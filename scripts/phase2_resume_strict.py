#!/usr/bin/env python3
"""
NewsEmo Phase 2:
strict field recovery + automatic full-pipeline continuation.

Uses existing phase2_autorun.py and local models.
No installation, no downloading, no cache deletion.
"""

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys

from collections import Counter
from pathlib import Path

ROOT = Path("/home/tahiti/NewsEmo")

sys.path.insert(0, str(ROOT / "scripts"))

import phase2_autorun as runner


# ============================================================
# SEMANTIC FEATURE DEFINITIONS
# ============================================================

LLM_DESCRIPTIONS = {
    "uncertainty":
        "degree of uncertainty or hedging in the wording",

    "urgency":
        "degree of urgency or time pressure in the wording",

    "causal_explicitness":
        "explicitness of causal statements",

    "attribution_strength":
        "strength of attribution to named sources",

    "quotation_density":
        "amount of directly quoted speech; zero if none exists",

    "sensational_language":
        "degree of sensational or attention-grabbing wording",
}

VLM_DESCRIPTIONS = {
    "visual_salience":
        "prominence of visually salient subjects",

    "people_presence":
        "visibility of people in the image",

    "scene_uncertainty":
        "ambiguity of the visible scene",

    "image_text_alignment":
        "literal correspondence between caption and image",

    "image_text_complementarity":
        "additional information supplied by the image",

    "visual_emphasis":
        "degree of dramatic visual emphasis",
}

ORIGINAL_SEMANTIC_STAGE = runner.semantic_stage


# ============================================================
# STRICT NUMERICAL RESPONSE PARSER
# ============================================================

def parse_scalar(raw, key, kind):
    # Existing parser: key=value, JSON, numeric fields.
    found = runner.extract_numbers(raw, [key], kind)

    if key in found:
        return found[key]

    # Alternative JSON formats:
    # {"value":0.2}, {"score":0.2}, etc.
    obj = runner.parse_object(raw)

    if isinstance(obj, dict):
        for alias in ("value", "score", "number", "rating"):
            value = obj.get(alias)

            if runner.good(key, value, kind):
                return float(value)

    # Plain numeric response: 0.2
    text = str(raw).strip()

    text = re.sub(
        r"^```(?:json|text)?\s*|\s*```$",
        "",
        text,
        flags=re.I,
    ).strip()

    pattern = (
        r"(?:score|value|answer)?\s*[:=]?\s*"
        r"([+]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"\s*[.,;]?\s*"
    )

    match = re.fullmatch(pattern, text, re.I)

    if match:
        value = match.group(1)

        if runner.good(key, value, kind):
            return float(value)

    return None


# ============================================================
# SHORT SCALAR PROMPTS
# ============================================================

def numeric_prompt(key, text, kind, attempt):
    descriptions = (
        LLM_DESCRIPTIONS
        if kind == "llm"
        else VLM_DESCRIPTIONS
    )

    definition = descriptions[key]

    text = re.sub(
        r"^nan\s+",
        "",
        str(text),
        flags=re.I,
    ).strip()

    instructions = [
        "Reply with exactly ONE decimal number between 0 and 1. "
        "No key name, JSON, explanation or Markdown.",

        "Return only one numeric score. "
        "Example formatting: 0.25. No other characters.",

        "Estimate a score from 0 to 1. "
        "Use 0.0 when the described property is absent. "
        "Return only the number.",

        "Your complete response must consist of one decimal "
        "number, such as 0.0, 0.5 or 1.0.",

        "Output exactly one valid floating-point number "
        "between 0 and 1. Nothing else.",
    ]

    instruction = instructions[attempt % len(instructions)]

    return (
        f"Evaluate ONLY this observable feature: {key}.\n"
        f"Definition: {definition}.\n"
        "Do not predict ground-truth labels, emotions, "
        "or news authenticity.\n"
        f"{instruction}\n"
        "NEWS TEXT/CAPTION:\n"
        + text[:1600]
    )


def string_prompt(key, text, attempt):
    descriptions = {
        "frame":
            "Name the main narrative frame in at most six words.",

        "stance":
            "Name the writer's stance in one or two words.",

        "focus":
            "Name the main subject in at most six words.",

        "evidence":
            "Return a short exact substring of the news text, "
            "or an empty string.",

        "scene":
            "Describe the visible scene in at most six words.",

        "objects":
            "Name the visible objects in at most six words.",

        "activity":
            "Describe the visible activity in at most six words.",

        "visual_evidence":
            "Describe one directly observable image detail.",
    }

    instruction = descriptions[key]

    output_format = (
        f'Return exactly {{"{key}":"short phrase"}}.'
        if attempt % 2 == 0
        else "Return only the short phrase, without JSON."
    )

    return (
        f"Extract ONLY {key}. {instruction}\n"
        "Do not predict emotion or authenticity labels.\n"
        f"{output_format}\n"
        "TEXT/CAPTION:\n"
        + str(text)[:1100]
    )


def parse_string(raw, key, kind):
    parsed = runner.extract_strings(raw, [key], kind)

    if key in parsed:
        return parsed[key]

    obj = runner.parse_object(raw)

    if isinstance(obj, dict):
        for alias in ("value", "answer", "text"):
            value = obj.get(alias)

            if runner.good(key, value, kind):
                return value

    value = (
        str(raw)
        .strip()
        .strip("`")
        .strip()
        .strip('"')
        .strip()
    )

    if (
        len(value) < 180
        and not value.startswith(("{", "["))
        and runner.good(key, value, kind)
    ):
        return value

    return None


# ============================================================
# RECORD PROVENANCE
# ============================================================

def save_provenance(dataset, kind, uid, key, attempt):
    path = (
        ROOT / "reports"
        / "phase2_field_recovery_provenance.jsonl"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "dataset": dataset,
        "kind": kind,
        "uid": uid,
        "field": key,
        "method": "Qwen_local_scalar_repair",
        "attempt": attempt,
    }

    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# ============================================================
# STRICT FIELD-BY-FIELD RECOVERY
# ============================================================

def targeted_repair(dataset, kind):
    df = runner.candidate_rows(dataset, kind)

    cache_path = (
        ROOT / "cache" / "llm"
        / f"{dataset}_{kind}.jsonl"
    )

    cache, broken = runner.merge_cache(cache_path, kind)

    rows = [
        (
            str(row.uid),
            str(row.text),
            str(getattr(row, "image", "")),
        )
        for row in df.itertuples(index=False)
    ]

    pending = [
        (uid, text, image)
        for uid, text, image in rows
        if runner.missing(cache.get(uid, {}), kind)
    ]

    missing_fields = Counter(
        key
        for uid, _, _ in pending
        for key in runner.missing(cache.get(uid, {}), kind)
    )

    runner.say(
        "STRICT REPAIR",
        dataset,
        kind,
        "pending_records", len(pending),
        "missing_fields", dict(missing_fields),
        "broken_jsonl_lines", broken,
    )

    if not pending:
        runner.say("CACHE ALREADY COMPLETE", dataset, kind)
        return

    # Backup only once.
    if cache_path.is_file():
        backup = cache_path.with_name(
            cache_path.name + ".before_strict_repair"
        )

        if not backup.exists():
            shutil.copy2(cache_path, backup)
            runner.say("CACHE BACKUP", backup)

    model, processor = runner.model_session(kind)

    try:
        for index, (uid, text, image) in enumerate(pending, 1):
            features = dict(cache.get(uid, {}))

            required = runner.missing(features, kind)

            for key in required:
                numeric_keys = (
                    runner.NUM if kind == "llm"
                    else runner.VN
                )

                is_numeric = key in numeric_keys

                for attempt in range(5):
                    if is_numeric:
                        prompt = numeric_prompt(
                            key, text, kind, attempt
                        )
                        max_tokens = 48
                    else:
                        prompt = string_prompt(
                            key, text, attempt
                        )
                        max_tokens = 115

                    try:
                        answer = runner.generate(
                            model,
                            processor,
                            kind,
                            prompt,
                            imgpath=(
                                image if kind == "vlm"
                                else None
                            ),
                            max_tokens=max_tokens,
                        )

                        if is_numeric:
                            value = parse_scalar(
                                answer, key, kind
                            )
                        else:
                            value = parse_string(
                                answer, key, kind
                            )

                        if runner.good(key, value, kind):
                            features[key] = (
                                float(value)
                                if is_numeric
                                else value
                            )

                            save_provenance(
                                dataset,
                                kind,
                                uid,
                                key,
                                attempt + 1,
                            )

                            break

                        if attempt == 4:
                            runner.say(
                                "UNPARSEABLE",
                                dataset,
                                uid,
                                key,
                                repr(answer[:180]),
                            )

                    except Exception as exc:
                        runner.say(
                            "REPAIR RETRY",
                            dataset,
                            uid,
                            key,
                            attempt + 1,
                            repr(exc),
                        )

                # Save successfully recovered fields immediately.
                if key in features:
                    runner.cache_append(
                        cache_path,
                        uid,
                        features,
                        kind,
                    )

            remaining = runner.missing(features, kind)

            if remaining:
                runner.say(
                    "STILL INCOMPLETE",
                    dataset,
                    uid,
                    remaining,
                )

            elif (
                index <= 5
                or index % 25 == 0
                or index == len(pending)
            ):
                runner.say(
                    "REPAIRED",
                    dataset,
                    kind,
                    index,
                    "/",
                    len(pending),
                )

    finally:
        del model, processor
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Independent final audit from disk.
    verified, broken = runner.merge_cache(
        cache_path, kind
    )

    unresolved = {
        uid: runner.missing(verified.get(uid, {}), kind)
        for uid, _, _ in rows
        if runner.missing(verified.get(uid, {}), kind)
    }

    report = (
        ROOT / "reports"
        / f"phase2_missing_{dataset}_{kind}.json"
    )

    report.parent.mkdir(parents=True, exist_ok=True)

    report.write_text(
        json.dumps(
            unresolved,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    runner.say(
        "STRICT AUDIT",
        dataset,
        kind,
        "complete",
        len(rows) - len(unresolved),
        "/",
        len(rows),
        "remaining",
        len(unresolved),
    )

    if unresolved:
        raise RuntimeError(
            f"{dataset}/{kind}: "
            f"{len(unresolved)} records still incomplete. "
            f"Details: {report}"
        )


# ============================================================
# BULK EXTRACTION + AUTOMATIC REPAIR
# ============================================================

def recover_or_generate(dataset, kind):
    df = runner.candidate_rows(dataset, kind)

    path = (
        ROOT / "cache" / "llm"
        / f"{dataset}_{kind}.jsonl"
    )

    previous, _ = runner.merge_cache(path, kind)

    pending = sum(
        bool(runner.missing(previous.get(str(uid), {}), kind))
        for uid in df.uid
    )

    runner.say(
        "CACHE STATUS",
        dataset,
        kind,
        "total", len(df),
        "pending", pending,
    )

    if pending == 0:
        return

    # Large, new datasets: use the existing bulk extractor.
    if pending > 1000:
        try:
            ORIGINAL_SEMANTIC_STAGE(dataset, kind)

        except RuntimeError as exc:
            # The old extractor stops at the final completeness
            # audit. Recover that specific failure automatically.
            if "incomplete records" not in str(exc):
                raise

            runner.say(
                "BULK ENDED WITH INCOMPLETE FIELDS",
                "AUTO REPAIR STARTED",
            )

    # Already processed datasets:
    # repair missing fields, not complete records.
    targeted_repair(dataset, kind)


# ============================================================
# USE EXISTING COMPATIBLE VLM ENVIRONMENT
# ============================================================

def choose_vlm_python():
    code = (
        "import pandas,numpy,PIL,torch,transformers;"
        "from transformers import "
        "Qwen2_5_VLForConditionalGeneration"
    )

    for executable in (runner.QPY, runner.PY):
        check = subprocess.run(
            [str(executable), "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if check.returncode == 0:
            runner.say(
                "VLM PYTHON SELECTED",
                executable,
            )
            return executable

    raise RuntimeError(
        "Neither existing Python environment supports "
        "Qwen2.5-VL with pandas. "
        "No packages were installed."
    )


def patched_semantic_stage(dataset, kind):
    if kind == "vlm":
        # Run VLM in a compatible existing environment.
        executable = choose_vlm_python()

        result = subprocess.run(
            [
                str(executable),
                "-u",
                str(Path(__file__).resolve()),
                "--vlm-only",
                dataset,
            ],
            cwd=ROOT,
            check=False,
        )

        if result.returncode:
            raise RuntimeError(
                f"VLM {dataset} failed: "
                f"return code {result.returncode}"
            )
    else:
        recover_or_generate(dataset, kind)


# ============================================================
# CONTINUE ORIGINAL FULL PLAN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vlm-only",
        choices=["mirage", "clips"],
    )

    args = parser.parse_args()

    if args.vlm_only:
        recover_or_generate(
            args.vlm_only,
            "vlm",
        )

    else:
        runner.semantic_stage = patched_semantic_stage

        runner.say(
            "STRICT AUTORUNNER ACTIVE",
            "FIELD RECOVERY ENABLED",
        )

        runner.main()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        runner.say(
            "INTERRUPTED — caches preserved"
        )
        sys.exit(130)

    except Exception as exc:
        runner.say(
            "STOP",
            repr(exc),
            "— caches preserved",
        )
        sys.exit(1)
