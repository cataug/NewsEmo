#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tahiti/NewsEmo")
MAN = ROOT / "cache/manifests/inews.csv"
CACHE = ROOT / "cache/llm/inews_llm.jsonl"

NUM = (
    "uncertainty",
    "urgency",
    "causal_explicitness",
    "attribution_strength",
    "quotation_density",
    "sensational_language",
)

TXT = ("frame", "stance", "focus", "evidence")


def valid(features):
    if not isinstance(features, dict):
        return False

    for key in NUM:
        try:
            value = features[key]
            if isinstance(value, bool):
                return False
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                return False
        except (KeyError, TypeError, ValueError):
            return False

    for key in TXT:
        if not isinstance(features.get(key), str):
            return False
        if key != "evidence" and not features[key].strip():
            return False

    return True


def read_inputs():
    with MAN.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    data = {str(row["uid"]): row for row in rows}

    if len(data) != len(rows):
        raise RuntimeError("Duplicate UID in manifest")

    if len(data) < 100:
        raise RuntimeError("Unexpectedly small iNews manifest")

    return data


def read_cache():
    records = {}
    broken = 0
    duplicates = 0

    if not CACHE.is_file():
        return records, broken, duplicates

    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                uid = str(obj["uid"])

                if uid in records:
                    duplicates += 1

                records[uid] = obj.get("features")

            except (ValueError, KeyError, TypeError):
                broken += 1

    return records, broken, duplicates


def audit():
    data = read_inputs()
    records, broken, duplicates = read_cache()

    good = {
        uid: features
        for uid, features in records.items()
        if uid in data and valid(features)
    }

    missing = sorted(set(data) - set(good))
    invalid = sorted(
        uid for uid in records
        if uid in data and not valid(records[uid])
    )

    coverage = len(good) / len(data)

    print("\n" + "=" * 75)
    print("NEWS EMOTION — PHASE 2 / INEWS CACHE AUDIT")
    print("=" * 75)

    print("Expected:", len(data))
    print("Cached unique IDs:", len(records))
    print("Structurally valid:", len(good))
    print("Missing or invalid:", len(missing))
    print("Coverage:", f"{coverage:.2%}")
    print("Invalid JSONL lines:", broken)
    print("Duplicate cached IDs:", duplicates)
    print("Missing IDs:", missing[:30])
    print("Invalid feature IDs:", invalid[:30])

    fields = Counter()

    for features in records.values():
        if isinstance(features, dict):
            fields.update(features.keys())

    print("\n========== FEATURE COMPLETENESS ==========")

    for key in NUM + TXT:
        print(
            f"{key:26s} "
            f"{fields[key]:5d}/{len(records)}"
        )

    print("\n========== NUMERICAL DISTRIBUTIONS ==========")

    varying = 0
    numeric_stats = {}

    for key in NUM:
        values = [float(f[key]) for f in good.values()]

        if not values:
            continue

        mean = statistics.mean(values)
        std = statistics.pstdev(values)

        if std > 0.005:
            varying += 1

        numeric_stats[key] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }

        print(
            f"{key:26s} "
            f"mean={mean:.4f} "
            f"std={std:.4f} "
            f"min={min(values):.4f} "
            f"max={max(values):.4f}"
        )

    print("\n========== SEMANTIC DISTRIBUTIONS ==========")

    for key in ("frame", "stance", "focus"):
        counts = Counter(
            str(f[key]).strip().lower()
            for f in good.values()
        )

        print(
            key,
            "unique=", len(counts),
            "top=", counts.most_common(8),
        )

    evidence_total = 0
    evidence_matched = 0

    for uid, features in good.items():
        evidence = features["evidence"].strip().strip("\"' ")

        if evidence:
            evidence_total += 1

            if evidence.casefold() in data[uid].get("text", "").casefold():
                evidence_matched += 1

    print("\n========== EVIDENCE CHECK ==========")
    print("Nonempty evidence:", evidence_total)
    print("Exact substring matches:", evidence_matched)

    if evidence_total:
        print(
            "Exact match rate:",
            f"{evidence_matched/evidence_total:.2%}"
        )

    print("\n========== EXAMPLE RECORDS ==========")

    keys = sorted(good)

    if keys:
        positions = sorted(set([0, len(keys)//2, len(keys)-1]))

        for pos in positions:
            uid = keys[pos]

            print("\nUID:", uid)
            print(
                "TEXT:",
                data[uid].get("text", "")[:220]
            )
            print(
                "FEATURES:",
                json.dumps(
                    good[uid],
                    ensure_ascii=False
                )[:700]
            )

    report = {
        "expected": len(data),
        "cached_unique": len(records),
        "valid": len(good),
        "coverage": coverage,
        "missing": missing,
        "invalid_ids": invalid,
        "invalid_jsonl_lines": broken,
        "duplicate_ids": duplicates,
        "varying_numeric_fields": varying,
        "numeric_statistics": numeric_stats,
        "evidence_nonempty": evidence_total,
        "evidence_matched": evidence_matched,
    }

    output = ROOT / "reports/phase2_inews_llm_audit.json"

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nAudit saved:", output)

    return data, missing, report


def parse_response(answer):
    text = str(answer).strip()
    start = text.find("{")

    if start < 0:
        return None

    text = text[start:]

    candidates = [
        text,
        text.replace("\\'", "'"),
    ]

    candidates += [
        re.sub(r",\s*([}\]])", r"\1", candidate)
        for candidate in candidates[:]
    ]

    for candidate in candidates:
        try:
            obj, _ = json.JSONDecoder(
                strict=False
            ).raw_decode(candidate)

            if isinstance(obj, dict):
                return obj

        except (ValueError, TypeError):
            pass

    return None


def recover(missing, data):
    if not missing:
        print("[SKIP] No missing records")
        return []

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    model_path = str(
        ROOT / "models/qwen2_5_7b_instruct"
    )

    print("\n========== TARGETED LLM RECOVERY ==========")
    print("Model:", model_path)
    print("Records to recover:", len(missing))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    ).eval()

    backup = CACHE.with_name(
        CACHE.name + ".before_targeted_recovery"
    )

    if CACHE.is_file() and not backup.exists():
        shutil.copy2(CACHE, backup)
        print("Backup:", backup)

    failures = []

    for index, uid in enumerate(missing, 1):
        text = data[uid].get("text", "").strip()

        if not text:
            print("[ERROR] Empty text:", uid)
            failures.append(uid)
            continue

        success = False

        for attempt, (maxchars, tokens) in enumerate(
            [
                (2200, 320),
                (1500, 440),
                (950, 520),
            ],
            1,
        ):
            prompt = (
                "Analyze observable WRITING STYLE in this news text. "
                "Do not predict emotion, reader reactions, authenticity "
                "or any target label. "
                "Return exactly ONE compact valid JSON object. "
                "No Markdown, no explanations. "
                "Numerical values must be between 0 and 1. "
                "Required numeric keys: "
                + ", ".join(NUM)
                + ". Required short string keys: "
                + ", ".join(TXT)
                + ". Each string should contain at most 10 words. "
                "Evidence must be a short exact quotation "
                "from the text, or an empty string. "
                "Do not escape apostrophes using backslashes. "
                "\nTEXT:\n"
                + text[:maxchars]
            )

            try:
                inputs = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.inference_mode():
                    output = model.generate(
                        inputs,
                        max_new_tokens=tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                answer = tokenizer.decode(
                    output[0][inputs.shape[1]:],
                    skip_special_tokens=True,
                )

                features = parse_response(answer)

                if valid(features):
                    with CACHE.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "uid": uid,
                                    "features": features,
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        )

                    print(
                        f"[FIXED] {index}/{len(missing)} "
                        f"{uid} attempt={attempt}",
                        flush=True,
                    )

                    success = True
                    break

                print(
                    f"[RETRY] {uid} "
                    f"attempt={attempt} "
                    f"output={answer[:130]!r}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"[ERROR] {uid} "
                    f"attempt={attempt}: {exc!r}",
                    flush=True,
                )

        if not success:
            failures.append(uid)

    print("\nRECOVERY COMPLETE")
    print("Recovered:", len(missing) - len(failures))
    print("Still missing:", len(failures))
    print("Failures:", failures)

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["audit", "repair", "verify"],
    )
    args = parser.parse_args()

    data, missing, report = audit()

    if args.mode == "audit":
        if (
            report["coverage"] < 0.90
            or report["varying_numeric_fields"] < 3
        ):
            print(
                "[STOP] Suspicious cache. "
                "Further investigation required."
            )
            raise SystemExit(3)

        print(
            "\n[OK] Cache structure and numerical variation "
            "look plausible."
        )
        print(
            "This is not proof of semantic accuracy."
        )

    elif args.mode == "repair":
        failures = recover(missing, data)

        if failures:
            raise SystemExit(4)

    elif args.mode == "verify":
        if missing:
            print("[ERROR] Incomplete iNews LLM cache")
            raise SystemExit(5)

        print(
            "\n[OK] ALL INEWS LLM FEATURES "
            "STRUCTURALLY VALID"
        )


if __name__ == "__main__":
    main()
