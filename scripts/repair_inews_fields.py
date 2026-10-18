import csv
import json
import math
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT = Path("/home/tahiti/NewsEmo")
MAN = ROOT / "cache/manifests/inews.csv"
CACHE = ROOT / "cache/llm/inews_llm.jsonl"
REPORT = ROOT / "reports/inews_field_recovery.json"

NUM = (
    "uncertainty",
    "urgency",
    "causal_explicitness",
    "attribution_strength",
    "quotation_density",
    "sensational_language",
)
TXT = ("frame", "stance", "focus", "evidence")

DESCRIPTIONS = {
    "uncertainty": "Uncertainty or hedging in the wording",
    "urgency": "Urgency and time sensitivity of presentation",
    "causal_explicitness": "Explicitness of causes and effects",
    "attribution_strength": "Explicit attribution of claims to sources",
    "quotation_density": "Proportion of directly quoted speech",
    "sensational_language": "Sensational or attention-grabbing wording",
}

def valid(k, v):
    if k in NUM:
        if isinstance(v, bool):
            return False
        try:
            x = float(v)
            return math.isfinite(x) and 0 <= x <= 1
        except (TypeError, ValueError):
            return False
    return isinstance(v, str) and (k == "evidence" or bool(v.strip()))

def missing(f):
    return [k for k in NUM + TXT if not valid(k, f.get(k))]

def read_cache():
    data = {}
    invalid_lines = 0
    with CACHE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                uid = str(rec["uid"])
                feats = rec["features"]
                if not isinstance(feats, dict):
                    continue
                dest = data.setdefault(uid, {})
                for k in NUM + TXT:
                    v = feats.get(k)
                    if valid(k, v):
                        dest[k] = float(v) if k in NUM else v
            except (ValueError, TypeError, KeyError):
                invalid_lines += 1
    return data, invalid_lines

def parse_numbers(answer, keys):
    result = {}
    for key in keys:
        pat = (
            r'(?:^|[\s{,;])["\']?'
            + re.escape(key)
            + r'["\']?\s*[:=]\s*["\']?'
            + r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?![\d.])'
        )
        match = re.search(pat, answer, re.I | re.M)
        if match and valid(key, match.group(1)):
            result[key] = float(match.group(1))

    if len(keys) == 1 and keys[0] not in result:
        match = re.fullmatch(
            r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*',
            answer,
        )
        if match and valid(keys[0], match.group(1)):
            result[keys[0]] = float(match.group(1))
    return result

def parse_text(answer):
    result = {}
    start = answer.find("{")
    if start >= 0:
        for candidate in (
            answer[start:],
            answer[start:].replace("\\'", "'"),
        ):
            try:
                obj, _ = json.JSONDecoder(strict=False).raw_decode(candidate)
                if isinstance(obj, dict):
                    for k in TXT:
                        if valid(k, obj.get(k)):
                            result[k] = obj[k]
                    break
            except ValueError:
                pass
    for line in answer.splitlines():
        match = re.match(
            r'\s*["\']?(frame|stance|focus|evidence)["\']?\s*[:=]\s*(.*)',
            line, re.I,
        )
        if match:
            k = match.group(1).lower()
            value = match.group(2).strip().rstrip(",").strip('"')
            if valid(k, value):
                result[k] = value
    return result

def generate(model, tokenizer, prompt, max_tokens=128):
    import torch

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=1800,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.inference_mode():
        output = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][enc["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

def recover_one(uid, text, f, model, tokenizer):
    text = re.sub(r"^nan\s+", "", text, flags=re.I).strip()
    if not text:
        return False, 0

    calls = 0
    need_num = [k for k in NUM if k not in f]

    if need_num:
        definitions = "\n".join(
            f"{k}: {DESCRIPTIONS[k]}" for k in need_num
        )
        format_hint = "; ".join(
            f"{k}=<NUMBER>" for k in need_num
        )
        prompt = (
            "Evaluate only the following WRITING-STYLE dimensions "
            "of the news text. Do not infer emotion labels, human "
            "reactions, factual accuracy or fake/real status.\n"
            + definitions
            + "\nReturn ONLY the requested key=value numerical scores "
            "between 0 and 1, on a single line. Format: "
            + format_hint
            + "\nTEXT:\n" + text[:2000]
        )

        answer = generate(model, tokenizer, prompt, 150)
        calls += 1
        f.update(parse_numbers(answer, need_num))

        # One short scalar generation for any field still missing.
        for key in need_num:
            if key in f:
                continue

            prompt = (
                f"Evaluate ONLY {key}: {DESCRIPTIONS[key]}. "
                "Answer with exactly ONE number between 0 and 1. "
                "No explanation, no JSON, no other text. "
                "Do not predict any target label.\nTEXT:\n"
                + text[:1500]
            )

            answer = generate(model, tokenizer, prompt, 24)
            calls += 1
            f.update(parse_numbers(answer, [key]))

            if key not in f:
                print(
                    f"[FAILED NUMERIC] {uid} {key}: {answer[:100]!r}",
                    flush=True,
                )

    need_text = [k for k in TXT if k not in f]

    if need_text:
        prompt = (
            "Extract brief writing-style descriptors from this news. "
            "Do not predict emotional reactions or target labels. "
            "Return a compact JSON object with exactly these "
            "string fields: "
            + ", ".join(need_text)
            + ". frame=framing, stance=writer stance, "
            "focus=main subject, evidence=an exact short "
            "substring from the text (empty if none). "
            "No Markdown.\nTEXT:\n"
            + text[:1400]
        )

        answer = generate(model, tokenizer, prompt, 150)
        calls += 1
        found = parse_text(answer)

        for key in need_text:
            if key in found:
                f[key] = found[key]

        for key in need_text:
            if key in f:
                continue

            instruction = (
                "Return an exact short substring from the text, "
                "or an empty string if none exists."
                if key == "evidence"
                else "Return only one brief descriptive phrase."
            )
            prompt = (
                f"Extract ONLY {key} from the following news. "
                + instruction
                + " Do not predict emotion or the target label.\nTEXT:\n"
                + text[:1000]
            )
            answer = generate(model, tokenizer, prompt, 50)
            calls += 1
            value = answer.strip().strip("`").strip('"').strip()
            if valid(key, value):
                f[key] = value

    return not missing(f), calls

def main():
    with MAN.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    manifest = {r["uid"]: r for r in rows}

    if len(manifest) != len(rows):
        raise RuntimeError("Duplicate UIDs in manifest")

    cached, bad_lines = read_cache()
    pending = {
        uid: missing(cached.get(uid, {}))
        for uid in manifest
        if missing(cached.get(uid, {}))
    }

    print("=" * 72, flush=True)
    print("INEWS TARGETED FIELD RECOVERY", flush=True)
    print("Expected:", len(manifest), flush=True)
    print("Already complete:", len(manifest) - len(pending), flush=True)
    print("Remaining:", len(pending), flush=True)
    print("Bad JSONL lines:", bad_lines, flush=True)
    print(
        "Missing fields:",
        dict(Counter(k for keys in pending.values() for k in keys)),
        flush=True,
    )

    if not pending:
        print("[OK] All fields complete; no model loaded", flush=True)
        return

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    backup = CACHE.with_name(
        CACHE.name + ".backup_field_repair_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shutil.copy2(CACHE, backup)
    print("[BACKUP]", backup, flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = str(ROOT / "models/qwen2_5_7b_instruct")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    ).eval()

    recovered = 0
    total_calls = 0

    for index, uid in enumerate(sorted(pending), 1):
        features = dict(cached.get(uid, {}))
        text = manifest[uid].get("text", "")

        try:
            ok, calls = recover_one(
                uid, text, features, model, tokenizer
            )
            total_calls += calls
        except Exception as exc:
            ok = False
            print(f"[ERROR] {uid}: {exc!r}", flush=True)

        if ok:
            with CACHE.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"uid": uid, "features": features},
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )
                fh.flush()

            recovered += 1
            print(
                f"[FIXED] {index}/{len(pending)} "
                f"{uid} recovered={recovered}",
                flush=True,
            )
        else:
            print(
                f"[PARTIAL] {uid} missing={missing(features)}",
                flush=True,
            )

        if index % 20 == 0:
            print(
                f"[PROGRESS] {index}/{len(pending)} "
                f"recovered={recovered} calls={total_calls}",
                flush=True,
            )

    final, bad_lines = read_cache()
    unresolved = {
        uid: missing(final.get(uid, {}))
        for uid in manifest
        if missing(final.get(uid, {}))
    }

    summary = {
        "expected": len(manifest),
        "complete": len(manifest) - len(unresolved),
        "recovered_now": recovered,
        "unresolved": unresolved,
        "inference_calls": total_calls,
        "backup": str(backup),
        "repair_protocol": "targeted_numeric_and_text_v3",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== FINAL AUDIT ==========", flush=True)
    print("Complete:", summary["complete"], "/", len(manifest), flush=True)
    print("Recovered now:", recovered, flush=True)
    print("Still missing:", len(unresolved), flush=True)
    print("Qwen calls:", total_calls, flush=True)
    print("Report:", REPORT, flush=True)

    if unresolved:
        print(
            "[STOP] Remaining IDs:",
            list(unresolved.items())[:20],
            flush=True,
        )
        raise SystemExit(2)

    print("[OK] ALL INEWS FEATURES COMPLETE", flush=True)

if __name__ == "__main__":
    main()
