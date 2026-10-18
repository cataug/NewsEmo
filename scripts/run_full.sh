#!/usr/bin/env bash
# NewsEmo Phase 2: sequential, foreground, cached. No pip/no new venv/no downloads.
cd /home/tahiti/NewsEmo || exit 1
set -o pipefail
PY=/home/tahiti/Malashin_Projects/.venv_a100/bin/python
QPY=/home/tahiti/IconoBench/.venv_qwen/bin/python
SCRIPT=scripts/phase2.py
mkdir -p logs cache results reports

for exe in "$PY" "$QPY"; do
    if [[ ! -x "$exe" ]]; then
        echo "ERROR: unavailable existing environment: $exe"
        exit 2
    fi
    "$exe" - <<'PY'
import numpy, pandas, torch, transformers
print('ENV OK', 'torch',torch.__version__,'transformers',transformers.__version__)
PY
    if [[ $? -ne 0 ]]; then
        echo "ERROR: preflight failed for $exe; no package changes attempted"
        exit 3
    fi
done
"$PY" - <<'PY'
import sklearn, scipy, pyarrow, PIL, joblib
print('CLASSICAL / PARQUET IMPORTS OK')
PY
if [[ $? -ne 0 ]]; then exit 4; fi

run() {
    name="$1"; shift
    echo "========== $(date '+%F %T') $name =========="
    if "$@" 2>&1 | tee -a "logs/phase2_${name}.log"; then
        echo "[OK] $name"
    else
        echo "[ERROR] $name, see logs/phase2_${name}.log"
        return 1
    fi
}

run prepare "$PY" -u "$SCRIPT" prepare || exit 10
for ds in inews gne semeval mirage clips; do
    run "stats_${ds}" "$PY" -u "$SCRIPT" extract --dataset "$ds" --encoder stats || exit 11
done
for ds in inews gne mirage; do
    run "mini_${ds}" "$PY" -u "$SCRIPT" extract --dataset "$ds" --encoder minilm --batch 24 || exit 12
done
for ds in mirage clips; do
    run "clip_text_${ds}" "$PY" -u "$SCRIPT" extract --dataset "$ds" --encoder cliptext --batch 32 || exit 13
    run "clip_img_${ds}" "$PY" -u "$SCRIPT" extract --dataset "$ds" --encoder clipimage --batch 16 || exit 14
done
run dino_mirage "$PY" -u "$SCRIPT" extract --dataset mirage --encoder dinov2 --batch 16 || exit 15
for enc in cliptextL clipimageL; do
    run "${enc}_clips" "$PY" -u "$SCRIPT" extract --dataset clips --encoder "$enc" --batch 8 || exit 16
done

# Structured semantic features. Each unique record is inferred at most once.
for ds in inews gne mirage; do
    run "qwen_text_${ds}" "$PY" -u "$SCRIPT" llm --dataset "$ds" --encoder llm || exit 20
done

# Strict completeness: no zero-filled LLM feature tables enter the experiment.
"$PY" - <<'PY'
import pandas as pd,json
from pathlib import Path
root=Path('/home/tahiti/NewsEmo')
for ds in ('inews','gne','mirage'):
    expected=set(pd.read_csv(root/f'cache/manifests/{ds}.csv').uid.astype(str))
    present=set()
    for ln in (root/f'cache/llm/{ds}_llm.jsonl').open():
        try: present.add(str(json.loads(ln)['uid']))
        except Exception: pass
    missing=expected-present
    print(ds,'LLM',len(present),'/',len(expected),'missing',len(missing),flush=True)
    if missing: raise SystemExit('Incomplete LLM features: rerun qwen_text_'+ds)
PY
if [[ $? -ne 0 ]]; then exit 21; fi

# 3000 fixed MiRAGe images and 1000 NewsCLIPpings pairs for VLM.
run vlm_mirage "$QPY" -u "$SCRIPT" llm --dataset mirage --encoder vlm || exit 22
run vlm_clips "$QPY" -u "$SCRIPT" llm --dataset clips --encoder vlm || exit 23

# 190 baseline + 60 VAD + 20 intensity = 270 trainings
run baselines "$PY" -u "$SCRIPT" baseline || exit 30
run vad "$PY" -u "$SCRIPT" vad || exit 31
run intensity "$PY" -u "$SCRIPT" intensity || exit 32

# 130 ablations, 120 leave-source-out, 20 VLM matched subsets = 270 trainings
run ablations "$PY" -u "$SCRIPT" ablation || exit 33
run source_holdout "$PY" -u "$SCRIPT" source || exit 34
run vlm_matched "$PY" -u "$SCRIPT" vlm || exit 35

# 27 validation-only Pareto / random / single-objective searches
run pareto "$PY" -u "$SCRIPT" pareto || exit 40
run robustness "$PY" -u "$SCRIPT" robustness || exit 41
run external "$PY" -u "$SCRIPT" external || exit 42
run stress "$PY" -u "$SCRIPT" stress || exit 42

# SemEval: two existing models x two fixed prompts, independently on 1000 headlines.
for model in qwen2_5_7b_instruct qwen3_8b; do
    for prompt in 0 1; do
        run "semeval_${model}_${prompt}" "$PY" -u "$SCRIPT" semeval --model "$model" --prompt-id "$prompt" || exit 43
    done
done
run semeval_report "$PY" -u "$SCRIPT" semeval_report || exit 44
run bootstrap "$PY" -u "$SCRIPT" bootstrap || exit 45
run statistics "$PY" -u "$SCRIPT" statistics || exit 46
run report "$PY" -u "$SCRIPT" report || exit 47

echo '========== PHASE 2 COMPLETE =========='
