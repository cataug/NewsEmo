#!/usr/bin/env bash
# NewsEmo: reproducible downloads for four external news/emotion datasets.
# Runs in foreground; no nohup/tmux/background workers.
# Usage: bash setup_newsemo.sh [--full-visualnews] [--visualnews-articles] [--no-screenshots]

set -u

ROOT="${NEWS_EMO_ROOT:-$HOME/NewsEmo}"
FULL_VISUALNEWS=0
VISUALNEWS_ARTICLES=0
TRY_SCREENSHOTS=1
for arg in "$@"; do
    case "$arg" in
        --full-visualnews) FULL_VISUALNEWS=1 ;;
        --visualnews-articles) VISUALNEWS_ARTICLES=1 ;;
        --no-screenshots) TRY_SCREENSHOTS=0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$ROOT"/{datasets,logs,reports,scripts,results,cache}
# To keep live output and a log, call: bash setup_newsemo.sh 2>&1 | tee ~/newsemo_download.log

say() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
warn() { printf '[WARNING] %s\n' "$*" >&2; }

say "Project root: $ROOT"
if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 is unavailable. Install python3 and python3-venv."; exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    warn "git is unavailable. Install git."; exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    warn "curl is unavailable. Install curl."; exit 1
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    say "Creating an isolated Python environment"
    if ! python3 -m venv "$ROOT/.venv"; then
        warn "Could not create venv. On Ubuntu: sudo apt install python3-venv"
        exit 1
    fi
fi
PY="$ROOT/.venv/bin/python"
say "Installing download and validation dependencies"
if ! "$PY" -m pip install -U pip huggingface_hub hf_xet; then
    warn "pip installation failed; downloads requiring Hugging Face may not work"
fi

# Do not put HF_TOKEN into this script. If gated access was approved, use
# export HF_TOKEN=hf_... in the shell or run: ~/NewsEmo/.venv/bin/hf auth login

clone_or_update() {
    local url="$1" dst="$2"
    if [ -d "$dst/.git" ]; then
        say "Updating $dst"
        git -C "$dst" pull --ff-only || warn "Update failed: $dst; keeping existing files"
    elif [ -e "$dst" ]; then
        warn "Directory already exists and is not a git repository: $dst"
    else
        say "Cloning $url"
        git clone --depth 1 "$url" "$dst" || warn "Clone failed: $url"
    fi
}

download_file() {
    local url="$1" dst="$2" min_gb="${3:-0}"
    mkdir -p "$(dirname "$dst")"
    if [ -s "$dst" ] && [ "$min_gb" = "0" ]; then
        say "Already downloaded: $dst"
        return 0
    fi
    if [ "$min_gb" != "0" ]; then
        # Avoid huge downloads when there is not enough free space.
        local free_kb needed_kb
        free_kb=$(df -Pk "$(dirname "$dst")" | awk 'NR==2 {print $4}')
        local existing_kb=0
        if [ -f "$dst" ]; then
            existing_kb=$(( $(stat -c%s "$dst") / 1024 ))
        fi
        needed_kb=$(( min_gb * 1024 * 1024 - existing_kb ))
        if [ "$needed_kb" -lt 0 ]; then needed_kb=0; fi
        if [ -n "$free_kb" ] && [ "$free_kb" -lt "$needed_kb" ]; then
            warn "Not enough free disk space for $dst (estimated remaining need: $((needed_kb / 1024 / 1024)) GiB)"
            return 1
        fi
    fi
    say "Downloading: $url"
    if ! curl -fL --retry 5 --retry-delay 4 --connect-timeout 30 -C - \
        --progress-bar "$url" -o "$dst"; then
        warn "Failed/interrupted: $dst (partial file is preserved for resuming)"
        return 1
    fi
    return 0
}

# 1. iNews: five original author splits and codebook; public, ~26.5 MB.
say "[1/4] iNews public CSVs + codebook"
if "$PY" - "$ROOT/datasets/inews_public" <<'PYCODE'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="pitehu/inews_public", repo_type="dataset", local_dir=str(out))
expected = ["train.csv", "dev.csv", "test_personalization_public.csv",
            "test_generalization_public.csv", "test_cold_start_public.csv"]
missing = [name for name in expected if not (out/name).is_file()]
if missing:
    raise RuntimeError("Missing iNews CSVs: " + repr(missing))
print("[OK] iNews: public CSV splits are present")
PYCODE
then
    say "iNews public: complete"
else
    warn "iNews public: FAILED; see previous error"
fi
clone_or_update "https://github.com/pitehu/inews.git" "$ROOT/datasets/inews_code"

# 1b. Gated screenshots (~926 MB), only if user has legitimate approval.
if [ "$TRY_SCREENSHOTS" = "1" ]; then
    say "iNews screenshots: checking approved Hugging Face access"
    if "$PY" - "$ROOT/datasets/inews_screenshots" <<'PYCODE'
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
out = Path(sys.argv[1])
repo = "pitehu/inews_screenshots"
try:
    info = HfApi().repo_info(repo, repo_type="dataset")
    # Gated files require authenticated approval even though the page is public.
    files = [s.rfilename for s in info.siblings]
    if "inews_screenshots_hf_release.zip" not in files:
        print("[INFO] Screenshot archive not listed for anonymous access")
        sys.exit(3)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(out),
                      allow_patterns=["inews_screenshots_hf_release.zip", "manifest.csv", "README.md"])
    print("[OK] Authorized screenshots downloaded")
except Exception as e:
    print("[INFO] Screenshot access is not available:", type(e).__name__, str(e)[:240])
    print("Request approval at https://huggingface.co/datasets/pitehu/inews_screenshots")
    print("After approval, log in with: ~/NewsEmo/.venv/bin/hf auth login")
    sys.exit(3)
PYCODE
    then
        say "iNews gated screenshots downloaded"
        SHOT_ZIP="$ROOT/datasets/inews_screenshots/inews_screenshots_hf_release.zip"
        if [ -s "$SHOT_ZIP" ] && [ ! -f "$ROOT/datasets/inews_screenshots/.extracted.ok" ]; then
            if "$PY" -m zipfile -t "$SHOT_ZIP" >/dev/null 2>&1; then
                mkdir -p "$ROOT/datasets/inews_screenshots/extracted"
                if "$PY" -m zipfile -e "$SHOT_ZIP" "$ROOT/datasets/inews_screenshots/extracted"; then
                    touch "$ROOT/datasets/inews_screenshots/.extracted.ok"
                    say "iNews screenshots extracted"
                else
                    warn "Screenshot extraction failed"
                fi
            else
                warn "The screenshot archive does not pass ZIP integrity check"
            fi
        fi
    else
        warn "Screenshots skipped (approval/login may be required); text labels are still available"
    fi
else
    say "iNews screenshots: explicitly skipped"
fi

# 2. GoodNewsEveryone official ZIP.
say "[2/4] GoodNewsEveryone"
GNE_DIR="$ROOT/datasets/goodnewseveryone"
mkdir -p "$GNE_DIR"
GNE_ZIP="$GNE_DIR/goodnewseveryone-v1.0.zip"
GNE_URL="https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/goodnewseveryone/goodnewseveryone-v1.0.zip"
if [ -s "$GNE_ZIP" ] && ! "$PY" -m zipfile -t "$GNE_ZIP" >/dev/null 2>&1; then
    warn "Archive is incomplete. Re-downloading from scratch to prevent corruption"
    rm -f "$GNE_ZIP"
fi
if download_file "$GNE_URL" "$GNE_ZIP"; then
    if "$PY" -m zipfile -t "$GNE_ZIP" >/dev/null 2>&1; then
        if [ ! -f "$GNE_DIR/.extracted.ok" ]; then
            mkdir -p "$GNE_DIR/extracted"
            if "$PY" -m zipfile -e "$GNE_ZIP" "$GNE_DIR/extracted"; then
                touch "$GNE_DIR/.extracted.ok"
                say "GoodNewsEveryone: extracted"
            else
                warn "GoodNewsEveryone extraction failed"
            fi
        else
            say "GoodNewsEveryone: already extracted"
        fi
    else
        warn "Downloaded GoodNewsEveryone is not a valid ZIP"
    fi
fi

# 3. Individual human emotion annotations for SemEval-2007 Affective Text.
say "[3/4] SemEval-2007 Affective Text, disaggregated annotations"
clone_or_update "https://github.com/MichiganNLP/Affective-Text-Individual-Annotations.git" \
    "$ROOT/datasets/semeval2007_affective_text"

# 4. Sens-VisualNews labels and evaluation code (~small Git repository).
say "[4/4] Sens-VisualNews, four original JSON annotation splits"
clone_or_update "https://github.com/IDT-ITI/Sens-VisualNews.git" \
    "$ROOT/datasets/sens_visualnews"

# Large original VisualNews assets: opt-in only (91 GB for image archive).
VN_DIR="$ROOT/datasets/visualnews_original"
mkdir -p "$VN_DIR"
if [ "$FULL_VISUALNEWS" = "1" ]; then
    say "Downloading original VisualNews image archive (91 GB; NO automatic extraction)"
    download_file "https://www.cs.rice.edu/~vo9/visualnews/origin.tar" \
        "$VN_DIR/origin.tar" 95 || warn "VisualNews original images not fully downloaded"
else
    say "VisualNews 91 GB image archive skipped. To download: --full-visualnews"
fi
if [ "$VISUALNEWS_ARTICLES" = "1" ]; then
    say "Downloading VisualNews articles (~4.2 GB)"
    download_file "https://www.cs.rice.edu/~vo9/visualnews/articles.tar.gz" \
        "$VN_DIR/articles.tar.gz" 6 || warn "VisualNews articles not fully downloaded"
fi

# Manifest: what exists, counts, size, paths, and commands for next steps.
say "Generating local dataset inventory"
if ! "$PY" - "$ROOT" <<'PYCODE'
import csv, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
d = root / 'datasets'
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root), 'datasets': {}}

def record(name, path, required=None):
    path = d/path
    files = [p for p in path.rglob('*') if p.is_file() and '.git' not in p.parts and '.cache' not in p.parts] if path.exists() else []
    size = sum(p.stat().st_size for p in files)
    req = [path/x for x in (required or [])]
    item = {'path': str(path), 'files': len(files), 'bytes': size,
            'size_MiB': round(size/(1024**2), 2),
            'required_present': all(p.is_file() for p in req) if req else None,
            'missing_required': [str(p.relative_to(path)) for p in req if not p.is_file()]}
    report['datasets'][name] = item
    return item

inews = record('inews_public', Path('inews_public'),
               ['train.csv', 'dev.csv', 'test_personalization_public.csv',
                'test_generalization_public.csv', 'test_cold_start_public.csv'])
rows = {}
for p in (d/'inews_public').glob('*.csv'):
    with p.open(encoding='utf-8-sig', newline='') as f:
        rows[p.name] = max(0, sum(1 for _ in csv.reader(f)) - 1)
report['datasets']['inews_public']['csv_rows'] = rows
record('inews_code', Path('inews_code'))
record('inews_screenshots', Path('inews_screenshots'))
record('goodnewseveryone', Path('goodnewseveryone'), ['goodnewseveryone-v1.0.zip'])
record('semeval2007_affective_text', Path('semeval2007_affective_text'), ['affectivetext_test.xml'])
record('sens_visualnews', Path('sens_visualnews'),
       ['dataset/full_test.json', 'dataset/full_dev.json',
        'dataset/strict_test.json', 'dataset/strict_dev.json'])
record('visualnews_original', Path('visualnews_original'))
for f in ['full_dev.json', 'full_test.json', 'strict_dev.json', 'strict_test.json']:
    p = d/'sens_visualnews'/'dataset'/f
    if p.is_file():
        with p.open(encoding='utf-8') as stream:
            data = json.load(stream)
        report['datasets']['sens_visualnews'][f] = len(data)

p = root/'reports'/'dataset_inventory.json'
p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nDATASET INVENTORY')
for name, item in report['datasets'].items():
    required = item['required_present']
    status = 'OK' if required is True else ('INCOMPLETE' if required is False else 'CHECK')
    print(f'{name:29s} {status:10s} files={item["files"]:5d}  size={item["size_MiB"]:10.1f} MiB')
print('iNews rows by split:', rows)
print('Inventory:', p)
PYCODE
then
    warn "Inventory generation failed"
fi

say "Done. External dataset downloads inspected. See $ROOT/reports/dataset_inventory.json"
say "Useful commands: du -sh '$ROOT/datasets'/* ; cat '$ROOT/reports/dataset_inventory.json'"
