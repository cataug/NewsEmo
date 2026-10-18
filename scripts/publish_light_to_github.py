#!/usr/bin/env python3
import os, re, sys, shutil, fnmatch, tempfile, subprocess
from pathlib import Path
from collections import Counter

ROOT = Path('/home/tahiti/NewsEmo')
REMOTE = 'git@github.com:cataug/NewsEmo.git'

MAX_FILE = int(float(os.getenv('MAX_FILE_MB', '18')) * 1024**2)
MAX_TOTAL = int(float(os.getenv('MAX_TOTAL_MB', '450')) * 1024**2)

BLOCK_DIRS = {
    '.git', '.svn', '.hg', '.venv', 'venv', '__pycache__',
    '.cache', '.pytest_cache', '.mypy_cache', '.ipynb_checkpoints',
    'models', 'model', 'weights', 'checkpoints', 'ckpts',
    'datasets', 'dataset', 'data', 'raw', 'raw_data',
    'downloads', 'download', 'node_modules', 'wandb',
    'runs', 'tmp', 'temp', 'dist', 'build', 'site-packages',
}

BLOCK_CACHE = {
    'embeddings', 'llm', 'vlm', 'zeroshot',
    'features', 'models', 'phase2_extension'
}

BLOCK_EXT = {
    '.safetensors', '.pt', '.pth', '.ckpt', '.bin',
    '.gguf', '.onnx', '.h5', '.hdf5', '.pkl', '.joblib',
    '.key', '.pem', '.p12', '.pfx',
    '.avi', '.mp4', '.mkv', '.mov', '.wav', '.mp3', '.flac'
}

BLOCK_FILE = {
    '.env', '.netrc', '.npmrc', '.pypirc',
    'id_rsa', 'id_ed25519', 'id_ecdsa',
    'credentials.json', 'tokens.json', 'token.json',
    'secrets.json', 'authorized_keys', 'known_hosts',
    'git-credentials'
}

SECRET = [
    re.compile(s) for s in (
        rb'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b',
        rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
        rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}\b',
        rb'\bhf_[A-Za-z0-9]{30,}\b',
        rb'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b',
        rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE|ENCRYPTED PRIVATE) KEY-----',
        rb'(?i)Authorization\s*:\s*Bearer\s+[^\s"\']{20,}',
    )
]

def git(*args, cwd=None):
    print('+ git', *args, flush=True)
    subprocess.run(['git', *args], cwd=cwd, check=True)

def excluded(rel):
    p = rel.parts
    folders = [x.lower() for x in p[:-1]]

    # Publish source code, figures, results and lightweight logs.
    allowed = {
        'scripts', 'src', 'tests', 'docs', 'notebooks',
        'figures', 'figs', 'results', 'reports',
        'logs', 'analysis', 'experiments',
        'configs', 'config', '.github',
        'assets', 'paper', 'manuscript', 'tables',
        'utils'
    }

    if len(p) > 1:
        top = p[0].lower()

        if top == 'cache':
            if len(p) >= 3 and p[1].lower() != 'manifests':
                return 'generated cache excluded'
            if len(p) == 2 and p[1].lower() != 'placeholder':
                return 'cache root file excluded'

        elif top not in allowed:
            return 'non-publication directory excluded'

    if any(
        x in BLOCK_DIRS or x.startswith(('.venv', 'venv_', '.tox'))
        for x in folders
    ):
        return 'models, environment or raw data'

    if (
        len(p) > 2
        and p[0].lower() == 'cache'
        and p[1].lower() in BLOCK_CACHE
    ):
        return 'large inference cache'

    name = p[-1].lower()

    if (
        name in BLOCK_FILE
        or name.startswith(('.env.', 'id_rsa.', 'id_ed25519.'))
        or any(
            x in name for x in (
                'private_key', 'api_key', 'passwords', 'secret_key'
            )
        )
    ):
        return 'secret-bearing filename'

    if (
        rel.suffix.lower() in BLOCK_EXT
        or any(
            fnmatch.fnmatch(name, x)
            for x in ('*.pyc', '*.pyo', '*.swp', '*.tmp', '*.bak', '*~')
        )
    ):
        return 'weight, raw media or temporary file'

    return None

def priority(rel):
    top = rel.parts[0].lower()

    if len(rel.parts) == 1 or top in (
        'scripts', 'src', 'tests', '.github'
    ):
        return 0

    if top == 'figures':
        return 1

    if top in ('results', 'reports'):
        return 2

    if top == 'logs':
        return 3

    return 4

def main():
    if not (ROOT / 'scripts').is_dir():
        raise RuntimeError(f'No NewsEmo project: {ROOT}')

    if not shutil.which('git'):
        raise RuntimeError('Git executable is missing')

    skipped, candidates = Counter(), []

    # Scan all local files except excluded directories.
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        base = Path(base)

        dirs[:] = [
            d for d in dirs
            if not (base / d).is_symlink()
            and not excluded(
                (base / d / 'placeholder').relative_to(ROOT)
            )
        ]

        for filename in files:
            src = base / filename

            if src.is_symlink() or not src.is_file():
                continue

            rel = src.relative_to(ROOT)
            reason = excluded(rel)
            size = src.stat().st_size

            if size > MAX_FILE:
                reason = 'over per-file size cap'

            if reason:
                skipped[reason] += 1
                continue

            candidates.append(
                (priority(rel), rel.as_posix(), src, rel, size)
            )

    # Prioritize scripts and figures over bulky secondary files.
    candidates.sort()

    chosen, total = [], 0

    for _, _, src, rel, size in candidates:
        if total + size > MAX_TOTAL:
            skipped['over total size cap'] += 1
            continue

        content = src.read_bytes()

        if any(pattern.search(content) for pattern in SECRET):
            skipped['possible credential found'] += 1
            print('[SECURITY SKIP]', rel)
            continue

        chosen.append((src, rel))
        total += size

    print(
        f'SELECTED: {len(chosen)} files, '
        f'{total / 1024**2:.1f} MiB'
    )
    print('SKIPPED:', dict(skipped), flush=True)

    if not chosen:
        raise RuntimeError('No suitable files')

    # Clone existing GitHub history to a temporary directory.
    with tempfile.TemporaryDirectory(prefix='newsemo_git_') as tmp:
        checkout = Path(tmp) / 'repo'

        git('clone', '--depth', '1', REMOTE, str(checkout))

        branch = subprocess.check_output(
            ['git', 'branch', '--show-current'],
            cwd=checkout,
            text=True
        ).strip()

        if not branch:
            raise RuntimeError('Cannot determine target branch')

        print('REMOTE BRANCH:', branch)

        paths = Path(tmp) / 'pathspec.nul'

        with paths.open('wb') as f:
            for src, rel in chosen:
                dst = checkout / rel
                dst.parent.mkdir(parents=True, exist_ok=True)

                if dst.is_symlink():
                    raise RuntimeError(
                        f'Destination is symlink: {rel}'
                    )

                shutil.copy2(src, dst)
                f.write(os.fsencode(rel.as_posix()) + b'\0')

        # -f deliberately includes project figures and logs
        # even when they are listed in .gitignore.
        git(
            'add', '-f',
            f'--pathspec-from-file={paths}',
            '--pathspec-file-nul',
            cwd=checkout
        )

        diff = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'],
            cwd=checkout
        ).returncode

        if diff == 0:
            print('[OK] No new changes to upload')
            return

        if diff != 1:
            raise RuntimeError('git diff failed')

        summary = subprocess.check_output(
            ['git', 'diff', '--cached', '--shortstat'],
            cwd=checkout,
            text=True
        ).strip()
        print('[STAGED SUMMARY]', summary, flush=True)

        name = subprocess.run(
            ['git', 'config', 'user.name'],
            cwd=checkout,
            capture_output=True,
            text=True
        ).stdout.strip()

        email = subprocess.run(
            ['git', 'config', 'user.email'],
            cwd=checkout,
            capture_output=True,
            text=True
        ).stdout.strip()

        if not name:
            git('config', 'user.name', 'cataug', cwd=checkout)
            name = 'cataug'

        if not email:
            git(
                'config',
                'user.email',
                'cataug@users.noreply.github.com',
                cwd=checkout
            )
            email = 'cataug@users.noreply.github.com'

        if not (name and email):
            raise RuntimeError(
                'Set git config --global user.name '
                'and user.email first'
            )

        git(
            'commit', '-m',
            'Publish lightweight NewsEmo source, figures and results',
            cwd=checkout
        )

        # No --force, no rewriting history.
        git(
            'push', 'origin',
            f'HEAD:refs/heads/{branch}',
            cwd=checkout
        )

        print(
            f'[SUCCESS] {len(chosen)} files published '
            f'to {REMOTE}, branch {branch}'
        )
        print(
            'Local project untouched; '
            'no forced pushes or remote deletions'
        )

try:
    main()
except (Exception, KeyboardInterrupt) as exc:
    print('[STOP]', str(exc), file=sys.stderr)
    sys.exit(1)
