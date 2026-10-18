#!/usr/bin/env python3

import json
import re
from pathlib import Path
from datetime import datetime, timezone

HOME = Path("/home/tahiti")
ROOT = HOME / "NewsEmo"
DEST = ROOT / "models"
REPORT = ROOT / "reports" / "phase2_model_registry.json"

DEST.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# KNOWN EXISTING MODEL DIRECTORIES
# ----------------------------------------------------------

SEARCH_ROOTS = [
    HOME / "CoMUL/models",
    HOME / "Forensics/models",
    HOME / "GreenToken/models",
    HOME / "MARINE_DATASETS/MARINE_MODELS",
    HOME / "ARTeccv/models",
    HOME / "TimeBound/models",
    HOME / "AgentUnlearn/models",
    HOME / "crypto/models",
    HOME / "UME2026/models/clip",
    HOME / "IconoBench/models",
    HOME / "GALA/models",
    HOME / "Spec2Test/models",
]

CACHE_ROOTS = [
    HOME / ".cache/huggingface/hub",
    HOME / "crypto/hf_cache/hub",
    HOME / "UME2026/models/hf_cache",
    HOME / "UME2026/models/hf_cache/hub",
    HOME / "UME2026/models/hf_cache/openclip",
]

KNOWN = {
    "qwen2_5_vl_7b": [
        HOME / "CoMUL/models/qwen2_5_vl_7b_instruct",
    ],
    "qwen2_5_vl_3b": [
        HOME / "ARTeccv/models/qwen2_5_vl_3b_instruct",
    ],
    "qwen3_8b": [
        HOME / "AgentUnlearn/models/Qwen3-8B",
        HOME / ".cache/huggingface/hub/"
               "models--Qwen--Qwen3-8B/snapshots/"
               "b968826d9c46dd6066d109eabc6255188de91218",
    ],
    "qwen2_5_coder_7b": [
        HOME / "crypto/models/Qwen2.5-Coder-7B-Instruct",
    ],
    "clip_vit_b32": [
        HOME / "Forensics/models/clip_vit_base_patch32",
        HOME / "GreenToken/models/clip_vit_base_patch32",
        HOME / "UME2026/models/clip/clip-vit-base-patch32",
    ],
    "clip_vit_l14": [
        HOME / "MARINE_DATASETS/MARINE_MODELS/clip_vit_large",
    ],
    "dinov2_small": [
        HOME / "Forensics/models/dinov2_small",
        HOME / "GreenToken/models/dinov2_small",
    ],
    "minilm_l6_v2": [
        HOME / "Forensics/models/minilm_l6_v2",
    ],
}

TARGETS = [
    "qwen2_5_7b_instruct",
    "qwen3_8b",
    "qwen2_5_coder_7b",
    "qwen2_5_vl_7b",
    "qwen2_5_vl_3b",
    "clip_vit_b32",
    "clip_vit_l14",
    "dinov2_small",
    "minilm_l6_v2",
    "openclip_laion_b32",
]

def normalize(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_candidates():
    candidates = []

    # First: already known locations
    for group in KNOWN.values():
        candidates.extend(group)

    # Second: model registry directories
    for parent in SEARCH_ROOTS:
        if not parent.is_dir():
            continue

        candidates.extend(
            p for p in parent.iterdir()
            if p.is_dir()
        )

    # Third: Hugging Face snapshots
    keywords = [
        "qwen2.5",
        "qwen3",
        "clip",
        "dinov2",
        "minilm",
        "all-minilm",
    ]

    for cache in CACHE_ROOTS:
        if not cache.is_dir():
            continue

        for repo in cache.glob("models--*"):
            if not any(
                k in repo.name.lower()
                for k in keywords
            ):
                continue

            candidates.extend(
                repo.glob("snapshots/*")
            )

    # Deduplicate resolved paths
    result = []
    seen = set()

    for p in candidates:
        try:
            if not p.is_dir():
                continue

            resolved = p.resolve(strict=True)

            if resolved in seen:
                continue

            seen.add(resolved)
            result.append(resolved)

        except (OSError, RuntimeError):
            pass

    return result


def validate_model(path):
    """
    Only inspect metadata/weights.
    No imports of transformers or torch.
    No model loading or downloading.
    """

    cfg = read_json(path / "config.json")
    openclip_cfg = read_json(
        path / "open_clip_config.json"
    )

    weights = []

    for pattern in (
        "*.safetensors",
        "pytorch_model*.bin",
        "open_clip_pytorch_model*.bin",
        "*.pt",
    ):
        weights.extend(path.glob(pattern))

    weights = sorted(set(weights))

    weights = [
        p for p in weights
        if p.is_file() and p.stat().st_size > 1000000
    ]

    if not weights:
        return None

    # Check sharded weight index if present
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index = path / index_name

        if not index.is_file():
            continue

        obj = read_json(index)
        required = set(
            obj.get("weight_map", {}).values()
        )

        if required:
            missing = [
                x for x in required
                if not (path / x).is_file()
            ]

            if missing:
                return None

    if not cfg and not openclip_cfg:
        return None

    total_bytes = sum(p.stat().st_size for p in weights)

    return {
        "config": cfg,
        "openclip_config": openclip_cfg,
        "weights": len(weights),
        "weight_gb": round(total_bytes / 1e9, 3),
    }


def identify(path, meta):
    name = normalize(path)
    cfg = meta["config"]
    model_type = str(
        cfg.get("model_type", "")
    ).lower()

    # Qwen models
    if "qwen25vl7b" in name and "qwen2_5_vl" in model_type:
        return "qwen2_5_vl_7b"

    if "qwen25vl3b" in name and "qwen2_5_vl" in model_type:
        return "qwen2_5_vl_3b"

    if "qwen25coder7b" in name and "qwen2" in model_type:
        return "qwen2_5_coder_7b"

    if (
        "qwen257binstruct" in name
        and "coder" not in name
        and "vl" not in name
        and model_type == "qwen2"
    ):
        return "qwen2_5_7b_instruct"

    if "qwen38b" in name and model_type == "qwen3":
        return "qwen3_8b"

    # Hugging Face CLIP
    if model_type == "clip":
        vision = cfg.get("vision_config", {})
        patch = vision.get("patch_size")

        if patch == 32:
            return "clip_vit_b32"

        if patch == 14:
            return "clip_vit_l14"

    # DINOv2 small
    if model_type == "dinov2":
        hidden = cfg.get("hidden_size")

        if hidden == 384:
            return "dinov2_small"

    # MiniLM text embeddings
    if (
        "minilm" in name
        and model_type in ("bert", "mpnet")
    ):
        return "minilm_l6_v2"

    # LAION OpenCLIP (different loading API)
    if (
        meta["openclip_config"]
        and "laion" in name
        and ("vitb32" in name or "vitb032" in name)
    ):
        return "openclip_laion_b32"

    return None


def create_link(name, source):
    dst = DEST / name

    if dst.is_symlink():
        try:
            if dst.resolve(strict=True) == source:
                return "ALREADY_LINKED"
        except OSError:
            pass

        return "EXISTS_DIFFERENT_TARGET"

    if dst.exists():
        return "EXISTS_NOT_LINK"

    dst.symlink_to(
        source,
        target_is_directory=True,
    )

    return "LINKED"


def main():
    candidates = discover_candidates()

    print("=" * 76)
    print("NEWSEMO — EXISTING MODEL DISCOVERY")
    print("=" * 76)
    print("Candidate directories:", len(candidates))
    print("No model downloads. No package installations.")
    print()

    discovered = {}

    for path in candidates:
        meta = validate_model(path)

        if meta is None:
            continue

        kind = identify(path, meta)

        if kind is None:
            continue

        discovered.setdefault(kind, []).append({
            "path": str(path),
            "weights": meta["weights"],
            "weight_gb": meta["weight_gb"],
        })

    registry = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": str(ROOT),
        "models": {},
    }

    for kind in TARGETS:
        options = discovered.get(kind, [])

        # Prefer directories in the remembered order.
        preferred = [
            p.resolve()
            for p in KNOWN.get(kind, [])
            if p.is_dir()
        ]

        options.sort(
            key=lambda x: (
                str(Path(x["path"])) not in
                [str(p) for p in preferred],
                -x["weight_gb"],
            )
        )

        if not options:
            print(f"[MISSING] {kind}")

            registry["models"][kind] = {
                "status": "MISSING",
            }
            continue

        best = options[0]
        source = Path(best["path"])

        status = create_link(kind, source)

        print(f"[{status}] {kind}")
        print(f"   Source: {source}")
        print(f"   Weights: {best['weight_gb']:.2f} GB")

        registry["models"][kind] = {
            "status": status,
            "source": str(source),
            "link": str(DEST / kind),
            "weights_gb": best["weight_gb"],
            "alternatives": options[1:],
        }

    # Text LLM alias: use exact Qwen2.5-7B if present;
    # otherwise explicitly fall back to existing Qwen3-8B.
    text_choice = None

    for name in (
        "qwen2_5_7b_instruct",
        "qwen3_8b",
    ):
        entry = registry["models"].get(name, {})

        if entry.get("status") in (
            "LINKED",
            "ALREADY_LINKED",
        ):
            text_choice = name
            break

    if text_choice:
        alias = DEST / "text_llm"
        target = DEST / text_choice

        if not alias.exists() and not alias.is_symlink():
            alias.symlink_to(
                target,
                target_is_directory=True,
            )

        print()
        print("[TEXT LLM SELECTED]", text_choice)
        registry["text_llm_selected"] = text_choice

    else:
        print()
        print("[WARNING] No main text LLM available")

    REPORT.write_text(
        json.dumps(
            registry,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("MODEL LINKS")
    print("=" * 76)

    for p in sorted(DEST.iterdir()):
        if p.is_symlink():
            try:
                print(
                    f"{p.name:28s} -> "
                    f"{p.resolve(strict=True)}"
                )
            except OSError:
                print(f"{p.name:28s} -> BROKEN")

    print()
    print("Registry:", REPORT)
    print("No original model files modified.")


if __name__ == "__main__":
    main()
