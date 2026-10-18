from pathlib import Path

ROOT = Path("/home/tahiti/NewsEmo")
MODELS = ROOT / "models"

# Text LLMs
TEXT_LLM = MODELS / "qwen2_5_7b_instruct"
QWEN3_8B = MODELS / "qwen3_8b"
QWEN_CODER = MODELS / "qwen2_5_coder_7b"

# Vision-language models
VISION_LLM = MODELS / "qwen2_5_vl_7b"
VISION_LLM_SMALL = MODELS / "qwen2_5_vl_3b"

# Image-text encoders
CLIP_BASE = MODELS / "clip_vit_b32"
CLIP_LARGE = MODELS / "clip_vit_l14"

# Unimodal encoders
DINO = MODELS / "dinov2_small"
MINILM = MODELS / "minilm_l6_v2"

ALL_MODELS = {
    "Qwen2.5-7B": TEXT_LLM,
    "Qwen3-8B": QWEN3_8B,
    "Qwen2.5-Coder-7B": QWEN_CODER,
    "Qwen2.5-VL-7B": VISION_LLM,
    "Qwen2.5-VL-3B": VISION_LLM_SMALL,
    "CLIP-B32": CLIP_BASE,
    "CLIP-L14": CLIP_LARGE,
    "DINOv2-S": DINO,
    "MiniLM-L6": MINILM,
}

if __name__ == "__main__":
    print("=" * 70)
    print("NEWSEMO MODEL REGISTRY")
    print("=" * 70)

    for name, path in ALL_MODELS.items():
        status = "OK" if path.is_dir() else "MISSING"
        print(f"[{status:7}] {name:22} {path.resolve()}")

    missing = [
        name for name, path in ALL_MODELS.items()
        if not path.is_dir()
    ]

    if missing:
        raise SystemExit(f"Missing models: {missing}")

    print("\nALL MODEL DIRECTORIES AVAILABLE")
