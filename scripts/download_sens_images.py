#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path("/home/tahiti/NewsEmo")

ANNOTATIONS = (
    ROOT / "datasets" / "sens_visualnews" / "dataset"
)

OUTPUT = ROOT / "datasets" / "visualnews_original"
REPORT = ROOT / "reports" / "sens_images_report.json"


def canonical_path(value):
    """
    Convert a VisualNews path to:
    source/images/folder/file.jpg

    Handles tar entries with leading ./ or origin/.
    """
    text = str(value).replace("\\", "/")
    parts = Path(text).parts

    if ".." in parts:
        return None

    if len(parts) < 4:
        return None

    tail = parts[-4:]

    if tail[1] != "images":
        return None

    return "/".join(tail)


def load_required_images():
    required = set()

    annotation_files = [
        "full_dev.json",
        "full_test.json",
        "strict_dev.json",
        "strict_test.json",
    ]

    for filename in annotation_files:
        path = ANNOTATIONS / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            image = canonical_path(record["image"])

            if image is None:
                raise ValueError(
                    f"Invalid image path: {record['image']}"
                )

            required.add(image)

        print(
            f"[ANNOTATIONS] {filename}: {len(records)} rows",
            flush=True,
        )

    return required


def is_complete(path):
    return path.is_file() and path.stat().st_size > 512


def save_report(required):
    found = sorted(
        p for p in required
        if is_complete(OUTPUT / p)
    )

    missing = sorted(set(required) - set(found))

    report = {
        "dataset": "Sens-VisualNews",
        "required": len(required),
        "downloaded": len(found),
        "missing": len(missing),
        "missing_paths": missing,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("SENS-VISUALNEWS IMAGE STATUS")
    print("=" * 65)
    print("Required:", len(required))
    print("Present: ", len(found))
    print("Missing: ", len(missing))
    print("Output:  ", OUTPUT)
    print("Report:  ", REPORT)

    if missing:
        print("First missing files:")
        for p in missing[:10]:
            print("  ", p)

    return not missing


def extract_stream(required):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    remaining = {
        p for p in required
        if not is_complete(OUTPUT / p)
    }

    print(
        f"\nAlready present: {len(required) - len(remaining)}",
        flush=True,
    )
    print(f"Still needed:   {len(remaining)}", flush=True)

    if not remaining:
        print("[OK] All images already present")
        return

    scanned = 0
    extracted = 0
    bytes_written = 0

    print("\nReading VisualNews TAR stream...", flush=True)

    with tarfile.open(
        fileobj=sys.stdin.buffer,
        mode="r|",
        bufsize=1024 * 1024,
    ) as archive:

        for member in archive:
            scanned += 1

            if not member.isfile():
                continue

            name = canonical_path(member.name)

            if name not in remaining:
                if scanned % 50000 == 0:
                    print(
                        f"[SCAN] entries={scanned:,} "
                        f"extracted={extracted:,} "
                        f"remaining={len(remaining):,}",
                        flush=True,
                    )
                continue

            destination = OUTPUT / name

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = destination.with_suffix(
                destination.suffix + ".partial"
            )

            source = archive.extractfile(member)

            if source is None:
                continue

            try:
                with open(temporary, "wb") as dst:
                    shutil.copyfileobj(
                        source,
                        dst,
                        length=1024 * 1024,
                    )

                if temporary.stat().st_size <= 512:
                    temporary.unlink(missing_ok=True)
                    continue

                os.replace(temporary, destination)

                extracted += 1
                bytes_written += destination.stat().st_size
                remaining.remove(name)

                if extracted % 250 == 0:
                    print(
                        f"[EXTRACT] {extracted:,} new "
                        f"| remaining={len(remaining):,} "
                        f"| saved={bytes_written/1024**3:.2f} GiB",
                        flush=True,
                    )

            except Exception:
                temporary.unlink(missing_ok=True)
                raise

    print("\nFinished scanning archive", flush=True)
    print("Archive entries scanned:", scanned)
    print("New images extracted:", extracted)
    print("New bytes written:", bytes_written)

    if remaining:
        print(
            "[WARNING] Some requested images were not found "
            "in the source archive.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check local image coverage",
    )

    args = parser.parse_args()

    required = load_required_images()

    print("\nUnique required images:", len(required), flush=True)

    if args.check:
        complete = save_report(required)
        sys.exit(0 if complete else 2)

    try:
        extract_stream(required)
    finally:
        save_report(required)


if __name__ == "__main__":
    main()
