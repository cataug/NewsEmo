#!/usr/bin/env python3

import csv
import json
import os
import shutil
import sys
import zipfile

from pathlib import Path
from collections import Counter

R = Path("/home/tahiti/NewsEmo")
D = R / "datasets"
OUT = R / "reports" / "phase2_audit.json"

csv.field_size_limit(10_000_000)

IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

NAMES = [
    "inews_public",
    "inews_screenshots",
    "goodnewseveryone",
    "semeval2007_affective_text",
    "sens_visualnews",
    "mirage_news",
    "newsclippings_ready",
    "image_text_fake_news",
    "n24news",
    "visualnews_original",
]

report = {
    "root": str(R),
    "datasets": {},
    "issues": [],
}


def walk(folder):
    ans = []

    if not folder.is_dir():
        return ans

    for base, dirs, files in os.walk(folder):
        dirs[:] = [
            x for x in dirs
            if x not in (".git", ".cache", "__pycache__")
        ]

        ans += [
            Path(base) / f
            for f in files
            if not (Path(base) / f).is_symlink()
        ]

    return ans


def jshape(obj):
    if isinstance(obj, list):
        s = obj[0] if obj else None

        data = {
            "records": len(obj),
            "fields": list(s)[:35] if isinstance(s, dict) else [],
        }

        if isinstance(s, dict):
            for key in (
                "gt", "label", "emotion",
                "Discrete", "category"
            ):
                if key in s:
                    data["label_counts"] = dict(
                        Counter(
                            str(x.get(key))
                            for x in obj
                        ).most_common(30)
                    )
                    data["label_field"] = key
                    break

        return data

    if isinstance(obj, dict):
        return {
            "root_keys": list(obj)[:35],
            "array_sizes": {
                k: len(v)
                for k, v in obj.items()
                if isinstance(v, list)
            },
        }

    return {"type": type(obj).__name__}


def jsoninfo(p):
    data = {"file": str(p.relative_to(R))}

    try:
        if p.stat().st_size > 125_000_000:
            return {
                **data,
                "skipped": "over 125 MB",
            }

        with p.open(encoding="utf-8-sig") as f:
            data.update(jshape(json.load(f)))

    except Exception as e:
        data["error"] = str(e)

    return data


def tableinfo(p):
    info = {"file": str(p.relative_to(R))}

    try:
        with p.open(
            newline="",
            encoding="utf-8-sig",
            errors="replace",
        ) as f:

            reader = csv.DictReader(
                f,
                delimiter="\t" if p.suffix == ".tsv" else ",",
            )

            fields = reader.fieldnames or []
            info["fields"] = fields[:60]

            labels = Counter()
            sources = Counter()
            posts = set()
            missing = Counter()
            rows = 0

            for row in reader:
                rows += 1

                if "Discrete" in fields:
                    labels[row.get("Discrete", "")] += 1

                if "Page.Name" in fields:
                    sources[row.get("Page.Name", "")] += 1

                if "Post_ID" in fields:
                    posts.add(row.get("Post_ID", ""))

                for key in fields:
                    if not str(row.get(key) or "").strip():
                        missing[key] += 1

            info.update(
                rows=rows,
                unique_posts=len(posts),
                label_counts=dict(labels.most_common(25)),
                source_counts=dict(sources.most_common(25)),
                missing_counts=dict(missing.most_common(20)),
            )

            if "Post_ID" in fields:
                info["_posts"] = posts

    except Exception as e:
        info["error"] = str(e)

    return info


def zipinfo(p):
    info = {"file": str(p.relative_to(R))}

    try:
        with zipfile.ZipFile(p) as z:
            fs = [
                f for f in z.infolist()
                if not f.is_dir()
            ]

            pics = [
                f for f in fs
                if Path(f.filename).suffix.lower() in IMG
            ]

            info.update(
                valid=True,
                members=len(fs),
                images=len(pics),
                uncompressed_bytes=sum(
                    f.file_size for f in fs
                ),
                image_examples=[
                    f.filename for f in pics[:3]
                ],
            )

            # Проверяем CRC трёх изображений.
            # Весь архив не распаковываем.
            checked = 0

            for item in pics:
                if item.file_size < 5_000_000:
                    with z.open(item) as h:
                        while h.read(1024 * 1024):
                            pass

                    checked += 1

                if checked == 3:
                    break

            info["crc_checked_images"] = checked

            # Проверяем JSON-разметку внутри ZIP.
            info["json_metadata"] = []

            for item in fs:
                if (
                    item.filename.endswith(".json")
                    and item.file_size < 80_000_000
                ):
                    try:
                        with z.open(item) as h:
                            meta = jshape(json.load(h))

                        info["json_metadata"].append({
                            "file": item.filename,
                            **meta,
                        })

                    except Exception as e:
                        info["json_metadata"].append({
                            "file": item.filename,
                            "error": str(e),
                        })

                if len(info["json_metadata"]) >= 3:
                    break

    except Exception as e:
        info.update(
            valid=False,
            error=str(e),
        )

    return info


def parquetinfo(p):
    info = {"file": str(p.relative_to(R))}

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(p)
        fields = pf.schema_arrow

        info.update(
            valid=True,
            rows=pf.metadata.num_rows,
            columns=fields.names,
            row_groups=pf.metadata.num_row_groups,
            possible_image_bytes=[
                str(x)[:180]
                for x in fields
                if (
                    "image" in x.name.lower()
                    and (
                        "binary" in str(x.type).lower()
                        or "struct" in str(x.type).lower()
                    )
                )
            ],
        )

    except ImportError:
        info.update(
            valid=None,
            reason="pyarrow unavailable",
        )

    except Exception as e:
        info.update(
            valid=False,
            error=str(e),
        )

    return info


if not D.is_dir():
    sys.exit(f"ERROR: no datasets directory: {D}")


print("=== NEWSEMO PHASE 2: READ-ONLY AUDIT ===", flush=True)

files_by_name = {}

# ----------------------------------------------------------
# 1. INVENTORY
# ----------------------------------------------------------

for name in NAMES:
    paths = walk(D / name)
    files_by_name[name] = paths

    pics = sum(
        p.suffix.lower() in IMG
        for p in paths
    )

    size = sum(
        p.stat().st_size
        for p in paths
    )

    info = {
        "files": len(paths),
        "images_on_disk": pics,
        "size_GB": round(size / 1e9, 3),
        "file_types": dict(
            Counter(
                p.suffix.lower() or "[none]"
                for p in paths
            )
        ),
        "archives": [],
        "parquet": [],
        "structured": [],
        "partial": [],
    }

    report["datasets"][name] = info

    print(
        f"{name:27s} "
        f"files={len(paths):6d} "
        f"images={pics:6d} "
        f"GB={size/1e9:.3f}",
        flush=True,
    )


# ----------------------------------------------------------
# 2. DATA / ARCHIVE VALIDATION
# ----------------------------------------------------------

print("\n=== FILES / LABELS / ARCHIVES ===", flush=True)

for name, paths in files_by_name.items():

    info = report["datasets"][name]

    for p in paths:
        ext = p.suffix.lower()

        if ext == ".zip":
            x = zipinfo(p)
            info["archives"].append(x)

            print(
                "[ZIP]", x["file"],
                "valid=", x.get("valid"),
                "images=", x.get("images", 0),
                "files=", x.get("members", 0),
                flush=True,
            )

            for meta in x.get("json_metadata", []):
                print(
                    "    [INSIDE JSON]",
                    meta["file"],
                    "records=", meta.get("records"),
                    "fields=",
                    meta.get(
                        "fields",
                        meta.get("root_keys", []),
                    )[:12],
                    flush=True,
                )

        elif ext == ".parquet":
            x = parquetinfo(p)
            info["parquet"].append(x)

            print(
                "[PARQUET]",
                x["file"],
                "rows=", x.get("rows"),
                "columns=", x.get("columns", [])[:15],
                "image_bytes=",
                bool(x.get("possible_image_bytes")),
                flush=True,
            )

        elif ext == ".part":
            with p.open("rb") as f:
                b = f.read(80)

            sig = (
                "HTML/XML"
                if b.lstrip().startswith(b"<")
                else repr(b[:24])
            )

            info["partial"].append({
                "file": str(p.relative_to(R)),
                "bytes": p.stat().st_size,
                "signature": sig,
            })

            print(
                "[PART]",
                p.relative_to(R),
                "bytes=", p.stat().st_size,
                "signature=", sig,
                flush=True,
            )

        elif name in (
            "inews_public",
            "goodnewseveryone",
            "semeval2007_affective_text",
            "sens_visualnews",
        ):

            if ext in (".csv", ".tsv"):
                x = tableinfo(p)
                info["structured"].append(x)

                print(
                    "[TABLE]",
                    x["file"],
                    "rows=", x.get("rows"),
                    "classes=",
                    list(x.get("label_counts", {}))[:12],
                    flush=True,
                )

            elif ext == ".json":
                x = jsoninfo(p)
                info["structured"].append(x)

                print(
                    "[JSON]",
                    x["file"],
                    "records=", x.get("records"),
                    "keys=",
                    x.get(
                        "fields",
                        x.get("root_keys", []),
                    )[:12],
                    flush=True,
                )

            elif (
                ext in (".txt", ".conll", ".ann")
                and p.stat().st_size < 2_000_000
            ):
                with p.open(
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    lines = sum(
                        bool(line.strip())
                        for line in f
                    )

                info["structured"].append({
                    "file": str(p.relative_to(R)),
                    "nonempty_lines": lines,
                })

                print(
                    "[TEXT]",
                    p.relative_to(R),
                    "nonempty_lines=", lines,
                    flush=True,
                )


# ----------------------------------------------------------
# 3. INEWS SPLIT OVERLAP
# ----------------------------------------------------------

print("\n=== INEWS CROSS-SPLIT POST OVERLAP ===", flush=True)

splits = {}

for x in report["datasets"]["inews_public"]["structured"]:
    if (
        x.get("file", "").endswith(".csv")
        and "_posts" in x
    ):
        splits[Path(x["file"]).name] = x.pop("_posts")

overlap = {}

for i, a in enumerate(sorted(splits)):
    for b in sorted(splits)[i + 1:]:
        n = len(splits[a] & splits[b])

        overlap[a + " <> " + b] = n

        if n:
            print(
                "[SHARED POSTS]",
                a, b, n,
                "(may be intended)",
                flush=True,
            )

report["inews_shared_posts"] = overlap


# ----------------------------------------------------------
# 4. DATA READINESS
# ----------------------------------------------------------

print("\n=== DATA READINESS ===", flush=True)

I = report["datasets"]

newsclip = sum(
    x.get("images", 0)
    for x in I["newsclippings_ready"]["archives"]
    if x.get("valid")
)

mirage_rows = sum(
    x.get("rows", 0)
    for x in I["mirage_news"]["parquet"]
)

checks = {
    "iNews human annotation tables":
        any(
            x.get("rows", 0) > 0
            for x in I["inews_public"]["structured"]
        ),

    "iNews original screenshots":
        I["inews_screenshots"]["images_on_disk"] > 0,

    "GoodNewsEveryone data":
        bool(I["goodnewseveryone"]["structured"])
        or bool(I["goodnewseveryone"]["archives"]),

    "SemEval data":
        bool(I["semeval2007_affective_text"]["structured"]),

    "Sens-VisualNews labels":
        bool(I["sens_visualnews"]["structured"]),

    "Sens original images":
        I["visualnews_original"]["images_on_disk"] > 0,

    "MiRAGeNews Parquet present":
        len(I["mirage_news"]["parquet"]) > 0,

    "MiRAGeNews Parquet readable":
        mirage_rows > 0,

    "NewsCLIPpings images inside valid ZIP":
        newsclip > 0,

    "N24News image collection":
        I["n24news"]["images_on_disk"] >= 1000,

    "Figshare valid archive":
        any(
            x.get("valid")
            for x in I["image_text_fake_news"]["archives"]
        ),
}

for key, val in checks.items():
    print(
        ("[YES] " if val else "[ NO] ") + key,
        flush=True,
    )

report["readiness"] = checks

report["summary"] = {
    "mirage_rows": mirage_rows,
    "newsclippings_images_in_zip": newsclip,
    "disk_free_GB": round(
        shutil.disk_usage(R).free / 1e9,
        2,
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)

OUT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print("\nSaved:", OUT, flush=True)
print("No datasets modified. No installations.", flush=True)
