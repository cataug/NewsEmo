#!/usr/bin/env python3
"""
NewsEmo — Phase 1: Human-annotated iNews baselines.

Models:
    Majority baseline
    TF-IDF + Logistic Regression
    TF-IDF + Linear SVM
    TF-IDF + Random Forest (100 trees)

Evaluation:
    Official train/dev/cold-start test splits
    Leakage-controlled feature construction
    Validation-only hyperparameter selection
    Accuracy, balanced accuracy, macro-F1, weighted-F1
    Post-cluster bootstrap 95% confidence intervals
    Per-source performance

Outputs:
    dataset_audit.json
    validation_search.csv
    heldout_results.csv
    heldout_predictions.csv
    sources_*.csv
    experiment_protocol.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.svm import LinearSVC


SEED = 42
ROOT = Path("/home/tahiti/NewsEmo")

REQUIRED = [
    "Post_ID",
    "Annotator_ID",
    "Message",
    "Link.Text",
    "Discrete",
    "Page.Name",
]


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def load_split(path, name):
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["Post_ID", "Discrete"]).copy()

    df["Post_ID"] = (
        df["Post_ID"].astype(str).str.strip()
    )

    df["Discrete"] = (
        df["Discrete"].astype(str).str.strip()
    )

    df = df[
        (df["Post_ID"] != "") &
        (df["Discrete"] != "")
    ].copy()

    df["news_text"] = (
        df["Message"].fillna("").astype(str).str.strip()
        + " . "
        + df["Link.Text"].fillna("").astype(str).str.strip()
    )

    df = df[df["news_text"].str.len() > 4].copy()

    df["source_group"] = (
        df["Page.Name"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    df["split"] = name

    return df.reset_index(drop=True)


def calculate_metrics(y_true, y_pred, classes):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=classes,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def post_bootstrap(
    y_true,
    y_pred,
    post_ids,
    classes,
    n_bootstrap=1000,
):
    """
    Cluster bootstrap by Post_ID.

    All annotations for a sampled post are resampled together.
    This avoids treating multiple ratings of one post as
    independent observations.
    """

    rng = np.random.default_rng(SEED)

    unique_posts, inverse = np.unique(
        post_ids,
        return_inverse=True,
    )

    groups = [
        np.flatnonzero(inverse == k)
        for k in range(len(unique_posts))
    ]

    results = []

    for _ in range(n_bootstrap):
        sampled_posts = rng.integers(
            0,
            len(groups),
            len(groups),
        )

        indices = np.concatenate([
            groups[j] for j in sampled_posts
        ])

        accuracy = accuracy_score(
            y_true[indices],
            y_pred[indices],
        )

        macro_f1 = f1_score(
            y_true[indices],
            y_pred[indices],
            labels=classes,
            average="macro",
            zero_division=0,
        )

        results.append([accuracy, macro_f1])

    results = np.asarray(results)

    return {
        "accuracy_ci95": np.quantile(
            results[:, 0],
            [0.025, 0.975],
        ).tolist(),
        "macro_f1_ci95": np.quantile(
            results[:, 1],
            [0.025, 0.975],
        ).tolist(),
    }


def build_model_specs():
    return {
        "majority": [
            (
                "constant",
                DummyClassifier(
                    strategy="most_frequent"
                ),
            ),
        ],

        "logreg_tfidf": [
            (
                f"C={c}",
                LogisticRegression(
                    C=c,
                    max_iter=2500,
                    class_weight="balanced",
                ),
            )
            for c in [0.1, 1.0, 10.0]
        ],

        "linear_svm_tfidf": [
            (
                f"C={c}",
                LinearSVC(
                    C=c,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            )
            for c in [0.1, 1.0, 10.0]
        ],

        "rf100_tfidf": [
            (
                "fixed",
                RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=min(8, os.cpu_count() or 1),
                    random_state=SEED,
                ),
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    root = args.root.resolve()

    data_dir = root / "datasets" / "inews_public"
    output_dir = root / "results" / "phase1_inews"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75, flush=True)
    print("NEWS EMOTION — PHASE 1", flush=True)
    print("HUMAN-ANNOTATED INEWS BASELINES", flush=True)
    print("=" * 75, flush=True)

    # --------------------------------------------------------
    # 1. Load original official splits
    # --------------------------------------------------------

    train = load_split(
        data_dir / "train.csv",
        "train",
    )

    dev = load_split(
        data_dir / "dev.csv",
        "dev",
    )

    test = load_split(
        data_dir / "test_cold_start_public.csv",
        "test_cold_start_public",
    )

    print("\nDATASET STATISTICS", flush=True)

    for name, df in [
        ("Train", train),
        ("Dev", dev),
        ("Cold-start test", test),
    ]:
        print(
            f"{name:20s} "
            f"annotations={len(df):5d} "
            f"posts={df['Post_ID'].nunique():5d} "
            f"users={df['Annotator_ID'].nunique():4d} "
            f"sources={df['source_group'].nunique():3d}",
            flush=True,
        )

    if len(dev) < 20 or len(test) < 20:
        raise RuntimeError(
            "Too few development/test examples."
        )

    # --------------------------------------------------------
    # 2. Leakage audit
    # --------------------------------------------------------

    test_overlap = (
        set(test.Post_ID)
        & (set(train.Post_ID) | set(dev.Post_ID))
    )

    if test_overlap:
        raise RuntimeError(
            f"Test leakage detected: "
            f"{len(test_overlap)} shared posts."
        )

    train_dev_overlap = (
        set(train.Post_ID) & set(dev.Post_ID)
    )

    if train_dev_overlap:
        print(
            f"[LEAKAGE GUARD] Removing "
            f"{len(train_dev_overlap)} overlapping posts "
            f"from training.",
            flush=True,
        )

        train = train[
            ~train.Post_ID.isin(train_dev_overlap)
        ].copy().reset_index(drop=True)

    classes = sorted(
        set(train.Discrete)
        | set(dev.Discrete)
        | set(test.Discrete)
    )

    print("\nEmotion classes:", classes, flush=True)

    unseen_classes = (
        set(test.Discrete) - set(train.Discrete)
    )

    if unseen_classes:
        print(
            "[WARNING] Unseen test classes:",
            sorted(unseen_classes),
            flush=True,
        )

    # --------------------------------------------------------
    # 3. Annotation agreement diagnostics
    # --------------------------------------------------------

    agreement = (
        test.groupby("Post_ID")["Discrete"]
        .agg([
            "count",
            lambda x: x.value_counts().iloc[0] / len(x),
        ])
    )

    agreement.columns = [
        "count",
        "majority_fraction",
    ]

    multi_annotated = agreement[
        agreement["count"] >= 2
    ]

    audit = {
        "dataset": "pitehu/inews_public",
        "target": "Human-annotated discrete emotion",
        "input_columns": [
            "Message",
            "Link.Text",
        ],
        "excluded_columns": [
            "Post_ID",
            "Annotator_ID",
            "Discrete",
            "V",
            "A",
            "D",
            "Page.Name",
            "Description",
            "Relevance",
            "Source",
            "User_Prompt",
        ],
        "classes": classes,
        "train_annotations": len(train),
        "dev_annotations": len(dev),
        "test_annotations": len(test),
        "train_posts": train.Post_ID.nunique(),
        "dev_posts": dev.Post_ID.nunique(),
        "test_posts": test.Post_ID.nunique(),
        "test_overlap_posts": len(test_overlap),
        "train_dev_overlap_posts_removed":
            len(train_dev_overlap),
        "multi_annotated_test_posts":
            len(multi_annotated),
        "within_post_majority_fraction":
            (
                float(
                    multi_annotated[
                        "majority_fraction"
                    ].mean()
                )
                if len(multi_annotated)
                else None
            ),
    }

    save_json(
        output_dir / "dataset_audit.json",
        audit,
    )

    # --------------------------------------------------------
    # 4. TF-IDF — fitted only on TRAIN
    # --------------------------------------------------------

    print("\nBUILDING TF-IDF FEATURES", flush=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )

    x_train = vectorizer.fit_transform(
        train.news_text
    )

    x_dev = vectorizer.transform(
        dev.news_text
    )

    x_test = vectorizer.transform(
        test.news_text
    )

    y_train = train.Discrete.to_numpy()
    y_dev = dev.Discrete.to_numpy()
    y_test = test.Discrete.to_numpy()

    print("Train:", x_train.shape, flush=True)
    print("Dev:  ", x_dev.shape, flush=True)
    print("Test: ", x_test.shape, flush=True)

    # --------------------------------------------------------
    # 5. Validation-only hyperparameter selection
    # --------------------------------------------------------

    search_rows = []
    selected_models = {}

    for family, variants in build_model_specs().items():
        print(
            f"\n{'=' * 65}\nMODEL: {family}\n{'=' * 65}",
            flush=True,
        )

        best = None

        for config, model in variants:
            model.fit(x_train, y_train)

            predictions = model.predict(x_dev)

            metrics = calculate_metrics(
                y_dev,
                predictions,
                classes,
            )

            search_rows.append({
                "model": family,
                "configuration": config,
                "split": "dev",
                **metrics,
            })

            print(
                f"  {config:12s} "
                f"macro-F1={metrics['macro_f1']:.4f} "
                f"accuracy={metrics['accuracy']:.4f}",
                flush=True,
            )

            if (
                best is None
                or metrics["macro_f1"] > best[0]
            ):
                best = (
                    metrics["macro_f1"],
                    config,
                    model,
                )

        selected_models[family] = (
            best[1],
            best[2],
        )

    pd.DataFrame(search_rows).to_csv(
        output_dir / "validation_search.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Untouched cold-start evaluation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75,
        flush=True,
    )
    print("FINAL HELD-OUT EVALUATION", flush=True)
    print("=" * 75, flush=True)

    result_rows = []

    prediction_table = test[
        [
            "Post_ID",
            "Annotator_ID",
            "Page.Name",
            "Discrete",
        ]
    ].copy()

    for family, (config, model) in selected_models.items():

        predictions = np.asarray(
            model.predict(x_test)
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
            classes,
        )

        ci = post_bootstrap(
            y_test,
            predictions,
            test.Post_ID.to_numpy(),
            classes,
            n_bootstrap=args.bootstrap,
        )

        result_rows.append({
            "model": family,
            "selected_config_by_validation": config,
            "split": "test_cold_start_public",
            "n_annotations": len(test),
            "n_posts": test.Post_ID.nunique(),
            **metrics,
            "accuracy_ci95_low":
                ci["accuracy_ci95"][0],
            "accuracy_ci95_high":
                ci["accuracy_ci95"][1],
            "macro_f1_ci95_low":
                ci["macro_f1_ci95"][0],
            "macro_f1_ci95_high":
                ci["macro_f1_ci95"][1],
        })

        prediction_table[
            f"pred_{family}"
        ] = predictions

        print(
            f"{family:20s} "
            f"macro-F1={metrics['macro_f1']:.4f} "
            f"CI95={ci['macro_f1_ci95']} "
            f"accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )

        # ----------------------------------------------------
        # 7. Performance by news source
        # ----------------------------------------------------

        source_rows = []

        grouped = test.assign(
            predicted_emotion=predictions
        ).groupby("source_group")

        for source, part in grouped:
            source_metrics = calculate_metrics(
                part.Discrete.to_numpy(),
                part.predicted_emotion.to_numpy(),
                classes,
            )

            source_rows.append({
                "model": family,
                "source": source,
                "n_annotations": len(part),
                "n_posts": part.Post_ID.nunique(),
                **source_metrics,
            })

        pd.DataFrame(source_rows).sort_values(
            "n_annotations",
            ascending=False,
        ).to_csv(
            output_dir / f"sources_{family}.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 8. Save reproducible results
    # --------------------------------------------------------

    results = pd.DataFrame(result_rows)

    results.to_csv(
        output_dir / "heldout_results.csv",
        index=False,
    )

    prediction_table.to_csv(
        output_dir / "heldout_predictions.csv",
        index=False,
    )

    save_json(
        output_dir / "experiment_protocol.json",
        {
            "seed": SEED,
            "dataset": "pitehu/inews_public",
            "test_split": "test_cold_start_public.csv",
            "selection": "Development macro-F1 only",
            "input_features": "TF-IDF of Message + Link.Text",
            "tfidf_fit": "train only",
            "bootstrap_unit": "Post_ID",
            "bootstrap_repetitions": args.bootstrap,
            "limitations": [
                "Text-only experiment",
                "Multiple subjective annotations per post",
                "Cold-start held-out evaluation",
                "News-source analysis is not demographic fairness",
                "Small source groups may have unstable metrics",
            ],
        },
    )

    print("\n" + "=" * 75, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 75, flush=True)

    print(
        results[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "macro_f1_ci95_low",
                "macro_f1_ci95_high",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ),
        flush=True,
    )

    print("\nResults saved to:", output_dir, flush=True)


if __name__ == "__main__":
    main()
