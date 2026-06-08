# evaluate_multilabel.py
"""
Score the multi-label predictions against the multi-label gold.

Multi-label needs different metrics than single-label accuracy:
  - exact_match    : fraction of docs whose predicted set == gold set (strict)
  - hamming_loss   : fraction of (doc, label) cells that are wrong (lower better)
  - micro_f1       : F1 pooled over all label decisions
  - macro_f1       : unweighted mean F1 over the real categories
  - f1_<CATEGORY>  : per-category F1 (treating each as one-vs-rest)

Reads : benchmark_multilabel_pred.jsonl   (gold_labels + per-model preds)
Writes: benchmark_multilabel_results.csv
"""
import json

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,       # exact match on binarized rows = subset accuracy
    hamming_loss,
)

from categories import CATEGORY_NAMES, MULTICLASS_LABELS

INPUT_PATH  = "benchmark_multilabel_pred.jsonl"
RESULTS_CSV = "benchmark_multilabel_results.csv"

# Binarize over ALL labels (incl. OTHER) so exact-match / hamming see OTHER too,
# but report per-category F1 only for the real categories.
mlb = MultiLabelBinarizer(classes=MULTICLASS_LABELS)
mlb.fit([MULTICLASS_LABELS])
CAT_IDX = [MULTICLASS_LABELS.index(c) for c in CATEGORY_NAMES]


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        raise SystemExit("No rows found.")

    prefixes = sorted({
        k.split("__", 1)[0] for k in rows[0] if k.endswith("__multilabel_pred")
    })

    y_true = mlb.transform([set(r["gold_labels"]) for r in rows])

    results = []
    for m in prefixes:
        pred_key = f"{m}__multilabel_pred"
        lat_key  = f"{m}__multilabel_latency_sec"
        if not any(pred_key in r for r in rows):
            continue

        y_pred = mlb.transform([set(r.get(pred_key, ["OTHER"])) for r in rows])

        # Per-category F1 (only the real categories' columns).
        per_cat = f1_score(
            y_true[:, CAT_IDX], y_pred[:, CAT_IDX],
            average=None, zero_division=0,
        )
        entry = {
            "model": m,
            "n": len(rows),
            "exact_match": accuracy_score(y_true, y_pred),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "micro_f1": f1_score(
                y_true[:, CAT_IDX], y_pred[:, CAT_IDX],
                average="micro", zero_division=0,
            ),
            "macro_f1": f1_score(
                y_true[:, CAT_IDX], y_pred[:, CAT_IDX],
                average="macro", zero_division=0,
            ),
            "avg_latency_sec": sum(r.get(lat_key, 0.0) for r in rows) / len(rows),
        }
        for cat, f1v in zip(CATEGORY_NAMES, per_cat):
            entry[f"f1_{cat}"] = float(f1v)
        results.append(entry)

    cat_cols = [f"f1_{c}" for c in CATEGORY_NAMES]
    col_order = (
        ["model", "n", "exact_match", "hamming_loss", "micro_f1", "macro_f1"]
        + cat_cols + ["avg_latency_sec"]
    )
    df = pd.DataFrame(results)[col_order].sort_values("macro_f1", ascending=False)
    df.to_csv(RESULTS_CSV, index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {RESULTS_CSV}")

    # How many gold docs are actually multi-label? (context for the comparison)
    multi = sum(1 for r in rows if len(set(r["gold_labels"]) & set(CATEGORY_NAMES)) > 1)
    print(f"\nGold docs with >1 real category: {multi}/{len(rows)}")


if __name__ == "__main__":
    main()
