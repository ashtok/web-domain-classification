# evaluate_strategies.py
"""
Compare the multi-class vs binary classification strategies.

For every (model, strategy) pair, score the predicted category against the
teacher `gold_label` and report:
  - accuracy            (overall, all categories incl. OTHER)
  - macro_f1            (unweighted mean F1 over the real categories)
  - f1_<CATEGORY>       (per-category F1, for MEDICAL/CYBERSECURITY/CLIMATE)
  - avg_latency_sec     (mean per-doc cost of that strategy)

Two CSVs are written:
  - benchmark_strategies_results.csv  (one row per model x strategy)
  - benchmark_strategies_delta.csv    (binary - multiclass, per model)

The delta table is the direct answer to "are there huge differences?".
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from categories import CATEGORY_NAMES, MULTICLASS_LABELS

INPUT_PATH   = "benchmark_strategies.jsonl"
RESULTS_CSV  = "benchmark_strategies_results.csv"
DELTA_CSV    = "benchmark_strategies_delta.csv"
STRATEGIES   = ["multiclass", "binary"]


def model_prefixes(row) -> list[str]:
    """Discover model prefixes from the *__multiclass_pred keys."""
    return sorted({
        k.split("__", 1)[0]
        for k in row.keys()
        if k.endswith("__multiclass_pred")
    })


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit("No rows found in input file.")

    y_true = [r["gold_label"] for r in rows]
    prefixes = model_prefixes(rows[0])

    results = []
    for m in prefixes:
        for strat in STRATEGIES:
            pred_key = f"{m}__{strat}_pred"
            lat_key  = f"{m}__{strat}_latency_sec"

            # Skip strategies that were never run for this model.
            if not any(pred_key in r for r in rows):
                continue

            y_pred = [r.get(pred_key, "OTHER") for r in rows]

            acc = accuracy_score(y_true, y_pred)
            macro = f1_score(
                y_true, y_pred,
                labels=CATEGORY_NAMES, average="macro", zero_division=0,
            )
            per_cat = f1_score(
                y_true, y_pred,
                labels=CATEGORY_NAMES, average=None, zero_division=0,
            )
            avg_lat = sum(r.get(lat_key, 0.0) for r in rows) / len(rows)

            entry = {
                "model": m,
                "strategy": strat,
                "n": len(rows),
                "accuracy": acc,
                "macro_f1": macro,
                "avg_latency_sec": avg_lat,
            }
            for cat, f1v in zip(CATEGORY_NAMES, per_cat):
                entry[f"f1_{cat}"] = float(f1v)
            results.append(entry)

    df = pd.DataFrame(results)
    cat_cols = [f"f1_{c}" for c in CATEGORY_NAMES]
    col_order = (
        ["model", "strategy", "n", "accuracy", "macro_f1"]
        + cat_cols + ["avg_latency_sec"]
    )
    df = df[col_order].sort_values(["model", "strategy"])
    df.to_csv(RESULTS_CSV, index=False)

    # Delta table: binary - multiclass per model (the headline comparison).
    delta_rows = []
    for m in df["model"].unique():
        mc = df[(df.model == m) & (df.strategy == "multiclass")]
        bn = df[(df.model == m) & (df.strategy == "binary")]
        if mc.empty or bn.empty:
            continue
        mc, bn = mc.iloc[0], bn.iloc[0]
        delta = {"model": m}
        for col in ["accuracy", "macro_f1"] + cat_cols + ["avg_latency_sec"]:
            delta[f"d_{col}"] = bn[col] - mc[col]
        delta_rows.append(delta)
    ddf = pd.DataFrame(delta_rows)
    ddf.to_csv(DELTA_CSV, index=False)

    pd.set_option("display.width", 200)
    print("=== Per model x strategy ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {RESULTS_CSV}")
    print("\n=== Delta (binary - multiclass) ===")
    print(ddf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nSaved to {DELTA_CSV}")


if __name__ == "__main__":
    main()
