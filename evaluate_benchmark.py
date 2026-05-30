# evaluate_benchmark.py
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

INPUT_PATH = "benchmark_1000_benchmarked.jsonl"
OUTPUT_CSV = "benchmark_results.csv"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if not rows:
    raise SystemExit("No rows found in input file.")

label_keys = [k for k in rows[0].keys() if k.endswith("_label") and k != "gpt_oss_label"]
model_prefixes = [k[:-6] for k in label_keys]

y_true = [r["gpt_oss_label"] for r in rows]
results = []

for prefix in model_prefixes:
    pred_key = f"{prefix}_label"
    lat_key = f"{prefix}_latency_sec"

    y_pred = [r.get(pred_key, "NON_MEDICAL") for r in rows]

    acc = accuracy_score(y_true, y_pred)

    # Get metrics for both classes
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["MEDICAL", "NON_MEDICAL"],
        average=None,
        zero_division=0,
    )

    avg_latency = sum(r.get(lat_key, 0.0) for r in rows) / len(rows)

    results.append({
        "model": prefix,
        "n": len(rows),
        "accuracy": acc,
        "precision_medical": float(p[0]),
        "recall_medical": float(r[0]),
        "f1_medical": float(f1[0]),
        "precision_non_medical": float(p[1]),
        "recall_non_medical": float(r[1]),
        "f1_non_medical": float(f1[1]),
        "avg_latency_sec": avg_latency,
    })

df = pd.DataFrame(results).sort_values(
    by=["f1_medical", "accuracy", "avg_latency_sec"],
    ascending=[False, False, True],
)

df.to_csv(OUTPUT_CSV, index=False)

print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved to {OUTPUT_CSV}")
