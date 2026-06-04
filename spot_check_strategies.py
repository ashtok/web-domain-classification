# spot_check_strategies.py
"""
Spot-check the benchmark + strategy predictions by printing example documents.

Three modes (pick with MODE):
  - "by_category" : N random gold examples per category (verify gold labels)
  - "disagree"    : docs where multiclass != binary for a given model
  - "vs_gold"     : docs where a model's multiclass pred != teacher gold_label

Read-only: reads benchmark_strategies.jsonl, prints to stdout. No API calls.
"""
import json
import random
import textwrap

from categories import MULTICLASS_LABELS

INPUT_PATH = "benchmark_strategies.jsonl"

MODE         = "by_category"   # "by_category" | "disagree" | "vs_gold"
N            = 4               # examples per group
MODEL        = "RedHatAI_gemma_3_27b_it_quantized.w4a16"  # for disagree / vs_gold
TEXT_CHARS   = 400
SEED         = 0


def show(row):
    print(f"  id:   {row.get('id')}")
    print(f"  url:  {row.get('url','')}")
    print(f"  gold: {row.get('gold_label')}")
    for k in sorted(row):
        if k.endswith("_pred"):
            print(f"  {k}: {row[k]}")
        if k.endswith("_binary_raw"):
            print(f"  {k}: {row[k]}")
    snippet = (row.get("text") or "")[:TEXT_CHARS].replace("\n", " ")
    print("  text:", textwrap.shorten(snippet, width=TEXT_CHARS, placeholder=" ..."))
    print("-" * 70)


def main():
    random.seed(SEED)
    rows = [json.loads(l) for l in open(INPUT_PATH, encoding="utf-8") if l.strip()]

    mc_key  = f"{MODEL}__multiclass_pred"
    bin_key = f"{MODEL}__binary_pred"

    if MODE == "by_category":
        for cat in MULTICLASS_LABELS:
            pool = [r for r in rows if r.get("gold_label") == cat]
            print("=" * 70)
            print(f"GOLD = {cat}   (showing {min(N, len(pool))} of {len(pool)})")
            print("=" * 70)
            for r in random.sample(pool, min(N, len(pool))):
                show(r)

    elif MODE == "disagree":
        pool = [r for r in rows if r.get(mc_key) != r.get(bin_key)]
        print("=" * 70)
        print(f"multiclass != binary for {MODEL}: {len(pool)} docs "
              f"(showing {min(N, len(pool))})")
        print("=" * 70)
        for r in random.sample(pool, min(N, len(pool))):
            show(r)

    elif MODE == "vs_gold":
        pool = [r for r in rows if r.get(mc_key) != r.get("gold_label")]
        print("=" * 70)
        print(f"multiclass != gold for {MODEL}: {len(pool)} docs "
              f"(showing {min(N, len(pool))})")
        print("=" * 70)
        for r in random.sample(pool, min(N, len(pool))):
            show(r)

    else:
        raise SystemExit(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
