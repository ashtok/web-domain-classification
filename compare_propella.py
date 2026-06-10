# compare_propella.py
"""
Cross-check our LLM topic labels against the independent propella annotations.

propella (openeurollm/propella-annotations, name="fineweb-2", split="deu_Latn")
annotates the SAME FineWeb-2 corpus with a `business_sector` field (a list of
sectors). We join it to our 2000 benchmark docs by `id`, map propella sectors to
our categories with an EXACT-NAME mapping (defensible, one sector -> one
category), and measure agreement (Cohen's kappa + raw agreement).

No LLM calls -- pure data join, so it runs while the inference API is down.

Modes:
  (default)        stream propella, join, cache raw sectors, then score.
  --remap          re-score from the cached propella_joined.jsonl WITHOUT
                   re-streaming (instant; use after editing SECTOR_MAP).

Outputs:
  propella_joined.jsonl       per-id: our_label + raw propella sectors (cache)
  propella_comparison.csv     short summary: one row per category
"""
import json
import os
import sys
import gc
from collections import Counter

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from categories import CATEGORY_NAMES

BENCHMARK_PATH = "benchmark.jsonl"
JOINED_PATH    = "propella_joined.jsonl"
RESULTS_CSV    = "propella_comparison.csv"

PROPELLA_NAME  = "fineweb-2"
PROPELLA_SPLIT = "deu_Latn"

# EXACT propella business_sector names -> our category. Tightest defensible map.
# (energy_utilities deliberately EXCLUDED from CLIMATE: it includes non-climate
#  energy such as grid/oil/gas; environmental_services is the clean match.)
SECTOR_MAP = {
    "MEDICAL":       {"healthcare_medical", "pharmaceutical_biotech"},
    "CYBERSECURITY": {"security_cyber"},
    "CLIMATE":       {"environmental_services"},
}


def sector_to_categories(sectors) -> set:
    """Map a doc's list of propella sectors to the set of our categories it hits."""
    secs = {s.lower() for s in sectors}
    return {cat for cat, names in SECTOR_MAP.items() if secs & names}


def stream_and_cache():
    """Stream propella, match our benchmark ids, cache raw sectors to JOINED_PATH."""
    from datasets import load_dataset

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench = {}
        for line in f:
            if line.strip():
                r = json.loads(line)
                bench[r["id"]] = r["gold_label"]
    want_ids = set(bench)
    print(f"Benchmark docs: {len(want_ids)}")

    print("Streaming propella to find matching ids...")
    ds = load_dataset(
        "openeurollm/propella-annotations",
        name=PROPELLA_NAME, split=PROPELLA_SPLIT, streaming=True,
    )

    matched, sector_vocab, scanned = {}, Counter(), 0
    try:
        for row in ds:
            scanned += 1
            rid = row.get("id")
            if rid in want_ids and rid not in matched:
                sectors = row.get("business_sector") or []
                if isinstance(sectors, str):
                    sectors = [sectors]
                matched[rid] = sectors
                sector_vocab.update(s.lower() for s in sectors)
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,}, matched {len(matched)}/{len(want_ids)}")
            if len(matched) == len(want_ids):
                print("  all ids matched, stopping early.")
                break
    finally:
        # Close the streaming reader cleanly so its prefetch thread doesn't
        # retry against closed fds during interpreter shutdown.
        del ds
        gc.collect()

    print(f"\nJoin coverage: {len(matched)}/{len(want_ids)} "
          f"({len(matched)/len(want_ids):.1%}) after scanning {scanned:,} rows")
    if not matched:
        raise SystemExit("No ids matched propella. Check dataset name/split/id format.")

    print("\nPropella business_sector vocabulary (top 40):")
    for sec, n in sector_vocab.most_common(40):
        print(f"  {n:5}  {sec}")

    with open(JOINED_PATH, "w", encoding="utf-8") as f:
        for rid, sectors in matched.items():
            f.write(json.dumps({
                "id": rid,
                "our_label": bench[rid],
                "propella_sectors": sectors,
            }, ensure_ascii=False) + "\n")
    print(f"\nCached {len(matched)} joined rows to {JOINED_PATH}")


def score():
    """Score our labels vs propella mapping from the cached JOINED_PATH."""
    if not os.path.exists(JOINED_PATH):
        raise SystemExit(f"{JOINED_PATH} not found. Run without --remap first.")
    with open(JOINED_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    print(f"\n=== Agreement (our label vs propella sector mapping) ===")
    print("Mapping:")
    for cat, names in SECTOR_MAP.items():
        print(f"  {cat:14} <- {', '.join(sorted(names))}")
    print()

    results = []
    for cat in CATEGORY_NAMES:
        ours = [1 if r["our_label"] == cat else 0 for r in rows]
        prop = [1 if cat in sector_to_categories(r["propella_sectors"]) else 0
                for r in rows]
        if sum(prop) == 0:
            print(f"{cat:14} propella has NO mapped sector -> not comparable")
            results.append({"category": cat, "comparable": False, "n": len(rows)})
            continue
        kappa = cohen_kappa_score(ours, prop)
        p, rec, f1, _ = precision_recall_fscore_support(
            ours, prop, average="binary", zero_division=0)
        agree = sum(1 for a, b in zip(ours, prop) if a == b) / len(rows)
        print(f"{cat:14} kappa={kappa:.3f}  agreement={agree:.1%}  "
              f"(our_pos={sum(ours)}, propella_pos={sum(prop)})")
        results.append({
            "category": cat, "comparable": True, "n": len(rows),
            "cohen_kappa": round(kappa, 4), "agreement": round(agree, 4),
            "our_positive": sum(ours), "propella_positive": sum(prop),
            "precision": round(p, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        })

    import pandas as pd
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved summary to {RESULTS_CSV}")


def main():
    if "--remap" not in sys.argv:
        stream_and_cache()
    else:
        print("--remap: scoring from cache, not re-streaming.")
    score()


if __name__ == "__main__":
    main()
