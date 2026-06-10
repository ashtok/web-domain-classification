# compare_propella.py
"""
Cross-check our LLM topic labels against the independent propella annotations.

propella (openeurollm/propella-annotations, name="fineweb-2", split="deu_Latn")
annotates the SAME FineWeb-2 corpus with, among other things, a `business_sector`
field (a list of sectors, e.g. healthcare_medical, pharmaceutical_biotech). We
join it to our 2000 benchmark docs by document `id`, map propella sectors to our
categories, and measure agreement.

No LLM calls — pure data join, so it runs fine while the inference API is down.

Reads : benchmark.jsonl                 (our gold_label per id)
        openeurollm/propella-annotations (streamed, indexed by id)
Writes : propella_comparison.csv          (per-category precision/recall + kappa)
         propella_joined.jsonl            (joined rows for inspection)

Because propella is BUSINESS SECTORS, MEDICAL maps cleanly; CYBERSECURITY and
CLIMATE may have no corresponding sector -> the comparison may only be
meaningful for MEDICAL. The script prints the full sector vocabulary so the
mapping below can be refined after seeing what actually exists.
"""
import json
import os
from collections import Counter

from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from categories import CATEGORY_NAMES, OTHER

BENCHMARK_PATH = "benchmark.jsonl"
JOINED_PATH    = "propella_joined.jsonl"
RESULTS_CSV    = "propella_comparison.csv"

PROPELLA_NAME  = "fineweb-2"
PROPELLA_SPLIT = "deu_Latn"

# Substring rules mapping a propella business_sector -> our category.
# Refine these after the printed sector vocabulary shows what really exists.
SECTOR_RULES = {
    "MEDICAL":       ["health", "medic", "pharma", "biotech", "hospital", "clinic"],
    "CYBERSECURITY": ["security", "cyber", "information_tech", "software", "it_"],
    "CLIMATE":       ["climate", "environment", "energy", "renewable", "sustainab"],
}


def sector_to_categories(sectors) -> set:
    """Map a doc's list of propella sectors to the set of our categories it hits."""
    cats = set()
    joined = " ".join(s.lower() for s in sectors)
    for cat, needles in SECTOR_RULES.items():
        if any(n in joined for n in needles):
            cats.add(cat)
    return cats


def main():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench = {json.loads(l)["id"]: json.loads(l)
                 for l in (line for line in f if line.strip())}
    want_ids = set(bench)
    print(f"Benchmark docs: {len(want_ids)}")

    print("Streaming propella to find matching ids (this can take a while)...")
    ds = load_dataset(
        "openeurollm/propella-annotations",
        name=PROPELLA_NAME,
        split=PROPELLA_SPLIT,
        streaming=True,
    )

    matched = {}
    sector_vocab = Counter()
    scanned = 0
    for row in ds:
        scanned += 1
        rid = row.get("id")
        if rid in want_ids and rid not in matched:
            sectors = row.get("business_sector") or []
            if isinstance(sectors, str):
                sectors = [sectors]
            matched[rid] = {
                "business_sector": sectors,
                "one_sentence_description": row.get("one_sentence_description", ""),
            }
            sector_vocab.update(s.lower() for s in sectors)
        if scanned % 500_000 == 0:
            print(f"  scanned {scanned:,}, matched {len(matched)}/{len(want_ids)}")
        if len(matched) == len(want_ids):
            print("  all ids matched, stopping early.")
            break

    print(f"\nJoin coverage: {len(matched)}/{len(want_ids)} "
          f"({len(matched)/len(want_ids):.1%}) after scanning {scanned:,} rows")
    if not matched:
        raise SystemExit("No ids matched propella. Check dataset name/split/id format.")

    print("\nPropella business_sector vocabulary (top 40):")
    for sec, n in sector_vocab.most_common(40):
        print(f"  {n:5}  {sec}")

    # Build joined records + per-category binary vectors.
    joined = []
    for rid, prop in matched.items():
        b = bench[rid]
        prop_cats = sector_to_categories(prop["business_sector"])
        joined.append({
            "id": rid,
            "our_label": b["gold_label"],
            "propella_sectors": prop["business_sector"],
            "propella_cats": sorted(prop_cats),
        })

    with open(JOINED_PATH, "w", encoding="utf-8") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-category agreement: treat as one-vs-rest.
    #   ours_pos     = our single-label gold == this category
    #   propella_pos = this category is in propella's mapped sectors
    print("\n=== Per-category agreement (our label vs propella sector mapping) ===")
    results = []
    for cat in CATEGORY_NAMES:
        ours = [1 if r["our_label"] == cat else 0 for r in joined]
        prop = [1 if cat in r["propella_cats"] else 0 for r in joined]
        if sum(prop) == 0:
            print(f"{cat:14} propella has NO mapped sector -> not comparable")
            results.append({"category": cat, "comparable": False})
            continue
        kappa = cohen_kappa_score(ours, prop)
        # Precision/recall of propella-positive as a predictor of our label.
        p, r_, f1, _ = precision_recall_fscore_support(
            ours, prop, average="binary", zero_division=0
        )
        agree = sum(1 for a, b2 in zip(ours, prop) if a == b2) / len(ours)
        print(f"{cat:14} kappa={kappa:.3f}  agreement={agree:.1%}  "
              f"( our_pos={sum(ours)}, propella_pos={sum(prop)})")
        results.append({
            "category": cat, "comparable": True, "n": len(joined),
            "cohen_kappa": kappa, "agreement": agree,
            "our_positive": sum(ours), "propella_positive": sum(prop),
            "precision": p, "recall": r_, "f1": f1,
        })

    import pandas as pd
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-category scores to {RESULTS_CSV}")
    print(f"Joined rows written to {JOINED_PATH}")


if __name__ == "__main__":
    main()
