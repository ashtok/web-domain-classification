# audit_propella_sample.py
"""
PHASE 1 of the propella tag-quality audit (NO LLM -- runs during API downtime).

To test "how good are propella's tags?" we must sample CONDITIONED ON PROPELLA,
not on our own labels. This streams propella once and reservoir-samples N random
docs for each target sector, saving each doc's id, full sector list, and
propella's own one_sentence_description (and text if present) for later judging.

This does NOT touch FineWeb-2 -- propella is small and fast to stream (our join
found all 2000 ids within ~300k rows), and propella already carries a short
description per doc, which is enough to judge whether its tag is plausible.

Writes: propella_audit_sample.jsonl   (the sampled docs to judge in Phase 2)
"""
import json
import os
import random

from datasets import load_dataset

PROPELLA_NAME  = "fineweb-2"
PROPELLA_SPLIT = "deu_Latn"
OUTPUT_PATH    = "propella_audit_sample.jsonl"

N_PER_SECTOR   = 1000
SEED           = 0

# propella sectors to audit, and the category each is supposed to mean for us.
TARGET_SECTORS = {
    "healthcare_medical":     "MEDICAL",
    "pharmaceutical_biotech": "MEDICAL",
    "security_cyber":         "CYBERSECURITY",
    "environmental_services": "CLIMATE",
}


def main():
    random.seed(SEED)
    ds = load_dataset(
        "openeurollm/propella-annotations",
        name=PROPELLA_NAME, split=PROPELLA_SPLIT, streaming=True,
    )

    # Reservoir sampling per sector: one pass, uniform random N per sector,
    # without holding the whole dataset in memory.
    reservoir = {s: [] for s in TARGET_SECTORS}
    seen = {s: 0 for s in TARGET_SECTORS}
    scanned = 0

    try:
        for row in ds:
            scanned += 1
            sectors = row.get("business_sector") or []
            if isinstance(sectors, str):
                sectors = [sectors]
            secs_lower = [s.lower() for s in sectors]
            for sec in TARGET_SECTORS:
                if sec in secs_lower:
                    seen[sec] += 1
                    rec = {
                        "id": row.get("id"),
                        "target_sector": sec,
                        "expected_category": TARGET_SECTORS[sec],
                        "business_sector": sectors,
                        "one_sentence_description": row.get("one_sentence_description", ""),
                        "text": row.get("text", ""),  # present in propella? kept if so
                    }
                    # Standard reservoir sampling.
                    if len(reservoir[sec]) < N_PER_SECTOR:
                        reservoir[sec].append(rec)
                    else:
                        j = random.randint(0, seen[sec] - 1)
                        if j < N_PER_SECTOR:
                            reservoir[sec][j] = rec
            if scanned % 500_000 == 0:
                got = {s: len(reservoir[s]) for s in TARGET_SECTORS}
                print(f"  scanned {scanned:,}  reservoir={got}")
    finally:
        del ds
        import gc
        gc.collect()

    all_samples = [r for recs in reservoir.values() for r in recs]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in all_samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nScanned {scanned:,} propella rows.")
    print("Sampled per sector (total docs seen with that sector):")
    for sec in TARGET_SECTORS:
        print(f"  {sec:24} sampled {len(reservoir[sec]):3}  of {seen[sec]:6} seen")
    has_text = sum(1 for r in all_samples if r.get("text"))
    has_desc = sum(1 for r in all_samples if r.get("one_sentence_description"))
    print(f"\nWrote {len(all_samples)} docs to {OUTPUT_PATH}")
    print(f"  with full text:       {has_text}")
    print(f"  with description:     {has_desc}")
    if not has_text and not has_desc:
        print("  WARNING: neither text nor description present -- Phase 2 will "
              "need a FineWeb-2 join. Check propella field names.")


if __name__ == "__main__":
    main()
