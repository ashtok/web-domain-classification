# fetch_med_with_text.py
# Collects 10 medical docs from propella-annotations (English)
# then finds their full text in FineWeb-2 by matching on id.

from datasets import load_dataset
import json

TARGET_SECTORS = {"healthcare_medical", "pharmaceutical_biotech"}
MAX_MEDICAL_IDS = 10
OUTPUT_PATH = "medical_with_text.jsonl"

# Step 1: collect 10 medical IDs from propella annotations (English)
print("Step 1: Scanning propella annotations for medical docs...")
annotations = load_dataset(
    "openeurollm/propella-annotations",
    name="fineweb-2",
    split="deu_Latn",
    streaming=True,
)

medical_annotations = {}  # id -> annotation row

for i, row in enumerate(annotations):
    sectors = row.get("business_sector", [])
    sector_set = set(sectors) if isinstance(sectors, list) else {sectors}
    if sector_set & TARGET_SECTORS:
        medical_annotations[row["id"]] = row
        print(f"  Found medical doc [{len(medical_annotations)}]: {row.get('one_sentence_description', '')[:80]}")
    if len(medical_annotations) >= MAX_MEDICAL_IDS:
        print(f"  Collected {MAX_MEDICAL_IDS} IDs after scanning {i:,} annotation rows.")
        break

print(f"\nStep 1 done. Medical IDs collected: {list(medical_annotations.keys())[:3]} ...")

# Step 2: stream FineWeb-2 (English) and match by id
print("\nStep 2: Streaming FineWeb-2 to find matching full texts...")
fw2 = load_dataset(
    "HuggingFaceFW/fineweb-2",
    name="deu_Latn",
    split="train",
    streaming=True,
)

found = 0
remaining = set(medical_annotations.keys())

with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for i, doc in enumerate(fw2):
        doc_id = doc.get("id")
        if doc_id in remaining:
            merged = dict(doc)
            merged.update(medical_annotations[doc_id])  # add propella fields
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            remaining.discard(doc_id)
            found += 1
            print(f"  Matched [{found}] id={doc_id}")
            print(f"    description: {medical_annotations[doc_id].get('one_sentence_description', '')}")
            print(f"    text preview: {(doc.get('text') or '')[:150]!r}")
            print()
            if not remaining:
                print("All 10 IDs matched!")
                break

        if i % 1_000_000 == 0 and i > 0:
            print(f"  Scanned {i:,} FineWeb-2 docs | Matched {found} | Remaining {len(remaining)}")

print(f"\nDone! Saved {found} merged docs to {OUTPUT_PATH}")
if remaining:
    print(f"Warning: {len(remaining)} IDs not found in FineWeb-2 stream (may be in a different shard).")
