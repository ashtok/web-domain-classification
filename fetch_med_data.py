from datasets import load_dataset
import json

dataset = load_dataset(
    "openeurollm/propella-annotations",
    name="fineweb-2",
    split="deu_Latn",
    streaming=True,
)

# All target sectors
TARGET_SECTORS = {
    # High priority
    "healthcare_medical",
    "pharmaceutical_biotech",
    "academic_research",
    # Medium priority
    "insurance_industry",
    "education_sector",
    "environmental_services",
    "government_public",
}

MAX_ROWS = 1_000_000
found = 0
sector_counts = {s: 0 for s in TARGET_SECTORS}

with open("medical_related_1M.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        sectors = row.get("business_sector", [])
        sector_set = set(sectors) if isinstance(sectors, list) else {sectors}

        matched = sector_set & TARGET_SECTORS
        if matched:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            found += 1
            for s in matched:
                sector_counts[s] += 1

            if found >= MAX_ROWS:
                print(f"Reached {MAX_ROWS:,} rows after scanning {i:,} total rows.")
                break

        if i % 1_000_000 == 0 and i > 0:
            print(f"Scanned {i:,} rows | Collected {found:,} | Breakdown: {sector_counts}")

print(f"\nDone! Saved {found:,} rows to medical_related_1M.jsonl")
print("\nFinal sector breakdown:")
for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
    print(f"  {sector}: {count:,}")
