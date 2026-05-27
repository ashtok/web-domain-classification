from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "openeurollm/propella-annotations",
    name="fineweb-2",
    split="deu_Latn",
    streaming=True,
)

sector_counter = Counter()

for i, row in enumerate(dataset):
    if i >= 100000:
        break
    sectors = row.get("business_sector", [])
    if isinstance(sectors, list):
        sector_counter.update(sectors)
    elif isinstance(sectors, str):
        sector_counter[sectors] += 1

print(f"Processed {i+1} rows\n")
for sector, count in sector_counter.most_common():
    print(f"{sector}: {count}")
