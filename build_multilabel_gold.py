# build_multilabel_gold.py
"""
Generate MULTI-LABEL ground truth for the existing benchmark.

The current benchmark.jsonl has a single-label `gold_label` (one of MEDICAL /
CYBERSECURITY / CLIMATE / OTHER). To evaluate the multi-label strategy fairly we
need multi-label gold: each doc gets the SET of categories that substantially
apply (e.g. {MEDICAL, CYBERSECURITY}). We produce it with the strong teacher
(gpt-oss-120b) using the multi-label prompt.

Reads  : benchmark.jsonl          (the same 2000 docs)
Writes : benchmark_multilabel.jsonl  (adds `gold_labels`: sorted list of labels)
Resumable: skips docs that already have `gold_labels`.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

from openai import OpenAI

from categories import (
    multilabel_system_prompt,
    parse_multilabel,
    message_text,
)

INPUT_PATH    = "benchmark.jsonl"
OUTPUT_PATH   = "benchmark_multilabel.jsonl"
TEACHER_MODEL = "hosted_vllm/gpt-oss-120b"
NUM_THREADS   = 20

client = OpenAI(
    api_key=os.environ["LSX_API_KEY"],
    base_url="https://litellm.professor-x.de/v1",
)
SYSTEM = multilabel_system_prompt()

lock = threading.Lock()
done_count = 0


def label_doc(row):
    """Return row enriched with gold_labels, or None on error."""
    global done_count
    text = (row.get("text") or "").strip()
    if not text:
        row["gold_labels"] = ["OTHER"]
        return row
    try:
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        labels = parse_multilabel(message_text(resp.choices[0].message))
    except Exception as e:
        print(f"Error id={row.get('id')}: {e}")
        return None

    row["gold_labels"] = sorted(labels)
    with lock:
        done_count += 1
        if done_count % 50 == 0:
            print(f"Labeled {done_count} docs...")
    return row


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # Resume: keep already-labeled rows, only process the rest.
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "gold_labels" in r:
                        existing[r["id"]] = r

    todo = [r for r in rows if r["id"] not in existing]
    print(f"{len(existing)} already labeled, {len(todo)} to do.")

    results = list(existing.values())
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        it = iter(todo)
        while True:
            batch = list(islice(it, NUM_THREADS * 4))
            if not batch:
                break
            futures = [ex.submit(label_doc, r) for r in batch]
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)

    # Preserve original benchmark order.
    order = {r["id"]: i for i, r in enumerate(rows)}
    results.sort(key=lambda r: order.get(r["id"], 1 << 30))
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick distribution report.
    from collections import Counter
    multi = sum(1 for r in results if len(r.get("gold_labels", [])) > 1)
    per_cat = Counter(l for r in results for l in r.get("gold_labels", []))
    print(f"\nDone. {len(results)} docs -> {OUTPUT_PATH}")
    print(f"Docs with >1 label: {multi}")
    print("Label frequency:", dict(per_cat))


if __name__ == "__main__":
    main()
