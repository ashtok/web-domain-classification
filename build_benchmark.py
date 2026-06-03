# build_benchmark.py
"""
Build a balanced multi-category benchmark set from FineWeb-2 (deu_Latn), labeled
by a strong "teacher" model (gpt-oss-120b) using the MULTI-CLASS prompt.

The teacher's label is the GROUND TRUTH (`gold_label`) used to score both
classification strategies later. We collect a target number of docs per category
(MEDICAL, CYBERSECURITY, CLIMATE) plus a quota of OTHER, so the benchmark is not
swamped by the natural ~99% non-topical distribution.

Re-streams FineWeb-2 fresh each run; writes one JSONL (the benchmark) so that
both strategy runs later score the *identical* documents.
"""
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

from datasets import load_dataset
from openai import OpenAI

from categories import (
    CATEGORY_NAMES,
    OTHER,
    multiclass_system_prompt,
    parse_multiclass_label,
)

OUTPUT_PATH    = "benchmark.jsonl"
TEACHER_MODEL  = "hosted_vllm/gpt-oss-120b"
PER_CATEGORY   = 500          # target positives per real category
TARGET_OTHER   = 500          # quota of OTHER docs
NUM_THREADS    = 20

client = OpenAI(
    api_key=os.environ["LSX_API_KEY"],
    base_url="https://litellm.professor-x.de/v1",
)

SYSTEM_PROMPT = multiclass_system_prompt()

# Per-label target counts: each real category + OTHER.
TARGETS = {name: PER_CATEGORY for name in CATEGORY_NAMES}
TARGETS[OTHER] = TARGET_OTHER

lock        = threading.Lock()
counts      = {label: 0 for label in TARGETS}
file_handle = None


def classify_and_store(doc):
    text = (doc.get("text") or "").strip()
    if not text:
        return None

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        elapsed = time.time() - start
        label = parse_multiclass_label(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error on id={doc.get('id')}: {e}")
        return None

    row = {
        "id":                  doc.get("id"),
        "text":                text,
        "url":                 doc.get("url", ""),
        "gold_label":          label,            # teacher = ground truth
        "teacher_latency_sec": elapsed,
    }

    with lock:
        if counts[label] < TARGETS[label]:
            counts[label] += 1
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            file_handle.flush()
            return (label, dict(counts))
    return None


def done():
    with lock:
        return all(counts[l] >= TARGETS[l] for l in TARGETS)


def main():
    global file_handle

    print("Loading FineWeb-2 deu_Latn stream...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="deu_Latn",
        split="train",
        streaming=True,
    )

    total_seen = 0
    BATCH_SIZE = NUM_THREADS * 4

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        file_handle = f
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            stream = iter(dataset)
            while not done():
                batch = list(islice(stream, BATCH_SIZE))
                if not batch:
                    break
                total_seen += len(batch)
                futures = {executor.submit(classify_and_store, d): d for d in batch}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        _, snapshot = result
                        progress = "  ".join(
                            f"{l}: {snapshot[l]}/{TARGETS[l]}" for l in TARGETS
                        )
                        print(f"[{total_seen} seen] {progress}")
                    if done():
                        break

    print(f"\nDone. Scanned ~{total_seen} docs.")
    print(f"Output: {OUTPUT_PATH}")
    with lock:
        for l in TARGETS:
            print(f"  {l}: {counts[l]}")


if __name__ == "__main__":
    main()
