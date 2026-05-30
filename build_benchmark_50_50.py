# build_benchmark_50_50.py
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from datasets import load_dataset
from openai import OpenAI

OUTPUT_PATH        = "benchmark_1000.jsonl"
MODEL_NAME         = "hosted_vllm/gpt-oss-120b"
TARGET_MEDICAL     = 500
TARGET_NON_MEDICAL = 500
NUM_THREADS        = 20   # tune up/down depending on API rate limits

client = OpenAI(
    api_key=os.environ["LSX_API_KEY"],
    base_url="https://litellm.professor-x.de/v1",
)

SYSTEM_PROMPT = (
    "You are a strict binary classifier for German web documents.\n"
    "Label the document MEDICAL only if its PRIMARY topic is one of:\n"
    "- Clinical medicine (symptoms, diagnosis, treatment, diseases, surgery)\n"
    "- Drugs / pharmaceuticals (medication names, dosages, side effects)\n"
    "- Medical research or clinical studies\n"
    "- Public health, epidemiology, health policy\n"
    "- Healthcare services (hospitals, doctors, insurance coverage of treatments)\n\n"
    "Label it NON_MEDICAL if:\n"
    "- It merely MENTIONS a medical term in passing\n"
    "- The primary topic is travel, food, law, tech, politics, sports, entertainment, "
    "automotive, finance, or any other non-medical domain\n"
    "- It is a product listing, forum post, or navigation page unrelated to healthcare\n\n"
    "Ask yourself: Would a medical professional or researcher consider this document "
    "directly relevant to their clinical or scientific work? If no, it is NON_MEDICAL.\n\n"
    "Respond with exactly one word: MEDICAL or NON_MEDICAL. No explanation."
)

# Thread-safe counters and file lock
lock              = threading.Lock()
medical_count     = 0
non_medical_count = 0
file_handle       = None


def extract_label(msg) -> str:
    content = (msg.content or "").strip().upper()
    if "NON_MEDICAL" in content or "NON MEDICAL" in content:
        return "NON_MEDICAL"
    if "MEDICAL" in content:
        return "MEDICAL"
    return "NON_MEDICAL"


def classify_and_store(doc):
    global medical_count, non_medical_count

    text = (doc.get("text") or "").strip()
    if not text:
        return None

    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        elapsed = time.time() - start
        label = extract_label(resp.choices[0].message)
    except Exception as e:
        print(f"Error on id={doc.get('id')}: {e}")
        return None

    row = {
        "id":                  doc.get("id"),
        "text":                text,
        "url":                 doc.get("url", ""),
        "gpt_oss_label":       label,
        "gpt_oss_latency_sec": elapsed,
    }

    with lock:
        if label == "MEDICAL" and medical_count < TARGET_MEDICAL:
            medical_count += 1
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            file_handle.flush()
            return ("MEDICAL", medical_count, non_medical_count)
        elif label == "NON_MEDICAL" and non_medical_count < TARGET_NON_MEDICAL:
            non_medical_count += 1
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            file_handle.flush()
            return ("NON_MEDICAL", medical_count, non_medical_count)

    return None


def done():
    with lock:
        return medical_count >= TARGET_MEDICAL and non_medical_count >= TARGET_NON_MEDICAL


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
    BATCH_SIZE = NUM_THREADS * 4  # pre-fetch docs ahead of threads

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        file_handle = f
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            stream = iter(dataset)
            while not done():
                # Grab next batch of docs from stream
                batch = list(islice(stream, BATCH_SIZE))
                if not batch:
                    break

                total_seen += len(batch)
                futures = {executor.submit(classify_and_store, doc): doc for doc in batch}

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        label, med, non = result
                        if (med + non) % 20 == 0:
                            print(
                                f"[{total_seen} seen] "
                                f"MEDICAL: {med}/{TARGET_MEDICAL}  "
                                f"NON_MEDICAL: {non}/{TARGET_NON_MEDICAL}"
                            )
                    if done():
                        break

    print(f"\nDone. Scanned ~{total_seen} docs.")
    print(f"Output: {OUTPUT_PATH}")
    with lock:
        print(f"MEDICAL: {medical_count}  NON_MEDICAL: {non_medical_count}")


if __name__ == "__main__":
    main()
