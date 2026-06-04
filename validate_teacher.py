# validate_teacher.py
"""
Validate the gpt-oss-120b ground-truth labels against a stronger / frontier model.

The benchmark's `gold_label` was produced by gpt-oss-120b. To check it isn't
systematically wrong, we re-label the SAME docs with a second strong model
(default: kimi-k2.6) using the identical multi-class prompt, then measure how
often the two agree. High agreement -> the gold labels are trustworthy.
Disagreements are written out so they can be eyeballed.

Reads  : benchmark.jsonl
Writes : benchmark_teacher_validation.jsonl   (per-doc validator labels)
         teacher_disagreements.jsonl           (only docs where they differ)
Resumable: skips docs already labeled by a given validator model.
"""
import json
import os
import time

from openai import OpenAI

from categories import multiclass_system_prompt, parse_multiclass_label, MULTICLASS_LABELS

INPUT_PATH       = "benchmark.jsonl"
OUTPUT_PATH      = "benchmark_teacher_validation.jsonl"
DISAGREE_PATH    = "teacher_disagreements.jsonl"
BASE_URL         = "https://litellm.professor-x.de/v1"

# Frontier model(s) to validate the gpt-oss teacher against.
VALIDATOR_MODELS = [
    "hosted_vllm/Kimi-K2.6",
]

client = OpenAI(api_key=os.environ["LSX_API_KEY"], base_url=BASE_URL)
SYSTEM = multiclass_system_prompt()


def safe_name(model: str) -> str:
    return model.replace("hosted_vllm/", "").replace("/", "_").replace("-", "_")


def label(model: str, text: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:4000]},
        ],
        temperature=0.0,
        max_tokens=50,
    )
    return parse_multiclass_label(resp.choices[0].message.content)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    # Resume: reload prior validation output keyed by id.
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r["id"]] = r

    out_rows = []
    for i, row in enumerate(rows, 1):
        row = existing.get(row["id"], row)
        text = (row.get("text") or "").strip()
        for model in VALIDATOR_MODELS:
            key = f"{safe_name(model)}_label"
            if key in row or not text:
                continue
            try:
                row[key] = label(model, text)
            except Exception as e:
                print(f"doc {i} id={row['id']} model={model}: {e}")
        out_rows.append(row)
        if i % 50 == 0:
            print(f"Processed {i}/{len(rows)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Agreement report + disagreement dump ---
    disagreements = []
    print("\n=== Agreement with gpt-oss gold_label ===")
    for model in VALIDATOR_MODELS:
        key = f"{safe_name(model)}_label"
        labeled = [r for r in out_rows if key in r]
        if not labeled:
            print(f"{model}: no labels produced.")
            continue
        agree = sum(1 for r in labeled if r[key] == r["gold_label"])
        n = len(labeled)
        print(f"\n{model}: {agree}/{n} = {agree/n:.1%} agreement")

        # per-gold-category agreement
        for cat in MULTICLASS_LABELS:
            sub = [r for r in labeled if r["gold_label"] == cat]
            if sub:
                a = sum(1 for r in sub if r[key] == cat)
                print(f"  gold={cat:14} {a}/{len(sub)} = {a/len(sub):.1%}")

        for r in labeled:
            if r[key] != r["gold_label"]:
                disagreements.append({
                    "id": r["id"],
                    "url": r.get("url", ""),
                    "gold_label": r["gold_label"],
                    "validator": model,
                    "validator_label": r[key],
                    "text": (r.get("text") or "")[:600],
                })

    with open(DISAGREE_PATH, "w", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(disagreements)} disagreements to {DISAGREE_PATH}")


if __name__ == "__main__":
    main()
