# classify_multilabel.py
"""
Run the MULTI-LABEL strategy: each model predicts the SET of categories that
apply to each doc (zero or more, OTHER exclusive). Scored later against the
multi-label gold from build_multilabel_gold.py.

Reads  : benchmark_multilabel.jsonl   (must contain `gold_labels`)
Writes : benchmark_multilabel_pred.jsonl
         adds per model `m`:  {m}__multilabel_pred  (sorted list)
                              {m}__multilabel_latency_sec
Resumable: skips (model) already present on a row.
"""
import json
import os
import time

from openai import OpenAI

from categories import (
    multilabel_system_prompt,
    parse_multilabel,
    message_text,
)

INPUT_PATH  = "benchmark_multilabel.jsonl"
OUTPUT_PATH = "benchmark_multilabel_pred.jsonl"
BASE_URL    = "https://litellm.professor-x.de/v1"

MODELS = [
    "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
    "hosted_vllm/RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "hosted_vllm/google/gemma-4-E4B-it",
]

client = OpenAI(api_key=os.environ["LSX_API_KEY"], base_url=BASE_URL)
SYSTEM = multilabel_system_prompt()


def safe_name(model: str) -> str:
    return model.replace("hosted_vllm/", "").replace("/", "_").replace("-", "_")


def predict(model: str, text: str):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:4000]},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    labels = parse_multilabel(message_text(resp.choices[0].message))
    return sorted(labels), (time.time() - t0)


def preflight(models):
    ok = []
    for model in models:
        try:
            predict(model, "Test.")
            ok.append(model)
            print(f"  [ok]   {model}")
        except Exception as e:
            print(f"  [SKIP] {model}: {str(e)[:160]}")
    return ok


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows or "gold_labels" not in rows[0]:
        raise SystemExit(
            f"{INPUT_PATH} missing `gold_labels`. Run build_multilabel_gold.py first."
        )

    print("Pre-flight model check...")
    models = preflight(MODELS)
    if not models:
        raise SystemExit("No accessible models.")

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
        for model in models:
            m = safe_name(model)
            pred_key = f"{m}__multilabel_pred"
            lat_key  = f"{m}__multilabel_latency_sec"
            if pred_key in row or not text:
                continue
            try:
                pred, lat = predict(model, text)
                row[pred_key] = pred
                row[lat_key]  = lat
            except Exception as e:
                print(f"doc {i} id={row['id']} model={model}: {e}")
        out_rows.append(row)
        if i % 50 == 0:
            print(f"Processed {i}/{len(rows)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done. Wrote {len(out_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
