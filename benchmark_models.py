# benchmark_models.py
import json
import os
import time
from openai import OpenAI

INPUT_PATH = "benchmark_1000.jsonl"
OUTPUT_PATH = "benchmark_1000_benchmarked.jsonl"
BASE_URL = "https://litellm.professor-x.de/v1"

MODELS = [
    "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
    "hosted_vllm/RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8",
    "hosted_vllm/google/gemma-4-E4B-it",
]

SYSTEM_PROMPT = (
    "You are a strict binary classifier for German web documents.\n"
    "Label MEDICAL if the PRIMARY topic is clinical medicine, drugs/pharmaceuticals, "
    "medical research, public health, or healthcare services.\n"
    "Label NON_MEDICAL if the primary topic is anything else (travel, tech, law, politics, "
    "sports, entertainment, automotive, finance, food, etc.), even if it mentions medical terms.\n"
    "Output ONLY one word: MEDICAL or NON_MEDICAL. No explanation, no reasoning."
)

client = OpenAI(
    api_key=os.environ["LSX_API_KEY"],
    base_url=BASE_URL,
)

def extract_label(msg):
    content = (msg.content or "").strip().upper()
    if "NON_MEDICAL" in content or "NON MEDICAL" in content:
        return "NON_MEDICAL"
    if "MEDICAL" in content:
        return "MEDICAL"

    # Check reasoning_content (gpt-oss, some Qwen models with thinking)
    reasoning = getattr(msg, "reasoning_content", "") or ""

    # Check provider_specific_fields (vLLM/LiteLLM wrapping)
    if not reasoning and isinstance(msg.provider_specific_fields, dict):
        reasoning = (
            msg.provider_specific_fields.get("reasoning_content")
            or msg.provider_specific_fields.get("reasoning")
            or msg.provider_specific_fields.get("thinking")
            or msg.provider_specific_fields.get("message")
            or ""
        )

    if reasoning:
        tail = reasoning[-500:].upper()
        if "NON_MEDICAL" in tail or "NON MEDICAL" in tail:
            return "NON_MEDICAL"
        if "MEDICAL" in tail:
            return "MEDICAL"

    return "NON_MEDICAL"

def safe_name(model):
    return model.replace("hosted_vllm/", "").replace("/", "_").replace("-", "_")

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])

    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue

            text = (row.get("text") or "").strip()
            if not text:
                continue

            for model in MODELS:
                s = safe_name(model)
                label_key = f"{s}_label"
                lat_key = f"{s}_latency_sec"

                if label_key in row and lat_key in row:
                    continue

                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text[:4000]},
                        ],
                        temperature=0.0,
                        max_tokens=50,
                    )
                    label = extract_label(resp.choices[0].message)
                except Exception as e:
                    print(f"Error on doc {i} id={row['id']} model={model}: {e}")
                    continue
                t1 = time.time()

                row[label_key] = label
                row[lat_key] = t1 - t0

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"Processed {i}/{len(rows)}")

if __name__ == "__main__":
    main()
