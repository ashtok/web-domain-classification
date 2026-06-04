# classify_strategies.py
"""
Run BOTH classification strategies over the benchmark, for every candidate model:

  1. multiclass : one call per doc -> {MEDICAL, CYBERSECURITY, CLIMATE, OTHER}
  2. binary     : one call per category per doc (POSITIVE/NEGATIVE), then collapsed
                  into a single predicted category for an apples-to-apples comparison.

Both strategies score the SAME benchmark.jsonl docs, so any metric difference is
attributable to the strategy, not the data. Latency is recorded per strategy:
  - multiclass latency = the single call
  - binary latency     = SUM of the per-category calls (the true cost of that strategy)

Output JSONL augments each row with, per model `m` and strategy `s`:
  {m}__{s}_pred           predicted category (MEDICAL/CYBERSECURITY/CLIMATE/OTHER)
  {m}__{s}_latency_sec    latency for that strategy on that doc
Binary also stores the raw per-category decisions under {m}__binary_raw.

Resumable: skips (model, strategy) pairs already present on a row.
"""
import json
import os
import time

from openai import OpenAI

from categories import (
    CATEGORY_NAMES,
    OTHER,
    multiclass_system_prompt,
    binary_system_prompt,
    parse_multiclass_label,
    parse_binary_label,
    message_text,
)

INPUT_PATH  = "benchmark.jsonl"
OUTPUT_PATH = "benchmark_strategies.jsonl"
BASE_URL    = "https://litellm.professor-x.de/v1"

MODELS = [
    "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
    "hosted_vllm/RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "hosted_vllm/google/gemma-4-E4B-it",
    # Additional small / fast models requested for the speed-vs-quality tradeoff.
    # Names must exactly match the team allowlist on the LiteLLM endpoint.
    "hosted_vllm/ibm-granite/granite-4.1-3b",
    "hosted_vllm/Qwen/Qwen3.6-35B-A3B-FP8",
    "hosted_vllm/Microsoft/Phi-4-mini-instruct",
    "hosted_vllm/Qwen/Qwen3.5-9B",
]

# Multi-class already won decisively, so by default new models are run with
# multi-class only (skips the 3x-cost binary pass). Set to False to also run
# binary for every model.
MULTICLASS_ONLY = True

# Conflict resolution for the binary strategy: if several binaries fire,
# the earliest category in this order wins. All-negative -> OTHER.
PRIORITY = CATEGORY_NAMES  # MEDICAL > CYBERSECURITY > CLIMATE

client = OpenAI(api_key=os.environ["LSX_API_KEY"], base_url=BASE_URL)

MULTICLASS_PROMPT = multiclass_system_prompt()
BINARY_PROMPTS    = {c: binary_system_prompt(c) for c in CATEGORY_NAMES}


def safe_name(model: str) -> str:
    return model.replace("hosted_vllm/", "").replace("/", "_").replace("-", "_")


def call(model: str, system: str, text: str) -> tuple[str, float]:
    """One chat completion. Returns (raw_content, latency_sec)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text[:4000]},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return message_text(resp.choices[0].message), (time.time() - t0)


def run_multiclass(model: str, text: str) -> tuple[str, float]:
    raw, lat = call(model, MULTICLASS_PROMPT, text)
    return parse_multiclass_label(raw), lat


def run_binary(model: str, text: str) -> tuple[str, float, dict]:
    """Run one binary call per category, collapse to a single predicted category."""
    raw_decisions = {}
    total_lat = 0.0
    for cat in CATEGORY_NAMES:
        raw, lat = call(model, BINARY_PROMPTS[cat], text)
        raw_decisions[cat] = parse_binary_label(raw, cat)
        total_lat += lat
    # Collapse: first positive in priority order, else OTHER.
    pred = OTHER
    for cat in PRIORITY:
        if raw_decisions[cat] == cat:
            pred = cat
            break
    return pred, total_lat, raw_decisions


def preflight(models):
    """Ping each model once; drop any that are inaccessible so a bad name
    doesn't burn 2000 calls per model before failing."""
    ok = []
    for model in models:
        try:
            run_multiclass(model, "Test.")
            ok.append(model)
            print(f"  [ok]   {model}")
        except Exception as e:
            print(f"  [SKIP] {model}: {str(e)[:160]}")
    return ok


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    print("Pre-flight model check...")
    models = preflight(MODELS)
    if not models:
        raise SystemExit("No accessible models. Check names / allowlist.")

    # Resume support: load already-written rows keyed by id.
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
        if not text:
            out_rows.append(row)
            continue

        for model in models:
            m = safe_name(model)

            mc_pred_key = f"{m}__multiclass_pred"
            mc_lat_key  = f"{m}__multiclass_latency_sec"
            if mc_pred_key not in row:
                try:
                    pred, lat = run_multiclass(model, text)
                    row[mc_pred_key] = pred
                    row[mc_lat_key]  = lat
                except Exception as e:
                    print(f"[multiclass] doc {i} id={row['id']} model={model}: {e}")

            bin_pred_key = f"{m}__binary_pred"
            bin_lat_key  = f"{m}__binary_latency_sec"
            bin_raw_key  = f"{m}__binary_raw"
            if not MULTICLASS_ONLY and bin_pred_key not in row:
                try:
                    pred, lat, raw = run_binary(model, text)
                    row[bin_pred_key] = pred
                    row[bin_lat_key]  = lat
                    row[bin_raw_key]  = raw
                except Exception as e:
                    print(f"[binary] doc {i} id={row['id']} model={model}: {e}")

        out_rows.append(row)
        if i % 25 == 0:
            print(f"Processed {i}/{len(rows)}")

    # Rewrite the full output (rows may have been enriched in place).
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for r in out_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(out_rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
