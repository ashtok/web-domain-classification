# latency_probe.py
"""
Quick standalone latency probe: hits each model N times with the same short
classification prompt and reports timing stats. Independent of the benchmark
files — just answers "how fast is each model on the endpoint right now".

Usage:
    uv run python latency_probe.py
"""
import os
import statistics as st
import time

from openai import OpenAI

from categories import multiclass_system_prompt

BASE_URL = "https://litellm.professor-x.de/v1"
N        = 10   # calls per model

MODELS = [
    "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
    "hosted_vllm/RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "hosted_vllm/google/gemma-4-E4B-it",
]

# A single representative German doc snippet to classify.
SAMPLE_TEXT = (
    "Krankenversicherung der Rentner: In der Krankenversicherung der Rentner "
    "(KVdR) sind alle Personen pflichtversichert, die die Voraussetzungen fuer "
    "eine Rente der gesetzlichen Rentenversicherung erfuellen und einen Antrag "
    "gestellt haben."
)

client = OpenAI(api_key=os.environ["LSX_API_KEY"], base_url=BASE_URL)
SYSTEM = multiclass_system_prompt()


def time_call(model):
    t0 = time.time()
    client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": SAMPLE_TEXT},
        ],
        temperature=0.0,
        max_tokens=50,
    )
    return time.time() - t0


for model in MODELS:
    # one warm-up call (not counted) to avoid cold-start skew
    try:
        time_call(model)
    except Exception as e:
        print(f"{model}: warm-up failed: {e}")
        continue

    times = []
    for _ in range(N):
        try:
            times.append(time_call(model))
        except Exception as e:
            print(f"{model}: call failed: {e}")

    if times:
        print(f"{model}")
        print(f"  n={len(times)}  mean={st.mean(times):.3f}s  "
              f"median={st.median(times):.3f}s  "
              f"min={min(times):.3f}s  max={max(times):.3f}s")
