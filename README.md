# Web Domain Classification (FineWeb-2, German)

Classify German web documents from the **FineWeb-2** corpus into topical
categories — **MEDICAL**, **CYBERSECURITY**, **CLIMATE** (plus a catch-all
**OTHER**) — using LLMs served on the LSX cluster (OpenAI-compatible API).

The project's goal is to pick a classification **strategy** and **model** that are
good enough and cheap enough to label web documents at large scale, and to
validate that the labels are trustworthy.

## What's here

- A **benchmark**: a balanced set of 2,000 German documents, labeled by a strong
  "teacher" model (gpt-oss-120b) that serves as ground truth.
- Experiments comparing **how** to classify:
  - **multi-class** — one label per document (the chosen approach)
  - **binary** — one yes/no model per category
  - **multi-label** — allow several labels per document
- **Validation** of the labels against a second frontier model (Kimi-K2.6) and
  against the independent **propella** annotations.

Adding a new category (e.g. "finance") is a one-entry change in
[`categories.py`](categories.py) — everything else reads from there.

## Key findings so far

- **Multi-class is the best strategy** — higher accuracy than binary on every
  model, and ~3× cheaper (1 API call per doc instead of one per category).
- **Top models** (vs. the gpt-oss teacher, 2,000 docs):

  | Model | Accuracy | Latency |
  | --- | --- | --- |
  | gemma-3-27b | 97.0% | 0.14s |
  | Mistral-Small-24B | 96.2% | 0.12s |
  | gemma-4-E4B | 96.5% | 0.23s |

- **Labels are trustworthy**: Kimi-K2.6 agrees with the teacher labels 96.2%,
  and the independent propella sector tags agree strongly (Cohen's κ 0.84–0.95).
- **Multi-label ≈ multi-class here** — these German docs are almost entirely
  single-topic (only 1 of 2,000 had more than one category).

## Quick start

```bash
# 1. Install dependencies (uv-managed environment)
uv sync

# 2. Set your LSX API key
export LSX_API_KEY="$(cat ~/.lsx_api_key)"

# 3. Build the benchmark, classify, evaluate
uv run python build_benchmark.py
uv run python classify_strategies.py
uv run python evaluate_strategies.py
```

You need to be on the university network/VPN to reach the LSX API.

## Documentation

- **[USAGE.md](USAGE.md)** — full technical guide: every script, the data files,
  how the experiments fit together, and how to run them (including on the SLURM
  cluster).

## Requirements

Python ≥ 3.12, managed with [`uv`](https://docs.astral.sh/uv/). Dependencies:
`datasets`, `openai`, `scikit-learn`, `pandas`.
