# Topical Classification of FineWeb-2 (German) Web Documents

This project builds a pipeline to classify German web documents from the
**FineWeb-2** corpus (`HuggingFaceFW/fineweb-2`, `deu_Latn`) into topical
categories, using LLMs hosted on the LSX cluster (OpenAI-compatible LiteLLM
proxy).

The categories are **MEDICAL**, **CYBERSECURITY** and **CLIMATE** (plus a catch-all
**OTHER**). New categories are added by editing a single config file
([`categories.py`](categories.py)) — everything downstream reads from it.

The classification operates directly on the **FineWeb-2 document text** (truncated
to ~4000 chars). An earlier prototype used the short `one_sentence_description`
field from `openeurollm/propella-annotations`; that approach has been dropped in
favour of the full text.

---

## Pipeline overview

The pipeline is three stages, each re-streaming FineWeb-2 fresh as needed:

1. **`build_benchmark.py`** — Streams FineWeb-2 and labels documents with a strong
   *teacher* model (`gpt-oss-120b`) using the multi-class prompt. Collects a
   **balanced** set (target N per category + a quota of OTHER) so the benchmark
   isn't swamped by the natural ~99% non-topical distribution. The teacher's label
   is stored as `gold_label` and treated as ground truth. → `benchmark.jsonl`

2. **`classify_strategies.py`** — Re-classifies the benchmark docs with every
   candidate model, under **both** strategies:
   - **multiclass**: one call per doc → `{MEDICAL, CYBERSECURITY, CLIMATE, OTHER}`
   - **binary**: one call *per category* (POSITIVE/NEGATIVE), then collapsed into a
     single predicted category (first positive by priority order, else OTHER).

   Records predicted category + latency per (model, strategy). Resumable.
   → `benchmark_strategies.jsonl`

3. **`evaluate_strategies.py`** — Scores both strategies for all models against
   `gold_label`: accuracy, macro-F1, per-category F1, avg latency. Also emits a
   **delta table** (`binary − multiclass`) that directly answers "are the two
   strategies meaningfully different?".
   → `benchmark_strategies_results.csv`, `benchmark_strategies_delta.csv`

> **Note on latency:** the binary strategy makes N calls per document (one per
> category), so its per-doc latency is the **sum** of those calls — i.e. its true
> cost is roughly N× the multi-class cost. This is reflected in `avg_latency_sec`.

### Legacy (medical-only) scripts

The original medical-only experiment is still in the repo for reference:
`build_benchmark_50_50.py`, `benchmark_models.py`, `evaluate_benchmark.py`
(→ `benchmark_results.csv`). The new `*_strategies.py` scripts supersede these.

---

## Environment setup (`uv`)

```bash
uv sync
```

Installs `datasets`, `openai`, `scikit-learn`, `pandas` from `pyproject.toml` /
`uv.lock`.

## LSX API configuration

The LSX cluster exposes an OpenAI-compatible API (LiteLLM proxy). You need a course
API key and to be on the university network/VPN.

```bash
export LSX_API_KEY="$(cat ~/.lsx_api_key)"   # or set it directly
```

All scripts read `os.environ["LSX_API_KEY"]` and use
`base_url="https://litellm.professor-x.de/v1"`.

---

## Running the pipeline

```bash
export LSX_API_KEY="$(cat ~/.lsx_api_key)"

# 1. Build the balanced multi-category benchmark (teacher = gpt-oss-120b)
uv run python build_benchmark.py

# 2. Classify with all models under both strategies
uv run python classify_strategies.py

# 3. Compare multi-class vs binary
uv run python evaluate_strategies.py
```

### Configuration knobs

- **Categories & prompts** — [`categories.py`](categories.py): `CATEGORIES` dict.
- **Benchmark size / balance** — [`build_benchmark.py`](build_benchmark.py):
  `PER_CATEGORY`, `TARGET_OTHER`, `NUM_THREADS`, `TEACHER_MODEL`.
- **Candidate models** — [`classify_strategies.py`](classify_strategies.py):
  `MODELS`, and `PRIORITY` (binary conflict resolution order).

---

## Earlier medical-only benchmark results

From the original binary medical experiment (`benchmark_results.csv`),
labels scored against the gpt-oss-120b teacher over 1000 docs (500/500):

| Model                          | Accuracy | MEDICAL F1 | MEDICAL Recall | Avg latency |
| ------------------------------ | -------- | ---------- | -------------- | ----------- |
| gemma-3-27b-it-quantized.w4a16 | 97.8%    | 0.978      | 99.8%          | 0.236 s     |
| gemma-4-E4B-it                 | 97.6%    | 0.977      | 99.8%          | 0.304 s     |
| Mistral-Small-3.2-24B          | 97.5%    | 0.975      | 98.2%          | **0.160 s** |

Spot checks of MEDICAL examples (Alzheimer's, biomarkers, surgery, patient
symptoms, psychiatry, headaches, epilepsy, LASIK) were all correctly labeled.

---

## Limitations and notes

- **Ground truth is itself an LLM.** Metrics measure *agreement with the
  gpt-oss-120b teacher*, not absolute correctness. A small human-labeled gold set
  would be needed to validate the teacher, especially for the rarer categories.
- **Binary strategy cost.** N calls per doc makes it ~N× slower and N× the API
  load of multi-class; weigh that against any quality gain.
- **CLIMATE is the hardest category** to label cleanly (easy to confuse with
  general weather / environment content); inspect its predictions most carefully.
- The natural distribution is extremely imbalanced (~1% per topical category),
  which is why the benchmark is built with per-category quotas.

---

## Possible next steps

- Add a small human-labeled gold set to validate the teacher labels per category.
- Pick the winning (model, strategy) combination for the large-scale labeling run.
- Parallelize the large-scale run (the medical estimate was ~1k medical docs per
  ~85k scanned).
