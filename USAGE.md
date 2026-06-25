# Technical Usage Guide

In-depth guide to the web-domain-classification pipeline: the scripts, the data
files they produce, how the experiments fit together, the configuration knobs,
and how to run everything (locally and on the SLURM cluster).

For a high-level overview see [README.md](README.md).

---

## 1. Concepts

### Categories
All categories, their definitions, and the prompts/parsers live in one place:
[`categories.py`](categories.py). The categories are **MEDICAL**,
**CYBERSECURITY**, **CLIMATE**, and the catch-all **OTHER**. Adding a category is
a single entry in the `CATEGORIES` dict — every other script imports from here,
so prompts, parsing, and scoring update automatically.

`categories.py` exposes three label "shapes":

| Strategy | Prompt builder | Parser | Output |
| --- | --- | --- | --- |
| multi-class | `multiclass_system_prompt()` | `parse_multiclass_label()` | exactly one of MEDICAL/CYBER/CLIMATE/OTHER |
| binary | `binary_system_prompt(cat)` | `parse_binary_label()` | `<CAT>` or `NON_<CAT>` |
| multi-label | `multilabel_system_prompt()` | `parse_multilabel()` | a set of labels (OTHER exclusive) |

`message_text(msg)` is a shared helper that extracts the answer from a chat
message — it prefers `content`, and falls back to the **tail** of
`reasoning_content` for reasoning models (e.g. Qwen) that stream their thinking
instead of returning a one-word answer. All parsers scan from the tail so they
pick the model's *conclusion*, not its first guess.

### Ground truth ("teacher")
We don't have human labels. Instead a strong model — **gpt-oss-120b** — labels
the benchmark, and those labels are treated as ground truth (`gold_label`). All
candidate models are scored by **agreement with the teacher**. This is fast and
cheap but circular by construction; see [§7 Caveats](#7-caveats-read-this).

### The LSX API
All inference goes through the LSX cluster's OpenAI-compatible LiteLLM proxy at
`https://litellm.professor-x.de/v1`. Every script reads the key from
`os.environ["LSX_API_KEY"]`. You must be on the university network/VPN.

```bash
export LSX_API_KEY="$(cat ~/.lsx_api_key)"
```

Model IDs must exactly match the team allowlist on the endpoint (including org
prefixes like `RedHatAI/`, `moonshotai/`). A wrong name returns a 401 that looks
like an auth error — the message lists the allowed models. The classify/validate
scripts run a **pre-flight** ping that skips inaccessible models before the main
loop, so a typo no longer wastes thousands of calls.

---

## 2. The strategy comparison pipeline (main experiment)

Three stages. Each re-streams FineWeb-2 fresh as needed; the benchmark file is
written once so all later stages score the **identical** documents.

### Stage 1 — build the benchmark
[`build_benchmark.py`](build_benchmark.py)

Streams `HuggingFaceFW/fineweb-2` (deu_Latn), labels each doc with the teacher
(multi-class prompt), and collects a **balanced** set — a per-category quota plus
an OTHER quota — so the benchmark isn't swamped by the natural ~99% non-topical
distribution.

- Output: `benchmark.jsonl` — one row per doc: `id`, `text`, `url`, `gold_label`,
  `teacher_latency_sec`.
- Knobs: `PER_CATEGORY`, `TARGET_OTHER`, `NUM_THREADS`, `TEACHER_MODEL`.
- Threaded (default 20 workers); writes incrementally and exits when all quotas
  are met.

```bash
uv run python build_benchmark.py
```

### Stage 2 — classify with both strategies
[`classify_strategies.py`](classify_strategies.py)

Runs every candidate model under **multi-class** and (optionally) **binary** on
the same benchmark docs.

- multi-class: one call per doc.
- binary: one call *per category* (POSITIVE/NEGATIVE), collapsed to a single
  predicted category by `PRIORITY` order (first positive wins, else OTHER).
- `MULTICLASS_ONLY = True` skips the 3×-cost binary pass for new models (binary
  was already settled).
- Output: `benchmark_strategies.jsonl`, adding per model `m`:
  `{m}__multiclass_pred`, `{m}__multiclass_latency_sec`, and (if binary runs)
  `{m}__binary_pred`, `{m}__binary_latency_sec`, `{m}__binary_raw`.
- **Resumable**: skips (model, strategy) pairs already present on a row. Re-runs
  only the new work.
- Knobs: `MODELS`, `MULTICLASS_ONLY`, `PRIORITY`.

```bash
uv run python classify_strategies.py
```

> **Latency note:** binary records the **sum** of its per-category calls, so its
> per-doc latency is ~3× multi-class — that's its real cost.

### Stage 3 — evaluate
[`evaluate_strategies.py`](evaluate_strategies.py)

Scores both strategies for all models against `gold_label`: accuracy, macro-F1,
per-category F1, average latency. Also emits a **delta table** (binary −
multi-class) that directly answers "is one strategy meaningfully better?".

- Outputs: `benchmark_strategies_results.csv`, `benchmark_strategies_delta.csv`.

```bash
uv run python evaluate_strategies.py
```

### Supporting tools
- [`check_latency.py`](check_latency.py) — recomputes latency stats (mean,
  **median**, p90, max) from the recorded times in `benchmark_strategies.jsonl`.
  No API calls. Use **median** for reporting — a single slow outlier (a 72 s
  call) inflates the mean.
- [`latency_probe.py`](latency_probe.py) — fires N timed calls per model on a
  fixed prompt for a clean model-speed comparison under current load.
- [`spot_check_strategies.py`](spot_check_strategies.py) — prints example docs by
  category, or where strategies/models disagree, for manual inspection. Modes:
  `by_category`, `disagree`, `vs_gold`.

---

## 3. Multi-label experiment

Tests whether allowing **several labels per document** changes results (and how
it compares to binary). Has its own gold because the single-label `gold_label`
can't score multi-label predictions.

| Step | Script | Output |
| --- | --- | --- |
| gold | [`build_multilabel_gold.py`](build_multilabel_gold.py) | `benchmark_multilabel.jsonl` (`gold_labels`: list) |
| classify | [`classify_multilabel.py`](classify_multilabel.py) | `benchmark_multilabel_pred.jsonl` |
| evaluate | [`evaluate_multilabel.py`](evaluate_multilabel.py) | `benchmark_multilabel_results.csv` |

```bash
uv run python build_multilabel_gold.py
uv run python classify_multilabel.py
uv run python evaluate_multilabel.py
```

Multi-label metrics differ from single-label: **exact-match** (predicted set ==
gold set), **Hamming loss**, **micro/macro-F1**, per-category F1. The evaluator
also prints how many gold docs are genuinely multi-label — the headline number,
since with near-zero overlap multi-label collapses to multi-class.

**OTHER is exclusive**: `parse_multilabel` drops OTHER if any real category is
present, and returns `{OTHER}` only when nothing applies.

---

## 4. Label validation

Two independent checks that the teacher's labels are trustworthy.

### 4a. Frontier cross-check — [`validate_teacher.py`](validate_teacher.py)
Re-labels all 2,000 benchmark docs with a second strong model (default
Kimi-K2.6) using the same multi-class prompt, then reports agreement with
`gold_label` overall and per category. Disagreements are dumped to
`teacher_disagreements.jsonl` for inspection.

- Outputs: `benchmark_teacher_validation.jsonl`, `teacher_disagreements.jsonl`.
- `VALIDATOR_MODELS` can hold more than one frontier model.

```bash
uv run python validate_teacher.py
```

### 4b. propella comparison — [`compare_propella.py`](compare_propella.py)
Cross-checks our labels against the **independent** `openeurollm/propella-annotations`
(same FineWeb-2 corpus, annotated with `business_sector`). Joins the 2,000
benchmark docs by `id`, maps sectors → our categories (exact-name map at the top
of the file), and reports **Cohen's κ** + agreement per category.

- Default run streams propella, caches the raw join to `propella_joined.jsonl`,
  and writes `propella_comparison.csv`.
- `--remap` re-scores from the cache **without re-streaming** — edit `SECTOR_MAP`
  and recompute κ instantly.

```bash
uv run python compare_propella.py            # stream + score
uv run python compare_propella.py --remap    # re-score from cache
```

> **Sector mapping is deliberately tight** (one sector → one category, exact
> names). Loose substring rules over-counted (e.g. `technology_software` leaking
> into CYBERSECURITY); the exact map is defensible.

---

## 5. propella tag-quality audit

The comparison in §4b measures *agreement on docs we sampled* — it cannot measure
how many docs **propella over-tagged**, because those only appear if our own
sampling happened to include them. To estimate propella's false-positive rate we
sample **conditioned on propella's tags** instead.

### Phase 1 — sample (no API) — [`audit_propella_sample.py`](audit_propella_sample.py)
Streams propella and collects the **first N docs per target sector**, then stops
(propella has tens of millions of rows — a full pass is wasteful). Saves id,
sectors, and propella's `one_sentence_description`.

- Output: `propella_audit_sample.jsonl`.
- Knobs: `N_PER_SECTOR`, `TARGET_SECTORS`.
- **Run on the login node** (compute nodes have no internet — see §6).

### Phase 2 — judge (needs API) — [`audit_propella_judge.py`](audit_propella_judge.py)
gpt-oss classifies each sampled doc; per-sector **reject rate** = fraction where
our label ≠ propella's expected category = propella's apparent false-positive
rate.

- Outputs: `propella_audit_judged.jsonl`, `propella_audit_results.csv`.
- Resumable.

```bash
uv run python audit_propella_sample.py    # Phase 1
uv run python audit_propella_judge.py     # Phase 2
```

> propella carries **no full document text**, only a one-sentence description, so
> Phase 2 judges from that. A thin description can lose the topic signal, so the
> reject rates are an **upper bound** on propella's error — see [§7](#7-caveats-read-this).

---

## 6. Running on the SLURM cluster

For long/detached runs use a batch job so it survives an SSH disconnect:
[`jobs/run_propella_audit.sh`](jobs/run_propella_audit.sh).

```bash
PHASE=1 sbatch jobs/run_propella_audit.sh     # sampling only
sbatch jobs/run_propella_audit.sh             # full audit (PHASE=all)
```

Common commands:

```bash
squeue -u $USER                               # your jobs (PD pending, R running)
tail -f jobs/logs/propella_audit_<JOBID>.out  # live log
scancel <JOBID>                               # cancel
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed   # post-mortem
```

> **Compute nodes have no internet.** Any step that **streams from HuggingFace**
> (`build_benchmark.py`, `build_multilabel_gold.py`, `compare_propella.py`,
> `audit_propella_sample.py`) will **hang** on a compute node. Run those on the
> **login node** (in `tmux`). Steps that only hit the **LSX API** can run as jobs
> if the API host is reachable from compute nodes; otherwise also use `tmux` on
> the login node.

```bash
tmux new -s work        # detach: Ctrl-b then d ; reattach: tmux attach -t work
```

---

## 7. Caveats (read this)

These shape how the results should be reported.

1. **Ground truth is an LLM.** All benchmark accuracy/F1 numbers measure
   *agreement with gpt-oss-120b*, not absolute correctness. Kimi (96.2%) and
   propella (κ 0.84–0.95) corroborate the labels, but a human-labeled set would
   be the only true validation.

2. **The multi-class vs binary comparison slightly favors multi-class** — the
   gold labels were generated with the multi-class prompt. The measured gap
   (3.5–6.4% accuracy) and the 3× cost difference are larger than that bias can
   explain, but state it when reporting.

3. **The propella comparison can't measure propella's false positives.** The
   benchmark was sampled by our own labels, so propella docs we never sampled are
   invisible. That's why the §5 audit samples by propella's tags instead.

4. **The audit's reject rates are an upper bound.** propella has no full text, so
   judging uses its one-sentence description; thin descriptions can cause our
   false-OTHERs. The clean version judges against full FineWeb-2 text (a one-pass
   join over the ~4,000 sampled ids) — not yet built.

5. **Reasoning models need room.** `max_tokens` is 2048 (not 50) so models like
   Qwen3.x can finish thinking and still emit a label; `message_text` reads the
   reasoning tail. Their latency is therefore higher and not directly comparable
   to non-reasoning models.

6. **Latency = wall-clock per call on the shared endpoint**, so it reflects
   server load, not just model compute. Report **medians** (outliers inflate the
   mean). It is not comparable to raw tok/s figures.

---

## 8. Data files

`.jsonl` files are git-ignored (regenerable); `.csv` result summaries are
committed.

| File | Produced by | Contents |
| --- | --- | --- |
| `benchmark.jsonl` | build_benchmark | 2,000 balanced docs + teacher `gold_label` |
| `benchmark_strategies.jsonl` | classify_strategies | per-model multi-class/binary preds |
| `benchmark_strategies_results.csv` | evaluate_strategies | per model × strategy metrics |
| `benchmark_strategies_delta.csv` | evaluate_strategies | binary − multi-class deltas |
| `benchmark_multilabel.jsonl` | build_multilabel_gold | docs + `gold_labels` |
| `benchmark_multilabel_pred.jsonl` | classify_multilabel | per-model label-set preds |
| `benchmark_multilabel_results.csv` | evaluate_multilabel | multi-label metrics |
| `benchmark_teacher_validation.jsonl` | validate_teacher | Kimi labels per doc |
| `teacher_disagreements.jsonl` | validate_teacher | docs where Kimi ≠ teacher |
| `propella_joined.jsonl` | compare_propella | benchmark id ↔ propella sectors (cache) |
| `propella_comparison.csv` | compare_propella | per-category κ + agreement |
| `propella_audit_sample.jsonl` | audit_propella_sample | propella docs sampled by sector |
| `propella_audit_judged.jsonl` | audit_propella_judge | per-doc verdicts |
| `propella_audit_results.csv` | audit_propella_judge | per-sector reject rate |

---

## 9. Legacy scripts

From the original medical-only experiment, kept for reference and superseded by
the `*_strategies.py` pipeline:
`build_benchmark_50_50.py`, `benchmark_models.py`, `evaluate_benchmark.py`
(→ `benchmark_results.csv`).
