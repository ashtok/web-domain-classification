# Medical vs Non‑Medical Classification on FineWeb‑2 (Propella Annotations)

This project builds a simple pipeline to classify web documents from the FineWeb‑2 corpus as **MEDICAL** or **NON_MEDICAL**, using:

- **propella-annotations** as metadata (business sector, one_sentence_description, etc.).[web:7]  
- **FineWeb‑2 (German)** as the underlying web corpus.[web:5]  
- An **LSX‑hosted LLM** (Mistral‑Small, OpenAI‑compatible API) for classification.

The goal is to explore how far we can get using only the propella annotations as input to the LLM, and to prepare a medical‑focused subset for further experiments.

---

## Data sources

- **FineWeb‑2** (`HuggingFaceFW/fineweb-2`): large multilingual web text dataset used for LLM pretraining.[web:5]  
- **propella-annotations** (`openeurollm/propella-annotations`): model‑generated annotations (18 properties) for several corpora, including FineWeb‑2.[web:7]  

In this project we primarily use:

- `openeurollm/propella-annotations` with `name="fineweb-2"` and `split="deu_Latn"` (German).  
- The `business_sector` and `one_sentence_description` fields to identify and describe potentially medical content.

---

## Repository structure

Files in this directory:

- `fetch_med_data.py`  
  Streams `openeurollm/propella-annotations` (FineWeb‑2, German) and **filters documents by `business_sector`** to build a candidate medical‑related subset (e.g. including `healthcare_medical`, `pharmaceutical_biotech`, etc.).

- `fetch_med_with_text.py`  
  Prototype script to **join annotations with full FineWeb‑2 text** by document `id`. This is experimental and can be slow because it streams FineWeb‑2 to find matching IDs.

- `llm_med_classify.py`  
  Main script that **calls the LSX OpenAI‑compatible API** (Mistral‑Small) and labels each document as `MEDICAL` or `NON_MEDICAL` using the propella annotation fields.

- `main.py`  
  Placeholder/entrypoint for future orchestration (currently unused).

- `medical_data.jsonl`  
  JSONL file containing propella annotation rows pre‑filtered to “medical‑related” sectors (German).

- `medical_data_llm_labeled.jsonl`  
  Output JSONL where each row from `medical_data.jsonl` is augmented with an LLM label in the field `medical_llm_label`.

- `medical_with_text.jsonl`  
  (Optional/experimental) Output from `fetch_med_with_text.py` that merges annotations with full FineWeb‑2 document text for a small set of IDs.

- `pyproject.toml`, `uv.lock`  
  Project configuration and dependency lockfile for `uv` (Python package & environment manager).

---

## Environment setup (using `uv`)

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python environment. From this directory:

```bash
# Create/update the virtualenv and install dependencies
uv sync
```

This reads `pyproject.toml` / `uv.lock` and installs the necessary packages (e.g. `datasets`, `openai`).

---

## LSX API configuration

The LSX cluster exposes an **OpenAI‑compatible** API (LiteLLM proxy). You need a course API key and to be on the university network/VPN.

1. Store your key once (optional but safer):

   ```bash
   echo "YOUR_COURSE_API_KEY" > ~/.lsx_api_key
   chmod 600 ~/.lsx_api_key
   ```

2. In each session, load it into the environment:

   ```bash
   export LSX_API_KEY="$(cat ~/.lsx_api_key)"
   ```

`llm_med_classify.py` reads this value via:

```python
client = OpenAI(
    api_key=os.environ["LSX_API_KEY"],
    base_url="https://litellm.professor-x.de/v1",
)
```

The current model used is:

```python
MODEL_NAME = "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"
```

You can switch to another LSX‑hosted chat model by changing this string.

---

## 1. Building the medical‑related subset

`fetch_med_data.py` streams the propella‑annotated German FineWeb‑2 and selects rows whose `business_sector` intersects a set of target sectors (e.g. `healthcare_medical`, `pharmaceutical_biotech`, `academic_research`, plus some broader sectors).

Example (simplified) logic:

```python
dataset = load_dataset(
    "openeurollm/propella-annotations",
    name="fineweb-2",
    split="deu_Latn",
    streaming=True,
)

TARGET_SECTORS = {
    "healthcare_medical",
    "pharmaceutical_biotech",
    "academic_research",
    "insurance_industry",
    "education_sector",
    "environmental_services",
    "government_public",
}
```

Run:

```bash
uv run python fetch_med_data.py
```

This will create a JSONL file (e.g. `medical_related_1M.jsonl`) containing rows that match at least one target sector. A curated, smaller subset of this is stored as `medical_data.jsonl` for LLM classification.

---

## 2. LLM-based classification (MEDICAL vs NON_MEDICAL)

`llm_med_classify.py` reads `medical_data.jsonl` and, for each row:

1. Builds a short input text for the LLM using propella annotations, e.g.:

   ```python
   text = (
       f"Description: {row.get('one_sentence_description', '')}\n"
       f"Sectors: {', '.join(row.get('business_sector', []))}\n"
       f"Content type: {', '.join(row.get('content_type', []))}\n"
       f"Content quality: {row.get('content_quality', '')}\n"
       f"Technical level: {', '.join(row.get('technical_content', []))}\n"
       f"Audience: {row.get('audience_level', '')}"
   )
   ```

2. Sends this text to the LSX‑hosted Mistral model with a system prompt that defines `MEDICAL` vs `NON_MEDICAL` and asks for exactly one label.

3. Adds the resulting label as `row["medical_llm_label"]`.

4. Writes the updated row to `medical_data_llm_labeled.jsonl`.

To run on a small sample (default `MAX_DOCS = 50`):

```bash
export LSX_API_KEY="$(cat ~/.lsx_api_key)"
uv run llm_med_classify.py
```

You will see progress output like:

```text
[1] label=MEDICAL | text='German-language description of the job duties...'
...
Reached MAX_DOCS = 50, stopping.
Done. Wrote 50 labeled docs to medical_data_llm_labeled.jsonl
```

To process the full dataset, set `MAX_DOCS = None` in `llm_med_classify.py` (and be mindful of runtime and API quota).

---

## 3. (Optional) Joining annotations with full FineWeb‑2 text

`fetch_med_with_text.py` is an experimental script that:

1. Collects a small number of medical `id`s from `openeurollm/propella-annotations` (e.g. 10).  
2. Streams `HuggingFaceFW/fineweb-2` (German) and, for each document, checks if its `id` matches one of the collected IDs.  
3. When a match is found, merges the full FineWeb‑2 document (including `text`) with the propella annotation fields and writes them to `medical_with_text.jsonl`.

Because FineWeb‑2 is very large, this streaming join can be slow and is mainly for small experiments, not for the full dataset.

Run:

```bash
uv run python fetch_med_with_text.py
```

and inspect the resulting `medical_with_text.jsonl`.

---

## Limitations and notes

- **No full text in annotations**: `openeurollm/propella-annotations` only contains metadata and short descriptions, not the full web page text. The LLM classifier currently operates on this metadata, not on full documents.[web:7]  
- **Joining with FineWeb‑2 is expensive**: Matching annotation IDs to full text requires streaming FineWeb‑2, which is large and slow for more than a small number of documents.[web:5]  
- **LLM labels are not perfect**: The MEDICAL vs NON_MEDICAL labels are produced by a general‑purpose LLM and can misclassify borderline cases (e.g. legal decisions about medical topics, corporate news involving healthcare companies). Manual inspection or additional heuristics may be needed for high‑precision subsets.

---

## Possible next steps

- Manually evaluate a sample (e.g. 100–200 docs) to estimate classification accuracy.  
- Adjust the prompt or sector filter to be more conservative for MEDICAL.  
- If needed, extend the pipeline to work with full FineWeb‑2 text for a smaller, carefully selected subset of documents.
