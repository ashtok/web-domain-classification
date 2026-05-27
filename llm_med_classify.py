# llm_med_classify.py
import json
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["LSX_API_KEY"],
    base_url="https://litellm.professor-x.de/v1",
)

MODEL_NAME = "hosted_vllm/RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"
INPUT_PATH = "medical_data.jsonl"
OUTPUT_PATH = "medical_data_llm_labeled.jsonl"

MAX_DOCS = 50  # increase to None to process everything

SYSTEM_PROMPT = (
    "You are a text classifier. "
    "Classify the given document metadata as MEDICAL or NON_MEDICAL.\n"
    "MEDICAL means the document content itself is clinical or health-related: "
    "patient advice, drug information, disease descriptions, medical procedures, "
    "clinical guidelines, biomedical research, or public health advisories.\n"
    "NON_MEDICAL means documents that merely mention hospitals, pharma companies, "
    "or medical keywords in a non-clinical context "
    "(e.g. hospital architecture, pharmaceutical packaging, legal rulings, corporate news).\n"
    "Respond with exactly one word: MEDICAL or NON_MEDICAL. No explanation."
)


def extract_label_from_msg(msg) -> str:
    content = (msg.content or "").strip().upper()
    if content:
        if "NON_MEDICAL" in content or "NON MEDICAL" in content:
            return "NON_MEDICAL"
        if "MEDICAL" in content:
            return "MEDICAL"

    reasoning = getattr(msg, "reasoning_content", "") or ""
    if not reasoning and isinstance(msg.provider_specific_fields, dict):
        reasoning = (
            msg.provider_specific_fields.get("reasoning_content")
            or msg.provider_specific_fields.get("reasoning")
            or ""
        )
    if reasoning:
        tail = reasoning[-300:].upper()
        if "NON_MEDICAL" in tail or "NON MEDICAL" in tail:
            return "NON_MEDICAL"
        if "MEDICAL" in tail:
            return "MEDICAL"

    return "NON_MEDICAL"


def classify_medical(text: str) -> str:
    if not text or not text.strip():
        return "NON_MEDICAL"

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    return extract_label_from_msg(resp.choices[0].message)


def main():
    processed = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            row = json.loads(line)

            # Combine all annotation fields for richer context
            text = (
                f"Description: {row.get('one_sentence_description', '')}\n"
                f"Sectors: {', '.join(row.get('business_sector', []))}\n"
                f"Content type: {', '.join(row.get('content_type', []))}\n"
                f"Content quality: {row.get('content_quality', '')}\n"
                f"Technical level: {', '.join(row.get('technical_content', []))}\n"
                f"Audience: {row.get('audience_level', '')}"
            )

            try:
                label = classify_medical(text)
            except Exception as e:
                print(f"Error for doc {processed + 1}: {e}")
                label = "ERROR"

            row["medical_llm_label"] = label
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            processed += 1
            desc = row.get("one_sentence_description", "")[:80]
            print(f"[{processed}] label={label} | text={desc!r}")

            if MAX_DOCS is not None and processed >= MAX_DOCS:
                print(f"Reached MAX_DOCS = {MAX_DOCS}, stopping.")
                break

    print(f"Done. Wrote {processed} labeled docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
