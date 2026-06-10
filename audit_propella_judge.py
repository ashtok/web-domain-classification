# audit_propella_judge.py
"""
PHASE 2 of the propella tag-quality audit (NEEDS the inference API).

Reads the propella-conditioned sample from Phase 1 and asks our teacher model
(gpt-oss-120b) to classify each doc. For each propella sector we audit, the
REJECT RATE = fraction of sampled docs where our model does NOT assign the
category that sector is supposed to mean. That reject rate is propella's
apparent FALSE-POSITIVE rate for that sector -- the thing the earlier
benchmark-vs-propella comparison structurally could not measure.

Judges from full `text` if present, else propella's `one_sentence_description`.

Reads : propella_audit_sample.jsonl
Writes: propella_audit_judged.jsonl   (per-doc verdict)
        propella_audit_results.csv     (per-sector reject rate)
Resumable: skips docs already judged.
"""
import json
import os
import time
from collections import defaultdict

from openai import OpenAI

from categories import (
    multiclass_system_prompt,
    parse_multiclass_label,
    message_text,
)

INPUT_PATH   = "propella_audit_sample.jsonl"
JUDGED_PATH  = "propella_audit_judged.jsonl"
RESULTS_CSV  = "propella_audit_results.csv"
MODEL        = "hosted_vllm/gpt-oss-120b"
BASE_URL     = "https://litellm.professor-x.de/v1"

client = OpenAI(api_key=os.environ["LSX_API_KEY"], base_url=BASE_URL)
SYSTEM = multiclass_system_prompt()


def judge(text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text[:4000]},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return parse_multiclass_label(message_text(resp.choices[0].message))


def doc_text(r: dict) -> str:
    """Prefer full text; fall back to propella's own description."""
    return (r.get("text") or "").strip() or (r.get("one_sentence_description") or "").strip()


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    existing = {}
    if os.path.exists(JUDGED_PATH):
        with open(JUDGED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r["id"], r["target_sector"]] = r

    out = []
    for i, r in enumerate(rows, 1):
        key = (r["id"], r["target_sector"])
        if key in existing and "our_label" in existing[key]:
            out.append(existing[key])
            continue
        text = doc_text(r)
        if not text:
            r["our_label"] = None        # nothing to judge
        else:
            try:
                r["our_label"] = judge(text)
            except Exception as e:
                print(f"doc {i} id={r['id']}: {e}")
                continue
        r["agrees"] = (r["our_label"] == r["expected_category"])
        out.append(r)
        if i % 25 == 0:
            print(f"Judged {i}/{len(rows)}")

    with open(JUDGED_PATH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-sector reject rate = propella's apparent false-positive rate.
    by_sector = defaultdict(list)
    for r in out:
        if r.get("our_label") is not None:
            by_sector[r["target_sector"]].append(r)

    print("\n=== propella tag-quality audit ===")
    print("reject_rate = fraction where OUR model disagrees with propella's "
          "sector (apparent false positives)\n")
    results = []
    for sec, recs in by_sector.items():
        n = len(recs)
        agree = sum(1 for r in recs if r["agrees"])
        reject = n - agree
        expected = recs[0]["expected_category"]
        # what our model said instead, for the rejects
        misc = defaultdict(int)
        for r in recs:
            if not r["agrees"]:
                misc[r["our_label"]] += 1
        print(f"{sec:24} -> {expected:14} n={n:3}  "
              f"reject={reject:3}/{n}  (we said instead: {dict(misc)})")
        results.append({
            "propella_sector": sec,
            "expected_category": expected,
            "n": n,
            "agree": agree,
            "reject": reject,
            "reject_rate": round(reject / n, 4) if n else None,
        })

    import pandas as pd
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
