# check_latency.py
"""
Sanity-check the latency numbers behind the strategy comparison.

For each model, report multiclass-call latency stats (mean/median/p90/max) so we
can see whether gemma-4 being 'slower' than gemma-3 is a real central-tendency
effect or driven by a few slow (queued) outliers. Also report the binary
per-call mean (binary total / 3 categories) for an apples-to-apples model-speed
comparison independent of the 1-vs-3-calls strategy difference.
"""
import json
import statistics as st

INPUT_PATH = "benchmark_strategies.jsonl"

rows = [json.loads(l) for l in open(INPUT_PATH, encoding="utf-8") if l.strip()]

prefixes = sorted({
    k.split("__", 1)[0] for k in rows[0] if k.endswith("__multiclass_pred")
})


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    p90 = xs[int(0.9 * (len(xs) - 1))]
    return {
        "n": len(xs),
        "mean": st.mean(xs),
        "median": st.median(xs),
        "p90": p90,
        "max": max(xs),
    }


for m in prefixes:
    mc = stats([r.get(f"{m}__multiclass_latency_sec") for r in rows])
    # binary total = the 3 category calls summed (the true per-doc cost)
    bin_total = stats([r.get(f"{m}__binary_latency_sec") for r in rows])
    # ...and divided by 3 for a per-call model-speed comparison
    bin_percall = stats([
        r.get(f"{m}__binary_latency_sec") / 3
        for r in rows if r.get(f"{m}__binary_latency_sec") is not None
    ])
    print("=" * 70)
    print(m)
    print(f"  multiclass (1 call/doc):     "
          f"mean={mc['mean']:.3f}  median={mc['median']:.3f}  "
          f"p90={mc['p90']:.3f}  max={mc['max']:.3f}")
    print(f"  binary TOTAL (3 calls/doc):  "
          f"mean={bin_total['mean']:.3f}  median={bin_total['median']:.3f}  "
          f"p90={bin_total['p90']:.3f}  max={bin_total['max']:.3f}")
    print(f"  binary per-call:             "
          f"mean={bin_percall['mean']:.3f}  median={bin_percall['median']:.3f}  "
          f"p90={bin_percall['p90']:.3f}  max={bin_percall['max']:.3f}")
