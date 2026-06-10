#!/bin/bash
#SBATCH --job-name=propella_audit
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/%x_%j.out
#SBATCH --error=jobs/logs/%x_%j.err
#SBATCH --qos=normal

# ── propella tag-quality audit ────────────────────────────────────────────────
#
# Phase 1 (audit_propella_sample.py)  : stream propella, reservoir-sample docs
#                                        per sector. NO API needed.
# Phase 2 (audit_propella_judge.py)   : judge sampled docs with gpt-oss.
#                                        NEEDS the LSX inference API + LSX_API_KEY.
#
# No GPU requested: these scripts only stream data / call the inference API.
#
# Parameters (override via env vars):
#   PHASE   1 | 2 | all     (default: all)
#               1   = sampling only (safe to run while the API is down)
#               2   = judging only (assumes the sample file already exists)
#               all = sampling then judging
#
# Examples:
#   # Sampling only, now, even if the API is down:
#   PHASE=1 sbatch jobs/run_propella_audit.sh
#
#   # Full audit once the API is back:
#   sbatch jobs/run_propella_audit.sh
#
PHASE=${PHASE:-all}

PROJECT=/data/42-julia-hpc-rz-wuenlp/s472389/caidas/annotation_data/medical

echo "=========================================="
echo "  propella tag-quality audit"
echo "  Phase   : $PHASE"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Started : $(date)"
echo "=========================================="

cd "$PROJECT" || { echo "ERROR: cannot cd to $PROJECT"; exit 1; }
mkdir -p jobs/logs

# Load the API key from file if not already in the environment.
if [ -z "$LSX_API_KEY" ] && [ -f "$HOME/.lsx_api_key" ]; then
    export LSX_API_KEY="$(cat "$HOME/.lsx_api_key")"
fi

EXIT_CODE=0

# ── Phase 1: sampling (no API) ────────────────────────────────────────────────
if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
    echo "[INFO] Phase 1: sampling propella (no API needed)"
    uv run python audit_propella_sample.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Phase 1 failed (exit $EXIT_CODE); not proceeding to Phase 2."
        echo "Finished: $(date)"
        exit $EXIT_CODE
    fi
fi

# ── Phase 2: judging (needs API) ──────────────────────────────────────────────
if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
    if [ -z "$LSX_API_KEY" ]; then
        echo "[WARN] LSX_API_KEY not set -- skipping Phase 2 (judging needs the API)."
        echo "       Re-run with PHASE=2 once the key is available."
    else
        echo "[INFO] Phase 2: judging sampled docs with gpt-oss"
        uv run python audit_propella_judge.py
        EXIT_CODE=$?
    fi
fi

echo "=========================================="
echo "  Finished : $(date)   Exit: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
