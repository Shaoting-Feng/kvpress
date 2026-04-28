#!/usr/bin/env bash
# End-to-end: generate answers -> F1 score -> simulate -> plot.
# Run from this directory (kvpress repo /experiments/aa_lcr/).
#
# Set ATTN_IMPL=kernels-community/vllm-flash-attn3 on Hopper machines for
# FA3 single-shot prefill. Default is "eager" + chunked prefill (Blackwell-safe
# fallback, much slower).

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$HERE"

# Use whatever python is on PATH unless overridden.
PYTHON="${PYTHON:-python}"
ATTN_IMPL="${ATTN_IMPL:-eager}"

mkdir -p logs

# 1. Generate
log_run() {
  local split="$1" outpath="$2"
  local extra=()
  if [ -s "$outpath" ]; then
    echo "[run] $split: $outpath exists, resuming"
    extra+=("--resume")
  fi
  "$PYTHON" run_aa_lcr_gptoss.py \
    --in "$split.csv" --out "$outpath" \
    --attn-impl "$ATTN_IMPL" \
    "${extra[@]}" \
    > "logs/${split}_run.out" 2> "logs/${split}_run.err"
}
log_run profiling output_profiling.csv
log_run testing output_testing.csv

# 2. No-garbage check (vs ground truth) on profiling output
"$PYTHON" evaluate_no_garbage.py --in output_profiling.csv \
  --out scores_profiling_vs_gt.csv > logs/no_garbage_check.txt
echo "----- no-garbage check -----"
cat logs/no_garbage_check.txt
echo "----------------------------"

# 3. F1 vs uncompressed (used by the simulator)
"$PYTHON" evaluate_quality_f1.py --in output_profiling.csv --out scores_profiling.csv --use-base
"$PYTHON" evaluate_quality_f1.py --in output_testing.csv  --out scores_testing.csv  --use-base

# 4. Simulate and plot
"$PYTHON" simulate.py
