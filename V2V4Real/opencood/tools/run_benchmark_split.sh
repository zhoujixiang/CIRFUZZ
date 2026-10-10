#!/usr/bin/env bash
# Run benchmark on a subset of models on one GPU.
# Usage: run_benchmark_split.sh <gpu_id> <max_frames> <repeats> <warmup> <name:fusion> ...
set -u
GPU=$1; MAX_FRAMES=$2; REPEATS=$3; WARMUP=$4; shift 4
PY=/home/ustc_user1/miniconda3/envs/v2v4real/bin/python
BASE=/bdata3/root/v2x/V2V4Real
OUT=$BASE/opencood/bench_inference_time
mkdir -p "$OUT"
cd "$BASE"
for spec in "$@"; do
  m="${spec%%:*}"; f="${spec##*:}"
  echo "== [gpu$GPU] $m ($f) start $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m opencood.tools.benchmark_inference_time \
      --model_dir "$BASE/model/$m" --fusion_method "$f" \
      --max_frames "$MAX_FRAMES" --repeats "$REPEATS" --warmup "$WARMUP" \
      --output_json "$OUT/${m}.json" 2>&1 \
      | grep -vE "UserWarning|warnings.warn|RuntimeWarning|invalid value"
  echo "== [gpu$GPU] $m done $(date +%H:%M:%S)"
done
echo "SPLIT DONE gpu$GPU"
