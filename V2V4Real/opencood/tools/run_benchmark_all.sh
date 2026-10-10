#!/usr/bin/env bash
# Benchmark per-frame inference time for all paired V2V4Real models.
set -u
PY=/home/ustc_user1/miniconda3/envs/v2v4real/bin/python
BASE=/bdata3/root/v2x/V2V4Real
OUT=$BASE/opencood/bench_inference_time
mkdir -p "$OUT"
GPU=${GPU_ID:-1}

MAX_FRAMES=${MAX_FRAMES:-100}
REPEATS=${REPEATS:-3}
WARMUP=${WARMUP:-10}

# name:fusion_method
MODELS=(
  "coalign_8_09:intermediate"
  "coalign_ego_8_09:intermediate"
  "CoBEVT:intermediate"
  "CoBEVT_ego_8_30:intermediate"
  "cost:intermediate"
  "cost_ego:intermediate"
  "early_fusion:early"
  "early_fusion_ego:early"
  "Fcooper:intermediate"
  "Fcooper_ego:intermediate"
  "pointpillar_late:late"
  "pointpillar_late_ego:late"
  "second_late:late"
  "second_late_ego:late"
  "second_early:early"
  "second_early_ego:early"
)

cd "$BASE"
for spec in "${MODELS[@]}"; do
  m="${spec%%:*}"; f="${spec##*:}"
  echo "============================================================"
  echo "== $m  ($f)  gpu=$GPU"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -m opencood.tools.benchmark_inference_time \
      --model_dir "$BASE/model/$m" \
      --fusion_method "$f" \
      --max_frames "$MAX_FRAMES" \
      --repeats "$REPEATS" \
      --warmup "$WARMUP" \
      --output_json "$OUT/${m}.json" 2>&1 \
      | grep -vE "UserWarning|warnings.warn|RuntimeWarning|invalid value"
done
echo "ALL DONE -> $OUT"
