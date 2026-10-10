#!/bin/bash
# Evaluate all 5 trained variants (v3-v7) on the test set.
# Assumes trainings are complete (net_epoch60.pth exists in each log dir).
#
# Usage: bash eval_all_variants.sh
# Output: for each variant, <log_dir>/v2x/{eval,error_eval,short_eval,short_error_eval,...}.yaml
set -e

PY=/home/ustc_user1/miniconda3/envs/v2v4real/bin/python
cd /bdata3/root/v2x/V2V4Real/opencood

# variant name -> log dir glob
eval_model() {
  local label="$1"
  local dir="$2"
  echo "=========== Evaluating $label ==========="
  echo "log dir: $dir"
  if [ ! -f "$dir/net_epoch60.pth" ]; then
    echo "  [SKIP] net_epoch60.pth not found"
    return
  fi
  # GPU isolation: use an idle GPU. Pass via env. Pick one free GPU.
  # Free GPU logic: run on GPU where memory.used is smallest.
  local gpu
  gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sort -t, -k2 -n | head -1 | cut -d, -f1)
  echo "  using GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" tools/inference.py \
      --model_dir "$dir" \
      --fusion_method early
  echo "  done: $dir/v2x/eval.yaml"
}

# Use the exact log dirs (find latest for each)
BASE=/bdata3/root/v2x/V2V4Real/opencood/logs
eval_model "v3" "$(ls -td $BASE/point_pillar_early_fusion_ego_preserving_2026* | head -1)"
eval_model "v4" "$(ls -td $BASE/*v4_2026* | head -1)"
eval_model "v5" "$(ls -td $BASE/*v5_2026* | head -1)"
eval_model "v6" "$(ls -td $BASE/*v6_2026* | head -1)"
eval_model "v7" "$(ls -td $BASE/*v7_2026* | head -1)"

echo "=========== All evaluations complete ==========="
