#!/usr/bin/env bash
# Distill a patch encoder according to configs/distill/<name>.json.
# Usage:  bash scripts/distill.sh configs/distill/conchv15.json
set -euo pipefail

CONFIG="${1:?Usage: bash scripts/distill.sh configs/distill/<name>.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load global paths first, then the per-experiment config (env-style flatten).
source "$SCRIPT_DIR/load_paths.sh" "$REPO_ROOT/configs/paths.json"
source "$SCRIPT_DIR/load_paths.sh" "$CONFIG"

# Compute optional flag arrays based on config booleans.
EXTRA_ARGS=()
if [[ "${COMPILE:-0}" == "1" ]]; then EXTRA_ARGS+=(--compile); fi
if [[ "${ONLY_AFFINITY:-0}" == "1" ]]; then EXTRA_ARGS+=(--only_affinity); fi
if [[ "${ONLY_DISTANCE:-0}" == "1" ]]; then EXTRA_ARGS+=(--only_distance); fi
if [[ "${WANDB:-0}" == "1" ]]; then EXTRA_ARGS+=(--wandb); else EXTRA_ARGS+=(--no_wandb); fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export TOKENIZERS_PARALLELISM=false

NPROC="${NPROC:-8}"

torchrun --nproc_per_node="$NPROC" --master_port "$MASTER_PORT" distill/distill.py \
  --data_dir "$DISTILL_DATA_DIR" \
  --output_base "$DISTILL_OUTPUT_DIR" \
  --eval_cache_dir "$EVAL_CACHE_DIR" \
  --teacher_model "$TEACHER_MODEL" \
  --student_dim "$STUDENT_DIM" \
  --student_depth "$STUDENT_DEPTH" \
  --student_heads "$STUDENT_HEADS" \
  --student_mlp_ratio "$STUDENT_MLP_RATIO" \
  --student_patch_size "$STUDENT_PATCH_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --max_steps "$MAX_STEPS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --optimizer "$OPTIMIZER" \
  --precision "$PRECISION" \
  --structure_temperature ${STRUCTURE_TEMPERATURE} \
  --structure_loss_weight "${STRUCTURE_LOSS_WEIGHT:-0.0}" \
  --cls_structure_loss_weight "${CLS_STRUCTURE_LOSS_WEIGHT:-0.0}" \
  --global_structure_loss_weight "${GLOBAL_STRUCTURE_LOSS_WEIGHT:-1.0}" \
  --feat_loss_weight "${FEAT_LOSS_WEIGHT:-0.0}" \
  --cls_loss_weight "${CLS_LOSS_WEIGHT:-0.0}" \
  --cosine_loss_weight "${COSINE_LOSS_WEIGHT:-0.0}" \
  --lr_warmup_steps 1000 \
  --lr_schedule cosine \
  --min_lr_ratio 0.01 \
  --image_index_workers "${IMAGE_INDEX_WORKERS:-32}" \
  --eval_iter 500 \
  --linear_prob_iter 500 \
  --pool_method cls \
  --log_every 50 \
  --save_every "${SAVE_EVERY:-10000}" \
  "${EXTRA_ARGS[@]}"
