#!/usr/bin/env bash
# Train the MLP projector that maps distilled-student features back into the
# teacher's eval-space. Used for ablation A6 / fill_projected experiments.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/load_paths.sh" "$REPO_ROOT/configs/paths.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29503}"
NPROC="${NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}"

STUDENT_CKPT_PATH="${STUDENT_CKPT_PATH:?Set STUDENT_CKPT_PATH to the distilled checkpoint .pt}"
OUTPUT_PATH="${OUTPUT_PATH:-${MLP_PROJECTOR_DIR}/conchv15_student384_to_teacher768.pt}"

torchrun --nproc_per_node="$NPROC" --master_port "$MASTER_PORT" \
    distill/train_mlp_projector.py \
    --data_dir              "$DISTILL_DATA_DIR" \
    --teacher_model         "${TEACHER_MODEL:-conchv15}" \
    --student_ckpt_path     "$STUDENT_CKPT_PATH" \
    --output_path           "$OUTPUT_PATH" \
    --mlp_hidden_dim        "${MLP_HIDDEN_DIM:-768}" \
    --mlp_num_hidden_layers "${MLP_NUM_HIDDEN_LAYERS:-2}" \
    --mlp_activation        "${MLP_ACTIVATION:-silu}" \
    --batch_size            "${BATCH_SIZE:-1024}" \
    --max_steps             "${MAX_STEPS:-5000}" \
    --lr                    "${LR:-1e-3}" \
    --weight_decay          "${WEIGHT_DECAY:-1e-4}" \
    --lr_warmup_steps       "${LR_WARMUP_STEPS:-500}" \
    --lr_schedule cosine --min_lr_ratio 0.01 \
    --image_index_workers   "${IMAGE_INDEX_WORKERS:-32}" \
    --log_every 50 --save_every 0 \
    --precision "${PRECISION:-fp16}" --compile \
    $([[ "${WANDB:-0}" == "1" ]] && echo "--wandb --wandb_log_every ${WANDB_LOG_EVERY:-10}" || echo "--no_wandb")
