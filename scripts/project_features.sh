#!/usr/bin/env bash
# Project distilled student features through an MLP projector to align them
# with the teacher's eval-space (used by ablation A6 and as feature fill-ins).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/load_paths.sh" "$REPO_ROOT/configs/paths.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29504}"
NPROC="${NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}"

PROJECTOR_PATH="${PROJECTOR_PATH:-${MLP_PROJECTOR_DIR}/conchv15_student384_to_teacher768.pt}"
INPUT_DIR="${INPUT_DIR:-${DISTILLED_FEATURE_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DISTILLED_PROJECT_FEATURE_DIR}}"

EXTRA=()
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA+=(--overwrite)

torchrun --nproc_per_node="$NPROC" --master_port "$MASTER_PORT" \
    distill/project_features.py \
    --projector_path  "$PROJECTOR_PATH" \
    --input_dir       "$INPUT_DIR" \
    --output_dir      "$OUTPUT_DIR" \
    --chunk_size      "${CHUNK_SIZE:-4096}" \
    --precision       "${PRECISION:-fp16}" \
    "${EXTRA[@]}"
