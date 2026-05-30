#!/usr/bin/env bash
# Run the distilled student encoder over WSI patches and write per-slide
# .pt feature tensors that the speculative-encoding sampler consumes.
#
# Inputs (one of):
#   --checkpoint <student.pt>     ← the distilled student you trained
#   --teacher_model <name>        ← the original teacher (HF download)
#                                   names: conchv15 | uni | virchow | prism |
#                                          provgigapath | conch | …
# Plus:
#   --wsi_root   <dir of raw .svs>
#   --coord_root <CLAM-0402 coord_root>  (one .npy per slide, x/y/tile_size_lv0)
#   --output_root <where to write per-slide .pt>
#
# Output: a mirror of coord_root with one <slide_id>.pt of shape [N_patches, D]
#         per slide.
#
# Usage examples:
#
#   # Distilled student
#   bash scripts/extract_features.sh \
#       --checkpoint outputs/distilled_models/<run>/checkpoint_10000.pt \
#       --wsi_root  /data/raw_wsi/cm16 \
#       --coord_root $COORD_ROOT/cm16 \
#       --output_root $DISTILLED_FEATURE_ROOT/cm16
#
#   # Original teacher (CONCH v1.5)
#   bash scripts/extract_features.sh \
#       --teacher_model conchv15 \
#       --wsi_root  /data/raw_wsi/cm16 \
#       --coord_root $COORD_ROOT/cm16 \
#       --output_root $FEATURE_ROOT/cm16
#
#   # multi-GPU (8 ranks), uses NPROC env var
#   NPROC=8 bash scripts/extract_features.sh ...
#
# Required env vars (autoloaded from configs/paths.json):
#   COORD_ROOT, DISTILLED_FEATURE_ROOT  (default values for the *_root args)
#
# Optional env vars: BATCH_SIZE, NUM_WORKERS, PRECISION, OVERWRITE, NPROC

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/load_paths.sh" "$REPO_ROOT/configs/paths.json"

CHECKPOINT_PATH=""
TEACHER_MODEL=""
WSI_ROOT=""
COORD_ROOT_ARG=""
OUTPUT_ROOT_ARG=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|--checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --teacher_model|--teacher)      TEACHER_MODEL="$2";   shift 2 ;;
    --wsi_root)                     WSI_ROOT="$2";        shift 2 ;;
    --coord_root)                   COORD_ROOT_ARG="$2";  shift 2 ;;
    --output_root)                  OUTPUT_ROOT_ARG="$2"; shift 2 ;;
    --overwrite)                    EXTRA+=(--overwrite); shift ;;
    *)                              EXTRA+=("$1");        shift ;;
  esac
done

if [[ -z "$CHECKPOINT_PATH" && -z "$TEACHER_MODEL" ]]; then
  echo "Pass exactly one of --checkpoint <student.pt> or --teacher_model <name>." >&2
  exit 2
fi
if [[ -n "$CHECKPOINT_PATH" && -n "$TEACHER_MODEL" ]]; then
  echo "--checkpoint and --teacher_model are mutually exclusive." >&2
  exit 2
fi

: "${WSI_ROOT:?--wsi_root <directory of .svs> is required}"
COORD_ROOT_ARG="${COORD_ROOT_ARG:-${COORD_ROOT}}"
# Default output: distilled root for student, FEATURE_ROOT for teacher.
if [[ -z "$OUTPUT_ROOT_ARG" ]]; then
  if [[ -n "$CHECKPOINT_PATH" ]]; then OUTPUT_ROOT_ARG="${DISTILLED_FEATURE_ROOT}"
  else                                  OUTPUT_ROOT_ARG="${FEATURE_ROOT}"
  fi
fi

NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-fp16}"

CMD=("${PYTHON:-python}" distill/extract_features.py
     --wsi_root        "$WSI_ROOT"
     --coord_root      "$COORD_ROOT_ARG"
     --output_root     "$OUTPUT_ROOT_ARG"
     --batch_size      "$BATCH_SIZE"
     --num_workers     "$NUM_WORKERS"
     --precision       "$PRECISION"
     "${EXTRA[@]}")
if [[ -n "$CHECKPOINT_PATH" ]]; then
  CMD+=(--checkpoint_path "$CHECKPOINT_PATH")
else
  CMD+=(--teacher_model "$TEACHER_MODEL")
fi

if [[ "$NPROC" -gt 1 ]]; then
  exec torchrun --nproc_per_node="$NPROC" --master_port "${MASTER_PORT:-29503}" \
       "${CMD[@]:1}"   # drop the leading $PYTHON for torchrun
else
  exec "${CMD[@]}"
fi
