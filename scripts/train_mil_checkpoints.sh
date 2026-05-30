#!/usr/bin/env bash
# Train every MIL aggregator (9 archs × 5 folds) on the full bag, saving
# per-fold "best" checkpoints to ``$CHECKPOINT_DIR``. The output layout is
# what `eval.py --model <arch>` expects — the eval pipeline simply re-loads
# these and runs inference at the requested patch budget.
#
# Run this ONCE per dataset (no sampling involved); afterwards the eval
# CLI can be called with any budget / sampler combo.
#
# Usage
# -----
#   bash scripts/train_mil_checkpoints.sh --dataset cm16
#   bash scripts/train_mil_checkpoints.sh --dataset cm17
#   bash scripts/train_mil_checkpoints.sh --dataset nsclc
#
# Env vars (resolved through configs/paths.json)
# ----------------------------------------------
#   FEATURE_ROOT       CONCH v1.5 patch features
#   COORD_ROOT         CLAM-0402 coord layout
#   CM16_RAW_ROOT      official CM16 lesion-annotation zip (only for --dataset cm16)
#   CM17_LABEL_CSV     official CM17 stages.csv          (only for --dataset cm17)
#   CHECKPOINT_DIR     where to write <arch>_fold<N>_best.pt
#                      default: outputs/mil_checkpoints/<dataset>/checkpoints
#
# Hyperparameters (paper defaults)
# --------------------------------
#   train_epoch=30, lr=1e-4, wd=1e-5, eval_interval=5
#
# Override any of those by exporting TRAIN_EPOCH / LR / WD / EVAL_INTERVAL.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/load_paths.sh" "$REPO_ROOT/configs/paths.json"

DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --output_dir|--out) CHECKPOINT_DIR="$2"; shift 2 ;;
    --mil_archs) shift; MIL_ARCHS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MIL_ARCHS+=("$1"); shift; done ;;
    *) echo "[train_mil] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  echo "Usage: bash scripts/train_mil_checkpoints.sh --dataset {cm16|cm17|nsclc}" >&2
  exit 2
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/mil_checkpoints/${DATASET}}"
mkdir -p "$CHECKPOINT_DIR"

TRAIN_EPOCH="${TRAIN_EPOCH:-30}"
LR="${LR:-1e-4}"
WD="${WD:-1e-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

# Default = train all 9 MIL aggregators (Tab. 1 row set).
if [[ -z "${MIL_ARCHS+x}" ]]; then
  MIL_ARCHS=(abmil clam_sb dftd dsmil ilra rrt mha transmil wikg)
fi

EXTRA_ARGS=()
case "$DATASET" in
  cm16)
    EXTRA_ARGS+=(--cm16_split_mode official --cm16_raw_root "$CM16_RAW_ROOT")
    ;;
  cm17)
    if [[ -z "${CM17_LABEL_CSV:-}" ]]; then
      echo "[train_mil] CM17_LABEL_CSV must be set for --dataset cm17." >&2
      exit 2
    fi
    EXTRA_ARGS+=(--label_csv "$CM17_LABEL_CSV")
    ;;
  nsclc)
    : ;;
  *)
    echo "[train_mil] unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

echo "[train_mil] dataset=$DATASET  archs=${MIL_ARCHS[*]}"
echo "[train_mil] checkpoint_dir=$CHECKPOINT_DIR"
echo "[train_mil] train_epoch=$TRAIN_EPOCH lr=$LR wd=$WD eval_interval=$EVAL_INTERVAL"

"$PYTHON" evaluator/runners/mil_comparison.py \
    --dataset "$DATASET" \
    --feature_root "$FEATURE_ROOT" \
    --mil_archs "${MIL_ARCHS[@]}" \
    --train_epoch "$TRAIN_EPOCH" \
    --lr "$LR" \
    --wd "$WD" \
    --eval_interval "$EVAL_INTERVAL" \
    --output_dir "$CHECKPOINT_DIR" \
    "${EXTRA_ARGS[@]}"

echo "[train_mil] done. Pass CHECKPOINT_DIR=$CHECKPOINT_DIR/checkpoints to eval.py."
