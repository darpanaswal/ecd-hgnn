#!/bin/bash
#SBATCH --job-name=ecd
#SBATCH --output=/dev/null                 # no tiny driver logs
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100

set -euo pipefail

# Usage:
#   sbatch [--array=START-END%CONCURRENCY] slurm.sh <base|pos|classw|pos_classw> <euclidean|lorentz|poincare>
# Examples:
#   sbatch slurm.sh base euclidean
#   sbatch --array=0-1%2 slurm.sh pos lorentz
#   sbatch --array=0-1%4 slurm.sh classw poincare
#   sbatch --array=0-5%4 slurm.sh pos_classw euclidean

CONFIG="${1:-}"
SPACE="${2:-poincare}"

if [[ -z "$CONFIG" ]]; then
  echo "Usage: sbatch [--array=START-END%CONCURRENCY] slurm.sh <base|pos|classw|pos_classw> <euclidean|lorentz|poincare>"
  exit 1
fi
case "$CONFIG" in
  base|pos|classw|pos_classw) ;;
  *) echo "Invalid CONFIG '$CONFIG'. Choose: base, pos, classw, pos_classw"; exit 1;;
esac
case "$SPACE" in
  euclidean|lorentz|poincare) ;;
  *) echo "Invalid SPACE '$SPACE'. Choose: euclidean, lorentz, poincare"; exit 1;;
esac

# Modules & env
module load anaconda3/2024.06/gcc-13.2.0
module load cuda/12.2.1/gcc-11.2.0
module load gcc/11.2.0/gcc-4.8.5
source activate ecdgnn

mkdir -p output_logs

# Per-chunk log only
CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK_LOG="output_logs/${SPACE}-${CONFIG}.chunk${CHUNK_ID}.out"

# Grids
DROPOUTS=(0 0.1 0.2 0.25 0.3)
POS_DIMS=(16 32 64 128)
CLASS_W_SETS=("0.6678,1.9897" "0.8,1.6" "1,1.5")  # comma-separated to avoid word-splitting

declare -a RUN_SPECS=()
if [[ "$CONFIG" == "base" ]]; then
  for d in "${DROPOUTS[@]}"; do RUN_SPECS+=("dropout=$d"); done
elif [[ "$CONFIG" == "pos" ]]; then
  for d in "${DROPOUTS[@]}"; do
    for pd in "${POS_DIMS[@]}"; do RUN_SPECS+=("dropout=$d pos_embed_dim=$pd"); done
  done
elif [[ "$CONFIG" == "classw" ]]; then
  for d in "${DROPOUTS[@]}"; do
    for cw in "${CLASS_W_SETS[@]}"; do RUN_SPECS+=("dropout=$d class_weights=$cw"); done
  done
elif [[ "$CONFIG" == "pos_classw" ]]; then
  for d in "${DROPOUTS[@]}"; do
    for cw in "${CLASS_W_SETS[@]}"; do
      for pd in "${POS_DIMS[@]}"; do RUN_SPECS+=("dropout=$d pos_embed_dim=$pd class_weights=$cw"); done
    done
  done
fi

TOTAL=${#RUN_SPECS[@]}
CHUNK_SIZE=10
START=$(( ${SLURM_ARRAY_TASK_ID:-0} * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
(( END > TOTAL )) && END=$TOTAL

# If no array was given (e.g., base), run only the first chunk
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  START=0
  END=$(( TOTAL < CHUNK_SIZE ? TOTAL : CHUNK_SIZE ))
fi

if (( START >= TOTAL )); then
  echo "Chunk ${CHUNK_ID} out of range for ${CONFIG} (TOTAL=${TOTAL}); nothing to do." | tee -a "$CHUNK_LOG"
  exit 0
fi

# Chunk header (per-chunk log)
{
  echo "========== CHUNK START =========="
  echo "Config: ${CONFIG} | SPACE: ${SPACE}"
  echo "Chunk: ${CHUNK_ID} | Host: $(hostname) | Time: $(date -Is)"
  echo "Total combos: ${TOTAL} | This chunk indices: [${START}, ${END}) (≤ ${CHUNK_SIZE})"
  if [[ "${CHUNK_ID}" == "0" ]]; then
    echo "CUDA info:"; nvidia-smi || true
  fi
  echo "================================="
} | tee -a "$CHUNK_LOG"

# Run this chunk sequentially
for (( i=START; i<END; i++ )); do
  spec="${RUN_SPECS[$i]}"

  # Parse key=val pairs
  declare -A KV=()
  for pair in $spec; do
    k="${pair%%=*}"
    v="${pair#*=}"
    KV["$k"]="$v"
  done

  run_args=(--task ecd --select_manifold "$SPACE" --compute_roc_auc)
  run_args+=(--dropout "${KV[dropout]}")
  TAG="cfg=${CONFIG} space=${SPACE} do=${KV[dropout]}"

  if [[ "$CONFIG" == "pos" || "$CONFIG" == "pos_classw" ]]; then
    run_args+=(--use_pos_tags --pos_embed_dim "${KV[pos_embed_dim]}")
    TAG="${TAG} pos=${KV[pos_embed_dim]}"
  fi
  if [[ "$CONFIG" == "classw" || "$CONFIG" == "pos_classw" ]]; then
    IFS=',' read -r cw0 cw1 <<< "${KV[class_weights]}"
    run_args+=(--use_class_weights --class_weight_values "$cw0" "$cw1")
    TAG="${TAG} cw=[$cw0,$cw1]"
  fi

  {
    echo ""
    echo "----- RUN START (global idx $i) -----"
    echo "Args: python main.py ${run_args[*]}"
    echo "Time: $(date -Is)"
  } | tee -a "$CHUNK_LOG"

  if ! srun -u python main.py "${run_args[@]}" 2>&1 | tee -a "$CHUNK_LOG"; then
    echo "RUN FAILED (idx $i): ${TAG}" | tee -a "$CHUNK_LOG"
  else
    echo "RUN DONE (idx $i): ${TAG}" | tee -a "$CHUNK_LOG"
  fi
done

echo "========== CHUNK END ($(date -Is)) ==========" | tee -a "$CHUNK_LOG"