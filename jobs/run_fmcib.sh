#!/bin/bash
#SBATCH --job-name=fmcib
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/fmcib/%x_%j.out

set -euo pipefail

# Extract FMCIB features (from the nodule blocks, full-CT fallback). Submit from jobs/:
#   mkdir -p logs/fmcib
#   sbatch run_fmcib.sh [script args...]
#   LIMIT=20 sbatch run_fmcib.sh /out/luna25_fmcib_features.csv --save-patch /out/patch.npy
# Output: $MODELS/fmcib/features/luna25_fmcib_features.csv

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FMCIB_IMAGE=/mnt/data/unisr-data/apptainer_images/images/fmcib/image.sif

set -a
source "$SCRIPT_DIR/.env"
set +a

: "${DATASETS:?DATASETS is not set}"
: "${MODELS:?MODELS is not set}"
[[ -f "$FMCIB_IMAGE" ]] || { echo "Missing container image: $FMCIB_IMAGE"; exit 1; }
[[ -f "$MODELS/fmcib/model_weights.torch" ]] || { echo "Missing FMCIB weights"; exit 1; }
mkdir -p "$MODELS/fmcib/features"

# fmcib loads model_weights.torch from the CWD -> --pwd /weights
apptainer exec --nv \
    --bind "$DATASETS:/datasets:ro" \
    --bind "$MODELS/fmcib:/weights:ro" \
    --bind "$MODELS/fmcib/features:/out" \
    --bind "$PROJECT_ROOT:/project:ro" \
    --pwd /weights \
    --env LIMIT="${LIMIT:-0}" \
    "$FMCIB_IMAGE" \
    /app/.venv/bin/python -u /project/analysis/extract_fmcib_features.py "$@"
