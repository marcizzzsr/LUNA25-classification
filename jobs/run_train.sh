#!/bin/bash
#SBATCH --job-name=hiera-luna25
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train/%x_%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve project paths
#
# Jobs are submitted from the jobs/ directory:
#   cd jobs
#   sbatch infer.sh
#
# SLURM_SUBMIT_DIR points to the directory where sbatch was invoked.
# -----------------------------------------------------------------------------

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SLOW_DATASETS=/mnt/data/unisr-data/datasets

echo "SCRIPT_DIR   = $SCRIPT_DIR"
echo "PROJECT_ROOT = $PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Load environment
# -----------------------------------------------------------------------------

set -a
source "$SCRIPT_DIR/.env"
set +a

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

echo "Sanity checks..."

: "${IMAGE_DIR:?IMAGE_DIR is not set}"
: "${DATASETS:?DATASETS is not set}"
: "${MODELS:?MODELS is not set}"


[[ -f "$IMAGE_DIR" ]]  || { echo "Missing container image: $IMAGE_DIR"; exit 1; }
[[ -d "$DATASETS" ]]  || { echo "Missing datasets dir: $DATASETS"; exit 1; }
[[ -d "$MODELS" ]]  || { echo "Missing models dir: $MODELS"; exit 1; }


echo "Done sanity checks!"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

apptainer exec \
    --nv \
    --bind "$DATASETS:/datasets" \
    --bind "$SLOW_DATASETS:/datasets/slow" \
    --bind "$MODELS:/models" \
    --pwd "$PROJECT_ROOT" \
    --env LUNA25_DATA_DIR=/datasets/LUNA25/luna25_nodule_blocks \
    --env LUNA25_RESOURCES_DIR=/models/hiera-luna25 \
    --env LUNA25_RESULTS_DIR=/models/hiera-luna25/results \
    --env LUNA25_DIAMETER_CSV=/datasets/slow/LUNA25/additional-annotations/csv-annotations/lesion_nlst_diameters_nolocal.csv \
    "$IMAGE_DIR" \
    python -u "$PROJECT_ROOT/finetune.py" --k_fold "$@"
