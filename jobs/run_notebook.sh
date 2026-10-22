#!/bin/bash
#SBATCH --job-name=hiera-luna25-notebook
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/notebook/%x_%j.out

set -euo pipefail

# -----------------------------------------------------------------------------
# Resolve project paths
#
# Jobs are submitted from the jobs/ directory:
#   cd jobs
#   mkdir -p logs/notebook
#   sbatch run_notebook.sh [notebook]   # default: visualize_nodule_size.ipynb
# -----------------------------------------------------------------------------

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NOTEBOOK="$PROJECT_ROOT/${1:-visualize_nodule_size.ipynb}"
# the notebooks read from /datasets/slow
SLOW_DATASETS=/mnt/data/unisr-data/datasets

echo "SCRIPT_DIR   = $SCRIPT_DIR"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "NOTEBOOK     = $NOTEBOOK"

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
: "${MODELS:?MODELS is not set}"


[[ -f "$IMAGE_DIR" ]]  || { echo "Missing container image: $IMAGE_DIR"; exit 1; }
[[ -d "$SLOW_DATASETS" ]]  || { echo "Missing slow datasets dir: $SLOW_DATASETS"; exit 1; }
[[ -d "$MODELS" ]]  || { echo "Missing models dir: $MODELS"; exit 1; }
[[ -f "$NOTEBOOK" ]]  || { echo "Missing notebook: $NOTEBOOK"; exit 1; }


echo "Done sanity checks!"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

# install nbconvert into $MODELS (persists across jobs) if the image lacks it

apptainer exec \
    --bind "$SLOW_DATASETS:/datasets/slow" \
    --bind "$MODELS:/models" \
    --pwd "$PROJECT_ROOT" \
    --env PYTHONPATH="/models/hiera-luna25/.pylib" \
    --env LUNA25_RESOURCES_DIR=/models/hiera-luna25 \
    --env LUNA25_RESULTS_DIR=/models/hiera-luna25/results \
    "$IMAGE_DIR" \
    bash -c '
        set -euo pipefail
        python -c "import nbconvert" 2>/dev/null || pip install --quiet --target /models/hiera-luna25/.pylib nbconvert
        python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$0"
    ' "$NOTEBOOK"
