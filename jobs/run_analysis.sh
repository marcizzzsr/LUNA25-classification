#!/bin/bash
#SBATCH --job-name=hiera-luna25-analysis
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/analysis/%x_%j.out

set -euo pipefail

# Run an analysis script inside the container. Submit from jobs/:
#   cd jobs
#   mkdir -p logs/analysis
#   sbatch run_analysis.sh analysis/diagnostics_truncation.py [args...]
#   sbatch --export=ALL,LIMIT=50 run_analysis.sh analysis/diagnostics_truncation.py   # smoke test

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPT="$PROJECT_ROOT/${1:?usage: sbatch run_analysis.sh analysis/<script>.py [args...]}"
shift
SLOW_DATASETS=/mnt/data/unisr-data/datasets

set -a
source "$SCRIPT_DIR/.env"
set +a

: "${IMAGE_DIR:?IMAGE_DIR is not set}"
: "${MODELS:?MODELS is not set}"
[[ -f "$IMAGE_DIR" ]] || { echo "Missing container image: $IMAGE_DIR"; exit 1; }
[[ -d "$SLOW_DATASETS" ]] || { echo "Missing slow datasets dir: $SLOW_DATASETS"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Missing script: $SCRIPT"; exit 1; }

apptainer exec \
    --bind "$SLOW_DATASETS:/datasets/slow" \
    --bind "$MODELS:/models" \
    --pwd "$PROJECT_ROOT" \
    --env PYTHONPATH="/models/hiera-luna25/.pylib" \
    --env LUNA25_RESOURCES_DIR=/models/hiera-luna25 \
    --env LUNA25_RESULTS_DIR=/models/hiera-luna25/results \
    --env LIMIT="${LIMIT:-0}" \
    "$IMAGE_DIR" \
    python -u "$SCRIPT" "$@"
