#!/bin/bash
#SBATCH --job-name=sws
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/sws/%x_%j.out

set -euo pipefail

# Duke ResNet50-SWS++ inference on the Hiera validation lesions (PiNS image). Submit from jobs/:
#   mkdir -p logs/sws
#   sbatch run_sws.sh
#   LIMIT=20 sbatch run_sws.sh
# Patches: $SLOW_DATASETS/LUNA25/sws_patches_64q/  Predictions: $MODELS/DLCSD-models/results/sws-pp-20261028/

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PINS_IMAGE=/mnt/data/unisr-data/apptainer_images/images/pins/image.sif
SLOW_DATASETS=/mnt/data/unisr-data/datasets

set -a
source "$SCRIPT_DIR/.env"
set +a

: "${DATASETS:?DATASETS is not set}"
: "${MODELS:?MODELS is not set}"
[[ -f "$PINS_IMAGE" ]] || { echo "Missing container image: $PINS_IMAGE"; exit 1; }
[[ -f "$MODELS/DLCSD-models/DukeLungRADS_CancerCL_Fold1_resnet50WSPP.pt" ]] || { echo "Missing SWS++ weights"; exit 1; }
mkdir -p "$SLOW_DATASETS/LUNA25/sws_patches_64q" "$MODELS/DLCSD-models/results"

apptainer exec --nv \
    --bind "$DATASETS:/datasets:ro" \
    --bind "$SLOW_DATASETS/LUNA25/sws_patches_64q:/patches" \
    --bind "$MODELS/DLCSD-models:/dlcsd" \
    --bind "$PROJECT_ROOT:/project:ro" \
    --env LIMIT="${LIMIT:-0}" \
    "$PINS_IMAGE" \
    /usr/local/bin/python3 -u /project/analysis/sws_inference.py
