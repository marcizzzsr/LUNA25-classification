# Portions of this code are adapted from luna25-baseline-public
# Source: https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/experiment_config.py
# License: Apache License 2.0 (see https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/LICENSE)

# Please note that this configuration is provided as a template. The parameters included represent the submitted version for the LUNA25 Challenge.

import os
from pathlib import Path


class Configuration(object):
    def __init__(self, kind="k-fold") -> None:
        assert kind in ["all-data", "k-fold", "submission"], \
            f"Invalid kind: {kind}. Must be one of ['all-data', 'k-fold', 'submission']."
        self.kind = kind
        root = Path(__file__).resolve().parent

        # Directories
        # Defaults are relative to the repository; override them with the LUNA25_* environment variables.
        self.WORKDIR = root
        self.RESOURCES = Path(os.environ.get("LUNA25_RESOURCES_DIR", root / "resources"))
        # Path to the nodule blocks folder provided for the LUNA25 training.
        self.DATADIR = Path(os.environ.get("LUNA25_DATA_DIR", root.parent / "luna25_nodule_blocks"))
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path(os.environ.get("LUNA25_SPLITS_DIR", root / "data_splits"))
        # Results will be saved in the results directory
        # inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = Path(os.environ.get("LUNA25_RESULTS_DIR", root / "results"))
        # NLST diameters per AnnotationID, used for the size-matched validation AUC.
        self.DIAMETER_CSV = Path(os.environ.get(
            "LUNA25_DIAMETER_CSV",
            "/mnt/data/unisr-data/datasets/LUNA25/additional-annotations/csv-annotations/lesion_nlst_diameters_nolocal.csv"))
        self.CONFIGS_DIR = self.WORKDIR / "configs"
        if kind != "submission":
            self.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Train
        self.EXPERIMENT_NAME = "finetune-hiera"  # must contain "hiera"
        self.MODE = "3D"
        self.SEED = 2025
        self.NUM_WORKERS = 8
        # The effective batch size is self.MINI_BATCH_SIZE * self.GRADIENT_ACCUMULATION_STEPS.
        self.MINI_BATCH_SIZE = 64
        self.GRADIENT_ACCUMULATION_STEPS = 1
        self.PATIENCE = 10
        # Model selection: size-matched AUC (pair weight sigma on log-diameter),
        # smoothed with a centered moving average over SMOOTH_WINDOW epochs.
        self.SELECTION_SIGMA = 0.15
        self.SMOOTH_WINDOW = 3
        # Focal loss (gamma, alpha) for training; None = BCE.
        self.FOCAL = None
        # Class-balanced WeightedRandomSampler; False = plain shuffling.
        self.BALANCED_SAMPLER = True
        self.NUM_GPUS = 1
        self.NUM_FOLDS = 5

        # Data & Augmentation
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.DEPTH_PX = 16
        # Output z spacing in mm; None = isotropic SIZE_MM / SIZE_PX
        self.Z_SPACING_MM = None
        # Random isotropic zoom range (min, max) for training; None = off. <1 zooms out, >1 zooms in
        self.ZOOM = None
        self.PATCH_SIZE = [64, 128, 128]
        self.ROTATION = ((-45, 45), (-45, 45), (-45, 45))
        self.TRANSLATION = True
        self.MIXUP = {
            "ENABLE": True,
            "ALPHA": 0.8,
            "CUTMIX_ALPHA": 1.0,
            "PROB": 1.0,
            "SWITCH_PROB": 0.5,
            "LABEL_SMOOTH_VALUE": 0.1
        }

        # Hiera
        self.DROP_PATH_RATE = 0.2
        self.HEAD_DROPOUT = 0.2
        self.HEAD_INIT_SCALE = 0.001
        self.PRETRAINED_MODEL = "mae_hiera_large_16x224"
        self.PRETRAINED_MODEL_CONFIGS = {
            "mae_hiera_base_16x224": {
                "embed_dim": 96,
                "num_heads": 1,
                "stages": (2, 3, 16, 3),
                "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth"
            },
            "mae_hiera_base_plus_16x224": {
                "embed_dim": 112,
                "num_heads": 2,
                "stages": (2, 3, 16, 3),
                "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth"
            },
            "mae_hiera_large_16x224": {
                "embed_dim": 144,
                "num_heads": 2,
                "stages": (2, 6, 36, 4),
                "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth"
            },
            "mae_hiera_huge_16x224": {
                "embed_dim": 256,
                "num_heads": 4,
                "stages": (2, 6, 36, 4),
                "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth"
            }
        }
        self.FUSION_HEAD_ENABLED = False
        # Attemp to fix the bug (https://arxiv.org/pdf/2311.05613)
        self.HACK = {
            "ENABLE": False,
            "NPY_DIR_POS_EMBED": self.CONFIGS_DIR / "pos_embed.npy",
            "NPY_DIR_POS_EMBED_WINDOW": self.CONFIGS_DIR / "pos_embed_window.npy",
            "sam2.1_hiera_tiny": {
                "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
            },
            "sam2.1_hiera_small": {
                "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
            },
            "sam2.1_hiera_base_plus": {
                "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
            },
            "sam2.1_hiera_large": {
                "checkpoint": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
            }
        }

        # Solver
        self.OPTIMIZING_METHOD = "adamw"
        self.BASE_LR = 5e-5
        self.COSINE_END_LR = 5e-8
        self.MAX_EPOCH = 50
        self.BETAS = (0.9, 0.999)
        self.WEIGHT_DECAY = 0.05
        # Unused if OPTIMIZING_METHOD is not "sgd"
        self.MOMENTUM = 0.9
        self.DAMPENING = 0.0
        self.NESTEROV = True
        # Start the warm up from BASE_LR * WARMUP_FACTOR.
        self.WARMUP_FACTOR = 0.1
        # Gradually warm up the BASE_LR over this number of epochs.
        self.WARMUP_EPOCHS = 10
        # If True, start from the peak cosine learning rate after warm up.
        self.COSINE_AFTER_WARMUP = True
        # If True, perform no weight decay on parameter with one dimension (bias term, etc).
        self.ZERO_WD_1D_PARAM = True
        # The layer-wise decay of learning rate. Set to 1. to disable.
        self.LAYER_DECAY = 1.0
        self.CLIP_GRAD_L2NORM = 5.0

        if kind == "all-data":
            self.CSV_DIR = Path(os.environ.get("LUNA25_ANNOTATION_DIR", root.parent))
            self.CSV_DIR_TRAIN = self.CSV_DIR / "LUNA25_Public_Training_Development_Data.csv"
            self.EPOCHS = 29


config = Configuration(kind="submission")
