"""Counterfactual rescaling inference: logits of each fold's model on validation crops zoomed by s."""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torchvision
torchvision.disable_beta_transforms_warning()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from experiment_config import config
from dataloader import get_data_loader
from models.model_hiera import Hiera3D

BLOCKS_DIR = "/datasets/slow/LUNA25/luna25_nodule_blocks"
WORKSPACE = "/models/hiera-luna25"
RESULTS_DIR = f"{WORKSPACE}/results/finetune-hiera-3D-20260814"
OUT_CSV = Path(f"{WORKSPACE}/figures/diagnostics/rescale_predictions.csv")
SCALES = [0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
FOLDS = range(1, 6)
BATCH_SIZE, WORKERS = 64, 4

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_pred = pd.read_csv(f"{RESULTS_DIR}/all_folds_predictions.csv")
t0 = time.time()
out = []
for k in FOLDS:
    model = Hiera3D(image_size=config.SIZE_PX, image_depth=config.DEPTH_PX, kind="finetuned").cuda()
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/fold_{k}/best_metric_model.pth", map_location="cuda"))
    model.eval()
    df_fold = df_pred[df_pred["fold"] == k].reset_index(drop=True)
    for s in SCALES:
        loader = get_data_loader(BLOCKS_DIR, df_fold, mode="3D", workers=WORKERS, batch_size=BATCH_SIZE,
                                 size_px=config.SIZE_PX, size_mm=config.SIZE_MM / s,
                                 rotations=None, translations=None)
        with torch.no_grad():
            logits = torch.cat([model(b["image"].cuda()).squeeze(-1).float().cpu() for b in loader])
        out.append(pd.DataFrame({"AnnotationID": df_fold["AnnotationID"], "fold": k, "label": df_fold["label"],
                                 "scale": s, "logit": logits.numpy()}))
        print(f"fold {k} scale {s}: {len(logits)} crops, {time.time() - t0:.0f}s", flush=True)
pd.concat(out).to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} in {time.time() - t0:.0f}s", flush=True)
