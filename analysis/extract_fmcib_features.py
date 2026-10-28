"""Extract FMCIB 4096-dim features for every LUNA25 lesion (GPU required).

Patches come from the pre-extracted nodule blocks (64x128x128 at native spacing, nodule at the
centre voxel), reproducing fmcib's preprocessing: 1 mm isotropic linear resampling, 50^3 crop with
the nodule at index 25, (HU + 1024) / 3072. Blocks that do not cover +-25 mm (thin slices) or with
a non-identity direction fall back to fmcib's own full-CT pipeline (get_features).

Usage: extract_fmcib_features.py [out_csv] [--save-patch PATH.npy]
Env: LIMIT=N processes only the first N lesions (output name gets a _limitN suffix).
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates

if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available; refusing to run FMCIB on CPU.")

from fmcib.models import fmcib_model
from fmcib.run import get_features

DATASETS = "/datasets"
ANNOTATIONS = f"{DATASETS}/LUNA25/annotations.csv"
IMAGES_DIR = f"{DATASETS}/LUNA25/luna25_images"
BLOCKS_DIR = f"{DATASETS}/LUNA25/luna25_nodule_blocks"
BATCH_SIZE, WORKERS = 32, 8
PATCH, HALF = 50, 25  # fmcib crops [c - 25, c + 25) on the 1 mm grid -> nodule at index 25
FEATURES = [f"pred_{i}" for i in range(4096)]


def load_block(annotation_id):
    img = np.load(f"{BLOCKS_DIR}/image/{annotation_id}.npy", mmap_mode="r")
    meta = np.load(f"{BLOCKS_DIR}/metadata/{annotation_id}.npy", allow_pickle=True).item()
    return img, np.asarray(meta["spacing"], float), np.asarray(meta["transform"], float)


def block_usable(img, spacing, transform):
    centre = np.array(img.shape) // 2
    covered = np.all(centre * spacing >= HALF) and np.all((np.array(img.shape) - 1 - centre) * spacing >= HALF - 1)
    return covered and np.allclose(transform, np.eye(3))


class BlockPatches(torch.utils.data.Dataset):
    def __init__(self, annotation_ids):
        self.ids = list(annotation_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img, spacing, _ = load_block(self.ids[i])
        centre = np.array(img.shape) // 2
        offsets = np.arange(PATCH) - HALF  # mm, 1 mm grid
        grid = np.meshgrid(*[c + offsets / s for c, s in zip(centre, spacing)], indexing="ij")
        patch = map_coordinates(img.astype(np.float32), grid, order=1, mode="nearest")
        patch = (patch + 1024) / 3072  # (z, y, x), same layout as fmcib's transposed LPS patch
        return torch.from_numpy(patch[None].astype(np.float32))


ap = argparse.ArgumentParser()
ap.add_argument("out_csv", nargs="?", default="/out/luna25_fmcib_features.csv")
ap.add_argument("--save-patch", default=None, help="save the preprocessed 50^3 patch of the first lesion (.npy)")
args = ap.parse_args()

limit = int(os.environ.get("LIMIT", "0"))
out_csv = Path(args.out_csv)
if limit > 0:
    out_csv = out_csv.with_name(f"{out_csv.stem}_limit{limit}{out_csv.suffix}")
out_csv.parent.mkdir(parents=True, exist_ok=True)

ann = pd.read_csv(ANNOTATIONS)
if limit > 0:
    ann = ann.iloc[:limit]
print(f"device: cuda ({torch.cuda.get_device_name(0)}), lesions: {len(ann)}", flush=True)

t0 = time.time()
use_block = np.array([block_usable(*load_block(a)) for a in ann.AnnotationID])
block_ids, ct_rows = ann.AnnotationID[use_block], ann[~use_block]
print(f"from blocks: {len(block_ids)}, from full CT: {len(ct_rows)}", flush=True)

model = fmcib_model(eval_mode=True).cuda()  # weights from model_weights.torch in the CWD
loader = torch.utils.data.DataLoader(BlockPatches(block_ids), batch_size=BATCH_SIZE, num_workers=WORKERS)
with torch.no_grad():
    feats = [model(x.cuda()).cpu().numpy() for x in loader]
parts = [pd.DataFrame(np.concatenate(feats) if feats else np.empty((0, 4096)), columns=FEATURES,
                      index=block_ids.values)]

if args.save_patch and len(block_ids):
    patch = BlockPatches(block_ids[:1])[0].numpy().squeeze()
    np.save(args.save_patch, patch)
    c = patch[HALF - 5:HALF + 5, HALF - 5:HALF + 5, HALF - 5:HALF + 5]
    print(f"patch check: center10^3 mean {c.mean():.4f} vs whole mean {patch.mean():.4f}", flush=True)

if len(ct_rows):
    inp = pd.DataFrame({
        "image_path": f"{IMAGES_DIR}/" + ct_rows["SeriesInstanceUID"] + ".mha",
        "coordX": ct_rows["CoordX"], "coordY": ct_rows["CoordY"], "coordZ": ct_rows["CoordZ"],
    })
    with tempfile.TemporaryDirectory(dir=out_csv.parent) as tmp:
        tmp_csv = Path(tmp) / "input.csv"
        inp.to_csv(tmp_csv, index=False)
        ct_feats = get_features(str(tmp_csv), batch_size=8, num_workers=WORKERS)
    parts.append(pd.DataFrame(ct_feats[FEATURES].values, columns=FEATURES, index=ct_rows.AnnotationID.values))
dt = time.time() - t0

out = pd.concat(parts).reindex(ann.AnnotationID).rename_axis("AnnotationID").reset_index()
assert len(out) == len(ann), (len(out), len(ann))
assert out["AnnotationID"].is_unique
assert not out.isna().any().any()
assert out.shape[1] == 4097
out.to_csv(out_csv, index=False)
print(f"elapsed {dt:.1f}s, {len(out) / dt:.2f} lesions/s; wrote {out_csv} {out.shape}", flush=True)
