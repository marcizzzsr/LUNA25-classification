"""Duke ResNet50-SWS++ (DLCS-trained) inference on the LUNA25 lesions of the Hiera validation folds (GPU required).

Reproduces the authors' pipeline (arXiv 2405.04605 v5, repo README, PiNS candidate_worldCoord_patchExtarctor_pipeline.py,
ct_classification/test.py): BSpline resampling to 0.703125 x 0.703125 x 1.25 mm, HU clip [-1000, 500] -> [0, 1],
64^3 voxel patch around the world coordinate (zero padding), saved as NIfTI with sitk and read back through their test
transform; score = softmax(logits)[:, 1]. Existing patches are not re-extracted.

Usage: sws_inference.py   Env: LIMIT=N processes only the first N lesions (outputs get a _limitN suffix).
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Resized, ScaleIntensityRanged
from sklearn.metrics import roc_auc_score

if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available; refusing to run on CPU.")

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data_splits"
IMAGES_DIR = "/datasets/LUNA25/luna25_images"
PATCH_DIR = Path("/patches")
WEIGHTS = "/dlcsd/DukeLungRADS_CancerCL_Fold1_resnet50WSPP.pt"
OUT_DIR = Path("/dlcsd/results/sws-pp-20261028")
SPACING = [0.703125, 0.703125, 1.25]
PATCH = 64
LIMIT = int(os.environ.get("LIMIT", "0"))
SUFFIX = f"_limit{LIMIT}" if LIMIT else ""


def resample_img(itk_image, out_spacing):
    # copy of the README's resample_img (image branch)
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_size = [int(np.round(original_size[i] * (original_spacing[i] / out_spacing[i]))) for i in range(3)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)


def extract_ct(args):
    series_uid, rows = args
    todo = [r for r in rows if not (PATCH_DIR / "nifti" / f"{r['AnnotationID']}.nii.gz").exists()]
    if not todo:
        return series_uid, None
    try:
        ct_image = resample_img(sitk.ReadImage(f"{IMAGES_DIR}/{series_uid}.mha"), SPACING)
        scale = ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, b_min=0, b_max=1, clip=True)
        numpyImage = scale({"image": torch.from_numpy(sitk.GetArrayFromImage(ct_image))})["image"].numpy()
        w = h = d = PATCH
        for r in todo:
            # PiNS crop + padding, verbatim
            voxelCoord = ct_image.TransformPhysicalPointToIndex([float(r["CoordX"]), float(r["CoordY"]), float(r["CoordZ"])])
            start_x, end_x = int(voxelCoord[0] - w / 2), int(voxelCoord[0] + w / 2)
            start_y, end_y = int(voxelCoord[1] - h / 2), int(voxelCoord[1] + h / 2)
            start_z, end_z = int(voxelCoord[2] - d / 2), int(voxelCoord[2] + d / 2)
            X, Y, Z = int(voxelCoord[0]), int(voxelCoord[1]), int(voxelCoord[2])
            numpy_to_save_np = numpyImage[max(start_z, 0):end_z, max(start_y, 0):end_y, max(start_x, 0):end_x]
            if np.any(numpy_to_save_np.shape != (d, h, w)):
                dZ, dY, dX = numpyImage.shape
                numpy_to_save_np = np.pad(numpy_to_save_np, ((max(d // 2 - Z, 0), d // 2 - min(dZ - Z, d // 2)),
                                                             (max(h // 2 - Y, 0), h // 2 - min(dY - Y, h // 2)),
                                                             (max(w // 2 - X, 0), w // 2 - min(dX - X, w // 2))),
                                          mode="constant", constant_values=0.)
            patch_image = sitk.GetImageFromArray(numpy_to_save_np)
            patch_image.SetSpacing(ct_image.GetSpacing())
            patch_image.SetDirection(ct_image.GetDirection())
            patch_image.SetOrigin(ct_image.GetOrigin())
            sitk.WriteImage(patch_image, str(PATCH_DIR / "nifti" / f"{r['AnnotationID']}.nii.gz"))
        return series_uid, None
    except Exception as e:
        return series_uid, repr(e)


# --- lesions of the validation folds
df = pd.concat([pd.read_csv(SPLITS_DIR / f"val_fold_{k}.csv").assign(fold=k) for k in range(1, 6)], ignore_index=True)
assert df["AnnotationID"].is_unique
if LIMIT:
    df = df.head(LIMIT)
cols = ["AnnotationID", "SeriesInstanceUID", "label", "fold", "CoordX", "CoordY", "CoordZ"]
print(f"{len(df)} lesions, {df['SeriesInstanceUID'].nunique()} CTs", flush=True)

# --- (a) patch extraction
(PATCH_DIR / "nifti").mkdir(parents=True, exist_ok=True)
jobs = [(uid, g.to_dict("records")) for uid, g in df.groupby("SeriesInstanceUID")]
errors = []
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))) as pool:
    for i, (uid, err) in enumerate(pool.imap_unordered(extract_ct, jobs), 1):
        if err:
            errors.append({"SeriesInstanceUID": uid, "error": err})
            print(f"ERROR {uid}: {err}", flush=True)
        if i % 200 == 0:
            print(f"extracted {i}/{len(jobs)} CTs", flush=True)
df[cols].to_csv(PATCH_DIR / f"patches{SUFFIX}.csv", index=False)
pd.DataFrame(errors, columns=["SeriesInstanceUID", "error"]).to_csv(PATCH_DIR / f"patches_errors{SUFFIX}.csv", index=False)
df = df[~df["SeriesInstanceUID"].isin([e["SeriesInstanceUID"] for e in errors])]

# --- (b) inference, with the authors' test transform
test_transforms = Compose([
    LoadImaged("img"),
    EnsureChannelFirstd("img", channel_dim="no_channel"),
    Resized("img", spatial_size=[PATCH] * 3, mode="trilinear", align_corners=True),
    EnsureTyped("img"),
])
files = [{"img": str(PATCH_DIR / "nifti" / f"{a}.nii.gz")} for a in df["AnnotationID"]]
loader = monai.data.DataLoader(monai.data.Dataset(files, transform=test_transforms), batch_size=32, num_workers=8)

device = torch.device("cuda")
model = monai.networks.nets.resnet50(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2)
state = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
model.to(device).eval()

probs = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for batch in loader:
        probs.append(torch.softmax(model(batch["img"].to(device)).float(), dim=1)[:, 1].cpu().numpy())
df["probability"] = np.concatenate(probs)

# --- outputs (Hiera results format) and pooled AUC
out = df[["AnnotationID", "label", "probability", "fold"]]
assert out["probability"].notna().all()
for k, g in out.groupby("fold"):
    (OUT_DIR / f"fold_{k}").mkdir(parents=True, exist_ok=True)
    g.drop(columns="fold").to_csv(OUT_DIR / f"fold_{k}" / f"predictions{SUFFIX}.csv", index=False)
out.to_csv(OUT_DIR / f"all_folds_predictions{SUFFIX}.csv", index=False)
print(f"failed CTs: {len(errors)}; n = {len(out)}, positives = {int(out['label'].sum())}")
print(f"probability min/median/max: {out['probability'].min():.4f} {out['probability'].median():.4f} {out['probability'].max():.4f}")
if out["label"].nunique() == 2:
    print(f"pooled AUC: {roc_auc_score(out['label'], out['probability']):.4f}")
