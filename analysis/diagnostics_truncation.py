"""Truncation diagnostics: how much of each MedSAM lesion mask falls outside the model's fixed crop box.

Crop box = annotation centre +/- 6.25 mm in z (DEPTH_PX * 0.78 / 2) and +/- 25 mm in y/x (SIZE_MM / 2).
Set LIMIT=<n> to process only the first n annotations (smoke test).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATA_DIR = "/datasets/slow/LUNA25"
WORKSPACE = "/models/hiera-luna25"
PRED_CSV = f"{WORKSPACE}/results/finetune-hiera-3D-20260814/all_folds_predictions.csv"
STATS_CSV = f"{DATA_DIR}/additional-annotations/csv-annotations/lesion_stats.csv"
MASK_DIR = Path(f"{DATA_DIR}/LUNA25-MedSAM2/instance-labels")
BBOX_JSON = f"{DATA_DIR}/additional-annotations/MedSAMv2_bbox/luna25_bbox_center_size_v2.json"
OUT_CSV = Path(f"{WORKSPACE}/figures/diagnostics/truncation_metrics.csv")
HALF_Z, HALF_XY = 6.25, 25.0   # mm
BBOX_MAX_DIST_MM = 10.0        # max distance between annotation centre and bbox centre to call it a match
LIMIT = int(os.environ.get("LIMIT", 0))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pred = pd.read_csv(PRED_CSV)
pred["absolute_error"] = (pred["probability"] - pred["label"]).abs()
stats = pd.read_csv(STATS_CSV)
df = pred[["AnnotationID", "fold", "label", "absolute_error"]].merge(
    stats[["AnnotationID", "SeriesInstanceUID", "CoordX", "CoordY", "CoordZ", "matched", "matched_instance_id",
           "nodule_volume_mm3", "query_voxel_x", "query_voxel_y", "query_voxel_z"]], on="AnnotationID", how="left")
print(f"{len(pred)} predictions, {df['SeriesInstanceUID'].notna().sum()} found in lesion_stats", flush=True)
df = df[df["SeriesInstanceUID"].notna() & df["matched"].fillna(False).astype(bool)]
if LIMIT:
    df = df.head(LIMIT)

bboxes = {}  # series uid -> (n, 6) array of [cx, cy, cz, sx, sy, sz]
for e in json.load(open(BBOX_JSON)):
    bboxes[e["image"].rsplit(".", 1)[0]] = np.array(e["box"], dtype=float).reshape(-1, 6)


def world_to_index(img, x, y, z):
    return np.array(img.TransformPhysicalPointToContinuousIndex((x, y, z)))  # (ix, iy, iz)


rows, missing, val = [], [], []
t0 = time.time()
for n, (uid, g) in enumerate(df.groupby("SeriesInstanceUID", sort=False)):
    path = MASK_DIR / f"{uid}.nii.gz"
    if not path.exists():
        missing += list(g["AnnotationID"])
        continue
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayViewFromImage(img)  # (z, y, x)
    sp_x, sp_y, sp_z = img.GetSpacing()
    voxel_mm3 = sp_x * sp_y * sp_z
    for r in g.itertuples():
        inst = int(r.matched_instance_id)
        iz, iy, ix = np.nonzero(arr == inst)
        # voxel-coordinate convention check: CoordX/Y/Z -> continuous index vs query_voxel_*
        cidx = world_to_index(img, r.CoordX, r.CoordY, r.CoordZ)
        qv = np.array([r.query_voxel_x, r.query_voxel_y, r.query_voxel_z])
        centre_val = arr[tuple(np.clip(np.round(cidx[::-1]).astype(int), 0, np.array(arr.shape) - 1))]
        val.append((*(cidx - qv), centre_val == inst))
        if len(iz) == 0:
            missing.append(r.AnnotationID)
            continue
        # world position of every mask voxel, relative to the annotation centre (axis-aligned scans)
        # vectorised: origin + D @ (spacing * idx)
        D = np.array(img.GetDirection()).reshape(3, 3)
        idx = np.stack([ix, iy, iz], 1) * np.array([sp_x, sp_y, sp_z])
        pw = np.array(img.GetOrigin()) + idx @ D.T
        d = pw - np.array([r.CoordX, r.CoordY, r.CoordZ])  # (x, y, z) offsets in mm
        ext = np.ptp(d, axis=0) + np.abs(D @ np.array([sp_x, sp_y, sp_z]))  # voxel-edge extent (x, y, z)
        out_z = np.abs(d[:, 2]) > HALF_Z
        out = out_z | (np.abs(d[:, 0]) > HALF_XY) | (np.abs(d[:, 1]) > HALF_XY)
        half = np.abs(D @ np.array([sp_x, sp_y, sp_z])) / 2
        touches = bool(((np.abs(d).max(0) + half) >= np.array([HALF_XY, HALF_XY, HALF_Z])).any())
        cen = d.mean(0)
        # bounding box JSON centres are in the x/y-flipped frame (like mask_centroid_* in lesion_stats)
        b = bboxes.get(uid)
        bsz = bsxy = np.nan
        if b is not None:
            dist = np.linalg.norm(b[:, :3] - [-r.CoordX, -r.CoordY, r.CoordZ], axis=1)
            if dist.min() <= BBOX_MAX_DIST_MM:
                k = b[dist.argmin()]
                bsz, bsxy = k[5], max(k[3], k[4])
        rows.append(dict(
            AnnotationID=r.AnnotationID, fold=r.fold, label=r.label, absolute_error=r.absolute_error,
            nodule_volume_mm3=r.nodule_volume_mm3, z_extent_mm=ext[2], y_extent_mm=ext[1], x_extent_mm=ext[0],
            xy_extent_mm=max(ext[0], ext[1]), frac_outside=out.mean(), frac_outside_z=out_z.mean(),
            touches_border=touches, trunc_z=bool(ext[2] > 2 * HALF_Z or out.any()),
            centroid_offset_z_mm=abs(cen[2]), centroid_offset_xy_mm=float(np.hypot(cen[0], cen[1])),
            bbox_sz_mm=bsz, bbox_sxy_mm=bsxy, mask_volume_mm3=len(iz) * voxel_mm3))
    if n % 100 == 0:
        print(f"series {n}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}: {len(out)} rows, {len(missing)} without usable mask, {time.time() - t0:.0f}s", flush=True)

# ---- validation ----
v = np.array(val, dtype=float)
print(f"voxel convention: continuous index - query_voxel (x,y,z) abs max = {np.abs(v[:, :3]).max(0)}, "
      f"median = {np.median(np.abs(v[:, :3]), 0)}; mask==instance at annotation centre: {v[:, 3].mean():.1%}")
m = out.dropna(subset=["nodule_volume_mm3"])
rel = (m["mask_volume_mm3"] / m["nodule_volume_mm3"] - 1).abs()
print(f"mask volume vs nodule_volume_mm3: median rel diff {rel.median():.2e}, max {rel.max():.2e}, "
      f"within 1%: {(rel < 0.01).mean():.1%}")
bb = out.dropna(subset=["bbox_sz_mm"])
print(f"bbox matched: {len(bb)}/{len(out)}; corr(z_extent, bbox_sz) = {bb['z_extent_mm'].corr(bb['bbox_sz_mm']):.3f}, "
      f"corr(xy_extent, bbox_sxy) = {bb['xy_extent_mm'].corr(bb['bbox_sxy_mm']):.3f}")
print(f"trunc_z {out['trunc_z'].mean():.1%}, touches_border {out['touches_border'].mean():.1%}, "
      f"z_extent>12.5 {(out['z_extent_mm'] > 12.5).mean():.1%}")
