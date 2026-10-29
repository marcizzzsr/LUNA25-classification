"""Coverage of the nodule blocks for FMCIB's 50 mm patch: shape, spacing and mm extent per axis.

FMCIB needs +-25 mm of real data around the nodule; blocks are 64x128x128 at native spacing with the
nodule at the centre voxel, so thin-slice series fall short in z. extract_fmcib_features.py sends
those lesions to the full-CT pipeline instead.

Run: cd jobs && sbatch run_analysis.sh analysis/fmcib_block_coverage.py
"""
import numpy as np
import pandas as pd

BLOCKS_DIR = "/datasets/slow/LUNA25/luna25_nodule_blocks"
ANNOTATIONS = "/datasets/slow/LUNA25/LUNA25_Public_Training_Development_Data.csv"
HALF_MM = 25

ann = pd.read_csv(ANNOTATIONS)
rows = []
for a in ann["AnnotationID"]:
    img = np.load(f"{BLOCKS_DIR}/image/{a}.npy", mmap_mode="r")
    meta = np.load(f"{BLOCKS_DIR}/metadata/{a}.npy", allow_pickle=True).item()
    sp = np.asarray(meta["spacing"], float)  # (z, y, x)
    rows.append({"AnnotationID": a, "shape": str(img.shape), "dtype": str(img.dtype),
                 "identity_direction": bool(np.allclose(meta["transform"], np.eye(3))),
                 **{f"s{ax}": s for ax, s in zip("zyx", sp)},
                 **{f"e{ax}": n * s for ax, n, s in zip("zyx", img.shape, sp)}})
d = pd.DataFrame(rows)

print(f"lesions {len(d)}; shapes {d['shape'].value_counts().to_dict()}; dtypes {d['dtype'].value_counts().to_dict()}")
print(f"non-identity direction: {int((~d['identity_direction']).sum())}")
print(d[["sz", "sy", "sx", "ez", "ey", "ex"]].describe(percentiles=[.01, .05, .5]).round(2).to_string())
for ax in "zyx":
    print(f"{ax}: extent < {2 * HALF_MM} mm in {int((d[f'e{ax}'] < 2 * HALF_MM).sum())} blocks")
