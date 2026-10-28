"""Check that FMCIB features from the nodule blocks match fmcib's own full-CT pipeline.

Compares, lesion by lesion, the block features of the first 20 lesions with the reference features
fmcib's get_features computed on the full CTs (fullct_limit20_reference.csv, produced once with the
original full-CT version of extract_fmcib_features.py). Result on 2026-10-27: median cosine 0.996,
min 0.987, own nearest neighbour 20/20; the residual comes from fmcib truncating the centre voxel.

Run: cd jobs
     LIMIT=20 sbatch run_fmcib.sh /out/luna25_fmcib_features.csv --save-patch /out/patch_first_block.npy
     sbatch run_analysis.sh analysis/fmcib_block_vs_fullct.py
"""
import numpy as np
import pandas as pd

FEATURES_DIR = "/models/fmcib/features"
REF_CSV = f"{FEATURES_DIR}/fullct_limit20_reference.csv"
NEW_CSV = f"{FEATURES_DIR}/luna25_fmcib_features_limit20.csv"

ref = pd.read_csv(REF_CSV).set_index("AnnotationID")
new = pd.read_csv(NEW_CSV).set_index("AnnotationID").loc[ref.index]
a, b = ref.values, new.values
na = a / np.linalg.norm(a, axis=1, keepdims=True)
nb = b / np.linalg.norm(b, axis=1, keepdims=True)
cos = (na * nb).sum(1)
pearson = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]
# how much closer is each lesion to its own reference than to the other lesions' references
sim = nb @ na.T
off = sim[~np.eye(len(sim), dtype=bool)]

print(f"cosine own   : min {cos.min():.4f} median {np.median(cos):.4f}")
print(f"pearson own  : min {min(pearson):.4f} median {np.median(pearson):.4f}")
print(f"cosine others: median {np.median(off):.4f} max {off.max():.4f}")
print(f"own is nearest neighbour for {(sim.argmax(1) == np.arange(len(sim))).sum()}/{len(sim)} lesions")
print(f"rel. L2 diff : median {np.median(np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)):.4f}")
