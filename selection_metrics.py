# Validation metrics used for model selection.

import numpy as np


def size_matched_auc(y, p, d, sigma=0.15):
    """
    Mann-Whitney AUC over malignant-benign pairs, each pair weighted by
    exp(-(ln d_m - ln d_b)^2 / 2 sigma^2), so only pairs of similar diameter count.
    Lesions with missing diameter (NaN) are ignored.
    """
    y, p, d = np.asarray(y), np.asarray(p), np.asarray(d, dtype=float)
    keep = np.isfinite(d) & (d > 0)
    y, p, log_d = y[keep], p[keep], np.log(d[keep])
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")

    w = np.exp(-(log_d[pos][:, None] - log_d[neg][None, :]) ** 2 / (2 * sigma ** 2))
    diff = p[pos][:, None] - p[neg][None, :]
    wins = (diff > 0) + 0.5 * (diff == 0)
    return float((w * wins).sum() / w.sum())


def spec_at_sens(y, p, sens=0.95):
    """
    Specificity at the threshold that keeps `sens` of the malignant lesions above it.
    """
    y, p = np.asarray(y), np.asarray(p)
    threshold = np.quantile(p[y == 1], 1 - sens)
    return float((p[y == 0] < threshold).mean())
