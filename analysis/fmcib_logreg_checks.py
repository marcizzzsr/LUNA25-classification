"""Diagnostics for a fmcib_logreg_oof.py run: inner-CV stability and per-fold log-odds scales.

Pooled thresholds (95% spec / 95% sens) need comparable scores across folds: if the pooled AUC is
clearly below the per-fold mean, the fold models produce log-odds on different scales (this is how
class_weight="balanced" in one fold showed up, see fmcib-logreg-20261027 vs -cwnone-).

Run: cd jobs && sbatch run_analysis.sh analysis/fmcib_logreg_checks.py [run_dir]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "/models/hiera-luna25/results/fmcib-logreg-cwnone-20261027"
FOLDS = range(1, 6)
N_INNER = 5

pd.set_option("display.width", 200)
print("== inner CV: mean AUC per (C, class_weight)")
tabs = []
for k in FOLDS:
    r = pd.read_csv(f"{RUN_DIR}/fold_{k}/cv_results.csv")
    r["cw"] = r["param_logisticregression__class_weight"].fillna("none")
    best = r.loc[r["rank_test_score"] == 1].iloc[0]
    by_cw = r.groupby("cw")["mean_test_score"].max().round(4).to_dict()
    print(f"fold {k}: best {best.mean_test_score:.4f} (se ~{best.std_test_score / np.sqrt(N_INNER):.4f}); "
          f"configs within 0.002 of best: {int((r.mean_test_score >= best.mean_test_score - 0.002).sum())}; "
          f"best per class_weight {by_cw}")
    tabs.append(r.pivot(index="param_logisticregression__C", columns="cw", values="mean_test_score").add_prefix(f"f{k}_"))
t = pd.concat(tabs, axis=1)
print(t.loc[(t.index > 1e-5) & (t.index < 1e-1)].round(4).to_string())

print("\n== OOF AUC")
d = pd.read_csv(f"{RUN_DIR}/all_folds_predictions.csv")
per_fold = [roc_auc_score(g.label, g.log_odds) for _, g in d.groupby("fold")]
print(f"per fold {np.round(per_fold, 4)} mean {np.mean(per_fold):.4f} | pooled {roc_auc_score(d.label, d.log_odds):.4f}")

print("\n== log-odds per fold and class")
q = d.groupby(["fold", "label"]).log_odds.describe(percentiles=[.05, .5, .95])[["count", "mean", "std", "5%", "50%", "95%"]]
print(q.round(2).to_string())
thr_spec = d[d.label == 0].log_odds.quantile(.95)
thr_sens = d[d.label == 1].log_odds.quantile(.05)
print(f"\npooled thresholds: 95% spec (benign q95) = {thr_spec:.2f}, 95% sens (malignant q05) = {thr_sens:.2f}")
print("share of each fold's benigns above the pooled 95%-spec threshold:",
      d[d.label == 0].groupby("fold").log_odds.apply(lambda s: round((s > thr_spec).mean(), 3)).to_dict())
# within-fold standardised log-odds, as a diagnostic of the scale effect only
d["z"] = d.groupby("fold").log_odds.transform(lambda s: (s - s.mean()) / s.std())
print(f"pooled AUC after per-fold z-scoring (diagnostic only): {roc_auc_score(d.label, d.z):.4f}")
