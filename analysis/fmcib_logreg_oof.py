"""FMCIB features -> per-fold L2 logistic regression; out-of-fold predictions in Hiera results format.

Hyperparameters are chosen only by an inner grouped CV on each fold's training set. The test set is
never used for fitting or predicting (only for the disjointness asserts). Outer AUCs are printed last,
for information only."""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data_splits"
FEATURES_CSV = "/models/fmcib/features/luna25_fmcib_features.csv"
OUT_DIR = Path("/models/hiera-luna25/results/fmcib-logreg-cwnone-20261027")
FOLDS = range(1, 6)
SEED = 42
C_GRID = np.logspace(-5, 2, 15)
CW_GRID = [None]  # fixed: "balanced" only shifts the intercept (AUC-neutral) and makes fold scores incomparable when pooled
MAX_ITER = 5000
N_INNER = 5
N_EXPECTED = 4968

# --- load features and splits
t_start = time.time()
df_feat = pd.read_csv(FEATURES_CSV)
feat_cols = [c for c in df_feat.columns if c.startswith("pred_")]
df_feat[feat_cols] = df_feat[feat_cols].astype(np.float64)  # lbfgs works in float64
df_feat = df_feat.set_index("AnnotationID")
assert df_feat.index.is_unique and len(feat_cols) == 4096
train = {k: pd.read_csv(SPLITS_DIR / f"train_fold_{k}.csv") for k in FOLDS}
val = {k: pd.read_csv(SPLITS_DIR / f"val_fold_{k}.csv") for k in FOLDS}
test = pd.read_csv(SPLITS_DIR / "test_set.csv")
print(f"features {df_feat.shape}, loaded in {time.time() - t_start:.0f}s", flush=True)

# --- fold asserts
ann = {k: set(val[k]["AnnotationID"]) for k in FOLDS}
pat = {k: set(val[k]["PatientID"]) for k in FOLDS}
for k in FOLDS:
    assert len(ann[k]) == len(val[k]), f"duplicate AnnotationID in val_fold_{k}"
    others = [j for j in FOLDS if j != k]
    for j in others:
        if j > k:
            assert not ann[k] & ann[j], f"val folds {k},{j} share lesions"
            assert not pat[k] & pat[j], f"val folds {k},{j} share patients"
    assert set(train[k]["AnnotationID"]) == set().union(*[ann[j] for j in others]), f"train_{k} != union of other val folds"
    assert len(train[k]) == len(set(train[k]["AnnotationID"]))
    assert not pat[k] & set(train[k]["PatientID"]), f"patient shared between val_{k} and train_{k}"
    assert not ann[k] & set(test["AnnotationID"]), f"val_{k} and test share lesions"
    assert not pat[k] & set(test["PatientID"]), f"val_{k} and test share patients"
    assert set(train[k]["AnnotationID"]) | ann[k] <= set(df_feat.index), f"fold {k}: lesion without features"
assert sum(len(ann[k]) for k in FOLDS) == N_EXPECTED

OUT_DIR.mkdir(parents=True, exist_ok=True)
rows, preds = [], []
for k in FOLDS:
    t0 = time.time()
    (OUT_DIR / f"fold_{k}").mkdir(exist_ok=True)
    tr, va = train[k], val[k]
    X_tr, y_tr, g_tr = df_feat.loc[tr["AnnotationID"], feat_cols].to_numpy(), tr["label"].to_numpy(), tr["PatientID"].to_numpy()
    X_va, y_va = df_feat.loc[va["AnnotationID"], feat_cols].to_numpy(), va["label"].to_numpy()

    pipe = make_pipeline(VarianceThreshold(0.0), StandardScaler(),
                         LogisticRegression(penalty="l2", solver="lbfgs", max_iter=MAX_ITER))
    search = GridSearchCV(pipe, {"logisticregression__C": C_GRID, "logisticregression__class_weight": CW_GRID},
                          scoring="roc_auc", refit=True, n_jobs=-1,
                          cv=StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + k))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_tr, y_tr, groups=g_tr)
    # may undercount: warnings raised in parallel worker processes do not propagate here
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    best = search.best_estimator_
    n_iter = int(best[-1].n_iter_[0])  # final refit convergence, checked directly
    best_C = float(search.best_params_["logisticregression__C"])
    best_cw = search.best_params_["logisticregression__class_weight"]
    res = pd.DataFrame(search.cv_results_)
    info = {
        "fold": k, "best_C": best_C, "best_class_weight": best_cw,
        "inner_auc_mean": float(search.best_score_),
        "inner_auc_std": float(res.loc[search.best_index_, "std_test_score"]),
        "n_features_kept": int(best[0].get_support().sum()), "n_features_total": len(feat_cols),
        "n_iter": n_iter, "max_iter": MAX_ITER, "converged": bool(n_iter < MAX_ITER),
        "inner_convergence_warnings": int(n_warn),
        "C_at_grid_edge": bool(best_C in (float(C_GRID.min()), float(C_GRID.max()))),
        "seed": SEED + k, "C_grid": [float(c) for c in C_GRID], "class_weight_grid": CW_GRID,
        "n_train": int(len(tr)), "n_train_pos": int(y_tr.sum()), "n_val": int(len(va)), "n_val_pos": int(y_va.sum()),
    }

    log_odds = best.decision_function(X_va)
    prob = best.predict_proba(X_va)[:, 1]
    assert np.allclose(prob, expit(log_odds), atol=1e-9)
    df_k = pd.DataFrame({"AnnotationID": va["AnnotationID"].to_numpy(), "label": y_va,
                         "probability": prob, "log_odds": log_odds})
    df_k.to_csv(OUT_DIR / f"fold_{k}" / "predictions.csv", index=False)
    with open(OUT_DIR / f"fold_{k}" / "search.json", "w") as f:
        json.dump(info, f, indent=2)
    res.to_csv(OUT_DIR / f"fold_{k}" / "cv_results.csv", index=False)
    rows.append({kk: vv for kk, vv in info.items() if kk not in ("C_grid", "class_weight_grid")})
    preds.append(df_k.assign(fold=k))

    print(f"fold {k}: C={best_C:.3g} class_weight={best_cw} inner AUC {info['inner_auc_mean']:.4f}"
          f"±{info['inner_auc_std']:.4f}, features kept {info['n_features_kept']}/{len(feat_cols)}, "
          f"n_iter {n_iter}/{MAX_ITER} converged={info['converged']}, inner warnings {n_warn}, "
          f"edge={info['C_at_grid_edge']}, {time.time() - t0:.0f}s", flush=True)

# --- combine and check
pd.DataFrame(rows).to_csv(OUT_DIR / "hyperparameters.csv", index=False)
df_all = pd.concat(preds, ignore_index=True)[["AnnotationID", "label", "probability", "fold", "log_odds"]]
df_all.to_csv(OUT_DIR / "all_folds_predictions.csv", index=False)
assert len(df_all) == N_EXPECTED and df_all["AnnotationID"].is_unique
assert not df_all.isna().any().any()
labels = pd.concat([val[k] for k in FOLDS]).set_index("AnnotationID")["label"]
assert (df_all["label"].to_numpy() == labels.loc[df_all["AnnotationID"]].to_numpy()).all()

# information only: nothing above depends on outer AUCs
for k in FOLDS:
    d = df_all[df_all["fold"] == k]
    print(f"fold {k} outer AUC {roc_auc_score(d['label'], d['probability']):.4f}", flush=True)
print(f"pooled outer AUC {roc_auc_score(df_all['label'], df_all['probability']):.4f}", flush=True)
print(f"wrote {OUT_DIR} in {time.time() - t_start:.0f}s", flush=True)
