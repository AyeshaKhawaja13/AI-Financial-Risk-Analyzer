"""Train, compare, tune, calibrate, evaluate, explain, save.  python -m src.train [--demo]"""
import argparse, json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, brier_score_loss, f1_score, make_scorer,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from xgboost import XGBClassifier
from .config import BASE, FIG_DIR, METRICS_PATH, MODEL_DIR, MODEL_PATH, REPORTS, SEED, TARGET
from .data import clean, load_raw
from .explain import global_importance
from .features import ENGINEERED, build_pipeline

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": "recall", "f1": "f1"}


def model_zoo(spw: float) -> dict:
    return {
        "dummy": build_pipeline(DummyClassifier(strategy="prior")),
        "logreg": build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED), scale=True),
        "random_forest": build_pipeline(RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        "hist_gb": build_pipeline(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)),
        "xgboost": build_pipeline(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)),
    }


GRIDS = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "random_forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, 14, None],
                      "clf__min_samples_leaf": [3, 5, 10, 20], "clf__max_features": ["sqrt", 0.5]},
    "hist_gb": {"clf__learning_rate": [0.03, 0.06, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
                "clf__l2_regularization": [0, 1, 10], "clf__min_samples_leaf": [20, 50, 100]},
    "xgboost": {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [3, 4, 5, 6],
                "clf__learning_rate": [0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 5, 10]},
}


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "brier": brier_score_loss(y, p), "threshold": float(thr)}


def fairness_audit(df_te, y, p, thr):
    """Subgroup metrics on the TEST set (sex is NOT a model input; education and age are)."""
    d = pd.DataFrame({"sex": df_te["sex"].values, "education": df_te["education"].values,
                      "age_band": pd.cut(df_te["age"], [17, 29, 39, 49, 59, 120],
                                         labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str).values,
                      "y": np.asarray(y), "p": p, "pred": (p >= thr).astype(int)})
    rows = []
    for g in ["sex", "education", "age_band"]:
        for k, s in d.groupby(g):
            tp = int(((s.pred == 1) & (s.y == 1)).sum())
            rows.append({"attribute": g, "group": str(k), "n": len(s), "actual_default_rate": float(s.y.mean()),
                         "flagged_rate": float(s.pred.mean()), "recall": tp / max(int((s.y == 1).sum()), 1),
                         "precision": tp / max(int((s.pred == 1).sum()), 1), "mean_predicted_prob": float(s.p.mean())})
    return pd.DataFrame(rows)


def main(demo=False, n_iter=15, tune=True):
    MODEL_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_raw(demo))
    X, y = df[BASE], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train {Xtr.shape}, test {Xte.shape}, positive rate train={ytr.mean():.4f} test={yte.mean():.4f}")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    zoo = model_zoo((ytr == 0).sum() / (ytr == 1).sum())

    rows = []
    for name, pipe in zoo.items():
        r = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=SCORING)
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in SCORING},
                     "pr_auc_std": float(np.std(r["test_pr_auc"]))})
        print(f"CV {name:14s} " + " ".join(f"{k}={rows[-1][k]:.4f}" for k in SCORING))
    cvdf = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    REPORTS.mkdir(exist_ok=True); cvdf.to_csv(REPORTS / "cv_results.csv", index=False)

    best_name = cvdf[cvdf.model != "dummy"].iloc[0]["model"]
    best = zoo[best_name]
    if tune and best_name in GRIDS:
        print(f"Tuning {best_name} ({n_iter} iterations, scoring=PR-AUC)...")
        s = RandomizedSearchCV(best, GRIDS[best_name], n_iter=n_iter, scoring="average_precision",
                               cv=cv, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        best, best_params = s.best_estimator_, s.best_params_
        print("Best params:", best_params, "CV PR-AUC:", round(s.best_score_, 4))
    else:
        best.fit(Xtr, ytr); best_params = {}

    # Calibration + threshold chosen ONLY from out-of-fold training predictions (no test leakage)
    oof = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    calib = IsotonicRegression(out_of_bounds="clip").fit(oof, ytr)
    oof_c = calib.predict(oof)
    pr, rc, th = precision_recall_curve(ytr, oof_c)
    f1s = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    thr = float(th[np.argmax(f1s)])

    raw_te = best.predict_proba(Xte)[:, 1]
    p_te = calib.predict(raw_te)
    test = metrics_at(yte, p_te, thr)
    test["brier_uncalibrated"] = float(brier_score_loss(yte, raw_te))
    test["confusion_matrix"] = pd.crosstab(yte, (p_te >= thr).astype(int)).values.tolist()
    print("TEST:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test.items()})

    RocCurveDisplay.from_predictions(yte, p_te); plt.savefig(FIG_DIR / "roc_curve.png", dpi=120); plt.close()
    PrecisionRecallDisplay.from_predictions(yte, p_te); plt.savefig(FIG_DIR / "pr_curve.png", dpi=120); plt.close()
    ConfusionMatrixDisplay.from_predictions(yte, (p_te >= thr).astype(int)); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=120); plt.close()

    bundle = {"pipeline": best, "calibrator": calib, "threshold": thr, "features": BASE,
              "engineered_features": ENGINEERED, "model_name": best_name,
              "background": np.median(best[:-1].transform(Xtr), axis=0),
              "data_source": "SYNTHETIC DEMO DATA - not real" if demo else "UCI Default of Credit Card Clients (Taiwan)"}
    fair = fairness_audit(df.loc[Xte.index], yte, p_te, thr)
    fair.to_csv(REPORTS / "fairness_audit.csv", index=False)
    gaps = {a: float(g.recall.max() - g.recall.min()) for a, g in fair[fair.n >= 30].groupby("attribute")}
    print("Fairness audit (recall gap across groups):", {k: round(v, 3) for k, v in gaps.items()})
    imp, method = global_importance(bundle, Xte)
    pd.Series(imp).head(15)[::-1].plot.barh(title=f"Mean |contribution| ({method})", figsize=(8, 6))
    plt.tight_layout(); plt.savefig(FIG_DIR / "feature_importance.png", dpi=120); plt.close()

    bundle["global_importance"] = imp
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"data_source": bundle["data_source"], "best_model": best_name,
                                        "best_params": {k: str(v) for k, v in best_params.items()},
                                        "seed": SEED, "cv_results": cvdf.to_dict("records"),
                                        "test_metrics": test, "fairness_recall_gaps": gaps, "global_importance": imp}, indent=2))
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, for testing the code only")
    ap.add_argument("--n-iter", type=int, default=15)
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    main(a.demo, a.n_iter, not a.no_tune)
