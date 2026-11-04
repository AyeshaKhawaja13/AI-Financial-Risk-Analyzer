"""Per-prediction explanations: SHAP for tree models (RF/XGBoost), occlusion for the rest."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .features import ENGINEERED


def contributions(bundle: dict, X):
    """Returns (matrix n x features, method). Positive = pushes risk up (model-score space)."""
    pipe = bundle["pipeline"]
    pre, clf = pipe[:-1], pipe[-1]
    Z = pre.transform(X)
    if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
        try:
            import shap
            sv = shap.TreeExplainer(clf).shap_values(Z)
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            return sv, "shap"
        except Exception:
            pass
    base = np.asarray(bundle["background"])
    p = clf.predict_proba(Z)[:, 1]
    out = np.zeros_like(Z, dtype=float)
    for j in range(Z.shape[1]):
        Zj = Z.copy()
        Zj[:, j] = base[j]
        out[:, j] = p - clf.predict_proba(Zj)[:, 1]
    return out, "occlusion"


def global_importance(bundle: dict, X, max_rows: int = 2000):
    X = X.sample(min(len(X), max_rows), random_state=42)
    c, method = contributions(bundle, X)
    imp = np.abs(c).mean(axis=0)
    return dict(sorted(zip(ENGINEERED, imp.tolist()), key=lambda kv: -kv[1])), method
