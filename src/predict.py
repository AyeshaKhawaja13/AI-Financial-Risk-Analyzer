import joblib
import numpy as np
import pandas as pd
from .config import BASE, MODEL_PATH
from .explain import contributions


def load_bundle(path=MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m src.train")
    return joblib.load(path)


def risk_level(p: float, thr: float) -> str:
    return "High" if p >= thr else "Medium" if p >= thr / 2 else "Low"


def predict(bundle: dict, records: list, top_k: int = 5) -> list:
    df = pd.DataFrame(records).reindex(columns=BASE)
    raw = bundle["pipeline"].predict_proba(df)[:, 1]
    probs = bundle["calibrator"].predict(raw)
    contrib, method = contributions(bundle, df)
    eng = bundle["pipeline"]["fe"].transform(df)
    names = list(eng.columns)
    results = []
    for i, p in enumerate(probs):
        order = np.argsort(-np.abs(contrib[i]))[:top_k]
        factors = []
        for j in order:
            v = eng.iloc[i, j]
            factors.append({"feature": names[j], "value": None if pd.isna(v) else float(v),
                            "impact": float(contrib[i, j]),
                            "direction": "increases risk" if contrib[i, j] > 0 else "decreases risk"})
        results.append({"risk_probability": float(p), "risk_level": risk_level(p, bundle["threshold"]),
                        "flagged_high_risk": bool(p >= bundle["threshold"]),
                        "decision_threshold": float(bundle["threshold"]),
                        "explanation_method": method, "top_factors": factors})
    return results
