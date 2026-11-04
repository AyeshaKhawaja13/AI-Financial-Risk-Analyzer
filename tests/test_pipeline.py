import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from api.main import EXAMPLE, app
from src.config import BASE, SENSITIVE, TARGET
from src.data import clean, generate_synthetic
from src.features import ENGINEERED, FeatureEngineer, Winsorizer, build_pipeline


def test_clean_removes_duplicates_and_maps_undocumented_codes():
    raw = generate_synthetic(1000)
    df = clean(raw)
    assert len(df) < len(raw)
    assert set(df["education"].unique()) <= {1, 2, 3, 4}
    assert 0 not in set(df["marriage"].unique())


def test_sensitive_attributes_not_model_inputs():
    assert not set(SENSITIVE) & set(BASE)


def test_feature_engineering_shape_and_no_inf():
    X = clean(generate_synthetic(500))[BASE].copy()
    X.loc[X.index[0], "limit_bal"] = 0
    out = FeatureEngineer().transform(X)
    assert list(out.columns) == ENGINEERED and not np.isinf(out.values).any()


def test_winsorizer_uses_train_bounds_only():
    w = Winsorizer().fit(np.arange(100.0).reshape(-1, 1))
    assert w.transform(np.array([[1e9]]))[0, 0] <= 100


def test_pipeline_outputs_probabilities():
    df = clean(generate_synthetic(3000))
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True).fit(df[BASE], df[TARGET])
    p = pipe.predict_proba(df[BASE].head(20))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_api_validation():
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.post("/predict", json={**EXAMPLE, "age": 5}).status_code == 422
        assert c.post("/predict", json={**EXAMPLE, "pay_1": 50}).status_code == 422
