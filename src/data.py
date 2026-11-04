"""Loading, cleaning and a synthetic generator (synthetic is for testing code only, NOT real)."""
import numpy as np
import pandas as pd
from .config import BASE, BILL, PAID, PAY, RAW_CSV, RENAME, SEED, SENSITIVE, TARGET, TARGET_ALIASES


def generate_synthetic(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    z = r.normal(size=n)  # latent risk
    limit = np.round(r.lognormal(11.5, 0.8, n), -4).clip(10000, 800000)
    d = {"limit_bal": limit, "sex": r.choice([1, 2], n),
         "education": r.choice([1, 2, 3, 4, 0, 5, 6], n, p=[.35, .45, .15, .02, .01, .01, .01]),
         "marriage": r.choice([1, 2, 3, 0], n, p=[.45, .5, .04, .01]), "age": r.integers(21, 70, n)}
    for i, c in enumerate(PAY):
        d[c] = np.clip(np.round(z * (1.1 - 0.1 * i) + r.normal(0, 1, n) - 0.3), -2, 8).astype(int)
    for i, c in enumerate(BILL):
        d[c] = np.round(limit * np.clip(r.beta(2, 3, n) + 0.08 * z, 0, 1.3))
    for i, c in enumerate(PAID):
        d[c] = np.round(d[BILL[i]].clip(0) * r.uniform(0, 0.5, n))
    d[TARGET] = (r.random(n) < 1 / (1 + np.exp(-(-1.55 + 0.9 * z)))).astype(int)
    df = pd.DataFrame(d)
    return pd.concat([df, df.sample(30, random_state=seed)], ignore_index=True)


def load_raw(demo: bool = False) -> pd.DataFrame:
    if demo:
        return generate_synthetic()
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_CSV}. Put credit_default.csv in data/raw/ "
                                "(UCI id=350), or use --demo for synthetic data (testing only).")
    df = pd.read_csv(RAW_CSV)
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("id", "unnamed: 0")])
    df = df.rename(columns=RENAME)
    for a in TARGET_ALIASES:
        if a in df.columns and TARGET not in df.columns:
            df = df.rename(columns={a: TARGET})
    missing = set(BASE + SENSITIVE + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level rules only (nothing learned from data), safe before the split."""
    df = df.drop_duplicates()
    df = df[(df["age"] >= 18) & (df["age"] <= 110)].copy()
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})  # undocumented codes -> 'other'
    df["marriage"] = df["marriage"].replace({0: 3})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)
