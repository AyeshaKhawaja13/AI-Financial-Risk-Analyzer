"""EDA + statistical tests. Run: python -m src.eda [--demo]"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from .config import BASE, FIG_DIR, REPORTS, SENSITIVE, TARGET
from .data import clean, load_raw


def run(demo: bool = False) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True)
    raw = load_raw(demo)
    num = BASE + SENSITIVE
    out = {"source": "SYNTHETIC DEMO" if demo else "UCI Default of Credit Card Clients",
           "rows": len(raw), "duplicates": int(raw.duplicated().sum()),
           "missing_pct": (raw.isna().mean() * 100).round(2).to_dict(),
           "class_balance": raw[TARGET].value_counts(normalize=True).round(4).to_dict(),
           "undocumented_education_codes": int(raw["education"].isin([0, 5, 6]).sum()),
           "undocumented_marriage_code_0": int((raw["marriage"] == 0).sum())}
    q1, q3 = raw[BASE].quantile(0.25), raw[BASE].quantile(0.75)
    iqr = q3 - q1
    out["iqr_outliers"] = ((raw[BASE] < q1 - 1.5 * iqr) | (raw[BASE] > q3 + 1.5 * iqr)).sum().to_dict()

    df = clean(raw)
    tests = {}
    for c in BASE:
        a, b = df.loc[df[TARGET] == 1, c], df.loc[df[TARGET] == 0, c]
        u, p = stats.mannwhitneyu(a, b)
        tests[c] = {"median_default": float(a.median()), "median_non_default": float(b.median()),
                    "mannwhitney_p": float(p), "rank_biserial": float(1 - 2 * u / (len(a) * len(b)))}
    out["mannwhitney_by_feature"] = tests
    out["chi2_categorical_vs_target"] = {}
    for c in ["sex", "education", "marriage"]:
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[c], df[TARGET]))
        out["chi2_categorical_vs_target"][c] = {"chi2": float(chi2), "p": float(p),
                                                "default_rate_by_group": df.groupby(c)[TARGET].mean().round(4).to_dict()}
    sp = df[num + [TARGET]].corr(method="spearman")
    out["spearman_with_target"] = sp[TARGET].drop(TARGET).round(4).to_dict()

    sns.countplot(x=TARGET, data=df); plt.title("Class balance"); plt.savefig(FIG_DIR / "class_balance.png", dpi=120); plt.close()
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for a, c in zip(ax.ravel(), ["limit_bal", "age", "bill_1", "paid_1"]):
        s = df[c]; sns.histplot(s.clip(upper=s.quantile(0.99)), bins=40, ax=a); a.set_title(c)
    df.groupby("pay_1")[TARGET].mean().plot.bar(ax=ax[1, 1], title="Default rate by latest repayment status (pay_1)")
    df.groupby("education")[TARGET].mean().plot.bar(ax=ax[1, 2], title="Default rate by education")
    plt.tight_layout(); plt.savefig(FIG_DIR / "distributions.png", dpi=110); plt.close()
    plt.figure(figsize=(11, 9)); sns.heatmap(sp, cmap="coolwarm", center=0)
    plt.title("Spearman correlation"); plt.tight_layout(); plt.savefig(FIG_DIR / "correlation.png", dpi=110); plt.close()

    (REPORTS / "eda_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["source", "rows", "duplicates", "class_balance", "undocumented_education_codes"]}, indent=2))
    print("Saved reports/eda_summary.json and reports/figures/*.png")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true")
    run(ap.parse_args().demo)
