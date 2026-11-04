import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src.config import BASE, FIG_DIR, METRICS_PATH
from src.predict import load_bundle, predict

st.set_page_config(page_title="AI Financial Risk Analyzer", layout="wide")
st.title("AI Financial Risk Analyzer")


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(f"{e}"); st.stop()
if "SYNTHETIC" in bundle["data_source"]:
    st.warning("This model was trained on SYNTHETIC demo data. Predictions are not meaningful.")

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Batch (CSV)", "Model performance", "Data analytics"])

with tab1:
    st.caption("Repayment status: -2 no consumption, -1 paid duly, 0 revolving credit, 1-9 = months of payment delay. Month 1 = most recent.")
    c = st.columns(3)
    limit = c[0].number_input("Credit limit (NT$)", 1000.0, 1e8, 120000.0, 10000.0)
    edu = c[1].selectbox("Education", [1, 2, 3, 4], index=1, format_func=lambda v: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[v])
    age = c[2].number_input("Age", 18, 110, 35)
    rec = dict(limit_bal=limit, education=edu, age=age)
    h = st.columns(3); h[0].markdown("**Repayment status**"); h[1].markdown("**Bill amount**"); h[2].markdown("**Amount paid**")
    for i in range(1, 7):
        r_ = st.columns(3)
        rec[f"pay_{i}"] = r_[0].number_input(f"pay_{i}", -2, 9, 0, key=f"p{i}", label_visibility="collapsed")
        rec[f"bill_{i}"] = r_[1].number_input(f"bill_{i}", -1e7, 1e8, 0.0, 100.0, key=f"b{i}", label_visibility="collapsed")
        rec[f"paid_{i}"] = r_[2].number_input(f"paid_{i}", 0.0, 1e8, 0.0, 100.0, key=f"a{i}", label_visibility="collapsed")
    if st.button("Analyze risk", type="primary"):
        r = predict(bundle, [rec], top_k=8)[0]
        m = st.columns(3)
        m[0].metric("Default risk probability", f"{r['risk_probability']:.1%}")
        m[1].metric("Risk level", r["risk_level"])
        m[2].metric("Decision threshold", f"{r['decision_threshold']:.1%}")
        st.progress(min(r["risk_probability"], 1.0))
        f = pd.DataFrame(r["top_factors"])[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(f["feature"], f["impact"], color=["#c0392b" if v > 0 else "#2e86c1" for v in f["impact"]])
        ax.set_title(f"Top contributing features ({r['explanation_method']}); red = raises risk")
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(r["top_factors"]), use_container_width=True)

with tab2:
    st.caption("CSV columns: " + ", ".join(BASE))
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        try:
            d = pd.read_csv(up)
            miss = set(BASE) - set(d.columns)
            if miss:
                st.error(f"Missing columns: {sorted(miss)}")
            else:
                res = predict(bundle, d[BASE].to_dict("records"), top_k=1)
                d["risk_probability"] = [x["risk_probability"] for x in res]
                d["risk_level"] = [x["risk_level"] for x in res]
                st.dataframe(d)
                st.download_button("Download results", d.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")

with tab3:
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        st.write(f"**Data:** {m['data_source']} | **Best model:** {m['best_model']}")
        st.subheader("Cross-validation (train set)"); st.dataframe(pd.DataFrame(m["cv_results"]))
        st.subheader("Held-out test metrics")
        st.json({k: v for k, v in m["test_metrics"].items() if k != "confusion_matrix"})
        cols = st.columns(3)
        for col, n in zip(cols, ["roc_curve", "pr_curve", "confusion_matrix"]):
            if (FIG_DIR / f"{n}.png").exists():
                col.image(str(FIG_DIR / f"{n}.png"))
        fa = FIG_DIR.parent / "fairness_audit.csv"
        if fa.exists():
            st.subheader("Fairness audit (test set, subgroup metrics)"); st.dataframe(pd.read_csv(fa))
        if (FIG_DIR / "feature_importance.png").exists():
            st.image(str(FIG_DIR / "feature_importance.png"))
    else:
        st.info("Run training first.")

with tab4:
    shown = False
    for n in ["class_balance", "distributions", "correlation"]:
        if (FIG_DIR / f"{n}.png").exists():
            st.image(str(FIG_DIR / f"{n}.png")); shown = True
    if not shown:
        st.info("Run `python -m src.eda` to generate analytics.")
