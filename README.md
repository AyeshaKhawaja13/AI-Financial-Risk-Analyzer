# AI Financial Risk Analyzer

End-to-end ML system that predicts the probability a credit card client defaults on next month's payment, using the **UCI Default of Credit Card Clients** dataset (Taiwan, 2005; 30,000 clients, 23 features, ~22% default).

**Pipeline:** data → EDA and statistical tests → cleaning → feature engineering → model comparison with CV → tuning → probability calibration → explainability → fairness audit → FastAPI → Streamlit.

## Features
- **Leakage-safe:** split first; winsorization quantiles, imputation and scaling are fit inside a sklearn `Pipeline` on training data only; duplicates removed before splitting.
- **Feature engineering** from 6 months of history: max/mean delay, months delayed, delay trend, credit utilization (avg, latest, trend), over-limit months, repayment ratio.
- **5 models** (Dummy, Logistic Regression, Random Forest, HistGradientBoosting, XGBoost), stratified 5-fold CV: ROC-AUC, PR-AUC, Precision, Recall, F1.
- **Imbalance:** class weights / `scale_pos_weight` (no resampling).
- **Tuning** with randomized search on PR-AUC; **isotonic calibration** and the decision threshold (max F1) are learned from out-of-fold training predictions only. The test set is evaluated once.
- **Explainability:** SHAP for tree models, occlusion fallback for the others.
- **Fairness:** `sex` and `marriage` are NOT model inputs. A subgroup audit by sex, education and age band is saved to `reports/fairness_audit.csv`.
- **FastAPI** with strict validation, batch endpoint; **Streamlit** dashboard with single prediction, CSV batch scoring, performance, fairness and EDA tabs.
- Unit tests and GitHub Actions CI.

## Setup
```powershell
python -m pip install -r requirements.txt
```
(Optionally create a venv first: `python -m venv .venv`, then activate it.)

## Data
Place `credit_default.csv` in `data/raw/` (columns `X1..X23` + `default`, or the original UCI names `LIMIT_BAL, PAY_0 ...`).
Easiest way:
```powershell
python -m pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; d=fetch_ucirepo(id=350); df=d.data.features.copy(); df['default']=d.data.targets.iloc[:,0]; df.to_csv('data/raw/credit_default.csv', index=False)"
```

## Run
```powershell
.\run.ps1 eda          # or: python -m src.eda
.\run.ps1 train        # or: python -m src.train      (add --no-tune for a fast run)
.\run.ps1 api          # http://127.0.0.1:8000/docs
.\run.ps1 dashboard    # http://localhost:8501
.\run.ps1 test
```
If PowerShell blocks scripts: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; Unblock-File .\run.ps1`, or run the `python -m ...` commands directly.
`python -m src.train --demo` uses synthetic data purely to test the code; its numbers mean nothing.

## Results
Intentionally left blank until you train on the real data. Everything is saved in `models/metrics.json`, `reports/cv_results.csv`, `reports/fairness_audit.csv`. Fill in:

| Model | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| _from reports/cv_results.csv_ | | |

## Sample API call
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
 "limit_bal": 120000, "education": 2, "age": 35,
 "pay_1": 2, "pay_2": 2, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
 "bill_1": 3200, "bill_2": 2900, "bill_3": 2600, "bill_4": 2400, "bill_5": 2000, "bill_6": 1800,
 "paid_1": 0, "paid_2": 1000, "paid_3": 1000, "paid_4": 800, "paid_5": 800, "paid_6": 700}'
```
Response shape (values depend on your trained model):
```json
{"risk_probability": 0.0, "risk_level": "Low|Medium|High", "flagged_high_risk": false,
 "decision_threshold": 0.0, "explanation_method": "shap|occlusion",
 "top_factors": [{"feature": "...", "value": 0.0, "impact": 0.0, "direction": "increases risk"}]}
```
`pay_1` = most recent month (September); `pay_6` = April. Status: -2 no consumption, -1 paid duly, 0 revolving, 1-9 months delayed.
Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`.

## Structure
```
src/ config, data, features, eda, train, explain, predict    api/main.py    app/dashboard.py
tests/    data/raw (not committed)    models/    reports/    run.ps1
```

## Limitations and ethical considerations
- Data is from one Taiwanese bank in 2005 (credit card defaults); it will not generalize to other countries, products or eras without retraining and validation.
- Only approved cardholders appear in the data (selection bias); the label covers one month only.
- Undocumented codes exist in `EDUCATION` (0, 5, 6) and `MARRIAGE` (0); they are mapped to "other".
- `sex` and `marriage` are excluded from the model, but other features (age, education, limit) can act as proxies. The fairness audit shows subgroup differences; it is a diagnostic, not a guarantee of fairness, and subgroup counts can be small.
- Explanations show model behaviour, not causes. Calibration and thresholds depend on the ~22% base rate and must be redone if the population shifts.
- Educational/portfolio project. **Not for real credit decisions.** Real use needs legal review (e.g. fair-lending rules), human oversight, adverse-action explanations and monitoring.
