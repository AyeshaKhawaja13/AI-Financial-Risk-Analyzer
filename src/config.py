from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "credit_default.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
TARGET = "default"

PAY = [f"pay_{i}" for i in range(1, 7)]    # repayment status, pay_1 = most recent month (Sept 2005)
BILL = [f"bill_{i}" for i in range(1, 7)]  # bill amount
PAID = [f"paid_{i}" for i in range(1, 7)]  # amount paid
X_NAMES = ["limit_bal", "sex", "education", "marriage", "age"] + PAY + BILL + PAID  # = X1..X23

RENAME = {f"X{i + 1}": n for i, n in enumerate(X_NAMES)}
RENAME.update({"LIMIT_BAL": "limit_bal", "SEX": "sex", "EDUCATION": "education", "MARRIAGE": "marriage",
               "AGE": "age", "PAY_0": "pay_1", **{f"PAY_{i}": f"pay_{i}" for i in range(2, 7)},
               **{f"BILL_AMT{i}": f"bill_{i}" for i in range(1, 7)},
               **{f"PAY_AMT{i}": f"paid_{i}" for i in range(1, 7)}})
TARGET_ALIASES = ["default payment next month", "default.payment.next.month", "Y"]

SENSITIVE = ["sex", "marriage"]  # NOT used as model inputs; used only for the fairness audit
BASE = ["limit_bal", "education", "age"] + PAY + BILL + PAID  # model input features (21)
