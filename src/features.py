"""Feature engineering + preprocessing. Everything learned from data is fit inside the Pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import BASE, BILL, PAID, PAY

NEW = ["max_delay", "mean_delay", "months_delayed", "delay_trend", "avg_utilization", "latest_utilization",
       "util_trend", "over_limit_months", "total_paid", "payment_ratio", "avg_bill"]
ENGINEERED = BASE + NEW


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless: payment-behaviour, utilization and repayment-ratio features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reindex(columns=BASE).astype(float)
        lim = X["limit_bal"].replace(0, np.nan)
        pay, bill, paid = X[PAY], X[BILL], X[PAID]
        X["max_delay"] = pay.max(axis=1)
        X["mean_delay"] = pay.mean(axis=1)
        X["months_delayed"] = (pay >= 1).sum(axis=1)
        X["delay_trend"] = X["pay_1"] - X["pay_6"]
        X["avg_utilization"] = bill.mean(axis=1) / lim
        X["latest_utilization"] = X["bill_1"] / lim
        X["util_trend"] = (X["bill_1"] - X["bill_6"]) / lim
        X["over_limit_months"] = bill.gt(lim, axis=0).sum(axis=1)
        X["total_paid"] = paid.sum(axis=1)
        X["payment_ratio"] = paid.sum(axis=1) / (bill.clip(lower=0).sum(axis=1) + 1)
        X["avg_bill"] = bill.mean(axis=1)
        return X[ENGINEERED].replace([np.inf, -np.inf], np.nan)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips outliers to quantiles learned on the TRAINING data only."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_pipeline(model, scale: bool = False) -> Pipeline:
    steps = [("fe", FeatureEngineer()), ("wins", Winsorizer()),
             ("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps + [("clf", model)])
