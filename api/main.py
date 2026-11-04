from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field, create_model
from src.config import BILL, PAID, PAY
from src.predict import load_bundle, predict

EXAMPLE = {"limit_bal": 120000, "education": 2, "age": 35,
           **{p: v for p, v in zip(PAY, [2, 2, 0, 0, 0, 0])},
           **{b: v for b, v in zip(BILL, [3200, 2900, 2600, 2400, 2000, 1800])},
           **{p: v for p, v in zip(PAID, [0, 1000, 1000, 800, 800, 700])}}

_fields = {"limit_bal": (float, Field(..., gt=0, le=1e8, description="Credit limit (NT$)")),
           "education": (int, Field(..., ge=1, le=4, description="1 grad school, 2 university, 3 high school, 4 other")),
           "age": (int, Field(..., ge=18, le=110))}
for c in PAY:
    _fields[c] = (int, Field(0, ge=-2, le=9, description="-2 no use, -1 paid duly, 0 revolving, 1..9 months delayed"))
for c in BILL:
    _fields[c] = (float, Field(0, ge=-1e7, le=1e8, description="Bill amount"))
for c in PAID:
    _fields[c] = (float, Field(0, ge=0, le=1e8, description="Amount paid"))
Applicant = create_model("Applicant", __config__=ConfigDict(json_schema_extra={"example": EXAMPLE}), **_fields)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.bundle = load_bundle()
    except FileNotFoundError:
        app.state.bundle = None
    yield


app = FastAPI(title="AI Financial Risk Analyzer", version="1.0.0", lifespan=lifespan)


def _bundle():
    if app.state.bundle is None:
        raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
    return app.state.bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.bundle is not None}


@app.get("/model-info")
def model_info():
    b = _bundle()
    return {"model": b["model_name"], "data_source": b["data_source"], "threshold": b["threshold"],
            "features": b["features"], "global_importance": dict(list(b["global_importance"].items())[:10])}


@app.post("/predict")
def predict_one(a: Applicant):
    return predict(_bundle(), [a.model_dump()])[0]


@app.post("/predict/batch")
def predict_batch(items: List[Applicant]):
    if not 1 <= len(items) <= 1000:
        raise HTTPException(422, "Send between 1 and 1000 records.")
    return predict(_bundle(), [i.model_dump() for i in items])
