from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import joblib, json

app = FastAPI(title="Retail Demand Estimation & Stockout Risk API")

model = joblib.load("best_demand_forecasting_model.pkl")
maps = json.load(open("category_maps.json"))

FEATURES = ["Product_Code", "Warehouse", "Product_Category", "Year", "Month", "Day",
            "Weekday", "Quarter", "Month_sin", "Month_cos", "Weekday_sin", "Weekday_cos",
            "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_std_7"]

class PredictRequest(BaseModel):
    product_code: str        # e.g. "Product_1359"
    warehouse: str           # e.g. "Whse_J"
    product_category: str    # e.g. "Category_019"
    date: str                # YYYY-MM-DD
    lag_1: float
    lag_7: float
    lag_30: float
    rolling_mean_7: float
    rolling_std_7: float
    current_stock: float     # stock on hand, used by the stockout rule

@app.get("/")
def home():
    return {"message": "Demand API is running. Use POST /predict or open /docs"}

@app.post("/predict")
def predict(r: PredictRequest):
    try:
        d = datetime.strptime(r.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")
    try:
        pc = maps["Product_Code"][r.product_code]
        wh = maps["Warehouse"][r.warehouse]
        cat = maps["Product_Category"][r.product_category]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown value: {e}")

    wd = d.weekday()
    row = {
        "Product_Code": pc, "Warehouse": wh, "Product_Category": cat,
        "Year": d.year, "Month": d.month, "Day": d.day, "Weekday": wd,
        "Quarter": (d.month - 1) // 3 + 1,
        "Month_sin": float(np.sin(2 * np.pi * d.month / 12)),
        "Month_cos": float(np.cos(2 * np.pi * d.month / 12)),
        "Weekday_sin": float(np.sin(2 * np.pi * wd / 7)),
        "Weekday_cos": float(np.cos(2 * np.pi * wd / 7)),
        "lag_1": r.lag_1, "lag_7": r.lag_7, "lag_30": r.lag_30,
        "rolling_mean_7": r.rolling_mean_7, "rolling_std_7": r.rolling_std_7,
    }
    X = pd.DataFrame([row])[list(model.feature_names_in_)]
    pred = max(float(model.predict(X)[0]), 0.0)

    # Explicit inventory rule: at risk if predicted demand exceeds stock on hand
    at_risk = pred > r.current_stock
    return {
        "product": r.product_code,
        "predicted_demand": round(pred, 1),
        "current_stock": r.current_stock,
        "stockout_risk": bool(at_risk),
        "action": "Do not promote; reorder" if at_risk else "OK to promote",
    }