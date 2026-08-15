from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

app = FastAPI(title="Demand Forecasting API")

CATEGORIES = {
    "cama_mesa_banho": {
        "model": joblib.load("models/model_cama_mesa_banho.pkl"),
        "data": pd.read_csv("data/weekly_data_cama_mesa_banho.csv", parse_dates=["date"])
    },
    "beleza_saude": {
        "model": joblib.load("models/model_beleza_saude.pkl"),
        "data": pd.read_csv("data/weekly_data_beleza_saude.csv", parse_dates=["date"])
    }
}

FEATURE_COLS = ["month", "week_of_year", "lag_1", "lag_2", "lag_4", "rolling_mean_4",
                "is_black_friday_week", "is_world_cup_week",
                "time_index", "pct_change_1", "diff_1", "trend_slope_4"]

class ForecastRequest(BaseModel):
    category: str
    weeks_ahead: int = 4

class ForecastResponse(BaseModel):
    category: str
    forecast: list

@app.get("/")
def root():
    return {"message": "Demand Forecasting API", "available_categories": list(CATEGORIES.keys())}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    if request.category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Category not found. Available: {list(CATEGORIES.keys())}")
    
    if request.weeks_ahead < 1 or request.weeks_ahead > 12:
        raise HTTPException(status_code=400, detail="weeks_ahead must be between 1 and 12")

    model = CATEGORIES[request.category]["model"]
    history = CATEGORIES[request.category]["data"].copy().sort_values("date").reset_index(drop=True)

    predictions = []
    working_history = history["units_sold"].tolist()
    last_date = history["date"].max()
    last_time_index = history["time_index"].max()

    for step in range(request.weeks_ahead):
        next_date = last_date + timedelta(weeks=step + 1)
        
        lag_1 = working_history[-1]
        lag_2 = working_history[-2]
        lag_4 = working_history[-4]
        rolling_mean_4 = np.mean(working_history[-5:-1])
        diff_1 = working_history[-1] - working_history[-2]
        pct_change_1 = (working_history[-1] - working_history[-2]) / working_history[-2] if working_history[-2] != 0 else 0
        
        recent_4 = working_history[-4:]
        trend_slope_4 = np.polyfit(np.arange(4), recent_4, 1)[0]

        features = pd.DataFrame([{
            "month": next_date.month,
            "week_of_year": next_date.isocalendar()[1],
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_4": lag_4,
            "rolling_mean_4": rolling_mean_4,
            "is_black_friday_week": 0,
            "is_world_cup_week": 0,
            "time_index": last_time_index + step + 1,
            "pct_change_1": pct_change_1,
            "diff_1": diff_1,
            "trend_slope_4": trend_slope_4
        }])[FEATURE_COLS]

        pred = float(model.predict(features)[0])
        pred = max(0, pred)

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_units": round(pred, 1)
        })

        working_history.append(pred)

    return ForecastResponse(category=request.category, forecast=predictions)