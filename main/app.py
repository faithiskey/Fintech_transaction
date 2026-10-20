from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# 1. Locate and load the trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "xgboost_transaction_amount_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Fintech Transaction Prediction API",
    description="API for predicting transaction amounts using an XGBoost model.",
    version="1.0.0"
)


# --------------------------------------------------
# 3. Input schema
# --------------------------------------------------

class TransactionRequest(BaseModel):
    Transaction_Mode: str
    Transaction_Type: str
    Transaction_Source: str
    Transaction_Destination: str
    Year: int
    Month: int
    Hour: int


# --------------------------------------------------
# 4. Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "model_loaded": True
    }


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_transaction(request: TransactionRequest):

    input_data = pd.DataFrame([{
        "Transaction_Mode": request.Transaction_Mode,
        "Transaction_Type": request.Transaction_Type,
        "Transaction_Source": request.Transaction_Source,
        "Transaction_Destination": request.Transaction_Destination,
        "Year": request.Year,
        "Month": request.Month,
        "Hour": request.Hour
    }])

    prediction = model.predict(input_data)

    predicted_amount = float(prediction[0])

    return {
        "predicted_transaction_amount": predicted_amount
    }