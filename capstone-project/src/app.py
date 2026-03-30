import os
import pickle
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize FastAPI application
app = FastAPI(title="Conversion Model API", description="Production API for Demand & Conversion Forecasting")

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Setup Templates and Static Files
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Global variables for model and metrics
model = None
accuracy_percentage = None
expected_features = []

# Pydantic schemas
class PredictRequest(BaseModel):
    num_touchpoints: int
    unique_channels: int
    time_to_conversion_hours: float
    first_channel: str
    last_channel: str

class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float

@app.on_event("startup")
def load_assets():
    global model, accuracy_percentage, expected_features
    
    print("Loading model and calculating metrics...")
    
    # 1. Load the Model
    model_path = os.path.join(PROJECT_ROOT, "model", "trained_model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        
    # 2. Compute Accuracy
    data_path = os.path.join(PROJECT_ROOT, "data", "processed_data.csv")
    try:
        df = pd.read_csv(data_path)
        X = df.drop(columns=['User_ID', 'converted'])
        y = df['converted']
        expected_features = list(X.columns)
        
        # We use the same train/test split seed as the training pipeline to get the exact test accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_percentage = round(accuracy * 100, 2)
        print(f"Metrics calculated. Test Accuracy: {accuracy_percentage}%")
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        accuracy_percentage = 0.0

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "accuracy": accuracy_percentage
    })

@app.get("/manual", response_class=HTMLResponse)
async def serve_manual(request: Request):
    return templates.TemplateResponse("manual.html", {
        "request": request
    })

@app.get("/api/metrics")
async def get_metrics():
    return {"accuracy": accuracy_percentage}

@app.get("/api/feature-importance")
async def get_feature_importance():
    if not model or not expected_features:
        return {"error": "Model not loaded"}
    
    importances = model.feature_importances_
    features_dict = dict(zip(expected_features, [float(i) for i in importances]))
    sorted_features = dict(sorted(features_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_features

@app.get("/api/budget-optimizer")
async def get_budget_optimizer(total_budget: float = 100000):
    try:
        import json
        attr_path = os.path.join(PROJECT_ROOT, "src", "attribution_comparison.json")
        with open(attr_path, "r") as f:
            data = json.load(f)
        
        markov = data.get("markov_attribution", {})
        summary = data.get("comparison_summary", "")
        
        # Calculate allocations based on markov probabilities
        allocations = {}
        for channel, score in markov.items():
            allocations[channel] = round(score * total_budget, 2)
            
        return {
            "total_budget": total_budget,
            "allocations": allocations,
            "markov_scores": markov,
            "summary": summary
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/predict", response_model=PredictResponse)
async def predict(request_data: PredictRequest):
    global model, expected_features
    if not model:
        return {"error": "Model not loaded"}

    # Initialize a dictionary with zeros for all expected features
    input_dict = {col: 0 for col in expected_features}
    
    # Set numeric features
    input_dict['num_touchpoints'] = request_data.num_touchpoints
    input_dict['unique_channels'] = request_data.unique_channels
    input_dict['time_to_conversion_hours'] = request_data.time_to_conversion_hours
    
    # Set encoded categorical features
    first_channel_col = f"first_channel_{request_data.first_channel}"
    last_channel_col = f"last_channel_{request_data.last_channel}"
    
    if first_channel_col in input_dict:
        input_dict[first_channel_col] = 1
    if last_channel_col in input_dict:
        input_dict[last_channel_col] = 1
        
    # Create DataFrame with the exact column order expected by the model
    input_df = pd.DataFrame([input_dict], columns=expected_features)
    
    # Ensure types match the model requirements (e.g., bool or int for dummy columns)
    # The processed_data used True/False string or boolean map
    for col in input_df.columns:
        if 'channel' in col:
            input_df[col] = input_df[col].astype(bool)
    
    # Predict
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    
    prediction_label = "Converted" if prediction == 1 else "Not Converted"
    
    return PredictResponse(
        prediction=prediction,
        prediction_label=prediction_label,
        probability=probability
    )
