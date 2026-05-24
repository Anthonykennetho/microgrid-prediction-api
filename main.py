import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Instantiate the core FastAPI application
app = FastAPI(
    title="Sub-Saharan Micro-Grid Telemetry Optimization Engine",
    description="Production API delivering real-time solar yield forecasts and grid outage risk assessments.",
    version="1.0.0"
)

# 2. Pre-load serialized ML Pipelines directly into memory to optimize execution speeds
try:
    regression_model = joblib.load('solar_yield_regression_pipeline.joblib')
    classification_model = joblib.load('grid_status_classification_pipeline.joblib')
    print("=== Production pipelines loaded successfully into RAM ===")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load pipeline files from disk. Details: {str(e)}")

# 3. Define the Input Data Contract using Pydantic Models for structural validation
class MicroGridTelemetryInput(BaseModel):
    Ambient_Temp_C: float = Field(..., description="Ambient temperature in degrees Celsius", example=31.5)
    Solar_Irradiance_W_m2: float = Field(..., description="Solar irradiance in Watts per square meter", example=750.0)
    Cloud_Cover_pct: float = Field(..., description="Cloud cover percentage (0-100%)", example=12.5)
    Relative_Humidity_pct: float = Field(..., description="Relative humidity percentage (0-100%)", example=65.0)
    Battery_SoC_pct: float = Field(..., description="Battery State of Charge percentage (0-100%)", example=82.0)
    Community_Demand_kW: float = Field(..., description="Instantaneous community consumption load in kW", example=45.2)
    Soiling_Index: float = Field(..., description="Harmattan dust accumulation factor coefficient", example=1.2)

# 4. Define the Output Data Contract for predictable JSON API responses
class PredictionResponse(BaseModel):
    predicted_solar_output_kWh: float
    predicted_grid_status: str
    status_code: int = 200

# 5. Define a standard health-check route for continuous uptime tracking
@app.get("/")
def read_root():
    return {
        "system_status": "ONLINE",
        "active_models": ["Solar_Yield_Regressor", "Grid_Status_Classifier"],
        "region_scope": "Sub-Saharan Micro-Grid Cluster A"
    }

# 6. Define the core Inference Route for unified dual predictions
@app.post("/predict", response_model=PredictionResponse)
async def generate_predictions(payload: MicroGridTelemetryInput):
    try:
        # Convert incoming JSON payload into an explicit dictionary matching our schema
        input_dict = {
            'Ambient_Temp_C': payload.Ambient_Temp_C,
            'Solar_Irradiance_W_m2': payload.Solar_Irradiance_W_m2,
            'Cloud_Cover_pct': payload.Cloud_Cover_pct,
            'Relative_Humidity_pct': payload.Relative_Humidity_pct,
            'Battery_SoC_pct': payload.Battery_SoC_pct,
            'Community_Demand_kW': payload.Community_Demand_kW,
            'Soiling_Index': payload.Soiling_Index
        }
        
        # Wrap into a Pandas DataFrame to maintain matching column orientation and names
        input_df = pd.DataFrame([input_dict])
        
        # Execute model predictions using the fully automated pipelines
        raw_reg_pred = regression_model.predict(input_df)[0]
        raw_clf_pred = classification_model.predict(input_df)[0]
        
        # Post-processing: Ensure regression predictions do not violate physical lower limits (0 kWh)
        final_reg_pred = max(0.0, float(raw_reg_pred))
        
        return PredictionResponse(
            predicted_solar_output_kWh=round(final_reg_pred, 2),
            predicted_grid_status=str(raw_clf_pred)
        )
        
    except Exception as err:
        # Prevent server crashes by wrapping runtime exceptions in a standard HTTP 500 block
        raise HTTPException(
            status_code=500, 
            detail=f"Inference execution engine failure: {str(err)}"
        )
