import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI(
    title="Sub-Saharan Micro-Grid Telemetry Optimization Engine",
    description="Production API and Dashboard for predicting solar yield and grid stability.",
    version="1.0.0"
)

# Global variables for the pipelines
solar_pipeline = None
grid_pipeline = None

# Load the production pipelines into memory
try:
    solar_pipeline = joblib.load("solar_yield_regression_pipeline.joblib")
    grid_pipeline = joblib.load("grid_status_classification_pipeline.joblib")
    print("=== Production pipelines loaded successfully into RAM ===")
except Exception as e:
    print(f"Error loading model pipelines: {str(e)}")

# Define data contract for incoming telemetry requests
class MicroGridTelemetryInput(BaseModel):
    Ambient_Temp_C: float
    Solar_Irradiance_W_m2: float
    Cloud_Cover_pct: float
    Relative_Humidity_pct: float
    Battery_SoC_pct: float
    Community_Demand_kW: float
    Soiling_Index: float

# Define data contract for outgoing prediction responses
class PredictionResponse(BaseModel):
    predicted_solar_output_kWh: float
    predicted_grid_status: str
    status_code: int

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MicroGridTelemetryInput):
    if solar_pipeline is None or grid_pipeline is None:
        raise HTTPException(status_code=500, detail="Model engines are not loaded on the server.")
    
    try:
        # Convert incoming JSON payload to a DataFrame matching the model training structure
        input_data = pd.DataFrame([{
            'Ambient_Temp_C': payload.Ambient_Temp_C,
            'Solar_Irradiance_W_m2': payload.Solar_Irradiance_W_m2,
            'Cloud_Cover_pct': payload.Cloud_Cover_pct,
            'Relative_Humidity_pct': payload.Relative_Humidity_pct,
            'Battery_SoC_pct': payload.Battery_SoC_pct,
            'Community_Demand_kW': payload.Community_Demand_kW,
            'Soiling_Index': payload.Soiling_Index
        }])
        
        # Execute Regression Inference
        raw_solar_prediction = float(solar_pipeline.predict(input_data)[0])
        solar_prediction = max(0.0, round(raw_solar_prediction, 2)) # Safeguard against negative anomalies
        
        # Execute Classification Inference
        grid_prediction = str(grid_pipeline.predict(input_data)[0])
        
        return PredictionResponse(
            predicted_solar_output_kWh=solar_prediction,
            predicted_grid_status=grid_prediction,
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution engine failure: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sub-Saharan Microgrid Analytics Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 font-sans min-h-screen flex flex-col">
        
        <header class="border-b border-slate-800 bg-slate-900/50 backdrop-blur sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="bg-amber-500 text-slate-950 p-2 rounded-lg font-black tracking-wider">
                    <i class="fa-solid fa-charging-station text-xl"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight">VANGUARD-OPTIMA</h1>
                    <p class="text-xs text-slate-400">Microgrid Telemetry Control Center</p>
                </div>
            </div>
            <div class="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-full text-xs text-emerald-400 font-medium">
                <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Cloud Predictive Engine Active
            </div>
        </header>

        <main class="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <section class="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold mb-1 flex items-center gap-2 text-amber-400">
                        <i class="fa-solid fa-sliders text-sm"></i> Telemetry Parameters
                    </h2>
                    <p class="text-sm text-slate-400 mb-6">Adjust live environmental inputs to optimize real-time generation constraints.</p>
                    
                    <form id="telemetryForm" class="space-y-5">
                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Ambient_Temp_C"><i class="fa-solid fa-temperature-half w-4"></i> Ambient Temp</label>
                                <span class="text-amber-400 font-bold"><span id="val_Ambient_Temp_C">31.5</span> °C</span>
                            </div>
                            <input type="range" id="Ambient_Temp_C" name="Ambient_Temp_C" min="10" max="50" step="0.1" value="31.5" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Ambient_Temp_C').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Solar_Irradiance_W_m2"><i class="fa-solid fa-sun w-4"></i> Solar Irradiance</label>
                                <span class="text-amber-400 font-bold"><span id="val_Solar_Irradiance_W_m2">750</span> W/m²</span>
                            </div>
                            <input type="range" id="Solar_Irradiance_W_m2" name="Solar_Irradiance_W_m2" min="0" max="1200" step="5" value="750" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Solar_Irradiance_W_m2').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Cloud_Cover_pct"><i class="fa-solid fa-cloud w-4"></i> Cloud Cover</label>
                                <span class="text-amber-400 font-bold"><span id="val_Cloud_Cover_pct">12.5</span> %</span>
                            </div>
                            <input type="range" id="Cloud_Cover_pct" name="Cloud_Cover_pct" min="0" max="100" step="0.1" value="12.5" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Cloud_Cover_pct').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Relative_Humidity_pct"><i class="fa-solid fa-droplet w-4"></i> Relative Humidity</label>
                                <span class="text-amber-400 font-bold"><span id="val_Relative_Humidity_pct">65</span> %</span>
                            </div>
                            <input type="range" id="Relative_Humidity_pct" name="Relative_Humidity_pct" min="10" max="100" step="1" value="65" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Relative_Humidity_pct').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Battery_SoC_pct"><i class="fa-solid fa-battery-three-quarters w-4"></i> Battery Charge (SoC)</label>
                                <span class="text-amber-400 font-bold"><span id="val_Battery_SoC_pct">82</span> %</span>
                            </div>
                            <input type="range" id="Battery_SoC_pct" name="Battery_SoC_pct" min="0" max="100" step="1" value="82" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Battery_SoC_pct').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Community_Demand_kW"><i class="fa-solid fa-house-lightning w-4"></i> Community Demand</label>
                                <span class="text-amber-400 font-bold"><span id="val_Community_Demand_kW">45.2</span> kW</span>
                            </div>
                            <input type="range" id="Community_Demand_kW" name="Community_Demand_kW" min="5" max="200" step="0.1" value="45.2" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Community_Demand_kW').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                                <label for="Soiling_Index"><i class="fa-solid fa-dust w-4"></i> Panel Dust/Soiling Index</label>
                                <span class="text-amber-400 font-bold"><span id="val_Soiling_Index">1.2</span></span>
                            </div>
                            <input type="range" id="Soiling_Index" name="Soiling_Index" min="0.5" max="5.0" step="0.1" value="1.2" class="w-full accent-amber-500 bg-slate-800 rounded-lg h-2 cursor-pointer" oninput="document.getElementById('val_Soiling_Index').innerText=this.value">
                        </div>
                    </form>
                </div>

                <button type="button" onclick="runOptimization()" class="w-full mt-6 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-slate-950 font-bold py-3.5 px-4 rounded-xl shadow-lg transition-all duration-150 transform active:scale-[0.98] flex items-center justify-center gap-2 text-sm tracking-wide">
                    <i class="fa-solid fa-bolt-lightning"></i> Run Matrix Inference
                </button>
            </section>

            <section class="lg:col-span-7 flex flex-col gap-6">
                
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between flex-1 relative overflow-hidden group">
                    <div class="absolute -right-4 -bottom-4 text-slate-800/20 text-9xl font-black select-none pointer-events-none transition-transform group-hover:scale-105 duration-300">
                        <i class="fa-solid fa-solar-panel"></i>
                    </div>
                    <div>
                        <span class="text-xs font-bold text-slate-400 tracking-wider uppercase bg-slate-800 px-2.5 py-1 rounded-md">Machine Learning Yield Regressor</span>
                        <h3 class="text-md font-medium mt-3 text-slate-300">Predicted Solar Array Output</h3>
                    </div>
                    <div class="my-6">
                        <span id="output_solar" class="text-5xl md:text-6xl font-black text-amber-400 tracking-tight transition-all duration-300">221.66</span>
                        <span class="text-xl font-bold text-slate-400 ml-1">kWh</span>
                    </div>
                    <div class="text-xs text-slate-400 flex items-center gap-1.5 border-t border-slate-800/60 pt-3">
                        <i class="fa-solid fa-circle-check text-emerald-400"></i> High-resolution continuous scalar prediction model
                    </div>
                </div>

                <div id="status_card" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between flex-1 relative overflow-hidden group border-l-4 border-l-emerald-500 transition-all duration-300">
                    <div class="absolute -right-4 -bottom-4 text-slate-800/20 text-9xl font-black select-none pointer-events-none transition-transform group-hover:scale-105 duration-300">
                        <i class="fa-solid fa-tower-cell"></i>
                    </div>
                    <div>
                        <span class="text-xs font-bold text-slate-400 tracking-wider uppercase bg-slate-800 px-2.5 py-1 rounded-md">Deep Classifier Network</span>
                        <h3 class="text-md font-medium mt-3 text-slate-300">Grid Operational Stability Status</h3>
                    </div>
                    <div class="my-6">
                        <span id="output_grid" class="text-4xl md:text-5xl font-black tracking-wide text-emerald-400 transition-all duration-300 uppercase">Stable</span>
                    </div>
                    <div class="text-xs text-slate-400 flex items-center gap-1.5 border-t border-slate-800/60 pt-3">
                        <i class="fa-solid fa-circle-nodes text-slate-400"></i> Probability matrix mapping network loading patterns
                    </div>
                </div>

            </section>
        </main>

        <script>
            async function runOptimization() {
                // Grab layout target DOM variables
                const solarOut = document.getElementById('output_solar');
                const gridOut = document.getElementById('output_grid');
                const statusCard = document.getElementById('status_card');
                
                // Add calculation animation placeholder effect
                solarOut.innerText = "---";
                gridOut.innerText = "CALCULATING...";
                gridOut.className = "text-2xl font-bold tracking-wide text-amber-400 animate-pulse uppercase";

                // Map inputs dynamically into exact data contract matching Pydantic structure
                const payload = {
                    Ambient_Temp_C: parseFloat(document.getElementById('Ambient_Temp_C').value),
                    Solar_Irradiance_W_m2: parseFloat(document.getElementById('Solar_Irradiance_W_m2').value),
                    Cloud_Cover_pct: parseFloat(document.getElementById('Cloud_Cover_pct').value),
                    Relative_Humidity_pct: parseFloat(document.getElementById('Relative_Humidity_pct').value),
                    Battery_SoC_pct: parseFloat(document.getElementById('Battery_SoC_pct').value),
                    Community_Demand_kW: parseFloat(document.getElementById('Community_Demand_kW').value),
                    Soiling_Index: parseFloat(document.getElementById('Soiling_Index').value)
                };

                try {
                    // Send HTTP POST asynchronous fetch call to local/production endpoint
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!response.ok) throw new Error("Server engine error status");
                    
                    const data = await response.json();

                    // Apply visual updates cleanly to the page layout
                    solarOut.innerText = data.predicted_solar_output_kWh.toFixed(2);
                    gridOut.innerText = data.predicted_grid_status;

                    // Update UI Alert Badge styling colors dynamically based on safety threshold metrics
                    if(data.predicted_grid_status.toLowerCase() === 'stable' || data.predicted_grid_status.toLowerCase() === 'normal') {
                        gridOut.className = "text-4xl md:text-5xl font-black tracking-wide text-emerald-400 uppercase transition-all duration-300";
                        statusCard.className = "bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between flex-1 relative overflow-hidden group border-l-4 border-l-emerald-500 transition-all duration-300";
                    } else {
                        gridOut.className = "text-4xl md:text-5xl font-black tracking-wide text-rose-500 uppercase transition-all duration-300 animate-pulse";
                        statusCard.className = "bg-slate-900 border border-rose-950 rounded-2xl p-6 flex flex-col justify-between flex-1 relative overflow-hidden group border-l-4 border-l-rose-500 shadow-[0_0_30px_rgba(244,63,94,0.15)] transition-all duration-300";
                    }

                } catch (error) {
                    console.error("Inference Error:", error);
                    solarOut.innerText = "ERR";
                    gridOut.innerText = "ENGINE ERROR";
                    gridOut.className = "text-xl font-bold tracking-wide text-rose-500 uppercase";
                    statusCard.className = "bg-slate-900 border border-rose-900 rounded-2xl p-6 flex flex-col justify-between flex-1 relative overflow-hidden group border-l-4 border-l-rose-500 transition-all duration-300";
                }
            }
            
            // Automatically execute a baseline calculation on dashboard load
            window.onload = runOptimization;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
