# Vanguard-Optima: Sub-Saharan Micro-Grid Telemetry Optimization Engine

Vanguard-Optima is a production-grade, full-stack machine learning microservice built to manage and optimize decentralized microgrid power infrastructures in sub-Saharan Africa. The system hosts two aligned predictive engines that ingest real-time environmental data to instantly project continuous solar power yield and classify local electrical distribution grid stability states.

The application features a dual-purpose architecture: an automated data ingestion API for machine-to-machine integrations, a unified developer documentation layer, and a beautiful, high-performance web dashboard styled with Tailwind CSS for human operator telemetry monitoring.

---

## System Architecture & Stack

The application relies on the enterprise standard $N-1$ rule for machine learning deployments, utilizing **Python 3.12** to guarantee perfect wheel binary compatibility and avoid uncompiled source-code build runtime exceptions.

* **Core Framework:** FastAPI (Asynchronous High-Performance Web Engine)
* **Production Server:** Uvicorn (ASGI standard implementation)
* **Inference Compute Engines:** Scikit-Learn 1.6.1 & Pandas 2.2.2
* **Model Management:** Joblib 1.5.3
* **Data Validation Layout:** Pydantic v2
* **Frontend UI Presentation:** Hypertext Markup Language (HTML5) styled via Tailwind CSS Core Framework Engine and FontAwesome Integration Vectors.

---

## Machine Learning Framework Specifications

The microservice operates concurrently over two separate serialized pipelines loaded straight into the server RAM upon bootstrap initialization:

1. **Solar Yield Regressor (`solar_yield_regression_pipeline.joblib`):** Ingests environmental metrics to output continuous electrical scalar power predictions measured in kilowatt-hours ($\text{kWh}$). Features an operational logic safeguard preventing negative anomalies ($\ge 0.0$).
2. **Grid Stability Classifier (`grid_status_classification_pipeline.joblib`):** Maps distribution constraints and generation variables across localized multi-class probability metrics to return grid state classifications (e.g., `"Stable"`, `"Unstable"`).

---

## Core Data Schema Contracts

### 1. Inbound Telemetry Payload (`MicroGridTelemetryInput`)

Any program or operator interacting with the execution engine must supply these variables within a verified JSON schema structure:

| Parameter Field Target | Target Unit Syntax | Minimum Bounds | Maximum Bounds | Core Baseline Example |
| :--- | :--- | :--- | :--- | :--- |
| `Ambient_Temp_C` | Celsius ($^\circ\text{C}$) | $10.0$ | $50.0$ | $31.5$ |
| `Solar_Irradiance_W_m2` | Watts per sq. meter ($\text{W/m}^2$) | $0.0$ | $1200.0$ | $750.0$ |
| `Cloud_Cover_pct` | Percentage ($\%$) | $0.0$ | $100.0$ | $12.5$ |
| `Relative_Humidity_pct`| Percentage ($\%$) | $10.0$ | $100.0$ | $65.0$ |
| `Battery_SoC_pct` | Percentage ($\%$) | $0.0$ | $100.0$ | $82.0$ |
| `Community_Demand_kW` | Kilowatts ($\text{kW}$) | $5.0$ | $200.0$ | $45.2$ |
| `Soiling_Index` | Cleanliness Scalar Factor | $0.5$ | $5.0$ | $1.2$ |

### 2. Outbound Prediction Response (`PredictionResponse`)

```json
{
  "predicted_solar_output_kWh": 221.66,
  "predicted_grid_status": "Stable",
  "status_code": 200
}
