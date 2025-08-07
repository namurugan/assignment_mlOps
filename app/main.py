from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import traceback
from app.logger import get_logger
import time

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

logger = get_logger()
app = FastAPI()

# Prometheus metrics
MODEL_UP = Gauge("model_up", "Model availability (1 = up, 0 = down)")
TOTAL_REQUESTS = Counter("total_requests", "Total number of requests")
SUCCESSFUL_PREDICTIONS = Counter("successful_predictions", "Successful prediction count")
FAILED_PREDICTIONS = Counter("failed_predictions", "Failed prediction count")
ERROR_COUNT = Counter("error_count", "Error counts by status", ["status_code"])
LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")
MODEL_VERSION = Gauge("model_version", "Version of the model")

# Load model and set status/version
try:
    model = joblib.load("model/CaliforniaHousingModel.pkl")
    MODEL_UP.set(1)
    MODEL_VERSION.set(1.0)  # Set your model version as float here
except Exception as e:
    logger.error(f"Model failed to load: {e}")
    MODEL_UP.set(0)

class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

@app.middleware("http")
async def log_requests_and_metrics(request: Request, call_next):
    TOTAL_REQUESTS.inc()
    start_time = time.time()

    try:
        # Read request body
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8") if body_bytes else ""
        logger.info(f"REQUEST: {request.method} {request.url} BODY: {body_text}")

        response: Response = await call_next(request)

        # Read response body
        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk
        response.body_iterator = iterate_in_chunks(resp_body)

        try:
            resp_log = json.loads(resp_body.decode())
        except Exception:
            resp_log = resp_body.decode(errors="ignore")
        logger.info(f"RESPONSE: status_code={response.status_code} body={resp_log}")

        LATENCY.observe(time.time() - start_time)

        if response.status_code >= 400:
            FAILED_PREDICTIONS.inc()
            ERROR_COUNT.labels(str(response.status_code)).inc()
        else:
            SUCCESSFUL_PREDICTIONS.inc()

        return response

    except Exception:
        logger.error(f"Unhandled middleware error:\n{traceback.format_exc()}")
        FAILED_PREDICTIONS.inc()
        ERROR_COUNT.labels("500").inc()
        raise

async def iterate_in_chunks(data):
    yield data

@app.post("/predict")
def predict(data: HousingInput):
    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        logger.info(f"INPUT: {input_dict} | PREDICTION: {prediction}")
        return {"prediction": prediction}
    except Exception:
        logger.error(f"ERROR:\n{traceback.format_exc()}")
        return {"error": "Prediction failed due to internal error."}

@app.get("/metrics")
def metrics_endpoint():
    """Expose metrics in Prometheus format"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
