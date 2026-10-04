from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from src.inference.predict import Predictor

# Global dictionary to hold the loaded predictor
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model into memory once
    ml_models["predictor"] = Predictor(config_path="configs/config.yaml", model_version="1")
    yield
    # Shutdown: Clear memory
    ml_models.clear()

app = FastAPI(title="Chest Cancer Classification API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read the file bytes directly from the request
    contents = await file.read()
    
    # Pass bytes to the globally loaded model instance
    result = ml_models["predictor"].predict(contents)
    
    return {
        "filename": file.filename,
        **result
    }