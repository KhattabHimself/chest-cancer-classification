import ast
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from tensorflow import keras
from src.features.preprocessing import PREPROCESS_FNS
from src.utils.common import setup_mlflow

# 1. Initialize MLflow dynamically from your config file
config = setup_mlflow("configs/config.yaml")
MODEL_NAME = config["mlflow"]["registered_model_name"]

# Image path is typically passed via command line (argparse), 
# but defined here as a variable for testing.
IMAGE_PATH = "data/test/normal/6.png"

def run_prediction(model_version, img_path):
    # 1. Use the client to find the original run_id for this registered model
    client = MlflowClient()
    version_details = client.get_model_version(name=MODEL_NAME, version=model_version)
    run_id = version_details.run_id
    
    # 2. Load the model directly from the Model Registry
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = mlflow.tensorflow.load_model(model_uri)
    
    # 3. Load metadata from the original run
    params = mlflow.get_run(run_id).data.params
    backbone = params["backbone"]
    image_size = tuple(ast.literal_eval(params["image_size"]))
    
    class_indices = mlflow.artifacts.load_dict(f"runs:/{run_id}/class_indices.json")
    idx_to_class = {v: k for k, v in class_indices.items()}
    preprocess = PREPROCESS_FNS[backbone]

    # 4. Load and preprocess the image
    img = keras.utils.load_img(img_path, target_size=image_size)
    x = preprocess(np.expand_dims(keras.utils.img_to_array(img), axis=0))

    # 5. Predict
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    
    # 6. Output results
    print(f"Prediction: {idx_to_class[pred_idx]} ({probs[pred_idx]:.2%})")
    print("\nClass Breakdown:")
    for i, p in enumerate(probs):
        print(f"  {idx_to_class[i]}: {p:.2%}")

if __name__ == "__main__":
    # Test version 1 of your registered model
    run_prediction(model_version="1", img_path=IMAGE_PATH)