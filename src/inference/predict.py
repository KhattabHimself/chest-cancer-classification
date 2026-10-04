import ast
import io
import json
import os
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from tensorflow import keras

from src.features.preprocessing import PREPROCESS_FNS
from src.utils.common import setup_mlflow

class Predictor:
    def __init__(self, config_path="configs/config.yaml", model_version="1"):
        # in Docker MODEL_DIR points at the exported model; locally fall back to MLflow
        model_dir = os.getenv("MODEL_DIR")
        if model_dir:
            self._load_from_dir(model_dir)
        else:
            self._load_from_mlflow(config_path, model_version)
        self.preprocess = PREPROCESS_FNS[self.backbone]

    def _load_from_dir(self, model_dir):
        with open(f"{model_dir}/meta.json") as f:
            meta = json.load(f)
        self.model = keras.models.load_model(f"{model_dir}/model.h5")
        self.backbone = meta["backbone"]
        self.image_size = tuple(meta["image_size"])
        self.idx_to_class = dict(enumerate(meta["class_names"]))

    def _load_from_mlflow(self, config_path, model_version):
        config = setup_mlflow(config_path)
        self.model_name = config["mlflow"]["registered_model_name"]
        self.model_version = model_version

        client = MlflowClient()
        version_details = client.get_model_version(name=self.model_name, version=self.model_version)
        run_id = version_details.run_id

        model_uri = f"models:/{self.model_name}/{self.model_version}"
        self.model = mlflow.tensorflow.load_model(model_uri)

        params = mlflow.get_run(run_id).data.params
        self.backbone = params["backbone"]
        self.image_size = tuple(ast.literal_eval(params["image_size"]))

        class_indices = mlflow.artifacts.load_dict(f"runs:/{run_id}/class_indices.json")
        self.idx_to_class = {v: k for k, v in class_indices.items()}

    def predict(self, image_bytes: bytes):
        # Load directly from memory using io.BytesIO
        img = keras.utils.load_img(io.BytesIO(image_bytes), target_size=self.image_size)
        x = self.preprocess(np.expand_dims(keras.utils.img_to_array(img), axis=0))

        probs = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        breakdown = {self.idx_to_class[i]: float(p) for i, p in enumerate(probs)}

        return {
            "prediction": self.idx_to_class[pred_idx],
            "confidence": float(probs[pred_idx]),
            "class_breakdown": breakdown
        }

if __name__ == "__main__":
    # Local testing logic for Step 7
    predictor = Predictor()
    with open("data/test/normal/6.png", "rb") as f:
        print(predictor.predict(f.read()))