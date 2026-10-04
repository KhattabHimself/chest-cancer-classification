import json
import ast
import os
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.common import setup_mlflow


def export_model(version="1", out_dir="models/production"):
    # pull the registered model + what it needs at inference time out of MLflow
    config = setup_mlflow()
    name = config["mlflow"]["registered_model_name"]
    run_id = MlflowClient().get_model_version(name=name, version=version).run_id

    model = mlflow.tensorflow.load_model(f"models:/{name}/{version}")
    params = mlflow.get_run(run_id).data.params
    class_indices = mlflow.artifacts.load_dict(f"runs:/{run_id}/class_indices.json")

    os.makedirs(out_dir, exist_ok=True)
    # .h5, not .keras: Keras 2.15 on Windows writes .keras weight paths with
    # backslashes, which then fail to load in the Linux container
    model.save(f"{out_dir}/model.h5")

    # class names ordered by index, so JSON's string keys don't bite us
    meta = {
        "model_name": name,
        "model_version": version,
        "run_id": run_id,
        "backbone": params["backbone"],
        "image_size": list(ast.literal_eval(params["image_size"])),
        "class_names": sorted(class_indices, key=class_indices.get),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Exported {name} v{version} to {out_dir}")


if __name__ == "__main__":
    export_model()
