import yaml
import mlflow
from pathlib import Path
import logging
import sys
import os

PROJECT_ROOT= Path(__file__).resolve().parent.parent.parent
def get_logger(name="mlops-pipeline"):
    log_dir=PROJECT_ROOT / "logs"
    os.makedirs(log_dir, exist_ok= True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(module)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(name)

#Initialize Global logger
logger= get_logger()


def read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_mlflow(config_path: str = "configs/config.yaml"):
    config = read_yaml(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    return config
