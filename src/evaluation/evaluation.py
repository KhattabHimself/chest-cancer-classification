import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from tensorflow import keras

from src.features.preprocessing import make_generator
from src.utils.common import read_yaml, setup_mlflow, logger

# 1. Initialize MLflow and load config paths
config = setup_mlflow("configs/config.yaml")

# 2. Load model parameters
params = read_yaml("params.yaml")["default"]

# Extract dynamic variables
MODEL_NAME = config["mlflow"]["registered_model_name"]


def evaluate_model(model_version):
    # Load Model from the Registry instead of a specific Run ID
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = mlflow.tensorflow.load_model(model_uri)
    
    # Use parameters dynamically loaded from yaml
    _, _, test_gen = make_generator(
        config["data"]["dir"], 
        params["backbone"], 
        params["image_size"], 
        params["batch_size"]
    )

    
    # evaluate the model on test data
    logger.info("Evaluating model...")
    test_loss, test_acc = model.evaluate(test_gen)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels and class names from the generator
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_labels)
    # print("Classification Report:\n", report)
    logger.info(f"Classification report:\n{report}")
    
    # 3. Save the confusion matrix PNG using config paths
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs(config["reports"]["figures_dir"], exist_ok=True)
    cm_path = config["reports"]["confusion_matrix"]
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion_matrix saved to {cm_path}")
    # 4. Write test metrics using config paths
    metrics = {
        "test_loss": test_loss, 
        "test_acc": test_acc
    }
    logger.info(f"Metric saved to {config['reports']['metrics']}")
    
    
    # Extract just the directory path from the full metrics filepath
    metrics_dir = os.path.dirname(config["reports"]["metrics"])
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(config["reports"]["metrics"], "w") as f:
        json.dump(metrics, f, indent=4)

    # 5. Log metrics and the PNG to MLflow
    with mlflow.start_run():
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_artifact(cm_path, artifact_path="figures")
        logger.info("Logged evaluation metrics and artifacts to MLflow")

if __name__ =="__main__":
    # Supply the version number of your registered model
    evaluate_model(model_version="1")