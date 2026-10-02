import mlflow
import optuna
from tensorflow import keras

from src.training.train import train_model, DEFAULT_PARAMS, train_and_register


def objective(trials):
    params={
        **DEFAULT_PARAMS,
        "backbone": trials.suggest_categorical("backbone", ["mobilenetv2", "efficientnetb0"]),
        "learning_rate": trials.suggest_float("learning_rate", 1e-5, 1e-2, log= True),
        "dropout": trials.suggest_float("dropout", 0.1, 0.5),
        "batch_size": trials.suggest_categorical("batch_size", [16,32]),
    }
    keras.backend.clear_session()

    #each trial is a child under the parent Optuna_search
    with mlflow.start_run(nested= True):
        mlflow.log_params(params)
        _, val_acc, history, class_indices= train_model(params)
        for metric_name, values in history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step=epoch)
        mlflow.log_metric("best_val_accuracy", val_acc)
    return val_acc

def run_search(n_trials=20):
    with mlflow.start_run(run_name= "optuna_search"):
        study= optuna.create_study(direction= "maximize")
        study.optimize(objective, n_trials= n_trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_accuracy", study.best_value)
    return study.best_params

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("chest-cancer-classification")
    best= run_search()
    print("Best Parameters:", best)
    # best only has the tuned keys (backbone, lr, dropout, batch_size),
    # so merge it on top of the defaults to get a complete params dict
    final_params = {**DEFAULT_PARAMS, **best}
    train_and_register(final_params)