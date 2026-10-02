from src.features.preprocessing import make_generator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
import mlflow


BACKBONES = {"mobilenetv2": MobileNetV2, "efficientnetb0": EfficientNetB0}


def build_model(params):
    # pretrained backbone without its ImageNet head, frozen so we only train our head
    base = BACKBONES[params["backbone"]](
        input_shape=(*params["image_size"], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    # our own head: feature map -> vector -> dropout -> class probabilities
    model = keras.Sequential([
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(params["num_classes"], activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(params):
    # 1. data: preprocessing matches the chosen backbone
    train_gen, val_gen, _ = make_generator(
        params["data_dir"], params["backbone"], params["image_size"], params["batch_size"]
    )

    # 2. model
    model = build_model(params)

    # 3. stop when val_loss stops improving and keep the best epoch's weights
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # 4. train
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params["epochs"],
        callbacks=[early_stop],
    )

    # 5. return the model and its best validation accuracy (tune.py uses the score)
    best_val_acc = max(hist.history["val_accuracy"])
    return model, best_val_acc, hist.history, train_gen.class_indices


# default settings for a single run (will move to params.yaml later)
DEFAULT_PARAMS = {
    "data_dir": "data",
    "backbone": "mobilenetv2",
    "image_size": (224, 224),
    "num_classes": 4,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "dropout": 0.2,
    "epochs": 20,
}

def train_and_register (params):
    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params(params)

        # Unpacked class_indices from train_model
        model, val_acc, history, class_indices= train_model(params)


        for metric_name, values in history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric_name, value, step= epoch)
        mlflow.log_metric("best_val_accuracy", val_acc)
        #Logged the dictionary as an MLflow artifact
        mlflow.log_dict(class_indices, "class_indices.json")
        mlflow.tensorflow.log_model(
            model,
            name="model",
            registered_model_name= "chest-cancer-classifier",
        )
    return model

if __name__ == "__main__":#
    model, val_acc, __ = train_model(DEFAULT_PARAMS)
    print(f"Best validation accuracy: {val_acc:.2%}")