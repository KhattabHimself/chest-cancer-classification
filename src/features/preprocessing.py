import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet_v2, efficientnet

PREPROCESS_FNS= {
    "mobilenetv2": mobilenet_v2.preprocess_input  ,
    # "resnet50": resnet50.preprocess_input ,
    "efficientnetb0": efficientnet.preprocess_input
}

def make_generator(data_dir, backbone, image_size, batch_size, augment= True):
    preprocess= PREPROCESS_FNS[backbone]
    aug= dict(    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1) if augment else {}
    train_gen= ImageDataGenerator(preprocessing_function= preprocess , **aug).flow_from_directory(
        f"{data_dir}/train",
        batch_size= batch_size,
        target_size= image_size,
        shuffle= True,
        class_mode= 'categorical',
        seed=42,
        color_mode='rgb'
    )
    val_gen= ImageDataGenerator(preprocessing_function= preprocess).flow_from_directory(
            f"{data_dir}/valid",
            batch_size= batch_size,
            target_size= image_size,
            shuffle= False,
            class_mode= 'categorical',
            seed=42,
            color_mode='rgb'
        )
    test_gen= ImageDataGenerator(preprocessing_function= preprocess).flow_from_directory(
            f"{data_dir}/test",
            batch_size= batch_size,
            target_size= image_size,
            shuffle= False,
            class_mode= 'categorical',
            seed=42,
            color_mode='rgb'
        )
    return train_gen, val_gen, test_gen
