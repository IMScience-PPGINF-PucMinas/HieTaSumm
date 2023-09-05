from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from PIL import Image
import tensorflow as tf
import numpy as np


def get_layer_name_feature_extraction(model: Model) -> str:
    model_name: str = model.name
    if model_name == "vgg16":
        layer_name_feature_extraction: str = 'fc2'
    elif model_name == "resnet50":
        layer_name_feature_extraction: str = 'predictions'
    else:
        layer_name_feature_extraction: str = ''
        print("Warning: model name unknown. Default layer name: '{}'".format(layer_name_feature_extraction))
    return layer_name_feature_extraction


def extract_resnet_features(frame_path: str, model: Model) -> np.ndarray:
    # Creation of a new keras model instance without the last layer
    layer_name_feature_extraction: str = get_layer_name_feature_extraction(model)
    model_feature_vect: Model = Model(inputs=model.input, outputs=model.get_layer(layer_name_feature_extraction).output)

    # Image processing
    img_size_model: tuple = get_img_size_model(model)
    img: tf.Tensor = tf.keras.utils.load_img(frame_path, target_size=img_size_model)
    img_arr: np.ndarray = np.array(img)
    img_: np.ndarray = preprocess_input(np.expand_dims(img_arr, axis=0))

    # Visual feature extraction
    feature_vect: np.ndarray = model_feature_vect.predict(img_)

    return feature_vect


def rgb_sim_preprocess_image(frame_dir: str) -> np.ndarray:
    frame: Image.Image = Image.open(frame_dir)
    # make sure images have the same dimensions, use .resize to scale image 2 to match image 1 dimensions
    # I am also reducing the shape by half just to save some processing power
    frame_reshape: Image.Image = frame.resize((round(frame.size[0] * 0.5), round(frame.size[1] * 0.5)))
    # convert the images to (R,G,B) arrays
    frame_array: np.ndarray = np.array(frame_reshape)
    # flatten the arrays so they are 1-dimensional vectors
    frame_array: np.ndarray = frame_array.flatten()
    # divide the arrays by 255, the maximum RGB value to make sure every value is on a 0-1 scale
    frame_array: np.ndarray = frame_array / 255
    return frame_array


def get_img_size_model(model: Model) -> tuple:
    model_name: str = model.name
    if model_name == "vgg16":
        img_size_model: tuple = (224, 224)
    elif model_name == "resnet50":
        img_size_model: tuple = (224, 224)
    else:
        img_size_model: tuple = (224, 224)
        print("Warning: model name unknown. Default image size: {}".format(img_size_model))
    return img_size_model
