"""Implements the Keras Sequential model."""

import keras
from models import get_eye_tracker_model
from keras import backend as K
from keras import layers, models
from keras.optimizers import Adam
from keras.utils import np_utils


import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def compile_model(model, learning_rate):
    # 1e-3
    adam = Adam(lr=learning_rate) 
    model.compile(loss='mse',
                  optimizer=adam)
    return model


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                      outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )
        builder.save()


def generator_input(input_file, chunk_size, batch_size=64):
    """Generator function to produce features and labels
       needed by keras fit_generator.
    """

if __name__ == "__main__":
    model = get_eye_tracker_model(3, 64, 64)
    compile_model(model, 1e-3)