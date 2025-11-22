from tensorflow.keras import layers,Input
from tensorflow.keras import backend as K
import tensorflow as tf
from keras import Model
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def Attention_Inception_Bi_LSTM():
    inputs = Input(shape=(200,200,3))
    c = layers.Conv2D(32,(3,3),(1,1),activation=None, padding='same')(inputs)
    c1 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c)
    ### weight 
    c2 = layers.Conv2D(64,(1,1),(1,1),activation=None, padding='same')(c1)
    c3 = layers.Conv2D(64,(3,3),(1,1),activation=None, padding='same')(c1)
    c4 = layers.Conv2D(64,(5,5),(1,1),activation=None, padding='same')(c1)
    # attention vector
    c5 = layers.Add()([c2,c3,c4])
    c6 = layers.MaxPooling2D(pool_size=(2,2),strides=(1,1), padding='same')(c5)
    c7 = layers.Conv2D(32,(3,3),(1,1),activation=None, padding='same')(c6)
    ### weight 
    c8 = layers.Conv2D(64,(1,1),(1,1),activation=None, padding='same')(c7)
    c9 = layers.Conv2D(64,(3,3),(1,1),activation=None, padding='same')(c7)
    c10 = layers.Conv2D(64,(5,5),(1,1),activation='sigmoid', padding='same')(c7)
    # attention vector
    c11 = layers.Add()([c8,c9,c10])
    c12 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same')(c11)
    c13 = layers.Reshape(target_shape=[-1, 64])(c12)
    c14 = layers.Bidirectional(layers.LSTM(128,activation='tanh'))(c13)
    model= Model(inputs, c14)


def Extract(data):
    from tensorflow.keras.models import load_model
    model = load_model("Model save/model_AttenIncBiLSTM")
    Extract_feature = model.predict(data)
    return Extract_feature

def Extracting(model,data):
    Extract_feature = model.predict(data)
    return Extract_feature

