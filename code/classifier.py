from tensorflow.keras import layers,callbacks,regularizers,Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class squash(layers.Layer):
  def __init__(self, num_outputs):
    super(squash, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)


def Proposed(x_train,ytrain,n_class,bt=64,path=None):
    input_ = Input(shape=x_train.shape[1:])
    con = layers.Conv1D(128,3,strides=1, activation='relu')(input_)
    con = layers.Conv1D(64,5,strides=1, activation='relu')(con)
    con = layers.BatchNormalization()(con)
    con = layers.Conv1D(128,3,strides=1, activation='relu')(con)
    con = layers.BatchNormalization()(con)
    bigru = layers.Bidirectional(layers.GRU(128*7,activation="tanh"))(con)
    encoded = Dense(256*3)(bigru)
    encoded = squash(256*3)(encoded)
    encoded= layers.Dropout(0.3)(encoded)
    encoded = Dense(128*4, activation='relu')(encoded)
    encoded = squash(128*4)(encoded) 
    encoded= layers.Dropout(0.3)(encoded) 
    encoded = Dense(64*4, activation='relu')(encoded) 
    encoded = squash(64*4)(encoded) 
    encoded= layers.Dropout(0.3)(encoded) 
    encoded = Dense(32*4, activation='relu')(encoded)
    encoded = squash(32*4)(encoded)
    encoded= layers.Dropout(0.3)(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(n_class, activation='softmax')(encoded)
    model = Model(input_, encoded)
    for lay in model.layers[:7]:
        lay.trainable = False
    callback = callbacks.EarlyStopping(monitor="accuracy",patience=100,mode="auto",restore_best_weights=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, ytrain,epochs=100,batch_size=bt,verbose=False,validation_split=0.1,callbacks=[callback])
    if path==None:return model
    else:model.save(path)

def CNN(x_train,ytrain,n_class,bt=64,path=None):
    
    input_ = Input(shape=x_train.shape[1:])
    cnn = layers.Conv1D(128,3,strides=1, activation='relu')(input_)
    cnn = layers.MaxPooling1D(pool_size=2, padding="same")(cnn)
    cnn = layers.Conv1D(64,3,strides=1, activation='relu')(cnn)
    cnn = layers.MaxPooling1D(pool_size=2, padding="same")(cnn)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.Conv1D(32,3,strides=1, activation='relu')(cnn)
    cnn = layers.MaxPooling1D(pool_size=2, padding="same")(cnn)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.Flatten()(cnn)
    cnn = Dense(64, activation='relu')(cnn)
    cnn = Dense(n_class, activation='softmax')(cnn)
    model = Model(input_, cnn)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, ytrain,epochs=100,batch_size=bt,verbose=False,validation_split=0.1)
    if path==None:return model
    else:model.save(path)

def Bilstm(x_train,ytrain,n_class,bt=64,path=None):
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
    
    model = Sequential()
    model.add(layers.Embedding(x_train.shape[0]+1,128,input_length=x_train.shape[1]))
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dense(32))
    model.add(layers.Dense(n_class, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, ytrain,epochs=100,batch_size=bt,verbose=False,validation_split=0.1)
    if path==None:return model
    else:model.save(path)

def Autoencoder(x_train,ytrain,n_class,bt=64,path=None):
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
    
    model = Sequential()
    model.add(layers.Dense(256,input_dim=x_train.shape[1]))
    model.add(layers.Dense(256*2, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256*2, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_class, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, ytrain,epochs=100,batch_size=bt,verbose=False,validation_split=0.1)
    if path==None:return model
    else:model.save(path)


def Rnn(x_train,ytrain,n_class,bt=64,path=None):
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
    
    model = Sequential()
    model.add(layers.Embedding(x_train.shape[0]+1,256,input_length=x_train.shape[1]))
    model.add(layers.LSTM(128,activation="tanh"))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_class, activation='softmax'))
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train, ytrain,epochs=100,batch_size=bt,verbose=False,validation_split=0.1)
    if path==None:return model
    else:model.save(path)

def predition(model_name,ydata):
    from tensorflow.keras.models import load_model
    model = load_model(model_name)
    ypred = model.predict(ydata)
    return ypred
    