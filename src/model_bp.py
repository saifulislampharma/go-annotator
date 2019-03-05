from keras.layers import (Flatten,Dense,Input,AveragePooling2D,LSTM, GlobalAveragePooling2D, Lambda, GRU,
        BatchNormalization,Conv2D,Reshape,Dropout,Bidirectional, concatenate, add, Activation, ZeroPadding2D)

from keras.models import Model
from keras.backend import int_shape
import keras.backend as K
from keras import optimizers
from keras import regularizers
from keras.layers.advanced_activations import PReLU
from . import data_preparation_bp_leve1 as data_preparations


nb_classes = data_preparations.NO_CLASSES


def build_model():
    
    b = Input(shape = (1000, 26))
    
    a = Dense(50, activation='relu')(b)
    a = Lambda(lambda x:K.expand_dims(x, axis = -1))(a)


    a1 = a
    x1 = ZeroPadding2D(padding = (3//2, 0)) (a1)
    x1 = Conv2D(64,(3, 50),padding= "valid", activation = None)(x1)
    x1 = Lambda(lambda x:K.squeeze(x, axis = 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
   
    a2 = a
    x2 = ZeroPadding2D(padding = (7//2, 0)) (a2)
    x2 = Conv2D(64,(7, 50),padding='valid', activation = None)(x2)
    x2 = Lambda(lambda x:K.squeeze(x, axis = 2))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    a3 = a
    x3 = ZeroPadding2D(padding = (11//2, 0)) (a3)
    x3 = Conv2D(64,(11, 50),padding='valid',activation = None)(x3)
    x3 = Lambda(lambda x:K.squeeze(x, axis = 2))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
  
    # concatenating 3 filters
    x_cnn = concatenate([ x1, x2, x3 ])
  

    # RNN network
    x_rnn = Bidirectional(GRU(300 ,activation='relu',return_sequences = True), merge_mode='concat')(x_cnn)
    #x_rnn = Bidirectional(GRU(300 ,activation='relu',return_sequences = True), merge_mode='concat')(x_rnn)
    

    # concatenating rnn with cnn
    x = concatenate([x_rnn, x_cnn])
    x = Lambda(lambda x: K.mean(x, axis = 2, keepdims = False))(x)
    #x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation ='sigmoid')(x)



    return Model(inputs = b, outputs = x)



## for testing models
if __name__ == "__main__":
    m = build_model()
    m.summary()
