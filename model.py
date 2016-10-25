from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, Adagrad, SGD#, Nadam
from keras.regularizers import l2
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import convert_kernel
import tensorflow as tf
import data_prep

def get_model():
    #with K.tf.device('/gpu:1'):
    model = Sequential()
    #model.add(Activation(activation=center_normalize, input_shape=(1,128,128)))

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(96,96,3), dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(16, 3, 3, border_mode='valid', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2),  dim_ordering='tf'))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='valid', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='tf'))
    model.add(PReLU())
    model.add(BatchNormalization())
    if True:
        model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Convolution2D(256, 3, 3, border_mode='same', dim_ordering='tf'))
        model.add(PReLU())
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    if False:
        model.add(Dense(1024, W_regularizer=l2(1e-4)))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    #plot(model, to_file='cnn_model.png')

    adam = Adam(lr=0.0001)

    #nadam = Nadam()
    adadelta = Adadelta()
    adagrad = Adagrad()
    model.compile(optimizer=adadelta, loss='categorical_crossentropy', metrics=['accuracy'])
    return model, 'cnn_model_5_96_shift_adadelta_BC.hdf5'


'''
# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from data_prep import load

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

X, y = load()
net1.fit(X, y)
'''