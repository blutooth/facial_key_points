from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, Adagrad, SGD#, Nadam
from keras.regularizers import l2
from keras import backend as K
from keras.models import load_model
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import convert_kernel
from sklearn.metrics import mean_squared_error
import keras.backend as K
import numpy as np
import data_prep
from matplotlib import pyplot
from keras.callbacks import EarlyStopping

# scale by 48 so same as challenge loss, cos scaled y by y/48
def RMSE(y_true, y_pred):
    return 48*K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_model_conv():
    #with K.tf.device('/gpu:1'):
    model = Sequential()
    #model.add(Activation(activation=center_normalize, input_shape=(1,128,128)))
    model.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(1,96,96)))
    model.add(PReLU())
    model.add(BatchNormalization())

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(20, 3, 3, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

    model.add(Convolution2D(40, 6, 6, border_mode='same'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))



    model.add(Flatten())
    model.add(Dense(30))

    #plot(model, to_file='cnn_model.png')

    adam = Adam(lr=0.1)

    #nadam = Nadam()
    adadelta = Adadelta()
    adagrad = Adagrad()
    model.compile(optimizer=adam, loss=RMSE,metrics=['mean_squared_error'])
    return model


# simple rubbish neural net model
def get_model():
    model = Sequential()
    model.add(Dense(1000, input_dim=9216))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(30))
    model.add(Activation('R'))

    adadelta = Adadelta()
    model.compile(optimizer=adadelta, loss=RMSE,metrics=['accuracy'])
    return model

model = get_model_conv()
X,y = data_prep.load2d()

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X,y,validation_split=0.2,nb_epoch=10,callbacks=[early_stopping])

save =True; load=False
# save model
if save:
    model.save('model_weights')
if load:
    model=load_model('model_weights')

X_test,blah = data_prep.load2d(test=True)
y_pred =model.predict(X_test,batch_size=64)
print(y_pred)


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

plot=True
if plot:

    y_pred = model.predict(X_test)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X_test[i], y_pred[i], ax)

    pyplot.show()
