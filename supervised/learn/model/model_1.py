from keras.layers import  Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, PReLU
from keras.models import Sequential


def my_model(NUM_CLASSES, input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.25))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model