import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
from list_to_pictures import list_pictures
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
import pandas as pd
import matplotlib.pyplot as plt
# グリッドサーチCVの導入
from sklearn.model_selection import GridSearchCV
# svmのインポート
from sklearn import svm


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".jpg"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(curDir, file)
                img_path_list.append(img_path)
    return img_path_list


def Mynet(img_height, img_width, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))
    return model


if __name__ == "__main__":
    NOT_BALL_PATH = "./trimmed_imgs/4-0"
    BALL_PATH = "./ball1"
    NOT_BALL = 0
    BALL = 1
    X = []
    Y = []
    for path in make_img_list(NOT_BALL_PATH):
        img = load_img(path)
        array = img_to_array(img)
        array /= 255
        X.append(array)
        Y.append(NOT_BALL)
    for path in make_img_list(BALL_PATH):
        img = load_img(path)
        array = img_to_array(img)
        array /= 255
        X.append(array)
        Y.append(BALL)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X.shape)
    print(X[0].shape)
    Y = np_utils.to_categorical(Y, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    input_shape = X_train[0].shape

    print(X_train.shape)
    print(input_shape)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, batch_size=5, epochs=100, validation_data=(X_test, y_test), verbose = 0)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()
    # テストデータに適用
    predict_classes = model.predict_classes(X_test)
    # マージ。yのデータは元に戻す
    mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(y_test, axis=1)})
    # confusion matrix
    pd.crosstab(mg_df['class'], mg_df['predict'])


