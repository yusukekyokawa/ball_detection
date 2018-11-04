from PIL import Image
import numpy as np
import os
import pandas as pd
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC


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


def img_to_matrix(path):
    img = Image.open(path)
    img_array = np.asarray(img)
    return img_array


def flatten_image(img):
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def main():
    X = []
    Y = []
    NOT_BALL_PATH = "trimmed_imgs/4-0"
    BALL_PATH = "ball1"
    NOT_BALL = 0
    BALL = 1
    for path in make_img_list(NOT_BALL_PATH):
        print(path)
        img = img_to_matrix(path)
        print(img)
        img = flatten_image(img)
        X.append(img)
        Y.append(NOT_BALL)

    for path in make_img_list(BALL_PATH):
        img = img_to_matrix(path)
        img = flatten_image(img)
        X.append(img)
        Y.append(BALL)

    X = np.array(X)
    print(X.shape)

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    print(X.shape)


