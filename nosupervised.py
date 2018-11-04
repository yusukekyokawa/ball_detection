import os
import shutil
import numpy  as np
import cv2
from skimage import data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".png"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(curDir, file)
                print("targetpaths is")
                print(img_path)
                print("-----------------------------")
                img_path_list.append(img_path)
    return img_path_list


for path in make_img_list("trimmed_imgs/4-0"):
    img = cv2.imread(path)
    print(img.shape)
    plt.figure()
    plt.imshow(np.asarray(img))


def img_to_matrix(img):
    img_array = np.asarray(img)
    return img_array

def flatten_img(img_array):
    s = img_array.shape[0] * img_array.shape[1] * img_array.shape[2]
    img_width = img_array.reshpe(1, s)

    return img_width
def make_dataset(img_dir):
    dataset = []
    for i in make_img_list(img_dir):
        img = cv2.imread(path)

        img = img_to_matrix(img)
        img = flatten_img(img)

        dataset.append(img)

    dataset = np.array(dataset)
    return dataset
    print(dataset.shape)
    print("Dataset make done")


def pca(dataset):
    n = dataset.shape[0]
    batch_size = 30
    ipca = IncrementalPCA(n_components=100)

    for i in range(n//batch_size):
        r_dataset = ipca.partial_fit(dataset[i * batch_size:(i + 1) * batch_size])
    r_dataset = ipca.transform(dataset)
    return r_dataset
    print(r_dataset.shape)
    print("PCA done")


def kmeans(r_dataset, img_path, save_dir):
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(r_dataset)
    labels = kmeans.labels_
    print("K-means clustering done.")

    for i in range(n_clusters):
        label = np.where(labels==i)[0]

        if not os.path.exists("label" + str(i)):
            os.makedirs("label" + str(i))

        for j in label:
            img = cv2.imread(make_img_list(img_path)[j])
            fname = make_img_list(img_path)[j].split('/')[-1]
            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, img)
    print("Image placing done.")

if __name__ == "__main__":
    img_dir = "trimmed_imgs/4-0"
    save_dir = "nosupervised"
    dataset = make_dataset(img_dir)
    r_dataset = pca(dataset)
    kmeans(r_dataset, img_dir, save_dir)