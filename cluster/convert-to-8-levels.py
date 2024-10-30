import os
import sys
import time
import numpy
import cv2
import pickle

import subprocess

from sklearn.cluster import KMeans
from matplotlib import pyplot, cm

#from image_processing import convert_to_std_gray, remove_empty_borders

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        f.close()
    return model

def save_model(filename, model):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        f.close()


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


if __name__ == '__main__':

    extension = ".jpg"
    K = 12

    for line in sys.stdin:
        source_filename = line.strip()
        target_filename = source_filename[:-len(extension)] + f"-k{K}.npy"
        print(source_filename, target_filename)
        #
        img = cv2.imread(source_filename, cv2.IMREAD_UNCHANGED)

        # Provar les de 1812x1812
        # Provar les de 15kx15k
        img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA) # Deixar mida original de les imatges
        #
        print('img', img.shape, img.min(), img.max())
        cv2.imwrite(source_filename[:-len(extension)] + "-512.png", img)
        #
        # Original KMeans
        # Va agafant les imatges una darrere de l'altra
        # Carrega el RGB de la imatge i l'imprimeix per stdout
        kmeans = KMeans(n_clusters = K, init = 'k-means++', n_init = 1, max_iter = 100, tol = 1.0e-4, verbose = 1)
        kmeans.fit(img.reshape(-1, 3))
        pred = kmeans.predict(img.reshape(-1, 3))
        img2 = kmeans.cluster_centers_[pred].reshape(img.shape) / 255.
        #
        print('img2', img2.shape, img2.min(), img2.max())
        #
        img3 = rgb2gray(img2)
        #
        print('img3', img3.shape, img3.min(), img3.max())
        #
        img3 = (255 * img3).astype(numpy.uint8)
        #
        print('img3', img3.shape, img3.min(), img3.max())
        #
        grey_levels = numpy.unique(img3)
        cluster_levels = numpy.linspace(0, 255, K).astype(numpy.uint8)
        print(cluster_levels)
        img4 = img3.copy()
        for i in range(len(grey_levels)):
            #img3[img3 == grey_levels[i]] = len(grey_levels) - i - 1
            img4[img3 == grey_levels[i]] = cluster_levels[len(grey_levels) - i - 1]
        print(img.shape, img2.shape, img3.shape, img4.shape, grey_levels)
        numpy.save(file = target_filename, arr = img4)
        target_filename = target_filename[:-3] + "png"
        print(source_filename, target_filename)
        cv2.imwrite(target_filename, img4)
        '''
        img = numpy.load(target_filename)
        target_filename = target_filename[:-3] + "png"
        print(source_filename, target_filename)
        cv2.imwrite(target_filename, img)
        '''
