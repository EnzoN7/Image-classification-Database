import os
import numpy as np
from PIL import Image
import time


def load_data(_datapath: str, _classes, _dataset: str ='train', _imagesize: int =64):
    num_images = 0
    for i in range(len(_classes)):
        try:
            dirs = sorted(os.listdir(_datapath + _dataset + '/' + _classes[i]))
            num_images += len(dirs)
        except FileNotFoundError:
            continue

    x = np.zeros((num_images, _imagesize, _imagesize, 3))
    y = np.zeros((num_images, 1))

    print("DATASET         : " + _datapath + _dataset)
    print("NOMBRE D'IMAGES : " + str(num_images) + "\n")

    current_index = 0

    # Parcours des différents répertoires pour collecter les images
    for idx_class in range(len(_classes)):
        try:
            dirs = sorted(os.listdir(_datapath + _dataset + '/' + _classes[idx_class]))
            num_images += len(dirs)
        except FileNotFoundError:
            continue

        # Chargement des images
        for idx_img in range(len(dirs)):
            item = dirs[idx_img]
            if os.path.isfile(_datapath + _dataset + '/' + _classes[idx_class] + '/' + item):
                # Ouverture de l'image
                img = Image.open(_datapath + _dataset + '/' + _classes[idx_class] + '/' + item)
                # Conversion de l'image en RGB
                img = img.convert('RGB')
                # Redimensionnement de l'image et écriture dans la variable de retour x
                img = img.resize(size=(_imagesize, _imagesize))
                x[current_index] = np.asarray(img) / 255
                # Écriture du label associé dans la variable de retour y
                y[current_index] = idx_class
                current_index += 1

    return x, y
