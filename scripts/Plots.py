from random import randint
import matplotlib.pyplot as plt
import time
import numpy as np


def plot_training_analysis(_history, _metrics):
    acc = _history.history[_metrics]
    val_acc = _history.history['val_' + _metrics]
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', linestyle="--",label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', linestyle="--",label='Training loss')
    plt.plot(epochs, val_loss,'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def print_false_values(_model, _labels, _xtest, _ytest):
    for idx in range(len(_xtest)):
        data = []
        data.append(_xtest[idx])
        data = np.asarray(data)
        dim = data[0].shape
        data = data.astype(np.float32).reshape(data.shape[0], dim[0], dim[1], dim[2])

        prediction = _model.predict(data)
        idxBestPrediction = np.argmax(prediction)
        bestPrediction = _labels[idxBestPrediction]

        if bestPrediction != _labels[int(_ytest[idx])]:
           res = prediction[0][idxBestPrediction] * 100
           print("PREDICTIONS sur la donnée n°" + str(idx) + "/" + str(len(_xtest) - 1))
           print('RESULTAT : ' + bestPrediction + ' / ' + "{0:.2f}%".format(res))
           print('ATTENDU  : ' +  _labels[int(_ytest[idx])] + '\n')
           time.sleep(0.1)


def plot_random_images(_xtrain, _ytrain, _labels):
    plt.figure(figsize=(9, 9))
    
    indices = [randint(0, len(_xtrain) - 1) for i in range(0, 9)]
    for i in range(0, 9):
        plt.subplot(3, 3, i+1)
        plt.title(_labels[int(_ytrain[indices[i]])])
        plt.imshow(_xtrain[indices[i]])

    plt.tight_layout()
    plt.show()


def plot_candidates(_model, _labels, _xtest):
    names = ['Anne Hidalgo', 'Emmanuel Macron', 'Eric Zemmour', 'Fabien Roussel', 'Jean Lassalle', 'Jean-Luc Mélenchon',
             'Marine Le Pen', 'Nathalie Arthaud', 'Nicolas Dupont-Aignan', 'Philippe Poutou', 'Valérie Pécresse', 'Yannick Jadot']
    plt.figure(figsize=(20, 15))
    for idx in range(len(_xtest)):
        data = []
        data.append(_xtest[idx])
        data = np.asarray(data)
        dim = data[0].shape
        data = data.astype(np.float32).reshape(data.shape[0], dim[0], dim[1], dim[2])

        prediction = _model.predict(data)
        idxBestPrediction = np.argmax(prediction)
        bestPrediction = _labels[idxBestPrediction]
        res = prediction[0][idxBestPrediction] * 100

        plt.subplot(3, 4, idx + 1)
        plt.title(names[idx] + "=" + bestPrediction + " ({0:.2f}%)".format(res))
        plt.imshow(_xtest[idx])