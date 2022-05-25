import numpy as np
import matplotlib.pyplot as plt
from random import randint


def test_data(_model, _labels, _xtest, _ytest):
    idx = randint(0, len(_xtest) - 1)
    data = []
    data.append(_xtest[idx])
    data = np.asarray(data)
    dim = data[0].shape
    data = data.astype(np.float32).reshape(data.shape[0], dim[0], dim[1], dim[2])

    # Calcul des statistiques de prédiction
    prediction = _model.predict(data)
    idxBestPrediction = np.argmax(prediction)
    bestPrediction = _labels[idxBestPrediction]
    res = prediction[0][idxBestPrediction] * 100

    # Affichage des résultats
    print("PREDICTIONS sur la donnée n°" + str(idx) + "/" + str(len(_xtest) - 1))
    for i in range(0, len(_labels)):
        print('     ' + _labels[i] + ' -> ' + "{0:.2f}%".format(prediction[0][i] * 100.))
    print('\nRESULTAT : ' + bestPrediction + ' / ' + "{0:.2f}%".format(res))
    print("ATTENDU  : " + _labels[int(_ytest[idx])])
    plt.imshow(_xtest[idx])