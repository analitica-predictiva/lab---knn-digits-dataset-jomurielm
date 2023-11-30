"""
Clasificación usando k-NN - Digits Dataset
-----------------------------------------------------------------------------------------

En este laboratio se construirá un clasificador usando k-NN para el dataset de digitos.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets


def pregunta_01():
    """
    Complete el código presentado a continuación.
    """

    # Cargue el dataset digits
    digits = ____.____()

    # Imprima los nombres de la variable target del dataset
    print(____.____)

    # Imprima las dimensinoes de matriz de datos
    print(____.____.____)

    # Imprima las dimensiones del vector de salida
    print(____.____.____)


def pregunta_02():
    """
    Complete el código presentado a continuación.
    """
    # Importe KNeighborsClassifier de sklearn.neighbors
    from ____ import ____

    # Importe train_test_split de sklearn.model_selection
    from ____ import ____

    # Cargue el dataset digits
    digits = ____.____()

    # Cree los vectors de características y de salida
    X = ____.____
    y = ____.____

    # Divida los datos de entrenamiento y prueba. Los conjuntos de datos están
    # estratificados. La semilla del generador de números aleatorios es 42.
    # El tamaño del test es del 20%
    X_train, X_test, y_train, y_test = ____(
        ____, ____, test_size=____, random_state=____, stratify=____
    )

    # Cree un clasificador con siete vecinos
    knn = ____

    # Entrene el clasificador
    ____

    # Imprima la precisión (score) del clasificador en el conjunto de datos de prueba
    print(round(knn.score(____, ____), 4))


def pregunta_03():
    """
    Complete el código presentado a continuación.
    """

    # Importe KNeighborsClassifier de sklearn.neighbors
    from ____ import ____

    # Importe train_test_split de sklearn.model_selection
    from ____ import ____

    # Cargue el dataset digits
    digits = ____.____()

    # Cree los vectors de características y de salida
    X = ____.____
    y = ____.____

    # Divida los datos de entrenamiento y prueba. Los conjuntos de datos están
    # estratificados. La semilla del generador de números aleatorios es 42.
    X_train, X_test, y_train, y_test = ____(
        ____, ____, test_size=____, random_state=____, stratify=____
    )

    # Inicialice los arreglos para almacenar la precisión para las muestras de
    # entrenamiento y de prueba
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Se itera sobre diferentes valores de vecinos
    for i, k in enumerate(neighbors):
        # Cree un clasificador con k vecinos
        knn = ____

        # Entrene el clasificador con los datos de entrenamiento
        ____

        # Calcule la precisión para el conjunto de datos de entrenamiento
        train_accuracy[i] = knn.score(____, ____)

        # Calcule la precisión para el conjunto de datos de prueba
        test_accuracy[i] = knn.score(____, ____)

    # Almacenamiento de los resultados como un dataframe
    df = pd.DataFrame(
        {
            "k": neighbors,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
    )

    return df
