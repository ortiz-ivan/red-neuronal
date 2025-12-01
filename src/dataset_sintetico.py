import numpy as np


def generar_espiral(points_per_class=300, num_classes=3, noise=0.2):
    """
    Genera un dataset en forma de espiral multicapa.

    Devuelve:
    X: matriz de datos (N, 2)
    y: etiquetas (N,)
    """
    N = points_per_class * num_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=int)

    for class_idx in range(num_classes):
        ix = range(points_per_class * class_idx, points_per_class * (class_idx + 1))

        r = np.linspace(0.0, 1, points_per_class)  # radio
        t = (
            np.linspace(class_idx * 4, (class_idx + 1) * 4, points_per_class)
            + np.random.randn(points_per_class) * noise
        )  # ángulo

        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_idx

    return X, y
