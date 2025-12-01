import numpy as np
import os
import struct


def load_idx_images(path):
    """Carga un archivo IDX de imágenes en un array NumPy de forma (N, 784)."""
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Archivo inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        return data.astype(np.float32)


def load_idx_labels(path):
    """Carga un archivo IDX de etiquetas en un array NumPy de forma (N,)."""
    with open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Archivo inválido: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int64)


def one_hot(labels, num_classes=10):
    """Convierte etiquetas a one-hot."""
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh


def load_fashion_mnist(root="data/fashion-mnist/"):
    """Carga y procesa Fashion-MNIST desde archivos IDX ya descomprimidos."""

    paths = {
        "train_images": os.path.join(root, "train-images-idx3-ubyte"),
        "train_labels": os.path.join(root, "train-labels-idx1-ubyte"),
        "test_images": os.path.join(root, "t10k-images-idx3-ubyte"),
        "test_labels": os.path.join(root, "t10k-labels-idx1-ubyte"),
    }

    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Archivo no encontrado: {p}")

    X_train = load_idx_images(paths["train_images"]) / 255.0
    y_train = load_idx_labels(paths["train_labels"])

    X_test = load_idx_images(paths["test_images"]) / 255.0
    y_test = load_idx_labels(paths["test_labels"])

    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    return X_train, y_train, X_test, y_test
