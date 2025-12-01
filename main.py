import numpy as np
import gzip
import os
import urllib.request
from src.layers import Dense, Dropout
from src.activations import ReLU
from src.losses import CrossEntropyLoss
from src.model import NeuralNetwork
from src.train import Trainer
import matplotlib.pyplot as plt


# -----------------------------
# 1. Descargar dataset
# -----------------------------
def download_mnist(dataset="fashion"):
    base_url = (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        if dataset == "fashion"
        else "http://yann.lecun.com/exdb/mnist/"
    )
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    path = "./data/fashion-mnist/"
    os.makedirs(path, exist_ok=True)

    for fname in files.values():
        out_path = os.path.join(path, fname)
        if not os.path.exists(out_path):
            print(f"Descargando {fname}...")
            urllib.request.urlretrieve(base_url + fname, out_path)
        else:
            print(f"{fname} ya existe, saltando descarga.")


def load_mnist(path="./data/fashion-mnist/"):
    def load_images(filename):
        with gzip.open(os.path.join(path, filename), "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    def load_labels(filename):
        with gzip.open(os.path.join(path, filename), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    X_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    X_test = load_images("t10k-images-idx3-ubyte.gz")
    y_test = load_labels("t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test


# -----------------------------
# 2. Preparar dataset
# -----------------------------
download_mnist(dataset="fashion")
X_train, y_train, X_test, y_test = load_mnist()


# One-hot encoding
def one_hot(labels, num_classes=10):
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh


y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

# -----------------------------
# 3. Definir arquitectura
# -----------------------------
layers = [
    Dense(28 * 28, 128),
    ReLU(),
    Dropout(0.3),
    Dense(128, 64),
    ReLU(),
    Dropout(0.3),
    Dense(64, 10),
]

loss = CrossEntropyLoss()
model = NeuralNetwork(layers, loss)
trainer = Trainer(model, lr=0.1, batch_size=64, epochs=30, verbose=True)

# -----------------------------
# 4. Entrenar
# -----------------------------
history = trainer.fit(X_train, y_train_oh, X_test, y_test_oh)

# -----------------------------
# 5. Evaluar
# -----------------------------
test_loss, test_acc = trainer.evaluate(X_test, y_test_oh)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# -----------------------------
# 6. Predicciones de ejemplo
# -----------------------------
preds = model.predict(X_test)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"P: {preds[i]}, T: {y_test[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
