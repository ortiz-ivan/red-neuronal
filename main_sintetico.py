import numpy as np
import matplotlib.pyplot as plt
from src.layers import Dense, Dropout
from src.activations import ReLU
from src.losses import CrossEntropyLoss
from src.model import NeuralNetwork
from src.train import Trainer
from src.dataset_sintetico import generar_espiral

# -----------------------------
# 1. Generar dataset sintético
# -----------------------------
X, y = generar_espiral(points_per_class=300, num_classes=3)

# Dividir en train / validation (80/20)
num_train = int(0.8 * X.shape[0])
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

train_idx = indices[:num_train]
val_idx = indices[num_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# Convertir etiquetas a one-hot
def one_hot(labels, num_classes):
    oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh

y_train_oh = one_hot(y_train, 3)
y_val_oh = one_hot(y_val, 3)

# -----------------------------
# 2. Definir arquitectura
# -----------------------------
layers = [
    Dense(2, 64),
    ReLU(),
    Dropout(0.3),
    Dense(64, 32),
    ReLU(),
    Dropout(0.3),
    Dense(32, 3)  # salida: 3 clases
]

loss = CrossEntropyLoss()
model = NeuralNetwork(layers=layers, loss=loss)

# -----------------------------
# 3. Configurar entrenador
# -----------------------------
trainer = Trainer(model, lr=0.1, batch_size=32, epochs=50, verbose=True)

# -----------------------------
# 4. Entrenar
# -----------------------------
history = trainer.fit(X_train, y_train_oh, X_val, y_val_oh)

# -----------------------------
# 5. Evaluar en validación
# -----------------------------
val_loss, val_acc = trainer.evaluate(X_val, y_val_oh)
print(f"\nEvaluación final en validación -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# -----------------------------
# 6. Predicciones
# -----------------------------
preds = model.predict(X_val)
print("Ejemplo de predicciones:", preds[:10])
print("Etiquetas reales:", y_val[:10])

# -----------------------------
# 7. Graficar frontera de decisión
# -----------------------------
def plot_decision_boundary(model, X, y, h=0.01):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("Frontera de decisión de la red neuronal")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

plot_decision_boundary(model, X, y)
