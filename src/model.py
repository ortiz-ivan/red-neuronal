import numpy as np


class NeuralNetwork:
    def __init__(self, layers, loss):
        """
        layers: lista de objetos tipo Dense, ReLU, Dropout, etc.
        loss: instancia de la función de pérdida (por ejemplo CrossEntropyLoss)
        """
        self.layers = layers
        self.loss = loss

    def _set_training_mode(self, mode):
        """
        Activa/desactiva training en todas las capas que tengan ese atributo.
        Ej: Dropout(training=True/False).
        """
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = mode

    def forward(self, X):
        """
        Ejecuta el forward completo a través de todas las capas.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_loss):
        """
        Ejecuta el backward completo recorriendo las capas en orden inverso.
        """
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self, lr):
        """
        Actualiza los parámetros de todas las capas entrenables.
        """
        for layer in self.layers:
            if hasattr(layer, "update_params"):
                layer.update_params(lr)

    def train_step(self, X, y_true, lr):
        """
        Un paso completo de entrenamiento:
        - activa modo entrenamiento
        - forward
        - loss
        - backward
        - actualización de parámetros
        """
        # activar dropout
        self._set_training_mode(True)

        logits = self.forward(X)
        loss_value = self.loss.forward(logits, y_true)
        grad_loss = self.loss.backward()

        self.backward(grad_loss)
        self.update_params(lr)

        return loss_value

    def predict(self, X):
        """
        Forward en modo evaluación.
        Devuelve las clases predichas.
        """
        # desactivar dropout
        self._set_training_mode(False)

        logits = self.forward(X)

        probs = self._softmax(logits)
        return probs.argmax(axis=1)

    @staticmethod
    def _softmax(logits):
        shift = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shift)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
