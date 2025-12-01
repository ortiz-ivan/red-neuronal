import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None  # salida del softmax
        self.y_true = None  # etiquetas one-hot

    def forward(self, logits, y_true):
        """
        logits: salida lineal (NO aplicar softmax aquí)
        y_true: one-hot (batch_size, num_classes)
        Devuelve el promedio de la pérdida en el batch.
        """

        # --- Softmax estable ---
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shift)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.y_true = y_true

        # Evitar log(0)
        eps = 1e-12
        correct_logprobs = -np.sum(y_true * np.log(self.probs + eps), axis=1)

        return np.mean(correct_logprobs)

    def backward(self):
        """
        Gradiente de la función softmax + cross-entropy.
        dX = (probs - y_true) / batch_size
        """
        batch_size = self.y_true.shape[0]
        return (self.probs - self.y_true) / batch_size
