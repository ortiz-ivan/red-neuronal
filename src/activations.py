import numpy as np


class ReLU:
    def __init__(self):
        self.X = None  # se guarda la entrada para el backward

    def forward(self, X):
        """
        ReLU(x) = max(0, x)
        X: (batch_size, n_features)
        """
        self.X = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        """
        d/dx ReLU(x) = 1 si x > 0, 0 si x <= 0
        grad_output: gradiente que viene de la siguiente capa
        """
        relu_grad = (self.X > 0).astype(np.float32)
        return grad_output * relu_grad


class Softmax:
    def __init__(self):
        self.out = None  # se guarda la salida para el backward

    def forward(self, X):
        """
        Softmax estable: exp(x - max(x)) / sum(exp(x - max(x)))
        X: (batch_size, num_classes)
        """
        # Estabilización numérica restando el máximo de cada fila
        shift = X - np.max(X, axis=1, keepdims=True)
        exp_scores = np.exp(shift)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.out = probs
        return probs

    def backward(self, grad_output):
        """
        Gradiente general del softmax (no combinado con cross-entropy).
        grad_output: (batch_size, num_classes)

        Para cada muestra:
            J = diag(out) - out*out^T
            dX = J @ grad_output
        """
        batch_size, num_classes = grad_output.shape
        dX = np.zeros_like(grad_output, dtype=np.float32)

        for i in range(batch_size):
            y = self.out[i].reshape(-1, 1)
            jac = np.diagflat(y) - (y @ y.T)
            dX[i] = jac @ grad_output[i]

        return dX
