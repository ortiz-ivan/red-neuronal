import numpy as np


class Dense:
    def __init__(self, input_dim, output_dim):
        """
        Capa densa totalmente conectada.
        input_dim: número de características de entrada
        output_dim: número de neuronas de salida
        """

        # Inicialización Xavier (ideal para ReLU y Softmax)
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim)).astype(
            np.float32
        )
        self.b = np.zeros((1, output_dim), dtype=np.float32)

        # Espacios para almacenar datos del forward
        self.X = None

        # Gradientes
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        Forward pass: X @ W + b
        X: (batch_size, input_dim)
        return: (batch_size, output_dim)
        """
        self.X = X  # Se guarda para usar en el backward
        return X @ self.W + self.b

    def backward(self, grad_output):
        """
        Backward pass.
        grad_output: gradiente de la capa siguiente, shape (batch_size, output_dim)
        Calcula:
            dW = X^T @ grad_output
            db = sum(grad_output)
            dX = grad_output @ W^T
        """
        batch_size = self.X.shape[0]

        # Gradientes respecto a los parámetros
        self.dW = (self.X.T @ grad_output) / batch_size
        self.db = np.sum(grad_output, axis=0, keepdims=True) / batch_size

        # Gradiente respecto a la entrada
        dX = grad_output @ self.W.T

        return dX

    def update_params(self, lr):
        """
        Actualiza los parámetros usando descenso de gradiente.
        lr: learning rate
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Dropout:
    def __init__(self, p):
        """
        p: probabilidad de apagar una neurona (0 < p < 1)
        """
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, X):
        """
        Durante entrenamiento:
            mask ~ Bernoulli(1 - p)
            out = X * mask / (1 - p)
        Durante evaluación:
            out = X
        """
        if self.training:
            # Bernoulli mask
            self.mask = (np.random.rand(*X.shape) > self.p).astype(np.float32)
            return X * self.mask / (1.0 - self.p)
        else:
            return X

    def backward(self, grad_output):
        """
        El gradiente fluye solo por neuronas activas.
        """
        if self.training:
            return grad_output * self.mask / (1.0 - self.p)
        else:
            return grad_output
