import numpy as np


class Trainer:
    def __init__(self, model, lr=0.01, batch_size=32, epochs=20, verbose=True):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def _iterate_minibatches(self, X, y):
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        best_val_loss = float("inf")
        overfitting_detected = False

        for epoch in range(1, self.epochs + 1):
            epoch_losses = []

            for Xb, yb in self._iterate_minibatches(X_train, y_train):
                loss = self.model.train_step(Xb, yb, self.lr)
                epoch_losses.append(loss)

            train_loss = np.mean(epoch_losses)
            train_acc = self._accuracy(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss > best_val_loss:
                    overfitting_detected = True
                else:
                    best_val_loss = val_loss
            else:
                val_loss = None

            if self.verbose:
                print(
                    f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}",
                    end="",
                )
                if X_val is not None:
                    print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print()

        if self.verbose and overfitting_detected:
            print("Advertencia: posible overfitting detectado.")

        return history

    def evaluate(self, X, y):
        logits = self.model.forward(X)
        loss = self.model.loss.forward(logits, y)

        preds = logits.argmax(axis=1)
        acc = np.mean(preds == y)

        return loss, acc

    def _accuracy(self, X, y):
        preds = self.model.predict(X)
        return np.mean(preds == y)
