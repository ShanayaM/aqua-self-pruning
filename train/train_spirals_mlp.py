import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor, Adam
from nn.mlp import Linear, ReLU, Sequential
from train.dataset.spirals import generate_spiral_dataset


def train_spiral_mlp():
    np.random.seed(123)

    X, y = generate_spiral_dataset(n_samples=600, n_classes=3, noise=0.123)

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = Sequential(
        Linear(2, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 3)
    )

    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    epochs = 250
    batch_size = 32
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(X_train))
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        for start in range(0, len(X_train), batch_size):
            xb = Tensor(X_shuf[start:start + batch_size], requires_grad=False)
            yb = y_shuf[start:start + batch_size]

            logits = model(xb)
            loss = logits.softmax_loss(yb)

            if not np.isfinite(loss.data):
                raise RuntimeError(f"Training diverged (NaN/Inf loss) at epoch {epoch}!")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_logits = model(Tensor(X_train, requires_grad=False))
        val_logits = model(Tensor(X_val, requires_grad=False))

        train_loss = float(train_logits.softmax_loss(y_train).data)
        val_loss = float(val_logits.softmax_loss(y_val).data)

        train_acc = np.mean(np.argmax(train_logits.data, axis=1) == y_train)
        val_acc = np.mean(np.argmax(val_logits.data, axis=1) == y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%")

    return model, X, y, history


def plot_results(model, X, y, history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_title("Cross-Entropy Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.array(history["train_acc"]) * 100, label="Train Acc")
    axes[1].plot(np.array(history["val_acc"]) * 100, label="Val Acc", linestyle="--")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
    grid = Tensor(np.c_[xx.ravel(), yy.ravel()], requires_grad=False)
    preds = np.argmax(model(grid).data, axis=1).reshape(xx.shape)

    axes[2].contourf(xx, yy, preds, cmap="viridis", alpha=0.35)
    axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)
    axes[2].set_title("Learned Spiral Decision Boundary")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    plt.tight_layout()
    plt.savefig("part2_learning_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    model, X, y, history = train_spiral_mlp()
    plot_results(model, X, y, history)
