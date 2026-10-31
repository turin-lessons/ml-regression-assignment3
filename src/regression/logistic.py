"""Binary logistic regression helpers."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y, p) -> float:
    """l(y, p) = -[y log p + (1 - y) log(1 - p)]"""
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradient(x, y, p):
    """Gradient of the logistic loss w.r.t. beta for one sample: (p - y) x"""
    return (p - y) * x


def gd_step(beta, x, y, lr):
    """One gradient-descent update. Returns (new_beta, info dict before/after)."""
    z0 = beta @ x
    p0 = sigmoid(z0)
    g = gradient(x, y, p0)
    new_beta = beta - lr * g
    z1 = new_beta @ x
    p1 = sigmoid(z1)
    info = {
        "z0": z0, "p0": p0, "loss0": binary_cross_entropy(y, p0),
        "grad": g,
        "z1": z1, "p1": p1, "loss1": binary_cross_entropy(y, p1),
    }
    return new_beta, info


def evaluate_threshold(y_true, proba, threshold):
    """Turn probabilities into labels at a threshold and score them."""
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"cm": cm, "fp": int(fp), "fn": int(fn), "accuracy": accuracy_score(y_true, pred)}
