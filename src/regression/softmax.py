"""Softmax (multiclass) helpers."""
import numpy as np


def softmax(Z):
    """Numerically stable softmax applied row-wise.

    Subtracting the row maximum does not change the result (softmax is
    shift-invariant) but prevents exp() from overflowing for large logits.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def per_sample_log_loss(P, y):
    """l_i = -log p_{i, c_i}, where c_i is the true class of sample i."""
    return -np.log(P[np.arange(len(y)), y])


def multiclass_log_loss(P, y) -> float:
    return float(per_sample_log_loss(P, y).mean())
