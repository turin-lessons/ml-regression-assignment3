"""Sanity checks for the helper functions and key assignment numbers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regression.data import load_json, load_logits, load_xy  # noqa: E402
from regression.linear import (fit_least_squares, gradient_descent, polynomial_features,  # noqa: E402
                               r2_score, stability_bound)
from regression.logistic import gd_step, sigmoid  # noqa: E402
from regression.softmax import softmax  # noqa: E402


def test_q1_coefficients():
    x, y = load_xy("q1_code_review.csv")
    b0, b1 = fit_least_squares(polynomial_features(x, 1), y)
    assert round(b0, 3) == 9.643 and round(b1, 3) == 7.690
    assert r2_score(y, b0 + b1 * x) > 0.99


def test_q2_gd_converges_below_bound_and_diverges_above():
    x, y = load_xy("q2_ci_build.csv")
    beta_star = (x @ y) / (x @ x)
    bound = stability_bound(x)
    beta, _ = gradient_descent(x, y, alpha=0.5 * bound, iters=500)
    assert beta == pytest.approx(beta_star, abs=1e-3)
    _, hist = gradient_descent(x, y, alpha=1.1 * bound, iters=20)
    assert hist[-1] > hist[0]


def test_q4_update_lowers_loss():
    cfg = load_json("q4_single_sample.json")
    _, info = gd_step(np.array(cfg["beta0"]), np.array(cfg["x"]), cfg["y"], cfg["learning_rate"])
    assert info["loss1"] < info["loss0"] and info["p1"] > info["p0"]


def test_softmax_properties():
    Z = load_logits()
    P = softmax(Z)
    assert np.allclose(P.sum(axis=1), 1) and (P >= 0).all()
    assert np.allclose(softmax(Z + 100), P)          # shift invariance
    assert np.isfinite(softmax(Z + 1000)).all()      # numerical stability


def test_sigmoid_midpoint():
    assert sigmoid(0) == 0.5
