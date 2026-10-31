"""Linear regression helpers: least squares, feature maps, metrics, gradient descent."""
import numpy as np


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """phi(x) = [1, x, x^2, ..., x^degree] for each sample (design matrix)."""
    return np.column_stack([x**d for d in range(degree + 1)])


def fit_least_squares(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve min ||y - Phi w||^2. Equivalent to w = (Phi^T Phi)^-1 Phi^T y."""
    return np.linalg.lstsq(Phi, y, rcond=None)[0]


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean((y - y_hat) ** 2))


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ---- Gradient descent for y_hat = beta * x (no intercept) ----

def loss_no_intercept(beta: float, x: np.ndarray, y: np.ndarray) -> float:
    """L(beta) = 1/n * sum (y_i - beta x_i)^2"""
    return float(np.mean((y - beta * x) ** 2))


def grad_no_intercept(beta: float, x: np.ndarray, y: np.ndarray) -> float:
    """dL/dbeta = -2/n * sum x_i (y_i - beta x_i)"""
    return float(-2 / len(x) * np.sum(x * (y - beta * x)))


def stability_bound(x: np.ndarray) -> float:
    """GD on this quadratic loss converges iff alpha < n / sum(x^2)."""
    return float(len(x) / np.sum(x**2))


def gradient_descent(x, y, alpha, beta=0.0, iters=50, tol=1e-8):
    """Returns (final beta, list of losses per iteration)."""
    history = [loss_no_intercept(beta, x, y)]
    for _ in range(iters):
        beta -= alpha * grad_no_intercept(beta, x, y)
        history.append(loss_no_intercept(beta, x, y))
        if abs(history[-2] - history[-1]) < tol:
            break
    return beta, history
