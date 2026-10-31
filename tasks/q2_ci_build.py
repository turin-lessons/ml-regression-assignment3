"""Q2 — CI build time: linear vs quadratic feature map, and gradient descent for y = beta x."""
import _path  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from regression.data import load_xy
from regression.linear import (fit_least_squares, gradient_descent, loss_no_intercept, mse,
                               polynomial_features, stability_bound)
from regression.plotting import save


def part_a(x, y):
    w_lin = fit_least_squares(polynomial_features(x, 1), y)
    w_quad = fit_least_squares(polynomial_features(x, 2), y)
    mse_lin = mse(y, polynomial_features(x, 1) @ w_lin)
    mse_quad = mse(y, polynomial_features(x, 2) @ w_quad)
    print(f"degree 1      w = {np.round(w_lin, 3)}  MSE = {mse_lin:.3f}")
    print(f"[1, x, x^2]   w = {np.round(w_quad, 3)}  MSE = {mse_quad:.3f}")

    xs = np.linspace(0.8, 6.2, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, color="black", zorder=3, label="observed")
    ax.plot(xs, polynomial_features(xs, 1) @ w_lin, label=f"degree 1 (MSE={mse_lin:.3f})")
    ax.plot(xs, polynomial_features(xs, 2) @ w_quad, label=f"quadratic (MSE={mse_quad:.3f})")
    ax.set(xlabel="Number of test suites", ylabel="Build time (min)", title="Q2a — Linear vs quadratic")
    ax.legend(); ax.grid(alpha=0.3)
    print("figure:", save(fig, "q2a_fits.png"))


def part_b(x, y):
    beta_star = (x @ y) / (x @ x)
    bound = stability_bound(x)
    print(f"closed-form beta* = {beta_star:.3f}, stability bound alpha < {bound:.3f}")

    alpha = 0.03
    beta, hist = gradient_descent(x, y, alpha)
    print(f"alpha={alpha}: beta={beta:.3f}, final loss={hist[-1]:.3f}, iterations={len(hist) - 1} -> converges")
    beta_bad, hist_bad = gradient_descent(x, y, 0.07, iters=15)
    print(f"alpha=0.07: beta={beta_bad:.3f}, loss={hist_bad[-1]:.3f} -> diverges")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist, marker="o", ms=3)
    ax[0].axhline(loss_no_intercept(beta_star, x, y), ls="--", color="gray", label="minimum")
    ax[0].set(title=f"Loss vs iteration (alpha={alpha})", xlabel="iteration", ylabel="L(beta)"); ax[0].legend()
    ax[1].semilogy(hist_bad, marker="o", ms=3, color="red")
    ax[1].set(title="alpha=0.07 (too large)", xlabel="iteration", ylabel="L(beta), log scale")
    for a in ax:
        a.grid(alpha=0.3)
    print("figure:", save(fig, "q2b_gd_loss.png"))


def main():
    x, y = load_xy("q2_ci_build.csv")
    print("--- Q2a ---"); part_a(x, y)
    print("--- Q2b ---"); part_b(x, y)


if __name__ == "__main__":
    main()
