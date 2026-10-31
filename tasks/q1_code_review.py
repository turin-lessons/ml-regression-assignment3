"""Q1 — Code-review time: simple linear regression y = b0 + b1 x."""
import _path  # noqa: F401  (adds src/ to sys.path)
import matplotlib.pyplot as plt
import numpy as np

from regression.data import load_xy
from regression.linear import fit_least_squares, mse, polynomial_features, r2_score
from regression.plotting import save


def main():
    x, y = load_xy("q1_code_review.csv")

    # a) fit
    b0, b1 = fit_least_squares(polynomial_features(x, 1), y)
    print(f"beta0 = {b0:.3f}  (fixed review overhead, minutes)")
    print(f"beta1 = {b1:.3f}  (extra minutes per 100 changed lines)")

    # b) predictions + metrics
    predict = lambda v: b0 + b1 * v
    print(f"y_hat(5.5) = {predict(5.5):.3f}  [interpolation]")
    print(f"y_hat(10)  = {predict(10):.3f}  [extrapolation]")
    y_hat = predict(x)
    print(f"MSE = {mse(y, y_hat):.3f}, R^2 = {r2_score(y, y_hat):.3f}")

    xs = np.linspace(0, 10.5, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, label="observed", zorder=3)
    ax.plot(xs, predict(xs), color="tab:orange", label=f"y = {b0:.3f} + {b1:.3f}x")
    ax.scatter([5.5, 10], [predict(5.5), predict(10)], color=["green", "red"], marker="X", s=80, zorder=4)
    ax.axvspan(1, 8, color="gray", alpha=0.08, label="observed range")
    ax.set(xlabel="PR size (hundreds of lines)", ylabel="Review time (min)", title="Q1 — Review time vs PR size")
    ax.legend(); ax.grid(alpha=0.3)
    print("figure:", save(fig, "q1_fit.png"))


if __name__ == "__main__":
    main()
