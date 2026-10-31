"""Q3 — Deployment failure: binary logistic regression and threshold trade-off."""
import _path  # noqa: F401
import numpy as np
from sklearn.linear_model import LogisticRegression

from regression.data import load_xy
from regression.logistic import evaluate_threshold, sigmoid


def main():
    X, y = load_xy("q3_deployment.csv")
    y = y.astype(int)

    # a) train (default L2 regularisation keeps weights finite on separable data)
    model = LogisticRegression().fit(X, y)
    b0, (b1, b2) = model.intercept_[0], model.coef_[0]
    print(f"beta0={b0:.3f}, beta1(change size)={b1:.3f}, beta2(test-failure rate)={b2:.3f}")

    x_new = np.array([1.2, 0.8])
    z = b0 + b1 * x_new[0] + b2 * x_new[1]
    print(f"z={z:.3f}, P(FAIL | [1.2, 0.8]) = {sigmoid(z):.3f}")

    # b) two thresholds
    proba = model.predict_proba(X)[:, 1]
    print("training probabilities:", np.round(proba, 3))
    for t in (0.5, 0.3):
        r = evaluate_threshold(y, proba, t)
        print(f"\nthreshold={t}\n{r['cm']}\nfalse alarms={r['fp']}, missed failures={r['fn']}, "
              f"accuracy={r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
