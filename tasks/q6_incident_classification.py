"""Q6 — Multiclass incident classification with softmax (multinomial logistic) regression."""
import _path  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from regression.data import load_xy
from regression.plotting import save
from regression.softmax import multiclass_log_loss, per_sample_log_loss

CLASSES = ["Frontend", "Backend", "Database"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def main():
    X, y = load_xy("q6_incidents.csv")
    y = y.astype(int)

    # a) train, probabilities, confusion matrix
    model = LogisticRegression().fit(X, y)  # multinomial (softmax) for >2 classes
    P = model.predict_proba(X)
    pred = P.argmax(axis=1)
    for i in range(len(y)):
        print(f"sample {i}: true={CLASSES[y[i]]:<9} P={np.round(P[i], 3)} pred={CLASSES[pred[i]]}")
    print("confusion matrix (rows=true):\n", confusion_matrix(y, pred))
    print(f"accuracy = {accuracy_score(y, pred):.3f}")

    # b) log loss
    losses = per_sample_log_loss(P, y)
    print("per-sample -log p_c:", np.round(losses, 3))
    print(f"multiclass log loss = {multiclass_log_loss(P, y):.3f}")
    i = int(losses.argmax())
    print(f"largest term: sample {i}, p_c={P[i, y[i]]:.3f}, -log p_c={losses[i]:.3f}")

    # c) coefficients
    print("W shape:", model.coef_.shape, "| intercept shape:", model.intercept_.shape)
    for k, name in enumerate(CLASSES):
        print(f"  {name:<9} w={np.round(model.coef_[k], 3)}, b={model.intercept_[k]:.3f}")

    # decision regions
    xx, yy = np.meshgrid(np.linspace(0.2, 1.9, 300), np.linspace(0, 1.9, 300))
    regions = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.contourf(xx, yy, regions, alpha=0.15, levels=[-0.5, 0.5, 1.5, 2.5], colors=COLORS)
    for k in range(3):
        ax.scatter(*X[y == k].T, color=COLORS[k], edgecolor="k", s=60, label=CLASSES[k])
    ax.set(xlabel="x1: error rate", ylabel="x2: DB latency", title="Q6 — Softmax decision regions")
    ax.legend()
    print("figure:", save(fig, "q6_regions.png"))


if __name__ == "__main__":
    main()
