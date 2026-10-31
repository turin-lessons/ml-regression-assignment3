"""Q5 — Softmax outputs: stable softmax, argmax check, shift invariance."""
import _path  # noqa: F401
import numpy as np

from regression.data import load_logits
from regression.softmax import softmax

CLASSES = np.array(["Frontend", "Backend", "Database"])
EXPECTED = ["Backend", "Frontend", "Database", "Frontend"]  # predicted in "Think first"


def main():
    Z = load_logits()
    P = softmax(Z)
    print("probabilities [Frontend, Backend, Database]:\n", np.round(P, 3))
    pred = CLASSES[P.argmax(axis=1)].tolist()
    print("argmax:", pred, "| matches prediction:", pred == EXPECTED)

    print("row sums:", np.round(P.sum(axis=1), 3), "| non-negative:", bool((P >= 0).all()))
    for c in (5, -3, 100):
        print(f"add c={c:>4}: max |P(Z+c) - P(Z)| = {np.abs(softmax(Z + c) - P).max():.1e}")

    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(Z[:1] + 1000) / np.exp(Z[:1] + 1000).sum()
    print("naive softmax(Z+1000):", naive, "| stable:", np.round(softmax(Z[:1] + 1000), 3))


if __name__ == "__main__":
    main()
