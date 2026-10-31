"""Q4 — Logistic loss and one gradient-descent update, computed by hand."""
import _path  # noqa: F401
import numpy as np

from regression.data import load_json
from regression.logistic import gd_step


def main():
    cfg = load_json("q4_single_sample.json")
    x, y = np.array(cfg["x"], dtype=float), cfg["y"]
    beta0 = np.array(cfg["beta0"], dtype=float)

    beta1, info = gd_step(beta0, x, y, cfg["learning_rate"])
    print(f"before: z={info['z0']:.3f}, p={info['p0']:.3f}, loss={info['loss0']:.3f}")
    print(f"gradient (p - y) x = {np.round(info['grad'], 3)}")
    print(f"beta(1) = {np.round(beta1, 3)}")
    print(f"after:  z={info['z1']:.3f}, p={info['p1']:.3f}, loss={info['loss1']:.3f}")
    print("p moved toward y=1 and loss decreased:", info["p1"] > info["p0"] and info["loss1"] < info["loss0"])


if __name__ == "__main__":
    main()
