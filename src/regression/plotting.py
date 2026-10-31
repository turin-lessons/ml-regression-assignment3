"""Shared plotting helpers. Figures are saved to outputs/figures/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed: scripts save PNG files
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
