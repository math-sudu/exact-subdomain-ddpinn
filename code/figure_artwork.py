"""Standalone PDF/SVG export used by the publication repository."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import figure_style as fs

ROOT = Path(__file__).resolve().parents[1]


def axes_mm(fig, rect):
    width, height = fig.get_size_inches() * 25.4
    x, y, w, h = rect
    return fig.add_axes((x / width, y / height, w / width, h / height))


def export(fig, name, size, thesis, sources, source, **bindings):
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    for extension in ("pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", bbox_inches=None,
                    facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Exported {name}: {size[0]} x {size[1]} mm PDF and SVG.")
