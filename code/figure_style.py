"""Portable plotting style for the released numerical figures."""

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

BLACK, RED, BLUE = "#000000", "#D62728", "#2055C7"
COLORS = (BLACK, RED, BLUE)
LINES = ("-", "--", "-.", ":")
MARKERS = ("o", "^", "s", "D", "v", "p")


def figure(width_mm, height_mm, **kwargs):
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "text.usetex": False, "font.size": 8.5,
        "axes.labelsize": 8.5, "axes.titlesize": 8.5, "axes.linewidth": 1.,
        "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "text.color": BLACK,
        "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.labelsize": 8., "ytick.labelsize": 8.,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "legend.fancybox": False, "legend.handlelength": 2.2,
        "lines.linewidth": 1.3, "lines.markersize": 4.,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "savefig.bbox": None, "savefig.transparent": False,
        "path.simplify": False,
    })
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), **kwargs)


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set(visible=True, linewidth=1., edgecolor=BLACK)
    ax.set_facecolor("white")
    ax.grid(False, which="both")
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=4., width=.9, colors=BLACK, pad=3)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=2., width=.7, colors=BLACK)
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "linear":
            axis.set_minor_locator(AutoMinorLocator(2))
    return ax


def series(index, *, role, replicate=0):
    color = COLORS[index % len(COLORS)]
    return dict(color=color,
                linestyle="none" if role == "measured" else LINES[(index + replicate) % 4],
                marker=MARKERS[index % len(MARKERS)] if role == "measured" else "none",
                markerfacecolor=color if replicate % 2 == 0 else "white",
                markeredgecolor=color, markeredgewidth=.7, markersize=4., linewidth=1.3)
