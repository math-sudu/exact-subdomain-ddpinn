"""Draw spatial, angular and loading-path views of radial convergence."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle

from figure_artwork import ROOT, axes_mm, export, fs
from plot_unloading_results import STEP_STYLE

DATA = ROOT / "results/unloading"
SIZE = (137, 168)
COMPONENTS = (
    (0, fs.BLACK, "-", "o", r"Wall $c$"),
    (1, fs.BLUE, "--", "s", r"Interface $C_\Gamma$"),
    (2, fs.RED, "-.", "^", r"Annulus $I$"),
)


def curve_axes(fig, rect, title, xlabel, ylabel):
    ax = axes_mm(fig, rect)
    fs.style_axes(ax)
    ax.set_title(title, fontsize=8.5, loc="left", pad=4)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(labelsize=7.5, pad=2)
    return ax


def stage_curves(ax, x, values, every=32):
    for k in (5, 8, 12):
        color, style, marker = STEP_STYLE[k]
        ax.plot(x, values[:, k-1], color=color, linestyle=style,
                marker=marker, markerfacecolor="white", markersize=2.5,
                markeredgewidth=.7, markevery=every, linewidth=1, label=rf"$k={k}$")


def radial_figure(data=DATA):
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    with np.load(data / "radial_kinematics.npz") as z:
        radius = z["radius_over_a"]
        theta = z["angle_deg"]
        epsilon = z["radial_strain"] * 1e6
        radial_u = z["radial_displacement_mm"]
        parts = z["fixed_contributions_mm"]
        angular = [z["wall_convergence_mm"], z["interface_motion_mm"], z["annulus_integral_mm"]]
    fig = fs.figure(*SIZE)
    scheme = axes_mm(fig, (5, 123, 39, 36))
    scheme.set(xlim=(-3.8, 3.8), ylim=(-3.55, 3.85), aspect="equal")
    scheme.set_axis_off()
    scheme.set_title("(a)  Fixed radial paths", fontsize=8.5, loc="left", pad=4)
    scheme.add_patch(Circle((0, 0), 3, facecolor="#edf3f8", edgecolor="#2055c7", lw=.8))
    scheme.add_patch(Circle((0, 0), 2, fill=False, edgecolor="#c3cdd7", lw=.6, linestyle=":"))
    scheme.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="black", lw=.8))
    scheme.text(0, .1, "Opening", ha="center", fontsize=7.5)
    scheme.text(-3.15, -2.65, r"$R_\Gamma=3a$", fontsize=8, color=fs.BLUE)
    scheme.text(-2.3, .3, r"$\Omega_{\mathrm{i}}$", fontsize=9, color=fs.BLUE)
    for j, loc in enumerate(metadata["fixed_locations"].values()):
        unit = np.array(loc["xy_over_a"])
        color, letter, marker = (fs.BLACK, "A", "o") if j == 0 else (fs.RED, "B", "s")
        scheme.plot([unit[0], 3*unit[0]], [unit[1], 3*unit[1]], color=color,
                    linestyle="-" if j == 0 else "--", lw=1)
        for r in (1, 3):
            scheme.plot(*(r*unit), color=color, marker=marker, markersize=3,
                        markerfacecolor="white", linestyle="none")
        scheme.annotate("", xy=2.53*unit, xytext=3*unit,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        anchor = 1.65*unit+.6*np.array([-unit[1], unit[0]])
        scheme.annotate(letter, unit, xytext=anchor, fontsize=8.5, color=color,
                        arrowprops=dict(arrowstyle="-", lw=.55, color=color))
    scheme.text(.35, 2.55, r"$C_\Gamma=-u_r(R_\Gamma)$", fontsize=7.5, color=fs.BLUE)
    scheme.text(1.4, 1.5, "Annular\nintegral $I$", fontsize=7.5, ha="center", va="center")

    for j, left in enumerate((57, 102)):
        ax = curve_axes(fig, (left, 125, 31, 34),
                        f"({chr(98+j)})  {'A' if j == 0 else 'B'}: radial strain",
                        r"Radius $r/a$", r"$\varepsilon_{rr}$ ($\mu\varepsilon$)")
        ax.set(xlim=(1, 3), xticks=[1, 2, 3], yscale="log", ylim=(1, 1000))
        ax.set_yticks([1, 10, 100, 1000], labels=["1", "10", "100", "1000"])
        stage_curves(ax, radius, epsilon[j])
        ax.legend(loc="lower left", fontsize=7.5, handlelength=1.65, labelspacing=.15,
                  borderpad=.1)

    ax = curve_axes(fig, (15, 90, 114, 22), r"(d)  Angular contributions, $k=12$",
                    r"Angle $\theta$ (deg)", "Contribution (mm)")
    for index, color, style, marker, name in COMPONENTS:
        values = angular[index][:, -1]
        ax.plot(np.r_[theta, 360], np.r_[values, values[0]], color=color, linestyle=style,
                marker=marker, markerfacecolor="white", markersize=2.5,
                markevery=60, linewidth=1, label=name)
    ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360], ylim=(0, .34), yticks=[0, .15, .3])
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.015), fontsize=7.5, ncol=3,
              handlelength=1.8, columnspacing=1.1, borderpad=0)
    for j, loc in enumerate(metadata["fixed_locations"].values()):
        angle = loc["angle_deg"]
        color, letter = (fs.BLACK, "A") if j == 0 else (fs.RED, "B")
        ax.axvline(angle, color=color, linestyle=":", lw=.7, alpha=.6)
        ax.annotate(letter, (angle, parts[j, -1, 0]), xytext=(3 if j == 0 else -11, -10),
                    textcoords="offset points", fontsize=8, color=color)

    load = np.arange(13)/12
    for component, left, title in ((1, 15, "(e)  Interface motion"), (2, 84, "(f)  Annular integral")):
        ax = curve_axes(fig, (left, 52, 45, 25), title,
                        r"Release fraction $\lambda$", r"$C_\Gamma$ (mm)" if component == 1 else r"$I$ (mm)")
        for j, letter in enumerate(("A", "B")):
            ax.plot(load, np.r_[0, parts[j, :, component]], color=(fs.BLACK, fs.RED)[j],
                    linestyle=("-", "--")[j], marker=("o", "s")[j], markerfacecolor="white",
                    linewidth=1, markersize=2.8, label=letter)
        ax.set(xlim=(0, 1.02), xticks=[0, .5, 1], ylim=(0, .135 if component == 1 else .21))
        ax.set_yticks([0, .06, .12] if component == 1 else [0, .1, .2])
        ax.legend(loc="upper left", fontsize=7.5, handlelength=1.8, labelspacing=.2)

    gap = parts[0] - parts[1]
    fractions = gap[-1, 1:] / gap[-1, 0]
    ax = curve_axes(fig, (15, 10, 45, 27), "(g)  A minus B",
                    r"Release fraction $\lambda$", "Difference (mm)")
    for index, color, style, marker, name in COMPONENTS:
        ax.plot(load, np.r_[0, gap[:, index]], color=color, linestyle=style, marker=marker,
                markerfacecolor="white", markersize=2.8, linewidth=1, label=name)
    gap_limit = max(.135, .02*np.ceil(1.1*gap.max()/.02))
    ax.set(xlim=(0, 1.02), xticks=[0, .5, 1], ylim=(0, gap_limit),
           yticks=np.arange(0, gap_limit+.001, .06))
    ax.text(.04, .88, rf"$C_\Gamma$: {100*fractions[0]:.0f}% of final gap", transform=ax.transAxes,
            fontsize=7.5, color=fs.BLUE)
    ax.text(.04, .73, rf"$I$: {100*fractions[1]:.0f}%", transform=ax.transAxes,
            fontsize=7.5, color=fs.RED)

    # Each cumulative integral starts at the wall and retains its sign.
    integrals = radial_u - radial_u[:, :1]
    cumulative_gap = integrals[0] - integrals[1]
    ax = curve_axes(fig, (84, 10, 45, 27), "(h)  Accumulation from the wall",
                    r"Upper radius $r/a$", r"$\int_a^r(\varepsilon_{rr,A}-\varepsilon_{rr,B})\,\mathrm{d}r$ (mm)")
    stage_curves(ax, radius, cumulative_gap)
    ax.axhline(0, color=".65", linestyle=":", linewidth=.7, zorder=0)
    integral_limit = max(.055, .01*np.ceil(1.1*cumulative_gap[:, [4, 7, 11]].max()/.01))
    ax.set(xlim=(1, 3), xticks=[1, 2, 3], ylim=(-.05, integral_limit),
           yticks=np.arange(-.04, integral_limit+.001, .04))
    ax.legend(loc="upper left", fontsize=7.5, handlelength=1.65,
              labelspacing=.15, borderpad=.1)
    export(fig, "radial_kinematics", SIZE,
           "Spatial strain profiles and cumulative integrals explain how the larger near-wall strain at B coexists with greater annular and interface contributions to convergence at A; angular and loading histories retain the signed kinematic decomposition.",
           [(data / name).relative_to(ROOT).as_posix() for name in
            ("radial_kinematics.npz", "radial_contributions.csv", "metadata.json")],
           "code/plot_radial_kinematics.py",
           panel_map={"a": "Annulus and fixed A/B rays with inward interface motion",
                      "b": "Full radial strain along A at k=5,8,12",
                      "c": "Full radial strain along B at k=5,8,12",
                      "d": "Final convergence, interface motion and annular integral at all wall angles",
                      "e": "A/B interface-motion histories", "f": "A/B annular-integral histories",
                      "g": "A minus B convergence and its two signed contributions",
                      "h": "A minus B integral accumulated from the wall, at k=5,8,12"},
           radial_steps=[5, 8, 12], radial_strain_scale="logarithmic, common 1-1000 microstrain",
           radial_strain_definition="Spatial derivative of the full excavation displacement, not plastic strain alone",
           samples_per_profile=720, radial_samples=257,
           contributions_order=["convergence", "interface", "annulus"],
           final_contributions_mm=parts[:, -1].tolist(), final_gap_mm=gap[-1].tolist(),
           final_gap_fractions=fractions.tolist(),
           cumulative_integral_definition="u_r(r)-u_r(a), evaluated on each fixed ray",
           schematic_arrows="Inward displacement direction; arrow lengths are illustrative",
           fixed_locations=metadata["fixed_locations"],
           reference="Current problem geometry and Origin linked field/path compositions; fixed-ray reconstruction of the manuscript radial relation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--figure-directory", type=Path)
    args = parser.parse_args()
    if args.figure_directory:
        import figure_artwork
        figure_artwork.FIGURES = args.figure_directory.resolve()
    radial_figure(args.data.resolve())
