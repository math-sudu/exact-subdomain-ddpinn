"""Plot retained unloading histories, whole-space kinematics and section gauges."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import ConnectionPatch

import figure_artwork as art
import plot_unloading_results as unloading
from figure_artwork import ROOT, axes_mm, fs

DATA = ROOT / "results/history_analysis"
OUTPUT = ROOT / "figures"
BASELINE = ROOT / "results/unloading"
HISTORY = ROOT / "results/orientation_extension"
META = json.loads((BASELINE / "metadata.json").read_text())


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def axis(fig, rect, title, xlabel, ylabel):
    ax = axes_mm(fig, rect)
    fs.style_axes(ax)
    ax.set_title(title, loc="left", fontsize=8.5, pad=5)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(labelsize=7.5, pad=2)
    return ax


def export(fig, name, size, thesis, sources):
    art.export(fig, name, size, thesis, sources, "code/plot_history_analysis.py")


def unloading_evolution():
    size = (180, 133)
    fig = fs.figure(*size)
    h = np.load(DATA / "wall_history.npz")
    angle = h["angle_deg"]
    slip = h["accumulated_slip"][0] * 100
    release = np.arange(1, 13) / 12
    ax = axis(fig, (17, 76, 61, 39), "(a) Wall slip through unloading",
              r"Wall angle $\theta$ (deg)", r"Release fraction $\lambda$")
    im = ax.pcolormesh(angle, release, slip.T, shading="nearest",
                       cmap=unloading.field_palette(), norm=SymLogNorm(.001, vmin=0, vmax=.35), rasterized=True)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0, 1)
    bar = fig.colorbar(im, cax=axes_mm(fig, (81, 76, 2, 39)))
    bar.ax.tick_params(labelsize=7, pad=1)
    bar.set_ticks([0, .001, .01, .35])
    bar.set_ticklabels(["0", ".001", ".01", ".35"])
    bar.set_label(r"$s_j^p$ (%)", fontsize=8, labelpad=2)
    bar.minorticks_off()
    ax = axis(fig, (109, 76, 61, 39), "(b) Growth of signed-slip sectors",
              r"Release fraction $\lambda$", r"Peak $s_j^p$ (%)")
    branches = [r for r in rows(DATA / "slip_sectors.csv") if r["case"] == "bedding_oblique"]
    for sign, color in (("positive", fs.RED), ("negative", fs.BLUE)):
        for half in (0,):
            data = [r for r in branches if r["branch"] == f"{sign}_{half}"]
            ax.plot([float(r["load_step"])/12 for r in data],
                    [float(r["accumulated_slip"])*100 for r in data],
                    color=color, linestyle=("-", "--")[half], marker=("o", "s")[half],
                    markersize=2.8, markerfacecolor="white", linewidth=1,
                    label=("+ shear" if sign == "positive" else "− shear"))
    ax.set_xlim(.35, 1.02)
    ax.legend(fontsize=7, loc="upper left", frameon=False, handlelength=1.8)
    ax = axis(fig, (17, 17, 65, 39), "(c) Signed slip increments",
              r"Wall angle $\theta$ (deg)", r"$\Delta\gamma_j^p$ (%)")
    for k, color, style in ((5, fs.BLACK, ":"), (8, fs.RED, "--"), (9, fs.BLUE, "-."), (12, fs.BLACK, "-")):
        ax.plot(angle, h["slip_increment"][0, :, k-1]*100, color=color, linestyle=style,
                linewidth=1, label=f"k={k}")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(-.085, .085)
    ax.legend(fontsize=7, ncol=4, frameon=False, loc="upper center", columnspacing=.8, handlelength=1.5)
    ax = axis(fig, (109, 17, 61, 39), "(d) Lobe and convergence directions",
              r"Release fraction $\lambda$", r"Angle modulo $180^\circ$")
    ax.plot(release, angle[h["convergence_mm"][0].argmax(axis=0)] % 180,
            color=fs.BLACK, marker="o", markersize=3, markerfacecolor="white", label="Convergence peak")
    for sign, color, label in (("positive", fs.RED, "+ shear lobe"), ("negative", fs.BLUE, "− shear lobe")):
        data = [r for r in branches if r["branch"] == f"{sign}_0"]
        ax.plot([float(r["load_step"])/12 for r in data], [float(r["angle_deg"]) for r in data],
                color=color, linestyle="--", marker="s", markersize=3, markerfacecolor="white", label=label)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0, .4))
    export(fig, "unloading_evolution", size,
           "Opposite slip sectors approach similar peak activity while the convergence direction changes little.",
           ["results/history_analysis/wall_history.npz", "results/history_analysis/slip_sectors.csv"])


def whole_space_kinematics():
    size = (180, 130)
    fig = fs.figure(*size)
    stress = rows(DATA / "fixed_sector_stress_paths.csv")
    locations = json.loads((DATA / "metadata.json").read_text())["fixed_locations"]
    ax = axis(fig, (17, 76, 64, 40), "(a) Stress paths at fixed wall locations",
              r"Normal compression $-\sigma_n$ (MPa)", r"$\tau_j$ (MPa)")
    for name, color, style, label in (
        ("early_negative_lobe", fs.BLUE, "--", "C"),
        ("final_maximum_slip", fs.RED, "-", "B"),
        ("final_maximum_closure", fs.BLACK, "-.", "A")):
        points = [r for r in stress if r["location"] == name]
        ax.plot([-float(r["sigma_n_MPa"]) for r in points], [float(r["tau_j_MPa"]) for r in points],
                color=color, linestyle=style, marker="o", markersize=2.5,
                markerfacecolor="white", linewidth=1,
                label=rf"{label}: ${locations[name]['angle_deg']:g}^\circ$")
    compression = np.linspace(0, 1.1, 80)
    strength = .2 + compression*np.tan(np.deg2rad(25))
    shear = np.sqrt(strength*(strength+.02))
    ax.plot(compression, shear, ":", color=".6", linewidth=.7)
    ax.plot(compression, -shear, ":", color=".6", linewidth=.7)
    ax.legend(fontsize=7, frameon=False, loc="center left")
    inner = np.load(BASELINE / "radial_kinematics.npz")
    ax = axis(fig, (108, 76, 62, 40), "(b) Near-field radial strain",
              r"Radius $r/a$", r"$\varepsilon_{rr}$ ($\mu\varepsilon$)")
    for i, color, style, name in ((0, fs.BLACK, "-", "A"), (1, fs.RED, "--", "B")):
        ax.plot(inner["radius_over_a"], inner["radial_strain"][i, :, -1]*1e6,
                color=color, linestyle=style, linewidth=1, label=name)
    ax.set_xlim(1, 3)
    ax.legend(fontsize=8, frameon=False)
    full = np.load(DATA / "whole_space_radial.npz")
    r = full["radius_over_a"]
    ax = axis(fig, (17, 16, 64, 40), "(c) Accumulation through the exterior",
              r"Radius $r/a$", r"Cumulative $H(r)$ (mm)")
    for i, color, style, name in ((0, fs.BLACK, "-", "A"), (1, fs.RED, "--", "B")):
        ax.plot(r, full["cumulative_mm"][i, :, -1], color=color, linestyle=style, linewidth=1.2, label=name)
        ax.axhline(full["convergence_mm"][i, -1], color=color, linestyle=":", linewidth=.7)
    ax.axvline(3, color=".5", linestyle="-.", linewidth=.7)
    ax.text(3.3, .012, r"$R_\Gamma$", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 3, 10, 100], labels=["1", "3", "10", "100"])
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axis(fig, (108, 16, 62, 40), "(d) Spatial formation of the A−B gap",
              r"Radius $r/a$", r"$H_A(r)-H_B(r)$ (mm)")
    for k, color, style in ((5, fs.BLACK, ":"), (8, fs.RED, "--"), (9, fs.BLUE, "-."), (12, fs.BLACK, "-")):
        gap = full["cumulative_mm"][0, :, k-1]-full["cumulative_mm"][1, :, k-1]
        ax.plot(r, gap, color=color, linestyle=style, linewidth=1, label=f"k={k}")
    ax.axhline(0, color=".6", linewidth=.5)
    ax.axvline(3, color=".5", linestyle="-.", linewidth=.7)
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 3, 10, 100], labels=["1", "3", "10", "100"])
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    export(fig, "whole_space_kinematics", size,
           "Fixed stress paths and the full radial field link local sliding to distributed convergence.",
           ["results/history_analysis/fixed_sector_stress_paths.csv", "results/history_analysis/whole_space_radial.npz",
            "results/unloading/radial_kinematics.npz"])


def orientation_gauges():
    size = (180, 137)
    fig = fs.figure(*size)
    h = np.load(DATA / "wall_history.npz")
    cases = ["bedding_oblique", "bedding_aligned", "bedding_crossed"]
    labels = ["Oblique", "Parallel", "Perpendicular"]
    grid = np.load(BASELINE / "fields.npz")
    norm = SymLogNorm(.001, vmin=0, vmax=.35)
    cmap = unloading.field_palette()
    inset_meshes = []
    for i, (case, label) in enumerate(zip(cases, labels)):
        x = 15 + i*58
        ax = axis(fig, (x, 85, 44, 38), f"({chr(97+i)}) {label}",
                  r"Gauge angle $\theta$ (deg)", r"$D$ (mm)" if i == 0 else "")
        for k, color, style in ((5, fs.BLACK, ":"), (8, fs.RED, "--"), (12, fs.BLUE, "-")):
            ax.plot(h["angle_deg"][:360], h["diametric_convergence_mm"][i, :, k-1],
                    color=color, linestyle=style, linewidth=1, label=f"k={k}")
        slip_angle = h["angle_deg"][h["accumulated_slip"][i, :, -1].argmax()] % 180
        ax.axvline(slip_angle, color=".5", linestyle="-.", linewidth=.7)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, .65)
        ax.set_xticks([0, 90, 180])
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        source = np.load(DATA / "sources" / f"{case}.npz")
        ax = axis(fig, (x, 23, 44, 40), f"({chr(100+i)}) Vertical plastic sources",
                  r"Release fraction $\lambda$", r"$D^p_{90}$ (mm)" if i == 0 else "")
        release = np.r_[0, source["steps"]/12]
        for j, color, style, region in ((0, fs.RED, "-", r"$\Omega_{\mathrm{S}}$: sides"),
                                       (1, fs.BLUE, "--", r"$\Omega_{\mathrm{T}}$: top/bottom")):
            ax.plot(release, np.r_[0, source["common_region_source_total_mm"][1, :, j]],
                    color=color, linestyle=style, linewidth=1, label=region)
        ax.plot(release, np.r_[0, source["source_total_mm"][1]], color=fs.BLACK,
                linestyle=":", linewidth=1, label="Sum")
        ax.plot(source["steps"]/12, source["retained_mm"][1]-source["elastic_mm"][1],
                color=fs.BLACK, linestyle="none", marker="o", markersize=2.5,
                markerfacecolor="white", label=r"$D_{90}-D^{el}_{90}$")
        ax.set(xlim=(0, 1.02), ylim=(-.004, .13), xticks=[0, .5, 1])
        if i == 0:
            fig.legend(*ax.get_legend_handles_labels(), fontsize=7.5, frameon=False,
                       loc="center", bbox_to_anchor=(.52, 73/size[1]), ncol=4,
                       handlelength=2, columnspacing=2, borderpad=0)
        with np.load(HISTORY / case / "fields.npz") as f:
            np.testing.assert_allclose(f["radius_over_a"], grid["radius_over_a"], atol=1e-12)
            np.testing.assert_allclose(np.deg2rad(f["angle_deg"]), grid["angle_rad"], atol=1e-12)
            values = f["accumulated_slip_strain"][:, :, -1]
        inset = axes_mm(fig, (x+.8, 40, 22, 22))
        inset_meshes.append(unloading.field_axes(inset, grid, values*100, norm, cmap, ""))
        inset.set(xlabel="", ylabel="", xticks=[], yticks=[])
        inset.set_axis_off()
        for direction, color, style in ((45, fs.BLACK, "-"), ((90, 225, 135)[i], fs.RED, "--")):
            vector = .8*np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
            inset.plot([-vector[0], vector[0]], [-vector[1], vector[1]],
                       color=color, linestyle=style, linewidth=1)
        for direction in (45, 135, 225, 315):
            vector = np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
            inset.plot([vector[0], 3*vector[0]], [vector[1], 3*vector[1]],
                       color=".5", linestyle=":", linewidth=.55)
        for j, point, region, color, bend in (
            (0, (2.35, 0), r"$\Omega_{\mathrm{S}}$", fs.RED, -.12),
            (1, (0, -2.35), r"$\Omega_{\mathrm{T}}$", fs.BLUE, .12),
        ):
            inset.text(*point, region, ha="center", va="center", fontsize=7.5, color=color,
                       bbox=dict(fc="white", ec="none", pad=.05))
            arrow = ConnectionPatch(
                xyA=point, coordsA=inset.transData,
                xyB=(source["steps"][-2]/12, source["common_region_source_total_mm"][1, -2, j]),
                coordsB=ax.transData, arrowstyle="-|>", connectionstyle=f"arc3,rad={bend}",
                color=color, linewidth=.7, mutation_scale=7, shrinkA=7, shrinkB=3,
            )
            fig.add_artist(arrow)
    bar = fig.colorbar(inset_meshes[0], cax=axes_mm(fig, (60, 8, 70, 2)), orientation="horizontal")
    bar.set_ticks([0, .001, .01, .35], labels=["0", "0.001", "0.01", "0.35"])
    bar.minorticks_off()
    bar.set_label(r"Final accumulated slip $s_j^p$ (%)", fontsize=8, labelpad=1)
    bar.ax.tick_params(labelsize=7, pad=1, length=2)
    for mesh in inset_meshes[1:]:
        mesh.colorbar = bar
    export(fig, "orientation_gauges", size,
           "Fixed regional sources connect orientation-dependent slip histories to the same vertical gauge.",
           ["results/history_analysis/wall_history.npz"] +
           [(HISTORY / case / "fields.npz").relative_to(ROOT).as_posix() for case in cases] +
           [f"results/history_analysis/sources/{case}.npz" for case in cases])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "unloading_evolution", "whole_space_kinematics", "orientation_gauges"), default="all")
    args = parser.parse_args()
    art.FIGURES = OUTPUT
    OUTPUT.mkdir(exist_ok=True)
    for name in ("unloading_evolution", "whole_space_kinematics", "orientation_gauges"):
        if args.figure in ("all", name):
            globals()[name]()
