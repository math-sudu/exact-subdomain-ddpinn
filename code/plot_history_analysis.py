"""Plot retained unloading histories, whole-space kinematics and section gauges."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize, SymLogNorm

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


def field(fig, rect, fields, values, norm, title, quantity, ticks):
    ax = axes_mm(fig, rect)
    mesh = unloading.field_axes(ax, fields, values, norm, unloading.field_palette(), title)
    x, y, w, _ = rect
    bar = fig.colorbar(mesh, cax=axes_mm(fig, (x, y-11, w, 2)), orientation="horizontal")
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{value:g}" for value in ticks])
    bar.minorticks_off()
    bar.set_label(quantity, fontsize=8, labelpad=1)
    bar.ax.tick_params(labelsize=7, pad=1, length=2)
    return ax


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
    size = (180, 190)
    fig = fs.figure(*size)
    h = np.load(DATA / "wall_history.npz")
    cases = ["bedding_oblique", "bedding_aligned", "bedding_crossed"]
    labels = ["Oblique", "Parallel", "Perpendicular"]
    grid = np.load(BASELINE / "fields.npz")
    for i, (case, label) in enumerate(zip(cases, labels)):
        x = 15 + i*58
        with np.load(HISTORY / case / "fields.npz") as f:
            np.testing.assert_allclose(f["radius_over_a"], grid["radius_over_a"], atol=1e-12)
            np.testing.assert_allclose(np.deg2rad(f["angle_deg"]), grid["angle_rad"], atol=1e-12)
            values = f["accumulated_slip_strain"][:, :, -1]
            ax = field(fig, (x+3, 137, 38, 38), grid, values*100,
                       SymLogNorm(.001, vmin=0, vmax=.35), f"({chr(97+i)}) {label}",
                       r"$s_j^p$ (%)", [0, .001, .01, .35])
            for direction, color, style in ((45, fs.BLACK, "-"), ((90, 225, 135)[i], fs.RED, "--")):
                vector = .8*np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                ax.plot([-vector[0], vector[0]], [-vector[1], vector[1]], color=color, linestyle=style, linewidth=1)
            beta = (0, 135, 45)[i]
            for direction in (beta, beta+90, beta+180, beta+270):
                vector = np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                ax.plot([vector[0], 3*vector[0]], [vector[1], 3*vector[1]], color=".5", linestyle=":", linewidth=.65)
            for direction, sign in ((beta+45, "+"), (beta+135, "−")):
                point = 2.25*np.array([np.cos(np.deg2rad(direction)), np.sin(np.deg2rad(direction))])
                ax.text(*point, sign, ha="center", va="center", fontsize=10,
                        bbox=dict(fc="white", ec="none", pad=.1))
        ax = axis(fig, (x, 78, 44, 36), f"({chr(100+i)}) Diametric shortening",
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
        ax = axis(fig, (x, 17, 44, 36), f"({chr(103+i)}) Vertical plastic sources",
                  r"Release fraction $\lambda$", r"$D^p_{90}$ (mm)" if i == 0 else "")
        release = np.r_[0, source["steps"]/12]
        for j, color, style, region in ((0, fs.RED, "-", r"$\Omega_+$"),
                                       (1, fs.BLUE, "--", r"$\Omega_-$")):
            ax.plot(release, np.r_[0, source["region_source_total_mm"][1, :, j]],
                    color=color, linestyle=style, linewidth=1, label=region)
        ax.plot(release, np.r_[0, source["source_total_mm"][1]], color=fs.BLACK,
                linestyle=":", linewidth=1, label="Sum")
        ax.plot(source["steps"]/12, source["retained_mm"][1]-source["elastic_mm"][1],
                color=fs.BLACK, linestyle="none", marker="o", markersize=2.5,
                markerfacecolor="white", label=r"$D_{90}-D^{el}_{90}$")
        ax.set(xlim=(0, 1.02), ylim=(-.004, .13), xticks=[0, .5, 1])
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left", handlelength=1.4,
                      labelspacing=.15, borderpad=.1)
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
