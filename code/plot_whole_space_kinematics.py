"""Connected spatial, local-path and whole-space views for Figure 7."""
from __future__ import annotations

import csv
import json

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Circle, ConnectionPatch, Rectangle

from figure_artwork import axes_mm, fs
from plot_unloading_results import field_grid, field_palette

SIZE = (170, 182)
LOCATIONS = (
    ("final_maximum_closure", "A", fs.BLACK, "o", "-"),
    ("final_maximum_slip", "B", fs.RED, "s", "--"),
    ("early_negative_lobe", "C", fs.BLUE, "^", "-."),
)
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def numeric_axis(fig, box, xlabel, ylabel, *, small=False):
    ax = axes_mm(fig, box)
    fs.style_axes(ax)
    ax.set_xlabel(xlabel, fontsize=7.5 if small else 8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7.5 if small else 8, labelpad=2)
    ax.tick_params(labelsize=7 if small else 7.5, pad=2, length=3)
    ax.tick_params(which="minor", length=1.7)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
    return ax


def heading(fig, x, y, text, *, size=9):
    return fig.text(x / SIZE[0], y / SIZE[1], text, fontsize=size, va="bottom")


def radial_axis(ax, ymax, *, labels=True):
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 3, 10, 100], labels=["1", "3", "10", "100"])
    ax.axvspan(1, 3, color=".94", zorder=0, lw=0)
    ax.axvline(3, color=".45", ls="-.", lw=.75, zorder=1)
    if labels:
        ax.text(1.7, ymax, "Interior", ha="center", va="top", fontsize=7.5)
        ax.text(17, ymax, "Exact elastic exterior", ha="center", va="top", fontsize=7.5)


def draw(data, baseline):
    fig = fs.figure(*SIZE)
    meta = json.loads((data / "metadata.json").read_text())
    physics = json.loads((baseline / "metadata.json").read_text())["physics"]
    locations = meta["fixed_locations"]
    with (data / "fixed_sector_stress_paths.csv").open(newline="", encoding="utf-8") as stream:
        stress = list(csv.DictReader(stream))
    field = np.load(data / "kinematics_figure.npz")
    near = np.load(baseline / "radial_kinematics.npz")
    full = np.load(data / "whole_space_radial.npz")
    rg = float(field["interface_radius_over_a"])

    heading(fig, 4, 176, "(a) Fixed locations and stress paths")
    spatial = axes_mm(fig, (3, 101, 80, 73))
    spatial.set(xlim=(-4.5, 4.5), ylim=(-4.1, 4.1), aspect="equal")
    spatial.set_axis_off()
    x, y = field_grid(field)
    levels = np.array([0, 10, 20, 40, 60, 80, 120, 160, 240, 400, 700, 1100])
    cmap = field_palette()
    norm = BoundaryNorm(levels, cmap.N)
    mesh = spatial.pcolormesh(x, y, field["radial_strain"] * 1e6, cmap=cmap, norm=norm,
                              shading="flat", rasterized=True, zorder=0)
    mesh.set_gid("final-total-radial-strain-inner-and-exact-exterior")
    spatial.add_patch(Circle((0, 0), 1, fc="white", ec="black", lw=.9, zorder=3))
    spatial.add_patch(Circle((0, 0), rg, fill=False, ec="white", lw=2.1, zorder=2))
    spatial.add_patch(Circle((0, 0), rg, fill=False, ec=".25", lw=.9, ls="-.", zorder=3))
    spatial.add_patch(Circle((0, 0), 4, fill=False, ec=".55", lw=.5, zorder=1))
    for key, name, color, marker, style in LOCATIONS:
        unit = np.array(locations[key]["xy_over_a"])
        spatial.plot([unit[0], 3.9 * unit[0]], [unit[1], 3.9 * unit[1]],
                     color=color, ls=style, lw=1, path_effects=HALO, zorder=4,
                     gid=f"fixed-ray-{name}")
        spatial.plot(*unit, marker=marker, ms=4, mfc="white", mec=color, mew=.9, zorder=5)
        label_xy = {"A": (-1.35, 3.9), "B": (3.75, 1.35), "C": (-3.8, 1.95)}[name]
        spatial.text(*label_xy, rf"{name}: {locations[key]['angle_deg']:g}°", color=color,
                     fontsize=8, ha="center", va="center", path_effects=HALO, zorder=6)
    for end, label in (((.65, 0), "$x$"), ((0, .65), "$y$")):
        spatial.annotate("", end, xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", lw=.6, mutation_scale=5), zorder=5)
        spatial.text(end[0]+.08, end[1]+.04, label, fontsize=7.5)
    spatial.text(0, -.48, "$r=a$", ha="center", fontsize=7.5)
    spatial.annotate(r"$\Gamma:\ r=3a$", (-2.12, -2.12), xytext=(-4.15, -3.2),
                     fontsize=8, arrowprops=dict(arrowstyle="-", lw=.6),
                     path_effects=HALO, zorder=6)
    spatial.text(2.55, -3.25, r"$k=12$", fontsize=8, ha="center", path_effects=HALO)

    # An exact spatial window exposes the narrow B-sector peak at native resolution.
    bounds = (.91, 1.45, .14, .64)
    spatial.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                               fc="none", ec="white", lw=2.2, zorder=6))
    spatial.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                               fc="none", ec=fs.RED, lw=.75, zorder=7))
    detail = numeric_axis(fig, (62, 103, 22, 21), "$x/a$", "$y/a$", small=True)
    inset_mesh = detail.pcolormesh(x, y, field["radial_strain"] * 1e6, cmap=cmap, norm=norm,
                                  shading="flat", rasterized=True)
    detail.add_patch(Circle((0, 0), 1, fc="white", ec="black", lw=.7))
    unit = np.array(locations["final_maximum_slip"]["xy_over_a"])
    detail.plot([unit[0], 1.6 * unit[0]], [unit[1], 1.6 * unit[1]],
                color=fs.RED, ls="--", lw=.9, path_effects=HALO)
    detail.plot(*unit, "s", mfc="white", mec=fs.RED, ms=3, mew=.7)
    detail.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal", xticks=[1, 1.4], yticks=[.2, .6])
    detail.text(.96, .96, "B detail", transform=detail.transAxes, ha="right", va="top",
                fontsize=7.5, path_effects=HALO)
    fig.add_artist(ConnectionPatch(xyA=(bounds[1], bounds[2]), coordsA=spatial.transData,
                                   xyB=(0, 1), coordsB=detail.transAxes, arrowstyle="-",
                                   color=fs.RED, lw=.65, zorder=6))
    bar = fig.colorbar(mesh, cax=axes_mm(fig, (8, 95, 68, 2.1)), orientation="horizontal",
                      ticks=[0, 20, 80, 240, 1100], spacing="uniform")
    bar.set_ticklabels(["0", "20", "80", "240", "1100"])
    bar.ax.tick_params(labelsize=7, pad=1, length=2)
    bar.minorticks_off()
    bar.set_label(r"Final radial strain $\varepsilon_{rr}$ ($\mu\varepsilon$)", fontsize=8, labelpad=1)
    inset_mesh.colorbar = bar

    path = numeric_axis(fig, (104, 139, 61, 34), r"$-\sigma_n$ (MPa)", r"$\tau_j$ (MPa)")
    compression = np.linspace(0, 1.08, 101)
    resistance = physics["joint_cohesion_mpa"] + compression * np.tan(np.deg2rad(physics["joint_phi_deg"]))
    shear = np.sqrt(resistance * (resistance + 2 * physics["joint_smoothing_mpa"]))
    for sign in (1, -1):
        path.plot(compression, sign * shear, color=".55", ls=":", lw=.8)
    for key, name, color, marker, style in LOCATIONS:
        points = [row for row in stress if row["location"] == key]
        xx = np.array([-float(row["sigma_n_MPa"]) for row in points])
        yy = np.array([float(row["tau_j_MPa"]) for row in points])
        path.plot(xx, yy, color=color, ls=style, lw=1.1, marker=marker, ms=2.6,
                  mfc="white", mew=.7, gid=f"fixed-stress-path-{name}-all-13-states")
        path.plot(xx[-1], yy[-1], marker, color=color, ms=3.5)
        path.annotate("", (xx[9], yy[9]), xytext=(xx[8], yy[8]),
                      arrowprops=dict(arrowstyle="->", color=color, lw=1, mutation_scale=6))
        path.text(xx[-1]+.015, yy[-1]+(.055 if name != "C" else -.09), name,
                  color=color, fontsize=8, path_effects=HALO)
    path.set(xlim=(-.03, 1.08), ylim=(-.76, .72), xticks=[0, .5, 1], yticks=[-.5, 0, .5])
    path.text(.46, .91, r"$k=0\rightarrow12$", transform=path.transAxes, fontsize=7.5)
    path.annotate("0", (1, 0), xytext=(.91, .19), fontsize=7.5,
                  arrowprops=dict(arrowstyle="-", lw=.5))
    heading(fig, 104, 175, "Wall stress paths", size=8)

    strain = numeric_axis(fig, (104, 101, 61, 25), r"Radius $r/a$", r"$\varepsilon_{rr}$ ($\mu\varepsilon$)")
    for i, (_, name, color, _, style) in enumerate(LOCATIONS[:2]):
        strain.plot(near["radius_over_a"], near["radial_strain"][i, :, -1]*1e6,
                    color=color, ls=style, lw=1.15, label=name)
    strain.set(xlim=(1, 3), ylim=(0, 920), xticks=[1, 2, 3], yticks=[0, 400, 800])
    strain.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right", handlelength=1.7)
    heading(fig, 104, 128, "(b) Radial profiles, $k=12$", size=8.5)
    strain.annotate("Broad A response", (1.75, np.interp(1.75, near["radius_over_a"], near["radial_strain"][0,:,-1]*1e6)),
                    xytext=(1.7, 360), fontsize=7.5, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=.6, mutation_scale=5))
    fig.add_artist(ConnectionPatch(xyA=(1, 1), coordsA=detail.transAxes,
                                   xyB=(1.001, np.interp(1.001, near["radius_over_a"], near["radial_strain"][1,:,-1]*1e6)),
                                   coordsB=strain.transData, arrowstyle="->", color=fs.RED,
                                   connectionstyle="arc3,rad=-.1", lw=.7, mutation_scale=6,
                                   shrinkA=4, shrinkB=3, zorder=1))

    heading(fig, 15, 80, r"(c) Whole-space accumulation, $k=12$", size=8.5)
    cumulative = numeric_axis(fig, (15, 22, 70, 55), r"Radius $r/a$", r"$H(r)$ (mm)")
    radii = full["radius_over_a"]
    radial_axis(cumulative, .343)
    for i, (_, name, color, _, style) in enumerate(LOCATIONS[:2]):
        h = full["cumulative_mm"][i, :, -1]
        closure = full["convergence_mm"][i, -1]
        cumulative.plot(radii, h, color=color, ls=style, lw=1.25, label=name)
        cumulative.axhline(closure, color=color, ls=":", lw=.7)
        cumulative.text(4.7, closure+.007, rf"$c_{name}={closure:.4f}$ mm", fontsize=7.5,
                        color=color, va="bottom")
        at_interface = float(np.interp(rg, radii, h))
        cumulative.plot(rg, at_interface, "o" if i == 0 else "s", ms=3.6,
                        mfc="white", mec=color, mew=.8)
    cumulative.set(ylim=(-.008, .36), yticks=[0, .1, .2, .3])
    cumulative.text(.36, .18, r"$H(r)=\int_a^r\varepsilon_{rr}(s)\,\mathrm{d}s$",
                    transform=cumulative.transAxes, fontsize=8)
    cumulative.text(.36, .07, r"$H(\infty)=c=-u_r(a)$", transform=cumulative.transAxes, fontsize=8)
    cumulative.text(3.16, .012, r"$R_\Gamma$", fontsize=8, color=".35")

    heading(fig, 105, 80, "(d) Formation of the A−B difference", size=8.5)
    gap_ax = numeric_axis(fig, (105, 22, 60, 55), r"Radius $r/a$", r"$H_A(r)-H_B(r)$ (mm)")
    radial_axis(gap_ax, .244, labels=False)
    gap_ax.axhline(0, color=".55", lw=.6)
    for k, color, style in ((5, fs.BLACK, ":"), (8, fs.RED, "--"), (9, fs.BLUE, "-."), (12, fs.BLACK, "-")):
        gap = full["cumulative_mm"][0, :, k-1] - full["cumulative_mm"][1, :, k-1]
        gap_ax.plot(radii, gap, color=color, ls=style, lw=1.15, label=f"k={k}")
    gap_ax.set(ylim=(-.02, .258), yticks=[0, .05, .1, .15, .2, .25])
    gap_ax.legend(fontsize=7.2, frameon=False, loc="upper right", bbox_to_anchor=(1, .89),
                  ncol=2, columnspacing=.8, handlelength=1.4, borderpad=0, labelspacing=.4)
    final_gap = full["cumulative_mm"][0, :, -1] - full["cumulative_mm"][1, :, -1]
    crossing_index = np.flatnonzero((final_gap[:-1] < 0) & (final_gap[1:] >= 0))[0]
    crossing = float(np.interp(0, final_gap[crossing_index:crossing_index+2], radii[crossing_index:crossing_index+2]))
    gap_ax.plot(crossing, 0, "o", ms=4, mfc="white", mec=fs.BLACK, mew=.9, zorder=5)
    zoom = numeric_axis(fig, (112, 60, 20, 14), r"$r/a$", "", small=True)
    zoom.plot(radii, final_gap * 1000, color=fs.BLACK, lw=1)
    zoom.axhline(0, color=".55", lw=.55)
    zoom.plot(crossing, 0, "o", ms=3, mfc="white", mec=fs.BLACK, mew=.7)
    zoom.set(xlim=(1, 1.8), ylim=(-16, 25), xticks=[1, 1.8], yticks=[-10, 0, 20])
    zoom.tick_params(labelsize=7, pad=1)
    zoom.text(.07, .95, r"$k=12$ ($\mu$m)", transform=zoom.transAxes, va="top", fontsize=7)
    zoom.annotate(f"{crossing:.2f}", (crossing, 0), xytext=(1.55, -12), fontsize=7,
                  arrowprops=dict(arrowstyle="-", lw=.5))
    gap_ax.add_patch(Rectangle((1, -.016), .8, .041, fill=False, ec=".45", lw=.55, ls=":"))
    fig.add_artist(ConnectionPatch(xyA=(1, .025), coordsA=gap_ax.transData,
                                   xyB=(1, -16), coordsB=zoom.transData, arrowstyle="-",
                                   color=".4", lw=.55, zorder=1))

    # Keep the established background interpretation next to the total gap.
    elastic_gap = float(field["elastic_wall_mm"][0] - field["elastic_wall_mm"][1])
    total_gap = float(field["total_wall_mm"][0] - field["total_wall_mm"][1])
    residual_gap = total_gap - elastic_gap
    heading(fig, 15, 4.5,
            rf"Final $c_A-c_B$:   {elastic_gap:+.4f} (elastic)  {residual_gap:+.4f} (nonelastic)  =  {total_gap:+.4f} mm",
            size=8.5)
    bindings = dict(
        field=dict(quantity="Total excavation radial strain", step=12,
                   window="1 < r/a < 4", range_microstrain=[float(field["radial_strain"].min()*1e6), float(field["radial_strain"].max()*1e6)],
                   color_boundaries_microstrain=levels.tolist(), detail_bounds_xy_over_a=list(bounds),
                   derivative="Exact saved inner displacement derivatives and exact exterior complex-potential derivatives"),
        locations=locations, final_gap_mm=dict(total=total_gap, elastic=elastic_gap, nonelastic=residual_gap),
        final_gap_zero_radius_over_a=crossing,
    )
    return fig, SIZE, bindings
