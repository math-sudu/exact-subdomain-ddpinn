"""Detailed process and field figures for the exterior-operator extensions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.tri import Triangulation
from matplotlib.ticker import FuncFormatter, NullFormatter

from figure_artwork import ROOT, axes_mm, export, fs
from interface_extension import opening_boundary
from pilot import config as C


DATA = ROOT / "results/interface_extension"
SETTINGS = json.loads((ROOT / "configs/interface_extension.json").read_text(encoding="utf-8"))
STYLES = {
    "uniform": (fs.BLACK, "-", "o", "Uniform"),
    "material": (fs.RED, "--", "s", "Material"),
    "wavelength": (fs.BLUE, "-.", "^", "Wavelength"),
    "compliance": (fs.BLACK, ":", "D", "Compliance"),
}


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def results(case, metric):
    series = []
    for seed in SETTINGS["seeds"]:
        directory = DATA / case / f"seed_{seed}" / metric
        rows = read_rows(directory / "trajectory.csv")
        if rows[-1]["phase"] != "complete":
            raise RuntimeError(f"Incomplete result: {directory}")
        series.append((rows, dict(np.load(directory / "fields.npz"))))
    return series


def axis(fig, rect, title, xlabel, ylabel):
    ax = axes_mm(fig, rect)
    fs.style_axes(ax)
    ax.set_title(title, loc="left", fontsize=8.3, pad=5)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.tick_params(labelsize=7.2, pad=2)
    return ax


def paired_curves(ax, all_series, key, *, root=False):
    stop = min(float(rows[-1]["evaluations"]) for series in all_series.values() for rows, _ in series)
    work = np.linspace(0, stop, 180)
    for metric, series in all_series.items():
        color, linestyle, marker, name = STYLES[metric]
        values = []
        for rows, _ in series:
            x = np.array([float(row["evaluations"]) for row in rows])
            y = np.array([float(row[key]) for row in rows])
            y = np.sqrt(y) if root else y
            x, indices = np.unique(x, return_index=True)
            y = y[indices]
            ax.plot(x, y, color=color, alpha=.22, linewidth=.45)
            values.append(np.interp(work, x, y))
        ax.plot(work, np.median(values, axis=0), color=color, linestyle=linestyle,
                marker=marker, markerfacecolor="white", markersize=2.6,
                markevery=26, linewidth=1, label=name)
    ax.set_xlim(0, stop)
    ax.set_xticks([0, 1200, 2400, 3500])
    ax.axvline(1200, color=".65", lw=.6, linestyle=":")


def log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def interface_figure():
    size = (137, 176)
    fig = fs.figure(*size)
    cases = ("circle_base", "circle_stiff")
    all_results = {case: {metric: results(case, metric) for metric in STYLES} for case in cases}
    for col, (case, ratio) in enumerate(zip(cases, (2.625, 5.0))):
        left = 16 + 68 * col
        ax = axis(fig, (left, 136, 45, 30), f"({chr(97+col)})  $E_h/E_v={ratio:g}$",
                  "Gradient evaluations", "Maximum convergence (mm)")
        paired_curves(ax, all_results[case], "convergence_max_mm")
        ax.axhline(0, color=".75", lw=.5)
        if col == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, 1.005),
                       ncol=4, fontsize=7.5, handlelength=1.7, columnspacing=1)
        ax.text(.17, .09, "Adam", ha="center", transform=ax.transAxes, fontsize=7, color=".4")
        ax.text(.67, .09, "L-BFGS", ha="center", transform=ax.transAxes, fontsize=7, color=".4")

        ax = axis(fig, (left, 95, 45, 29), f"({chr(99+col)})" +
                  ("  Interface residual" if col == 0 else ""),
                  "Gradient evaluations", r"$\sqrt{q_C}$")
        paired_curves(ax, all_results[case], "interface_compliance", root=True)
        log_y(ax)

        ax = axis(fig, (left, 55, 45, 28), f"({chr(101+col)})" +
                  ("  Residual spectrum" if col == 0 else ""),
                  "Fourier mode $m$", r"$\|\widehat{\boldsymbol{r}}_m\|_2$ (MPa)")
        for metric, series in all_results[case].items():
            color, linestyle, marker, name = STYLES[metric]
            values = [np.linalg.norm(data["interface_coefficients"], axis=0) * C.SIGMA_REF/1e6
                      for _, data in series]
            modes = np.arange(len(values[0]))
            for v in values:
                ax.plot(modes[1:-1], v[1:-1], color=color, alpha=.20, lw=.45)
            ax.plot(modes[1:-1], np.median(values, axis=0)[1:-1], color=color,
                    linestyle=linestyle, linewidth=1, marker=marker, markersize=2.2,
                    markerfacecolor="white", markevery=[0, 3, 9, 29, 99, 200], label=name)
        ax.set(xscale="log", xlim=(1, modes[-2]))
        log_y(ax)
        ax.set_xticks([1, 10, 100, modes[-2]], labels=["1", "10", "100", str(modes[-2])])
        ax.axvline(SETTINGS["quadrature"]["boundary"]//2-1, color=".65", lw=.6, linestyle=":")

        ax = axis(fig, (left, 14, 45, 29), f"({chr(103+col)})" +
                  ("  Paired endpoints" if col == 0 else ""),
                  "Interface metric", r"Final $\sqrt{q_C}$")
        matrix = np.array([[np.sqrt(float(all_results[case][metric][j][0][-1]["interface_compliance"]))
                            for metric in STYLES] for j in range(len(SETTINGS["seeds"]))])
        for j, values in enumerate(matrix):
            ax.plot(np.arange(4), values, color=".72", lw=.65, zorder=1)
        for i, metric in enumerate(STYLES):
            color, _, marker, _ = STYLES[metric]
            ax.plot(np.full(len(matrix), i), matrix[:, i], linestyle="none", marker=marker,
                    color=color, markerfacecolor="white", markersize=3.5, markeredgewidth=.9)
        ax.set(xlim=(-.35, 3.35), xticks=range(4))
        log_y(ax)
        ax.set_xticklabels([STYLES[metric][3] for metric in STYLES], rotation=25, ha="right")
    export(fig, "interface_metric_study", size,
           "Paired initial states and equal initial interface scale isolate the material and wavelength contributions to training and residual spectra.",
           ["results/interface_extension/summary.csv", "configs/interface_extension.json"],
           "code/plot_extensions.py --figure interface",
           variability="Three paired initializations; thin lines and connected final points show every seed; thick curves are medians.")


def field_map(ax, data, case, norm):
    xy = data["xy"]
    displacement = np.linalg.norm(data["displacement"], axis=-1) * C.U_REF * 1000
    tri = Triangulation(xy[:, 0], xy[:, 1])
    centers = xy[tri.triangles].mean(1)
    alpha = np.deg2rad(case["opening_angle_deg"])
    local = centers @ np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    inside = (local[:, 0]/case["opening_axes"][0])**2 + (local[:, 1]/case["opening_axes"][1])**2 < 1
    tri.set_mask(inside)
    artist = ax.tripcolor(tri, displacement, shading="gouraud", cmap="jet", norm=norm, rasterized=True)
    theta = np.linspace(0, 2*np.pi, 721)
    wall, _, _ = opening_boundary(theta, case["opening_axes"], case["opening_angle_deg"])
    ax.fill(wall[:, 0], wall[:, 1], facecolor="white", edgecolor="black", lw=.65)
    ax.plot(3*np.cos(theta), 3*np.sin(theta), color="black", lw=.6)
    ax.plot([-2.4, -1.5], [1.5, 2.4], color="white", lw=1.2)
    ax.text(-2.8, 2.65, r"$\phi_b=45^\circ$", fontsize=7, color="black")
    ax.set(xlim=(-3.15, 3.15), ylim=(-3.15, 3.15), aspect="equal", xticks=[-3, 0, 3], yticks=[-3, 0, 3])
    return artist


def angular_profile(data, field):
    n0, n1 = int(data["n_pde"]), int(data["n_pde"])+int(data["n_wall"])
    wall_xy = data["xy"][n0:n1]
    radial = wall_xy/np.linalg.norm(wall_xy, axis=-1, keepdims=True)
    theta = np.rad2deg(np.arctan2(wall_xy[:, 1], wall_xy[:, 0])) % 360
    if field == "convergence":
        value = -(data["displacement"][n0:n1]*radial).sum(-1)*C.U_REF*1000
    elif field == "stress":
        tangent = np.stack((-data["wall_normal"][:, 1], data["wall_normal"][:, 0]), -1)
        stress = data["stress_increment"][n0:n1]+data["insitu"]
        value = (stress[:, 0]*tangent[:, 0]**2+stress[:, 1]*tangent[:, 1]**2
                 +2*stress[:, 3]*tangent[:, 0]*tangent[:, 1])*C.SIGMA_REF/1e6
    else:
        xy = data["xy"][n1:]
        theta = np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0])) % 360
        value = (data["displacement"][n1:]*xy/3).sum(-1)*C.U_REF*1000
    order = np.argsort(theta)
    return theta[order], value[order]


def geometry_figure():
    size = (137, 178)
    fig = fs.figure(*size)
    row_axes = [[], [], []]
    cases = [case for case in SETTINGS["cases"] if case["name"] in ("circle_base", "circle_stiff", "ellipse_staggered")]
    all_results = {case["name"]: results(case["name"], "compliance") for case in cases}
    limit = max(np.linalg.norm(series[0][1]["displacement"], axis=-1).max() * C.U_REF * 1000
                for series in all_results.values())
    norm = Normalize(0, np.ceil(limit*20)/20)
    meshes = []
    for col, case in enumerate(cases):
        left = 13 + 43*col
        series = all_results[case["name"]]
        title = ("Circular, $E_h/E_v=2.625$", "Circular, $E_h/E_v=5$", "Elliptical, $a_1/a_2=1.96$")[col]
        ax = axis(fig, (left, 122, 32, 32), f"({chr(97+col)})", "$x/a$", "$y/a$")
        artist = field_map(ax, series[0][1], case, norm)
        meshes.append(artist)
        ax.text(.5, 1.18, title, ha="center", transform=ax.transAxes, fontsize=7.4)
        if col == 2:
            ax.text(.5, 1.07, r"Major axis: $30^\circ$", ha="center",
                    transform=ax.transAxes, fontsize=7.1)

        for row, (bottom, ylabel, field) in enumerate(((82, "Radial convergence (mm)", "convergence"),
                                                       (47, r"$\sigma_{tt}$ (MPa)", "stress"),
                                                       (12, r"Interface $u_r$ (mm)", "interface"))):
            ax = axis(fig, (left, bottom, 32, 25), f"({chr(100+3*row+col)})" +
                      ("  " + ("Wall convergence", "Tangential stress", "Interface motion")[row]
                       if col == 0 else ""),
                      r"Angle $\theta$ (deg)" if row == 2 else "", ylabel)
            row_axes[row].append(ax)
            for j, (_, data) in enumerate(series):
                theta, value = angular_profile(data, field)
                color, style, marker = ((fs.BLACK, "-", "o"), (fs.RED, "--", "s"), (fs.BLUE, "-.", "^"))[j]
                ax.plot(theta, value, color=color, linestyle=style, linewidth=.9,
                        marker=marker, markerfacecolor="white", markersize=2.1, markevery=64,
                        label=f"Seed {SETTINGS['seeds'][j]}")
            ax.set(xlim=(0, 360), xticks=[0, 180, 360])
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 115.5/size[1]),
                           ncol=3, fontsize=7.2, handlelength=1.7, columnspacing=1)
    for group in row_axes:
        lower = min(ax.get_ylim()[0] for ax in group)
        upper = max(ax.get_ylim()[1] for ax in group)
        for ax in group:
            ax.set_ylim(lower, upper)
    cax = axes_mm(fig, (39, 168, 60, 2))
    cb = fig.colorbar(artist, cax=cax, orientation="horizontal")
    for mesh in meshes:
        mesh.colorbar = cb
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=7, pad=1)
    cb.set_label("Displacement magnitude (mm)", fontsize=7.5, labelpad=1)
    export(fig, "geometry_extension", size,
           "The same circular exterior construction couples anisotropic interiors with circular and elliptical openings; complete angular profiles reveal the regional response across three initializations.",
           ["results/interface_extension/summary.csv", "configs/interface_extension.json"],
           "code/plot_extensions.py --figure geometry",
           representative="Field maps use the first predefined initialization; all three initializations appear in the curves.")


def orientation_figure(directory, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = config["cases"]
    sources = {}
    for case in cases:
        source = directory / case["name"]
        sources[case["name"]] = dict(
            fields=dict(np.load(source / "fields.npz")),
            wall=read_rows(source / "wall_profiles.csv"),
            paths=read_rows(source / "stress_paths.csv"),
            metadata=json.loads((source / "metadata.json").read_text(encoding="utf-8")))
    size = (137, 180)
    fig = fs.figure(*size)
    limit = max(source["fields"]["accumulated_slip_strain"][:, :, -1].max()*100
                for source in sources.values())
    norm = SymLogNorm(linthresh=.001, linscale=.65, vmin=0, vmax=np.ceil(limit*10)/10)
    cmap = colormaps["turbo"].copy()
    cmap.set_bad("#f2f2f2")
    meshes = []
    convergence_limit = max(float(row["convergence_mm"]) for source in sources.values() for row in source["wall"])*1.07
    slip_limit = max(float(row["accumulated_slip_strain"])*100 for source in sources.values() for row in source["wall"])*1.07
    compression_limit = max(-float(row["sigma_n_MPa"]) for source in sources.values()
                            for row in source["paths"])*1.1
    theta = np.linspace(0, 2*np.pi, 721)
    for col, case in enumerate(cases):
        left = 12 + 43*col
        data = sources[case["name"]]
        fields, metadata = data["fields"], data["metadata"]
        ax = axis(fig, (left, 122, 32, 32), f"({chr(97+col)})", "$x/a$", "$y/a$")
        angles = np.deg2rad(fields["angle_deg"])
        radius = fields["radius_over_a"]
        x = radius[:, None]*np.cos(angles)[None]
        y = radius[:, None]*np.sin(angles)[None]
        close = lambda v: np.concatenate((v, v[:, :1]), axis=1)
        ax.fill(3*np.cos(theta), 3*np.sin(theta), facecolor="#f2f2f2", edgecolor="none", zorder=0)
        artist = ax.pcolormesh(close(x), close(y),
                              close(fields["accumulated_slip_strain"][:, :, -1]*100),
                              cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        meshes.append(artist)
        ax.fill(np.cos(theta), np.sin(theta), facecolor="white", edgecolor="black", linestyle="-", lw=.6)
        ax.plot(3*np.cos(theta), 3*np.sin(theta), color="black", linestyle="-", lw=.6)
        beta = np.deg2rad(case["joint_normal_deg"]+90)
        ax.plot(.8*np.array([-1, 1])*np.cos(beta), .8*np.array([-1, 1])*np.sin(beta),
                color=fs.RED, linestyle="--", lw=.9)
        ax.plot([-.37, .37], [-.37, .37], color=fs.BLACK, linestyle="-", lw=.9)
        for letter, location, color, marker in (
                ("A", "maximum_convergence", fs.BLACK, "o"),
                ("B", "maximum_slip", fs.RED, "s")):
            p = np.asarray(metadata["fixed_locations"][location]["xy_over_a"])
            ax.plot(*p, marker=marker, color=color, markerfacecolor="white", ms=3, mew=.7)
            ax.annotate(letter, p, xytext=.43*p, color=color, fontsize=7.2, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", pad=.2),
                        arrowprops=dict(arrowstyle="-", color=color, lw=.5, shrinkA=2, shrinkB=3))
        ax.set(xlim=(-3.15, 3.15), ylim=(-3.15, 3.15), aspect="equal", xticks=[-3, 0, 3], yticks=[-3, 0, 3])
        title = ("Oblique", "Parallel", "Perpendicular")[col]
        ax.text(.5, 1.18, title + rf" ($\beta_j={case['joint_normal_deg']:g}^\circ$)",
                ha="center", transform=ax.transAxes, fontsize=7.4)

        wall = data["wall"]
        final = [r for r in wall if int(r["load_step"]) == config["load_steps"]]
        ax = axis(fig, (left, 85, 28, 25), f"({chr(100+col)})" +
                  ("  Wall profiles" if col == 0 else ""),
                  r"Angle $\theta$ (deg)", "Convergence (mm)" if col == 0 else "")
        twin = ax.twinx()
        fs.style_axes(twin)
        twin.tick_params(labelsize=7, pad=1, colors=fs.RED)
        twin.set_ylabel("Accumulated slip (%)" if col == 2 else "", color=fs.RED, labelpad=1, fontsize=7.5)
        angle = np.array([float(r["angle_deg"]) for r in final])
        ax.plot(angle, [float(r["convergence_mm"]) for r in final], color=fs.BLACK, lw=.9)
        twin.plot(angle, [float(r["accumulated_slip_strain"])*100 for r in final],
                  color=fs.RED, ls="--", lw=.9)
        ax.set(xlim=(0, 360), xticks=[0, 180, 360], ylim=(0, convergence_limit))
        twin.set_ylim(0, slip_limit)

        ax = axis(fig, (left, 48, 28, 25), f"({chr(103+col)})" +
                  ("  Peak histories" if col == 0 else ""),
                  "Release fraction", "Peak convergence (mm)" if col == 0 else "")
        twin = ax.twinx()
        fs.style_axes(twin)
        twin.tick_params(labelsize=7, pad=1, colors=fs.RED)
        twin.set_ylabel("Peak slip (%)" if col == 2 else "", color=fs.RED, labelpad=1, fontsize=7.5)
        c, s = [0.], [0.]
        for k in range(1, config["load_steps"]+1):
            rows = [r for r in wall if int(r["load_step"]) == k]
            c.append(max(float(r["convergence_mm"]) for r in rows))
            s.append(max(float(r["accumulated_slip_strain"])*100 for r in rows))
        release = np.linspace(0, 1, len(c))
        ax.plot(release, c, color=fs.BLACK, marker="o", ms=2.4, mfc="white", lw=.9)
        twin.plot(release, s, color=fs.RED, ls="--", marker="s", ms=2.4, mfc="white", lw=.9)
        ax.set(xlim=(0, 1), xticks=[0, .5, 1], ylim=(0, convergence_limit))
        twin.set_ylim(0, slip_limit)

        ax = axis(fig, (left, 11, 32, 25), f"({chr(106+col)})" +
                  ("  Stress paths" if col == 0 else ""),
                  r"$-\sigma_n$ (MPa)", r"$\tau_j$ (MPa)")
        compression = np.linspace(0, compression_limit, 180)
        strength = config["physics"]["joint_cohesion_mpa"] + compression*np.tan(np.deg2rad(config["physics"]["joint_phi_deg"]))
        delta = config["physics"]["joint_smoothing_mpa"]
        tau = np.sqrt((strength+delta)**2-delta**2)
        ax.fill_between(compression, -tau, tau, color=".95", zorder=0)
        ax.plot(compression, tau, color=".5", lw=.6, ls=":")
        ax.plot(compression, -tau, color=".5", lw=.6, ls=":")
        for location, color, style, marker, name in (
                ("maximum_convergence", fs.BLACK, "-", "o", "A"),
                ("maximum_slip", fs.RED, "--", "s", "B")):
            rows = [r for r in data["paths"] if r["location"] == location]
            ax.plot([-float(r["sigma_n_MPa"]) for r in rows],
                    [float(r["tau_j_MPa"]) for r in rows], color=color, ls=style,
                    marker=marker, ms=2.3, mfc="white", lw=.8, label=name)
        ax.set(xlim=(0, compression_limit), ylim=(-1.05*tau[-1], 1.05*tau[-1]))
        ax.legend(fontsize=7, loc="lower left" if col == 2 else "upper left", handlelength=1.3, borderpad=.15)
    cax = axes_mm(fig, (39, 170, 60, 2))
    cb = fig.colorbar(artist, cax=cax, orientation="horizontal", ticks=[0, .001, .01, .1, .2])
    for mesh in meshes:
        mesh.colorbar = cb
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(labelsize=7, pad=1)
    cb.set_ticklabels(["0", "0.001", "0.01", "0.1", "0.2"])
    cb.minorticks_off()
    cb.set_label("Accumulated weak-plane slip strain (%) · symmetric log", fontsize=7.5, labelpad=1)
    export(fig, "orientation_extension", size,
           "Matched unloading protocols isolate weak-plane direction through spatial slip, complete wall profiles, loading histories and local stress paths.",
           [config_path.relative_to(ROOT).as_posix()] +
           [str((directory / case["name"] / filename).relative_to(ROOT)).replace("\\", "/")
            for case in cases for filename in ("fields.npz", "wall_profiles.csv", "stress_paths.csv", "metadata.json")],
           "code/plot_extensions.py --figure orientation",
           locations="A and B are each case's fixed final maximum-convergence and maximum-slip locations.",
           slip_color_scale="Symmetric log; linear core 0.001 percent.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("interface", "geometry", "orientation", "all"), default="all")
    parser.add_argument("--orientation-data", type=Path, default=ROOT / "results/orientation_extension")
    parser.add_argument("--orientation-config", type=Path, default=ROOT / "configs/orientation_extension.json")
    args = parser.parse_args()
    if args.figure in ("interface", "all"):
        interface_figure()
    if args.figure in ("geometry", "all"):
        geometry_figure()
    if args.figure in ("orientation", "all"):
        orientation_figure(args.orientation_data.resolve(), args.orientation_config.resolve())


if __name__ == "__main__":
    main()
