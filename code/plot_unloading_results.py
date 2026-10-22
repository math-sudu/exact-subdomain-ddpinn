"""Render composite field, profile and history figures from the saved path."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap, SymLogNorm
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MultipleLocator
from figure_artwork import ROOT, axes_mm, export, fs

DATA = ROOT / "results/unloading"
SIZE = (137, 127)
LOCATION_IDS = {"final_maximum_closure": "A", "final_maximum_slip": "B"}
ZOOM = (.72, 1.12, -.65, -.15)
SLIP_LINEAR_CORE = .001  # Percent; shared symmetric-log transition.
STEP_STYLE = {
    3: (fs.BLACK, ":", "o"), 5: (fs.BLACK, "-", "o"),
    6: (fs.RED, "--", "^"), 8: (fs.RED, "--", "^"),
    9: (fs.BLUE, "-.", "D"), 12: (fs.BLUE, "-", "s"),
}


def read_csv(name):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fixed_locations(metadata):
    for i, (name, letter) in enumerate(LOCATION_IDS.items()):
        location = metadata["fixed_locations"][name]
        style = fs.series(i, role="model")
        style.update(marker=("o", "s")[i], markersize=3, markerfacecolor="white")
        yield name, letter, location, style, rf'{letter}: $\theta={location["angle_deg"]:g}^\circ$'


def configure_zoom(metadata):
    global ZOOM
    x, y = metadata["fixed_locations"]["final_maximum_slip"]["xy_over_a"]
    if not (ZOOM[0] <= x <= ZOOM[1] and ZOOM[2] <= y <= ZOOM[3]):
        ZOOM = (x-.2, x+.2, y-.275, y+.225)


def mark_locations(ax, metadata):
    for _, letter, location, style, _ in fixed_locations(metadata):
        point = np.array(location["xy_over_a"])
        ax.plot(*point, **{**style, "linestyle": "none", "zorder": 6})
        ax.annotate(letter, point, xytext=.40*point, color=style["color"],
                    fontsize=8, ha="center", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=style["color"], lw=.6,
                                    shrinkA=3, shrinkB=3))


def field_grid(z):
    edges = z["radius_edges_over_a"]
    angle = np.r_[z["angle_rad"]-np.pi/720, 2*np.pi-np.pi/720]
    return edges[:, None]*np.cos(angle), edges[:, None]*np.sin(angle)


def field_axes(ax, z, values, norm, cmap, title, *, zoom=False,
               show_x=True, show_y=True):
    x, y = field_grid(z)
    mesh = ax.pcolormesh(x, y, values, norm=norm, cmap=cmap, shading="flat", rasterized=True)
    ax.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="black", linewidth=.7))
    ax.add_patch(Circle((0, 0), 3, fill=False, edgecolor="black", linewidth=.7))
    ax.set(xlim=(-3.15, 3.15), ylim=(-3.15, 3.15), aspect="equal",
           xlabel=r"$x/a$", ylabel=r"$y/a$", xticks=[-3, 0, 3], yticks=[-3, 0, 3])
    if zoom:
        ax.set(xlim=ZOOM[:2], ylim=ZOOM[2:])
        ax.xaxis.set_major_locator(MultipleLocator(.2))
        ax.yaxis.set_major_locator(MultipleLocator(.2))
    fs.style_axes(ax)
    ax.tick_params(labelsize=7.5, pad=2, labelbottom=show_x, labelleft=show_y,
                   length=3)
    ax.set_xlabel(r"$x/a$" if show_x else "")
    ax.set_ylabel(r"$y/a$" if show_y else "")
    ax.set_title(title, fontsize=8.5, pad=4, loc="left")
    ax.xaxis.labelpad = 1.5
    ax.yaxis.labelpad = 1.5
    return mesh


def zoom_box(ax, letter):
    ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1]-ZOOM[0], ZOOM[3]-ZOOM[2],
                           fill=False, edgecolor="black", linewidth=.65, linestyle="--"))
    anchor = (2.1*np.sign(np.mean(ZOOM[:2])), 1.8*np.sign(np.mean(ZOOM[2:])))
    corner = (ZOOM[0] if anchor[0] < 0 else ZOOM[1], ZOOM[2] if anchor[1] < 0 else ZOOM[3])
    ax.annotate(f"({letter})", corner, xytext=anchor,
                fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="-", lw=.6))


def weak_plane_basis(ax):
    origin = np.array([.94*np.mean(ZOOM[:2]), ZOOM[3]-.13])
    for direction, text in ((np.array([1, 0]), r"$n_j$"), (np.array([0, 1]), r"$t_j$")):
        end = origin + .075*direction
        ax.annotate("", end, xytext=origin, arrowprops=dict(arrowstyle="->", lw=.7, mutation_scale=6))
        ax.text(*(end + .022*direction), text, fontsize=8, ha="center", va="center")


def mark_zoom_location(ax, metadata):
    """Locate the same wall point in the global map and the near-wall detail."""
    point = metadata["fixed_locations"]["final_maximum_slip"]["xy_over_a"]
    ax.plot(*point, marker="s", markersize=3, markerfacecolor="white",
            markeredgecolor=fs.RED, linestyle="none", zorder=6)
    ax.annotate("B", point, xytext=.90*np.array(point), color=fs.RED, fontsize=8,
                ha="center", va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color=fs.RED, lw=.6,
                                shrinkA=1, shrinkB=3))


def shared_bar(fig, meshes, title, ticks, *, bottom=54):
    cax = axes_mm(fig, (29, bottom, 80, 2.2))
    bar = fig.colorbar(meshes[0], cax=cax, orientation="horizontal", ticks=ticks)
    bar.set_ticklabels([f"{value:g}" for value in ticks])
    bar.minorticks_off()
    bar.set_label(title, fontsize=8, labelpad=1)
    bar.ax.tick_params(labelsize=7.5, direction="in", length=2, pad=1.2)
    for mesh in meshes:
        mesh.colorbar = bar


def bottom_axes(fig, left, title, ylabel, *, width=48):
    ax = axes_mm(fig, (left, 10, width, 25))
    fs.style_axes(ax)
    ax.set_title(title, fontsize=8.5, loc="left", pad=5)
    ax.set_ylabel(ylabel, fontsize=8.5, labelpad=2)
    ax.tick_params(labelsize=7.5, pad=2)
    ax.xaxis.labelpad = 2
    return ax


def save(fig, name, thesis, files, *, size=SIZE, **bindings):
    export(fig, name, size, thesis, [(DATA / file).relative_to(ROOT).as_posix() for file in files],
           "code/plot_unloading_results.py", **bindings)


def field_palette():
    return ListedColormap(matplotlib.colormaps["turbo"](np.linspace(.08, .95, 256)),
                         name="fem_turbo")


def displacement_figure(z, metadata, walls):
    """Resolve incremental changes hidden by the cumulative displacement maps."""
    size = (137, 169)
    fig = fs.figure(*size)
    u = z["displacement_mm"].astype(np.float64)
    du = np.diff(u, axis=2, prepend=np.zeros_like(u[:, :, :1]))
    magnitude = np.linalg.norm(du, axis=-1)
    cmap = field_palette()
    increment_limit = np.ceil(magnitude.max()*100)/100
    levels = np.linspace(0, increment_limit, 25)
    norm = BoundaryNorm(levels, cmap.N)
    steps, meshes = [4, 5, 6, 8, 10, 12], []
    details = [
        ("g", (-.9, .4, .75, 2.05), "A sector", 0),
        ("h", (.7, 2, -.95, .35), "B sector", 1),
    ]
    for j, (letter, bounds, title, location_index) in enumerate(details):
        point = np.array(list(metadata["fixed_locations"].values())[location_index]["xy_over_a"])
        if not (bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]):
            center = 1.4*point
            details[j] = (letter, (center[0]-.65, center[0]+.65,
                                  center[1]-.65, center[1]+.65), title, location_index)
    for i, step in enumerate(steps):
        row, col = divmod(i, 3)
        ax = axes_mm(fig, ((11, 55, 99)[col], (132, 97)[row], 30, 30))
        meshes.append(field_axes(ax, z, magnitude[:, :, step-1], norm, cmap,
                                rf"({chr(97+i)})  $k={step}$",
                                show_x=row == 1, show_y=col == 0))
        if step in (4, 12):
            mark_locations(ax, metadata)
        if step == 12:
            for letter, bounds, _, _ in details:
                ax.add_patch(Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0],
                                       bounds[3]-bounds[2], fill=False, edgecolor="white",
                                       linewidth=.8, linestyle="--"))
                text = (2.4*np.sign(np.mean(bounds[:2])),
                        (2.6 if letter == "g" else 2.4)*np.sign(np.mean(bounds[2:])))
                anchor = (bounds[0] if text[0] < 0 else bounds[1],
                          bounds[2] if text[1] < 0 else bounds[3])
                ax.annotate(f"({letter})", anchor, xytext=text, fontsize=7.5,
                            ha="center", arrowprops=dict(arrowstyle="-", lw=.6))
    shared_bar(fig, meshes, r"Incremental displacement $|\Delta\mathbf{u}_k|$ (mm)",
               np.arange(0, increment_limit+.005, .01), bottom=86)
    for j, (letter, bounds, title, location_index) in enumerate(details):
        ax = axes_mm(fig, ((13, 57)[j], 45, 26, 26))
        meshes.append(field_axes(ax, z, magnitude[:, :, 11], norm, cmap,
                                 f"({letter})  {title}, $k=12$"))
        ax.set(xlim=bounds[:2], ylim=bounds[2:])
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        loc = list(metadata["fixed_locations"].values())[location_index]
        direction = np.array(loc["xy_over_a"])
        color, marker = (fs.BLACK, "o") if j == 0 else (fs.RED, "s")
        ax.plot([direction[0], 3*direction[0]], [direction[1], 3*direction[1]],
                color="white", linewidth=2.3)
        ax.plot([direction[0], 3*direction[0]], [direction[1], 3*direction[1]],
                color=color, linewidth=.9, linestyle="--" if j else "-")
        ax.plot(*direction, color=color, marker=marker, markerfacecolor="white",
                markersize=3, linestyle="none")
        ax.annotate("A" if j == 0 else "B", direction, xytext=(-10, 7) if j == 0 else (7, -13),
                    textcoords="offset points", color=color, fontsize=8.5)
        meshes[-1].colorbar = meshes[0].colorbar

    increment = axes_mm(fig, (99, 45, 30, 26))
    fs.style_axes(increment)
    increment.set_title(r"(i)  Wall increments", fontsize=8.5, loc="left", pad=4)
    increment.set(xlim=(.5, 12.5), ylim=(0, .045), xticks=[1, 4, 8, 12],
                  yticks=[0, .02, .04], ylabel=r"$\Delta c_k$ (mm)")
    increment.tick_params(labelsize=7.5, pad=2)
    increment.yaxis.labelpad = 1.5
    wall_values = np.array([[float(r["closure_mm"]) for r in walls if int(r["load_step"]) == k]
                            for k in range(1, 13)])
    wall_increment = np.diff(wall_values, axis=0, prepend=np.zeros_like(wall_values[:1]))
    for i, (name, letter) in enumerate(LOCATION_IDS.items()):
        angle = metadata["fixed_locations"][name]["angle_deg"]
        values = wall_increment[:, int(round(angle/.5))]
        increment.plot(np.arange(1, 13), values, color=(fs.BLACK, fs.RED)[i],
                       linestyle=("-", "--")[i], marker=("o", "s")[i],
                       markerfacecolor="white", markersize=2.5, linewidth=1, label=letter)
    increment.plot(np.arange(1, 13), wall_increment.mean(1), color=fs.BLUE, linestyle="-.",
                   marker="^", markerfacecolor="white", markersize=2.5, linewidth=1, label="Mean")
    increment.legend(loc="upper left", fontsize=7, handlelength=1.4,
                     borderpad=.15, labelspacing=.25)
    increment.set_xlabel(r"Increment $k$", fontsize=8, labelpad=1.5)

    with np.load(DATA / "radial_kinematics.npz") as radial:
        queried = radial["displacement_mm"]
        radial_increment = np.linalg.norm(np.diff(queried, axis=2,
                                          prepend=np.zeros_like(queried[:, :, :1])), axis=-1)
        radius = radial["radius_over_a"]
    for j, left in enumerate((15, 85)):
        ax = axes_mm(fig, (left, 9, 44, 24))
        fs.style_axes(ax)
        ax.set_title(f"({chr(106+j)})  Along {'A' if j == 0 else 'B'}", fontsize=8.5, loc="left", pad=4)
        ax.set(xlim=(1, 3), ylim=(0, increment_limit), xticks=[1, 2, 3],
               yticks=np.arange(0, increment_limit+.001, .02),
               xlabel=r"Radius $r/a$", ylabel=r"$|\Delta\mathbf{u}_k|$ (mm)")
        ax.tick_params(labelsize=7.5, pad=2)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 1.5
        for step in (5, 8, 12):
            color, linestyle, marker = STEP_STYLE[step]
            ax.plot(radius, radial_increment[j, :, step-1], color=color,
                    linestyle=linestyle, marker=marker, markerfacecolor="white",
                    markersize=2.5, markevery=32, linewidth=1, label=rf"$k={step}$")
        ax.legend(loc="upper right", fontsize=7.5, handlelength=1.8, labelspacing=.25)
    save(fig, "unloading_displacement",
         "Equal release increments develop different spatial displacement patterns; paired A/B details, per-step wall motion and radial profiles separate local increment growth from cumulative convergence.",
         ["fields.npz", "wall_profiles.csv", "radial_kinematics.npz", "metadata.json"], size=size,
         steps=steps, quantity="norm(u_k - u_(k-1)); vector subtraction precedes the norm",
         panel_map={**{chr(97+i): f"Incremental displacement magnitude at k={k}" for i, k in enumerate(steps)},
                    "g": "Final increment in the A sector with radial trace",
                    "h": "Final increment in the B sector with radial trace",
                    "i": "Per-increment convergence at A/B and the circumferential mean",
                    "j": "Radial increment magnitudes at A for k=5,8,12",
                    "k": "Radial increment magnitudes at B for k=5,8,12"},
         color_limits_mm=[0, increment_limit], color_boundaries_mm=levels.tolist(),
         colormap="turbo", colormap_range=[.08, .95], color_scale="linear",
         detail_bounds_xy_over_a=[list(item[1]) for item in details],
         radial_profile_steps=[5, 8, 12], radial_samples=257,
         reference="c545231: displacement details and radial traces, reassembled around increment fields",
         fixed_locations=metadata["fixed_locations"], location_labels=LOCATION_IDS)


def wall_figure(z, metadata, walls):
    size = (137, 157)
    fig = fs.figure(*size)
    magnitude = np.linalg.norm(z["displacement_mm"][:, :, 11], axis=-1)
    cmap = field_palette()
    levels = np.linspace(0, .35, 29)
    disp_ax = axes_mm(fig, (11, 120, 30, 30))
    disp_mesh = field_axes(disp_ax, z, magnitude, BoundaryNorm(levels, cmap.N), cmap,
                           r"(a)  $|\mathbf{u}|$, $k=12$", show_x=False)
    mark_locations(disp_ax, metadata)
    cax = axes_mm(fig, (44, 121, 2, 28))
    bar = fig.colorbar(disp_mesh, cax=cax, ticks=[0, .1, .2, .3, .35])
    bar.ax.set_title("mm", fontsize=7.5, pad=3)
    bar.ax.tick_params(labelsize=7.5, direction="in", length=2, pad=1.2)
    bar.minorticks_off()
    disp_mesh.colorbar = bar

    # A common symmetric-log scale shows onset and signed late-stage amplitudes.
    signed = 100*z["signed_slip_strain"]
    signed_cmap = matplotlib.colormaps["RdBu_r"].copy()
    signed_cmap.set_bad("#f2f2f2")
    signed_norm = SymLogNorm(linthresh=SLIP_LINEAR_CORE, linscale=.65,
                            vmin=-.35, vmax=.35, base=10)
    meshes = []
    panels = [
        ("b", 5, False, (57, 120, 30, 30)),
        ("c", 5, True, (103.6, 120, 24, 30)),
        ("d", 8, False, (11, 85, 30, 30)),
        ("e", 12, False, (57, 85, 30, 30)),
        ("f", 12, True, (103.6, 85, 24, 30)),
    ]
    for letter, step, is_zoom, rect in panels:
        ax = axes_mm(fig, rect)
        values = signed[:, :, step-1]
        title = rf"({letter})  $k={step}$" + (" detail" if is_zoom else "")
        meshes.append(field_axes(ax, z, values, signed_norm, signed_cmap, title,
                                  zoom=is_zoom,
                                  show_x=letter in "def",
                                  show_y=is_zoom or letter == "d"))
        if is_zoom:
            mark_zoom_location(ax, metadata)
            if step == 5:
                weak_plane_basis(ax)
        else:
            mark_locations(ax, metadata)
            if step in (5, 12):
                zoom_box(ax, "c" if step == 5 else "f")
    cax = axes_mm(fig, (26, 74, 100, 2.3))
    bar = fig.colorbar(meshes[0], cax=cax, orientation="horizontal",
                       ticks=[-.35, -.01, -.001, 0, .001, .01, .35])
    bar.set_ticklabels(["−0.35", "−0.01", "−0.001", "0", "0.001", "0.01", "0.35"])
    bar.set_label(r"Signed slip $\gamma_j^p$ (%) · symmetric log", fontsize=8, labelpad=1)
    bar.ax.tick_params(labelsize=7.5, direction="in", length=2, pad=1.2)
    bar.minorticks_off()
    for mesh in meshes:
        mesh.colorbar = bar

    profiles = [
        (47, "(g)  Wall convergence", r"$c$ (mm)", "closure_mm", 1,
         [3, 6, 9, 12], (0, .34), [0, .15, .3]),
        (28, "(h)  Accumulated slip", r"$s_j^p$ (%)", "accumulated_slip_strain", 100,
         [5, 8, 12], (-.01, .37), [0, .15, .3]),
        (9, "(i)  Signed slip", r"$\gamma_j^p$ (%)", "signed_slip_strain", 100,
         [5, 8, 12], (-.37, .37), [-.3, 0, .3]),
    ]
    by_step = {}
    for step in {3, 5, 6, 8, 9, 12}:
        by_step[step] = sorted((r for r in walls if int(r["load_step"]) == step),
                               key=lambda r: float(r["angle_deg"]))
    closure = None
    for bottom, title, ylabel, quantity, factor, steps, limits, ticks in profiles:
        height = 15 if quantity == "closure_mm" else 13
        ax = fig.add_axes((14/size[0], bottom/size[1], 115/size[0], height/size[1]),
                          sharex=closure)
        if closure is None:
            closure = ax
        fs.style_axes(ax)
        ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360],
               ylim=limits, yticks=ticks, ylabel=ylabel)
        ax.tick_params(labelsize=7.5, pad=2, labelbottom=bottom == 9)
        ax.yaxis.labelpad = 2
        ax.text(.012, 1.03, title, transform=ax.transAxes, fontsize=8.5, va="bottom")
        if quantity == "signed_slip_strain":
            ax.axhline(0, color=".6", linewidth=.6, zorder=0)
        for step in steps:
            theta = np.array([float(r["angle_deg"]) for r in by_step[step]])
            values = factor*np.array([float(r[quantity]) for r in by_step[step]])
            color, linestyle, marker = STEP_STYLE[step]
            ax.plot(np.r_[theta, 360], np.r_[values, values[0]],
                    color=color, linestyle=linestyle, linewidth=1,
                    marker=marker, markerfacecolor="white", markersize=2.5,
                    markeredgewidth=.65, markevery=60, label=rf"$k={step}$")
        for _, letter, loc, style, _ in fixed_locations(metadata):
            point = next(r for r in by_step[12] if float(r["angle_deg"]) == loc["angle_deg"])
            value = factor*float(point[quantity])
            ax.axvline(loc["angle_deg"], color=style["color"], alpha=.4,
                       linestyle=":", linewidth=.65, ymax=.76, zorder=0)
            ax.plot(loc["angle_deg"], value,
                    **{**style, "linestyle": "none", "zorder": 5})
            offset = (4, -9) if quantity == "closure_mm" and letter == "A" else (4, 3)
            if letter == "B":
                offset = (6, -10)
            ax.annotate(letter, (loc["angle_deg"], value), xytext=offset,
                        textcoords="offset points", fontsize=8, color=style["color"])
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.005),
                  frameon=False, ncol=len(steps), fontsize=7.5,
                  handlelength=1.7, columnspacing=.8, handletextpad=.4,
                  borderaxespad=0, borderpad=.1)
    ax.set_xlabel(r"Wall angle $\theta$ (deg)", labelpad=2)
    save(fig, "unloading_wall_history",
         "The final displacement field locates regional convergence; signed-slip maps at onset, development and final release connect two local details to three multi-stage wall profiles at the same A/B angles.",
         ["fields.npz", "wall_profiles.csv", "metadata.json"], size=size,
         panel_map={"a": "Final displacement magnitude",
                    "b": "Signed slip at k=5", "c": "Signed-slip detail at k=5",
                    "d": "Signed slip at k=8", "e": "Signed slip at k=12",
                    "f": "Signed-slip detail at k=12",
                    "g": "Convergence profiles at k=3,6,9,12",
                    "h": "Accumulated-slip profiles at k=5,8,12",
                    "i": "Signed-slip profiles at k=5,8,12"},
         displacement_step=12, displacement_limits_mm=[0, .35],
         displacement_scale="linear", displacement_color_boundaries_mm=levels.tolist(),
         displacement_colormap="turbo", colormap_range=[.08, .95],
         signed_field_steps=[5, 8, 12], signed_slip_limits_percent=[-.35, .35],
         signed_slip_scale="symmetric log, base 10", signed_slip_colormap="RdBu_r",
         signed_slip_linear_core_percent=SLIP_LINEAR_CORE,
         signed_slip_linscale=.65,
         zoom_steps=[5, 12], zoom_xy_over_a=list(ZOOM),
         closure_steps=[3, 6, 9, 12], accumulated_profile_steps=[5, 8, 12],
         signed_profile_steps=[5, 8, 12], shared_angular_axis_deg=[0, 360],
         samples_per_profile=720, periodic_closure="Append the value at 0 degrees at 360 degrees.",
         markers="Open symbols every 60 samples; all samples retained in each line.",
         fixed_locations=metadata["fixed_locations"], location_labels=LOCATION_IDS)


def plastic_figure(z, metadata, summary):
    size = (137, 144)
    fig = fs.figure(*size)
    values = z["accumulated_equivalent"]*1e6
    cmap = field_palette()
    limit = float(10**np.ceil(np.log10(values.max())))
    norm = SymLogNorm(linthresh=1, linscale=.65, vmin=0, vmax=limit)
    steps, meshes = [4, 5, 6, 8, 10, 12], []
    for i, step in enumerate(steps):
        row, column = divmod(i, 3)
        ax = axes_mm(fig, ((10, 54, 98)[column], (105, 68)[row], 32, 32))
        title = rf"({chr(97+i)})  $k={step}$"
        if step == 4:
            title += ": pre-onset"
        elif step == 5:
            title += ": onset"
        meshes.append(field_axes(ax, z, values[:, :, step-1], norm, cmap,
                                  title, show_x=row == 1,
                                  show_y=column == 0))
        if step in (5, 12):
            zoom_box(ax, "g" if step == 5 else "h")
        if step in (4, 5, 12):
            mark_locations(ax, metadata)
    for i, step in enumerate((5, 12)):
        ax = axes_mm(fig, ((13, 57)[i], 9, 24, 30))
        meshes.append(field_axes(ax, z, values[:, :, step-1], norm, cmap,
                                 rf"({chr(103+i)})  $k={step}$ detail",
                                 zoom=True))
        mark_zoom_location(ax, metadata)
        if step == 5:
            weak_plane_basis(ax)
    shared_bar(fig, meshes,
               r"Accumulated equivalent plastic strain $\kappa$ ($\mu\varepsilon$, symmetric log)",
               [0, 1, 10, 100, limit], bottom=54)
    growth = axes_mm(fig, (98, 9, 31, 30))
    fs.style_axes(growth)
    growth.set_title("(i)  Peak strain", fontsize=8.5, loc="left", pad=5)
    release = np.array([float(r["release_fraction"]) for r in summary])
    peak = np.array([float(r["maximum_accumulated_equivalent"])*1e6 for r in summary])
    growth.plot(np.r_[0, release], np.r_[0, peak], color=fs.BLACK, linestyle="-",
                marker="o", markerfacecolor="white", markersize=2.5, linewidth=1)
    for step in (5, 8, 12):
        color, _, marker = STEP_STYLE[step]
        growth.plot(release[step-1], peak[step-1], marker=marker, markersize=3.8,
                    markerfacecolor="white", color=color, linestyle="none", zorder=4)
    peak_limit = np.ceil(peak.max()/100)*100
    growth.set(xlim=(0, 1.02), ylim=(0, peak_limit), xticks=[0, .5, 1],
               yticks=[0, peak_limit/2, peak_limit])
    growth.set_xlabel(r"Release $\lambda$", fontsize=8.5, labelpad=2)
    growth.set_ylabel(r"$\kappa_{\max}$ ($\mu\varepsilon$)", fontsize=8.5, labelpad=2)
    growth.tick_params(labelsize=7.5, pad=2)
    growth.text(.07, .87, rf"{peak[-1]:.1f} $\mu\varepsilon$", transform=growth.transAxes, fontsize=8)
    save(fig, "unloading_plastic_zone",
         "Six full strain fields, matched near-wall details and the twelve-increment peak history connect spatial plastic evolution to its growing intensity.",
         ["fields.npz", "load_summary.csv", "metadata.json"], size=size, steps=steps,
         panel_map={**{chr(97+i): f"Accumulated equivalent strain at k={k}"
                       for i, k in enumerate(steps)},
                    "g": "Near-wall detail at k=5", "h": "Near-wall detail at k=12",
                    "i": "Peak accumulated equivalent strain through all twelve increments"},
         quantity="accumulated_equivalent * 1e6", color_limits_microstrain=[0, limit],
         color_scale="symmetric log", colormap="turbo", colormap_range=[.08, .95],
         linear_core_microstrain=1, zoom_steps=[5, 12], zoom_xy_over_a=list(ZOOM),
         fixed_locations=metadata["fixed_locations"], location_labels=LOCATION_IDS)


def stress_path_figure(metadata):
    rows = read_csv("fixed_location_stress_paths.csv")
    histories = read_csv("fixed_location_histories.csv")
    # The two exports must describe the same fixed points, not moving extrema.
    stress_by_key = {(r["location"], int(r["load_step"])): r for r in rows}
    for r in histories:
        other = stress_by_key[(r["location"], int(r["load_step"]))]
        for quantity in ("closure_mm", "accumulated_slip_strain"):
            assert np.isclose(float(r[quantity]), float(other[quantity]), atol=1e-14, rtol=1e-12)
    fig = fs.figure(*SIZE)
    path_ax = axes_mm(fig, (16, 87, 80, 32))
    physics = metadata["physics"]
    normal = np.linspace(-1.07, .03, 300)
    delta = physics["joint_smoothing_mpa"]
    capacity = physics["joint_cohesion_mpa"] - normal*np.tan(np.deg2rad(physics["joint_phi_deg"]))
    shear = np.sqrt((capacity+delta)**2-delta**2)
    path_ax.fill_between(normal, -shear, shear, color=".96", zorder=0)
    path_ax.plot(normal, shear, color=".5", linestyle=":", linewidth=1.1, label=r"$f_j=0$")
    path_ax.plot(normal, -shear, color=".5", linestyle=":", linewidth=1.1)
    panels = [axes_mm(fig, (14, 48, 48, 25)), axes_mm(fig, (83, 48, 46, 25)),
              bottom_axes(fig, 14, "(e)  Wall convergence", r"$c$ (mm)"),
              bottom_axes(fig, 83, "(f)  Accumulated slip", r"$s_j^p$ (%)", width=46)]
    quantities = [("sigma_n_MPa", 1), ("tau_j_MPa", 1), ("closure_mm", 1), ("accumulated_slip_strain", 100)]
    offsets = [{4: (-8, -9), 8: (0, 7), 12: (-3, -9)},
               {4: (-8, -4), 5: (0, -12), 8: (0, -11), 12: (7, 5)}]
    endpoints = {}
    slip_path = sorted((r for r in rows if r["location"] == "final_maximum_slip"),
                       key=lambda r: int(r["load_step"]))
    onset = next(int(r["load_step"]) for r in slip_path if float(r["accumulated_slip_strain"]) > 0)
    for i, (name, letter, loc, style, text) in enumerate(fixed_locations(metadata)):
        path = sorted((r for r in rows if r["location"] == name), key=lambda r: int(r["load_step"]))
        x = np.array([float(r["sigma_n_MPa"]) for r in path])
        y = np.array([float(r["tau_j_MPa"]) for r in path])
        release = np.array([float(r["release_fraction"]) for r in path])
        path_ax.plot(x, y, **style, label=letter)
        for step, offset in offsets[i].items():
            if letter == "B" and y[-1] > 0:
                offset = (offset[0], -offset[1])
                if step == onset:
                    offset = (8, -12)
            path_ax.annotate(str(step), (x[step], y[step]), xytext=offset,
                             textcoords="offset points", fontsize=7.5, color=style["color"],
                             ha="center", va="center")
        for start, end in ((1, 2), (9, 10)):
            path_ax.annotate("", xy=(x[end], y[end]), xytext=(x[start], y[start]),
                             arrowprops=dict(arrowstyle="->", color=style["color"], lw=.8,
                                             shrinkA=3, shrinkB=3, mutation_scale=7))
        if letter == "B":
            path_ax.scatter([x[onset]], [y[onset]], s=110, facecolors="none", edgecolors=style["color"], linewidths=.65)
        for ax, (quantity, factor) in zip(panels, quantities):
            values = factor*np.array([float(r[quantity]) for r in path])
            ax.plot(release, values, **style, label=letter)
        endpoints[letter] = float(path[-1]["closure_mm"])
    path_ax.set(xlim=(-1.07, .03), ylim=(-.76, .76),
                xlabel=r"Normal stress $\sigma_n$ (MPa)", ylabel=r"Shear stress $\tau_j$ (MPa)")
    fs.style_axes(path_ax)
    path_ax.set_title("(a)  Total-stress paths on the weak plane", fontsize=8.5, loc="left", pad=5)
    path_ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=7.5, handlelength=1.5, columnspacing=.8)
    path_ax.annotate("0", (-1, 0), xytext=(-8, -3), textcoords="offset points", fontsize=7.5)
    path_ax.xaxis.labelpad = 2
    path_ax.yaxis.labelpad = 2
    locator = axes_mm(fig, (103, 90, 29, 29))
    locator.add_patch(Circle((0, 0), 1, facecolor="#f3f3f3", edgecolor="black", linewidth=.8))
    locator.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal")
    locator.set_axis_off()
    locator.set_title("(b)  Fixed wall points", fontsize=8.5, loc="center", pad=4)
    mark_locations(locator, metadata)
    locator.annotate("", (1.3, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=".5", lw=.65))
    locator.annotate("", (0, 1.3), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=".5", lw=.65))
    locator.text(1.3, -.2, r"$n_j$", fontsize=8, ha="center")
    locator.text(.12, 1.17, r"$t_j$", fontsize=8)
    for name, letter, _, _, text in fixed_locations(metadata):
        fig.text(117.5/SIZE[0], (86 if letter == "A" else 81)/SIZE[1], text,
                 ha="center", fontsize=8, color=fs.BLACK if letter == "A" else fs.RED)
    for ax in panels:
        fs.style_axes(ax)
        ax.set(xlim=(0, 1.02), xticks=[0, .5, 1], xlabel=r"Release fraction $\lambda$")
        ax.axvline(onset/12, color=".65", linestyle=":", linewidth=.7, zorder=0)
        ax.tick_params(labelsize=7.5, pad=2)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
    panels[0].set(ylim=(-1.1, .04), ylabel=r"$\sigma_n$ (MPa)")
    panels[0].set_title("(c)  Normal stress", fontsize=8.5, loc="left", pad=5)
    shear_history = np.array([float(r["tau_j_MPa"]) for r in rows])
    panels[1].set(ylim=(np.floor((shear_history.min()-.04)*10)/10,
                        np.ceil((shear_history.max()+.04)*10)/10), ylabel=r"$\tau_j$ (MPa)")
    panels[1].set_title("(d)  Shear stress", fontsize=8.5, loc="left", pad=5)
    for ax in panels[:2]:
        ax.set_xlabel("")
    panels[2].set_ylim(0, .37)
    panels[3].set_ylim(0, .27)
    ratio = 100*endpoints["B"]/endpoints["A"]
    panels[2].text(.05, .85, rf"Final $c_B/c_A={ratio:.1f}\%$", transform=panels[2].transAxes,
                   fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", pad=.8))
    panels[3].text(.08, .8, f"Step {onset}", transform=panels[3].transAxes, fontsize=8, color=".4")
    panels[3].annotate("", xy=(onset/12, .17), xytext=(.24, .2), arrowprops=dict(arrowstyle="-", lw=.65, color=".4"))
    save(fig, "unloading_stress_paths",
         f"The same two fixed wall points connect stress paths, normal and shear histories, convergence and accumulated slip; B develops slip at step {onset} and ends at {ratio:.1f} percent of A's convergence.",
         ["fixed_location_stress_paths.csv", "fixed_location_histories.csv", "metadata.json"],
         panel_map={"a": "Total weak-plane stress paths and both rounded yield branches",
                    "b": "Fixed A/B coordinates and weak-plane basis",
                    "c": "Normal stress history", "d": "Shear stress history",
                    "e": "Convergence history", "f": "Accumulated-slip history"},
         fixed_locations=metadata["fixed_locations"], location_labels=LOCATION_IDS,
         yield_curve="sqrt(tau_j^2 + delta_j^2)-delta_j + sigma_n*tan(phi_j)-c_j=0",
         onset_guide_step=onset, convergence_ratio_percent=ratio, initial_state="Total initial stress; zero excavation displacement and slip.")


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--figure-directory", type=Path, default=ROOT / "figures")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--figure", choices=["displacement", "wall"])
    args = parser.parse_args()
    DATA = args.data.resolve()
    if args.figure_directory:
        import figure_artwork
        figure_artwork.FIGURES = args.figure_directory.resolve()
    selected = args.figure
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    configure_zoom(metadata)
    summary = read_csv("load_summary.csv")
    if selected in (None, "displacement", "wall", "plastic"):
        with np.load(DATA / "fields.npz") as z:
            if selected in (None, "displacement"):
                displacement_figure(z, metadata, read_csv("wall_profiles.csv"))
            if selected in (None, "wall"):
                wall_figure(z, metadata, read_csv("wall_profiles.csv"))


if __name__ == "__main__":
    main()
