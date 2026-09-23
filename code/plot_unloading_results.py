"""Render composite field, profile and history figures from the saved path."""
from __future__ import annotations
import argparse
import csv
import json
import numpy as np
import contourpy
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, SymLogNorm
from matplotlib.patches import Circle, Rectangle
from figure_artwork import ROOT, axes_mm, export, fs

DATA = ROOT / "results/unloading"
SIZE = (137, 127)
LOCATION_IDS = {"final_maximum_closure": "A", "final_maximum_slip": "B"}
ZOOM = (.72, 1.12, -.65, -.15)
SLIP_DISPLAY_THRESHOLD = .0001  # Percent; display masking only.
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


def field_axes(ax, z, values, norm, cmap, title, *, contour=None, zoom=False,
               show_x=True, show_y=True):
    x, y = field_grid(z)
    mesh = ax.pcolormesh(x, y, values, norm=norm, cmap=cmap, shading="flat", rasterized=True)
    if contour is not None:
        angle = np.r_[z["angle_rad"], z["angle_rad"][0]+2*np.pi]
        xc = z["radius_over_a"][:, None]*np.cos(angle)
        yc = z["radius_over_a"][:, None]*np.sin(angle)
        closed_values = np.ma.column_stack((values, values[:, 0]))
        for line in contourpy.contour_generator(x=xc, y=yc, z=closed_values).lines(contour):
            ax.plot(line[:, 0], line[:, 1], color="black", linewidth=.65,
                    linestyle=":" if contour < 0 else "-")
    ax.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="black", linewidth=.7))
    ax.add_patch(Circle((0, 0), 3, fill=False, edgecolor="black", linewidth=.7))
    ax.set(xlim=(-3.15, 3.15), ylim=(-3.15, 3.15), aspect="equal",
           xlabel=r"$x/a$", ylabel=r"$y/a$", xticks=[-3, 0, 3], yticks=[-3, 0, 3])
    if zoom:
        ax.set(xlim=ZOOM[:2], ylim=ZOOM[2:], xticks=[.8, 1], yticks=[-.6, -.4, -.2])
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
    ax.annotate(f"({letter})", (ZOOM[1], ZOOM[2]), xytext=(2.1, -1.8),
                fontsize=7.5, ha="center", arrowprops=dict(arrowstyle="-", lw=.6))


def weak_plane_basis(ax):
    origin = np.array([.76, -.29])
    for direction, text in ((np.array([1, 0]), r"$n_j$"), (np.array([0, 1]), r"$t_j$")):
        end = origin + .075*direction
        ax.annotate("", end, xytext=origin, arrowprops=dict(arrowstyle="->", lw=.7, mutation_scale=6))
        ax.text(*(end + .022*direction), text, fontsize=8, ha="center", va="center")


def mark_zoom_location(ax, metadata):
    """Locate the same wall point in the global map and the near-wall detail."""
    point = metadata["fixed_locations"]["final_maximum_slip"]["xy_over_a"]
    ax.plot(*point, marker="s", markersize=3, markerfacecolor="white",
            markeredgecolor=fs.RED, linestyle="none", zorder=6)
    ax.annotate("B", point, xytext=(.87, -.39), color=fs.RED, fontsize=8,
                ha="right", va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color=fs.RED, lw=.6,
                                shrinkA=1, shrinkB=3))


def shared_bar(fig, meshes, title, ticks, *, log=False, bottom=54):
    cax = axes_mm(fig, (29, bottom, 80, 2.2))
    bar = fig.colorbar(meshes[0], cax=cax, orientation="horizontal",
                       ticks=ticks, extend="min" if log else "neither")
    if log:
        bar.set_ticklabels([str(t) for t in ticks])
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
    export(fig, name, size, thesis, [f"results/unloading/{file}" for file in files],
           "code/plot_unloading_results.py", **bindings)


def field_palette():
    return ListedColormap(matplotlib.colormaps["turbo"](np.linspace(.08, .95, 256)),
                         name="fem_turbo")


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

    # A common symmetric-log scale exposes onset while retaining signed
    # late-stage amplitudes. Masking changes the display, never source values.
    signed = 100*z["signed_slip_strain"]
    signed_cmap = matplotlib.colormaps["RdBu_r"].copy()
    signed_cmap.set_bad("#f2f2f2")
    signed_norm = SymLogNorm(linthresh=SLIP_LINEAR_CORE, linscale=.65,
                            vmin=-.25, vmax=.25, base=10)
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
        values = np.ma.masked_inside(signed[:, :, step-1],
                                     -SLIP_DISPLAY_THRESHOLD, SLIP_DISPLAY_THRESHOLD)
        title = rf"({letter})  $k={step}$" + (" detail" if is_zoom else "")
        meshes.append(field_axes(ax, z, values, signed_norm, signed_cmap, title,
                                  contour=-.005 if is_zoom else None, zoom=is_zoom,
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
                       ticks=[-.25, -.01, -.001, 0, .001, .01, .25])
    bar.set_ticklabels(["−0.25", "−0.01", "−0.001", "0", "0.001", "0.01", "0.25"])
    bar.set_label(r"Signed slip $\gamma_j^p$ (%) · symmetric log", fontsize=8, labelpad=1)
    bar.ax.tick_params(labelsize=7.5, direction="in", length=2, pad=1.2)
    bar.minorticks_off()
    for mesh in meshes:
        mesh.colorbar = bar

    profiles = [
        (47, "(g)  Wall convergence", r"$c$ (mm)", "closure_mm", 1,
         [3, 6, 9, 12], (0, .34), [0, .15, .3]),
        (28, "(h)  Accumulated slip", r"$s_j^p$ (%)", "accumulated_slip_strain", 100,
         [5, 8, 12], (-.01, .28), [0, .1, .2]),
        (9, "(i)  Signed slip", r"$\gamma_j^p$ (%)", "signed_slip_strain", 100,
         [5, 8, 12], (-.27, .27), [-.25, 0, .25]),
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
                offset = (-12, -10) if quantity == "accumulated_slip_strain" else (-12, 4)
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
         signed_field_steps=[5, 8, 12], signed_slip_limits_percent=[-.25, .25],
         signed_slip_scale="symmetric log, base 10", signed_slip_colormap="RdBu_r",
         signed_slip_linear_core_percent=SLIP_LINEAR_CORE,
         signed_slip_linscale=.65, signed_slip_display_threshold_percent=SLIP_DISPLAY_THRESHOLD,
         zoom_steps=[5, 12], zoom_xy_over_a=list(ZOOM), zoom_contour_percent=-.005,
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
    cmap.set_under("#f2f2f2")
    cmap.set_bad("#f2f2f2")
    norm, steps, meshes = LogNorm(1, 2000), [4, 5, 6, 8, 10, 12], []
    for i, step in enumerate(steps):
        row, column = divmod(i, 3)
        ax = axes_mm(fig, ((10, 54, 98)[column], (105, 68)[row], 32, 32))
        title = rf"({chr(97+i)})  $k={step}$"
        if step == 4:
            title += ": pre-onset"
        elif step == 5:
            title += ": onset"
        meshes.append(field_axes(ax, z, values[:, :, step-1], norm, cmap,
                                  title, contour=10, show_x=row == 1,
                                  show_y=column == 0))
        if step in (5, 12):
            zoom_box(ax, "g" if step == 5 else "h")
        if step in (4, 5, 12):
            mark_locations(ax, metadata)
    for i, step in enumerate((5, 12)):
        ax = axes_mm(fig, ((13, 57)[i], 9, 24, 30))
        meshes.append(field_axes(ax, z, values[:, :, step-1], norm, cmap,
                                 rf"({chr(103+i)})  $k={step}$ detail",
                                 contour=10, zoom=True))
        mark_zoom_location(ax, metadata)
        if step == 5:
            weak_plane_basis(ax)
    shared_bar(fig, meshes,
               r"Accumulated equivalent plastic strain $\kappa$ ($\mu\varepsilon$, log)",
               [1, 10, 100, 1000, 2000], log=True, bottom=54)
    growth = axes_mm(fig, (98, 9, 31, 30))
    fs.style_axes(growth)
    growth.set_title("(i)  Footprint area", fontsize=8.5, loc="left", pad=5)
    release = np.array([float(r["release_fraction"]) for r in summary])
    area = np.array([float(r["footprint_10microstrain_area_m2"]) for r in summary])
    growth.plot(np.r_[0, release], np.r_[0, area], color=fs.BLACK, linestyle="-",
                marker="o", markerfacecolor="white", markersize=2.5, linewidth=1)
    for step in (5, 8, 12):
        color, _, marker = STEP_STYLE[step]
        growth.plot(release[step-1], area[step-1], marker=marker, markersize=3.8,
                    markerfacecolor="white", color=color, linestyle="none", zorder=4)
    growth.set(xlim=(0, 1.02), ylim=(0, 1.4), xticks=[0, .5, 1], yticks=[0, .7, 1.4])
    growth.set_xlabel(r"Release $\lambda$", fontsize=8.5, labelpad=2)
    growth.set_ylabel(r"$A_{10}$ (m$^2$)", fontsize=8.5, labelpad=2)
    growth.tick_params(labelsize=7.5, pad=2)
    growth.text(.07, .87, rf"{area[-1]:.3f} m$^2$", transform=growth.transAxes, fontsize=8)
    radius = float(summary[-1]["footprint_10microstrain_outer_radius_over_a"])
    growth.text(.07, .72, rf"Reach: $ {radius:.2f}a$", transform=growth.transAxes, fontsize=8)
    save(fig, "unloading_plastic_zone",
         "Six full fields resolve the pre-onset, onset, early expansion and late plastic footprint; matched local details and its twelve-increment area history link local intensity to spatial extent.",
         ["fields.npz", "load_summary.csv", "metadata.json"], size=size, steps=steps,
         panel_map={**{chr(97+i): f"Accumulated equivalent strain at k={k}"
                       for i, k in enumerate(steps)},
                    "g": "Near-wall detail at k=5", "h": "Near-wall detail at k=12",
                    "i": "Area above 10 microstrain through all twelve increments"},
         quantity="accumulated_equivalent * 1e6", color_limits_microstrain=[1, 2000],
         color_scale="log", colormap="turbo", colormap_range=[.08, .95],
         below_display_limit="Light gray for kappa < 1 microstrain.",
         contour_microstrain=10, zoom_steps=[5, 12], zoom_xy_over_a=list(ZOOM),
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
    shear = -np.sqrt((capacity+delta)**2-delta**2)
    path_ax.plot(normal, shear, color=".5", linestyle=":", linewidth=1.1, label=r"$f_j=0$")
    panels = [axes_mm(fig, (14, 48, 48, 25)), axes_mm(fig, (83, 48, 46, 25)),
              bottom_axes(fig, 14, "(e)  Wall convergence", r"$c$ (mm)"),
              bottom_axes(fig, 83, "(f)  Accumulated slip", r"$s_j^p$ (%)", width=46)]
    quantities = [("sigma_n_MPa", 1), ("tau_j_MPa", 1), ("closure_mm", 1), ("accumulated_slip_strain", 100)]
    offsets = [{4: (-8, -9), 8: (0, 7), 12: (-3, -9)},
               {4: (-8, -4), 5: (0, -12), 8: (0, -11), 12: (7, 5)}]
    endpoints = {}
    for i, (name, letter, loc, style, text) in enumerate(fixed_locations(metadata)):
        path = sorted((r for r in rows if r["location"] == name), key=lambda r: int(r["load_step"]))
        x = np.array([float(r["sigma_n_MPa"]) for r in path])
        y = np.array([float(r["tau_j_MPa"]) for r in path])
        release = np.array([float(r["release_fraction"]) for r in path])
        path_ax.plot(x, y, **style, label=letter)
        for step, offset in offsets[i].items():
            path_ax.annotate(str(step), (x[step], y[step]), xytext=offset,
                             textcoords="offset points", fontsize=7.5, color=style["color"],
                             ha="center", va="center")
        for start, end in ((1, 2), (9, 10)):
            path_ax.annotate("", xy=(x[end], y[end]), xytext=(x[start], y[start]),
                             arrowprops=dict(arrowstyle="->", color=style["color"], lw=.8,
                                             shrinkA=3, shrinkB=3, mutation_scale=7))
        if letter == "B":
            path_ax.scatter([x[5]], [y[5]], s=110, facecolors="none", edgecolors=style["color"], linewidths=.65)
        for ax, (quantity, factor) in zip(panels, quantities):
            values = factor*np.array([float(r[quantity]) for r in path])
            ax.plot(release, values, **style, label=letter)
        endpoints[letter] = float(path[-1]["closure_mm"])
    path_ax.set(xlim=(-1.07, .03), ylim=(-.74, .06),
                xlabel=r"Normal stress $\sigma_n$ (MPa)", ylabel=r"Shear stress $\tau_j$ (MPa)")
    fs.style_axes(path_ax)
    path_ax.set_title("(a)  Total-stress paths on the weak plane", fontsize=8.5, loc="left", pad=5)
    path_ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=7.5, handlelength=1.5, columnspacing=.8)
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
    fig.text(117.5/SIZE[0], 86/SIZE[1], r"A: $107^\circ$", ha="center", fontsize=8, color=fs.BLACK)
    fig.text(117.5/SIZE[0], 81/SIZE[1], r"B: $338^\circ$", ha="center", fontsize=8, color=fs.RED)
    for ax in panels:
        fs.style_axes(ax)
        ax.set(xlim=(0, 1.02), xticks=[0, .5, 1], xlabel=r"Release fraction $\lambda$")
        ax.axvline(5/12, color=".65", linestyle=":", linewidth=.7, zorder=0)
        ax.tick_params(labelsize=7.5, pad=2)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
    panels[0].set(ylim=(-1.1, .04), ylabel=r"$\sigma_n$ (MPa)")
    panels[0].set_title("(c)  Normal stress", fontsize=8.5, loc="left", pad=5)
    panels[1].set(ylim=(-.65, .04), ylabel=r"$\tau_j$ (MPa)")
    panels[1].set_title("(d)  Shear stress", fontsize=8.5, loc="left", pad=5)
    for ax in panels[:2]:
        ax.set_xlabel("")
    panels[2].set_ylim(0, .37)
    panels[3].set_ylim(0, .27)
    ratio = 100*endpoints["B"]/endpoints["A"]
    panels[2].text(.05, .85, rf"$c_B/c_A={ratio:.1f}\%$", transform=panels[2].transAxes, fontsize=8.5)
    panels[3].text(.08, .8, "Step 5", transform=panels[3].transAxes, fontsize=8, color=".4")
    panels[3].annotate("", xy=(5/12, .17), xytext=(.24, .2), arrowprops=dict(arrowstyle="-", lw=.65, color=".4"))
    save(fig, "unloading_stress_paths",
         "The same two fixed wall points connect stress paths, normal and shear histories, convergence and accumulated slip; B approaches yield at step five and ends at 60.8 percent of A's convergence.",
         ["fixed_location_stress_paths.csv", "fixed_location_histories.csv", "metadata.json"],
         panel_map={"a": "Total weak-plane stress paths and rounded yield branch",
                    "b": "Fixed A/B coordinates and weak-plane basis",
                    "c": "Normal stress history", "d": "Shear stress history",
                    "e": "Convergence history", "f": "Accumulated-slip history"},
         fixed_locations=metadata["fixed_locations"], location_labels=LOCATION_IDS,
         yield_curve="sqrt(tau_j^2 + delta_j^2)-delta_j + sigma_n*tan(phi_j)-c_j=0",
         onset_guide_step=5, convergence_ratio_percent=ratio, initial_state="Total initial stress; zero excavation displacement and slip.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--figure", choices=["wall", "plastic", "stress"])
    select.add_argument("--stress-paths-only", action="store_true")
    select.add_argument("--plastic-only", action="store_true")
    args = parser.parse_args()
    selected = "stress" if args.stress_paths_only else "plastic" if args.plastic_only else args.figure
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    summary = read_csv("load_summary.csv")
    if selected in (None, "wall", "plastic"):
        with np.load(DATA / "fields.npz") as z:
            if selected in (None, "wall"):
                wall_figure(z, metadata, read_csv("wall_profiles.csv"))
            if selected in (None, "plastic"):
                plastic_figure(z, metadata, summary)
    if selected in (None, "stress"):
        stress_path_figure(metadata)


if __name__ == "__main__":
    main()
