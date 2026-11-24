"""Independent quadrilateral finite-element solution of the first release."""

import numpy as np
import pandas as pd
from skfem import Basis, BilinearForm, ElementQuad1, ElementVector, FacetBasis, LinearForm, MeshQuad, asm, condense, solve
def wall_nodes(contour, spacing):
    """Place arc junctions, vertical extrema and every measured chord on nodes."""
    crown = contour["bounds_max"][1]
    elevations = [0., crown, contour["bounds_min"][1], *(crown-h for h in (2.5, 6., 9.5))]
    segments = []
    for arc in contour["circular_arcs"]:
        x, y = arc["center"]
        radius = arc["radius"]
        start = arc["start_rad"]
        stop = start + arc["sweep_rad"]
        split = [start, stop]
        for level in elevations:
            sine = (level-y)/radius
            if abs(sine) > 1 + 1e-12:
                continue
            angle = np.arcsin(np.clip(sine, -1., 1.))
            for value in (angle, np.pi-angle):
                value %= 2*np.pi
                if start + 1e-9 < value < stop - 1e-9:
                    split.append(value)
        limits = np.unique(np.round(split, 12))
        for left, right in zip(limits, limits[1:]):
            count = int(np.ceil(radius*(right-left)/spacing))
            theta = np.linspace(left, right, count, endpoint=False)
            segments.append(np.column_stack([x+radius*np.cos(theta), y+radius*np.sin(theta)]))
    return np.concatenate(segments)

def elastic_movements(settings, contour, outer_radii, wall_spacing, layers):
    wall = wall_nodes(contour, wall_spacing)
    n = len(wall)
    radius = outer_radii*settings["length_scale_m"]
    relative = radius/np.linalg.norm(wall, axis=1)
    fraction = np.linspace(0., 1., layers+1)[:, None]
    nodes = (wall[None, :, :]*np.power(relative, fraction)[:, :, None]).reshape(-1, 2)
    inner = (np.arange(layers)[:, None]*n + np.arange(n)).ravel()
    next_inner = (inner//n)*n + (inner+1)%n
    quad = np.array([inner, inner+n, next_inner+n, next_inner])
    mesh = MeshQuad(nodes.T, quad)
    element = ElementVector(ElementQuad1())
    basis = Basis(mesh, element)

    material = settings["elastic_constants"]
    eh, ev, shear = (1000*material[k] for k in ("Eh_gpa", "Ev_gpa", "Gvh_gpa"))
    nvh, nhh = material["nu_vh"], material["nu_hh"]
    compliance = np.array([[1/eh, -nvh/ev, -nhh/eh],
                           [-nvh/ev, 1/ev, -nvh/ev],
                           [-nhh/eh, -nvh/ev, 1/eh]])
    in_plane = np.linalg.inv(compliance)[:2, :2]
    angle = np.deg2rad(material["bedding_dip_deg"])
    h = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])

    def strains(field):
        gradient = field.grad
        def projection(a, b):
            return np.einsum("i,ij...,j->...", a, gradient, b)
        return np.array([projection(h, h), projection(v, v)]), projection(h, v)+projection(v, h)

    @BilinearForm
    def stiffness(u, test, w):
        eu, gu = strains(u)
        et, gt = strains(test)
        return np.einsum("i...,ij,j...->...", et, in_plane, eu)+shear*gu*gt

    sx, sy, txy = settings["insitu_mpa"]
    stress = np.array([[sx, txy], [txy, sy]]) + settings["support_pressure_mpa"]*np.eye(2)

    @LinearForm
    def unloading(test, w):
        # w.n points out of the rock domain at the cavity wall.
        return -np.einsum("i...,ij,j...->...", test, stress, w.n)/settings["load_steps"]

    wall_facets = np.nonzero((mesh.facets < n).all(axis=0))[0]
    rhs = asm(unloading, FacetBasis(mesh, element, facets=wall_facets))
    fixed = basis.nodal_dofs[:, -n:].ravel()
    motion = solve(*condense(asm(stiffness, basis), rhs, D=fixed))
    u = 1000*motion[basis.nodal_dofs[:, :n]].T

    def closure(level):
        indices = np.flatnonzero(np.isclose(wall[:, 1], level, rtol=0., atol=1e-7))
        left, right = indices[np.argsort(wall[indices, 0])[[0, -1]]]
        return u[left, 0]-u[right, 0]

    top = np.argmax(wall[:, 1])
    bottom = np.argmin(wall[:, 1])
    return dict(sidewall_shortening=closure(0.), crown_settlement=-u[top, 1],
                invert_heave=u[bottom, 1], vertical_shortening=u[bottom, 1]-u[top, 1],
                **{g: closure(wall[top, 1]-d) for g, d in
                   zip(("upper_chord", "middle_chord", "lower_chord"), (2.5, 6., 9.5))}), len(inner)

def compare_first_release(settings, contour, spectral, neural):
    if spectral.plastic_area_m2 or neural.plastic_area_m2:
        raise ValueError("The first unloading step has yielded.")
    rows = []
    for radii, spacing, layers in ((80, .24, 100), (160, .15, 180)):
        gauges, elements = elastic_movements(settings, contour, radii, spacing, layers)
        for gauge, movement in gauges.items():
            key = gauge + "_mm"
            rows.append(dict(gauge=gauge, outer_radius_over_a=radii, elements=elements,
                             fe_mm=movement, direct_mm=spectral[key], full_network_mm=neural[key],
                             direct_difference_pct=100*(spectral[key]/movement-1),
                             full_network_difference_pct=100*(neural[key]/movement-1)))
    return pd.DataFrame(rows)
