"""Exact radial map of a convex, tangent circular-arc opening to a circle."""
import json
from pathlib import Path

import numpy as np
import torch


class HorseshoeGeometry:
    def __init__(self, source, length_scale=6.73, radius=4.0, role="initial_support_outer"):
        self.source = str(source)
        self.length_scale, self.radius = float(length_scale), float(radius)
        data = json.loads(Path(source).read_text("utf-8"))
        if data["confirmed_metres_per_drawing_unit"] != 1.0:
            raise ValueError("The CAD coordinate scale must be explicit")
        self.boundary = next(item for item in data["boundaries"] if item["role"] == role)
        self.arcs = self.boundary["circular_arcs"]
        if radius * length_scale <= max(np.linalg.norm(self.boundary["bounds_min"]),
                                        np.linalg.norm(self.boundary["bounds_max"])):
            raise ValueError("The coupling circle must enclose the complete opening")

    def wall(self, theta):
        """Radius, angular derivative and cavity-outward normal, in scaled units."""
        e = torch.stack((theta.cos(), theta.sin()), -1)
        et = torch.stack((-theta.sin(), theta.cos()), -1)
        radius = torch.zeros_like(theta)
        derivative = torch.zeros_like(theta)
        normal = torch.zeros_like(e)
        assigned = torch.zeros_like(theta, dtype=torch.bool)
        for arc in self.arcs:
            center = theta.new_tensor(arc["center"]) / self.length_scale
            a = arc["radius"] / self.length_scale
            dot = e @ center
            root2 = dot.square() + a*a - center.square().sum()
            root = root2.clamp_min(0).sqrt()
            candidate = dot + root
            radial_normal = candidate[..., None]*e-center
            angle = torch.atan2(radial_normal[..., 1], radial_normal[..., 0])
            relative = torch.remainder(angle-arc["start_rad"], 2*torch.pi)
            valid = ((relative <= arc["sweep_rad"]+1e-12) | (relative >= 2*torch.pi-1e-12)) & (root2 > 0) & (candidate > 0)
            n = radial_normal/a
            dr = -candidate*(n*et).sum(-1)/(n*e).sum(-1).clamp_min(1e-12)
            radius = torch.where(valid, candidate, radius)
            derivative = torch.where(valid, dr, derivative)
            normal = torch.where(valid[..., None], n, normal)
            assigned |= valid
        if not bool(assigned.all()):
            raise ValueError("A ray does not intersect the adopted finite arc boundary")
        return radius, derivative, normal

    def map(self, s, theta):
        wall, _, _ = self.wall(theta)
        r = wall+(self.radius-wall)*s
        return r[..., None]*torch.stack((theta.cos(), theta.sin()), -1)

    def inverse(self, xy):
        theta = torch.atan2(xy[..., 1], xy[..., 0])
        wall, _, _ = self.wall(theta)
        return (xy.norm(dim=-1)-wall)/(self.radius-wall), theta

    def horizontal_chord(self, depth_below_crown_m):
        """Reference-contour chord, without identifying it with a surveyed gauge."""
        y = self.boundary["bounds_max"][1]-depth_below_crown_m
        intersections = []
        for arc in self.arcs:
            cx, cy = arc["center"]
            square = arc["radius"]**2-(y-cy)**2
            if square < 0:
                continue
            for x in (cx-np.sqrt(square), cx+np.sqrt(square)):
                angle = np.arctan2(y-cy, x-cx)
                if (angle-arc["start_rad"]) % (2*np.pi) <= arc["sweep_rad"]+1e-12:
                    intersections.append(x)
        if len(intersections) < 2:
            raise ValueError("The requested reference level does not cross the opening")
        return [[min(intersections), y], [max(intersections), y]]

    def quadrature(self, nr, nt, *, dtype=torch.float64, device="cpu", shift=0.0):
        nodes, weights = np.polynomial.legendre.leggauss(nr)
        s = torch.as_tensor((nodes+1)/2, dtype=dtype, device=device)
        w = torch.as_tensor(weights/2, dtype=dtype, device=device)
        theta = (torch.arange(nt, dtype=dtype, device=device)+shift)*(2*torch.pi/nt)
        wall, _, _ = self.wall(theta)
        radius = wall[None]+(self.radius-wall[None])*s[:, None]
        weight = w[:, None]*radius*(self.radius-wall[None])*(2*torch.pi/nt)
        ss, tt = torch.meshgrid(s, theta, indexing="ij")
        return self.map(ss, tt).reshape(-1, 2), weight.reshape(-1)

    def boundary_quadrature(self, nt, *, dtype=torch.float64, device="cpu", shift=0.0):
        points, normals, weights = [], [], []
        for arc in self.arcs:
            count = max(4, int(np.ceil(nt*arc["sweep_rad"]/(2*np.pi))))+int(shift != 0)
            x, w = np.polynomial.legendre.leggauss(count)
            angle = torch.as_tensor(arc["start_rad"]+arc["sweep_rad"]*(x+1)/2, dtype=dtype, device=device)
            normal = torch.stack((angle.cos(), angle.sin()), -1)
            center = angle.new_tensor(arc["center"])/self.length_scale
            radius = arc["radius"]/self.length_scale
            points.append(center+radius*normal)
            normals.append(normal)
            weights.append(torch.as_tensor(w, dtype=dtype, device=device)*arc["sweep_rad"]*radius/2)
        return torch.cat(points), torch.cat(normals), torch.cat(weights)


def mapped_basis(geometry, xy, radial_order, angular_modes):
    """Real Chebyshev/Fourier basis and exact Cartesian first derivatives.

    The m=0 basis uses T_j(s)-1, j>=1, so only the outer-circle mean is
    removed. Nonconstant interior m=0 fields and all even modes remain.
    """
    s, theta = geometry.inverse(xy)
    r = xy.norm(dim=-1)
    wall, drwall, _ = geometry.wall(theta)
    z = 2*s-1
    t = [torch.ones_like(z), z]
    dt = [torch.zeros_like(z), torch.ones_like(z)]
    for _ in range(2, radial_order+1):
        dt.append(2*t[-1]+2*z*dt[-1]-dt[-2])
        t.append(2*z*t[-1]-t[-2])
    t, ds = torch.stack(t, -1), 2*torch.stack(dt, -1)
    mode = torch.arange(1, angular_modes+1, device=xy.device, dtype=xy.dtype)
    phase = theta[:, None]*mode
    angular = torch.cat((torch.ones_like(theta[:, None]), phase.cos(), phase.sin()), -1)
    angular_derivative = torch.cat((torch.zeros_like(theta[:, None]), -phase.sin()*mode, phase.cos()*mode), -1)
    value = t[:, :, None]*angular[:, None]
    derivative_s = ds[:, :, None]*angular[:, None]
    derivative_t = t[:, :, None]*angular_derivative[:, None]
    value[:, :, 0] -= 1
    value = value.flatten(1)[:, 1:]
    derivative_s = derivative_s.flatten(1)[:, 1:]
    derivative_t = derivative_t.flatten(1)[:, 1:]
    c, sn = theta.cos(), theta.sin()
    dsdr = 1/(geometry.radius-wall)
    dsdt = -(1-s)*drwall/(geometry.radius-wall)
    dx = derivative_s*(dsdr*c-dsdt*sn/r)[:, None]-derivative_t*(sn/r)[:, None]
    dy = derivative_s*(dsdr*sn+dsdt*c/r)[:, None]+derivative_t*(c/r)[:, None]
    return value, dx, dy
