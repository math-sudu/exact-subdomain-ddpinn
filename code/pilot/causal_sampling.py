"""Area-preserving polar quadrature for spatially resolved causal residuals."""

import numpy as np
import torch

from .causal_mixed import loss_groups as uniform_loss_groups
from .plastic_assessment import region_slices


def apply_quadrature(points, outer_radius, spec):
    radial, angular = int(spec['radial']), int(spec['angular'])
    if radial < 2 or angular < 4 or radial*angular != points.n_pde:
        raise ValueError('Polar quadrature shape must match interior point count')
    # Uniform area is uniform in t=(r^2-a^2)/(R^2-a^2), with a=1.
    z, w = np.polynomial.legendre.leggauss(radial)
    radius = np.sqrt(1+(outer_radius**2-1)*(z+1)/2)
    theta = 2*np.pi*(np.arange(angular)+float(spec['angular_shift']))/angular
    xy = np.stack((radius[:, None]*np.cos(theta), radius[:, None]*np.sin(theta)), -1).reshape(-1, 2)
    weights = np.broadcast_to(w[:, None]/(2*angular), (radial, angular)).copy().reshape(-1)
    points.xy_all = torch.cat((torch.as_tensor(xy, device=points.xy_all.device, dtype=points.xy_all.dtype), points.bnd_xy))
    points.interior_weights = torch.as_tensor(weights, device=points.xy_all.device, dtype=points.xy_all.dtype)
    return points


def loss_groups(h, r, pts, masses):
    if not hasattr(pts, 'interior_weights'):
        return uniform_loss_groups(h, r, pts, masses)
    w = pts.interior_weights
    if w.shape != (pts.n_pde,) or not torch.all(w > 0) or not torch.isclose(w.sum(), w.new_tensor(1.)):
        raise ValueError('Invalid area-mean quadrature weights')
    def regional(value):
        means = {}
        for name, sl in region_slices(pts).items():
            point_power = value[sl].square().reshape(len(value[sl]), -1).mean(-1)
            means[name] = (point_power*w).sum() if name == 'interior' else point_power.mean()
        return sum(masses[name]*v for name, v in means.items())
    return torch.stack(((r['equilibrium'].square().sum(-1)*w).sum(),
        r['wall'].square().sum(-1).mean(), r['interface'].square().mean(),
        regional(r['constitutive']), regional(r['complementarity']), r['gauge'].square().sum()))


def objective(h, r, pts, masses, weights):
    groups = loss_groups(h, r, pts, masses)
    return (groups*groups.new_tensor((*weights, 1.))).sum()
