"""Composite Gauss quadrature in normalized annular area, with full FFT boundaries.

Panel placement changes numerical resolution while positive weights still
integrate the same uniform physical area measure. It changes neither the
constitutive equations nor the relative residual/region coefficients.
"""

import numpy as np
import torch


def apply_composite_quadrature(points, outer_radius, *, area_breaks, radial_orders,
                               angular, angular_shift):
    breaks = np.asarray(area_breaks, dtype=float)
    if (breaks.ndim != 1 or len(breaks) < 2 or not np.isfinite(breaks).all()
            or breaks[0] != 0 or breaks[-1] != 1 or not np.all(np.diff(breaks) > 0)):
        raise ValueError('Area panels must strictly partition [0, 1]')
    if (len(radial_orders) != len(breaks)-1 or
            any(isinstance(n, bool) or int(n) != n or n < 2 for n in radial_orders)):
        raise ValueError('Each area panel needs an integer Gauss order of at least two')
    if (isinstance(angular, bool) or int(angular) != angular or angular < 4
            or not np.isfinite(angular_shift) or not np.isfinite(outer_radius) or outer_radius <= 1):
        raise ValueError('Invalid annular angular quadrature')
    angular = int(angular)
    if sum(radial_orders)*angular != points.n_pde:
        raise ValueError('Composite quadrature must match the declared interior point count')
    nodes, weights = [], []
    for lower, upper, order in zip(breaks[:-1], breaks[1:], radial_orders):
        z, w = np.polynomial.legendre.leggauss(int(order))
        nodes.append(lower+(upper-lower)*(z+1)/2)
        weights.append(w*(upper-lower)/2)
    area = np.concatenate(nodes)
    radial_weights = np.concatenate(weights)
    radius = np.sqrt(1+(outer_radius**2-1)*area)
    angle = 2*np.pi*(np.arange(angular)+float(angular_shift))/angular
    xy = np.stack((radius[:, None]*np.cos(angle), radius[:, None]*np.sin(angle)), -1).reshape(-1, 2)
    combined_weights = np.broadcast_to(radial_weights[:, None]/angular,
                                       (len(radius), angular)).copy().reshape(-1)
    like = points.xy_all
    points.xy_all = torch.cat((torch.as_tensor(xy, device=like.device, dtype=like.dtype), points.bnd_xy))
    points.interior_weights = torch.as_tensor(combined_weights, device=like.device, dtype=like.dtype)
    return points
