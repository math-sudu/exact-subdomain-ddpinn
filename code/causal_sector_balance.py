"""Area and contour quadrature of local annular-sector momentum residuals."""

import numpy as np
import torch


def sector_quadrature(inner, outer, lower, upper, order, like):
    if not 0 < inner < outer or not lower < upper or order < 2:
        raise ValueError('Require positive radial/angular intervals and quadrature order')
    z, w = np.polynomial.legendre.leggauss(order)
    radius = inner+(outer-inner)*(z+1)/2
    angle = lower+(upper-lower)*(z+1)/2
    rw, aw = w*(outer-inner)/2, w*(upper-lower)/2
    radial = np.stack((np.cos(angle), np.sin(angle)), -1)
    xy = (radius[:, None, None]*radial[None]).reshape(-1, 2)
    area_weights = (radius[:, None]*rw[:, None]*aw[None]).reshape(-1)
    contour_xy = [inner*radial, outer*radial]
    contour_normal = [-radial, radial]
    contour_weights = [inner*aw, outer*aw]
    for theta, sign in ((lower, -1), (upper, 1)):
        er = np.array([np.cos(theta), np.sin(theta)])
        et = sign*np.array([-np.sin(theta), np.cos(theta)])
        contour_xy.append(radius[:, None]*er)
        contour_normal.append(np.broadcast_to(et, (order, 2)))
        contour_weights.append(rw)
    cast = lambda x: torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return dict(xy=cast(xy), area_weights=cast(area_weights),
        contour_xy=cast(np.concatenate(contour_xy)),
        contour_normal=cast(np.concatenate(contour_normal)),
        contour_weights=cast(np.concatenate(contour_weights)), wall_count=order)


def momentum_integrals(stress, divergence, quadrature):
    normals, weights = quadrature['contour_normal'], quadrature['contour_weights']
    traction = torch.stack((stress[:, 0]*normals[:, 0]+stress[:, 3]*normals[:, 1],
                            stress[:, 3]*normals[:, 0]+stress[:, 1]*normals[:, 1]), -1)
    volume = (divergence*quadrature['area_weights'][:, None]).sum(0)
    flux = (traction*weights[:, None]).sum(0)
    absolute_volume = (divergence.norm(dim=-1)*quadrature['area_weights']).sum()
    return volume, flux, absolute_volume
