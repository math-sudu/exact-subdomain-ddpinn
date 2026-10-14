"""Differentiable circular unloading field shared by the causal increments."""

import numpy as np
import torch
from torch import nn

from . import config as C
from sf_core.material import TIMaterial
from sf_core.reference_solution import solve_circular_hole


class CircularElasticIncrement(nn.Module):
    def __init__(self, case, total_steps, like):
        super().__init__()
        if case["opening_axes"] != [1., 1.]:
            raise ValueError("The circular elastic background requires a unit circular opening")
        material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
        material = TIMaterial("unloading", material.Eh/C.E_REF, material.Ev/C.E_REF,
                              material.nu_vh, material.nu_hh, material.Gvh/C.E_REF,
                              material.phi)
        problem = solve_circular_hole(material, 1., *C.nondim_insitu(), n_modes=1)
        # Real storage preserves complex parts under Module.to(real_dtype).
        for name, value in dict(roots=[material.mu1, material.mu2],
                                displacement_map=[[material.p1, material.p2],
                                                  [material.q1, material.q2]],
                                potentials=np.asarray(problem.field.coeffs)[:, 0]/total_steps).items():
            value = np.asarray(value, dtype=complex)
            self.register_buffer(name, like.new_tensor(np.stack((value.real, value.imag), -1)))
        compliance = material.full_compliance()
        self.register_buffer("zz_map", like.new_tensor(-compliance[2, [0, 1, 5]]/compliance[2, 2]))

    def forward(self, xy):
        roots = torch.view_as_complex(self.roots)
        potentials = torch.view_as_complex(self.potentials)
        displacement_map = torch.view_as_complex(self.displacement_map)
        z = xy[:, :1] + xy[:, 1:]*roots
        a, b = (1-1j*roots)/2, (1+1j*roots)/2
        discriminant = torch.sqrt(z.square()-4*a*b)
        first, second = (z+discriminant)/(2*a), (z-discriminant)/(2*a)
        zeta = torch.where(first.abs() >= second.abs(), first, second)
        phi = potentials/zeta
        derivative = -potentials/(zeta.square()*(a-b/zeta.square()))
        displacement = 2*(phi @ displacement_map.T).real
        inplane = 2*torch.stack(((derivative*roots.square()).sum(-1),
                                 derivative.sum(-1), -(derivative*roots).sum(-1)), -1).real
        zz = inplane @ self.zz_map
        stress = torch.stack((inplane[:, 0], inplane[:, 1], zz, inplane[:, 2]), -1)
        return displacement, stress, xy.new_zeros((len(xy), 2))

    def spatial_fields(self, xy):
        directions = torch.eye(2, dtype=xy.dtype, device=xy.device)[:, None].expand(-1, len(xy), -1)
        values, derivatives = torch.vmap(lambda d: torch.func.jvp(self, (xy,), (d,)))(directions)
        return tuple(value[0] for value in values), derivatives
