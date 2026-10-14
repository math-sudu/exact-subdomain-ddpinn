"""Causal displacement, stress and plastic-multiplier increment networks."""

from copy import deepcopy
import torch
from torch import nn
from . import plasticity as P
from .mixed_plasticity import field_history
from .plastic_assessment import regional_mean_square


def trunk(hidden):
    layers, width = [], 2
    for size in hidden:
        layers.extend((nn.Linear(width, size), nn.Tanh()))
        width = size
    return nn.Sequential(*layers)


class IncrementFields(nn.Module):
    def __init__(self, hidden, input_scale, total_steps, multiplier_scale):
        super().__init__()
        self.input_scale, self.total_steps = float(input_scale), int(total_steps)
        self.kinematic_trunk, self.stress_trunk = trunk(hidden), trunk(hidden)
        self.kinematic_head = nn.Linear(hidden[-1], 4)
        self.stress_head = nn.Linear(hidden[-1], 4)
        self.register_buffer('multiplier_scale', multiplier_scale.detach().clone())
        # Zero perturbation, without requiring zero internal feature gradients
        # after the first accepted head update.
        for head in (self.kinematic_head, self.stress_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, xy):
        kin = self.kinematic_head(self.kinematic_trunk(xy/self.input_scale))/self.total_steps
        stress = self.stress_head(self.stress_trunk(xy/self.input_scale))/self.total_steps
        return kin[..., :2], stress, kin[..., 2:]*self.multiplier_scale

    def spatial_fields(self, xy):
        directions = torch.stack((P._spatial_direction(xy, 0), P._spatial_direction(xy, 1)))
        values, derivatives = torch.vmap(lambda d: torch.func.jvp(self, (xy,), (d,)))(directions)
        return tuple(value[0] for value in values), derivatives


class CausalMixedPINN(nn.Module):
    def __init__(self, hidden, input_scale, total_steps, stiffness):
        super().__init__()
        self.hidden, self.input_scale, self.total_steps = tuple(hidden), float(input_scale), int(total_steps)
        self.register_buffer('multiplier_scale', 1/stiffness.diag().abs().max())
        self.increments = nn.ModuleList()

    @property
    def n_steps(self):
        return len(self.increments)

    def make_increment(self):
        return IncrementFields(self.hidden, self.input_scale, self.total_steps,
                               self.multiplier_scale).to(self.multiplier_scale)

    def append_increment(self, *, copy_previous=True):
        for increment in self.increments:
            increment.requires_grad_(False)
        if copy_previous and self.increments:
            current = deepcopy(self.increments[-1]).requires_grad_(True)
        else:
            current = self.make_increment()
        self.increments.append(current)
        return current

    def forward(self, xy):
        values = [increment(xy) for increment in self.increments]
        u, stress, gamma = (torch.stack([value[i] for value in values], dim=1) for i in range(3))
        return u.cumsum(1), stress.cumsum(1), gamma


@torch.no_grad()
def prefix_cache(model, xy, stiffness, insitu, params):
    """Recompute the entire frozen prefix at precisely the supplied coordinates."""
    if model.n_steps == 0:
        z4, z2 = xy.new_zeros((len(xy),4)), xy.new_zeros((len(xy),2))
        return dict(displacement=z2,stress_increment=z4,strain=z4,
                    plastic_strain=z4,equilibrium=z2,xy=xy.detach().clone())
    parts = [field_history(model, x, stiffness, insitu, params) for x in xy.split(2048)]
    cache = {key: torch.cat([part[key][:,-1] for part in parts]).detach()
             for key in ('displacement','stress_increment','strain','plastic_strain','equilibrium')}
    cache['xy'] = xy.detach().clone()
    return cache


def current_residuals(increment, cache, pts, stiffness, insitu, sf, params, step,
                      *, projected_coefficients=None):
    xy = pts.xy_all
    if xy.shape != cache['xy'].shape or not torch.equal(xy, cache['xy']):
        raise ValueError('Prefix cache belongs to different coordinates; recompute history.')
    if not 1 <= step <= increment.total_steps:
        raise ValueError('Invalid physical loading step')
    if "increment_background" in cache:
        (du, ds, gamma), derivatives = increment.spatial_fields(
            xy, background_fields=cache["increment_background"],
            grid=cache.get("increment_grid"), coefficients=projected_coefficients)
    elif isinstance(increment, IncrementFields):
        (du, ds, gamma), derivatives = increment.spatial_fields(xy)
    else:
        # Functional parameter probes expose a callable rather than a module.
        (du, ds, gamma), derivatives = IncrementFields.spatial_fields(increment, xy)
    ddu, dds, _ = derivatives
    de = torch.stack((ddu[0,...,0],ddu[1,...,1],torch.zeros_like(ddu[0,...,0]),
                      ddu[1,...,0]+ddu[0,...,1]),-1)
    strain, stress = cache['strain']+de, cache['stress_increment']+ds
    u = cache['displacement']+du
    total = insitu+stress
    flow = (gamma[:,:1]*P.matrix_flow_direction(total,params)
            +gamma[:,1:]*P.joint_flow_direction(total,params))
    plastic = cache['plastic_strain']+flow
    kp = strain-torch.linalg.solve(stiffness,stress.unsqueeze(-1)).squeeze(-1)
    equilibrium = cache['equilibrium']+torch.stack((dds[0,...,0]+dds[1,...,3],
                                                   dds[0,...,3]+dds[1,...,1]),-1)
    yields = torch.stack((P.matrix_yield(total,params),P.joint_yield(total,params)),-1)
    scaled = gamma/increment.multiplier_scale
    fb = P._fischer_burmeister(scaled,-yields,params.complementarity_smoothing)
    n0,n1 = pts.n_pde,pts.n_pde+pts.n_wall
    exterior = sf.apply(u[n1:].T).T
    previous_defect = (cache['strain']-cache['plastic_strain'])@stiffness.T-cache['stress_increment']
    constitutive = (strain-plastic)@stiffness.T-stress
    r = dict(equilibrium=equilibrium[:n0],
        wall=P._traction4(stress[n0:n1],pts.nrm_wall)+(step/increment.total_steps)*pts.t0_wall,
        interface=P._traction4(stress[n1:],pts.nrm_gam)-exterior,
        constitutive=constitutive,flow=constitutive-previous_defect,complementarity=fb,
        gauge=u[n1:].mean(0))
    h = dict(displacement=u,stress_increment=stress,stress_total=total,strain=strain,
             kinematic_plastic_strain=kp,plastic_strain=plastic,multiplier_increment=gamma,
             scaled_multiplier=scaled,yield_values=yields)
    return h,r


def loss_groups(h,r,pts,masses):
    return torch.stack((r['equilibrium'].square().sum(-1).mean(),
        r['wall'].square().sum(-1).mean(),r['interface'].square().mean(),
        regional_mean_square(r['constitutive'],pts,masses),
        regional_mean_square(r['complementarity'],pts,masses),r['gauge'].square().sum()))


def objective(h,r,pts,masses,weights):
    groups = loss_groups(h,r,pts,masses)
    return (groups*groups.new_tensor((*weights,1.))).sum()
