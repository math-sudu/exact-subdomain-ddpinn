"""The cap-free direction may trade residual components while lowering merit."""

from types import SimpleNamespace
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from train_causal_response import propose
from causal_response_problem import CausalResponseProblem, material_dual_vector
from case import make_case
from pilot.causal_sampling import loss_groups
from pilot.torch_sf import SfCircle
from pilot import config as C


def test_cap_free_trial_changes_a_zero_residual_without_mutating_boundary():
    theta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    residual = lambda x: torch.stack((x[0]-2, x[0]))
    vector = residual(theta)
    problem = SimpleNamespace(theta=theta, vec=vector, base=vector.detach(),
        cache=None, pts=None, sf=None, step=6, residual=residual,
        jt=lambda v: torch.autograd.grad(vector, theta, v, retain_graph=True)[0].detach())
    trial, record = propose(problem, torch.ones_like(theta),
                            {'mu': .1, 'radius': 10.}, lambda: None)
    torch.testing.assert_close(trial, theta.new_tensor([2/2.1]))
    assert theta.item() == 0 and 'jv' not in vars(problem)
    assert residual(trial)[1] > 0
    assert record['actual_decrease'] > 0 and record['backtracks'] == 0
    assert record['radius'] == 10


def test_material_residual_matches_existing_norm_and_gradient():
    sf = SfCircle(C.nondim_lambda(C.material(2.625, 45)), 3., 32, 'cpu', dtype=torch.float64)
    traction = torch.randn(32, 2, dtype=torch.float64, requires_grad=True)
    factored = material_dual_vector(sf, traction).square().sum()
    direct = sf.dual_metric(traction.T)
    torch.testing.assert_close(factored, direct, rtol=1e-12, atol=1e-14)
    actual = torch.autograd.grad(factored, traction)[0]
    expected = torch.autograd.grad(direct, traction)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


def test_causal_dual_objective_keeps_other_terms_and_prefix():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root/'configs/initialization.json').read_text())
    spec.pop('composite_area_quadrature', None)
    spec.update(hidden=[6, 6], n_pde=16, boundary=16,
                assessment_n_pde=16, assessment_boundary=16,
                interior_quadrature=dict(radial=4, angular=4, angular_shift=.317))
    model, params, points, c, s0, sf = make_case(spec, device='cpu')
    for _ in range(6):
        current = model.append_increment(copy_previous=False)
        with torch.no_grad():
            current.kinematic_head.weight.normal_(0, .03)
            current.stress_head.weight.normal_(0, .03)
    problem = CausalResponseProblem(model, spec, {'train': (points, sf)}, c, s0, params)
    h, r = problem.fields(problem.theta)
    groups = loss_groups(h, r, points, spec['regional_masses'])
    expected = sum(groups[i]*w for i, w in enumerate((*spec['weights'], 1.)) if i != 2)
    expected += spec['weights'][2]*sf.dual_metric(r['interface'].T)
    actual = problem.vec.square().sum()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    expected_gradient = torch.autograd.grad(expected, problem.theta, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, problem.theta, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-13)
    assert problem.protected_unchanged()
