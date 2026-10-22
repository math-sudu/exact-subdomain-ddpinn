"""Verify the actual rounded flow equations and their loading derivatives."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from pilot import config as C, plasticity as P
from pilot.joint_return import joint_return
from pilot.orientation_tangent import joint_active_tangent


@pytest.mark.parametrize("angle", [0., 45., 90.])
def test_joint_return_preserves_constitutive_flow_and_active_tangent(angle):
    params = replace(P.WeakPlaneParameters(), joint_normal_deg=angle)
    stiffness = torch.tensor(C.material(2.625, 45.).plane_strain_stiffness4()/C.E_REF)
    normal, shear = P._joint_geometry(params, stiffness)
    # Start from an exactly yielding state and add a known plastic increment.
    for sign in (-1., 1.):
        yield_stress = stiffness.new_tensor([-.7, -.7, -.6, 0.])
        strength = params.joint_cohesion-(yield_stress@normal)*np.tan(np.deg2rad(params.joint_phi_deg))
        tau = sign*((strength+params.joint_smoothing)**2-params.joint_smoothing**2).sqrt()
        # Tensor stress producing tau, with zero normal traction on this plane.
        shear_stress = shear.clone()
        shear_stress[:3] *= 2
        yield_stress += tau*shear_stress
        gamma = stiffness.new_tensor(.12)
        flow = P.joint_flow_direction(yield_stress, params)
        trial = yield_stress+gamma*(stiffness@flow)
        result = joint_return(trial, stiffness, params)
        torch.testing.assert_close(result["stress"], yield_stress, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(result["gamma"], gamma, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(result["plastic"], gamma*flow, atol=1e-13, rtol=1e-12)
        tangent = joint_active_tangent(yield_stress, gamma, stiffness, params)
        torch.testing.assert_close(result["tangent"], tangent, atol=1e-12, rtol=1e-12)
        direction = stiffness.new_tensor([.2, -.3, 0., .4])
        function = lambda strain: joint_return(trial+stiffness@strain, stiffness, params)["stress"]
        _, derivative = torch.func.jvp(function, (torch.zeros_like(trial),), (direction,))
        finite = (function(1e-6*direction)-function(-1e-6*direction))/(2e-6)
        torch.testing.assert_close(derivative, tangent@direction, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(derivative, finite, atol=1e-9, rtol=1e-8)


def test_joint_return_unloads_elastically_and_vectorizes_mixed_active_sets():
    params = replace(P.WeakPlaneParameters(), joint_normal_deg=0.)
    stiffness = torch.tensor(C.material(2.625, 45.).plane_strain_stiffness4()/C.E_REF)
    trial = stiffness.new_tensor([[-.5, -1., -.6, .1], [-.5, -1., -.6, -.5]])
    result = joint_return(trial, stiffness, params)
    assert result["active"].tolist() == [False, True]
    torch.testing.assert_close(result["stress"][0], trial[0], atol=0., rtol=0.)
    torch.testing.assert_close(result["tangent"][0], stiffness, atol=0., rtol=0.)
    assert result["gamma"][0] == 0 and result["gamma"][1] > 0
    torch.testing.assert_close(result["stress"]+result["plastic"]@stiffness.T,
                               trial, atol=1e-13, rtol=1e-12)
    assert abs(float(P.joint_yield(result["stress"][1], params))) < 1e-13
    # Unloading from the returned stress must not remain on a forced active branch.
    unload = result["stress"][1].clone()
    unload[3] *= .9
    returned = joint_return(unload, stiffness, params)
    assert not returned["active"]
    torch.testing.assert_close(returned["stress"], unload, atol=0., rtol=0.)




def test_joint_return_rejects_unsupported_flow_and_apex_branches():
    stiffness = torch.tensor(C.material(2.625, 45.).plane_strain_stiffness4()/C.E_REF)
    params = replace(P.WeakPlaneParameters(), joint_normal_deg=0.)
    with pytest.raises(ValueError, match="zero weak-plane dilation"):
        joint_return(stiffness.new_tensor([-.5, -1., -.6, -.5]), stiffness,
                     replace(params, joint_psi_deg=5.))
    with pytest.raises(ValueError, match="rounded apex"):
        joint_return(stiffness.new_tensor([2., 0., 0., 0.]), stiffness, params)
    unsupported = stiffness.new_tensor([2., 0., 0., 0.])
    result = joint_return(unsupported, stiffness, params, check_branch=False)
    assert not torch.isfinite(result["stress"]).all()
    with pytest.raises(FloatingPointError, match="Nonfinite trial stress"):
        joint_return(torch.full_like(unsupported, torch.nan), stiffness, params)
