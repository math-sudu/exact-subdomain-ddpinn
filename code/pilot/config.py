"""Physical scales and transversely isotropic material constants."""

import numpy as np

from sf_core.material import TIMaterial

A_TUNNEL = 1.0
SX, SY, TXY = -1.0e6, -2.0e6, 0.0
SIGMA_REF = 2.0e6
E_REF = 10.5e9
U_REF = SIGMA_REF * A_TUNNEL / E_REF


def material(eh_over_ev=2.625, bedding_dip_deg=45.0):
    return TIMaterial(
        name="excavation", Eh=float(eh_over_ev) * E_REF, Ev=E_REF,
        nu_vh=0.2, nu_hh=0.25, Gvh=E_REF / 2.5,
        phi=np.deg2rad(float(bedding_dip_deg)),
    )


def nondim_lambda(mat):
    return mat.Lam / E_REF


def nondim_insitu():
    return SX / SIGMA_REF, SY / SIGMA_REF, TXY / SIGMA_REF
