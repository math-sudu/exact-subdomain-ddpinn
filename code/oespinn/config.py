"""Stress and stiffness scales and transversely isotropic material construction."""

import numpy as np
SIGMA_REF = 2.0e6

E_REF = 10.5e9

def material(eh_over_ev: float = 2.625, bedding_dip_deg: float = 45.0):
    """Construct a material with the specified modulus ratio and bedding angle.

    The normal modulus, Poisson ratios and shear modulus remain fixed.
    """

    from sf_core.material import TIMaterial

    ev = E_REF
    return TIMaterial(
        name=f"R{eh_over_ev:g}_D{bedding_dip_deg:g}",
        Eh=float(eh_over_ev) * ev,
        Ev=ev,
        nu_vh=0.2,
        nu_hh=0.25,
        Gvh=ev / 2.5,
        phi=np.deg2rad(float(bedding_dip_deg)),
    )
