import quivr as qv


class PhysicalParameters(qv.Table):
    """
    Physical parameters required for photometric magnitude predictions.

    Notes
    -----
    Currently, absolute magnitude is assumed to be in Johnson-Cousins V-band.
    """

    # Absolute magnitude (V-band)
    H_v = qv.Float64Column(nullable=True)

    # 1-sigma uncertainty in H_v (mag)
    H_v_sigma = qv.Float64Column(nullable=True)

    # Photometric slope parameter (H-G system)
    G = qv.Float64Column(nullable=True)

    # 1-sigma uncertainty in G
    G_sigma = qv.Float64Column(nullable=True)

    # Optional fit diagnostics (useful when these parameters are estimated from data)
    #
    # - `sigma_eff` is an empirical residual scatter estimate (mag). If per-point `mag_sigma`
    #   is missing, this will also absorb model error (e.g., incorrect fixed G).
    sigma_eff = qv.Float64Column(nullable=True)
    chi2_red = qv.Float64Column(nullable=True)
