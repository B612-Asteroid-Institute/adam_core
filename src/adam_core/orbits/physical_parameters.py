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

    # Photometric slope parameter (H-G system)
    G = qv.Float64Column(nullable=True)


