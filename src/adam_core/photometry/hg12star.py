import numpy as np


# Basis functions Phi1, Phi2, Phi3 from Penttila et al. (2016)
# Hermite cubic spline (Appendix A, Eq. A.1).
# xs in degrees; derivatives ds are in d(y)/d(alpha_rad) as given in Penttila Table A.2/A.3.
def _hermite_spline(x_deg, xs_deg, ys, ds_per_rad):
    x = np.deg2rad(np.asarray(x_deg, dtype=float))
    xs = np.deg2rad(xs_deg)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    dx = xs[j + 1] - xs[j]
    dy = ys[j + 1] - ys[j]
    t = (x - xs[j]) / dx
    a = ds_per_rad[j] * dx - dy
    b = -ds_per_rad[j + 1] * dx + dy
    result = (1 - t) * ys[j] + t * ys[j + 1] + t * (1 - t) * ((1 - t) * a + t * b)
    return float(result[0]) if scalar else result


# Spline knots Table A.2: xi1
_XI1_X = np.array([7.5, 30.0, 60.0, 90.0, 120.0, 150.0])
_XI1_Y = np.array(
    [7.5e-1, 3.3486016e-1, 1.3410560e-1, 5.1104756e-2, 2.1465687e-2, 3.6396989e-3]
)
_XI1_D = np.array(
    [
        -1.9098593,
        -5.5463432e-1,
        -2.4404599e-1,
        -9.4980438e-2,
        -2.1411424e-2,
        -9.1328612e-2,
    ]
)

# xi2
_XI2_X = np.array([7.5, 30.0, 60.0, 90.0, 120.0, 150.0])
_XI2_Y = np.array(
    [9.25e-1, 6.2884169e-1, 3.1755495e-1, 1.2716367e-1, 2.2373903e-2, 1.6505689e-4]
)
_XI2_D = np.array(
    [
        -5.7295780e-1,
        -7.6705367e-1,
        -4.5665789e-1,
        -2.8071809e-1,
        -1.1173257e-1,
        -8.6573138e-8,
    ]
)

# xi3 Table A.3
_XI3_X = np.array([0.0, 0.3, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0])
_XI3_Y = np.array(
    [
        1.0,
        8.3381185e-1,
        5.7735424e-1,
        4.2144772e-1,
        2.3174230e-1,
        1.0348178e-1,
        6.1733473e-2,
        1.6107006e-2,
        0.0,
    ]
)
_XI3_D = np.array(
    [
        -1.0630097e-1,
        -4.1180439e1,
        -1.0366915e1,
        -7.5784615,
        -3.6960950,
        -7.8605652e-1,
        -4.6527012e-1,
        -2.0459545e-1,
        0.0,
    ]
)


def _phi1(alpha_deg):
    a = np.asarray(alpha_deg, dtype=float)
    lin = 1.0 - (6.0 / np.pi) * np.deg2rad(a)
    spl = _hermite_spline(a, _XI1_X, _XI1_Y, _XI1_D)
    return np.where(a <= 7.5, lin, spl)


def _phi2(alpha_deg):
    a = np.asarray(alpha_deg, dtype=float)
    lin = 1.0 - (9.0 / (5.0 * np.pi)) * np.deg2rad(a)
    spl = _hermite_spline(a, _XI2_X, _XI2_Y, _XI2_D)
    return np.where(a <= 7.5, lin, spl)


def _phi3(alpha_deg):
    a = np.asarray(alpha_deg, dtype=float)
    spl = _hermite_spline(a, _XI3_X, _XI3_Y, _XI3_D)
    return np.where(a <= 30.0, spl, 0.0)


# TODO: Make unittests from this
# Sanity check using table B.4 from page 25 of Pentilla.
# Their value for phi_i(0) is wrong (should be 1, they have 0),
# the rest should match
# for alpha in [0.0, 0.35, 2.0, 5.5, 75]:
#    print(f"Alpha={alpha} phi1={_phi1(alpha)} phi2={_phi2(alpha)} phi3={_phi3(alpha)}")


def hg12star_correction(alpha_deg: np.ndarray, g12star: float) -> np.ndarray:
    """Compute alpha correction using H,G12* approximation.

    Parameters:
    -----------
    alpha_deg: np.ndarray
      angle Sun-object-observer in degrees
    g12star: float
      value of G12* parameter to use for computing G1 and G2

    Returns:
    --------
    Magnitude correction for the given alphas.
    """
    G1 = 0.84293649 * g12star
    G2 = 0.53513350 * (1.0 - g12star)
    G3 = 1.0 - G1 - G2
    combined = G1 * _phi1(alpha_deg) + G2 * _phi2(alpha_deg) + G3 * _phi3(alpha_deg)
    combined = np.maximum(combined, 1e-10)
    return -2.5 * np.log10(combined)
