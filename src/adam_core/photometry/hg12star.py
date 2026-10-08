import numpy as np
import numpy.typing as npt


# Basis functions Phi1, Phi2, Phi3 from Penttila et al. (2016)
# Hermite cubic spline (Appendix A, Eq. A.1).
# xs in degrees; derivatives ds are in d(y)/d(alpha_rad) as given in Penttila Table A.2/A.3.
#
# Interpolation only: x must lie within [xs[0], xs[-1]]. Callers are responsible
# for restricting x to the tabulated range first.
def _hermite_spline(
    x_deg: npt.NDArray[np.float64] | float,
    xs_deg: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    ds_per_rad: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | float:
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


def _phi1(alpha_deg: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    a = np.asarray(alpha_deg, dtype=float)
    lin = 1.0 - (6.0 / np.pi) * np.deg2rad(a)
    spl = _hermite_spline(a, _XI1_X, _XI1_Y, _XI1_D)
    return np.where(a <= 7.5, lin, spl)


def _phi2(alpha_deg: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    a = np.asarray(alpha_deg, dtype=float)
    lin = 1.0 - (9.0 / (5.0 * np.pi)) * np.deg2rad(a)
    spl = _hermite_spline(a, _XI2_X, _XI2_Y, _XI2_D)
    return np.where(a <= 7.5, lin, spl)


def _phi3(alpha_deg: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    a = np.asarray(alpha_deg, dtype=float)
    spl = _hermite_spline(a, _XI3_X, _XI3_Y, _XI3_D)
    return np.where(a <= 30.0, spl, 0.0)


# The Penttila et al. (2016) spline tables end at 150 deg (Table A.2), so the
# HG12* phase function is only defined on [0, 150] deg. The paper gives no
# prescription beyond the last knot, so angles outside the table are rejected
# rather than extrapolated.
_ALPHA_MAX_DEG = 150.0

# Penttila et al. (2016) Eq. 22: the G1/G2 pair is parameterized by the single
# G12* via these coefficients.  Defined once so `hg12star_correction` and its
# derivative below cannot drift apart.
_G1_PER_G12STAR = 0.84293649
_G2_PER_G12STAR = 0.53513350
# The combined basis sum can go non-positive for an unphysical G12* outside
# [0, 1] (as a fitter may try), which would break log10; clip it here.
_PHASE_FLOOR = 1e-10


def _check_alpha_domain(alpha_deg: npt.NDArray[np.float64] | float) -> None:
    """Raise if any phase angle falls outside the tabulated [0, 150] deg range.

    NaN is left alone (it compares False here and propagates to a NaN
    correction); it means "no phase angle", not an out-of-domain one.
    """
    a = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    outside = (a < 0.0) | (a > _ALPHA_MAX_DEG)
    if np.any(outside):
        offending = np.unique(a[outside])
        shown = ", ".join(f"{v:.2f}" for v in offending[:5])
        if len(offending) > 5:
            shown += ", ..."
        raise ValueError(
            "The HG12* phase function is only defined for phase angles in "
            f"[0, {_ALPHA_MAX_DEG:.2f}] deg, where the Penttila et al. (2016) spline "
            f"tables are given; got {int(np.sum(outside))} value(s) outside that "
            f"range: {shown} deg."
        )


def hg12star_correction(
    alpha_deg: npt.NDArray[np.float64] | float, g12star: float
) -> npt.NDArray[np.float64]:
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

    Raises:
    -------
    ValueError
      If any alpha falls outside [0, 150] deg, the range over which the
      Penttila et al. (2016) spline tables define the basis functions.
    """
    _check_alpha_domain(alpha_deg)
    combined, _, _, _ = _hg12star_basis_sum(alpha_deg, g12star)
    combined = np.maximum(combined, _PHASE_FLOOR)
    return -2.5 * np.log10(combined)


def _hg12star_basis_sum(
    alpha_deg: npt.NDArray[np.float64] | float, g12star: float
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """``(G1*phi1 + G2*phi2 + G3*phi3, phi1, phi2, phi3)`` for one ``G12*``."""
    phi1 = _phi1(alpha_deg)
    phi2 = _phi2(alpha_deg)
    phi3 = _phi3(alpha_deg)
    G1 = _G1_PER_G12STAR * g12star
    G2 = _G2_PER_G12STAR * (1.0 - g12star)
    return G1 * phi1 + G2 * phi2 + (1.0 - G1 - G2) * phi3, phi1, phi2, phi3


def _hg12star_correction_dg12star(
    alpha_deg: npt.NDArray[np.float64] | float, g12star: float
) -> npt.NDArray[np.float64]:
    """
    Analytic ``d(hg12star_correction)/d(G12*)`` at ``alpha_deg``.

    Supplying this to a least-squares fit makes the residual Jacobian exact,
    which is what lets a rank test distinguish a genuinely degenerate fit (e.g.
    a single phase angle, where ``G12*`` trades off against the absolute
    magnitude exactly) from a merely ill-conditioned one; a finite-difference
    Jacobian blurs that distinction at the ~1e-8 level.  Zero wherever
    `hg12star_correction` clips at ``_PHASE_FLOOR``, so the two agree on the
    flat region.
    """
    _check_alpha_domain(alpha_deg)
    combined, phi1, phi2, phi3 = _hg12star_basis_sum(alpha_deg, g12star)
    # d(G1)/dg = +_G1_PER_G12STAR, d(G2)/dg = -_G2_PER_G12STAR, and G3 = 1-G1-G2
    # so d(G3)/dg = _G2_PER_G12STAR - _G1_PER_G12STAR.
    d_combined = (
        _G1_PER_G12STAR * phi1
        - _G2_PER_G12STAR * phi2
        + (_G2_PER_G12STAR - _G1_PER_G12STAR) * phi3
    )
    return np.where(
        combined > _PHASE_FLOOR,
        -2.5 / np.log(10.0) * d_combined / np.maximum(combined, _PHASE_FLOOR),
        0.0,
    )
