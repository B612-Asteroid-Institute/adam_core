import warnings
from typing import Any, List, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from jax import config, jacfwd, jit, vmap
from scipy.optimize import least_squares

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.residuals import Residuals
from ..coordinates.transform import transform_coordinates
from ..dynamics.ephemeris import _generate_ephemeris_2body
from ..dynamics.propagation import _propagate_2body
from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from ..time.time import Timestamp
from ..utils.spice import get_perturber_state
from .evaluate import OrbitDeterminationObservations, evaluate_orbits
from .fitted_orbits import FittedOrbitMembers, FittedOrbits
from .observation_uncertainty import ObservationUncertaintyModel
from .outliers import calculate_max_outliers, remove_lowest_probability_observation

config.update("jax_enable_x64", True)

# Bucket size used to pad the analytic-Jacobian inputs so that JAX compiles
# one kernel per bucket rather than one per distinct observation count.
_JACOBIAN_PAD_MULTIPLE = 32

# Acceptance window for the weak-direction consistency check: the measured
# delta-chi2 at a 1-sigma displacement along the covariance's widest axis
# should be ~1 for a trustworthy covariance at a converged minimum.
_DELTA_CHI2_WINDOW = (0.1, 10.0)

# Robust loss functions supported by fit_least_squares. "linear" is ordinary
# weighted least squares (the cost is chi2); "huber" is Huber's M-estimator.
LossType = Literal["linear", "huber"]

# Huber transition point in units of whitened (1-sigma) residual components:
# residuals within +/- f_scale are treated quadratically, larger residuals
# linearly. 1.345 is the classical constant giving 95% asymptotic efficiency
# for Gaussian errors (Huber 1981; Huber & Ronchetti 2009, sec. 4.5).
HUBER_F_SCALE_DEFAULT = 1.345

_ROBUST_SCALE_EPS = np.finfo(np.float64).eps


def _validate_loss(loss: str, f_scale: float) -> None:
    if loss not in ("linear", "huber"):
        raise ValueError(f"loss must be one of 'linear', 'huber'; got {loss!r}")
    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError(f"f_scale must be a positive finite number; got {f_scale!r}")


def _robust_cost(
    residuals: npt.NDArray[np.float64], loss: str, f_scale: float
) -> float:
    """
    Objective minimized by `fit_least_squares` for the given loss, as a
    function of the whitened residual components ``r``.

    - ``"linear"``: ``sum(r**2)``, i.e. chi2.
    - ``"huber"``: ``f_scale**2 * sum(rho(r**2 / f_scale**2))`` with
      ``rho(z) = z`` for ``z <= 1`` and ``rho(z) = 2 sqrt(z) - 1`` otherwise,
      which is twice the ``cost`` reported by `scipy.optimize.least_squares`
      and reduces to chi2 when every component lies inside the quadratic core.
    """
    if loss == "linear":
        return float(residuals @ residuals)
    z = (residuals / f_scale) ** 2
    rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    return float(f_scale**2 * np.sum(rho))


def _robust_weights(
    residuals: npt.NDArray[np.float64], loss: str, f_scale: float
) -> npt.NDArray[np.float64]:
    """
    Iteratively-reweighted-least-squares weight ``rho'(z)`` of each whitened
    residual component at the solution: 1 inside the quadratic core and
    ``f_scale / |r|`` in the linear tail (Huber's psi(r) / r). All ones for
    ``"linear"``.
    """
    if loss == "linear":
        return np.ones_like(residuals)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.minimum(1.0, f_scale / np.abs(residuals))
    return np.asarray(np.where(np.isfinite(weights), weights, 1.0), dtype=np.float64)


def _robust_jacobian_scale(
    residuals: npt.NDArray[np.float64], loss: str, f_scale: float
) -> npt.NDArray[np.float64]:
    """
    Row scaling that turns the Jacobian of the whitened residual vector into
    the Gauss-Newton Jacobian of the robust cost, following the convention of
    `scipy.optimize.least_squares` (``sqrt(rho' + 2 rho'' r**2)``, floored at
    machine epsilon). For ``"linear"`` this is all ones. For ``"huber"`` the
    second derivative of the loss vanishes in the linear tail, so components
    beyond ``f_scale`` contribute (numerically) no curvature: the resulting
    ``inv(J^T J)`` is the covariance of the M-estimate in which downweighted
    observations carry no information, consistent with the local shape of
    the robust cost surface probed by `_weak_direction_delta_chi2`.
    """
    if loss == "linear":
        return np.ones_like(residuals)
    inside = np.abs(residuals) <= f_scale
    return np.asarray(
        np.where(inside, 1.0, np.sqrt(_ROBUST_SCALE_EPS)), dtype=np.float64
    )


def _observation_whitening_matrices(
    observations: OrbitDeterminationObservations,
) -> npt.NDArray[np.float64]:
    """
    Compute per-observation whitening matrices for the angular residuals.

    For each observation, the 2x2 (lon, lat) block of the coordinate
    covariance is corrected for cos(latitude) on the longitude axis (matching
    the convention used by `Residuals.calculate`) and factored as C = L L^T.
    The returned whitener is L^-1: applied to the cos(latitude)-corrected
    (lon, lat) residual vector it yields two components whose sum of squares
    equals the observation's chi2 (Mahalanobis distance squared).

    Missing (NaN) cross terms are treated as zero, matching the convention in
    `~adam_core.coordinates.residuals.calculate_chi2`.

    Parameters
    ----------
    observations : `OrbitDeterminationObservations` (N)
        Observations whose covariances to factor.

    Returns
    -------
    whiteners : `~numpy.ndarray` (N, 2, 2)
        Inverse lower-triangular Cholesky factor per observation.

    Raises
    ------
    ValueError
        If any observation has non-finite angular variances or a
        non-positive-definite (lon, lat) covariance block.
    """
    covariances = observations.coordinates.covariance.to_matrix()
    lat = observations.coordinates.lat.to_numpy(zero_copy_only=False)
    cos_lat = np.cos(np.radians(lat))

    blocks = covariances[:, 1:3, 1:3].copy()
    cross = blocks[:, 0, 1]
    cross = np.where(np.isfinite(cross), cross, 0.0)
    blocks[:, 0, 1] = cross
    blocks[:, 1, 0] = cross
    blocks[:, 0, 0] *= cos_lat**2
    blocks[:, 0, 1] *= cos_lat
    blocks[:, 1, 0] *= cos_lat

    finite = np.isfinite(blocks).all(axis=(1, 2))
    if not np.all(finite):
        bad = int(np.flatnonzero(~finite)[0])
        raise ValueError(
            f"Observation {observations.id[bad].as_py()!r} has non-finite "
            "(lon, lat) covariance entries; least-squares fitting requires "
            "finite angular uncertainties for every observation."
        )

    try:
        cholesky = np.linalg.cholesky(blocks)
    except np.linalg.LinAlgError:
        for i in range(len(blocks)):
            try:
                np.linalg.cholesky(blocks[i])
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"Observation {observations.id[i].as_py()!r} has a "
                    "non-positive-definite (lon, lat) covariance block."
                ) from None
        raise
    return np.linalg.inv(cholesky)


def residual_function(
    state_vector: npt.NDArray[np.float64],
    mjd_tdb: float,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
) -> npt.NDArray[np.float64]:
    """
    Compute the whitened angular residuals for a given Cartesian orbit with
    respect to a set of observations.

    Parameters
    ----------
    state_vector : `~numpy.ndarray` (6,)
        Heliocentric ecliptic state vector of the orbit in Cartesian
        coordinates.
    mjd_tdb : float
        Time of the state vector in MJD TDB.
    observations : `~adam_core.orbit_determination.OrbitDeterminationObservations` (N)
        Observations to compute residuals for.
    propagator : `~adam_core.propagator.Propagator`
        Propagator to use to generate ephemeris.

    Returns
    -------
    residuals : `~numpy.ndarray` (2N,)
        Whitened (lon, lat) residual components, two per observation: the
        cos(latitude)-corrected (observed - predicted) angular residual
        multiplied by the inverse Cholesky factor of the matching observation
        covariance block. The sum of squares over one observation's pair
        equals that observation's chi2 (Mahalanobis distance squared), so the
        least-squares objective matches the historical per-observation
        sqrt(chi2) convention while remaining differentiable at zero residual
        and preserving the direction of each residual on the sky.
    """
    # Generate ephemeris and compute residuals
    orbit = Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=state_vector[0:1],
            y=state_vector[1:2],
            z=state_vector[2:3],
            vx=state_vector[3:4],
            vy=state_vector[4:5],
            vz=state_vector[5:6],
            time=Timestamp.from_mjd([mjd_tdb], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        )
    )
    ephemeris = propagator.generate_ephemeris(
        orbit, observations.observers, max_processes=1
    )
    residuals = Residuals.calculate(observations.coordinates, ephemeris.coordinates)

    # Residuals.calculate returns (observed - predicted) values with the
    # longitude residual wrapped to [-180, 180] and cos(latitude)-corrected.
    # Whitening those (lon, lat) pairs with the matching covariance factor
    # yields components r such that sum(r**2) == sum(chi2).
    values = np.stack(residuals.values.to_numpy(zero_copy_only=False))[:, 1:3]
    whiteners = _observation_whitening_matrices(observations)
    whitened = np.einsum("nij,nj->ni", whiteners, values)
    return np.asarray(whitened, dtype=np.float64).reshape(-1)


@jit
def _whitened_model_angles_2body(
    state_vector: jnp.ndarray,
    epoch_mjd_tdb: float,
    times_mjd_tdb: jnp.ndarray,
    observer_states: jnp.ndarray,
    sun_states: jnp.ndarray,
    cos_lats: jnp.ndarray,
    whiteners: jnp.ndarray,
    mu_helio: float,
    mu_bary: float,
) -> jnp.ndarray:
    """
    Predicted whitened (lon, lat) angles from the 2-body model as a flat
    (2N,) vector.

    The heliocentric state is propagated with the universal-anomaly 2-body
    propagator from the fit epoch to each observation time, translated to the
    barycenter, and projected to topocentric equatorial spherical coordinates
    including light-time iteration (mirroring `generate_ephemeris_2body`).
    Each (lon, lat) pair is then cos(latitude)-corrected and whitened exactly
    like the residuals in `residual_function`, so the negated Jacobian of
    this function with respect to `state_vector` is the Jacobian of the
    whitened residual vector: the observation partials chained with the
    2-body state transition matrix.
    """

    def one(
        time: jnp.ndarray,
        observer_state: jnp.ndarray,
        sun_state: jnp.ndarray,
        cos_lat: jnp.ndarray,
        whitener: jnp.ndarray,
    ) -> jnp.ndarray:
        propagated = _propagate_2body(state_vector, epoch_mjd_tdb, time, mu_helio)
        ephemeris, _, _ = _generate_ephemeris_2body(
            propagated + sun_state, time, observer_state, mu_bary
        )
        corrected = jnp.stack([ephemeris[1] * cos_lat, ephemeris[2]])
        return jnp.asarray(whitener @ corrected)

    return vmap(one)(
        times_mjd_tdb, observer_states, sun_states, cos_lats, whiteners
    ).reshape(-1)


_whitened_model_angles_2body_jacobian = jit(
    jacfwd(_whitened_model_angles_2body, argnums=0)
)


def _analytic_jacobian_terms(
    observations: OrbitDeterminationObservations,
) -> dict[str, Any]:
    """
    Precompute the observation-dependent constants of the analytic 2-body
    Jacobian: observation times, barycentric observer and Sun states, and the
    whitening factors. Inputs are padded to a multiple of
    `_JACOBIAN_PAD_MULTIPLE` (padded rows have zero whiteners, so their
    Jacobian rows vanish and are sliced off) to limit JAX recompilation
    across arc sizes.
    """
    observers = observations.observers
    times_tdb = observers.coordinates.time.rescale("tdb")
    times = times_tdb.mjd().to_numpy(zero_copy_only=False)
    observer_states = transform_coordinates(
        observers.coordinates,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    ).values
    sun_states = get_perturber_state(
        OriginCodes.SUN,
        times_tdb,
        frame="ecliptic",
        origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    ).values
    lat = observations.coordinates.lat.to_numpy(zero_copy_only=False)
    cos_lats = np.cos(np.radians(lat))
    whiteners = _observation_whitening_matrices(observations)

    n = len(observations)
    n_padded = -(-n // _JACOBIAN_PAD_MULTIPLE) * _JACOBIAN_PAD_MULTIPLE
    pad = n_padded - n
    if pad:
        times = np.concatenate([times, np.repeat(times[-1:], pad)])
        observer_states = np.concatenate(
            [observer_states, np.repeat(observer_states[-1:], pad, axis=0)]
        )
        sun_states = np.concatenate(
            [sun_states, np.repeat(sun_states[-1:], pad, axis=0)]
        )
        cos_lats = np.concatenate([cos_lats, np.ones(pad)])
        whiteners = np.concatenate([whiteners, np.zeros((pad, 2, 2))])

    return {
        "n": n,
        "times": jnp.asarray(times),
        "observer_states": jnp.asarray(observer_states),
        "sun_states": jnp.asarray(sun_states),
        "cos_lats": jnp.asarray(cos_lats),
        "whiteners": jnp.asarray(whiteners),
        "mu_helio": float(Origin.from_kwargs(code=["SUN"]).mu()[0]),
        "mu_bary": float(Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"]).mu()[0]),
    }


def _analytic_jacobian(
    state_vector: npt.NDArray[np.float64],
    epoch_mjd_tdb: float,
    terms: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """
    Analytic-quality Jacobian of the whitened residual vector with respect to
    the epoch state: d(obs)/dx(t) chained through the 2-body state transition
    matrix, computed by forward-mode automatic differentiation. Residuals are
    (observed - predicted), hence the negation of the model Jacobian.
    """
    model_jacobian = _whitened_model_angles_2body_jacobian(
        jnp.asarray(state_vector),
        epoch_mjd_tdb,
        terms["times"],
        terms["observer_states"],
        terms["sun_states"],
        terms["cos_lats"],
        terms["whiteners"],
        terms["mu_helio"],
        terms["mu_bary"],
    )
    return -np.asarray(model_jacobian)[: 2 * terms["n"]]


def _central_difference_jacobian(
    state_vector: npt.NDArray[np.float64],
    mjd_tdb: float,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    rel_step: float = 1e-6,
    abs_step_floor: float = 1e-3,
) -> npt.NDArray[np.float64]:
    """
    Central-difference Jacobian of the whitened residual vector computed
    through the full residual pipeline (N-body propagation via `propagator`).

    Step sizes are h_k = rel_step * max(|x_k|, abs_step_floor), which produce
    covariances stable to four decimals over rel_step in [1e-7, 1e-5] on
    angles-only NEO arcs. This is the fallback covariance path when the
    2-body analytic Jacobian cannot be trusted, e.g. when the arc contains a
    planetary encounter that the 2-body state transition matrix does not
    model.
    """
    n_params = len(state_vector)
    jacobian = np.empty((2 * len(observations), n_params), dtype=np.float64)
    for k in range(n_params):
        step = rel_step * max(abs(state_vector[k]), abs_step_floor)
        plus = state_vector.copy()
        minus = state_vector.copy()
        plus[k] += step
        minus[k] -= step
        jacobian[:, k] = (
            residual_function(plus, mjd_tdb, observations, propagator)
            - residual_function(minus, mjd_tdb, observations, propagator)
        ) / (2.0 * step)
    return jacobian


def _weak_direction_delta_chi2(
    state_vector: npt.NDArray[np.float64],
    mjd_tdb: float,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    covariance_matrix: npt.NDArray[np.float64],
    cost_solution: float,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> float:
    """
    Measure the actual change in the fit cost at a 1-sigma displacement along
    the covariance's weakest-constrained direction.

    The cost is chi2 for ``loss="linear"`` and the robust cost of
    `_robust_cost` otherwise (``cost_solution`` must be that cost at
    ``state_vector``). If the covariance is trustworthy and the solution sits
    at the minimum of a locally quadratic cost surface, this is ~1 by
    construction (delta = sigma^2 * v^T C^-1 v = 1 for the eigenpair
    (sigma^2, v)). Values far below 1 reproduce the fabricated-confidence
    failure mode where the cost valley is flat over many claimed sigma;
    values far above 1 indicate the covariance overstates the uncertainty.
    The displacement is applied symmetrically so that a residual gradient
    (incomplete convergence) mostly cancels.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sigma = float(np.sqrt(eigenvalues[-1]))
    direction = eigenvectors[:, -1]

    residuals_plus = residual_function(
        state_vector + sigma * direction, mjd_tdb, observations, propagator
    )
    residuals_minus = residual_function(
        state_vector - sigma * direction, mjd_tdb, observations, propagator
    )
    cost_plus = _robust_cost(residuals_plus, loss, f_scale)
    cost_minus = _robust_cost(residuals_minus, loss, f_scale)
    return 0.5 * (cost_plus + cost_minus) - cost_solution


def _validated_covariance(
    covariance_matrix: npt.NDArray[np.float64],
    jacobian_method: str,
    state_vector: npt.NDArray[np.float64],
    mjd_tdb: float,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    cost_solution: float,
    loss: str = "linear",
    f_scale: float = 1.0,
    residuals_solution: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Run the weak-direction consistency check on a fit covariance and, for the
    analytic Jacobian path, fall back to the central-difference covariance
    when the check fails (the signature of a dynamics regime — such as a
    planetary encounter inside the arc — that the 2-body state transition
    matrix does not capture).

    ``cost_solution`` is the fit cost (chi2, or the robust cost for a robust
    ``loss``) at ``state_vector``. For a robust loss the central-difference
    fallback Jacobian is row-scaled with `_robust_jacobian_scale` using
    ``residuals_solution`` (recomputed if not given).
    """
    lower, upper = _DELTA_CHI2_WINDOW
    try:
        delta_chi2 = _weak_direction_delta_chi2(
            state_vector,
            mjd_tdb,
            observations,
            propagator,
            covariance_matrix,
            cost_solution,
            loss=loss,
            f_scale=f_scale,
        )
    except Exception as e:
        warnings.warn(
            "The fit covariance could not be validated: evaluating residuals "
            f"at a 1-sigma displacement along its weakest direction failed ({e}). "
            "The covariance may be unreliable in weakly constrained directions.",
            category=RuntimeWarning,
        )
        return covariance_matrix

    if lower <= delta_chi2 <= upper:
        return covariance_matrix

    if jacobian_method == "analytic":
        warnings.warn(
            "The analytic (2-body) fit covariance failed the weak-direction "
            f"consistency check (measured delta-chi2 = {delta_chi2:.3g} at a "
            "1-sigma displacement along the weakest axis; expected ~1). This "
            "typically indicates dynamics inside the arc that the 2-body "
            "state transition matrix does not model (e.g. a planetary "
            "encounter). Falling back to a central-difference Jacobian "
            "computed through the full residual pipeline.",
            category=RuntimeWarning,
        )
        jacobian = _central_difference_jacobian(
            state_vector, mjd_tdb, observations, propagator
        )
        if loss != "linear":
            if residuals_solution is None:
                residuals_solution = residual_function(
                    state_vector, mjd_tdb, observations, propagator
                )
            jacobian = (
                jacobian
                * _robust_jacobian_scale(residuals_solution, loss, f_scale)[:, None]
            )
        try:
            covariance_matrix = np.asarray(
                np.linalg.inv(jacobian.T @ jacobian), dtype=np.float64
            )
        except np.linalg.LinAlgError:
            warnings.warn(
                "The central-difference fallback covariance could not be "
                "computed. The solution covariance may be unreliable.",
                category=RuntimeWarning,
            )
            return covariance_matrix
        return _validated_covariance(
            covariance_matrix,
            "central",
            state_vector,
            mjd_tdb,
            observations,
            propagator,
            cost_solution,
            loss=loss,
            f_scale=f_scale,
            residuals_solution=residuals_solution,
        )

    if jacobian_method in ("2-point", "solver"):
        warnings.warn(
            "The fit covariance failed the weak-direction consistency check "
            f"(measured delta-chi2 = {delta_chi2:.3g} at a 1-sigma displacement "
            "along the weakest axis; expected ~1). The covariance was computed "
            "from the solver's forward-difference Jacobian, whose implied steps "
            "can sit below the numerical noise floor of the residual pipeline; "
            "in weakly constrained (line-of-sight) directions the resulting "
            "confidence is fabricated by finite-difference noise. Use "
            "jacobian='analytic' or jacobian='central' instead.",
            category=RuntimeWarning,
        )
    else:
        warnings.warn(
            "The fit covariance failed the weak-direction consistency check "
            f"(measured delta-chi2 = {delta_chi2:.3g} at a 1-sigma displacement "
            "along the weakest axis; expected ~1). The covariance may be "
            "unreliable in weakly constrained directions (the chi2 surface "
            "disagrees with the local quadratic model, e.g. because the "
            "solution has not fully converged along a flat valley).",
            category=RuntimeWarning,
        )
    return covariance_matrix


def fit_least_squares(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    ignore: Optional[List[str]] = None,
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
    jacobian: Literal["analytic", "central", "2-point"] = "analytic",
    validate_covariance: bool = True,
    loss: LossType = "linear",
    f_scale: float = HUBER_F_SCALE_DEFAULT,
    **kwargs,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Differentially correct an orbit using (optionally robust) least squares.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (1)
        Orbit to differentially correct (heliocentric ecliptic Cartesian).
    observations : `~adam_core.orbit_determination.OrbitDeterminationObservations` (N)
        Observations.
    propagator : `~adam_core.propagator.Propagator`
        Propagator to use to generate ephemeris.
    ignore : list of str
        List of observation IDs to ignore when fitting the orbit with least squares.
        These observations will be marked as outliers in the fitted orbit members.
    observatory_bias_model : `~adam_core.orbit_determination.ObservationUncertaintyModel`, optional
        Observation uncertainty model applied to the observations before fitting
        (e.g. inflating per-station sigmas from an observatory bias table). Default
        None leaves the observations unchanged. The model is applied once at this
        entry point and is not forwarded to nested calls.
    jacobian : {"analytic", "central", "2-point"}, optional
        How the residual Jacobian is obtained for both the solver and the
        solution covariance.

        - ``"analytic"`` (default): the observation partials (topocentric
          spherical projection including light time) chained with the 2-body
          state transition matrix, computed by forward-mode automatic
          differentiation and passed to `scipy.optimize.least_squares` as an
          exact Jacobian callable. Residual values remain full N-body through
          `propagator`. This removes the finite-difference noise that
          fabricates curvature — and hence collapses the covariance — in
          weakly constrained (line-of-sight) directions, and improves
          convergence along flat chi2 valleys.
        - ``"central"``: the solver uses its default forward differences, and
          the covariance is recomputed after the fit from a central-difference
          Jacobian of the whitened residual vector through the full residual
          pipeline. Use this when the arc contains dynamics the 2-body state
          transition matrix cannot represent (e.g. a deep planetary
          encounter).
        - ``"2-point"``: legacy behavior; the covariance is taken from the
          solver's forward-difference Jacobian. Retained for comparison; its
          covariance is untrustworthy in weakly constrained directions.

        Passing an explicit ``jac`` in ``**kwargs`` overrides this parameter
        and computes the covariance from the solver-reported Jacobian.
    validate_covariance : bool, optional
        If True (default), measure the actual chi2 change at a 1-sigma
        displacement along the covariance's weakest direction (two extra
        residual evaluations) and check it is ~1. On failure the analytic
        path falls back to the central-difference covariance; other paths
        emit a `RuntimeWarning`. This also serves as the guard against
        planetary encounters inside the arc invalidating the 2-body state
        transition matrix.
    loss : {"linear", "huber"}, optional
        Loss applied to the whitened residual components.

        - ``"linear"`` (default): ordinary weighted least squares; the
          objective is chi2. Outliers are handled only by hard rejection
          (``ignore`` here, the rejection loop in `iterative_fit`).
        - ``"huber"``: Huber's M-estimator. Components within ``f_scale``
          (1-sigma units) contribute quadratically, larger ones linearly, so
          a gross outlier's pull on the solution is bounded instead of
          growing with its residual. No observation is removed: every
          observation stays in the solution with an iteratively-reweighted
          least-squares weight of ``min(1, f_scale / |r|)`` per component,
          reported on the members (see Notes). Requires
          ``method='trf'`` (default) or ``'dogbox'``.
    f_scale : float, optional
        Huber transition point in units of whitened (1-sigma) residual
        components. Default `HUBER_F_SCALE_DEFAULT` (1.345), the classical
        constant giving 95% asymptotic efficiency for Gaussian errors.
        Ignored for ``loss="linear"``.
    **kwargs
        Additional keyword arguments to pass to `~scipy.optimize.least_squares`
        (``loss`` and ``f_scale`` are the named parameters above).
        Some of these parameters if not specified will be set to sensible defaults.
            xtol = 1e-12
            ftol = 1e-12
            gtol = 1e-12
            x_scale = "jac"
            bounds = (-np.inf, np.inf) (for each parameter)

    Returns
    -------
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (1)
        Fitted orbit.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N)
        Fitted orbit members. ``weight`` holds each observation's effective
        weight in the solution: 0 for ignored observations, 1 for fully
        weighted ones and, for ``loss="huber"``, the smaller of the two
        per-component Huber weights for downweighted observations.

    Notes
    -----
    With ``loss="huber"`` the reported ``chi2`` / ``reduced_chi2`` remain the
    plain (unweighted) chi2 of the included observations, so they stay
    comparable across losses; the minimized robust cost is not stored. The
    covariance is ``inv(J^T J)`` of the Gauss-Newton Jacobian of the robust
    cost (the convention of `scipy.optimize.least_squares`): components in
    the linear tail of the loss contribute no curvature, i.e. downweighted
    observations carry no information in the covariance, and the
    weak-direction consistency check probes the robust cost instead of chi2.
    """
    assert len(orbit) == 1, "Only one orbit can be differentially corrected"
    _validate_loss(loss, f_scale)

    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)

    # TODO: Investigate whether we want to add fitting for the epoch as well
    # Set up least squares problem
    if ignore is not None:
        mask = pc.invert(pc.is_in(observations.id, pa.array(ignore)))
        observations_to_include = observations.apply_mask(mask)
    else:
        observations_to_include = observations

    observed_values = observations_to_include.coordinates.values
    if np.any(np.isfinite(observed_values[:, [0, 3, 4, 5]])):
        raise ValueError(
            "fit_least_squares only supports angular (lon, lat) observations; "
            "found finite values in rho or velocity dimensions."
        )

    parameters = 6
    # Extract epoch and state vector from orbit
    epoch = orbit.coordinates.time.rescale("tdb").mjd().to_numpy(zero_copy_only=False)
    state_vector = orbit.coordinates.values[0]
    args = (epoch[0], observations_to_include, propagator)

    # Define some sensible defaults for the least squares fitting procedure
    if "xtol" not in kwargs:
        kwargs["xtol"] = 1e-12
    if "ftol" not in kwargs:
        kwargs["ftol"] = 1e-12
    if "gtol" not in kwargs:
        kwargs["gtol"] = 1e-12
    if "x_scale" not in kwargs:
        kwargs["x_scale"] = "jac"
    if "bounds" not in kwargs:
        kwargs["bounds"] = (
            np.full(parameters, -np.inf),
            np.full(parameters, np.inf),
        )
    if "args" in kwargs:
        kwargs.pop("args")
        warnings.warn(
            "The args parameter is not supported and will be ignored.",
            category=RuntimeWarning,
        )
    if loss != "linear":
        if kwargs.get("method") == "lm":
            raise ValueError(
                f"loss={loss!r} is not supported by method='lm'; use method='trf' "
                "(default) or 'dogbox'."
            )
        kwargs["loss"] = loss
        kwargs["f_scale"] = f_scale

    jacobian_method: str = jacobian
    if "jac" in kwargs:
        jacobian_method = "solver"
        warnings.warn(
            "An explicit jac was passed to least_squares; the jacobian "
            "parameter is ignored and the covariance is computed from the "
            "solver-reported Jacobian.",
            category=RuntimeWarning,
        )
    elif jacobian == "analytic":
        terms = _analytic_jacobian_terms(observations_to_include)
        epoch_mjd_tdb = epoch[0]

        def jac(x: npt.NDArray[np.float64], *_args: object) -> npt.NDArray[np.float64]:
            return _analytic_jacobian(x, epoch_mjd_tdb, terms)

        kwargs["jac"] = jac
    elif jacobian in ("central", "2-point"):
        kwargs["jac"] = "2-point"
    else:
        raise ValueError(
            f"jacobian must be one of 'analytic', 'central', '2-point'; "
            f"got {jacobian!r}"
        )

    # Run least squares
    solution = least_squares(residual_function, state_vector, args=args, **kwargs)

    # Extract solution state vector and covariance matrix
    mjd_tdb = epoch[0]
    x, y, z, vx, vy, vz = solution.x

    # Whitened residual components at the solution (unscaled by the loss) and
    # the value of the minimized objective (chi2 for the linear loss).
    residuals_solution = np.asarray(solution.fun, dtype=np.float64)
    cost_solution = _robust_cost(residuals_solution, loss, f_scale)

    if jacobian_method == "central":
        solution_jacobian = _central_difference_jacobian(
            solution.x, mjd_tdb, observations_to_include, propagator
        )
        if loss != "linear":
            # Match the solver's convention: the covariance is that of the
            # robust cost, whose Gauss-Newton Jacobian is the residual
            # Jacobian row-scaled by the loss curvature.
            solution_jacobian = (
                solution_jacobian
                * _robust_jacobian_scale(residuals_solution, loss, f_scale)[:, None]
            )
    else:
        # For a robust loss, scipy reports the Jacobian already scaled for
        # the loss (J^T J approximates the Hessian of the robust cost).
        solution_jacobian = solution.jac

    try:
        covariance_matrix = np.asarray(
            np.linalg.inv(solution_jacobian.T @ solution_jacobian),
            dtype=np.float64,
        )
    except np.linalg.LinAlgError:
        warnings.warn(
            "The covariance matrix could not be computed. The solution may be "
            "unreliable.",
            category=RuntimeWarning,
        )
        covariance_matrix = np.full((6, 6), np.nan)

    if validate_covariance and np.all(np.isfinite(covariance_matrix)):
        covariance_matrix = _validated_covariance(
            covariance_matrix,
            jacobian_method,
            solution.x,
            mjd_tdb,
            observations_to_include,
            propagator,
            cost_solution,
            loss=loss,
            f_scale=f_scale,
            residuals_solution=residuals_solution,
        )

    # Create orbit with solution state vector and use it to generate ephemeris
    # and calculate the residuals with respect to the observations
    orbit = Orbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[x],
            y=[y],
            z=[z],
            vx=[vx],
            vy=[vy],
            vz=[vz],
            time=Timestamp.from_mjd([mjd_tdb], scale="tdb"),
            covariance=CoordinateCovariances.from_matrix(
                covariance_matrix.reshape(1, 6, 6)
            ),
            origin=orbit.coordinates.origin,
            frame=orbit.coordinates.frame,
        ),
    )

    # Evaluate the solution orbit and return it as a fitted orbit and fitted orbit members
    # which contain the residuals with respect to the observations and the overall
    # quality of the fit
    fitted_orbit, fitted_orbit_members = evaluate_orbits(
        orbit,
        observations,
        propagator,
        parameters=parameters,
        ignore=ignore,
    )
    fitted_orbit = (
        fitted_orbit.set_column("iterations", [solution.nfev])
        .set_column("success", [solution.success])
        .set_column("status_code", [solution.status])
    )
    fitted_orbit_members = fitted_orbit_members.set_column(
        "solution", pc.invert(fitted_orbit_members.outlier)
    )

    # Effective weight of each observation in the solution: the smaller of its
    # two per-component IRLS weights (all ones for the linear loss), and 0 for
    # ignored observations. Members are in the order of `observations`.
    component_weights = _robust_weights(residuals_solution, loss, f_scale)
    included_weights = component_weights.reshape(-1, 2).min(axis=1)
    weights = np.zeros(len(observations), dtype=np.float64)
    if ignore is not None:
        weights[mask.to_numpy(zero_copy_only=False)] = included_weights
    else:
        weights[:] = included_weights
    fitted_orbit_members = fitted_orbit_members.set_column("weight", weights)

    return fitted_orbit, fitted_orbit_members


def iterative_fit(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    rchi2_threshold: float = 10.0,
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20.0,
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
    loss: LossType = "linear",
    f_scale: float = HUBER_F_SCALE_DEFAULT,
    **kwargs,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Iteratively fit an orbit using least squares with outlier rejection.

    Wraps `fit_least_squares` with an outlier rejection loop: after each fit,
    if the reduced chi2 exceeds `rchi2_threshold`, the observation with the
    worst residual is removed and the fit is repeated. This continues until
    the fit converges, no more outliers are allowed, or arc length / minimum
    observation constraints would be violated.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (1)
        Initial orbit to differentially correct.
    observations : `~adam_core.orbit_determination.OrbitDeterminationObservations` (N)
        Observations to fit against.
    propagator : `~adam_core.propagator.Propagator`
        Propagator to use to generate ephemeris.
    rchi2_threshold : float, optional
        Reduced chi2 threshold below which the fit is considered converged.
        Default is 10.0.
    min_obs : int, optional
        Minimum number of observations required to retain the fit.
        Default is 6.
    min_arc_length : float, optional
        Minimum arc length in days required to retain the fit.
        Default is 1.0.
    contamination_percentage : float, optional
        Maximum percentage of observations that may be rejected as outliers.
        Range is [0, 100]. Default is 20.0.
    observatory_bias_model : `~adam_core.orbit_determination.ObservationUncertaintyModel`, optional
        Observation uncertainty model applied to the observations before fitting
        (e.g. inflating per-station sigmas from an observatory bias table). Default
        None leaves the observations unchanged. The model is applied once at this
        entry point and is not forwarded to nested calls.
    loss : {"linear", "huber"}, optional
        Loss used by every `fit_least_squares` call. ``"huber"`` bounds the
        influence of large residuals by downweighting them instead of (or in
        addition to) removing observations; see `fit_least_squares`. To use
        Huber M-estimation as the sole outlier treatment, pass
        ``contamination_percentage=0.0`` so that the hard-rejection loop is
        disabled; with a non-zero contamination the two compose (a robust fit
        at every pass, with observations still removed while the reduced chi2
        exceeds ``rchi2_threshold``).
    f_scale : float, optional
        Huber transition point in units of whitened (1-sigma) residual
        components; see `fit_least_squares`. Ignored for ``loss="linear"``.
    **kwargs
        Additional keyword arguments passed to `fit_least_squares` (including
        `jacobian` and `validate_covariance`) and ultimately to
        `~scipy.optimize.least_squares`.

    Returns
    -------
    fitted_orbit : `~adam_core.orbit_determination.FittedOrbits` (1)
        Best fitted orbit found.
    fitted_orbit_members : `~adam_core.orbit_determination.FittedOrbitMembers` (N)
        Fitted orbit members with residuals, outlier flags and weights.
    """
    assert len(orbit) == 1, "Only one orbit can be iteratively fitted"

    if observatory_bias_model is not None:
        # Applied once here; the inflated observations (not the model) are
        # passed to the nested fit_least_squares calls below.
        observations = observatory_bias_model.apply(observations)

    num_obs = len(observations)
    max_outliers = calculate_max_outliers(num_obs, min_obs, contamination_percentage)

    ignore: List[str] = []
    best_fitted_orbit = None
    best_fitted_orbit_members = None

    for _ in range(max_outliers + 1):
        fitted_orbit, fitted_orbit_members = fit_least_squares(
            orbit,
            observations,
            propagator,
            ignore=ignore if ignore else None,
            loss=loss,
            f_scale=f_scale,
            **kwargs,
        )

        # Track the best fit seen so far (lowest reduced chi2 among successful fits)
        if best_fitted_orbit is None or (
            fitted_orbit.success[0].as_py()
            and fitted_orbit.reduced_chi2[0].as_py()
            < best_fitted_orbit.reduced_chi2[0].as_py()
        ):
            best_fitted_orbit = fitted_orbit
            best_fitted_orbit_members = fitted_orbit_members

        # Check convergence
        rchi2 = fitted_orbit.reduced_chi2[0].as_py()
        if rchi2 is not None and rchi2 <= rchi2_threshold:
            break

        # Stop if we've already used up all allowed outlier slots
        if len(ignore) >= max_outliers:
            break

        # Identify the worst non-outlier observation among the current solution members
        solution_members = fitted_orbit_members.apply_mask(
            pc.equal(fitted_orbit_members.outlier, False)
        )
        if len(solution_members) == 0:
            break

        obs_id, remaining_observations = remove_lowest_probability_observation(
            solution_members, observations
        )

        # Check that removing this observation still leaves enough arc length
        arc_length = remaining_observations.coordinates.time.mjd().to_numpy()
        if (
            len(arc_length) < min_obs
            or (arc_length.max() - arc_length.min()) < min_arc_length
        ):
            break

        ignore.append(obs_id)

    return best_fitted_orbit, best_fitted_orbit_members
