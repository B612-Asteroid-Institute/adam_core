import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import quivr as qv
import ray
from ray import ObjectRef

from .._rust.api import (
    generate_ephemeris_2body_numpy as rust_generate_ephemeris_2body_numpy,
    generate_ephemeris_2body_with_covariance_numpy as rust_generate_ephemeris_2body_with_covariance_numpy,
)
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..photometry.magnitude import (
    calculate_apparent_magnitude_v,
    calculate_apparent_magnitude_v_and_phase_angle,
    calculate_phase_angle,
)
from ..ray_cluster import initialize_use_ray
from ..utils.iter import _iterate_chunks
from .exceptions import DynamicsNumericalError

def _first_non_finite(values: np.ndarray) -> Optional[int]:
    bad = np.flatnonzero(~np.isfinite(values))
    return int(bad[0]) if bad.size > 0 else None


def _raise_ephemeris_numerical_error(
    *,
    reason: str,
    row_index: int,
    orbit_id: str,
    object_id: str,
    observation_time: float,
    light_time: Optional[float],
    max_iter: int,
    tol: float,
    lt_tol: float,
) -> None:
    raise DynamicsNumericalError(
        stage="ephemeris",
        reason=reason,
        context={
            "row_index": row_index,
            "orbit_id": orbit_id,
            "object_id": object_id,
            "observation_time_mjd_tdb": float(observation_time),
            "light_time_days": None if light_time is None else float(light_time),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "lt_tol": float(lt_tol),
        },
    )


def _generate_ephemeris_2body_serial(
    propagated_orbits: Orbits,
    observers: Observers,
    *,
    lt_tol: float,
    max_iter: int,
    tol: float,
    stellar_aberration: bool,
    predict_magnitudes: bool,
    predict_phase_angle: bool,
) -> Ephemeris:
    # Delegate to the public function's existing implementation, but without Ray.
    return generate_ephemeris_2body(
        propagated_orbits,
        observers,
        lt_tol=lt_tol,
        max_iter=max_iter,
        tol=tol,
        stellar_aberration=stellar_aberration,
        predict_magnitudes=predict_magnitudes,
        predict_phase_angle=predict_phase_angle,
        max_processes=1,
    )


@ray.remote
def ephemeris_2body_worker_ray(
    start: int,
    idx_chunk: np.ndarray,
    propagated_orbits: Orbits,
    observers: Observers,
    lt_tol: float,
    max_iter: int,
    tol: float,
    stellar_aberration: bool,
    predict_magnitudes: bool,
    predict_phase_angle: bool,
) -> Tuple[int, Ephemeris]:
    prop_chunk = propagated_orbits.take(idx_chunk)
    obs_chunk = observers.take(idx_chunk)
    eph = _generate_ephemeris_2body_serial(
        prop_chunk,
        obs_chunk,
        lt_tol=lt_tol,
        max_iter=max_iter,
        tol=tol,
        stellar_aberration=stellar_aberration,
        predict_magnitudes=predict_magnitudes,
        predict_phase_angle=predict_phase_angle,
    )
    return start, eph


def generate_ephemeris_2body(
    propagated_orbits: Orbits,
    observers: Observers,
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    predict_magnitudes: bool = True,
    *,
    predict_phase_angle: bool = False,
    max_processes: Optional[int] = 1,
    chunk_size: int = 100,
) -> Ephemeris:
    """
    Generate on-sky ephemerides for each propagated orbit as viewed by the observers.
    This function calculates the light time delay between the propagated orbit and the observer,
    and then propagates the orbit backward by that amount to when the light from object was actually
    emitted towards the observer ("astrometric coordinates").

    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true location, this is known as
    stellar aberration (often referred to in combination with other aberrations as "apparent
    coordinates"). Stellar aberration can optionally be applied after
    light time correction has been added but it should not be necessary when comparing to ephemerides
    of solar system small bodies extracted from astrometric catalogs. The stars to which the
    catalog is calibrated undergo the same aberration as the moving objects as seen from the observer.

    If stellar aberration is applied then the velocity of the input orbits are unmodified, only the position
    vector is modified with stellar aberration.

    For more details on aberrations see:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/abcorr.html
        https://ssd.jpl.nasa.gov/horizons/manual.html#defs


    Parameters
    ----------
    propagated_orbits : `~adam_core.orbits.orbits.Orbits` (N)
        Propagated orbits.
    observers : `~adam_core.observers.observers.Observers` (N)
        Observers for which to generate ephemerides. Orbits should already have been
        propagated to the same times as the observers.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days).
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson
        method.
    stellar_aberration : bool, optional
        Apply stellar aberration to the ephemerides.

    Returns
    -------
    ephemeris : `~adam_core.orbits.ephemeris.Ephemeris` (N)
        Topocentric ephemerides for each propagated orbit as observed by the given observers.
    """
    if max_processes is None:
        max_processes = mp.cpu_count()

    if max_processes > 1:
        initialize_use_ray(num_cpus=max_processes)

        num_entries = len(observers)
        assert len(propagated_orbits) == num_entries

        propagated_ref = ray.put(propagated_orbits)  # type: ignore[name-defined]
        observers_ref = ray.put(observers)  # type: ignore[name-defined]

        idx = np.arange(0, num_entries, dtype=np.int64)
        pending: List["ObjectRef"] = []  # type: ignore[name-defined]
        results: Dict[int, Ephemeris] = {}

        for idx_chunk in _iterate_chunks(idx, chunk_size):
            start = int(idx_chunk[0]) if len(idx_chunk) else 0
            pending.append(
                ephemeris_2body_worker_ray.remote(  # type: ignore[name-defined]
                    start,
                    idx_chunk,
                    propagated_ref,
                    observers_ref,
                    lt_tol,
                    max_iter,
                    tol,
                    stellar_aberration,
                    predict_magnitudes,
                    predict_phase_angle,
                )
            )

            if len(pending) >= max_processes * 1.5:
                finished, pending = ray.wait(pending, num_returns=1)  # type: ignore[name-defined]
                start_i, eph_i = ray.get(finished[0])  # type: ignore[name-defined]
                results[int(start_i)] = eph_i

        while pending:
            finished, pending = ray.wait(pending, num_returns=1)  # type: ignore[name-defined]
            start_i, eph_i = ray.get(finished[0])  # type: ignore[name-defined]
            results[int(start_i)] = eph_i

        chunks = [results[k] for k in sorted(results.keys())]
        return qv.concatenate(chunks) if chunks else Ephemeris.empty()

    num_entries = len(observers)
    assert (
        len(propagated_orbits) == num_entries
    ), "Orbits and observers must be paired and orbits must be propagated to observer times."

    # Transform both the orbits and observers to the barycenter if they are not already.
    #
    # Fast path: common workload uses SUN/ecliptic for both, on an aligned time grid.
    # In that case we can compute the SUN->SSB translation vectors once and apply them
    # to both orbits and observers (strictly equivalent, but avoids duplicate work).
    propagated_orbits_barycentric = None
    observers_barycentric = None
    try:
        po = propagated_orbits.coordinates
        obc = observers.coordinates
        po_origin = po.origin.code.to_numpy(zero_copy_only=False)
        ob_origin = obc.origin.code.to_numpy(zero_copy_only=False)
        if (
            str(po.frame) == "ecliptic"
            and str(obc.frame) == "ecliptic"
            and np.all(po_origin == OriginCodes.SUN.name)
            and np.all(ob_origin == OriginCodes.SUN.name)
        ):
            t_po = po.time.rescale("tdb")
            t_ob = obc.time.rescale("tdb")
            same_time = np.array_equal(
                t_po.days.to_numpy(zero_copy_only=False),
                t_ob.days.to_numpy(zero_copy_only=False),
            ) and np.array_equal(
                t_po.nanos.to_numpy(zero_copy_only=False),
                t_ob.nanos.to_numpy(zero_copy_only=False),
            )
            if same_time:
                from ..utils.spice import get_perturber_state

                sun_wrt_ssb = get_perturber_state(
                    OriginCodes.SUN,
                    t_po,
                    frame="ecliptic",
                    origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
                ).values
                coords_po = po.translate(
                    sun_wrt_ssb, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name
                )
                coords_ob = obc.translate(
                    sun_wrt_ssb, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name
                )
                propagated_orbits_barycentric = propagated_orbits.set_column(
                    "coordinates", coords_po
                )
                observers_barycentric = observers.set_column("coordinates", coords_ob)
    except Exception:
        propagated_orbits_barycentric = None
        observers_barycentric = None

    if propagated_orbits_barycentric is None or observers_barycentric is None:
        propagated_orbits_barycentric = propagated_orbits.set_column(
            "coordinates",
            transform_coordinates(
                propagated_orbits.coordinates,
                CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            ),
        )
        observers_barycentric = observers.set_column(
            "coordinates",
            transform_coordinates(
                observers.coordinates,
                CartesianCoordinates,
                frame_out="ecliptic",
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            ),
        )

    observer_coordinates = observers_barycentric.coordinates.values
    observer_codes = observers_barycentric.code.to_numpy(zero_copy_only=False)
    mu = observers_barycentric.coordinates.origin.mu()
    times = propagated_orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    orbits_array = propagated_orbits_barycentric.coordinates.values
    need_covariance = not propagated_orbits.coordinates.covariance.is_all_nan()
    if need_covariance:
        cartesian_covariances = propagated_orbits.coordinates.covariance.to_matrix()
    else:
        cartesian_covariances = None

    covariances_spherical_flat: Optional[np.ndarray] = None

    # Rust single-crossing path: one call fuses LT Newton + aberration +
    # ec->eq + cart->sph (+ Dual<6> Jacobian when covariance is needed).
    if need_covariance:
        cov_flat = np.ascontiguousarray(
            np.asarray(cartesian_covariances, dtype=np.float64).reshape(num_entries, 36)
        )
        rust_result = rust_generate_ephemeris_2body_with_covariance_numpy(
            orbits_array,
            cov_flat,
            observer_coordinates,
            mu,
            lt_tol=lt_tol,
            max_iter=max_iter,
            tol=tol,
            stellar_aberration=stellar_aberration,
        )
        assert rust_result is not None
        sph_r, lt_r, aberrated_r, cov_r = rust_result
        ephemeris_spherical = np.ascontiguousarray(sph_r, dtype=np.float64)
        light_time = np.ascontiguousarray(lt_r, dtype=np.float64)
        aberrated_orbits = np.ascontiguousarray(aberrated_r, dtype=np.float64)
        covariances_spherical_flat = np.ascontiguousarray(cov_r, dtype=np.float64)
    else:
        rust_result = rust_generate_ephemeris_2body_numpy(
            orbits_array,
            observer_coordinates,
            mu,
            lt_tol=lt_tol,
            max_iter=max_iter,
            tol=tol,
            stellar_aberration=stellar_aberration,
        )
        assert rust_result is not None
        sph_r, lt_r, aberrated_r = rust_result
        ephemeris_spherical = np.ascontiguousarray(sph_r, dtype=np.float64)
        light_time = np.ascontiguousarray(lt_r, dtype=np.float64)
        aberrated_orbits = np.ascontiguousarray(aberrated_r, dtype=np.float64)

    # Row-level error context for any NaN emitted by either backend. The Rust
    # kernel mirrors the JAX NaN policy (iter >= max_lt_iter → NaN light-time).
    def _error_context_for(row: int, reason: str) -> None:
        _raise_ephemeris_numerical_error(
            reason=reason,
            row_index=row,
            orbit_id=str(
                propagated_orbits_barycentric.orbit_id.to_numpy(
                    zero_copy_only=False
                )[row]
            ),
            object_id=str(
                propagated_orbits_barycentric.object_id.to_numpy(
                    zero_copy_only=False
                )[row]
            ),
            observation_time=float(times[row]),
            light_time=float(light_time[row]),
            max_iter=max_iter,
            tol=tol,
            lt_tol=lt_tol,
        )

    bad_lt = _first_non_finite(light_time)
    if bad_lt is not None:
        _error_context_for(bad_lt, "non_finite_light_time")
    bad_eph = _first_non_finite(ephemeris_spherical)
    if bad_eph is not None:
        _error_context_for(bad_eph, "non_finite_ephemeris_state")
    bad_aberrated = _first_non_finite(aberrated_orbits)
    if bad_aberrated is not None:
        _error_context_for(bad_aberrated, "non_finite_aberrated_state")

    # Compute emission times by subtracting light-time (in days) from the observation times.
    emission_times = propagated_orbits_barycentric.coordinates.time.add_fractional_days(
        pa.array(-light_time)
    )
    aberrated_coordinates = CartesianCoordinates.from_kwargs(
        x=aberrated_orbits[:, 0],
        y=aberrated_orbits[:, 1],
        z=aberrated_orbits[:, 2],
        vx=aberrated_orbits[:, 3],
        vy=aberrated_orbits[:, 4],
        vz=aberrated_orbits[:, 5],
        time=emission_times,
        origin=Origin.from_kwargs(
            code=np.full(num_entries, OriginCodes.SOLAR_SYSTEM_BARYCENTER.name)
        ),
        frame="ecliptic",
    )

    if need_covariance:
        covariances_spherical = CoordinateCovariances.from_matrix(
            covariances_spherical_flat.reshape(num_entries, 6, 6)
        )
    else:
        covariances_spherical = None

    spherical_coordinates = SphericalCoordinates.from_kwargs(
        time=propagated_orbits.coordinates.time,
        rho=ephemeris_spherical[:, 0],
        lon=ephemeris_spherical[:, 1],
        lat=ephemeris_spherical[:, 2],
        vrho=ephemeris_spherical[:, 3],
        vlon=ephemeris_spherical[:, 4],
        vlat=ephemeris_spherical[:, 5],
        covariance=covariances_spherical,
        origin=Origin.from_kwargs(code=observer_codes),
        frame="equatorial",
    )

    ephemeris = Ephemeris.from_kwargs(
        orbit_id=propagated_orbits_barycentric.orbit_id,
        object_id=propagated_orbits_barycentric.object_id,
        coordinates=spherical_coordinates,
        light_time=light_time,
        aberrated_coordinates=aberrated_coordinates,
    )

    want_alpha = bool(predict_phase_angle)
    want_mags = bool(predict_magnitudes)

    if not want_alpha and not want_mags:
        return ephemeris

    # Determine whether we can compute magnitudes (needs H and G).
    has_params = None
    H_v = None
    G = None
    if want_mags:
        H_v = propagated_orbits.physical_parameters.H_v.to_numpy(zero_copy_only=False)
        G = propagated_orbits.physical_parameters.G.to_numpy(zero_copy_only=False)
        has_params = np.isfinite(H_v) & np.isfinite(G)
        if not np.any(has_params):
            want_mags = False

    if not want_alpha and not want_mags:
        return ephemeris

    # Transform object and observer coordinates to heliocentric for photometry.
    if aberrated_coordinates is None:
        raise RuntimeError(
            "Internal error: aberrated coordinates are required for photometry but were not computed."
        )
    aberrated_heliocentric = transform_coordinates(
        aberrated_coordinates,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )
    observers_heliocentric = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        ),
    )

    alpha_deg = None
    mags = None
    if want_mags and want_alpha:
        assert H_v is not None and G is not None and has_params is not None
        mags, alpha_deg = calculate_apparent_magnitude_v_and_phase_angle(
            H_v=H_v,
            object_coords=aberrated_heliocentric,
            observer=observers_heliocentric,
            G=G,
        )
    elif want_alpha:
        alpha_deg = calculate_phase_angle(
            aberrated_heliocentric, observers_heliocentric
        )
    elif want_mags:
        assert H_v is not None and G is not None and has_params is not None
        mags = calculate_apparent_magnitude_v(
            H_v=H_v,
            object_coords=aberrated_heliocentric,
            observer=observers_heliocentric,
            G=G,
        )

    if alpha_deg is not None:
        alpha_deg = np.asarray(alpha_deg, dtype=np.float64)
        ephemeris = ephemeris.set_column(
            "alpha",
            pa.array(alpha_deg, mask=~np.isfinite(alpha_deg), type=pa.float64()),
        )

    if mags is not None:
        assert has_params is not None
        mags = np.asarray(mags, dtype=np.float64)
        valid = has_params & np.isfinite(mags)
        ephemeris = ephemeris.set_column(
            "predicted_magnitude_v", pa.array(mags, mask=~valid, type=pa.float64())
        )
    return ephemeris
