import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import quivr as qv
import ray
from ray import ObjectRef

from .._rust import (
    propagate_2body_arc_batch_numpy as rust_propagate_2body_arc_batch_numpy,
    propagate_2body_numpy as rust_propagate_2body_numpy,
    propagate_2body_with_covariance_numpy as rust_propagate_2body_with_covariance_numpy,
)
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin
from ..orbits.orbits import Orbits
from ..ray_cluster import initialize_use_ray
from ..time import Timestamp
from ..utils.iter import _iterate_chunks
from .exceptions import DynamicsNumericalError


@dataclass(frozen=True)
class ChiDiagnostics:
    """Lightweight diagnostic snapshot for fail-fast error reporting.

    `chi` is reported as NaN here — the universal-anomaly value is computed
    inside the Rust kernel and is not surfaced through the FFI for diagnostic
    use. The other fields are sufficient to triage stiff / non-physical
    propagation inputs (alpha sign indicates orbit type; finite checks catch
    NaN/Inf upstream).
    """

    dt: float
    mu: float
    r_norm: float
    v_norm: float
    alpha: float
    chi: float
    finite: bool


def _calc_chi_diagnostics(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    mu: float,
) -> ChiDiagnostics:
    r_arr = np.asarray(r, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    r_norm = float(np.linalg.norm(r_arr))
    v_norm = float(np.linalg.norm(v_arr))
    alpha = float(-(v_norm**2) / mu + 2.0 / r_norm) if r_norm > 0 else float("nan")
    finite = bool(
        np.isfinite(r_norm)
        and np.isfinite(v_norm)
        and np.isfinite(alpha)
    )
    return ChiDiagnostics(
        dt=float(dt),
        mu=float(mu),
        r_norm=r_norm,
        v_norm=v_norm,
        alpha=alpha,
        chi=float("nan"),
        finite=finite,
    )


def _first_non_finite_row(values: np.ndarray) -> Optional[int]:
    finite_mask = np.isfinite(values).all(axis=1)
    bad = np.flatnonzero(~finite_mask)
    return int(bad[0]) if bad.size > 0 else None


def _raise_non_finite_propagation_error(
    *,
    stage: str,
    reason: str,
    absolute_idx: int,
    orbit_id: str,
    object_id: str,
    orbit_row: np.ndarray,
    t0: float,
    t1: float,
    mu: float,
    max_iter: int,
    tol: float,
) -> None:
    diag = _calc_chi_diagnostics(
        orbit_row[0:3],
        orbit_row[3:6],
        t1 - t0,
        mu=mu,
    )
    raise DynamicsNumericalError(
        stage=stage,
        reason=reason,
        context={
            "row_index": absolute_idx,
            "orbit_id": orbit_id,
            "object_id": object_id,
            "t0": float(t0),
            "t1": float(t1),
            "dt": float(t1 - t0),
            "mu": float(mu),
            "r_norm": diag.r_norm,
            "v_norm": diag.v_norm,
            "alpha": diag.alpha,
            "chi": diag.chi,
            "chi_finite": diag.finite,
            "max_iter": int(max_iter),
            "tol": float(tol),
        },
    )


# Rough crossover from microbenchmark (Apple M1, 2026-04-25): below this
# n_times the warm-started serial arc per orbit beats the rayon-parallel
# batched cold-start path. Above it, parallel batch wins.
_ARC_PATH_DTS_THRESHOLD = 500


def _can_use_arc_path(num_entries: int, n_times: int) -> bool:
    """Decide whether the warm-started arc API is preferable to the
    rayon-parallel batched path for this propagation shape.

    Conditions:
      * we have ≥ 2 dts to amortize chi warm-starting,
      * n_times stays under the rayon crossover, AND
      * the total entry count divides evenly into per-orbit chunks
        (it always does in our flow, but assert it as a guard).
    """
    if n_times < 2 or n_times >= _ARC_PATH_DTS_THRESHOLD:
        return False
    if num_entries % n_times != 0:
        return False
    return True


def _run_2body_propagate(
    *,
    orbits_array_: np.ndarray,
    t0_: np.ndarray,
    t1_: np.ndarray,
    mu_: np.ndarray,
    cov_in_flat: Optional[np.ndarray],
    n_times: int,
    orbit_ids_: np.ndarray,
    object_ids_: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Propagate a flattened (orbit x time) batch using Rust when available,
    falling back to the legacy JAX chunked path. Raises
    `DynamicsNumericalError` with the same context/diagnostics as the legacy
    path on non-finite input or output.
    """
    num_entries = orbits_array_.shape[0]
    dt_ = t1_ - t0_

    bad_input = _first_non_finite_row(orbits_array_)
    if bad_input is not None:
        _raise_non_finite_propagation_error(
            stage="propagation",
            reason="non_finite_input_state",
            absolute_idx=bad_input,
            orbit_id=str(orbit_ids_[bad_input]),
            object_id=str(object_ids_[bad_input]),
            orbit_row=np.asarray(orbits_array_[bad_input], dtype=np.float64),
            t0=float(t0_[bad_input]),
            t1=float(t1_[bad_input]),
            mu=float(mu_[bad_input]),
            max_iter=max_iter,
            tol=tol,
        )

    cov_out_flat: Optional[np.ndarray] = None
    if cov_in_flat is not None:
        # cov_in_flat is per-orbit (n_orbits, 36); broadcast to per-entry.
        cov_per_entry = np.repeat(cov_in_flat, n_times, axis=0)
        rust_result = rust_propagate_2body_with_covariance_numpy(
            orbits_array_, cov_per_entry, dt_, mu_, max_iter, tol
        )
        assert rust_result is not None
        orbits_propagated, cov_out_flat = rust_result
        orbits_propagated = np.ascontiguousarray(orbits_propagated, dtype=np.float64)
        cov_out_flat = np.ascontiguousarray(cov_out_flat, dtype=np.float64)
    elif _can_use_arc_path(num_entries, n_times):
        # Single-orbit / many-dts pattern: the Rust arc API computes
        # orbit-only constants once per orbit and warm-starts the chi
        # solver from the previous dt's converged value. Rayon parallel
        # ACROSS orbits + serial warm-start WITHIN orbit. Wins 3-10× over
        # the rayon-parallel cold-start batched path when the per-orbit
        # n_times is small (~< 500).
        n_orbits = num_entries // n_times
        # Per-orbit base orbit and mu — the ((n_orbits, n_times, 6)
        # ordered) input has all n_times rows of orbit-i identical.
        base_orbits = orbits_array_[::n_times]
        base_mus = mu_[::n_times]
        dts_per_orbit = dt_.reshape(n_orbits, n_times)
        arc_out = rust_propagate_2body_arc_batch_numpy(
            base_orbits, dts_per_orbit, base_mus, max_iter, tol
        )
        assert arc_out is not None
        orbits_propagated = np.ascontiguousarray(arc_out, dtype=np.float64)
    else:
        rust_result = rust_propagate_2body_numpy(
            orbits_array_, dt_, mu_, max_iter, tol
        )
        assert rust_result is not None
        orbits_propagated = np.ascontiguousarray(rust_result, dtype=np.float64)

    bad_output = _first_non_finite_row(orbits_propagated)
    if bad_output is not None:
        _raise_non_finite_propagation_error(
            stage="propagation",
            reason="non_finite_output_state",
            absolute_idx=bad_output,
            orbit_id=str(orbit_ids_[bad_output]),
            object_id=str(object_ids_[bad_output]),
            orbit_row=np.asarray(orbits_array_[bad_output], dtype=np.float64),
            t0=float(t0_[bad_output]),
            t1=float(t1_[bad_output]),
            mu=float(mu_[bad_output]),
            max_iter=max_iter,
            tol=tol,
        )

    if orbits_propagated.shape[0] != num_entries:
        raise RuntimeError(
            f"Internal error: expected {num_entries} propagated rows, got {orbits_propagated.shape[0]}"
        )

    return orbits_propagated, cov_out_flat


def _propagate_2body_serial(
    orbits: Orbits,
    times: Timestamp,
    *,
    max_iter: int,
    tol: float,
) -> Orbits:
    """
    Serial (single-process) implementation of 2-body propagation.

    The Ray backend uses this function inside each worker.
    """
    # Extract and prepare data
    cartesian_orbits = orbits.coordinates.values
    t0 = orbits.coordinates.time.rescale("tdb").mjd()
    t1 = times.rescale("tdb").mjd()
    mu = orbits.coordinates.origin.mu()
    orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)
    object_ids = orbits.object_id.to_numpy(zero_copy_only=False)

    n_orbits = cartesian_orbits.shape[0]
    n_times = len(times)
    orbit_ids_ = np.repeat(orbit_ids, n_times)
    object_ids_ = np.repeat(object_ids, n_times)
    orbits_array_ = np.repeat(cartesian_orbits, n_times, axis=0)
    mu_ = np.repeat(mu, n_times)
    t0_ = np.repeat(t0, n_times)
    t1_ = np.tile(t1, n_orbits)

    # Preserve physical parameters by repeating per-orbit rows across times.
    pp_idx = np.repeat(np.arange(n_orbits), n_times).tolist()
    physical_parameters_ = orbits.physical_parameters.take(pp_idx)

    has_cov = not orbits.coordinates.covariance.is_all_nan()
    cov_in_flat = (
        orbits.coordinates.covariance.to_matrix().reshape(n_orbits, 36)
        if has_cov
        else None
    )

    orbits_propagated, cov_out_flat = _run_2body_propagate(
        orbits_array_=orbits_array_,
        t0_=t0_,
        t1_=t1_,
        mu_=mu_,
        cov_in_flat=cov_in_flat,
        n_times=n_times,
        orbit_ids_=orbit_ids_,
        object_ids_=object_ids_,
        max_iter=max_iter,
        tol=tol,
    )

    if has_cov:
        cartesian_covariances = CoordinateCovariances.from_matrix(
            cov_out_flat.reshape(-1, 6, 6)
        )
    else:
        cartesian_covariances = None

    origin_code = np.repeat(
        orbits.coordinates.origin.code.to_numpy(zero_copy_only=False), n_times
    )

    return Orbits.from_kwargs(
        orbit_id=orbit_ids_,
        object_id=object_ids_,
        physical_parameters=physical_parameters_,
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbits_propagated[:, 0],
            y=orbits_propagated[:, 1],
            z=orbits_propagated[:, 2],
            vx=orbits_propagated[:, 3],
            vy=orbits_propagated[:, 4],
            vz=orbits_propagated[:, 5],
            covariance=cartesian_covariances,
            time=Timestamp.from_mjd(t1_, scale="tdb"),
            origin=Origin.from_kwargs(code=origin_code),
            frame="ecliptic",
        ),
    )


@ray.remote
def propagate_2body_worker_ray(
    start: int,
    idx_chunk: np.ndarray,
    orbits: Orbits,
    times: Timestamp,
    max_iter: int,
    tol: float,
) -> Tuple[int, Orbits]:
    orbits_chunk = orbits.take(idx_chunk)
    propagated = _propagate_2body_serial(
        orbits_chunk,
        times,
        max_iter=max_iter,
        tol=tol,
    )
    return start, propagated


def propagate_2body(
    orbits: Orbits,
    times: Timestamp,
    max_iter: int = 1000,
    tol: float = 1e-14,
    *,
    max_processes: Optional[int] = 1,
    chunk_size: int = 100,
) -> Orbits:
    """
    Propagate orbits using the 2-body universal anomaly formalism.

    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits` (N)
        Cartesian orbits with position in units of au and velocity in units of au per day.
    times : Timestamp (M)
        Epochs to which to propagate each orbit. If a single epoch is given, all orbits are propagated to this
        epoch. If multiple epochs are given, then each orbit to will be propagated to each epoch.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits` (N*M)
        Orbits propagated to each MJD.
    """
    if max_processes is None:
        max_processes = mp.cpu_count()

    if max_processes <= 1:
        return _propagate_2body_serial(
            orbits,
            times,
            max_iter=max_iter,
            tol=tol,
        )

    initialize_use_ray(num_cpus=max_processes)

    # Put large inputs in object store once.
    orbits_ref = ray.put(orbits)  # type: ignore[name-defined]
    times_ref = ray.put(times)  # type: ignore[name-defined]

    idx = np.arange(0, len(orbits), dtype=np.int64)
    pending: List["ObjectRef"] = []  # type: ignore[name-defined]
    results: Dict[int, Orbits] = {}

    for idx_chunk in _iterate_chunks(idx, chunk_size):
        start = int(idx_chunk[0]) if len(idx_chunk) else 0
        pending.append(
            propagate_2body_worker_ray.remote(  # type: ignore[name-defined]
                start, idx_chunk, orbits_ref, times_ref, max_iter, tol
            )
        )

        if len(pending) >= max_processes * 1.5:
            finished, pending = ray.wait(pending, num_returns=1)  # type: ignore[name-defined]
            start_i, propagated_i = ray.get(finished[0])  # type: ignore[name-defined]
            results[int(start_i)] = propagated_i

    while pending:
        finished, pending = ray.wait(pending, num_returns=1)  # type: ignore[name-defined]
        start_i, propagated_i = ray.get(finished[0])  # type: ignore[name-defined]
        results[int(start_i)] = propagated_i

    chunks = [results[k] for k in sorted(results.keys())]
    return qv.concatenate(chunks) if chunks else Orbits.empty()
