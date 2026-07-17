"""Abstract propagator framework.

adam_core ships no concrete propagator and no Python composition: only
Rust-backed propagators (e.g. ``adam_assist.ASSISTPropagator``) are
supported. Each public surface -- :meth:`Propagator.propagate_orbits`,
:meth:`EphemerisMixin.generate_ephemeris`, and
:meth:`~adam_core.dynamics.impacts.ImpactMixin.detect_collisions` -- is the
abstract single-crossing contract that a backend implements as one
Python->Rust crossing. Covariance sampling/collapse, light-time/aberration,
photometry, and local parallelism (rayon) all live inside the backend; Ray, if
used at all, is only an optional outer distribution wrapper around a whole
backend call.

The former Python composition (per-chunk Ray fan-out, covariance
sample/propagate/collapse, the generic light-time ephemeris pipeline, and the
``_propagate_orbits``/``_generate_ephemeris`` hooks) has been deleted; its
behavior is replicated inside the Rust backends and validated by the
the downstream ``adam-assist`` Rust/parity test suites.
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..orbits.variants import VariantOrbits
from .types import EphemerisType, ObserverType, OrbitType, TimestampType

logger = logging.getLogger(__name__)

__all__ = [
    "EphemerisMixin",
    "Propagator",
    "EphemerisType",
    "ObserverType",
    "OrbitType",
    "TimestampType",
]

_CovarianceMethod = Literal["auto", "sigma-point", "monte-carlo"]


class EphemerisMixin(ABC):
    """Ephemeris-generation contract for propagators.

    Concrete (Rust-backed) propagators implement :meth:`generate_ephemeris` as a
    single Python->Rust crossing: covariance sampling, per-variant ephemeris,
    collapse to per-row covariance, light-time/aberration, and photometry all
    run inside the backend. adam_core provides no Python composition -- only the
    abstract contract below.
    """

    @abstractmethod
    def generate_ephemeris(
        self,
        orbits: OrbitType,
        observers: ObserverType,
        covariance: bool = False,
        covariance_method: _CovarianceMethod = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
        seed: Optional[int] = None,
        predict_magnitudes: bool = True,
        predict_phase_angle: bool = False,
    ) -> Ephemeris:
        """Generate ephemerides for each orbit as observed by each observer.

        Backends must return predicted ephemerides in a stable row order
        (orbit_id, time) -- or (orbit_id, variant_id, time) for variant
        ephemeris -- and, when ``covariance`` is set, propagate the orbit
        covariance by sampling/collapse inside the single Rust crossing.

        Parameters
        ----------
        orbits : Orbits (N)
            Orbits for which to generate ephemerides.
        observers : Observers (M)
            Observers for which to generate each orbit's ephemeris.
        covariance : bool, optional
            Propagate the orbit covariance matrices to the ephemeris.
        covariance_method : {'auto', 'sigma-point', 'monte-carlo'}, optional
            Sampling method used when ``covariance`` is set.
        num_samples : int, optional
            Number of monte-carlo samples to draw.
        chunk_size, max_processes, seed : optional
            Backend parallelism / determinism controls. Local parallelism is a
            backend (rayon) concern; ``max_processes`` is retained only for an
            optional outer distribution wrapper.
        predict_magnitudes, predict_phase_angle : bool, optional
            Whether to attach predicted V magnitude / phase angle columns.

        Returns
        -------
        ephemeris : Ephemeris or VariantEphemeris
        """
        ...


class Propagator(EphemerisMixin):
    """Abstract base for orbit propagators.

    Concrete (Rust-backed) propagators implement :meth:`propagate_orbits` as a
    single Python->Rust crossing. When ``covariance`` is set, the backend
    samples orbit variants, propagates them, and collapses the samples back to
    a covariance -- all inside Rust. adam_core ships no concrete propagator and
    no Python composition; only Rust backends are supported.
    """

    @abstractmethod
    def propagate_orbits(
        self,
        orbits: OrbitType,
        times: TimestampType,
        covariance: bool = False,
        covariance_method: _CovarianceMethod = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int = 100,
        max_processes: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> Union[Orbits, VariantOrbits]:
        """Propagate each orbit in ``orbits`` to each time in ``times``.

        Backends must return propagated orbits in a stable row order
        (orbit_id, time) -- or (orbit_id, variant_id, time) for VariantOrbits --
        and, when ``covariance`` is set, propagate the orbit covariance by
        sampling/collapse inside the single Rust crossing.

        Parameters
        ----------
        orbits : Orbits (N)
            Orbits to propagate.
        times : Timestamp (M)
            Times to which to propagate.
        covariance : bool, optional
            Propagate the orbit covariance matrices.
        covariance_method : {'auto', 'sigma-point', 'monte-carlo'}, optional
            Sampling method used when ``covariance`` is set.
        num_samples : int, optional
            Number of monte-carlo samples to draw.
        chunk_size, max_processes, seed : optional
            Backend parallelism / determinism controls. Local parallelism is a
            backend (rayon) concern; ``max_processes`` is retained only for an
            optional outer distribution wrapper.

        Returns
        -------
        propagated : Orbits or VariantOrbits
        """
        ...
