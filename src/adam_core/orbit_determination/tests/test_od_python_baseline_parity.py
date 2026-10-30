"""Replay applicable d21 OD numerical products without legacy workflow APIs.

The retained fixture records the accepted 2026-09-16 Python-baseline products.
This module replays only surviving numerical kernels and the declarative
Rust-owned EFCC18 pipeline. Removed executable model/fitter/provenance APIs are
not reconstructed here.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observations import efcc18
from ...observers import Observers
from ...orbits.orbits import Orbits
from ...time import Timestamp
from .._pipeline import apply_pipeline_once
from ..algorithms import fitting as dc
from ..algorithms import rejection
from ..algorithms.fitting import fit_least_squares
from ..algorithms.rejection import cmc2003_fit_detailed
from ..config import OdConfig, ResourceBindings
from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..observation_pipeline import Efcc18CatalogBiasCorrection, ObservationPipeline
from ._focused import explicit_config
from .test_differential_correction import TwoBodyPropagator

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "od_python_baseline_parity_fixture_2026-09-16.json"
)
TRUTH_STATE = np.array([0.9, 0.5, 0.05, -0.008, 0.012, 0.0005])
EPOCH_MJD_TDB = 61000.0


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    with FIXTURE.open() as f:
        return json.load(f)


def CartesianCoordinatesFromState(state: np.ndarray):  # noqa: N802
    from ...coordinates.cartesian import CartesianCoordinates

    return CartesianCoordinates.from_kwargs(
        x=state[0:1],
        y=state[1:2],
        z=state[2:3],
        vx=state[3:4],
        vy=state[4:5],
        vz=state[5:6],
        time=Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )


def _truth_orbit(state: np.ndarray = TRUTH_STATE) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["truth"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinatesFromState(state),
    )


def _guess_orbit() -> Orbits:
    x0 = TRUTH_STATE + np.array([1e-5, -8e-6, 5e-6, 1e-8, -1e-8, 5e-9])
    return _truth_orbit(x0).set_column(
        "orbit_id", pa.array(["guess"], pa.large_string())
    )


def _observations(
    inputs: dict[str, Any], with_outliers: bool = False, cross_term: float | None = None
) -> OrbitDeterminationObservations:
    n = len(inputs["lon"])
    times = Timestamp.from_mjd(np.array(inputs["mjd_tdb"]), scale="tdb")
    lat = np.array(inputs["lat"])
    if with_outliers:
        lat = lat.copy()
        for index, k in zip(inputs["outlier_index"], inputs["outlier_sigmas"]):
            lat[index] += k * np.sqrt(inputs["var_lat"][index])
    covariances = np.full((n, 6, 6), np.nan)
    covariances[:, 1, 1] = inputs["var_lon"]
    covariances[:, 2, 2] = inputs["var_lat"]
    if cross_term is not None:
        covariances[:, 1, 2] = cross_term
        covariances[:, 2, 1] = cross_term
    coordinates = SphericalCoordinates.from_kwargs(
        lon=np.array(inputs["lon"]),
        lat=lat,
        time=times,
        covariance=CoordinateCovariances.from_matrix(covariances),
        origin=Origin.from_kwargs(code=inputs["codes"]),
        frame="equatorial",
    )
    return OrbitDeterminationObservations.from_kwargs(
        id=[f"obs-{i:03d}" for i in range(n)],
        coordinates=coordinates,
        observers=Observers.from_codes(inputs["codes"], times),
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * n, rmsmag=[None] * n, band=inputs["band"]
        ),
        astcat=inputs["astcat"],
    )


def _positions(observations: OrbitDeterminationObservations) -> np.ndarray:
    return np.stack(
        [
            observations.coordinates.lon.to_numpy(zero_copy_only=False),
            observations.coordinates.lat.to_numpy(zero_copy_only=False),
        ],
        axis=1,
    )


def _cov(observations: OrbitDeterminationObservations) -> np.ndarray:
    return observations.coordinates.covariance.to_matrix()


def _assert_close(
    actual: Any, expected: Any, rtol: float, atol: float, label: str
) -> None:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, label
    np.testing.assert_array_equal(np.isnan(a), np.isnan(e), err_msg=label)
    mask = ~np.isnan(e)
    np.testing.assert_allclose(a[mask], e[mask], rtol=rtol, atol=atol, err_msg=label)


def _fit_summary(fitted: FittedOrbits, members: FittedOrbitMembers) -> dict[str, Any]:
    return {
        "state": np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]]),
        "covariance": fitted.coordinates.covariance.to_matrix()[0],
        "chi2": fitted.chi2[0].as_py(),
        "reduced_chi2": fitted.reduced_chi2[0].as_py(),
        "num_obs": fitted.num_obs[0].as_py(),
        "success": fitted.success[0].as_py(),
        "outlier": members.outlier.to_pylist(),
        "weight": members.weight.to_pylist(),
        "residual_chi2": members.residuals.chi2.to_pylist(),
    }


def _assert_fit(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    _assert_close(actual["state"], expected["state"], 1e-9, 1e-10, f"{label} state")
    _assert_close(
        actual["covariance"], expected["covariance"], 1e-6, 1e-22, f"{label} covariance"
    )
    _assert_close(actual["chi2"], expected["chi2"], 1e-6, 1e-9, f"{label} chi2")
    _assert_close(
        actual["reduced_chi2"], expected["reduced_chi2"], 1e-6, 1e-9, f"{label} rchi2"
    )
    _assert_close(
        actual["residual_chi2"],
        expected["residual_chi2"],
        1e-4,
        1e-6,
        f"{label} residual chi2",
    )
    _assert_close(actual["weight"], expected["weight"], 1e-5, 1e-9, f"{label} weight")
    assert actual["num_obs"] == expected["num_obs"], label
    assert actual["success"] == expected["success"], label
    assert actual["outlier"] == expected["outlier"], label


def _real_bias_path() -> Path:
    try:
        return efcc18.resolve_bias_dat()
    except FileNotFoundError:
        pytest.skip("EFCC18 bias.dat is not installed")


def _real_bias_table():
    return efcc18.load_efcc18_biases(_real_bias_path())


def _efcc_config(path: Path, exclude_astcats: tuple[str, ...]) -> OdConfig:
    base = explicit_config()
    return OdConfig.explicit(
        observation_pipeline=ObservationPipeline.ordered(
            [Efcc18CatalogBiasCorrection(exclude_astcats=exclude_astcats)]
        ),
        iod=base.for_iod(),
        refinement=base.for_refinement(),
        evaluation=base.for_evaluation(),
        resources=ResourceBindings({"efcc18_catalog_bias_table": path}),
    )


def test_core_config_pins_accepted_d21_two_body_constants() -> None:
    two_body = explicit_config().for_iod().candidate_correction_fit.two_body
    assert two_body.lt_tol == 1.0e-12
    assert two_body.ephemeris_max_iter == 1000
    assert two_body.max_lt_iter == 10


class TestEfcc18:
    def test_tiles_and_corrections_on_published_table(
        self, fixture: dict[str, Any]
    ) -> None:
        bias = _real_bias_table()
        expected = fixture["products"]["efcc18"]
        rng = np.random.default_rng(3)
        ra = np.concatenate([rng.uniform(0, 360, 60), [0.0, 359.9999, 45.0, 180.0]])
        dec = np.concatenate(
            [np.degrees(np.arcsin(rng.uniform(-1, 1, 60))), [89.99, -89.99, 0.0, 66.0]]
        )
        cats = [
            "UCAC4",
            "PPMXL",
            "Gaia1",
            "Tycho",
            "USNOB1",
            "Gaia2",
            None,
            "2MASS",
        ] * 8
        jd = 2451545.0 + rng.uniform(-3000, 9000, 64)
        assert efcc18.ra_dec_to_healpix(ra, dec).tolist() == expected["tiles"]
        assert efcc18.is_efcc18_covered(cats).tolist() == expected["covered"]
        _assert_close(
            efcc18.compute_efcc18_corrections(ra, dec, cats, jd, bias_table=bias),
            expected["corrections"],
            0.0,
            0.0,
            "corrections",
        )
        _assert_close(
            efcc18.compute_efcc18_corrections(
                ra,
                dec,
                cats,
                jd,
                bias_table=bias,
                exclude_astcats=efcc18.EFCC18_JPL_UNDEBIASED_ASTCATS,
            ),
            expected["corrections_jpl_excluded"],
            0.0,
            0.0,
            "corrections excluded",
        )
        checksum = expected["bias_table_checksum"]
        assert float(bias.astype(np.float64).sum()) == pytest.approx(
            checksum[0], rel=1e-12
        )
        np.testing.assert_array_equal(
            bias[12345].astype(np.float64).ravel(), checksum[2]
        )

    def test_debias_model_positions(self, fixture: dict[str, Any]) -> None:
        path = _real_bias_path()
        expected = fixture["products"]["efcc18_model"]
        observations = _observations(fixture["inputs"])

        debiased, output = apply_pipeline_once(
            observations, _efcc_config(path, exclude_astcats=())
        )
        assert output["pipeline_changed"]
        _assert_close(
            _positions(debiased), expected["positions"], 0.0, 1e-12, "positions"
        )
        assert np.array_equal(_cov(debiased), _cov(observations), equal_nan=True)

        excluded, excluded_output = apply_pipeline_once(
            observations,
            _efcc_config(path, tuple(efcc18.EFCC18_JPL_UNDEBIASED_ASTCATS)),
        )
        assert excluded_output["pipeline_changed"]
        _assert_close(
            _positions(excluded),
            expected["positions_jpl_excluded"],
            0.0,
            1e-12,
            "excluded",
        )
        assert np.array_equal(_cov(excluded), _cov(observations), equal_nan=True)


class TestDifferentialCorrectionKernels:
    def test_whitening_jacobian_and_robust_helpers(
        self, fixture: dict[str, Any]
    ) -> None:
        expected = fixture["products"]
        obs_cross = _observations(fixture["inputs"], cross_term=1e-9)
        whiteners = dc._observation_whitening_matrices(obs_cross)
        _assert_close(
            whiteners, expected["whitening"]["whiteners"], 1e-9, 1e-12, "whiteners"
        )
        terms = dc._analytic_jacobian_terms(obs_cross)
        jacobian = dc._analytic_jacobian(TRUTH_STATE, EPOCH_MJD_TDB, terms)
        _assert_close(
            jacobian, expected["whitening"]["jacobian"], 1e-8, 1e-9, "jacobian"
        )
        residuals = dc.residual_function(
            TRUTH_STATE, EPOCH_MJD_TDB, obs_cross, TwoBodyPropagator()
        )
        _assert_close(
            residuals,
            expected["whitening"]["residual_function"],
            0.0,
            1e-5,
            "residuals",
        )
        r = np.array([0.2, -1.0, 2.5, -4.0, 0.0, 1.345])
        assert dc._robust_cost(r, "huber", 1.345) == pytest.approx(
            expected["robust"]["cost"], rel=1e-12
        )
        assert dc._robust_cost(r, "linear", 1.0) == pytest.approx(
            expected["robust"]["linear_cost"], rel=1e-12
        )
        _assert_close(
            dc._robust_weights(r, "huber", 1.345),
            expected["robust"]["weights"],
            1e-12,
            0.0,
            "weights",
        )
        _assert_close(
            dc._robust_jacobian_scale(r, "huber", 1.345),
            expected["robust"]["scale"],
            1e-9,
            0.0,
            "scale",
        )

    def test_cmc2003_decision_kernels(self, fixture: dict[str, Any]) -> None:
        expected = fixture["products"]["cmc2003_kernels"]
        chi2 = np.ones(30)
        chi2[5] = 100.0
        chi2[6] = 30.0
        chi2[7] = 20.0
        selected = np.ones(30, dtype=bool)
        selected[3] = False
        chi2[3] = 6.9
        apparitions = np.array([0] * 25 + [1] * 5)
        new_selected, n_rejected, n_recovered, flags = rejection._cmc2003_select(
            chi2,
            selected,
            apparitions,
            chi2_reject=8.0,
            chi2_recover=7.0,
            chi2_frac=0.25,
            max_rejected_fraction=0.5,
            one_at_a_time=False,
        )
        assert new_selected.tolist() == expected["selected"]
        assert (n_rejected, n_recovered, sorted(flags)) == (
            expected["n_rejected"],
            expected["n_recovered"],
            expected["flags"],
        )
        residuals = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        jacobian = np.zeros((6, 6))
        jacobian[0, 0] = jacobian[1, 1] = jacobian[2, 0] = jacobian[3, 1] = 1.0
        jacobian[4, 2] = 0.7
        jacobian[5, 3] = 0.6
        jacobian[4, 3] = 0.1
        covariance = np.diag([0.5, 0.5, 0.99, 0.99, 0, 0]).astype(float)
        exp_chi2, exp_flags = rejection._expected_residual_chi2(
            residuals, jacobian, covariance, np.array([True, False, True])
        )
        _assert_close(exp_chi2, expected["expected_chi2"], 1e-9, 1e-12, "expected chi2")
        assert sorted(exp_flags) == expected["expected_flags"]
        assert (
            rejection._apparitions(
                np.array([10.0, 12.0, 400.0, 11.0, 401.0]), 180.0
            ).tolist()
            == expected["apparitions"]
        )


class TestFits:
    @pytest.fixture(scope="class")
    def propagator(self) -> TwoBodyPropagator:
        return TwoBodyPropagator()

    def test_linear_huber_ignore_and_cmc2003(
        self, fixture: dict[str, Any], propagator: TwoBodyPropagator
    ) -> None:
        inputs = fixture["inputs"]
        expected = fixture["products"]
        observations = _observations(inputs)
        contaminated = _observations(inputs, with_outliers=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_fit(
                _fit_summary(
                    *fit_least_squares(_guess_orbit(), observations, propagator)
                ),
                expected["fit_linear_clean"],
                "linear clean",
            )
            _assert_fit(
                _fit_summary(
                    *fit_least_squares(_guess_orbit(), contaminated, propagator)
                ),
                expected["fit_linear_contaminated"],
                "linear contaminated",
            )
            _assert_fit(
                _fit_summary(
                    *fit_least_squares(
                        _guess_orbit(), contaminated, propagator, loss="huber"
                    )
                ),
                expected["fit_huber_contaminated"],
                "huber contaminated",
            )
            _assert_fit(
                _fit_summary(
                    *fit_least_squares(
                        _guess_orbit(),
                        observations,
                        propagator,
                        ignore=["obs-002", "obs-020"],
                    )
                ),
                expected["fit_linear_ignore"],
                "linear ignore",
            )
            result = cmc2003_fit_detailed(_guess_orbit(), contaminated, propagator)
        _assert_fit(
            _fit_summary(result.fitted_orbit, result.fitted_orbit_members),
            expected["cmc2003_fit"],
            "cmc2003",
        )
        assert (result.n_iterations, result.n_rejected, result.n_recovered) == (
            expected["cmc2003_fit"]["n_iterations"],
            expected["cmc2003_fit"]["n_rejected"],
            expected["cmc2003_fit"]["n_recovered"],
        )
        assert sorted(result.flags) == sorted(expected["cmc2003_fit"]["flags"])
