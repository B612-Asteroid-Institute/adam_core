"""
Behavioral parity against the grid-validated Python OD baseline.

The fixture ``migration/artifacts/od_python_baseline_parity_fixture_2026-09-16.json``
records synthetic inputs together with every product the Python
``kk/obs-uncertainty-interface`` branch (``d56114ac``, JAX/numpy) produced for
them: the ``astcat`` / ``from_ades`` mapping, EFCC18 tiles and corrections on
JPL's ``bias.dat``, every sigma-interpreter covariance block (positions
provably unchanged), the whitening factors and analytic Jacobian, the robust
loss helpers, the CMC2003 decision kernels, linear / Huber / ignore fits, a
CMC2003 rejection run and a ``run_od`` provenance run with
``Composite([EFCC18Debias, EmpiricalCovariance])``. This test replays the same
inputs through the Rust-backed implementation and compares.

Tolerances: positions and covariance blocks of the models are exact; fitted
states 1e-9 relative, covariances 1e-6 relative (different SPICE and 2-body
backends behind the two environments). Products that need the published
EFCC18 table skip when ``bias.dat`` is not installed.
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
from ...dynamics.propagation import propagate_2body
from ...observations import efcc18
from ...observations.ades import ADESObservations
from ...observers import Observers
from ...orbits.orbits import Orbits
from ...time import Timestamp
from .. import differential_correction as dc
from .. import rejection
from ..evaluate import OrbitDeterminationObservations, OrbitDeterminationPhotometry
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..native_orbit_fitter import NativeOrbitFitter
from ..observation_uncertainty import (
    BIAS_TABLE_SCHEMA,
    CompositeModel,
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    IdentityModel,
    NightBatchDeweightingModel,
    PerformanceWeightedModel,
    SigmaFloorModel,
)
from ..od_orchestration import run_od
from ..rejection import cmc2003_fit_detailed
from ..veres2017 import VeresFloorModel, VeresReplaceModel
from .test_differential_correction import TwoBodyPropagator, fit_least_squares

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


def _truth_orbit(state: np.ndarray = TRUTH_STATE) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["truth"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinatesFromState(state),
    )


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


def _bias_table() -> pa.Table:
    rows = {
        "obs_code": ["500", "F51", "F51", "W84"],
        "band": [None, None, "g", None],
        "n_obs": [1000, 2000, 300, 800],
        "n_objects": [100, 200, 30, 80],
        "bias_ra_arcsec": [2.0, -1.0, -0.5, 0.05],
        "bias_ra_ci_low": [1.5, -1.2, -0.7, 0.0],
        "bias_ra_ci_high": [2.5, -0.8, -0.3, 0.1],
        "bias_dec_arcsec": [1.5, 0.5, 0.2, -3.0],
        "bias_dec_ci_low": [1.0, 0.3, 0.1, -3.5],
        "bias_dec_ci_high": [2.0, 0.7, 0.3, -2.5],
        "resid_var_ra": [0.36, 0.04, 0.09, 0.25],
        "resid_var_dec": [0.25, 0.04, 0.16, 0.16],
        "resid_cov_ra_dec": [0.09, -0.01, 0.02, 0.01],
        "resid_cov_n": [100, 500, 40, 5],
        "rms_ra_arcsec": [0.6, 0.2, 0.3, 0.5],
        "rms_dec_arcsec": [0.5, 0.2, 0.4, 0.4],
        "chi2_per_obs": [4.0, 2.5, 0.5, 9.0],
        "bias_significant": [True, True, False, False],
        "high_confidence": [True, True, False, False],
        "confidence_score": [1.0, 0.9, 0.3, 0.2],
    }
    return pa.table(rows, schema=BIAS_TABLE_SCHEMA)


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
        1e-6,
        1e-9,
        f"{label} residual chi2",
    )
    _assert_close(actual["weight"], expected["weight"], 1e-6, 1e-9, f"{label} weight")
    assert actual["num_obs"] == expected["num_obs"], label
    assert actual["success"] == expected["success"], label
    assert actual["outlier"] == expected["outlier"], label


def _real_bias_table():
    try:
        efcc18.resolve_bias_dat()
    except FileNotFoundError:
        pytest.skip("EFCC18 bias.dat not installed (pip install jpl-debias-2018)")
    return efcc18.load_efcc18_biases()


class TestFromAdes:
    def test_astcat_round_trip_and_covariance(self, fixture: dict[str, Any]) -> None:
        expected = fixture["products"]["from_ades"]
        n = 6
        ades = ADESObservations.from_kwargs(
            permID=["3000"] * n,
            trkSub=["a1234b"] * n,
            obsSubID=[f"s{i}" for i in range(n)],
            obsTime=Timestamp.from_mjd(60434.0 + 0.1 * np.arange(n), scale="utc"),
            ra=[240.0, 240.05, 15.0, 15.05, 100.0, 359.99],
            dec=[-15.0, -15.05, 10.0, 60.0, 80.0, -70.0],
            rmsRACosDec=[0.9659, 0.9657, None, 0.5, 0.25, 0.3],
            rmsDec=[1.0, 1.0, None, 0.25, 0.3, 0.3],
            rmsCorr=[0.2, None, None, -0.5, 0.1, None],
            mag=[20.0, 20.3, None, 21.4, 19.0, 18.5],
            rmsMag=[0.1, 0.2, None, 0.3, 0.1, 0.1],
            band=["r", "g", None, "r", "w", "o"],
            stn=["W84", "W84", "V00", "695", "F51", "T05"],
            mode=["CCD"] * n,
            astCat=["Gaia2", "Gaia2", "UCAC4", "PPMXL", "Gaia3", "Gaia2"],
        )
        od_ades = OrbitDeterminationObservations.from_ades(ades)
        assert od_ades.id.to_pylist() == expected["id"]
        assert od_ades.astcat.to_pylist() == expected["astcat"]
        assert od_ades.observers.code.to_pylist() == expected["codes"]
        _assert_close(_positions(od_ades), expected["positions"], 0.0, 0.0, "positions")
        _assert_close(
            _cov(od_ades)[:, 1:3, 1:3],
            expected["covariance_lonlat_block"],
            1e-12,
            0.0,
            "covariance",
        )


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
        bias = _real_bias_table()
        expected = fixture["products"]["efcc18_model"]
        observations = _observations(fixture["inputs"])
        debiased = EFCC18DebiasModel(bias_table=bias).apply(observations)
        _assert_close(
            _positions(debiased), expected["positions"], 0.0, 1e-12, "positions"
        )
        assert np.array_equal(_cov(debiased), _cov(observations), equal_nan=True)
        excluded = EFCC18DebiasModel(
            bias_table=bias, exclude_astcats=efcc18.EFCC18_JPL_UNDEBIASED_ASTCATS
        ).apply(observations)
        _assert_close(
            _positions(excluded),
            expected["positions_jpl_excluded"],
            0.0,
            1e-12,
            "excluded",
        )


class TestSigmaInterpreters:
    def test_every_model_matches_and_preserves_positions(
        self, fixture: dict[str, Any]
    ) -> None:
        table = _bias_table()
        models = {
            "identity": IdentityModel(table),
            "empirical_add": EmpiricalCovarianceModel(table),
            "empirical_replace": EmpiricalCovarianceModel(table, mode="replace"),
            "empirical_min5": EmpiricalCovarianceModel(table, min_resid_cov_n=5),
            "performance_weighted": PerformanceWeightedModel(table),
            "sigma_floor": SigmaFloorModel(table),
            "night_batch": NightBatchDeweightingModel(cap=4),
            "composite": CompositeModel(
                PerformanceWeightedModel(table), NightBatchDeweightingModel(cap=4)
            ),
            "veres_floor": VeresFloorModel(),
            "veres_floor_fill": VeresFloorModel(fill_missing=True),
            "veres_replace": VeresReplaceModel(),
            "veres_floor_nofallback": VeresFloorModel(fallback_sigma_arcsec=None),
        }
        observations = _observations(fixture["inputs"])
        obs_cross = _observations(fixture["inputs"], cross_term=1e-9)
        expected = fixture["products"]["models"]
        for name, model in models.items():
            for label, source in (("nan_cross", observations), ("cross", obs_cross)):
                key = f"{name}/{label}"
                result = model.apply(source)
                _assert_close(
                    _cov(result)[:, 1:3, 1:3],
                    expected[key]["covariance_block"],
                    1e-12,
                    0.0,
                    key,
                )
                assert np.array_equal(_positions(result), _positions(source)), key
                assert (result is source) == expected[key]["is_input"], key


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
        # Whitened units (sigma = 0.5"): 1e-5 sigma is 1.4e-9 arcsec between the
        # two environments' SPICE / 2-body backends.
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

    def test_run_od_composite_provenance(
        self, fixture: dict[str, Any], propagator: TwoBodyPropagator
    ) -> None:
        bias = _real_bias_table()
        expected = fixture["products"]["run_od_composite"]
        observations = _observations(fixture["inputs"])
        fitter = NativeOrbitFitter(
            propagator_class=TwoBodyPropagator,
            min_obs=6,
            rchi2_threshold=10.0,
            iod_rchi2_threshold=1e6,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_orbits, members = run_od(
                observations,
                fitter,
                CompositeModel(
                    EFCC18DebiasModel(bias_table=bias),
                    EmpiricalCovarianceModel(_bias_table()),
                ),
                propagator=propagator,
                object_id="synthetic",
            )
        members = members.sort_by("obs_id")
        assert len(fitted_orbits) == expected["n_orbits"]
        assert fitted_orbits.success.to_pylist() == expected["success"]
        assert fitted_orbits.object_id.to_pylist() == expected["object_id"]
        assert members.obs_id.to_pylist() == expected["obs_id"]
        assert members.astcat.to_pylist() == expected["astcat"]
        assert members.outlier.to_pylist() == expected["outlier"]
        for which in ("original", "used"):
            astrometry = getattr(members, f"{which}_astrometry")
            for name in ("lon", "lat", "sigma_lon", "sigma_lat", "cov_lonlat"):
                _assert_close(
                    astrometry.table[name].to_numpy(zero_copy_only=False),
                    expected[which][name],
                    1e-9,
                    1e-12,
                    f"{which} {name}",
                )
        # The IOD epoch may differ between IOD implementations: compare the
        # fitted orbit propagated to the common truth epoch.
        at_truth = propagate_2body(
            fitted_orbits.to_orbits().set_column(
                "coordinates.covariance",
                CoordinateCovariances.nulls(len(fitted_orbits)),
            ),
            Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
        )
        state = np.concatenate([at_truth.coordinates.r[0], at_truth.coordinates.v[0]])
        _assert_close(
            state, expected["state_at_truth_epoch"], 1e-9, 1e-10, "state at truth epoch"
        )
