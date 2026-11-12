from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv

from adam_core import _rust_native
from adam_core.coordinates import CartesianCoordinates, CometaryCoordinates, Origin
from adam_core.observations import SourceCatalog
from adam_core.orbits import NonGravitationalParameters, Orbits
from adam_core.time import Timestamp
from adam_core.utils.spice import OBSERVER_GEOMETRY_KERNELS

from ..bandpasses import BandpassComposition
from ..color_determination import (
    ColorFit,
    _color_fit_from_native_batch,
    _native_record_batch,
    estimate_colors,
)

DATA_DIR = Path(__file__).parent / "data"
COLOR_FIXTURES = sorted(DATA_DIR.glob("color_fixture_*.npz"))
with (DATA_DIR / "color_fit_goldens.json").open() as golden_file:
    COLOR_FIT_GOLDENS = json.load(golden_file)
LIST_FIELDS = (
    "filter_ids",
    "absolute_magnitudes",
    "absolute_magnitude_sigmas",
    "num_used_by_filter",
    "magnitude_covariance",
    "color_lhs_filter_ids",
    "color_rhs_filter_ids",
    "colors",
    "color_sigmas",
)
SCIENCE_LIST_FIELDS = (
    "absolute_magnitudes",
    "absolute_magnitude_sigmas",
    "magnitude_covariance",
    "color_lhs_filter_ids",
    "color_rhs_filter_ids",
    "colors",
    "color_sigmas",
)


def _load_fixture_observations(fx: np.lib.npyio.NpzFile) -> SourceCatalog:
    n = int(fx["mag_obs"].shape[0])
    return SourceCatalog.from_kwargs(
        id=np.arange(n).astype(str).tolist(),
        object_id=[str(fx["object_id"][0])] * n,
        exposure_id=fx["obsid"].astype(str).tolist(),
        time=Timestamp.from_kwargs(
            days=pa.array(fx["obstime_days"].tolist(), type=pa.int64()),
            nanos=pa.array(fx["obstime_nanos"].tolist(), type=pa.int64()),
            scale="utc",
        ),
        ra=fx["ra"].tolist(),
        dec=fx["dec"].tolist(),
        ra_sigma=[None] * n,
        dec_sigma=[None] * n,
        mag=fx["mag_obs"].tolist(),
        mag_sigma=fx["rmsmag"].tolist(),
        filter=fx["band"].astype(str).tolist(),
        observatory_code=fx["station"].astype(str).tolist(),
        catalog_id=fx["station"].astype(str).tolist(),
    )


def _load_fixture_orbits(fx: np.lib.npyio.NpzFile) -> Orbits:
    epoch = Timestamp.from_kwargs(
        days=pa.array([int(fx["epoch_days"][0])], type=pa.int64()),
        nanos=pa.array([int(fx["epoch_nanos"][0])], type=pa.int64()),
        scale="tdb",
    )
    return Orbits.from_kwargs(
        orbit_id=[None],
        object_id=[str(fx["object_id"][0])],
        coordinates=CometaryCoordinates.from_kwargs(
            q=fx["q"].tolist(),
            e=fx["e"].tolist(),
            i=fx["inc"].tolist(),
            raan=fx["node"].tolist(),
            ap=fx["argperi"].tolist(),
            tp=fx["peri_time"].tolist(),
            time=epoch,
            origin=Origin.from_kwargs(code=pa.repeat("SUN", len(epoch))),
            frame="ecliptic",
        ).to_cartesian(),
    )


def _fixture(name: str = "color_fixture_2025_MN25.npz") -> tuple[SourceCatalog, Orbits]:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"Missing fixture {name}")
    with np.load(path, allow_pickle=True) as fx:
        return _load_fixture_observations(fx), _load_fixture_orbits(fx)


def _row(result: ColorFit) -> dict[str, object]:
    assert len(result) == 1
    return result.table.to_pylist()[0]


def _assert_partition(row: dict[str, object]) -> None:
    assert row["num_obs"] == (
        row["num_invalid"]
        + row["num_unsupported_filter"]
        + row["num_clipped"]
        + row["num_used"]
    )


def test_facade_is_schema_marshalling_and_one_native_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, orbits = _fixture()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    target = "estimate_colors_ipc"

    for name in dir(_rust_native):
        native = getattr(_rust_native, name)
        if not callable(native):
            continue

        def tracked(
            *args: object,
            _name: str = name,
            _native: object = native,
            **kwargs: object,
        ) -> object:
            calls.append((_name, args, kwargs))
            if _name != target:
                raise AssertionError(f"unexpected chained native call: {_name}")
            assert callable(_native)
            return _native(*args, **kwargs)

        monkeypatch.setattr(_rust_native, name, tracked)

    result = estimate_colors(observations, orbits, "c1c2", force_g_bounds=False)

    assert _row(result)["status"] == "ok"
    assert [name for name, _, _ in calls] == [target]
    _, args, kwargs = calls[0]
    assert len(args) == 5
    assert isinstance(args[0], bytes) and isinstance(args[1], bytes)
    assert args[2] == list(map(str, OBSERVER_GEOMETRY_KERNELS))
    assert kwargs["projection"] is None
    assert kwargs["template_id"] is None
    assert kwargs["mix"] is None


def test_python_and_published_rust_bandpass_bundles_are_identical() -> None:
    root = Path(__file__).resolve().parents[4]
    python_data = root / "src/adam_core/photometry/bandpasses/data"
    rust_data = root / "rust/adam_core_rs_coords/data/bandpasses"
    filenames = sorted(path.name for path in python_data.glob("*.parquet"))
    assert filenames == [
        "asteroid_templates.parquet",
        "bandpass_curves.parquet",
        "observatory_band_map.parquet",
        "solar_spectrum.parquet",
        "template_bandpass_integrals.parquet",
    ]
    for filename in filenames:
        assert (rust_data / filename).read_bytes() == (
            python_data / filename
        ).read_bytes()


def test_color_uses_shared_bandpass_composition_and_explicit_mapping_types() -> None:
    hints = get_type_hints(estimate_colors)
    assert hints["color_term_composition"] == BandpassComposition | None
    assert hints["filter_projection"] == dict[str, str] | None or str(
        hints["filter_projection"]
    ).endswith("Mapping[str, str] | None")


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array([], type=field.type) for field in schema], schema=schema
    )


@pytest.mark.parametrize(
    "drift", ["missing", "order", "item", "nullability", "metadata", "type"]
)
def test_native_batch_schema_is_exact_before_quivr_wrapping(drift: str) -> None:
    exact = _empty_batch(ColorFit.schema)
    assert len(_color_fit_from_native_batch(exact)) == 0

    fields = list(ColorFit.schema)
    if drift == "missing":
        schema = pa.schema(fields[:-1])
    elif drift == "order":
        fields[3], fields[4] = fields[4], fields[3]
        schema = pa.schema(fields)
    elif drift == "item":
        fields[3] = pa.field(
            "filter_ids",
            pa.large_list(pa.field("value", pa.large_string(), nullable=False)),
            nullable=True,
        )
        schema = pa.schema(fields)
    elif drift == "nullability":
        fields[3] = pa.field(
            "filter_ids",
            pa.large_list(pa.field("item", pa.large_string(), nullable=True)),
            nullable=True,
        )
        schema = pa.schema(fields)
    elif drift == "metadata":
        schema = ColorFit.schema.with_metadata({b"unexpected": b"metadata"})
    else:
        fields[10] = pa.field(
            "colors",
            pa.large_list(pa.field("item", pa.float32(), nullable=False)),
            nullable=True,
        )
        schema = pa.schema(fields)
    drifted = _empty_batch(schema)
    with pytest.raises(ValueError, match="does not exactly match ColorFit.schema"):
        _color_fit_from_native_batch(drifted)


def _record_batch_ipc(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def test_public_facade_rejects_drifted_native_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations, orbits = _fixture()
    fields = list(ColorFit.schema)[:-1]
    drifted = _empty_batch(pa.schema(fields))

    def drifted_native(*_args: object, **_kwargs: object) -> bytes:
        return _record_batch_ipc(drifted)

    monkeypatch.setattr(_rust_native, "estimate_colors_ipc", drifted_native)
    with pytest.raises(ValueError, match="does not exactly match ColorFit.schema"):
        estimate_colors(observations, orbits, "c1c2")


@pytest.mark.parametrize(
    "golden",
    COLOR_FIT_GOLDENS["cases"],
    ids=lambda case: f"{case['fixture']}-{case['phi_type']}",
)
def test_public_facade_all_fixtures_all_phase_models(
    golden: dict[str, object],
) -> None:
    fixture_path = DATA_DIR / f"color_fixture_{golden['fixture']}.npz"
    with np.load(fixture_path, allow_pickle=True) as fx:
        observations = _load_fixture_observations(fx)
        orbits = _load_fixture_orbits(fx)
        result = estimate_colors(
            observations,
            orbits,
            golden["phi_type"],
            force_g_bounds=False,
        )
    row = _row(result)
    tolerance = COLOR_FIT_GOLDENS["absolute_tolerance"]

    assert row["status"] == "ok"
    assert row["filter_ids"] == golden["filter_ids"]
    assert row["num_used_by_filter"] == golden["num_used_by_filter"]
    for name in (
        "absolute_magnitudes",
        "absolute_magnitude_sigmas",
        "colors",
        "color_sigmas",
    ):
        np.testing.assert_allclose(
            row[name], golden[name], rtol=0.0, atol=tolerance, equal_nan=True
        )
    _assert_partition(row)
    assert row["num_obs"] == len(observations)
    assert sum(row["num_used_by_filter"]) == row["num_used"]
    assert len(row["magnitude_covariance"]) == len(row["filter_ids"]) ** 2

    forced = _row(
        estimate_colors(
            observations,
            orbits,
            golden["phi_type"],
            force_g_bounds=True,
        )
    )
    assert forced["status"] == golden["forced_status"]
    if golden["forced_status"] == "phase_param_out_of_bounds":
        assert forced["filter_ids"] == golden["filter_ids"]
        assert forced["num_used_by_filter"] == golden["num_used_by_filter"]
        for name in SCIENCE_LIST_FIELDS:
            assert forced[name] is None
    else:
        assert forced["filter_ids"] == golden["filter_ids"]
        assert forced["num_used_by_filter"] == golden["num_used_by_filter"]
        for name in (
            "absolute_magnitudes",
            "absolute_magnitude_sigmas",
            "colors",
            "color_sigmas",
        ):
            np.testing.assert_allclose(
                forced[name], golden[name], rtol=0.0, atol=tolerance
            )

    for name in LIST_FIELDS:
        array = result.table.column(name).combine_chunks()
        assert isinstance(array, pa.LargeListArray)
        assert array.values.null_count == 0


@pytest.mark.parametrize("filter_id", ["ATLAS_c", "ATLAS_o", "PS1_w", "DECam_VR", "V"])
def test_native_mode_fits_generic_canonical_filters(filter_id: str) -> None:
    observations, orbits = _fixture()
    observations = observations.set_column(
        "filter",
        pa.array([filter_id] * len(observations), type=observations.filter.type),
    )
    row = _row(estimate_colors(observations, orbits, "c1c2", force_g_bounds=False))
    assert row["status"] == "ok"
    assert row["filter_ids"] == [filter_id]
    assert row["num_used_by_filter"] == [row["num_used"]]
    assert row["color_lhs_filter_ids"] == []
    assert row["colors"] == []


def test_projection_facade_matches_frozen_direct_rust_goldens() -> None:
    observations, orbits = _fixture()
    bands = observations.filter.to_pylist()
    bands = ["SDSS_g" if band == "g" else band for band in bands]
    observations = observations.set_column(
        "filter", pa.array(bands, type=observations.filter.type)
    )
    projection = {
        "SDSS_g": "LSST_g",
        "LSST_i": "LSST_i",
        "LSST_r": "LSST_r",
    }
    projected = _row(
        estimate_colors(
            observations,
            orbits,
            "HG12star",
            force_g_bounds=False,
            filter_projection=projection,
            color_term_composition="C",
        )
    )
    assert projected["status"] == "ok"
    assert projected["filter_ids"] == ["LSST_g", "LSST_i", "LSST_r"]
    assert projected["num_unsupported_filter"] == 0
    # Frozen from direct public Rust (bitwise facade parity): native LSST_g H
    # plus the physical C-template SDSS_g -> LSST_g delta.
    assert projected["absolute_magnitudes"] == pytest.approx(
        [20.728124818909933, 20.170046369331104, 20.32369900320434],
        abs=5.0e-12,
    )

    missing_identity = dict(projection)
    del missing_identity["LSST_i"]
    partial = _row(
        estimate_colors(
            observations,
            orbits,
            "HG12star",
            force_g_bounds=False,
            filter_projection=missing_identity,
            color_term_composition=(0.4, 0.6),
        )
    )
    assert "LSST_i" not in partial["filter_ids"]
    assert partial["num_unsupported_filter"] > 0
    # Pins adapter mix order against the same direct-Rust consumer path.
    assert partial["absolute_magnitudes"] == pytest.approx(
        [20.721172158560325, 20.326516038223083], abs=5.0e-12
    )
    assert partial["colors"] == pytest.approx([0.39465612033724184], abs=5.0e-12)
    _assert_partition(partial)


@pytest.mark.parametrize(
    "projection,composition",
    [(None, "C"), ({"LSST_g": "LSST_g"}, None)],
)
def test_projection_and_composition_must_be_paired(
    projection: dict[str, str] | None,
    composition: BandpassComposition | None,
) -> None:
    observations, orbits = _fixture()
    with pytest.raises(ValueError, match="projection|composition"):
        estimate_colors(
            observations,
            orbits,
            "c1c2",
            filter_projection=projection,
            color_term_composition=composition,
        )


def test_strict_mapping_accounts_for_generic_fallback() -> None:
    observations, orbits = _fixture()
    stations = observations.observatory_code.to_pylist()
    bands = observations.filter.to_pylist()
    stations[0] = "500"
    bands[0] = "g"
    observations = observations.set_column(
        "observatory_code",
        pa.array(stations, type=observations.observatory_code.type),
    ).set_column("filter", pa.array(bands, type=observations.filter.type))

    loose = _row(estimate_colors(observations, orbits, "c1c2", force_g_bounds=False))
    strict = _row(
        estimate_colors(
            observations,
            orbits,
            "c1c2",
            force_g_bounds=False,
            strict_band_mapping=True,
        )
    )
    assert loose["num_unsupported_filter"] == 0
    assert strict["num_unsupported_filter"] == 1
    assert loose["num_used"] == strict["num_used"] + 1
    _assert_partition(loose)
    _assert_partition(strict)


def test_nonlexical_first_appearance_order_is_preserved() -> None:
    fixture_order = [COLOR_FIXTURES[2], COLOR_FIXTURES[0], COLOR_FIXTURES[1]]
    loaded = [np.load(path, allow_pickle=True) for path in fixture_order]
    try:
        observations = [_load_fixture_observations(fx) for fx in loaded]
        orbits = [_load_fixture_orbits(fx) for fx in loaded]
        expected = [table.object_id[0].as_py() for table in observations]
        result = estimate_colors(
            qv.concatenate(observations),
            qv.concatenate(orbits),
            "c1c2",
            force_g_bounds=False,
        )
    finally:
        for fx in loaded:
            fx.close()

    assert expected != sorted(expected)
    assert result.object_id.to_pylist() == expected


def test_multi_object_order_null_ids_and_orbit_cardinality_isolation() -> None:
    if len(COLOR_FIXTURES) < 3:
        pytest.skip("Need three fixtures")
    loaded = [np.load(path, allow_pickle=True) for path in COLOR_FIXTURES[:3]]
    try:
        observations = [_load_fixture_observations(fx) for fx in loaded]
        orbits = [_load_fixture_orbits(fx) for fx in loaded]
        first_id, second_id, third_id = [
            table.object_id[0].as_py() for table in observations
        ]
        first = observations[0].set_column(
            "object_id",
            pc.if_else(
                pa.array([True] + [False] * (len(observations[0]) - 1)),
                pa.nulls(len(observations[0]), pa.large_string()),
                observations[0].object_id,
            ),
        )
        result = estimate_colors(
            qv.concatenate([first, observations[1], observations[2]]),
            qv.concatenate([orbits[1], orbits[1], orbits[2]]),
            "c1c2",
            force_g_bounds=False,
        )
    finally:
        for fx in loaded:
            fx.close()

    rows = result.table.to_pylist()
    assert [row["object_id"] for row in rows] == [first_id, second_id, third_id]
    assert [row["status"] for row in rows] == [
        "no_orbit",
        "multiple_orbits",
        "ok",
    ]
    assert rows[0]["num_obs"] == len(observations[0]) - 1
    assert all(rows[0][name] is None for name in LIST_FIELDS)
    assert all(rows[1][name] is None for name in LIST_FIELDS)
    _assert_partition(rows[2])


def test_utc_orbit_epochs_and_call_level_frame_validation() -> None:
    observations, orbits = _fixture()
    utc_coordinates = orbits.coordinates.set_column(
        "time", orbits.coordinates.time.rescale("utc")
    )
    utc = orbits.set_column("coordinates", utc_coordinates)
    base = _row(estimate_colors(observations, orbits, "c1c2", force_g_bounds=False))
    converted = _row(estimate_colors(observations, utc, "c1c2", force_g_bounds=False))
    assert converted["colors"] == pytest.approx(base["colors"], abs=1.0e-12)

    c = orbits.coordinates
    equatorial = CartesianCoordinates.from_kwargs(
        x=c.x,
        y=c.y,
        z=c.z,
        vx=c.vx,
        vy=c.vy,
        vz=c.vz,
        time=c.time,
        origin=c.origin,
        frame="equatorial",
        covariance=c.covariance,
    )
    with pytest.raises(ValueError, match="requires ecliptic orbit coordinates"):
        estimate_colors(
            observations,
            orbits.set_column("coordinates", equatorial),
            "c1c2",
        )


def test_call_level_time_errors_preserve_exact_context() -> None:
    observations, orbits = _fixture()
    ancient_observations = observations.set_column(
        "time",
        Timestamp.from_kwargs(
            days=[-3_000_000] * len(observations),
            nanos=[0] * len(observations),
            scale="utc",
        ),
    )
    with pytest.raises(ValueError) as observation_error:
        estimate_colors(ancient_observations, orbits, "c1c2")
    assert str(observation_error.value) == (
        "cannot convert observation times to TDB: invalid record batch: "
        "eraUtctai failed with ERFA status -1"
    )

    ancient_epoch = Timestamp.from_kwargs(days=[-3_000_000], nanos=[0], scale="utc")
    ancient_orbit = orbits.set_column(
        "coordinates", orbits.coordinates.set_column("time", ancient_epoch)
    )
    with pytest.raises(ValueError) as epoch_error:
        estimate_colors(observations, ancient_orbit, "c1c2")
    assert str(epoch_error.value) == (
        "cannot convert orbit epochs to TDB: invalid record batch: "
        "eraUtctai failed with ERFA status -1"
    )


def test_per_object_errors_are_clean_rust_native_messages() -> None:
    observations, orbits = _fixture()
    codes = observations.observatory_code.to_pylist()
    codes[0] = "ZZZ"
    unknown_code = observations.set_column(
        "observatory_code",
        pa.array(codes, type=observations.observatory_code.type),
    )
    row = _row(estimate_colors(unknown_code, orbits, "c1c2"))
    assert row["status"] == "failed"
    assert row["error"] == (
        "ZZZ is not a valid MPC observatory code and was not found in SPICE kernels."
    )

    coordinates = orbits.coordinates
    earth = CartesianCoordinates.from_kwargs(
        x=coordinates.x,
        y=coordinates.y,
        z=coordinates.z,
        vx=coordinates.vx,
        vy=coordinates.vy,
        vz=coordinates.vz,
        time=coordinates.time,
        origin=Origin.from_kwargs(code=pa.repeat("EARTH", len(coordinates))),
        frame="ecliptic",
        covariance=coordinates.covariance,
    )
    row = _row(
        estimate_colors(
            observations,
            orbits.set_column("coordinates", earth),
            "c1c2",
        )
    )
    assert row["status"] == "failed"
    assert row["error"] == (
        "color estimation requires heliocentric (SUN-origin) orbit coordinates"
    )


def test_active_nongrav_is_isolated_failure() -> None:
    observations, orbits = _fixture()
    nongrav = Orbits.from_kwargs(
        orbit_id=orbits.orbit_id,
        object_id=orbits.object_id,
        coordinates=orbits.coordinates,
        non_gravitational_parameters=NonGravitationalParameters.from_kwargs(A1=[1e-12]),
    )
    row = _row(estimate_colors(observations, nongrav, "c1c2"))
    assert row["status"] == "failed"
    assert "non-gravitational" in row["error"]
    assert all(row[name] is None for name in LIST_FIELDS)


def _with_nulls(
    observations: SourceCatalog, column: str, mask: np.ndarray
) -> SourceCatalog:
    values = getattr(observations, column)
    return observations.set_column(
        column, pc.if_else(pa.array(mask), pa.nulls(len(mask), values.type), values)
    )


def test_mixed_null_photometry_precedence_and_partition() -> None:
    observations, orbits = _fixture("color_fixture_2025_MS34.npz")
    index = np.arange(len(observations)) % 9
    null_mag = (index == 0) | (index == 2)
    null_sigma = index == 4
    null_filter = (index == 2) | (index == 7)

    mixed = _with_nulls(observations, "mag", null_mag)
    mixed = _with_nulls(mixed, "mag_sigma", null_sigma)
    mixed = _with_nulls(mixed, "filter", null_filter)
    row = _row(estimate_colors(mixed, orbits, "c1c2", force_g_bounds=False))

    assert row["status"] == "ok"
    assert row["num_obs"] == len(observations)
    assert row["num_invalid"] == int(np.sum(null_mag | null_sigma))
    assert row["num_unsupported_filter"] == int(
        np.sum(null_filter & ~(null_mag | null_sigma))
    )
    _assert_partition(row)


@pytest.mark.parametrize(
    ("column", "invalid", "unsupported"),
    [("mag", True, False), ("mag_sigma", True, False), ("filter", False, True)],
)
def test_all_null_photometry_is_accounted(
    column: str, invalid: bool, unsupported: bool
) -> None:
    observations, orbits = _fixture("color_fixture_2025_MU8.npz")
    empty = _with_nulls(observations, column, np.ones(len(observations), dtype=bool))
    row = _row(estimate_colors(empty, orbits, "c1c2"))

    assert row["status"] == "no_usable_photometry"
    assert row["num_invalid"] == (len(observations) if invalid else 0)
    assert row["num_unsupported_filter"] == (len(observations) if unsupported else 0)
    assert all(row[name] is None for name in LIST_FIELDS)
    _assert_partition(row)


@pytest.mark.parametrize("empty_kind", ["canonical", "zero-row-ecliptic"])
def test_empty_orbits_produce_isolated_no_orbit_rows(empty_kind: str) -> None:
    observations, orbits = _fixture()
    second_object_id = "second-empty-orbit-object"
    second = observations.set_column(
        "object_id",
        pa.array([second_object_id] * len(observations), type=pa.large_string()),
    )
    observations = qv.concatenate([observations, second])
    empty_orbits = (
        Orbits.empty()
        if empty_kind == "canonical"
        else orbits.apply_mask(pa.array([False]))
    )

    rows = estimate_colors(observations, empty_orbits, "c1c2").table.to_pylist()

    assert [row["object_id"] for row in rows] == [
        orbits.object_id[0].as_py(),
        second_object_id,
    ]
    for no_orbit in rows:
        assert no_orbit["status"] == "no_orbit"
        assert no_orbit["num_obs"] == len(second)
        assert no_orbit["num_invalid"] is None
        assert all(no_orbit[name] is None for name in LIST_FIELDS)


def test_status_list_and_scalar_nullability() -> None:
    observations, orbits = _fixture()
    unmatched = orbits.set_column(
        "object_id", pa.array(["unmatched"], type=orbits.object_id.type)
    )
    no_orbit = _row(estimate_colors(observations, unmatched, "c1c2"))
    assert no_orbit["status"] == "no_orbit"
    assert no_orbit["num_obs"] == len(observations)
    assert no_orbit["num_invalid"] is None
    assert all(no_orbit[name] is None for name in LIST_FIELDS)

    unusable = observations.set_column(
        "mag", pa.nulls(len(observations), observations.mag.type)
    )
    none = _row(estimate_colors(unusable, orbits, "c1c2"))
    assert none["status"] == "no_usable_photometry"
    assert none["num_invalid"] == none["num_obs"]
    _assert_partition(none)
    assert all(none[name] is None for name in LIST_FIELDS)


def test_fresh_process_facade_lazily_initializes_observer_geometry_subset() -> None:
    code = textwrap.dedent("""
        import pyarrow as pa

        from adam_core import _rust_native
        from adam_core.photometry.color_determination import estimate_colors
        from adam_core.photometry.tests.test_color_determination import _fixture
        from adam_core.utils.spice import OBSERVER_GEOMETRY_KERNELS
        from adam_core.utils.spice_backend import get_backend

        backend = get_backend()
        inspect_registered_kernels = backend.registered_kernels
        inspect_mpc_obscodes_loaded = backend.mpc_obscodes_loaded
        assert inspect_registered_kernels() == []
        assert inspect_mpc_obscodes_loaded() == 0
        observations, orbits = _fixture()
        empty_orbits = orbits.apply_mask(pa.array([False] * len(orbits)))

        target = "estimate_colors_ipc"
        native_calls = []
        native_originals = {}
        for name in dir(_rust_native):
            native = getattr(_rust_native, name)
            if not callable(native):
                continue
            native_originals[name] = native

            def tracked_native(*args, _name=name, _native=native, **kwargs):
                native_calls.append(_name)
                if _name != target:
                    raise AssertionError(f"unexpected chained native call: {_name}")
                return _native(*args, **kwargs)

            setattr(_rust_native, name, tracked_native)

        backend_calls = []
        patched_backend_methods = []
        for name in dir(backend):
            if name.startswith("_"):
                continue
            method = getattr(backend, name)
            if not callable(method):
                continue
            patched_backend_methods.append(name)

            def blocked_backend_call(*args, _name=name, **kwargs):
                backend_calls.append(_name)
                raise AssertionError(f"unexpected chained backend call: {_name}")

            setattr(backend, name, blocked_backend_call)

        try:
            empty_result = estimate_colors(
                observations, empty_orbits, "c1c2", force_g_bounds=False
            )
            assert empty_result.status.to_pylist() == ["no_orbit"]
            assert inspect_registered_kernels() == []
            assert inspect_mpc_obscodes_loaded() == 0

            result = estimate_colors(
                observations, orbits, "c1c2", force_g_bounds=False
            )
        finally:
            for name, native in native_originals.items():
                setattr(_rust_native, name, native)
            for name in patched_backend_methods:
                delattr(backend, name)

        assert result.status.to_pylist() == ["ok"]
        assert native_calls == [target, target]
        assert backend_calls == []
        assert inspect_registered_kernels() == list(
            map(str, OBSERVER_GEOMETRY_KERNELS)
        )
        assert inspect_mpc_obscodes_loaded() > 0
        """)
    environment = dict(os.environ)
    environment["ADAM_CORE_REQUIRE_RUST_BACKEND"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[4],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_native_result_decoder_rejects_zero_or_multiple_batches() -> None:
    sink = pa.BufferOutputStream()
    schema = pa.schema([pa.field("value", pa.int64())])
    with pa.ipc.new_stream(sink, schema):
        pass
    with pytest.raises(ValueError, match="contains no record batch"):
        _native_record_batch(sink.getvalue().to_pybytes())

    sink = pa.BufferOutputStream()
    batch = pa.record_batch({"value": [1]})
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
        writer.write_batch(batch)
    with pytest.raises(ValueError, match="more than one record batch"):
        _native_record_batch(sink.getvalue().to_pybytes())
