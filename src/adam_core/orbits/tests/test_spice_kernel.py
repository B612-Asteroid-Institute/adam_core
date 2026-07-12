import json
from pathlib import Path

import numpy as np
import pytest
from adam_assist import ASSISTPropagator

from ...constants import KM_P_AU, S_P_DAY
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...time import Timestamp
from ...utils.spice_backend import get_backend
from ..orbits import Orbits
from ..arrow_bridge import orbits_to_ipc
from ..spice_kernel import (
    fit_chebyshev,
    orbits_to_spk,
    write_spkw03_segment,
    write_spkw09_segment,
)


def test_orbits_to_spk(tmp_path):
    # Create test orbit
    t0 = Timestamp.from_mjd([60000.0], scale="tdb")
    origin = Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"])

    # Create a simple circular orbit
    orbits = Orbits.from_kwargs(
        orbit_id=["test_orbit"],
        object_id=["test_object"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[1.0],
            z=[0.0],
            vx=[0.0],
            vy=[1.0],
            vz=[0.0],
            time=t0,
            origin=origin,
            frame="equatorial",
        ),
    )

    # Generate SPK file
    spk_file = tmp_path / "test.bsp"
    end_time = t0.add_fractional_days(100.0)

    orbits_to_spk(
        orbits,
        str(spk_file),
        start_time=t0,
        end_time=end_time,
        propagator=ASSISTPropagator(),
        step_days=1.0,
        window_days=10.0,
    )

    # Verify file exists
    assert spk_file.exists()

    # Load SPK and verify contents via the pure-Rust backend
    backend = get_backend()
    backend.furnsh(str(spk_file))
    try:
        et0 = t0.et()[0].as_py()
        state = backend.spkez(1000000, et0, "J2000", 0)

        # Convert backend state (km, km/s) to our units (au, au/day)
        state = np.asarray(state, dtype=np.float64).copy()
        state[:3] /= KM_P_AU
        state[3:] *= S_P_DAY / KM_P_AU

        input_state = orbits.coordinates.values[0]
        assert np.allclose(state, input_state, rtol=1e-10, atol=1e-10)
    finally:
        backend.unload(str(spk_file))


FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "spk_fit_fixture_2026-07-12.json"
)


def _fixture_coordinates():
    fixture = json.loads(FIXTURE_PATH.read_text())
    values = np.asarray(fixture["values"], dtype=np.float64)
    count = len(values)
    coordinates = CartesianCoordinates.from_kwargs(
        x=values[:, 0],
        y=values[:, 1],
        z=values[:, 2],
        vx=values[:, 3],
        vy=values[:, 4],
        vz=values[:, 5],
        time=Timestamp.from_kwargs(
            days=fixture["days"], nanos=[0] * count, scale="tdb"
        ),
        origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * count),
        frame="equatorial",
    )
    return fixture, coordinates


def test_fit_chebyshev_matches_frozen_legacy():
    fixture, coordinates = _fixture_coordinates()
    for case in fixture["cases"]:
        coefficients, mid, half = fit_chebyshev(
            coordinates,
            case["window_start"],
            case["window_end"],
            case["degree"],
        )
        np.testing.assert_allclose(
            coefficients,
            np.asarray(case["coefficients"]),
            rtol=2e-12,
            atol=2e-7,
            err_msg=case["name"],
        )
        assert mid == case["actual_mid"]
        assert half == case["actual_half"]


def _sampled_orbits(groups=2):
    rows = []
    orbit_ids = tuple(f"orbit_{index:03d}" for index in range(groups))
    for day_index in range(20):
        for orbit_index, orbit_id in enumerate(orbit_ids):
            t = float(day_index)
            rows.append(
                (
                    orbit_id,
                    60000 + day_index,
                    1.0 + orbit_index + 0.001 * t,
                    -0.5 + 0.002 * t,
                    0.1 * orbit_index,
                    0.0001,
                    0.0002,
                    -0.00005,
                )
            )
    # Reverse to pin Rust grouping by first appearance and per-group epoch sort.
    rows.reverse()
    return Orbits.from_kwargs(
        orbit_id=[row[0] for row in rows],
        object_id=[None] * len(rows),
        coordinates=CartesianCoordinates.from_kwargs(
            x=[row[2] for row in rows],
            y=[row[3] for row in rows],
            z=[row[4] for row in rows],
            vx=[row[5] for row in rows],
            vy=[row[6] for row in rows],
            vz=[row[7] for row in rows],
            time=Timestamp.from_kwargs(
                days=[row[1] for row in rows], nanos=[0] * len(rows), scale="tdb"
            ),
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * len(rows)),
            frame="equatorial",
        ),
    )


def test_orbits_to_spk_type9_no_provider_grouping_readback_and_atomicity(tmp_path):
    orbits = _sampled_orbits()
    output = tmp_path / "sampled.bsp"
    mappings = orbits_to_spk(
        orbits,
        str(output),
        start_time=Timestamp.from_mjd([60000.0], scale="tdb"),
        end_time=Timestamp.from_mjd([60019.0], scale="tdb"),
        propagator=None,
        target_id_start=2000,
        kernel_type="w09",
    )
    # Reversed rows encounter the lexically last generated orbit first.
    assert mappings == {"orbit_001": 2000, "orbit_000": 2001}
    assert output.exists()
    assert not output.with_suffix(".tmp").exists()

    backend = get_backend()
    backend.furnsh(str(output))
    try:
        for orbit_id, target_id in mappings.items():
            group = orbits.select("orbit_id", orbit_id).sort_by(
                ["coordinates.time.days", "coordinates.time.nanos"]
            )
            for index in (0, 10, 19):
                et = group.coordinates.time.et()[index].as_py()
                state = np.asarray(
                    backend.spkez(target_id, et, "J2000", 0), dtype=np.float64
                )
                state[:3] /= KM_P_AU
                state[3:] *= S_P_DAY / KM_P_AU
                np.testing.assert_allclose(
                    state, group.coordinates.values[index], rtol=2e-12, atol=2e-12
                )
    finally:
        backend.unload(str(output))


def test_public_type3_and_type9_segment_shims_are_one_native_call(tmp_path):
    from adam_core._rust import naif_spk_writer

    orbits = (
        _sampled_orbits()
        .select("orbit_id", "orbit_000")
        .sort_by(["coordinates.time.days", "coordinates.time.nanos"])
    )
    start_et = orbits.coordinates.time.et()[0].as_py()
    end_et = orbits.coordinates.time.et()[-1].as_py()

    writer3 = naif_spk_writer("shim-type3")
    write_spkw03_segment(
        orbits,
        writer3,
        3000,
        start_et,
        end_et,
        window_seconds=10 * S_P_DAY,
        cheby_degree=3,
    )
    output3 = tmp_path / "shim_type3.bsp"
    writer3.write(str(output3))
    assert output3.exists()

    writer9 = naif_spk_writer("shim-type9")
    write_spkw09_segment(orbits, writer9, 3001, 0.0, 0.0)
    output9 = tmp_path / "shim_type9.bsp"
    writer9.write(str(output9))
    assert output9.exists()


def test_spk_multi_summary_record_chain_reads_late_segments(tmp_path):
    orbits = _sampled_orbits(groups=30)
    output = tmp_path / "multi_summary.bsp"
    mappings = orbits_to_spk(
        orbits,
        str(output),
        start_time=Timestamp.from_mjd([60000.0], scale="tdb"),
        end_time=Timestamp.from_mjd([60019.0], scale="tdb"),
        kernel_type="w09",
        target_id_start=5000,
    )
    assert len(mappings) == 30
    backend = get_backend()
    backend.furnsh(str(output))
    try:
        for orbit_id in ("orbit_029", "orbit_004", "orbit_000"):
            target_id = mappings[orbit_id]
            group = orbits.select("orbit_id", orbit_id).sort_by(
                ["coordinates.time.days", "coordinates.time.nanos"]
            )
            et = group.coordinates.time.et()[10].as_py()
            state = np.asarray(
                backend.spkez(target_id, et, "J2000", 0), dtype=np.float64
            )
            state[:3] /= KM_P_AU
            state[3:] *= S_P_DAY / KM_P_AU
            np.testing.assert_allclose(
                state, group.coordinates.values[10], rtol=2e-12, atol=2e-12
            )
    finally:
        backend.unload(str(output))


def test_public_segment_writer_supports_multiple_summary_records(tmp_path):
    from adam_core._rust import naif_spk_writer

    orbits = _sampled_orbits(groups=30)
    writer = naif_spk_writer("multi-shim")
    for target_offset, (orbit_id, group) in enumerate(orbits.group_by_orbit_id()):
        group = group.sort_by(["coordinates.time.days", "coordinates.time.nanos"])
        write_spkw09_segment(group, writer, 6000 + target_offset, 0.0, 0.0)
    output = tmp_path / "multi_shim.bsp"
    writer.write(str(output))
    backend = get_backend()
    backend.furnsh(str(output))
    try:
        # Reversed input encounters orbit_029 first and orbit_000 last.
        group = orbits.select("orbit_id", "orbit_000").sort_by(
            ["coordinates.time.days", "coordinates.time.nanos"]
        )
        state = np.asarray(
            backend.spkez(6029, group.coordinates.time.et()[10].as_py(), "J2000", 0),
            dtype=np.float64,
        )
        state[:3] /= KM_P_AU
        state[3:] *= S_P_DAY / KM_P_AU
        np.testing.assert_allclose(
            state, group.coordinates.values[10], rtol=2e-12, atol=2e-12
        )
    finally:
        backend.unload(str(output))


def test_spk_invalid_kernel_type_and_native_timing(tmp_path):
    from adam_core import _rust_native

    orbits = _sampled_orbits()
    with pytest.raises(ValueError, match="Invalid kernel type: invalid"):
        orbits_to_spk(
            orbits,
            str(tmp_path / "invalid.bsp"),
            start_time=Timestamp.from_mjd([60000.0], scale="tdb"),
            end_time=Timestamp.from_mjd([60019.0], scale="tdb"),
            kernel_type="invalid",
        )
    samples = _rust_native.benchmark_spk_write_orbits_product(
        orbits_to_ipc(orbits),
        1,
        1,
        0,
        output_file=str(tmp_path / "timing.bsp"),
        target_id_start=4000,
        window_days=32.0,
        kernel_type="w09",
    )
    assert samples[0][0] > 0.0
