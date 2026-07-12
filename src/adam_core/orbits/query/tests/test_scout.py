"""Tests for the scout module."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_core import _rust_native
from adam_core._rust.arrow import table_from_record_batch
from adam_core.orbits.variants import VariantOrbits

from ..scout import ScoutObjectSummary, ScoutOrbit, scout_orbits_to_variant_orbits

SCOUT_DATA = Path(__file__).parent / "testdata" / "scout"


def test_scout_orbits_to_variant_orbits():
    """Test that scout orbits are correctly converted to variant orbits."""
    # Create a mock scout orbits table
    scout_data = {
        "idx": [0, 1],
        "epoch": ["60000.0", "60000.0"],
        "ec": ["0.5", "0.51"],
        "qr": ["1.0", "1.01"],
        "tp": ["59000.0", "59000.0"],
        "om": ["10.0", "10.1"],
        "w": ["50.0", "50.1"],
        "inc": ["10.0", "10.1"],
        "H": ["20.0", "20.0"],
        "dca": ["0.1", "0.1"],
        "tca": ["0.1", "0.1"],
        "moid": ["0.1", "0.1"],
        "vinf": ["0.1", "0.1"],
        "geoEcc": ["0.1", "0.1"],
        "impFlag": [0, 0],
    }
    scout_orbits = ScoutOrbit.from_kwargs(**scout_data)

    # Convert to variant orbits
    variant_orbits = scout_orbits_to_variant_orbits("2024AA", scout_orbits)

    # Check that the output has the expected structure
    assert len(variant_orbits) == len(scout_orbits)
    assert variant_orbits.coordinates.frame == "ecliptic"
    assert pc.all(pc.equal(variant_orbits.coordinates.origin.code, "SUN")).as_py()

    # Check that the object IDs are correct
    assert variant_orbits.object_id.to_pylist() == ["2024AA", "2024AA"]

    # Check that the orbit IDs are correct
    assert variant_orbits.orbit_id.to_pylist() == ["0", "1"]

    # Check that the variant IDs are unique
    assert len(pc.unique(variant_orbits.variant_id)) == len(scout_orbits)

    # Check that the time is correct
    np.testing.assert_array_equal(
        variant_orbits.coordinates.time.jd(), pc.cast(scout_orbits.epoch, pa.float64())
    )


def test_recorded_scout_summary_and_orbits_are_rust_owned():
    summary_payload = (SCOUT_DATA / "summary.json").read_text()
    summary = table_from_record_batch(
        ScoutObjectSummary,
        _rust_native.get_scout_objects_arrow(summary_payload),
    )
    assert len(summary) > 0
    object_id = summary.objectName[0].as_py()

    variants = table_from_record_batch(
        VariantOrbits,
        _rust_native.query_scout_arrow(
            [object_id], [(SCOUT_DATA / "orbits.json").read_text()]
        ),
    )
    assert len(variants) == 1000
    assert set(variants.object_id.to_pylist()) == {object_id}
    assert np.all(np.isfinite(variants.coordinates.values))


def test_scout_products_have_rust_owned_timing():
    for kind, payload in [
        (
            "neocc",
            (Path(__file__).parent / "testdata" / "neocc" / "2024YR4.ke1").read_text(),
        ),
        ("scout-summary", (SCOUT_DATA / "summary.json").read_text()),
        ("scout", (SCOUT_DATA / "orbits.json").read_text()),
    ]:
        samples = np.asarray(
            _rust_native.benchmark_query_client_processing(kind, [payload], 2, 2, 1)
        )
        assert samples.shape == (2, 2)
        assert np.all(samples > 0.0)
