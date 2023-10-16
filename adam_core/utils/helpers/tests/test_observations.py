from importlib.resources import files

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from ..observations import make_observations


def test_make_observations():
    # Test that we get two tables from make observations are
    # the expected length
    exposures, detections, associations = make_observations()
    assert len(exposures) == 2412
    assert len(detections) == 2520
    assert len(associations) == 2520

    # Load original ephemeris file
    ephemeris = pd.read_csv(
        files("adam_core.utils.helpers.data").joinpath("ephemeris.csv")
    )
    ephemeris.sort_values(by=["mjd_utc", "observatory_code"], inplace=True)
    ephemeris["obs_id"] = [f"obs_{i:04d}" for i in range(len(ephemeris))]

    # Test that the RAs, Decs, and associations were correctly set
    for object_id in associations.object_id.unique().to_numpy(zero_copy_only=False):

        # Get the obs_ids for this object from the associations table
        associations_mask = pc.equal(associations.object_id, object_id)
        associations_object = associations.apply_mask(associations_mask)
        obs_ids = associations_object.detection_id

        # Get the RA and Decs for this object from the detections table
        detections_mask = pc.is_in(detections.id, obs_ids)
        detections_object = detections.apply_mask(detections_mask)
        ras = detections_object.ra.to_numpy(zero_copy_only=False)
        decs = detections_object.dec.to_numpy(zero_copy_only=False)

        # Get the input ephemeris for this object
        ephemeris_object = ephemeris[ephemeris["targetname"] == object_id]
        obs_ids_ephemeris = ephemeris_object["obs_id"].values
        ras_ephemeris = ephemeris_object["RA"].values
        decs_ephemeris = ephemeris_object["DEC"].values

        # Test that the obs_ids are identical
        np.testing.assert_array_equal(obs_ids, obs_ids_ephemeris)

        # Test that the RA and Decs are close
        np.testing.assert_allclose(ras, ras_ephemeris, atol=1e-15, rtol=0)
        np.testing.assert_allclose(decs, decs_ephemeris, atol=1e-15, rtol=0)
