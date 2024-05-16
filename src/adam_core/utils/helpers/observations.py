from importlib.resources import files
from typing import Tuple

import numpy as np
import pandas as pd

from ...observations.associations import Associations
from ...observations.detections import PointSourceDetections
from ...observations.exposures import Exposures
from ...time import Timestamp


def make_observations() -> Tuple[Exposures, PointSourceDetections, Associations]:
    """
    Create an Exposures, PointSourceDetections, and Associations table
    using predicted ephemerides from JPL Horizons included in adam_core.

    Returns
    -------
    exposures : `~adam_core.observations.exposures.Exposures`
        Table of exposures.
    detections : `~adam_core.observations.detections.PointSourceDetections`
        Table of detections.
    associations : `~adam_core.observations.associations.Associations`
        Table of associations.
    """
    # Load the ephemeris (these are predicted ephemerides generated via JPL Horizons)
    ephemeris = pd.read_csv(
        files("adam_core.utils.helpers.data").joinpath("ephemeris.csv")
    )

    # Now lets create simulated exposures and detections
    exposure_times = {
        "X05": 30,
        "W84": 60,
    }
    for observatory_code in ephemeris["observatory_code"].unique():
        observatory_mask = ephemeris["observatory_code"] == observatory_code

        # For X05, lets give every observation a time thats between the start
        # and end time of an exposure. Note that with this current set up we
        # might get two exposures overlapping in time with different exposure start times
        if observatory_code == "X05":
            rng = np.random.default_rng(seed=20233202)
            observation_times = ephemeris[observatory_mask]["mjd_utc"].values
            exposure_start_times = observation_times + rng.uniform(
                -exposure_times[observatory_code] / 86400,
                0,
                size=len(observation_times),
            )
            ephemeris.loc[observatory_mask, "exposure_start"] = exposure_start_times

        # For everything else (which is just W84 at the moment) every observation is reported at the midpoint
        else:
            ephemeris.loc[observatory_mask, "exposure_start"] = ephemeris[
                observatory_mask
            ]["mjd_utc"].values - (exposure_times[observatory_code] / 2 / 86400)

    # Create an exposures table
    exposures = (
        ephemeris.groupby(by=["observatory_code", "exposure_start"])
        .size()
        .to_frame(name="num_obs")
        .reset_index()
    )
    exposures.sort_values(by="exposure_start", inplace=True, ignore_index=True)
    exposures.insert(
        0,
        "exposure_id",
        exposures["observatory_code"].astype(str)
        + "_"
        + [f"{i:04d}" for i in range(len(exposures))],
    )
    for observatory_code in exposures["observatory_code"].unique():
        exposures.loc[exposures["observatory_code"] == observatory_code, "duration"] = (
            exposure_times[observatory_code]
        )
    exposures["filter"] = "V"

    # Attached exposure IDs to the ephemerides
    ephemeris = ephemeris.merge(
        exposures[["observatory_code", "exposure_start", "exposure_id"]],
        on=["observatory_code", "exposure_start"],
    )

    # Create detections dataframe
    detections_df = ephemeris[
        ["exposure_id", "observatory_code", "mjd_utc", "RA", "DEC", "V", "targetname"]
    ].copy()
    detections_df.rename(
        columns={
            "RA": "ra",
            "DEC": "dec",
            "V": "mag",
        },
        inplace=True,
    )

    # Lets only report astrometric errors for one of the observatories
    detections_df.loc[detections_df["observatory_code"] == "X05", "ra_sigma"] = (
        0.1 / 3600
    )
    detections_df.loc[detections_df["observatory_code"] == "X05", "dec_sigma"] = (
        0.1 / 3600
    )
    # Lets report photometric errors for all observatories
    detections_df["mag_sigma"] = 0.1
    detections_df.sort_values(
        by=["mjd_utc", "observatory_code"], inplace=True, ignore_index=True
    )
    detections_df["detection_id"] = [f"obs_{i:04d}" for i in range(len(detections_df))]

    # Create exposures table
    exposures = Exposures.from_kwargs(
        id=exposures["exposure_id"],
        start_time=Timestamp.from_mjd(exposures["exposure_start"].values, scale="utc"),
        duration=exposures["duration"].values,
        filter=exposures["filter"].values,
        observatory_code=exposures["observatory_code"].values,
    )

    # Create detections table
    detections = PointSourceDetections.from_kwargs(
        id=detections_df["detection_id"],
        exposure_id=detections_df["exposure_id"],
        time=Timestamp.from_mjd(detections_df["mjd_utc"].values, scale="utc"),
        ra=detections_df["ra"].values,
        dec=detections_df["dec"].values,
        mag=detections_df["mag"].values,
        ra_sigma=detections_df["ra_sigma"].values,
        dec_sigma=detections_df["dec_sigma"].values,
        mag_sigma=detections_df["mag_sigma"].values,
    )

    # Create associations table
    associations = Associations.from_kwargs(
        detection_id=detections.id,
        object_id=detections_df["targetname"],
    )
    return exposures, detections, associations
