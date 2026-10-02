import logging
import time
import uuid
from typing import Literal, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv

from ..coordinates import CartesianCoordinates
from ..coordinates.origin import Origin
from ..coordinates.residuals import Residuals
from ..orbits.orbits import Orbits
from ..propagator import Propagator
from ..time import Timestamp
from ..utils.iter import _iterate_chunks
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits, drop_duplicate_orbits
from .gauss import MU, C
from .observation_uncertainty import ObservationUncertaintyModel

logger = logging.getLogger(__name__)

__all__ = ["initial_orbit_determination"]


def sort_by_id_and_time(
    linkages: qv.AnyTable,
    members: qv.AnyTable,
    observations: OrbitDeterminationObservations,
    linkage_column: str,
) -> Tuple[qv.AnyTable, qv.AnyTable]:
    """
    Sort linkages and linkage members by linkage ID and observation time.

    Parameters
    ----------
    linkages : qv.AnyTable
        Linkages to sort.
    members : qv.AnyTable
        Linkage members to sort.
    observations : Observations
        Observations from which linkage members were generated. Observations
        are used to determine the observation time of each linkage member.
    linkage_column : str
        Column name in the linkage table to use for sorting. For clusters
        this is "cluster_id" and for orbits this is "orbit_id".

    Returns
    -------
    linkages : qv.AnyTable
        Sorted linkages.
    members : qv.AnyTable
        Sorted linkage members.
    """
    from adam_core import _rust_native

    # One Rust crossing owns the observation-time join and the stable
    # (linkage id, observation time) ordering.
    time_table = observations.coordinates.time.table.combine_chunks()
    linkage_order, member_order = _rust_native.sort_linkages_by_id_and_time_numpy(
        linkages.table.column(linkage_column).to_pylist(),
        members.table.column(linkage_column).to_pylist(),
        members.table.column("obs_id").to_pylist(),
        observations.id.to_pylist(),
        time_table.column("days").chunk(0).to_numpy(zero_copy_only=False),
        time_table.column("nanos").chunk(0).to_numpy(zero_copy_only=False),
    )

    linkages = linkages.take(pa.array(linkage_order, type=pa.int64()))
    members = members.take(pa.array(member_order, type=pa.int64()))

    if linkages.fragmented():
        linkages = qv.defragment(linkages)
    if members.fragmented():
        members = qv.defragment(members)
    return linkages, members


def select_observations(
    observations: OrbitDeterminationObservations,
    method: Literal["combinations", "first+middle+last", "thirds"] = "combinations",
) -> npt.NDArray[np.str_]:
    """
    Selects which three observations to use for IOD depending on the method.

    Methods:
        'first+middle+last' : Grab the first, middle and last observations in time.
        'thirds' : Grab the middle observation in the first third, second third, and final third.
        'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
            non-coinciding observation times.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        Pandas DataFrame containing observations with at least a column of observation IDs and a column
        of exposure times.
    method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Which method to use to select observations.
        [Default = 'combinations']

    Returns
    -------
    obs_id : `~numpy.ndarray' (N, 3 or 0)
        An array of selected observation IDs. If three unique observations could
        not be selected then returns an empty array.
    """
    obs_ids = observations.id.to_numpy(zero_copy_only=False)
    if len(obs_ids) < 3:
        return np.array([])

    times = observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    from adam_core import _rust_native

    # One Rust crossing owns percentile/combination selection, arc-length and
    # midpoint ordering, and the unique-time filter; the legacy ValueError
    # message for unknown methods is preserved by the kernel.
    selected_index = np.asarray(
        _rust_native.select_observation_triplets_numpy(
            np.ascontiguousarray(times, dtype=np.float64), method
        ),
        dtype=np.int64,
    )

    # Return an empty array if no observations satisfy the criteria
    if selected_index.shape[0] == 0:
        return np.array([])

    return obs_ids[selected_index]


def _tables_from_native_iod(output: dict) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """Wrap one fused Rust IOD batch product without further computation."""
    orbit_ids = list(output["orbit_ids"])
    if not orbit_ids:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    states = np.asarray(output["states"], dtype=np.float64)
    epochs = Timestamp.from_mjd(
        np.asarray(output["epoch_mjd"], dtype=np.float64), scale="utc"
    )
    orbits = FittedOrbits.from_kwargs(
        orbit_id=orbit_ids,
        object_id=[None] * len(orbit_ids),
        coordinates=CartesianCoordinates.from_kwargs(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            vx=states[:, 3],
            vy=states[:, 4],
            vz=states[:, 5],
            time=epochs,
            origin=Origin.from_kwargs(code=["SUN"] * len(orbit_ids)),
            frame="ecliptic",
        ),
        arc_length=output["arc_length"],
        num_obs=output["num_obs"],
        chi2=output["chi2"],
        reduced_chi2=output["reduced_chi2"],
    )
    residuals = Residuals.from_kwargs(
        values=np.asarray(output["residual_values"], dtype=np.float64).tolist(),
        chi2=output["residual_chi2"],
        dof=output["residual_dof"],
        probability=output["residual_probability"],
    )
    members = FittedOrbitMembers.from_kwargs(
        orbit_id=output["member_orbit_ids"],
        obs_id=output["member_obs_ids"],
        residuals=residuals,
        solution=output["solution"],
        outlier=output["outlier"],
    )
    return orbits, members


def _native_initial_orbit_determination(
    prop: Propagator,
    observations: OrbitDeterminationObservations,
    linkage_ids: list[str],
    member_linkage_ids: list[str],
    member_obs_ids: list[str],
    *,
    min_obs: int,
    min_arc_length: float,
    contamination_percentage: float,
    rchi2_threshold: float,
    observation_selection_method: str,
    light_time: bool,
    chunk_size: int,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    output = prop.initial_orbit_determination(
        observations,
        linkage_ids,
        member_linkage_ids,
        member_obs_ids,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        rchi2_threshold=rchi2_threshold,
        observation_selection_method=observation_selection_method,
        light_time=light_time,
        chunk_size=chunk_size,
        mu=MU,
        speed_of_light=C,
    )
    return _tables_from_native_iod(output)


def iod_worker(
    linkage_ids: npt.NDArray[np.str_],
    observations: OrbitDeterminationObservations,
    linkage_members: FittedOrbitMembers,
    propagator: Type[Propagator],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    rchi2_threshold: float = 200,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations",
    linkage_id_col: str = "cluster_id",
    iterate: bool = False,
    light_time: bool = True,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:

    # Pre-compute linkage_id -> linkage_member rows and obs_id -> observation row
    # so each per-linkage lookup is O(1) rather than O(N_members)/O(N_observations).
    lm_linkage_ids = linkage_members.column(linkage_id_col).to_numpy(
        zero_copy_only=False
    )
    lm_obs_ids = linkage_members.obs_id.to_numpy(zero_copy_only=False)
    lm_rows_by_linkage: dict = {}
    for i, lid in enumerate(lm_linkage_ids):
        lm_rows_by_linkage.setdefault(lid, []).append(i)

    observations_id_arr = observations.id.to_numpy(zero_copy_only=False)
    obs_idx_by_id = {oid: i for i, oid in enumerate(observations_id_arr)}

    iod_orbits_list: list[FittedOrbits] = []
    iod_orbit_members_list: list[FittedOrbitMembers] = []
    for linkage_id in linkage_ids:
        time_start = time.time()
        logger.debug(f"Finding initial orbit for linkage {linkage_id}...")

        lm_rows = lm_rows_by_linkage.get(linkage_id, [])
        obs_row_idx = np.fromiter(
            (obs_idx_by_id[o] for o in lm_obs_ids[lm_rows] if o in obs_idx_by_id),
            dtype=np.int64,
        )
        observations_linkage = observations.take(obs_row_idx)

        # Sort observations by time
        observations_linkage = observations_linkage.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        iod_orbit, iod_orbit_orbit_members = iod(
            observations_linkage,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            rchi2_threshold=rchi2_threshold,
            contamination_percentage=contamination_percentage,
            observation_selection_method=observation_selection_method,
            iterate=iterate,
            light_time=light_time,
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
        )
        if len(iod_orbit) > 0:
            iod_orbit = iod_orbit.set_column("orbit_id", pa.array([linkage_id]))
            iod_orbit_orbit_members = iod_orbit_orbit_members.set_column(
                "orbit_id",
                pa.array([linkage_id for i in range(len(iod_orbit_orbit_members))]),
            )

        time_end = time.time()
        duration = time_end - time_start
        logger.debug(f"IOD for linkage {linkage_id} completed in {duration:.3f}s.")

        iod_orbits_list.append(iod_orbit)
        iod_orbit_members_list.append(iod_orbit_orbit_members)

    iod_orbits = (
        qv.concatenate(iod_orbits_list) if iod_orbits_list else FittedOrbits.empty()
    )
    iod_orbit_members = (
        qv.concatenate(iod_orbit_members_list)
        if iod_orbit_members_list
        else FittedOrbitMembers.empty()
    )
    return iod_orbits, iod_orbit_members


def iod(
    observations: OrbitDeterminationObservations,
    propagator: Type[Propagator],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    rchi2_threshold: float = 200,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations",
    iterate: bool = False,
    light_time: bool = True,
    propagator_kwargs: dict = {},
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run initial orbit determination on a set of observations believed to belong to a single
    object.

    Parameters
    ----------
    observations : `OrbitDeterminationObservations`
        Observations with at least the following attributes:
            id : Observation IDs [str],
            coordinates : SkyCoord object containing observation time, RA, and Dec
            observers : Observer object containing observatory positions
    min_obs : int, optional
        Minimum number of observations that must remain in the linkage. For example, if min_obs is set to 6 and
        a linkage has 8 observations, at most the two worst observations will be flagged as outliers if their individual
        chi2 values exceed the chi2 threshold.
    contamination_percentage : float, optional
        Maximum percent of observations that can flagged as outliers.
    rchi2_threshold : float, optional
        Maximum reduced chi2 required for an initial orbit to be accepted.
    observation_selection_method : {'first+middle+last', 'thirds', 'combinations'}, optional
        Selects which three observations to use for IOD depending on the method. The avaliable methods are:
            'first+middle+last' : Grab the first, middle and last observations in time.
            'thirds' : Grab the middle observation in the first third, second third, and final third.
            'combinations' : Return the observation IDs corresponding to every possible combination of three observations with
                non-coinciding observation times.
    iterate : bool, optional
        Accepted for compatibility; the Rust decision loop evaluates every
        candidate the same way and ignores it.
    light_time : bool, optional
        Correct preliminary orbit for light travel time.
    propagator : Type[Propagator], optional
        Which propagator to use for ephemeris generation.
    propagator_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        propagator.
    observatory_bias_model : `~adam_core.orbit_determination.ObservationUncertaintyModel`, optional
        Observation uncertainty model applied to the observations before fitting
        (e.g. inflating per-station sigmas from an observatory bias table). Default
        None leaves the observations unchanged. The model is applied once at this
        entry point and is not forwarded to nested calls.

    Returns
    -------
    iod_orbits : `FittedOrbits`
        Dataframe with orbits found in linkages.
            "orbit_id" : Orbit ID, a uuid [str],
            "epoch" : Epoch at which orbit is defined in MJD TDB [float],
            "x" : Orbit's ecliptic J2000 x-position in au [float],
            "y" : Orbit's ecliptic J2000 y-position in au [float],
            "z" : Orbit's ecliptic J2000 z-position in au [float],
            "vx" : Orbit's ecliptic J2000 x-velocity in au per day [float],
            "vy" : Orbit's ecliptic J2000 y-velocity in au per day [float],
            "vz" : Orbit's ecliptic J2000 z-velocity in au per day [float],
            "arc_length" : Arc length in days [float],
            "num_obs" : Number of observations that were within the chi2 threshold
                of the orbit.
            "chi2" : Total chi2 of the orbit calculated using the predicted location of the orbit
                on the sky compared to the consituent observations.

    iod_orbit_members : `FittedOrbitMembers`
        Dataframe of orbit members with the following columns:
            "orbit_id" : Orbit ID, a uuid [str],
            "obs_id" : Observation IDs [str], one ID per row.
            "residual_ra_arcsec" : Residual (observed - expected) equatorial J2000 Right Ascension in arcseconds [float]
            "residual_dec_arcsec" : Residual (observed - expected) equatorial J2000 Declination in arcseconds [float]
            "chi2" : Observation's chi2 [float]
            "gauss_sol" : Flag to indicate which observations were used to calculate the Gauss soluton [int]
            "outlier" : Flag to indicate which observations are potential outliers (their chi2 is higher than
                the chi2 threshold) [float]
    """
    # Apply the observatory bias model (if any) before fitting: inflates the
    # observation uncertainties once at this entry point.
    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)

    # Initialize the propagator
    prop = propagator(**propagator_kwargs)

    # One native crossing owns the complete preliminary-IOD decision loop on
    # providers exposing the fused work unit. UUID creation remains Python-
    # owned; the native batch uses it as the synthetic linkage/final orbit id.
    if hasattr(prop, "initial_orbit_determination"):
        orbit_id = uuid.uuid4().hex
        return _native_initial_orbit_determination(
            prop,
            observations,
            [orbit_id],
            [orbit_id] * len(observations),
            observations.id.to_pylist(),
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            rchi2_threshold=rchi2_threshold,
            observation_selection_method=observation_selection_method,
            light_time=light_time,
            chunk_size=1,
        )

    # Every other propagator runs the same Rust decision loop through the
    # callback route, driving its own ``generate_ephemeris``.
    from adam_core import _rust_native

    from .differential_correction import _native_problem

    orbit_id = uuid.uuid4().hex
    if len(observations) == 0:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    _, observed_ipc, observers_ipc = _native_problem(
        _placeholder_orbit(observations), observations
    )
    output = _rust_native.iod_fit_ipc(
        prop,
        observed_ipc,
        observers_ipc,
        iod_settings={
            "min_obs": min_obs,
            "min_arc_length": min_arc_length,
            "contamination_percentage": contamination_percentage,
            "rchi2_threshold": rchi2_threshold,
            "observation_selection_method": observation_selection_method,
            "light_time": light_time,
            "mu": MU,
            "speed_of_light": C,
        },
    )
    return tables_from_iod_output(orbit_id, observations, output)


def _placeholder_orbit(observations: OrbitDeterminationObservations) -> Orbits:
    """A one-row orbit table the IPC bridge needs (IOD takes no seed)."""
    return Orbits.from_kwargs(
        orbit_id=["iod"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017],
            vz=[0.0],
            time=observations.coordinates.time[0:1],
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def tables_from_iod_output(
    orbit_id: str,
    observations: OrbitDeterminationObservations,
    output: dict,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Wrap the dict of a Rust IOD decision (`iod_fit_ipc`, or the ``iod`` entry
    of a fused ``full_od`` output) into `FittedOrbits` / `FittedOrbitMembers`
    in the observations' order; empty tables when no orbit was accepted.
    """
    if not output["found"]:
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    state = np.asarray(output["state"], dtype=np.float64)
    orbit = FittedOrbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[None],
        coordinates=CartesianCoordinates.from_kwargs(
            x=state[0:1],
            y=state[1:2],
            z=state[2:3],
            vx=state[3:4],
            vy=state[4:5],
            vz=state[5:6],
            time=Timestamp.from_mjd(
                [float(output["epoch_mjd"])], scale=output["epoch_scale"]
            ),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
        arc_length=[output["arc_length"]],
        num_obs=[output["num_obs"]],
        chi2=[output["chi2"]],
        reduced_chi2=[output["reduced_chi2"]],
    )
    residuals = Residuals.from_kwargs(
        values=np.asarray(output["residual_values"], dtype=np.float64).tolist(),
        chi2=output["residual_chi2"],
        dof=output["residual_dof"],
        probability=output["residual_probability"],
    )
    members = FittedOrbitMembers.from_kwargs(
        orbit_id=[orbit_id] * len(observations),
        obs_id=observations.id,
        residuals=residuals,
        solution=output["solution"],
        outlier=output["outlier"],
    )
    return orbit, members


def initial_orbit_determination(
    observations: OrbitDeterminationObservations,
    linkage_members: FittedOrbitMembers,
    propagator: Type[Propagator],
    min_obs: int = 6,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20.0,
    rchi2_threshold: float = 10**3,
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations",
    iterate: bool = False,
    light_time: bool = True,
    linkage_id_col: str = "cluster_id",
    propagator_kwargs: dict = {},
    chunk_size: int = 1,
    max_processes: Optional[int] = 1,
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Run initial orbit determination on linkages found in observations.

    If `observatory_bias_model` is provided it is applied to the observations
    once at this entry point (e.g. inflating per-station sigmas from an
    observatory bias table) before any orbits are fit; the default None
    leaves the observations unchanged.
    """
    time_start = time.perf_counter()
    logger.info("Running initial orbit determination...")

    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)

    # The supported native provider owns the entire multi-linkage workflow in
    # one crossing; chunk_size is consumed by Rust and max_processes is kept
    # only for signature compatibility (native execution replaces Ray).
    native_prop = propagator(**propagator_kwargs)
    if hasattr(native_prop, "initial_orbit_determination"):
        if len(observations) == 0 or len(linkage_members) == 0:
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        linkage_ids = linkage_members.column(linkage_id_col).unique().to_pylist()
        return _native_initial_orbit_determination(
            native_prop,
            observations,
            linkage_ids,
            linkage_members.column(linkage_id_col).to_pylist(),
            linkage_members.obs_id.to_pylist(),
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            rchi2_threshold=rchi2_threshold,
            observation_selection_method=observation_selection_method,
            light_time=light_time,
            chunk_size=chunk_size,
        )

    iod_orbits_chunks: list[FittedOrbits] = []
    iod_orbit_members_chunks: list[FittedOrbitMembers] = []
    if len(observations) > 0 and len(linkage_members) > 0:
        # Extract linkage IDs
        linkage_ids = linkage_members.column(linkage_id_col).unique()

        # `max_processes` remains accepted for signature compatibility.
        # Native providers returned above; non-native providers use the
        # deterministic serial fallback instead of Python/Ray distribution.
        del max_processes
        for linkage_id_chunk in _iterate_chunks(linkage_ids, chunk_size):
            iod_orbits_chunk, iod_orbit_members_chunk = iod_worker(
                linkage_id_chunk,
                observations,
                linkage_members,
                min_obs=min_obs,
                min_arc_length=min_arc_length,
                contamination_percentage=contamination_percentage,
                rchi2_threshold=rchi2_threshold,
                observation_selection_method=observation_selection_method,
                iterate=iterate,
                light_time=light_time,
                linkage_id_col=linkage_id_col,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
            )
            iod_orbits_chunks.append(iod_orbits_chunk)
            iod_orbit_members_chunks.append(iod_orbit_members_chunk)

        iod_orbits = (
            qv.concatenate(iod_orbits_chunks)
            if iod_orbits_chunks
            else FittedOrbits.empty()
        )
        iod_orbit_members = (
            qv.concatenate(iod_orbit_members_chunks)
            if iod_orbit_members_chunks
            else FittedOrbitMembers.empty()
        )

        time_start_drop = time.time()
        logger.info("Removing duplicate initial orbits...")
        num_orbits = len(iod_orbits)
        iod_orbits, iod_orbit_members = drop_duplicate_orbits(
            iod_orbits,
            iod_orbit_members,
            subset=[
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.x",
                "coordinates.y",
                "coordinates.z",
                "coordinates.vx",
                "coordinates.vy",
                "coordinates.vz",
            ],
            keep="first",
        )
        time_end_drop = time.time()
        logger.info(f"Removed {num_orbits - len(iod_orbits)} duplicate clusters.")
        time_end_drop = time.time()
        logger.info(
            f"Inital orbit deduplication completed in {time_end_drop - time_start_drop:.3f} seconds."
        )

        # Sort initial orbits by orbit ID and observation time
        iod_orbits, iod_orbit_members = sort_by_id_and_time(
            iod_orbits, iod_orbit_members, observations, "orbit_id"
        )

    else:
        iod_orbits = FittedOrbits.empty()
        iod_orbit_members = FittedOrbitMembers.empty()

    time_end = time.perf_counter()
    logger.info(f"Found {len(iod_orbits)} initial orbits.")
    logger.info(
        f"Initial orbit determination completed in {time_end - time_start:.3f} seconds."
    )

    return iod_orbits, iod_orbit_members
