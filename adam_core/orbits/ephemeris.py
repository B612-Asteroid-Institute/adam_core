import warnings

import pyarrow as pa
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.spherical import SphericalCoordinates
from ..observers.observers import Observers


class Ephemeris(qv.Table):

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = SphericalCoordinates.as_column()

    # The coordinates as observed by the observer will be the result of
    # light emitted or reflected from the object at the time of the observation.
    # Light, however, has a finite speed and so the object's observed cooordinates
    # will be different from its actual geometric coordinates at the time of observation.
    # Aberrated coordinates are coordinates that account for the light travel time
    # from the time of emission/reflection to the time of observation
    alpha = qv.Float64Column(nullable=True)
    light_time = qv.Float64Column(nullable=True)
    aberrated_coordinates = CartesianCoordinates.as_column(nullable=True)

    def link_to_observers(
        self, observers: Observers, precision="ns"
    ) -> qv.MultiKeyLinkage["Ephemeris", Observers]:
        """
        Link these ephemerides back to the observers that generated them. This is useful if
        you want or need to use the observer's position as part of any computation for
        any given set of ephemerides.

        Not all propagators will return ephemerides exactly at the time of the input observers.
        As an example, PYOORB stores times as a single MJD, when converting from two integers to
        this singular float there will be a loss of precision. To mitigate this, the user may
        optionally define the precision to which would like to link back to observers. Times
        for both the ephemerides and observers will be rounded to this precision before linking.

        Parameters
        ----------
        observers : `~adam_core.observers.observers.Observers` (N)
            Observers that generated the ephemerides.
        precision : str, optional
            Precision to which to link back to observers, by default "ns".

        Returns
        -------
        `~qv.MultiKeyLinkage[
                `~adam_core.orbits.ephemeris.Ephemeris`,
                `~adam_core.observers.observersObservers
        ]`
            Linkage between ephemerides and observers.
        """
        if self.coordinates.time.scale != observers.coordinates.time.scale:
            observers = observers.set_column(
                "coordinates.time",
                observers.coordinates.time.rescale(self.coordinates.time.scale),
            )

        rounded = self.coordinates.time.rounded(precision)
        observers_rounded = observers.coordinates.time.rounded(precision)

        left_keys = {
            "days": rounded.days,
            "nanos": rounded.nanos,
            "observatory_code": self.coordinates.origin.code,
        }
        right_keys = {
            "days": observers_rounded.days,
            "nanos": observers_rounded.nanos,
            "observatory_code": observers.code,
        }
        linkage = qv.MultiKeyLinkage(self, observers, left_keys, right_keys)

        # Check to make sure we have the correct number of linkages by calculating the number
        # of unique observers in the table and checking to see if that matches the number of
        # unique keys in the linkage
        observers_table = pa.table(
            [
                observers.code,
                observers.coordinates.time.days,
                observers.coordinates.time.nanos,
            ],
            names=["observatory_code", "days", "nanos"],
        )
        unique_observers = observers_table.group_by(
            ["observatory_code", "days", "nanos"]
        ).aggregate([])

        expected_length = len(unique_observers)
        actual_length = len(linkage.all_unique_values)
        if expected_length != actual_length:
            warnings.warn(
                "The number of unique keys in the linkage does not match the number"
                " of unique observers. Linkage precision may be too low."
                f"Expected {expected_length} unique keys, got {actual_length}.",
                UserWarning,
            )
        return linkage
