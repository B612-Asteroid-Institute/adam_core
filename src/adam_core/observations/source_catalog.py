import healpy as hp
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from quivr.validators import and_, ge, le

from ..coordinates.covariances import CoordinateCovariances
from ..coordinates.origin import Origin
from ..coordinates.spherical import SphericalCoordinates
from ..observers.observers import Observers
from ..time import Timestamp
from .associations import Associations
from .detections import PointSourceDetections
from .exposures import Exposures
from .photometry import Photometry


class SourceCatalog(qv.Table):

    #: A unique identifier for the source
    id = qv.LargeStringColumn()
    #: An identifier for the exposure in which the source was detected
    exposure_id = qv.LargeStringColumn(nullable=True)
    #: The time at which the source was detected. In most cases, this will be
    #: the midpoint of the exposure. For Rubin Observatory, each detection within an
    #: exposure will have a time that takes into account the motion of the shutter
    #: across the focal plane.
    time = Timestamp.as_column()
    #: Right Ascension in degrees (J2000)
    ra = qv.Float64Column(validator=and_(ge(0), le(360)))
    #: Declination in degrees (J2000)
    dec = qv.Float64Column(validator=and_(ge(-90), le(90)))
    #: 1-sigma uncertainty in Right Ascension in arcseconds
    ra_sigma = qv.Float64Column(nullable=True)
    #: 1-sigma uncertainty in Declination in arcseconds
    dec_sigma = qv.Float64Column(nullable=True)
    #: Correlation between Right Ascension and Declination (dimensionless)
    radec_corr = qv.Float64Column(nullable=True, validator=and_(ge(-1), le(1)))
    #: Magnitude of the source in AB magnitudes
    mag = qv.Float64Column(nullable=True)
    #: 1-sigma uncertainty in the magnitude of the source in AB magnitudes
    mag_sigma = qv.Float64Column(nullable=True)

    # PSF parameters
    #: Full width at half maximum of the PSF in arcseconds
    fwhm = qv.Float64Column(nullable=True, validator=ge(0))
    #: Semi-major axis of PSF ellipse in arcseconds
    a = qv.Float64Column(nullable=True, validator=ge(0))
    #: 1-sigma uncertainty in semi-major axis of PSF ellipse in arcseconds
    a_sigma = qv.Float64Column(nullable=True)
    #: Semi-minor axis of PSF ellipse in arcseconds
    b = qv.Float64Column(nullable=True, validator=ge(0))
    #: 1-sigma uncertainty in semi-minor axis of PSF ellipse in arcseconds
    b_sigma = qv.Float64Column(nullable=True)
    #: Position angle of PSF ellipse in degrees (0 at the North Celestial Pole (NCP), increasing Eastward)
    pa = qv.Float64Column(nullable=True)
    #: 1-sigma uncertainty in position angle of PSF ellipse in degrees
    pa_sigma = qv.Float64Column(nullable=True)

    # Exposure Details
    #: The MPC observatory code
    observatory_code = qv.LargeStringColumn()
    #: The filter in which the source was detected
    filter = qv.LargeStringColumn(nullable=True)
    #: The start time of the exposure (typically corresponds
    #: to the moment the shutter opens)
    exposure_start_time = Timestamp.as_column(nullable=True)
    #: The exposure duration in seconds (typically corresponds
    #: to the amount of time the focal plane is exposed to light)
    exposure_duration = qv.Float64Column(nullable=True, validator=ge(0))
    #: The FWHM assuming a Gaussian PSF in arcseconds for the exposure
    exposure_seeing = qv.Float64Column(nullable=True, validator=ge(0))
    #: The magnitude of a point-source that would be detected at 5-sigma
    exposure_depth_5sigma = qv.Float64Column(nullable=True)

    # Association Details
    #: The ID of the solar system object associated with the source
    object_id = qv.LargeStringColumn(nullable=True)

    #: ID of the source catalog
    catalog_id = qv.LargeStringColumn()

    def detections(self) -> PointSourceDetections:
        """
        Return the detections in the source catalog.

        Returns
        -------
        detections : PointSourceDetections
            The detections in the source catalog.
        """
        return PointSourceDetections.from_kwargs(
            id=self.id,
            exposure_id=self.exposure_id,
            time=self.time,
            ra=self.ra,
            ra_sigma=self.ra_sigma,
            dec=self.dec,
            dec_sigma=self.dec_sigma,
            mag=self.mag,
            mag_sigma=self.mag_sigma,
        )

    def exposures(self) -> Exposures:
        """
        Return the unique exposures in the source catalog.

        Returns
        -------
        exposures : Exposures
            The unique exposures in the source catalog.
        """
        exposures = Exposures.from_kwargs(
            id=self.exposure_id,
            start_time=self.exposure_start_time,
            duration=self.exposure_duration,
            filter=self.filter,
            observatory_code=self.observatory_code,
            seeing=self.exposure_seeing,
            depth_5sigma=self.exposure_depth_5sigma,
        )
        exposures = exposures.drop_duplicates(subset=["id"])
        return exposures

    def associations(self) -> Associations:
        """
        Return the associations in the source catalog.

        Returns
        -------
        associations : Associations
            The unique associations in the source catalog.
        """
        associations = Associations.from_kwargs(
            detection_id=self.id,
            object_id=self.object_id,
        )
        return associations

    def photometry(self) -> Photometry:
        """
        Return the photometry in the source catalog.

        Returns
        -------
        photometry : Photometry
            The photometry in the source catalog.
        """
        photometry = Photometry.from_kwargs(
            time=self.time,
            mag=self.mag,
            mag_sigma=self.mag_sigma,
            filter=self.filter,
        )
        return photometry

    def coordinates(self) -> SphericalCoordinates:
        """
        Return the astrometry in the source catalog as SphericalCoordinates.

        Returns
        -------
        coordinates : SphericalCoordinates
            The astrometry in the source catalog.
        """
        covariance_matrix = np.empty((len(self), 6, 6))
        covariance_matrix.fill(np.nan)

        # Convert uncertainties from arcseconds to degrees
        ra_sigma = self.ra_sigma.to_numpy(zero_copy_only=False) / 3600.0
        dec_sigma = self.dec_sigma.to_numpy(zero_copy_only=False) / 3600.0
        radec_corr = self.radec_corr.to_numpy(zero_copy_only=False)

        # Calculate the covariance matrix in units of degrees^2
        cov_ra = ra_sigma**2
        cov_dec = dec_sigma**2
        cov_radec = radec_corr * ra_sigma * dec_sigma

        covariance_matrix[:, 1, 1] = cov_ra
        covariance_matrix[:, 2, 2] = cov_dec
        covariance_matrix[:, 1, 2] = cov_radec
        covariance_matrix[:, 2, 1] = cov_radec

        return SphericalCoordinates.from_kwargs(
            rho=None,
            lon=self.ra,
            lat=self.dec,
            vrho=None,
            vlon=None,
            vlat=None,
            time=self.time,
            covariance=CoordinateCovariances.from_matrix(covariance_matrix),
            origin=Origin.from_kwargs(code=self.observatory_code),
            frame="equatorial",
        )

    def observers(self, exposure_midpoint: bool = False) -> Observers:
        """
        Return the observers for each detection in the source catalog.

        Parameters
        ----------
        exposure_midpoint : bool
            If True, the observer locations are calculated at the midpoint of the exposures.
            If False, the observer locations are calculated at the time of the detections.

        Returns
        -------
        observers : Observers
            The observers in the source catalog.
        """
        if exposure_midpoint:
            half_exposure = pc.cast(
                pc.round(
                    pc.multiply(pc.divide(self.exposure_duration, 2.0), 1e9),
                ),
                pa.int64(),
            )
            return Observers.from_codes(
                self.observatory_code,
                self.exposure_start_time.add_nanos(half_exposure),
            )
        else:
            return Observers.from_codes(
                self.observatory_code,
                self.time,
            )

    def healpixels(self, nside: int = 16, nest: bool = True) -> npt.NDArray[np.int64]:
        """
        Return the healpixels for the source catalog.

        Parameters
        ----------
        nside : int
            The nside parameter for the healpix grid.
        nest : bool
            If True, the healpix grid is in the nested format.
            If False, the healpix grid is in the ring format.

        Returns
        -------
        healpixels : np.ndarray
            The healpixels for the source catalog.
        """
        return hp.ang2pix(
            nside,
            self.ra.to_numpy(zero_copy_only=False),
            self.dec.to_numpy(zero_copy_only=False),
            lonlat=True,
            nest=nest,
        )
