import numpy as np
from astropy.time import Time

from ..cartesian import CartesianCoordinates
from ..covariances import CoordinateCovariances
from ..origin import Origin
from ..times import Times

coords_ec = CartesianCoordinates.from_kwargs(
    time=Times.from_astropy(
        Time([59000.0, 59001.0, 59002.0], format="mjd", scale="tdb")
    ),
    x=[1, 2, 3],
    y=[4, 5, 6],
    z=[7, 8, 9],
    vx=[0.1, 0.2, 0.3],
    vy=[0.4, 0.5, 0.6],
    vz=[0.7, 0.8, 0.9],
    covariance=CoordinateCovariances.from_sigmas(
        np.array(
            [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            ]
        ),
    ),
    frame="ecliptic",
    origin=Origin.from_kwargs(code=["SUN" for i in range(3)]),
)

coords_eq = CartesianCoordinates.from_kwargs(
    time=coords_ec.time,
    x=coords_ec.x,
    y=coords_ec.y,
    z=coords_ec.z,
    vx=coords_ec.vx,
    vy=coords_ec.vy,
    vz=coords_ec.vz,
    covariance=coords_ec.covariance,
    frame="equatorial",
    origin=coords_ec.origin,
)


def test_coords_to_dataframe():
    # Cartesian coordinates defined in the ecliptic frame
    df = coords_ec.to_dataframe()
    np.testing.assert_equal(df["x_ec"].values, coords_ec.x.to_numpy())
    np.testing.assert_equal(df["y_ec"].values, coords_ec.y.to_numpy())
    np.testing.assert_equal(df["z_ec"].values, coords_ec.z.to_numpy())
    np.testing.assert_equal(df["vx_ec"].values, coords_ec.vx.to_numpy())
    np.testing.assert_equal(df["vy_ec"].values, coords_ec.vy.to_numpy())
    np.testing.assert_equal(df["vz_ec"].values, coords_ec.vz.to_numpy())
    np.testing.assert_equal(
        df["cov_x_x"].values, coords_ec.covariances.sigmas[:, 0] ** 2
    )
    np.testing.assert_equal(
        df["cov_y_y"].values, coords_ec.covariances.sigmas[:, 1] ** 2
    )
    np.testing.assert_equal(
        df["cov_z_z"].values, coords_ec.covariances.sigmas[:, 2] ** 2
    )
    np.testing.assert_equal(
        df["cov_vx_vx"].values, coords_ec.covariances.sigmas[:, 3] ** 2
    )
    np.testing.assert_equal(
        df["cov_vy_vy"].values, coords_ec.covariances.sigmas[:, 4] ** 2
    )
    np.testing.assert_equal(
        df["cov_vz_vz"].values, coords_ec.covariances.sigmas[:, 5] ** 2
    )

    # Cartesian coordinates defined in the equatorial frame
    df = coords_eq.to_dataframe()
    np.testing.assert_equal(df["x_eq"].values, coords_eq.x.to_numpy())
    np.testing.assert_equal(df["y_eq"].values, coords_eq.y.to_numpy())
    np.testing.assert_equal(df["z_eq"].values, coords_eq.z.to_numpy())
    np.testing.assert_equal(df["vx_eq"].values, coords_eq.vx.to_numpy())
    np.testing.assert_equal(df["vy_eq"].values, coords_eq.vy.to_numpy())
    np.testing.assert_equal(df["vz_eq"].values, coords_eq.vz.to_numpy())
    np.testing.assert_equal(
        df["cov_x_x"].values, coords_eq.covariances.sigmas[:, 0] ** 2
    )
    np.testing.assert_equal(
        df["cov_y_y"].values, coords_eq.covariances.sigmas[:, 1] ** 2
    )
    np.testing.assert_equal(
        df["cov_z_z"].values, coords_eq.covariances.sigmas[:, 2] ** 2
    )
    np.testing.assert_equal(
        df["cov_vx_vx"].values, coords_eq.covariances.sigmas[:, 3] ** 2
    )
    np.testing.assert_equal(
        df["cov_vy_vy"].values, coords_eq.covariances.sigmas[:, 4] ** 2
    )
    np.testing.assert_equal(
        df["cov_vz_vz"].values, coords_eq.covariances.sigmas[:, 5] ** 2
    )


def test_coords_from_dataframe():
    # Cartesian coordinates defined in the ecliptic frame
    df = coords_ec.to_dataframe()
    coords_ec2 = CartesianCoordinates.from_dataframe(df)
    assert coords_ec2.frame == coords_ec.frame

    # Cartesian coordinates defined in the equatorial frame
    df = coords_eq.to_dataframe()
    coords_eq2 = CartesianCoordinates.from_dataframe(df)
    assert coords_eq2.frame == coords_eq.frame
