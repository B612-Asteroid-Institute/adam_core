from typing import Type

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from ..utils import Indexable
from .conversions import convert_coordinates
from .coordinates import Coordinates

__all__ = ["calc_residuals"]


class Residuals(Indexable):
    def __init__(
        self,
        residuals,
        chi2,
        dof,
        mahalanobis_distance,
        probability,
        names,
    ):
        self._values = residuals
        self._chi2 = chi2
        self._dof = dof
        self._mahalanobis_distance = mahalanobis_distance
        self._probability = probability
        self._names = names

        # For each dimension in values add a class attribute
        for i, name in enumerate(self._names):
            self.__dict__[f"d{name}"] = self._values[:, i]

        Indexable.__init__(self)
        return

    def __len__(self):
        return len(self.values)

    @property
    def values(self):
        return self._values

    @property
    def chi2(self):
        return self._chi2

    @property
    def chi2_total(self):
        return self.chi2.sum(axis=1)

    @property
    def dof(self):
        return self._dof

    @property
    def mahalanobis_distance(self):
        return self._mahalanobis_distance

    @property
    def probabality(self):
        return self._probability

    @property
    def names(self):
        return self._names

    def to_df(self):
        data = {}
        for i, (k, v) in enumerate(self.names.items()):
            data[v] = self.values[:, i].filled()

        for i, (k, v) in enumerate(self.names.items()):
            data[f"{v}_chi2"] = self.chi2[:, i].filled()

        data["chi2"] = self.chi2_total
        data["dof"] = self.dof
        data["mahalanobis_distance"] = self.mahalanobis_distance
        data["probability"] = self.probabality

        return pd.DataFrame(data)


def calc_residuals(observed: Type[Coordinates], predicted: Type[Coordinates]):
    """
    Calculate the residuals beteween two sets of coordinates: one observed
    and another predicted.

    Parameters
    ----------
    observed : Coordinates
        Observed coordinates.
    predicted : Coordinates
        Predicted coordinates.

    Returns
    -------
    residuals : Residuals
        Residuals calculated between the observed and predicted coordinates (observed - predicted).
        Includes chi2, dof and mahalanobis distance if the covariance matrices for the observed
        coordinates are defined.
    """
    assert isinstance(observed, Coordinates)
    assert isinstance(predicted, Coordinates)
    assert type(observed) == type(predicted)
    assert len(observed) == len(predicted)
    if observed.times is not None and predicted.times is not None:
        np.testing.assert_equal(observed.times.tdb.mjd, predicted.times.tdb.mjd)

    if not predicted.has_units(observed.units):
        predicted = convert_coordinates(predicted, observed.units)

    residual_names = {}
    for k, v in observed.names.items():
        residual_names[k] = f"d{v}"

    # Caclulate the degrees of freedom for every coordinate
    # Number of coordinate dimensions less the number of masked quantities
    dof = observed.values.shape[1] - observed.values.mask.sum(axis=1)

    residuals = observed.values - predicted.values
    # if isinstance(observed, SphericalCoordinates):
    # residuals[:, 1] = np.where(residuals[:, 1] > 180., 360. - residuals[:, 1], residuals[:, 1])
    # residual_ra *= np.cos(np.radians(dec_pred))
    # residuals[:, 2] = np.where(residuals[:, 2] > 90., 360. - residuals[:, 2], residuals[:, 2])

    if observed.covariances is not None:
        chi2s = residuals**2 / observed.sigmas**2

        d_list = []
        p_list = []
        for i, (observed_i, predicted_i) in enumerate(zip(observed, predicted)):
            mask = observed_i.values.mask
            dof_i = dof[i]

            u = predicted_i.values[~mask].filled()
            v = observed_i.values.compressed()
            cov = observed_i.covariances.compressed().reshape(dof_i, dof_i)

            d_i = mahalanobis(u, v, np.linalg.inv(cov))
            p_i = 1 - chi2.cdf(d_i, dof_i)

            d_list.append(d_i)
            p_list.append(p_i)

        d = np.array(d_list)
        p = np.array(p_list)
    else:
        p = np.empty(len(observed))
        p.fill(np.NaN)
        d = np.empty(len(observed))
        d.fill(np.NaN)

        chi2s = np.ma.zeros(residuals.shape, dtype=float)
        chi2s.fill(np.NaN)
        chi2s.mask = residuals.mask
        chi2s.fill_value = np.NaN

    return Residuals(residuals, chi2s, dof, d, p, residual_names)
