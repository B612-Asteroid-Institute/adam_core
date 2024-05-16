import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from astroquery.jplsbdb import SBDB

from ..sbdb import NotFoundError, _convert_SBDB_covariances, query_sbdb


def test__convert_SBDB_covariances():
    sbdb_format = np.array(
        [
            ["e_e", "e_q", "e_tp", "e_raan", "e_ap", "e_i"],
            ["q_e", "q_q", "q_tp", "q_raan", "q_ap", "q_i"],
            ["tp_e", "tp_q", "tp_tp", "tp_raan", "tp_ap", "tp_i"],
            ["raan_e", "raan_q", "raan_tp", "raan_raan", "raan_ap", "raan_i"],
            ["ap_e", "ap_q", "ap_tp", "ap_raan", "ap_ap", "ap_i"],
            ["i_e", "i_q", "i_tp", "i_raan", "i_ap", "i_i"],
        ]
    )
    adam_core_format = np.array(
        [
            ["q_q", "q_e", "q_i", "q_raan", "q_ap", "q_tp"],
            ["e_q", "e_e", "e_i", "e_raan", "e_ap", "e_tp"],
            ["i_q", "i_e", "i_i", "i_raan", "i_ap", "i_tp"],
            ["raan_q", "raan_e", "raan_i", "raan_raan", "raan_ap", "raan_tp"],
            ["ap_q", "ap_e", "ap_i", "ap_raan", "ap_ap", "ap_tp"],
            ["tp_q", "tp_e", "tp_i", "tp_raan", "tp_ap", "tp_tp"],
        ]
    )
    sbdb_format = np.array([sbdb_format])
    adam_core_format = np.array([adam_core_format])

    cometary_covariances = _convert_SBDB_covariances(sbdb_format)
    # Some of the symmetric terms will be in reverse format: for example
    # q_e will be e_q where it should be q_e. So for places where
    # the order is reversed, let's try to flip them around and then lets
    # check for equality with the expected adam format.

    flip_mask = np.where(cometary_covariances != adam_core_format)
    for i, j, k in zip(*flip_mask):
        cometary_covariances[i, j, k] = "_".join(
            cometary_covariances[i, j, k].split("_")[::-1]
        )

    npt.assert_equal(cometary_covariances, adam_core_format)


def test_query_sbdb_for_ceres():
    with mock_sbdb_query("Ceres.json") as mock:
        result = query_sbdb(["Ceres"])
        mock.assert_called_once()

    assert len(result) == 1


def test_query_sbdb_for_2001vb():
    with mock_sbdb_query("2001VB.json") as mock:
        result = query_sbdb(["2001VB"])
        mock.assert_called_once()

    assert len(result) == 1


def test_query_sbdb_for_54509():
    # 54509 has a 7x7 covariance matrix so lets test that we
    # correctly convert it to the 6x6 format
    with mock_sbdb_query("54509.json") as mock:
        assert mock.return_value["orbit"]["covariance"]["data"].shape == (7, 7)
        assert mock.return_value["orbit"]["covariance"]["labels"] == [
            "e",
            "q",
            "tp",
            "node",
            "peri",
            "i",
            "A2",
        ]
        result = query_sbdb(["54509"])
        mock.assert_called_once()

    assert len(result) == 1
    assert result.coordinates.covariance.to_matrix().shape == (1, 6, 6)
    assert not np.isnan(result.coordinates.covariance.to_matrix()).all()


def test_query_sbdb_for_missing_value():
    with pytest.raises(NotFoundError):
        with mock_sbdb_query("missing.json"):
            query_sbdb(["missing"])


@contextmanager
def mock_sbdb_query(response_file: str):
    with patch("adam_core.orbits.query.sbdb.SBDB.query") as mock_sbdb_query:
        resp_path = os.path.join(
            os.path.dirname(__file__), "testdata", "sbdb", response_file
        )
        response_dict = json.load(open(resp_path, "r"))
        response_value = SBDB()._process_data(OrderedDict(response_dict))
        mock_sbdb_query.return_value = response_value
        yield mock_sbdb_query
