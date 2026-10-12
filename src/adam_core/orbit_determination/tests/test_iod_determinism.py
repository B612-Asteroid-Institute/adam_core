"""
IOD is deterministic in the caller's row order (PR #217 review round 2,
phase 1, item 4): the Rust driver sorts observations by epoch, station and
original index before searching triplets and maps every member array back
to the input order, so a shuffled copy of the observations yields the same
accepted orbit and the same per-observation flags.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from ..iod import iod
from .test_differential_correction import TwoBodyPropagator
from .test_run_od_null_covariance import two_body_arc  # noqa: F401


def flags_by_id(members) -> dict[str, tuple[bool, bool]]:
    return dict(
        zip(
            members.obs_id.to_pylist(),
            zip(members.solution.to_pylist(), members.outlier.to_pylist()),
        )
    )


@pytest.mark.parametrize("contamination_percentage", [0.0, 20.0])
def test_iod_is_invariant_to_observation_order(
    two_body_arc, contamination_percentage  # noqa: F811
) -> None:
    settings = dict(
        min_obs=6,
        rchi2_threshold=1e4,
        contamination_percentage=contamination_percentage,
    )
    fitted, members = iod(two_body_arc, TwoBodyPropagator, **settings)
    assert len(fitted) == 1
    assert members.obs_id.to_pylist() == two_body_arc.id.to_pylist()

    permutation = np.random.default_rng(123).permutation(len(two_body_arc))
    shuffled = two_body_arc.take(permutation)
    assert shuffled.id.to_pylist() != two_body_arc.id.to_pylist()
    fitted_shuffled, members_shuffled = iod(shuffled, TwoBodyPropagator, **settings)

    assert len(fitted_shuffled) == 1
    npt.assert_array_equal(
        fitted_shuffled.coordinates.values, fitted.coordinates.values
    )
    assert fitted_shuffled.coordinates.time.equals(fitted.coordinates.time)
    assert fitted_shuffled.num_obs[0].as_py() == fitted.num_obs[0].as_py()
    assert fitted_shuffled.chi2[0].as_py() == fitted.chi2[0].as_py()
    # Members come back in the caller's (shuffled) order with the same flags
    # per observation id.
    assert members_shuffled.obs_id.to_pylist() == shuffled.id.to_pylist()
    assert flags_by_id(members_shuffled) == flags_by_id(members)
    assert sum(members.solution.to_pylist()) == 3
    chi2 = members.residuals.chi2.to_numpy(zero_copy_only=False)
    chi2_shuffled = members_shuffled.residuals.chi2.to_numpy(zero_copy_only=False)
    npt.assert_array_equal(chi2_shuffled, chi2[permutation])
