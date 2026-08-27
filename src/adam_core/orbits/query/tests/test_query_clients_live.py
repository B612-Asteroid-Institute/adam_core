import os

import pytest

from ..neocc import query_neocc
from ..sbdb import query_sbdb_new
from ..scout import get_scout_objects, query_scout


@pytest.mark.skipif(
    os.environ.get("ADAM_CORE_LIVE_QUERY_CLIENTS") != "1",
    reason="set ADAM_CORE_LIVE_QUERY_CLIENTS=1 for external query integration gates",
)
def test_neocc_scout_sbdb_live_clients():
    neocc = query_neocc(["2024 YR4"])
    assert len(neocc) == 1

    summary = get_scout_objects()
    assert len(summary) > 0
    scout = query_scout([summary.objectName[0].as_py()])
    assert len(scout) > 0

    sbdb = query_sbdb_new(["Ceres"], max_attempts=2)
    assert len(sbdb) == 1
