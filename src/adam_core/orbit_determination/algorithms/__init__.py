"""Expert numerical OD kernels.

Canonical configured batch workflows are exported by the parent package; this
namespace exposes only focused expert kernels, not alternate orchestration or
provenance behavior.
"""

from .fitting import fit_least_squares, iterative_fit
from .gauss import gaussIOD
from .gibbs import calcGibbs
from .herrick_gibbs import calcHerrickGibbs
from .iod import IodCandidate, IodCandidateFailure, IodReport
from .rejection import cmc2003_fit_detailed

__all__ = [
    "IodCandidate",
    "IodCandidateFailure",
    "IodReport",
    "calcGibbs",
    "calcHerrickGibbs",
    "cmc2003_fit_detailed",
    "fit_least_squares",
    "gaussIOD",
    "iterative_fit",
]
