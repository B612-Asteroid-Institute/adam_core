from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DynamicsNumericalError(RuntimeError):
    """
    Raised when a dynamics computation produces numerically invalid output.
    """

    stage: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        details = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        if details:
            return f"DynamicsNumericalError(stage={self.stage!r}, reason={self.reason!r}, {details})"
        return f"DynamicsNumericalError(stage={self.stage!r}, reason={self.reason!r})"
