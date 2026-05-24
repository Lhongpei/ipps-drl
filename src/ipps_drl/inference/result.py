"""Result type returned by :class:`InferenceEngine.solve`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class InferenceResult:
    """Outcome of solving a single IPPS instance.

    Attributes:
        makespan: Best makespan found.
        schedule: ``(num_scheduled_ops, 5)`` numpy array; each row is
            ``[operation_id, machine_id, job_id, start_time, end_time]``.
            ``None`` when the caller asked to skip schedule extraction.
        method: Which inference method produced this result
            (``"greedy"`` / ``"sampling"`` / ``"mcts"``).
        wall_time_s: End-to-end wall time spent solving this instance, in seconds.
        problem: Identifier of the problem (file path, or whatever the caller passed).
        extras: Method-specific extra info — e.g. for ``sampling`` the per-rollout
            makespans, for ``mcts`` the search trajectory.
    """

    makespan: float
    schedule: Optional[np.ndarray]
    method: str
    wall_time_s: float
    problem: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        prob = f" problem={self.problem!r}" if self.problem else ""
        return (
            f"InferenceResult(method={self.method!r}, makespan={self.makespan:.2f},"
            f" wall={self.wall_time_s:.3f}s{prob})"
        )
