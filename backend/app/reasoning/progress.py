"""Non-convergence detection (Chapter 3)."""
from __future__ import annotations


class ProgressMonitor:
    """Detects non-convergence without asking the model whether it is stuck.

    Self-assessment is unreliable: a drifting agent reports optimism. Use an
    external, monotone signal instead.
    """

    def __init__(self, total_subgoals: int, patience: int = 3) -> None:
        self.total = max(total_subgoals, 1)
        self.patience = patience
        self.history: list[float] = []

    def record(self, satisfied: int, unknowns_open: int = 0) -> None:
        # Distance-to-goal in [0, 1]; unknowns weigh half a sub-goal.
        remaining = (self.total - satisfied) + 0.5 * unknowns_open
        self.history.append(max(remaining, 0.0) / self.total)

    def stalled(self) -> bool:
        """True when the last `patience` observations show no improvement."""
        if len(self.history) <= self.patience:
            return False
        window = self.history[-(self.patience + 1):]
        return all(b >= a - 1e-9 for a, b in zip(window, window[1:]))

    @property
    def distance(self) -> float:
        return self.history[-1] if self.history else 1.0
