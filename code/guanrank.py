from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class KaplanMeier:
    """Minimal Kaplan-Meier estimator used for GuanRank label construction."""

    unique_times: np.ndarray
    survival: np.ndarray

    @staticmethod
    def fit(time: np.ndarray, event: np.ndarray) -> "KaplanMeier":
        t = np.asarray(time, dtype=float)
        e = np.asarray(event, dtype=int)

        if t.ndim != 1 or e.ndim != 1 or len(t) != len(e):
            raise ValueError("time and event must be 1D arrays of equal length")

        order = np.argsort(t)
        t_sorted = t[order]
        e_sorted = e[order]

        unique_times = np.unique(t_sorted)
        survival = np.empty_like(unique_times, dtype=float)

        s = 1.0
        at_risk = len(t_sorted)
        for k, tk in enumerate(unique_times):
            mask = t_sorted == tk
            d = int(np.sum(e_sorted[mask] == 1))
            c = int(np.sum(e_sorted[mask] == 0))
            if at_risk > 0:
                s *= 1.0 - d / at_risk
            survival[k] = s
            at_risk -= d + c

        return KaplanMeier(unique_times=unique_times, survival=survival)

    def survival_at(self, t: float) -> float:
        if t < self.unique_times[0]:
            return 1.0
        idx = np.searchsorted(self.unique_times, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.unique_times) - 1))
        return float(self.survival[idx])

    def expected_time_given_survival(self, c: float) -> float:
        """Approximate E[T | T > c] via the KM curve, truncated at the largest observed time."""
        t_max = float(self.unique_times[-1])
        if c >= t_max:
            return c

        s_c = self.survival_at(c)
        if s_c <= 0.0:
            return c

        times = self.unique_times
        surv = self.survival
        start_idx = np.searchsorted(times, c, side="right")
        area = 0.0

        next_t = float(times[start_idx]) if start_idx < len(times) else t_max
        area += s_c * (next_t - c)

        for i in range(start_idx, len(times) - 1):
            area += float(surv[i]) * (float(times[i + 1]) - float(times[i]))

        expected_residual = area / s_c
        return c + expected_residual


def guanrank_labels(
    time: np.ndarray,
    event: np.ndarray,
    *,
    complete: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GuanRank-style labels and expected event times.

    The returned labels lie in [0, 1], where larger values correspond to worse prognosis
    (shorter expected time to event).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)

    km = KaplanMeier.fit(t, e)
    exp_time = t.copy()
    if complete:
        censored = e == 0
        for i in np.where(censored)[0]:
            exp_time[i] = km.expected_time_given_survival(float(t[i]))

    order = np.argsort(exp_time)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(exp_time), dtype=float)

    if len(exp_time) == 1:
        labels = np.array([0.5], dtype=float)
    else:
        labels = 1.0 - (ranks / (len(exp_time) - 1))

    return labels.astype(np.float32), exp_time.astype(np.float32)
