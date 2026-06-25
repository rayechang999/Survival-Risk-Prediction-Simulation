from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from lifelines.utils import concordance_index as c_index


from lifelines.utils.concordance import _concordance_summary_statistics 

import numpy as np


def concordance_index(
    event_time: np.ndarray,
    predicted_score: np.ndarray,
    event_observed: np.ndarray,
    tie_out =False
) -> float:
    t = np.asarray(event_time, dtype=np.float64).reshape(-1)
    s = np.asarray(predicted_score, dtype=np.float64).reshape(-1)
    e = np.asarray(event_observed, dtype=np.int64).reshape(-1)
    if tie_out==False:
        return c_index(t, s, e)
    else:
        concordant, tied_risk, comparable =_concordance_summary_statistics(t,s,e)
        return (concordant) / (comparable-tied_risk)

