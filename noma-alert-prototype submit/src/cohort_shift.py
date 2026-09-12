from __future__ import annotations

import pandas as pd

from .bias_analysis import SPECS
from .evaluation import metric_table


def cohort_shift_table(scored: pd.DataFrame) -> pd.DataFrame:
    heldout_a = scored[(scored["cohort"] == "A") & (scored["split"] == "test")]
    shifted_b = scored[scored["cohort"] == "B"]
    rows = []
    for label, df in [("Held-out Cohort A: post-discharge", heldout_a), ("Cohort B: nursing-home / LTC shift", shifted_b)]:
        mt = metric_table(df, SPECS)
        mt.insert(0, "evaluation_population", label)
        rows.append(mt)
    return pd.concat(rows, ignore_index=True)
