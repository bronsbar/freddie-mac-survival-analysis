"""
Cash-flow projection engine wrapping the Kundig-Sigrist (2025) LaGaBoost
competing-risks model.

The LaGaBoost model is a *yearly* discrete-time hazard. To slot it into the
existing monthly amortisation / survival recursion (RSFCashFlowEngine), we
convert each year's predicted P(event in year) into a constant monthly hazard
via:

    h_monthly = 1 - (1 - p_annual) ** (1/12)

Limitations
-----------
* Time-varying covariates (curr_loan_to_value, ir_spread, macros, n_months)
  are held flat at their start-of-projection values for the entire horizon
  except n_months, which is advanced by 12 between years. Project-forward
  scenarios with macro shocks would require feeding shocked covariates per
  year explicitly.
* The yearly-to-monthly conversion assumes a constant monthly hazard within
  each year, which is the standard discrete-time framing but slightly less
  realistic than the RSF/PTCM monthly-grid hazards.
"""

from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.alm.rsf_cash_flow_engine import RSFCashFlowEngine, RSFCashFlowConfig


@dataclass
class LaGaBoostCashFlowConfig(RSFCashFlowConfig):
    """Same dataclass shape as RSFCashFlowConfig."""
    pass


class LaGaBoostCashFlowEngine(RSFCashFlowEngine):
    """Cash-flow projection driven by a fitted CompetingRisksLaGaBoost model.

    Inherits :meth:`project_cash_flows`, :meth:`_project_batch` and
    :meth:`aggregate_portfolio` from :class:`RSFCashFlowEngine`; overrides
    :meth:`predict_hazards` and :meth:`predict_survival_curves` to query the
    LaGaBoost yearly model and convert annual probabilities to monthly hazards.

    Parameters
    ----------
    model : CompetingRisksLaGaBoost
        Fitted on the yearly panel (must include lat/lon, n_months, ir_spread).
    feature_cols : list[str]
        Same column ordering used at fit time.
    start_year : int
        Calendar year corresponding to the start of the projection.
    config : LaGaBoostCashFlowConfig, optional
    """

    def __init__(
        self,
        model,
        feature_cols,
        start_year: int,
        config: Optional[LaGaBoostCashFlowConfig] = None,
    ):
        # Bypass RSFCashFlowEngine.__init__: no rsf models to attach.
        self.model = model
        self.feature_cols = list(feature_cols)
        self.start_year = int(start_year)
        self.config = config or LaGaBoostCashFlowConfig()

    # ------------------------------------------------------------------
    def predict_hazards(
        self,
        X: np.ndarray,
        current_age: np.ndarray,
        T: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monthly cause-specific hazards from yearly LaGaBoost predictions.

        For each year offset y in 0, 1, ..., ceil(T/12)-1 we score the model on
        the loan covariates with `n_months` and `year` advanced by 12y, take the
        annual P(prepay), P(default), and emit the implied constant monthly
        hazards for the months falling in that year.
        """
        N = X.shape[0]
        h_p = np.zeros((N, T), dtype=np.float64)
        h_d = np.zeros((N, T), dtype=np.float64)

        # Build a DataFrame matching the model's expected schema
        df = pd.DataFrame(X, columns=self.feature_cols).copy()
        # n_months at projection start (already in df if it's a feature)
        if 'n_months' not in df.columns:
            df['n_months'] = current_age.astype(float)
        df['year'] = float(self.start_year)

        n_years = int(np.ceil(T / 12))
        for y in range(n_years):
            df_y = df.copy()
            df_y['year'] = float(self.start_year + y)
            if 'n_months' in df_y.columns:
                df_y['n_months'] = df_y['n_months'].astype(float) + 12 * y

            preds = self.model.predict_proba(df_y)
            p_p = np.clip(preds['prepay'], 1e-9, 1 - 1e-9)
            p_d = np.clip(preds['default'], 1e-9, 1 - 1e-9)

            # Convert annual probability to constant monthly hazard.
            # If P(no event in year) = (1 - h_m)^12 = 1 - p_annual, then
            # h_m = 1 - (1 - p_annual)^(1/12)
            hm_p = 1.0 - (1.0 - p_p) ** (1.0 / 12.0)
            hm_d = 1.0 - (1.0 - p_d) ** (1.0 / 12.0)

            t_start = 12 * y
            t_end = min(12 * (y + 1), T)
            n_months = t_end - t_start
            h_p[:, t_start:t_end] = np.tile(hm_p[:, None], (1, n_months))
            h_d[:, t_start:t_end] = np.tile(hm_d[:, None], (1, n_months))

        return h_p, h_d

    # ------------------------------------------------------------------
    def predict_survival_curves(
        self,
        X: np.ndarray,
        max_month: int = 360,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cause-specific marginal "no-event-yet" curves S_k(t) = ∏(1-h_k).

        Like :meth:`DeepPTCMCashFlowEngine.predict_survival_curves`, these are
        marginal probabilities, not classical cause-specific survival functions
        in the competing-risks sense.
        """
        h_p, h_d = self.predict_hazards(X, np.zeros(X.shape[0]), max_month)
        Sp = np.cumprod(1.0 - h_p, axis=1)
        Sd = np.cumprod(1.0 - h_d, axis=1)
        # Pad a leading 1.0 for month 0
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, Sp])[:, : max_month + 1], np.hstack([ones, Sd])[:, : max_month + 1]
