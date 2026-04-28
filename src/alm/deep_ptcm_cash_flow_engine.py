"""
Cash flow projection engine for ALM using the Deep Promotion Time Cure Model.

The Deep-PTCM is a static competing-risks model: theta_k(x) is determined at
origination from the loan's static features.  This engine extracts discrete
cause-specific hazards from the model's CIF and overall-survival outputs and
feeds them into the same monthly amortisation / cash-flow recursion as
``RSFCashFlowEngine`` (notebook 15), so the two can be compared directly.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd

from src.alm.rsf_cash_flow_engine import RSFCashFlowEngine, RSFCashFlowConfig


@dataclass
class DeepPTCMCashFlowConfig(RSFCashFlowConfig):
    """Configuration for Deep-PTCM cash-flow projection.

    Inherits all fields from :class:`RSFCashFlowConfig`.
    """
    pass


class DeepPTCMCashFlowEngine(RSFCashFlowEngine):
    """
    Cash-flow projection driven by a Deep-PTCM competing-risks model.

    The amortisation schedule, survival recursion and aggregation are inherited
    from :class:`RSFCashFlowEngine`; only hazard extraction is overridden.

    Parameters
    ----------
    ptcm_model : CompetingRisksDeepPTCM
        Fitted model with ``event_types_`` containing ``prepay_event`` and
        ``default_event``.
    prepay_event, default_event : int
        Event codes the model was trained with.
    config : DeepPTCMCashFlowConfig, optional
    """

    def __init__(
        self,
        ptcm_model,
        prepay_event: int = 1,
        default_event: int = 2,
        config: Optional[DeepPTCMCashFlowConfig] = None,
    ):
        # Bypass RSFCashFlowEngine.__init__ -- there are no RSF objects to attach.
        self.ptcm = ptcm_model
        self.prepay_event = prepay_event
        self.default_event = default_event
        self.config = config or DeepPTCMCashFlowConfig()

    def predict_hazards(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        current_age: np.ndarray,
        T: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Discrete cause-specific hazards from the PTCM CIF / S grids.

        For loan i at projection month t (1-indexed in the recursion):

            s_prev = current_age[i] + t
            s_curr = current_age[i] + t + 1
            h_k    = max(CIF_k(s_curr) - CIF_k(s_prev), 0) / max(S(s_prev), eps)

        S here is the *overall* survival from the PTCM (not a cause-specific
        survival), because the PTCM is a joint model and S(t) governs the
        risk-set size for both causes.

        Returns
        -------
        h_prepay, h_default : np.ndarray, each shape (N, T)
        """
        N = X.shape[0] if hasattr(X, 'shape') else len(X)
        current_age = np.asarray(current_age)
        max_age = int(np.max(current_age) + T + 1)

        ages = np.arange(0, max_age + 1, dtype=np.float32)

        cif_p = self.ptcm.predict_cumulative_incidence(
            X, event=self.prepay_event, times=ages,
        )
        cif_d = self.ptcm.predict_cumulative_incidence(
            X, event=self.default_event, times=ages,
        )
        S = self.ptcm.predict_survival(X, times=ages)

        ca = current_age.astype(int).reshape(-1, 1)
        t_idx = np.arange(T)[None, :]
        idx_prev = np.clip(ca + t_idx, 0, max_age)
        idx_curr = np.clip(ca + t_idx + 1, 0, max_age)

        rows = np.arange(N)[:, None]
        S_prev = np.maximum(S[rows, idx_prev], 1e-12)

        d_cif_p = np.maximum(cif_p[rows, idx_curr] - cif_p[rows, idx_prev], 0.0)
        d_cif_d = np.maximum(cif_d[rows, idx_curr] - cif_d[rows, idx_prev], 0.0)

        return d_cif_p / S_prev, d_cif_d / S_prev

    def predict_survival_curves(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        max_month: int = 360,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Marginal "no-event-yet-by-cause" curves S_k(t) := 1 - CIF_k(t).

        These are NOT cause-specific survival functions in the classical
        Kaplan-Meier sense; they are the marginal probabilities of not yet
        having experienced cause k by month t.  Provided for API parity with
        :meth:`RSFCashFlowEngine.predict_survival_curves`.
        """
        ages = np.arange(0, max_month + 1, dtype=np.float32)
        cif_p = self.ptcm.predict_cumulative_incidence(
            X, event=self.prepay_event, times=ages,
        )
        cif_d = self.ptcm.predict_cumulative_incidence(
            X, event=self.default_event, times=ages,
        )
        return 1.0 - cif_p, 1.0 - cif_d
