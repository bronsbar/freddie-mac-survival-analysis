"""
Cash flow projection engine for ALM using Random Survival Forest models.

Projects monthly mortgage cash flows using cause-specific RSF models,
accounting for prepayment and default as competing risks.

Unlike the Cox engine which uses h0(t)*exp(X*beta) with time-varying covariates,
the RSF engine extracts cause-specific hazards directly from predicted survival
functions.  RSF survival curves are predicted from snapshot feature vectors
(one per loan), so the macro environment is captured in the feature values at
prediction time rather than via month-by-month covariate updates.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class RSFCashFlowConfig:
    """Configuration for RSF-based cash flow projection."""
    lgd: float = 0.25
    projection_horizon: int = 360
    batch_size: int = 2_000
    max_hazard_total: float = 0.999


class RSFCashFlowEngine:
    """
    Vectorized cash flow projection engine using cause-specific RSF models.

    Extracts cause-specific hazards from RSF survival function predictions,
    then projects expected cash flows under competing risks.
    """

    def __init__(
        self,
        rsf_prepay,
        rsf_default,
        config: Optional[RSFCashFlowConfig] = None,
    ):
        """
        Parameters
        ----------
        rsf_prepay : RandomSurvivalForest
            Fitted scikit-survival RSF for prepayment.
        rsf_default : RandomSurvivalForest
            Fitted scikit-survival RSF for default.
        config : RSFCashFlowConfig, optional
            Projection configuration.
        """
        self.rsf_prepay = rsf_prepay
        self.rsf_default = rsf_default
        self.config = config or RSFCashFlowConfig()

    def predict_hazards(
        self,
        X: np.ndarray,
        current_age: np.ndarray,
        T: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cause-specific discrete hazards from RSF survival curves.

        For each loan, the RSF predicts a cause-specific survival function
        S_k(t|X).  The discrete hazard is:

            h_k(t) = 1 - S_k(t) / S_k(t-1)

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Feature matrix (snapshot features per loan).
        current_age : np.ndarray, shape (N,)
            Current loan age in months.
        T : int
            Number of projection months.

        Returns
        -------
        h_prepay, h_default : np.ndarray, each shape (N, T)
        """
        N = len(X)

        # Predict survival functions and evaluate on common time grid
        surv_prepay = self.rsf_prepay.predict_survival_function(X)
        surv_default = self.rsf_default.predict_survival_function(X)

        # Clamp time grid to the model's domain
        domain_max = int(min(
            surv_prepay[0].domain[1],
            surv_default[0].domain[1],
        ))
        max_age = min(int(current_age.max()) + T + 1, domain_max)
        time_grid = np.arange(max_age + 1, dtype=np.float64)

        S_p = np.array([fn(time_grid) for fn in surv_prepay])   # (N, max_age+1)
        S_d = np.array([fn(time_grid) for fn in surv_default])

        # Build age index arrays: loan age at projection month t
        t_arr = np.arange(T)[None, :]                             # (1, T)
        age_curr = current_age[:, None].astype(int) + t_arr + 1   # (N, T)
        age_prev = age_curr - 1

        age_curr = np.clip(age_curr, 0, max_age)
        age_prev = np.clip(age_prev, 0, max_age)

        row_idx = np.arange(N)[:, None]

        S_p_curr = S_p[row_idx, age_curr]
        S_p_prev = S_p[row_idx, age_prev]
        S_d_curr = S_d[row_idx, age_curr]
        S_d_prev = S_d[row_idx, age_prev]

        h_prepay = np.maximum(0.0, 1.0 - S_p_curr / np.maximum(S_p_prev, 1e-10))
        h_default = np.maximum(0.0, 1.0 - S_d_curr / np.maximum(S_d_prev, 1e-10))

        return h_prepay, h_default

    def predict_survival_curves(
        self,
        X: np.ndarray,
        max_month: int = 360,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cause-specific survival curves on a monthly grid.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Feature matrix.
        max_month : int
            Maximum month to evaluate.

        Returns
        -------
        S_prepay, S_default : np.ndarray, each shape (N, max_month+1)
            Survival curves evaluated at months 0..max_month.
        """
        surv_prepay = self.rsf_prepay.predict_survival_function(X)
        surv_default = self.rsf_default.predict_survival_function(X)

        # Clamp time grid to the model's domain to avoid extrapolation errors
        domain_max = min(
            surv_prepay[0].domain[1],
            surv_default[0].domain[1],
        )
        effective_max = min(max_month, int(domain_max))
        time_grid = np.arange(effective_max + 1, dtype=np.float64)

        S_p = np.array([fn(time_grid) for fn in surv_prepay])
        S_d = np.array([fn(time_grid) for fn in surv_default])

        return S_p, S_d

    def project_cash_flows(
        self,
        loans_df: pd.DataFrame,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Project cash flows for a portfolio of loans.

        Parameters
        ----------
        loans_df : pd.DataFrame
            One row per loan with: int_rate, orig_upb, orig_loan_term,
            current_loan_age.
        X : np.ndarray, shape (N, F)
            Feature matrix (snapshot features per loan).

        Returns
        -------
        dict[str, np.ndarray]
            Keys: 'interest', 'scheduled_principal', 'prepayment',
            'recovery', 'loss', 'total_cf', 'survival',
            'f_prepay', 'f_default', 'upb_schedule'.
            Each array has shape (N_loans, T_months).
        """
        N = len(loans_df)
        T = self.config.projection_horizon
        cfg = self.config

        int_rate = loans_df["int_rate"].values.astype(np.float64)
        orig_upb = loans_df["orig_upb"].values.astype(np.float64)
        term = loans_df["orig_loan_term"].values.astype(np.float64)
        current_age = loans_df["current_loan_age"].values.astype(np.float64)

        results = {
            k: np.zeros((N, T), dtype=np.float64)
            for k in [
                "interest", "scheduled_principal", "prepayment", "recovery",
                "loss", "total_cf", "survival", "f_prepay", "f_default",
                "upb_schedule",
            ]
        }

        for b_start in range(0, N, cfg.batch_size):
            b_end = min(b_start + cfg.batch_size, N)
            sl = slice(b_start, b_end)

            batch = self._project_batch(
                int_rate[sl], orig_upb[sl], term[sl], current_age[sl],
                X[b_start:b_end], T,
            )
            for key in results:
                results[key][sl] = batch[key]

        return results

    def _project_batch(
        self,
        int_rate: np.ndarray,
        orig_upb: np.ndarray,
        term: np.ndarray,
        current_age: np.ndarray,
        X_batch: np.ndarray,
        T: int,
    ) -> dict[str, np.ndarray]:
        """Project cash flows for a batch of loans."""
        n = len(int_rate)
        cfg = self.config

        # --- 1. AMORTIZATION SCHEDULE ---
        monthly_rate = int_rate / 100.0 / 12.0
        payment = np.where(
            monthly_rate > 0,
            orig_upb * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-term)),
            orig_upb / term,
        )

        factor_start = (1.0 + monthly_rate) ** current_age
        upb_start = np.where(
            monthly_rate > 0,
            orig_upb * factor_start - payment * (factor_start - 1.0) / monthly_rate,
            orig_upb - payment * current_age,
        )
        upb_start = np.maximum(upb_start, 0.0)

        upb = np.zeros((n, T), dtype=np.float64)
        interest = np.zeros((n, T), dtype=np.float64)
        sched_principal = np.zeros((n, T), dtype=np.float64)

        prev_upb = upb_start.copy()
        for t in range(T):
            total_age = current_age + t + 1
            active = total_age <= term
            int_t = prev_upb * monthly_rate * active
            prin_t = np.minimum(payment - int_t, prev_upb) * active
            prin_t = np.maximum(prin_t, 0.0)
            new_upb = np.maximum(prev_upb - prin_t, 0.0)

            interest[:, t] = int_t
            sched_principal[:, t] = prin_t
            upb[:, t] = new_upb
            prev_upb = new_upb

        # --- 2. CAUSE-SPECIFIC HAZARDS FROM RSF ---
        h_prepay, h_default = self.predict_hazards(X_batch, current_age, T)

        # Clip total hazard and re-scale proportionally
        h_sum = h_prepay + h_default
        scale = np.where(
            h_sum > cfg.max_hazard_total,
            cfg.max_hazard_total / h_sum,
            1.0,
        )
        h_prepay = h_prepay * scale
        h_default = h_default * scale

        # --- 3. SURVIVAL AND SUB-DENSITIES ---
        survival = np.zeros((n, T), dtype=np.float64)
        f_prepay = np.zeros((n, T), dtype=np.float64)
        f_default = np.zeros((n, T), dtype=np.float64)

        s_prev = np.ones(n, dtype=np.float64)
        for t in range(T):
            f_prepay[:, t] = h_prepay[:, t] * s_prev
            f_default[:, t] = h_default[:, t] * s_prev
            s_prev = s_prev * (1.0 - h_prepay[:, t] - h_default[:, t])
            survival[:, t] = s_prev

        # --- 4. EXPECTED CASH FLOWS ---
        s_start = np.ones((n, T), dtype=np.float64)
        s_start[:, 1:] = survival[:, :-1]

        exp_interest = s_start * interest
        exp_principal = s_start * sched_principal
        exp_prepay = f_prepay * upb
        exp_recovery = f_default * upb * (1.0 - cfg.lgd)
        exp_loss = f_default * upb * cfg.lgd
        total_cf = exp_interest + exp_principal + exp_prepay + exp_recovery

        return {
            "interest": exp_interest,
            "scheduled_principal": exp_principal,
            "prepayment": exp_prepay,
            "recovery": exp_recovery,
            "loss": exp_loss,
            "total_cf": total_cf,
            "survival": survival,
            "f_prepay": f_prepay,
            "f_default": f_default,
            "upb_schedule": upb,
        }

    def aggregate_portfolio(
        self,
        cf_results: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Aggregate loan-level cash flows to portfolio level.

        Parameters
        ----------
        cf_results : dict[str, np.ndarray]
            Output from project_cash_flows().

        Returns
        -------
        pd.DataFrame
            Monthly portfolio cash flows with columns for each component.
        """
        T = cf_results["total_cf"].shape[1]
        agg = pd.DataFrame({
            "month": np.arange(1, T + 1),
            "interest": cf_results["interest"].sum(axis=0),
            "scheduled_principal": cf_results["scheduled_principal"].sum(axis=0),
            "prepayment": cf_results["prepayment"].sum(axis=0),
            "recovery": cf_results["recovery"].sum(axis=0),
            "loss": cf_results["loss"].sum(axis=0),
            "total_cf": cf_results["total_cf"].sum(axis=0),
            "avg_survival": cf_results["survival"].mean(axis=0),
        })
        agg["cumulative_cf"] = agg["total_cf"].cumsum()
        agg["principal_return"] = (
            agg["scheduled_principal"] + agg["prepayment"] + agg["recovery"]
        )
        return agg
