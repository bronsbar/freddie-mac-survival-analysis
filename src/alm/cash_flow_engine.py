"""
Core cash flow projection engine for ALM.

Projects monthly mortgage cash flows using cause-specific Cox hazard models,
accounting for prepayment and default as competing risks.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CashFlowConfig:
    """Configuration for cash flow projection."""
    lgd: float = 0.25                  # Loss given default
    projection_horizon: int = 360      # Max months to project
    batch_size: int = 10_000           # Loans per batch
    max_hazard_total: float = 0.999    # Cap on total hazard per month


class MortgageCashFlowEngine:
    """
    Vectorized cash flow projection engine for mortgage portfolios.

    Uses cause-specific Cox hazard models to project expected cash flows
    under a given macro scenario.
    """

    def __init__(
        self,
        h0_prepay: np.ndarray,
        h0_default: np.ndarray,
        beta_prepay: np.ndarray,
        beta_default: np.ndarray,
        config: Optional[CashFlowConfig] = None,
    ):
        """
        Parameters
        ----------
        h0_prepay : np.ndarray
            Baseline discrete hazard for prepayment, length >= max loan age.
        h0_default : np.ndarray
            Baseline discrete hazard for default, length >= max loan age.
        beta_prepay : np.ndarray
            Cox coefficients for prepayment model, shape (N_features,).
        beta_default : np.ndarray
            Cox coefficients for default model, shape (N_features,).
        config : CashFlowConfig, optional
            Projection configuration.
        """
        self.h0_prepay = h0_prepay.astype(np.float64)
        self.h0_default = h0_default.astype(np.float64)
        self.beta_prepay = beta_prepay.astype(np.float64)
        self.beta_default = beta_default.astype(np.float64)
        self.config = config or CashFlowConfig()

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
        X : np.ndarray
            Covariate matrix shape (N_loans, T_months, N_features) from
            scenario_to_covariate_matrix().

        Returns
        -------
        dict[str, np.ndarray]
            Keys: 'interest', 'scheduled_principal', 'prepayment',
            'recovery', 'loss', 'total_cf', 'survival',
            'f_prepay', 'f_default', 'upb_schedule'.
            Each array has shape (N_loans, T_months).
        """
        N = len(loans_df)
        T = min(X.shape[1], self.config.projection_horizon)
        cfg = self.config

        # Extract loan-level arrays
        int_rate = loans_df["int_rate"].values.astype(np.float64)
        orig_upb = loans_df["orig_upb"].values.astype(np.float64)
        term = loans_df["orig_loan_term"].values.astype(np.float64)
        current_age = loans_df["current_loan_age"].values.astype(np.float64)

        # Allocate output arrays
        results = {
            "interest": np.zeros((N, T), dtype=np.float64),
            "scheduled_principal": np.zeros((N, T), dtype=np.float64),
            "prepayment": np.zeros((N, T), dtype=np.float64),
            "recovery": np.zeros((N, T), dtype=np.float64),
            "loss": np.zeros((N, T), dtype=np.float64),
            "total_cf": np.zeros((N, T), dtype=np.float64),
            "survival": np.zeros((N, T), dtype=np.float64),
            "f_prepay": np.zeros((N, T), dtype=np.float64),
            "f_default": np.zeros((N, T), dtype=np.float64),
            "upb_schedule": np.zeros((N, T), dtype=np.float64),
        }

        # Process in batches
        for b_start in range(0, N, cfg.batch_size):
            b_end = min(b_start + cfg.batch_size, N)
            batch_slice = slice(b_start, b_end)
            n_batch = b_end - b_start

            batch_results = self._project_batch(
                int_rate[batch_slice],
                orig_upb[batch_slice],
                term[batch_slice],
                current_age[batch_slice],
                X[b_start:b_end, :T, :],
                T,
            )

            for key in results:
                results[key][batch_slice] = batch_results[key]

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
        monthly_rate = int_rate / 100.0 / 12.0  # (n,)
        # Monthly payment
        payment = np.where(
            monthly_rate > 0,
            orig_upb * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-term)),
            orig_upb / term,
        )

        # Scheduled UPB, interest, principal at each month
        upb = np.zeros((n, T), dtype=np.float64)
        interest = np.zeros((n, T), dtype=np.float64)
        sched_principal = np.zeros((n, T), dtype=np.float64)

        # UPB at start of projection (after current_age payments)
        factor_start = (1.0 + monthly_rate) ** current_age
        upb_start = np.where(
            monthly_rate > 0,
            orig_upb * factor_start - payment * (factor_start - 1.0) / monthly_rate,
            orig_upb - payment * current_age,
        )
        upb_start = np.maximum(upb_start, 0.0)

        prev_upb = upb_start.copy()
        for t in range(T):
            total_age = current_age + t + 1
            # Past maturity: no more payments
            active = total_age <= term
            int_t = prev_upb * monthly_rate * active
            prin_t = np.minimum(payment - int_t, prev_upb) * active
            prin_t = np.maximum(prin_t, 0.0)
            new_upb = np.maximum(prev_upb - prin_t, 0.0)

            interest[:, t] = int_t
            sched_principal[:, t] = prin_t
            upb[:, t] = new_upb
            prev_upb = new_upb

        # --- 2. CAUSE-SPECIFIC HAZARDS ---
        # Linear predictor: X . beta -> (n, T)
        lp_prepay = np.einsum("ntf,f->nt", X_batch, self.beta_prepay)
        lp_default = np.einsum("ntf,f->nt", X_batch, self.beta_default)

        # Relative risk
        rr_prepay = np.exp(np.clip(lp_prepay, -20, 20))
        rr_default = np.exp(np.clip(lp_default, -20, 20))

        # Baseline hazard at each loan's age
        h_prepay = np.zeros((n, T), dtype=np.float64)
        h_default = np.zeros((n, T), dtype=np.float64)

        for t in range(T):
            ages = (current_age + t + 1).astype(int)  # loan_age at month t
            ages_clipped = np.clip(ages - 1, 0, len(self.h0_prepay) - 1)
            h_prepay[:, t] = self.h0_prepay[ages_clipped] * rr_prepay[:, t]
            ages_clipped_d = np.clip(ages - 1, 0, len(self.h0_default) - 1)
            h_default[:, t] = self.h0_default[ages_clipped_d] * rr_default[:, t]

        # Clip total hazard
        h_total = np.minimum(h_prepay + h_default, cfg.max_hazard_total)
        # Re-scale individual hazards proportionally if clipped
        scale = np.where(
            (h_prepay + h_default) > cfg.max_hazard_total,
            cfg.max_hazard_total / (h_prepay + h_default),
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
            s_prev = s_prev * (1.0 - h_total[:, t])
            survival[:, t] = s_prev

        # --- 4. EXPECTED CASH FLOWS ---
        # Survival at start of period = survival at end of previous period
        s_start = np.ones((n, T), dtype=np.float64)
        s_start[:, 1:] = survival[:, :-1]

        exp_interest = s_start * interest
        exp_principal = s_start * sched_principal
        exp_prepay = f_prepay * upb  # full payoff of remaining UPB
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
