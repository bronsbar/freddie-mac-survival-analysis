"""
Cash flow projection engine for ALM using cause-specific DeepHit models.

Projects monthly mortgage cash flows using cause-specific pycox DeepHit
(MLPVanilla) models, accounting for prepayment and default as competing risks.

Similar to the RSF engine: extracts cause-specific hazards from predicted
survival curves, then combines them under the competing risks framework.
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepHitCashFlowConfig:
    """Configuration for DeepHit-based cash flow projection."""
    lgd: float = 0.25
    projection_horizon: int = 360
    batch_size: int = 2_000
    max_hazard_total: float = 0.999


class DeepHitCashFlowEngine:
    """
    Vectorized cash flow projection engine using cause-specific DeepHit models.

    Each DeepHit model (pycox MLPVanilla) outputs logits over discrete time bins.
    After softmax, the PMF is converted to a survival curve on a monthly grid,
    from which cause-specific hazards are extracted.
    """

    def __init__(
        self,
        net_prepay: torch.nn.Module,
        net_default: torch.nn.Module,
        scaler,
        labtrans_cuts: np.ndarray,
        config: Optional[DeepHitCashFlowConfig] = None,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        net_prepay : torch.nn.Module
            Fitted pycox MLPVanilla for prepayment.
        net_default : torch.nn.Module
            Fitted pycox MLPVanilla for default.
        scaler : StandardScaler
            Feature scaler fitted on training data.
        labtrans_cuts : np.ndarray
            Time bin cut points from LabTransDiscreteTime.cuts.
        config : DeepHitCashFlowConfig, optional
            Projection configuration.
        device : str
            Torch device.
        """
        self.net_prepay = net_prepay.eval()
        self.net_default = net_default.eval()
        self.scaler = scaler
        self.cuts = labtrans_cuts.astype(np.float64)
        self.config = config or DeepHitCashFlowConfig()
        self.device = device
        self.domain_max = int(self.cuts[-1])

    def _predict_monthly_survival(
        self, net: torch.nn.Module, X_scaled: np.ndarray, max_month: int
    ) -> np.ndarray:
        """
        Predict cause-specific survival curve on a monthly grid.

        Parameters
        ----------
        net : torch.nn.Module
            The pycox MLPVanilla network.
        X_scaled : np.ndarray, shape (N, F)
            Scaled feature matrix.
        max_month : int
            Maximum month to evaluate.

        Returns
        -------
        S : np.ndarray, shape (N, max_month+1)
            Survival at months 0, 1, ..., max_month.
        """
        with torch.no_grad():
            x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = net(x_t).cpu().numpy()  # (N, n_bins)

        # Softmax → PMF → CIF at cut points
        logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        exp_l = np.exp(logits)
        pmf = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N, n_bins)
        cif_at_cuts = np.cumsum(pmf, axis=1)  # (N, n_bins)

        # Interpolate CIF to monthly grid
        months = np.arange(max_month + 1, dtype=np.float64)
        N = X_scaled.shape[0]
        cif_monthly = np.zeros((N, max_month + 1), dtype=np.float64)
        for i in range(N):
            cif_monthly[i] = np.interp(months, self.cuts, cif_at_cuts[i], left=0.0)

        # Survival = 1 - CIF
        S = np.maximum(1.0 - cif_monthly, 0.0)
        return S

    def predict_hazards(
        self,
        X: np.ndarray,
        current_age: np.ndarray,
        T: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cause-specific discrete hazards from DeepHit survival curves.

        For each loan, the discrete hazard is:
            h_k(t) = 1 - S_k(t) / S_k(t-1)

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Raw (unscaled) feature matrix.
        current_age : np.ndarray, shape (N,)
            Current loan age in months.
        T : int
            Number of projection months.

        Returns
        -------
        h_prepay, h_default : np.ndarray, each shape (N, T)
        """
        N = len(X)
        X_scaled = self.scaler.transform(X)

        max_age = min(int(current_age.max()) + T + 1, self.domain_max)

        S_p = self._predict_monthly_survival(self.net_prepay, X_scaled, max_age)
        S_d = self._predict_monthly_survival(self.net_default, X_scaled, max_age)

        # Build age index arrays
        t_arr = np.arange(T)[None, :]
        age_curr = current_age[:, None].astype(int) + t_arr + 1  # (N, T)
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
            Raw (unscaled) feature matrix.

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

        # --- 2. CAUSE-SPECIFIC HAZARDS FROM DEEPHIT ---
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
