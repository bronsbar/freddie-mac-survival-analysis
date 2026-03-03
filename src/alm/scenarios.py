"""
Macro scenario definitions and covariate path generation for ALM projections.

Translates economic scenarios into the 21-feature covariate matrix required
by the cause-specific Cox models, replicating the exact feature derivation
logic from notebooks/03b_create_loan_month_panel.ipynb.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MacroScenario:
    """
    Forward macro paths for ALM projection.

    Each array is length T (projection horizon in months).
    Index i corresponds to projection month i+1.
    """
    name: str
    mortgage30us: np.ndarray  # 30-year FRM rate path
    dgs10: np.ndarray         # 10-year Treasury yield path
    dgs3mo: np.ndarray        # 3-month Treasury bill rate path
    # State-level arrays: dict mapping state code -> np.ndarray of length T
    state_hpi: dict[str, np.ndarray] = field(default_factory=dict)
    state_unemployment: dict[str, np.ndarray] = field(default_factory=dict)
    national_hpi: Optional[np.ndarray] = None  # national HPI path

    @property
    def horizon(self) -> int:
        return len(self.mortgage30us)


def create_base_scenario(
    panel_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    state_hpi_df: pd.DataFrame,
    state_unemp_df: pd.DataFrame,
    horizon: int = 360,
    name: str = "base",
) -> MacroScenario:
    """
    Create base scenario by holding last observed macro values constant.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Loan-month panel (used to identify relevant states).
    macro_df : pd.DataFrame
        National macro data (fred_monthly_panel.parquet), DatetimeIndex.
    state_hpi_df : pd.DataFrame
        State HPI data (state_hpi.parquet), DatetimeIndex.
    state_unemp_df : pd.DataFrame
        State unemployment data (state_unemployment.parquet), DatetimeIndex.
    horizon : int
        Projection horizon in months.
    name : str
        Scenario name.

    Returns
    -------
    MacroScenario
    """
    # National rates: hold last observed constant
    mortgage30us = np.full(horizon, macro_df["MORTGAGE30US"].iloc[-1])
    dgs10 = np.full(horizon, macro_df["DGS10"].iloc[-1])
    dgs3mo = np.full(horizon, macro_df["DGS3MO"].iloc[-1])

    # States present in the portfolio
    states = panel_df["property_state"].unique()

    # State HPI
    state_hpi = {}
    for st in states:
        col = f"{st}_hpi"
        if col in state_hpi_df.columns:
            last_val = state_hpi_df[col].dropna().iloc[-1]
            state_hpi[st] = np.full(horizon, last_val)

    # National HPI (average across state HPIs)
    hpi_cols = [c for c in state_hpi_df.columns if c.endswith("_hpi")]
    last_national = state_hpi_df[hpi_cols].iloc[-1].mean()
    national_hpi = np.full(horizon, last_national)

    # State unemployment
    state_unemployment = {}
    for st in states:
        col = f"{st}_unemployment"
        if col in state_unemp_df.columns:
            last_val = state_unemp_df[col].dropna().iloc[-1]
            state_unemployment[st] = np.full(horizon, last_val)

    return MacroScenario(
        name=name,
        mortgage30us=mortgage30us,
        dgs10=dgs10,
        dgs3mo=dgs3mo,
        state_hpi=state_hpi,
        state_unemployment=state_unemployment,
        national_hpi=national_hpi,
    )


def apply_rate_shock(
    base: MacroScenario,
    shock_bps: int,
    name: Optional[str] = None,
) -> MacroScenario:
    """
    Apply parallel shift to all interest rates.

    Parameters
    ----------
    base : MacroScenario
        Base scenario to shock.
    shock_bps : int
        Basis points to shift (positive = rates up).
    name : str, optional
        Scenario name.

    Returns
    -------
    MacroScenario
    """
    shift = shock_bps / 100.0  # bps to percentage points
    if name is None:
        sign = "+" if shock_bps >= 0 else ""
        name = f"rate_{sign}{shock_bps}bp"

    return MacroScenario(
        name=name,
        mortgage30us=base.mortgage30us + shift,
        dgs10=base.dgs10 + shift,
        dgs3mo=base.dgs3mo + shift,
        state_hpi={k: v.copy() for k, v in base.state_hpi.items()},
        state_unemployment={k: v.copy() for k, v in base.state_unemployment.items()},
        national_hpi=base.national_hpi.copy() if base.national_hpi is not None else None,
    )


def apply_hpi_shock(
    base: MacroScenario,
    decline_pct: float = 20.0,
    decline_months: int = 24,
    name: Optional[str] = None,
) -> MacroScenario:
    """
    Apply gradual HPI decline over N months, then hold flat.

    Parameters
    ----------
    base : MacroScenario
        Base scenario.
    decline_pct : float
        Total percentage decline (e.g. 20 means -20%).
    decline_months : int
        Months over which the decline occurs.
    name : str, optional
        Scenario name.

    Returns
    -------
    MacroScenario
    """
    if name is None:
        name = f"hpi_-{decline_pct:.0f}pct_{decline_months}m"

    T = base.horizon
    # Multiplier path: linear decline then flat at trough
    multiplier = np.ones(T)
    ramp = np.linspace(1.0, 1.0 - decline_pct / 100.0, decline_months + 1)[1:]
    ramp_len = min(len(ramp), T)
    multiplier[:ramp_len] = ramp[:ramp_len]
    if ramp_len < T:
        multiplier[ramp_len:] = multiplier[ramp_len - 1]

    shocked_hpi = {}
    for st, path in base.state_hpi.items():
        shocked_hpi[st] = path * multiplier

    shocked_national = None
    if base.national_hpi is not None:
        shocked_national = base.national_hpi * multiplier

    return MacroScenario(
        name=name,
        mortgage30us=base.mortgage30us.copy(),
        dgs10=base.dgs10.copy(),
        dgs3mo=base.dgs3mo.copy(),
        state_hpi=shocked_hpi,
        state_unemployment={k: v.copy() for k, v in base.state_unemployment.items()},
        national_hpi=shocked_national,
    )


def apply_unemployment_shock(
    base: MacroScenario,
    increase_pct_pts: float = 3.0,
    ramp_months: int = 12,
    name: Optional[str] = None,
) -> MacroScenario:
    """
    Apply unemployment increase over N months, then hold flat.

    Parameters
    ----------
    base : MacroScenario
        Base scenario.
    increase_pct_pts : float
        Percentage point increase in unemployment rate.
    ramp_months : int
        Months over which increase occurs.
    name : str, optional
        Scenario name.

    Returns
    -------
    MacroScenario
    """
    if name is None:
        name = f"unemp_+{increase_pct_pts:.0f}pp_{ramp_months}m"

    T = base.horizon
    # Additive ramp
    addition = np.zeros(T)
    ramp = np.linspace(0.0, increase_pct_pts, ramp_months + 1)[1:]
    ramp_len = min(len(ramp), T)
    addition[:ramp_len] = ramp[:ramp_len]
    if ramp_len < T:
        addition[ramp_len:] = addition[ramp_len - 1]

    shocked_unemp = {}
    for st, path in base.state_unemployment.items():
        shocked_unemp[st] = path + addition

    return MacroScenario(
        name=name,
        mortgage30us=base.mortgage30us.copy(),
        dgs10=base.dgs10.copy(),
        dgs3mo=base.dgs3mo.copy(),
        state_hpi={k: v.copy() for k, v in base.state_hpi.items()},
        state_unemployment=shocked_unemp,
        national_hpi=base.national_hpi.copy() if base.national_hpi is not None else None,
    )


def _build_history(
    macro_df: pd.DataFrame,
    state_hpi_df: pd.DataFrame,
    state_unemp_df: pd.DataFrame,
    scenario: MacroScenario,
    n_history: int = 12,
) -> dict:
    """
    Build combined history + projection arrays for computing lagged features.

    Returns dict with keys for each macro series, each array of length
    (n_history + T) where the first n_history entries are historical.
    """
    T = scenario.horizon

    # National rates: last n_history historical + T projection
    mortgage_hist = macro_df["MORTGAGE30US"].dropna().values[-n_history:]
    dgs10_hist = macro_df["DGS10"].dropna().values[-n_history:]
    dgs3mo_hist = macro_df["DGS3MO"].dropna().values[-n_history:]

    mortgage_full = np.concatenate([mortgage_hist, scenario.mortgage30us])
    dgs10_full = np.concatenate([dgs10_hist, scenario.dgs10])
    dgs3mo_full = np.concatenate([dgs3mo_hist, scenario.dgs3mo])

    # State HPI
    hpi_cols = [c for c in state_hpi_df.columns if c.endswith("_hpi")]
    national_hist = state_hpi_df[hpi_cols].iloc[-n_history:].mean(axis=1).values
    national_full = np.concatenate([
        national_hist,
        scenario.national_hpi if scenario.national_hpi is not None
        else np.full(T, national_hist[-1])
    ])

    state_hpi_full = {}
    for st, path in scenario.state_hpi.items():
        col = f"{st}_hpi"
        if col in state_hpi_df.columns:
            hist = state_hpi_df[col].dropna().values[-n_history:]
            state_hpi_full[st] = np.concatenate([hist, path])

    state_unemp_full = {}
    for st, path in scenario.state_unemployment.items():
        col = f"{st}_unemployment"
        if col in state_unemp_df.columns:
            hist = state_unemp_df[col].dropna().values[-n_history:]
            state_unemp_full[st] = np.concatenate([hist, path])

    return {
        "mortgage30us": mortgage_full,
        "dgs10": dgs10_full,
        "dgs3mo": dgs3mo_full,
        "national_hpi": national_full,
        "state_hpi": state_hpi_full,
        "state_unemp": state_unemp_full,
        "n_history": n_history,
    }


def scenario_to_covariate_matrix(
    loans_df: pd.DataFrame,
    scenario: MacroScenario,
    macro_df: pd.DataFrame,
    state_hpi_df: pd.DataFrame,
    state_unemp_df: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Translate a MacroScenario + loan static attributes into the covariate
    matrix required by the Cox models.

    Replicates the exact feature derivation from 03b_create_loan_month_panel.ipynb.

    Parameters
    ----------
    loans_df : pd.DataFrame
        One row per loan with columns:
        - loan_sequence_number, property_state
        - int_rate, orig_upb, fico_score, dti_r, ltv_r
        - orig_loan_term (months)
        - current_loan_age (months since origination at projection start)
        - orig_MORTGAGE30US, orig_DGS10, orig_state_hpi
          (macro values at origination time)
    scenario : MacroScenario
        Forward macro paths.
    macro_df : pd.DataFrame
        Historical national macro data.
    state_hpi_df : pd.DataFrame
        Historical state HPI data.
    state_unemp_df : pd.DataFrame
        Historical state unemployment data.
    feature_names : list[str]
        Ordered feature names matching model.params_.index.

    Returns
    -------
    np.ndarray
        Shape (N_loans, T_months, N_features) float32 covariate matrix.
    """
    N = len(loans_df)
    T = scenario.horizon
    F = len(feature_names)

    # Build history for lagged features
    hist = _build_history(macro_df, state_hpi_df, state_unemp_df, scenario, n_history=12)
    nh = hist["n_history"]

    # Pre-extract loan-level arrays
    int_rate = loans_df["int_rate"].values.astype(np.float32)
    orig_upb = loans_df["orig_upb"].values.astype(np.float32)
    fico = loans_df["fico_score"].values.astype(np.float32)
    dti = loans_df["dti_r"].values.astype(np.float32)
    ltv = loans_df["ltv_r"].values.astype(np.float32)
    term = loans_df["orig_loan_term"].values.astype(np.float32)
    current_age = loans_df["current_loan_age"].values.astype(np.float32)
    states = loans_df["property_state"].values

    # Origination-time macro values
    orig_mortgage = loans_df["orig_MORTGAGE30US"].values.astype(np.float32)
    orig_dgs10 = loans_df["orig_DGS10"].values.astype(np.float32)
    orig_state_hpi_vals = loans_df["orig_state_hpi"].values.astype(np.float32)

    # Compute log_upb
    log_upb = np.log(orig_upb)

    # Compute amortization schedule for bal_repaid
    monthly_rate = int_rate / 100.0 / 12.0  # (N,)
    payment = orig_upb * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-term))  # (N,)
    # Handle zero-rate loans
    zero_rate_mask = monthly_rate == 0
    if zero_rate_mask.any():
        payment[zero_rate_mask] = orig_upb[zero_rate_mask] / term[zero_rate_mask]

    # Scheduled UPB at each future month
    # UPB(t) for t = current_age+1 .. current_age+T
    # We need bal_repaid = (orig_upb - UPB_sched(t)) / orig_upb * 100
    # UPB after n payments: UPB_n = orig_upb * (1+r)^n - payment * ((1+r)^n - 1) / r
    # Vectorized: compute UPB for each loan at each projection month
    upb_sched = np.zeros((N, T), dtype=np.float32)
    for t_idx in range(T):
        n = current_age + t_idx + 1  # total payments made (broadcast: N,)
        factor = (1.0 + monthly_rate) ** n
        upb_n = orig_upb * factor - payment * (factor - 1.0) / np.where(monthly_rate > 0, monthly_rate, 1.0)
        # Handle zero-rate
        if zero_rate_mask.any():
            upb_n[zero_rate_mask] = orig_upb[zero_rate_mask] - payment[zero_rate_mask] * n[zero_rate_mask] if isinstance(n, np.ndarray) else orig_upb[zero_rate_mask] - payment[zero_rate_mask] * n
        upb_sched[:, t_idx] = np.maximum(upb_n, 0.0)

    bal_repaid = (orig_upb[:, None] - upb_sched) / orig_upb[:, None] * 100.0

    # Build state lookup maps for vectorized access
    unique_states = np.unique(states)
    state_to_idx = {st: i for i, st in enumerate(unique_states)}
    loan_state_idx = np.array([state_to_idx.get(st, 0) for st in states])

    # State HPI array: (n_states, nh + T)
    state_hpi_arr = np.zeros((len(unique_states), nh + T), dtype=np.float32)
    for st, idx in state_to_idx.items():
        if st in hist["state_hpi"]:
            arr = hist["state_hpi"][st]
            state_hpi_arr[idx, :len(arr)] = arr

    # State unemployment array: (n_states, nh + T)
    state_unemp_arr = np.zeros((len(unique_states), nh + T), dtype=np.float32)
    for st, idx in state_to_idx.items():
        if st in hist["state_unemp"]:
            arr = hist["state_unemp"][st]
            state_unemp_arr[idx, :len(arr)] = arr

    # National arrays (nh + T)
    mortgage_full = hist["mortgage30us"].astype(np.float32)
    dgs10_full = hist["dgs10"].astype(np.float32)
    dgs3mo_full = hist["dgs3mo"].astype(np.float32)
    national_hpi_full = hist["national_hpi"].astype(np.float32)

    # Build feature name -> index map
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    # Allocate output
    X = np.zeros((N, T, F), dtype=np.float32)

    # Fill features for each projection month
    for t_idx in range(T):
        # Index into full history+projection arrays
        ti = nh + t_idx       # current time index
        ti_12 = ti - 12       # 12 months ago
        ti_3 = ti - 3         # 3 months ago
        loan_age = current_age + t_idx + 1  # (N,) loan age at this month

        # Static features
        if "int_rate" in feat_idx:
            X[:, t_idx, feat_idx["int_rate"]] = int_rate
        if "log_upb" in feat_idx:
            X[:, t_idx, feat_idx["log_upb"]] = log_upb
        if "orig_upb" in feat_idx:
            X[:, t_idx, feat_idx["orig_upb"]] = orig_upb
        if "fico_score" in feat_idx:
            X[:, t_idx, feat_idx["fico_score"]] = fico
        if "dti_r" in feat_idx:
            X[:, t_idx, feat_idx["dti_r"]] = dti
        if "ltv_r" in feat_idx:
            X[:, t_idx, feat_idx["ltv_r"]] = ltv

        # Behavioral: bal_repaid
        if "bal_repaid" in feat_idx:
            X[:, t_idx, feat_idx["bal_repaid"]] = bal_repaid[:, t_idx]

        # Behavioral: t_act_12m = min(12, loan_age) (assume performing)
        if "t_act_12m" in feat_idx:
            X[:, t_idx, feat_idx["t_act_12m"]] = np.minimum(12.0, loan_age)

        # Behavioral: delinquency counts = 0 (assume performing)
        if "t_del_30d_12m" in feat_idx:
            X[:, t_idx, feat_idx["t_del_30d_12m"]] = 0.0
        if "t_del_60d_12m" in feat_idx:
            X[:, t_idx, feat_idx["t_del_60d_12m"]] = 0.0

        # ppi_c_FRMA: int_rate - MORTGAGE30US(t)
        if "ppi_c_FRMA" in feat_idx:
            X[:, t_idx, feat_idx["ppi_c_FRMA"]] = int_rate - mortgage_full[ti]

        # ppi_o_FRMA: int_rate - MORTGAGE30US(origination) — static
        if "ppi_o_FRMA" in feat_idx:
            X[:, t_idx, feat_idx["ppi_o_FRMA"]] = int_rate - orig_mortgage

        # hpi_st_d_t_o: hpi_state(t) - hpi_state(origination)
        if "hpi_st_d_t_o" in feat_idx:
            hpi_t = state_hpi_arr[loan_state_idx, ti]
            X[:, t_idx, feat_idx["hpi_st_d_t_o"]] = hpi_t - orig_state_hpi_vals

        # TB10Y_d_t_o: DGS10(t) - DGS10(origination)
        if "TB10Y_d_t_o" in feat_idx:
            X[:, t_idx, feat_idx["TB10Y_d_t_o"]] = dgs10_full[ti] - orig_dgs10

        # FRMA30Y_d_t_o: MORTGAGE30US(t) - MORTGAGE30US(origination)
        if "FRMA30Y_d_t_o" in feat_idx:
            X[:, t_idx, feat_idx["FRMA30Y_d_t_o"]] = mortgage_full[ti] - orig_mortgage

        # hpi_st_log12m: log(hpi_state(t) / hpi_state(t-12))
        if "hpi_st_log12m" in feat_idx:
            hpi_t = state_hpi_arr[loan_state_idx, ti]
            hpi_t12 = state_hpi_arr[loan_state_idx, max(ti_12, 0)]
            ratio = np.where(hpi_t12 > 0, hpi_t / hpi_t12, 1.0)
            X[:, t_idx, feat_idx["hpi_st_log12m"]] = np.log(np.maximum(ratio, 1e-6))

        # hpi_r_st_us: hpi_state(t) / hpi_national(t)
        if "hpi_r_st_us" in feat_idx:
            hpi_t = state_hpi_arr[loan_state_idx, ti]
            nat_hpi_t = national_hpi_full[ti]
            X[:, t_idx, feat_idx["hpi_r_st_us"]] = np.where(
                nat_hpi_t > 0, hpi_t / nat_hpi_t, 1.0
            )

        # st_unemp_r12m: log(unemp(t) / unemp(t-12))
        if "st_unemp_r12m" in feat_idx:
            unemp_t = state_unemp_arr[loan_state_idx, ti]
            unemp_t12 = state_unemp_arr[loan_state_idx, max(ti_12, 0)]
            ratio = np.where(unemp_t12 > 0, unemp_t / unemp_t12, 1.0)
            X[:, t_idx, feat_idx["st_unemp_r12m"]] = np.log(np.maximum(ratio, 1e-6))

        # st_unemp_r3m: log(unemp(t) / unemp(t-3))
        if "st_unemp_r3m" in feat_idx:
            unemp_t = state_unemp_arr[loan_state_idx, ti]
            unemp_t3 = state_unemp_arr[loan_state_idx, max(ti_3, 0)]
            ratio = np.where(unemp_t3 > 0, unemp_t / unemp_t3, 1.0)
            X[:, t_idx, feat_idx["st_unemp_r3m"]] = np.log(np.maximum(ratio, 1e-6))

        # TB10Y_r12m: log(DGS10(t) / DGS10(t-12))
        if "TB10Y_r12m" in feat_idx:
            d10_t = dgs10_full[ti]
            d10_t12 = dgs10_full[max(ti_12, 0)]
            ratio = d10_t / d10_t12 if d10_t12 > 0 else 1.0
            X[:, t_idx, feat_idx["TB10Y_r12m"]] = np.log(max(ratio, 1e-6))

        # T10Y3MM: DGS10(t) - DGS3MO(t)
        if "T10Y3MM" in feat_idx:
            X[:, t_idx, feat_idx["T10Y3MM"]] = dgs10_full[ti] - dgs3mo_full[ti]

        # T10Y3MM_r12m: pct_change of T10Y3MM over 12 months
        if "T10Y3MM_r12m" in feat_idx:
            spread_t = dgs10_full[ti] - dgs3mo_full[ti]
            spread_t12 = dgs10_full[max(ti_12, 0)] - dgs3mo_full[max(ti_12, 0)]
            X[:, t_idx, feat_idx["T10Y3MM_r12m"]] = (
                (spread_t - spread_t12) / abs(spread_t12) if abs(spread_t12) > 1e-6 else 0.0
            )

    return X
