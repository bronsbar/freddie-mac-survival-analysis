"""
Interest rate risk metrics for ALM analysis.

NPV, modified/effective duration, modified/effective convexity, WAL.
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable


def compute_npv(
    cash_flows: np.ndarray,
    annual_rate: float,
) -> float:
    """
    Compute net present value of a monthly cash flow stream.

    Parameters
    ----------
    cash_flows : np.ndarray
        Monthly cash flows, shape (T,) for portfolio or (N, T) for loan-level.
    annual_rate : float
        Annual discount rate (e.g. 0.05 for 5%).

    Returns
    -------
    float or np.ndarray
        NPV. Scalar if input is 1D, array of shape (N,) if input is 2D.
    """
    monthly_rate = annual_rate / 12.0
    if cash_flows.ndim == 1:
        T = len(cash_flows)
        t = np.arange(1, T + 1)
        discount = (1.0 + monthly_rate) ** (-t)
        return float(np.sum(cash_flows * discount))
    else:
        T = cash_flows.shape[1]
        t = np.arange(1, T + 1)
        discount = (1.0 + monthly_rate) ** (-t)
        return np.sum(cash_flows * discount[None, :], axis=1)


def compute_modified_duration(
    cash_flows: np.ndarray,
    annual_rate: float,
    dr: float = 0.0001,
) -> float:
    """
    Compute modified duration using discount rate shock only.

    Duration = -(NPV(r+dr) - NPV(r-dr)) / (2 * dr * NPV(r))

    Parameters
    ----------
    cash_flows : np.ndarray
        Portfolio-level monthly cash flows, shape (T,).
    annual_rate : float
        Annual discount rate.
    dr : float
        Annual rate shock for finite-difference (default 1bp).

    Returns
    -------
    float
        Modified duration in years.
    """
    npv_base = compute_npv(cash_flows, annual_rate)
    npv_up = compute_npv(cash_flows, annual_rate + dr)
    npv_down = compute_npv(cash_flows, annual_rate - dr)

    if abs(npv_base) < 1e-10:
        return 0.0

    return -(npv_up - npv_down) / (2.0 * dr * npv_base)


def compute_effective_duration(
    engine,
    loans_df: pd.DataFrame,
    scenario_func: Callable,
    annual_rate: float,
    shock_bps: int = 100,
) -> float:
    """
    Compute effective duration by shocking both discount rate and macro covariates.

    This captures prepayment optionality: when rates change, borrower behavior
    (prepayment/default hazards) also changes.

    Parameters
    ----------
    engine : MortgageCashFlowEngine
        Cash flow engine.
    loans_df : pd.DataFrame
        Loan data.
    scenario_func : Callable
        Function(shock_bps: int) -> (X_shocked, MacroScenario) that returns
        the shocked covariate matrix and scenario.
    annual_rate : float
        Annual discount rate.
    shock_bps : int
        Basis points for parallel rate shock.

    Returns
    -------
    float
        Effective duration in years.
    """
    dr = shock_bps / 10000.0  # bps to decimal

    # Base case
    X_base, _ = scenario_func(0)
    cf_base = engine.project_cash_flows(loans_df, X_base)
    npv_base = compute_npv(cf_base["total_cf"].sum(axis=0), annual_rate)

    # Rate up
    X_up, _ = scenario_func(shock_bps)
    cf_up = engine.project_cash_flows(loans_df, X_up)
    npv_up = compute_npv(cf_up["total_cf"].sum(axis=0), annual_rate + dr)

    # Rate down
    X_down, _ = scenario_func(-shock_bps)
    cf_down = engine.project_cash_flows(loans_df, X_down)
    npv_down = compute_npv(cf_down["total_cf"].sum(axis=0), annual_rate - dr)

    if abs(npv_base) < 1e-10:
        return 0.0

    return -(npv_up - npv_down) / (2.0 * dr * npv_base)


def compute_modified_convexity(
    cash_flows: np.ndarray,
    annual_rate: float,
    dr: float = 0.0001,
) -> float:
    """
    Compute modified convexity using discount rate shock only.

    Convexity = (NPV(r+dr) + NPV(r-dr) - 2*NPV(r)) / (dr^2 * NPV(r))

    Parameters
    ----------
    cash_flows : np.ndarray
        Portfolio-level monthly cash flows, shape (T,).
    annual_rate : float
        Annual discount rate.
    dr : float
        Annual rate shock for finite-difference.

    Returns
    -------
    float
        Modified convexity.
    """
    npv_base = compute_npv(cash_flows, annual_rate)
    npv_up = compute_npv(cash_flows, annual_rate + dr)
    npv_down = compute_npv(cash_flows, annual_rate - dr)

    if abs(npv_base) < 1e-10:
        return 0.0

    return (npv_up + npv_down - 2.0 * npv_base) / (dr ** 2 * npv_base)


def compute_effective_convexity(
    engine,
    loans_df: pd.DataFrame,
    scenario_func: Callable,
    annual_rate: float,
    shock_bps: int = 100,
) -> float:
    """
    Compute effective convexity with shocked macro covariates.

    Parameters
    ----------
    engine : MortgageCashFlowEngine
        Cash flow engine.
    loans_df : pd.DataFrame
        Loan data.
    scenario_func : Callable
        Function(shock_bps) -> (X_shocked, MacroScenario).
    annual_rate : float
        Annual discount rate.
    shock_bps : int
        Basis points for parallel rate shock.

    Returns
    -------
    float
        Effective convexity.
    """
    dr = shock_bps / 10000.0

    X_base, _ = scenario_func(0)
    cf_base = engine.project_cash_flows(loans_df, X_base)
    npv_base = compute_npv(cf_base["total_cf"].sum(axis=0), annual_rate)

    X_up, _ = scenario_func(shock_bps)
    cf_up = engine.project_cash_flows(loans_df, X_up)
    npv_up = compute_npv(cf_up["total_cf"].sum(axis=0), annual_rate + dr)

    X_down, _ = scenario_func(-shock_bps)
    cf_down = engine.project_cash_flows(loans_df, X_down)
    npv_down = compute_npv(cf_down["total_cf"].sum(axis=0), annual_rate - dr)

    if abs(npv_base) < 1e-10:
        return 0.0

    return (npv_up + npv_down - 2.0 * npv_base) / (dr ** 2 * npv_base)


def compute_wal(
    principal_return: np.ndarray,
) -> float:
    """
    Compute weighted average life (WAL).

    WAL = sum(t * principal_return(t)) / sum(principal_return(t))

    Parameters
    ----------
    principal_return : np.ndarray
        Monthly principal returns (sched + prepay + recovery), shape (T,).

    Returns
    -------
    float
        WAL in years.
    """
    T = len(principal_return)
    t = np.arange(1, T + 1)
    total_principal = np.sum(principal_return)

    if total_principal < 1e-10:
        return 0.0

    # WAL in months, convert to years
    wal_months = np.sum(t * principal_return) / total_principal
    return float(wal_months / 12.0)


def compute_all_risk_metrics(
    portfolio_cf: pd.DataFrame,
    annual_rate: float,
    dr: float = 0.0001,
) -> dict[str, float]:
    """
    Compute all risk metrics from portfolio-level cash flows.

    Parameters
    ----------
    portfolio_cf : pd.DataFrame
        Output from MortgageCashFlowEngine.aggregate_portfolio().
    annual_rate : float
        Annual discount rate.
    dr : float
        Rate shock for finite-difference.

    Returns
    -------
    dict[str, float]
        Dictionary with NPV, modified_duration, modified_convexity, WAL.
    """
    total_cf = portfolio_cf["total_cf"].values
    principal_return = portfolio_cf["principal_return"].values

    return {
        "npv": compute_npv(total_cf, annual_rate),
        "modified_duration": compute_modified_duration(total_cf, annual_rate, dr),
        "modified_convexity": compute_modified_convexity(total_cf, annual_rate, dr),
        "wal_years": compute_wal(principal_return),
        "total_cash_flow": float(np.sum(total_cf)),
        "total_principal_return": float(np.sum(principal_return)),
        "total_interest": float(np.sum(portfolio_cf["interest"].values)),
        "total_loss": float(np.sum(portfolio_cf["loss"].values)),
    }
