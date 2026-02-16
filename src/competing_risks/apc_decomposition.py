"""
Bayesian APC Decomposition for Breeden & Crook (2022).

Implements the two-stage approach from the paper:
  Stage 1: Extract F(a), G(v), H(t) at the portfolio level via iterative
           backfitting with smoothing splines (GAM-style penalized smoothing,
           equivalent to an RW2 Bayesian prior on roughness).
  Stage 2: Use the extracted APC values as scalar features in the
           horizon-specific logistic regressions.

The backfitting operates on aggregated (age, vintage, caltime) cells for
efficiency (~thousands of cells rather than millions of loan-months).

Reference:
    Breeden, J.L. and Crook, J.N. (2022). "Multihorizon discrete time
    survival models." Journal of the Operational Research Society.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.interpolate import UnivariateSpline
from scipy.special import logit, expit


# ---------------------------------------------------------------------------
# Portfolio-level aggregation
# ---------------------------------------------------------------------------

def aggregate_portfolio_rates(
    panel: pd.DataFrame,
    event_code: int,
    min_cell_size: int = 10,
) -> Tuple[pd.DataFrame, Dict[int, object]]:
    """
    Aggregate loan-month panel to (loan_age, vintage_year) cells.

    Parameters
    ----------
    panel : pd.DataFrame
        Loan-month panel with columns: loan_age, vintage_year, year_month,
        event_code.
    event_code : int
        Event of interest (1=prepay, 2=default).
    min_cell_size : int
        Minimum number of at-risk observations per cell.

    Returns
    -------
    cells : pd.DataFrame
        Cell-level DataFrame with columns: loan_age, vintage_year,
        cal_time_idx, n_at_risk, n_events, rate.
    cal_time_map : dict
        Mapping from sequential integer index to year_month value.
    """
    df = panel[['loan_age', 'vintage_year', 'year_month', 'event_code']].copy()
    df['is_event'] = (df['event_code'] == event_code).astype(int)

    # Aggregate by (loan_age, vintage_year, year_month)
    cells = (
        df.groupby(['loan_age', 'vintage_year', 'year_month'])
        .agg(n_at_risk=('is_event', 'size'), n_events=('is_event', 'sum'))
        .reset_index()
    )

    # Filter small cells
    cells = cells[cells['n_at_risk'] >= min_cell_size].copy()

    # Create sequential calendar-time index
    sorted_periods = np.sort(cells['year_month'].unique())
    period_to_idx = {p: i for i, p in enumerate(sorted_periods)}
    cal_time_map = {i: p for p, i in period_to_idx.items()}

    cells['cal_time_idx'] = cells['year_month'].map(period_to_idx)
    cells['rate'] = cells['n_events'] / cells['n_at_risk']

    return cells, cal_time_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_group_mean(
    values: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted mean of values within each group level."""
    result = np.zeros_like(values, dtype=float)
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        w = weights[mask]
        w_sum = w.sum()
        if w_sum > 0:
            result[mask] = np.average(values[mask], weights=w)
    return result


def _smooth_1d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    smoothing: Optional[float] = None,
    degree: int = 3,
) -> Tuple[np.ndarray, UnivariateSpline]:
    """
    Fit a weighted smoothing spline to (x, y) data.

    Parameters
    ----------
    x : np.ndarray
        Covariate values (e.g., age, vintage index, caltime index).
    y : np.ndarray
        Response values (partial residuals).
    weights : np.ndarray
        Observation weights (typically n_at_risk).
    smoothing : float or None
        Smoothing factor for UnivariateSpline. None triggers automatic
        GCV-like selection (s = len(x), the default).
    degree : int
        Spline degree (default 3 = cubic).

    Returns
    -------
    fitted : np.ndarray
        Fitted values at each unique x (broadcast back to full array).
    spline : UnivariateSpline
        Fitted spline object for prediction.
    """
    # Aggregate to unique x values (weighted mean of y)
    unique_x = np.sort(np.unique(x))

    if len(unique_x) <= degree:
        # Too few unique values for spline; return weighted group means
        fitted = _weighted_group_mean(y, x, weights)
        # Create a simple spline from the means
        y_agg = np.array([
            np.average(y[x == ux], weights=weights[x == ux])
            for ux in unique_x
        ])
        w_agg = np.array([weights[x == ux].sum() for ux in unique_x])
        spl = UnivariateSpline(unique_x, y_agg, w=w_agg, k=1, s=0)
        return fitted, spl

    # Aggregate y and weights to unique x levels
    y_agg = np.array([
        np.average(y[x == ux], weights=weights[x == ux])
        for ux in unique_x
    ])
    w_agg = np.array([weights[x == ux].sum() for ux in unique_x])

    # Normalize weights for spline fitting
    w_norm = w_agg / w_agg.sum() * len(w_agg)

    # Fit spline
    k = min(degree, len(unique_x) - 1)
    spl = UnivariateSpline(unique_x, y_agg, w=w_norm, k=k, s=smoothing)

    # Evaluate at all original x values
    fitted = spl(x)

    return fitted, spl


# ---------------------------------------------------------------------------
# BreedenAPC class
# ---------------------------------------------------------------------------

class BreedenAPC:
    """
    Bayesian APC decomposition via iterative backfitting with smoothing splines.

    Decomposes portfolio-level event rates into:
        logit(rate) = intercept + F(age) + G(vintage) + H(caltime)

    where F, G, H are smooth functions estimated by penalized splines.
    The roughness penalty is equivalent to an RW2 Bayesian prior.

    Parameters
    ----------
    max_iter : int
        Maximum backfitting iterations.
    tol : float
        Convergence tolerance (max absolute change in F, G, H).
    smoothing_age : float or None
        Smoothing parameter for F(a). None = automatic (GCV).
    smoothing_vintage : float or None
        Smoothing parameter for G(v). None = automatic.
    smoothing_caltime : float or None
        Smoothing parameter for H(t). None = automatic.
    degree : int
        Spline degree (default 3 = cubic).
    min_cell_size : int
        Minimum observations per (age, vintage, caltime) cell.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        smoothing_age: Optional[float] = None,
        smoothing_vintage: Optional[float] = None,
        smoothing_caltime: Optional[float] = None,
        degree: int = 3,
        min_cell_size: int = 10,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing_age = smoothing_age
        self.smoothing_vintage = smoothing_vintage
        self.smoothing_caltime = smoothing_caltime
        self.degree = degree
        self.min_cell_size = min_cell_size

        # Fitted attributes (set by fit())
        self.intercept_: Optional[float] = None
        self.F_spline_: Optional[UnivariateSpline] = None
        self.G_spline_: Optional[UnivariateSpline] = None
        self.H_spline_: Optional[UnivariateSpline] = None
        self.F_values_: Optional[np.ndarray] = None
        self.G_values_: Optional[np.ndarray] = None
        self.H_values_: Optional[np.ndarray] = None
        self.cal_time_map_: Optional[Dict[int, object]] = None
        self.convergence_history_: Optional[List[float]] = None
        self.n_iterations_: Optional[int] = None
        self.cells_: Optional[pd.DataFrame] = None
        self.macro_regression_ = None

    def fit(
        self,
        panel: pd.DataFrame,
        event_code: int,
        log_fn=print,
    ) -> 'BreedenAPC':
        """
        Fit APC decomposition via iterative backfitting.

        Parameters
        ----------
        panel : pd.DataFrame
            Loan-month panel with columns: loan_age, vintage_year,
            year_month, event_code.
        event_code : int
            Event of interest (1=prepay, 2=default).
        log_fn : callable
            Logging function.

        Returns
        -------
        self
        """
        # Stage 1: Aggregate to cells
        cells, cal_time_map = aggregate_portfolio_rates(
            panel, event_code, min_cell_size=self.min_cell_size
        )
        self.cal_time_map_ = cal_time_map

        log_fn(f"  APC cells: {len(cells):,} (from {len(panel):,} loan-months)")

        # Work on logit scale
        eps = 1e-8
        rate_clipped = cells['rate'].values.clip(eps, 1 - eps)
        y = logit(rate_clipped)
        weights = cells['n_at_risk'].values.astype(float)

        age = cells['loan_age'].values.astype(float)
        vintage = cells['vintage_year'].values.astype(float)
        caltime = cells['cal_time_idx'].values.astype(float)

        # Initialize
        intercept = np.average(y, weights=weights)
        F = np.zeros(len(y))
        G = np.zeros(len(y))
        H = np.zeros(len(y))

        convergence_history = []

        for iteration in range(self.max_iter):
            F_old = F.copy()
            G_old = G.copy()
            H_old = H.copy()

            # Update F(age): smooth partial residuals y - intercept - G - H
            resid_F = y - intercept - G - H
            F, self.F_spline_ = _smooth_1d(
                age, resid_F, weights, self.smoothing_age, self.degree
            )

            # Update G(vintage): smooth partial residuals y - intercept - F - H
            resid_G = y - intercept - F - H
            G, self.G_spline_ = _smooth_1d(
                vintage, resid_G, weights, self.smoothing_vintage, self.degree
            )

            # Update H(caltime): smooth partial residuals y - intercept - F - G
            resid_H = y - intercept - F - G
            H, self.H_spline_ = _smooth_1d(
                caltime, resid_H, weights, self.smoothing_caltime, self.degree
            )

            # Zero-mean constraint on H (absorb mean into intercept)
            H_mean = np.average(H, weights=weights)
            intercept += H_mean
            H -= H_mean

            # Also zero-mean F and G for identifiability
            F_mean = np.average(F, weights=weights)
            intercept += F_mean
            F -= F_mean

            G_mean = np.average(G, weights=weights)
            intercept += G_mean
            G -= G_mean

            # Convergence check
            delta = max(
                np.max(np.abs(F - F_old)),
                np.max(np.abs(G - G_old)),
                np.max(np.abs(H - H_old)),
            )
            convergence_history.append(delta)

            if delta < self.tol:
                log_fn(f"  Converged after {iteration + 1} iterations "
                       f"(delta={delta:.2e})")
                break
        else:
            log_fn(f"  Warning: did not converge after {self.max_iter} "
                   f"iterations (delta={delta:.2e})")

        self.intercept_ = intercept
        self.F_values_ = F
        self.G_values_ = G
        self.H_values_ = H
        self.convergence_history_ = convergence_history
        self.n_iterations_ = len(convergence_history)
        self.cells_ = cells.copy()
        self.cells_['F_age'] = F
        self.cells_['G_vintage'] = G
        self.cells_['H_caltime'] = H

        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Add F_age, G_vintage, H_caltime columns to panel via spline lookup.

        Parameters
        ----------
        panel : pd.DataFrame
            Must have columns: loan_age, vintage_year, year_month.

        Returns
        -------
        pd.DataFrame
            Panel with added F_age, G_vintage, H_caltime columns.
        """
        panel = panel.copy()

        # F(age)
        ages = panel['loan_age'].values.astype(float)
        panel['F_age'] = self.F_spline_(ages)

        # G(vintage)
        vintages = panel['vintage_year'].values.astype(float)
        panel['G_vintage'] = self.G_spline_(vintages)

        # H(caltime) - need to map year_month to cal_time_idx
        # Build reverse map: year_month -> idx
        idx_map = {v: k for k, v in self.cal_time_map_.items()}
        cal_indices = panel['year_month'].map(idx_map)

        # For known calendar times, use spline; for unknown, extrapolate
        known_mask = cal_indices.notna()
        H_values = np.zeros(len(panel))
        if known_mask.any():
            H_values[known_mask] = self.H_spline_(
                cal_indices[known_mask].values.astype(float)
            )
        if (~known_mask).any():
            # For out-of-sample calendar times, use the last known H value
            max_idx = max(self.cal_time_map_.keys())
            H_values[~known_mask] = self.H_spline_(float(max_idx))

        panel['H_caltime'] = H_values

        # Zero-mean the transformed values (consistent with training)
        panel['F_age'] -= panel['F_age'].mean()
        panel['G_vintage'] -= panel['G_vintage'].mean()
        panel['H_caltime'] -= panel['H_caltime'].mean()

        return panel

    def fit_macro_regression(
        self,
        panel: pd.DataFrame,
        macro_cols: List[str],
    ) -> 'BreedenAPC':
        """
        Regress H(t) on macro variables via OLS.

        Operates on the unique calendar-time values from the fitted cells,
        linking the extracted H(t) curve to observable macro variables.

        Parameters
        ----------
        panel : pd.DataFrame
            Panel with year_month and macro variable columns.
        macro_cols : list of str
            Names of macro variable columns.

        Returns
        -------
        self
        """
        import statsmodels.api as sm

        # Get unique H(t) values by calendar time
        cells = self.cells_
        ht_by_time = (
            cells.groupby('cal_time_idx')
            .agg(
                H_caltime=('H_caltime', 'mean'),
                year_month=('year_month', 'first'),
                weight=('n_at_risk', 'sum'),
            )
            .reset_index()
        )

        # Get macro values by calendar time from panel
        available_macro = [c for c in macro_cols if c in panel.columns]
        if not available_macro:
            raise ValueError("No macro columns found in panel.")

        macro_by_time = (
            panel.groupby('year_month')[available_macro]
            .mean()
            .reset_index()
        )

        # Merge
        merged = ht_by_time.merge(macro_by_time, on='year_month', how='inner')
        merged = merged.dropna(subset=available_macro)

        if len(merged) < len(available_macro) + 2:
            raise ValueError(
                f"Too few calendar-time observations ({len(merged)}) "
                f"for {len(available_macro)} macro variables."
            )

        # OLS regression
        X = sm.add_constant(merged[available_macro].values.astype(float))
        y = merged['H_caltime'].values
        w = merged['weight'].values

        model = sm.WLS(y, X, weights=w).fit()
        self.macro_regression_ = model
        self._macro_cols = available_macro
        self._macro_cal_data = merged

        return self

    def predict_H_from_macro(
        self,
        macro_values: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict H(t) for out-of-sample calendar times using macro regression.

        Parameters
        ----------
        macro_values : pd.DataFrame
            DataFrame with macro variable columns matching those used in
            fit_macro_regression.

        Returns
        -------
        np.ndarray
            Predicted H(t) values.
        """
        import statsmodels.api as sm

        if self.macro_regression_ is None:
            raise RuntimeError("Must call fit_macro_regression() first.")

        available = [c for c in self._macro_cols if c in macro_values.columns]
        X = sm.add_constant(
            macro_values[available].values.astype(float),
            has_constant='add',
        )
        return self.macro_regression_.predict(X)

    def plot_components(
        self,
        axes=None,
        figsize: Tuple[int, int] = (16, 4),
    ):
        """
        Plot F(a), G(v), H(t) curves.

        Parameters
        ----------
        axes : array of matplotlib Axes, optional
            Three axes for F, G, H. If None, creates a new figure.
        figsize : tuple
            Figure size if creating new figure.

        Returns
        -------
        axes
        """
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=figsize)

        cells = self.cells_

        # F(age)
        age_data = (
            cells.groupby('loan_age')
            .agg(F=('F_age', 'mean'), w=('n_at_risk', 'sum'))
            .reset_index()
            .sort_values('loan_age')
        )
        axes[0].plot(age_data['loan_age'], age_data['F'], 'b-', linewidth=2)
        axes[0].set_xlabel('Loan Age (months)')
        axes[0].set_ylabel('F(a)')
        axes[0].set_title('Age Effect F(a)')
        axes[0].grid(True, alpha=0.3)

        # G(vintage)
        vin_data = (
            cells.groupby('vintage_year')
            .agg(G=('G_vintage', 'mean'), w=('n_at_risk', 'sum'))
            .reset_index()
            .sort_values('vintage_year')
        )
        axes[1].bar(
            vin_data['vintage_year'].astype(str), vin_data['G'],
            alpha=0.7, color='steelblue',
        )
        axes[1].set_xlabel('Vintage Year')
        axes[1].set_ylabel('G(v)')
        axes[1].set_title('Vintage Effect G(v)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')

        # H(caltime)
        cal_data = (
            cells.groupby('cal_time_idx')
            .agg(
                H=('H_caltime', 'mean'),
                year_month=('year_month', 'first'),
                w=('n_at_risk', 'sum'),
            )
            .reset_index()
            .sort_values('cal_time_idx')
        )
        axes[2].plot(
            cal_data['year_month'].astype(str), cal_data['H'],
            'b-', linewidth=1.5,
        )
        axes[2].set_xlabel('Calendar Time')
        axes[2].set_ylabel('H(t)')
        axes[2].set_title('Calendar Time Effect H(t)')
        n_ticks = len(cal_data)
        if n_ticks > 10:
            step = max(1, n_ticks // 10)
            axes[2].set_xticks(axes[2].get_xticks()[::step])
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        return axes
