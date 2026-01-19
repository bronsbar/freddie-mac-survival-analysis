"""
Cumulative Incidence Function (CIF) estimation for competing risks.

The cumulative incidence function gives the probability of experiencing
a specific event by time t, accounting for competing risks.

Key distinction:
- 1 - Kaplan-Meier is NOT the same as CIF when competing risks exist
- CIF properly accounts for subjects who experience competing events
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter
from sksurv.nonparametric import CensoringDistributionEstimator


def estimate_cif_aalen_johansen(
    durations: np.ndarray,
    event_codes: np.ndarray,
    event_of_interest: int = 1,
) -> AalenJohansenFitter:
    """
    Estimate cumulative incidence using Aalen-Johansen estimator.

    The Aalen-Johansen estimator is the non-parametric analog of
    Kaplan-Meier for competing risks.

    Parameters
    ----------
    durations : np.ndarray
        Survival/censoring times
    event_codes : np.ndarray
        Event codes (0=censored, 1=primary, 2=competing, etc.)
    event_of_interest : int
        Event code to estimate CIF for

    Returns
    -------
    AalenJohansenFitter
        Fitted estimator with cumulative_density_ attribute
    """
    ajf = AalenJohansenFitter()
    ajf.fit(durations, event_codes, event_of_interest=event_of_interest)
    return ajf


def estimate_cif_by_group(
    df: pd.DataFrame,
    group_col: str,
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    event_of_interest: int = 1,
) -> Dict[str, AalenJohansenFitter]:
    """
    Estimate CIF separately for each group.

    Parameters
    ----------
    df : pd.DataFrame
        Data with one row per subject
    group_col : str
        Column to group by
    duration_col : str
        Duration column
    event_col : str
        Event code column
    event_of_interest : int
        Event to estimate CIF for

    Returns
    -------
    Dict[str, AalenJohansenFitter]
        Dictionary mapping group values to fitted estimators
    """
    results = {}

    for group_val in df[group_col].unique():
        mask = df[group_col] == group_val
        if mask.sum() > 10:  # Minimum sample size
            ajf = estimate_cif_aalen_johansen(
                df.loc[mask, duration_col].values,
                df.loc[mask, event_col].values,
                event_of_interest=event_of_interest
            )
            results[group_val] = ajf

    return results


def estimate_cif_from_model(
    model,
    df: pd.DataFrame,
    times: np.ndarray,
    id_col: str = 'loan_sequence_number',
) -> pd.DataFrame:
    """
    Estimate CIF from a fitted discrete-time model.

    For discrete-time models:
    CIF(t) = sum over s<=t of: h(s) * S(s-1)

    where h(s) is the subdistribution hazard and S(s) is survival.

    Parameters
    ----------
    model : DiscreteTimeFineGray
        Fitted discrete-time model
    df : pd.DataFrame
        Data to predict on (loan-month panel)
    times : np.ndarray
        Time points for CIF estimation
    id_col : str
        Subject identifier column

    Returns
    -------
    pd.DataFrame
        CIF estimates at each time point
    """
    # Get hazard predictions
    cif_result = model.predict_cumulative_incidence(df, id_col=id_col)

    # Aggregate to get mean CIF at each time point
    cif_by_time = cif_result.groupby('loan_age')['cif'].mean()

    # Interpolate to requested times
    cif_interp = np.interp(times, cif_by_time.index, cif_by_time.values)

    return pd.DataFrame({'time': times, 'cif': cif_interp})


def plot_cumulative_incidence(
    estimators: Dict[str, AalenJohansenFitter],
    title: str = 'Cumulative Incidence Function',
    xlabel: str = 'Time (months)',
    ylabel: str = 'Cumulative Incidence',
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot cumulative incidence curves for multiple groups.

    Parameters
    ----------
    estimators : Dict[str, AalenJohansenFitter]
        Dictionary mapping group names to fitted estimators
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : Tuple[int, int]
        Figure size
    colors : List[str], optional
        Colors for each group
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    plt.Axes
        Plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if colors is None:
        colors = plt.cm.tab10.colors

    for i, (name, ajf) in enumerate(estimators.items()):
        color = colors[i % len(colors)]
        ajf.plot(ax=ax, label=name, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    return ax


def plot_cif_comparison(
    df: pd.DataFrame,
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot CIF for prepayment and default side by side.

    Parameters
    ----------
    df : pd.DataFrame
        Data with one row per subject
    duration_col : str
        Duration column
    event_col : str
        Event code column
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Figure with CIF plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # CIF for prepayment (event=1)
    ajf_prepay = estimate_cif_aalen_johansen(
        df[duration_col].values,
        df[event_col].values,
        event_of_interest=1
    )
    ajf_prepay.plot(ax=axes[0], color='steelblue')
    axes[0].set_title('Cumulative Incidence: Prepayment')
    axes[0].set_xlabel('Time (months)')
    axes[0].set_ylabel('Cumulative Incidence')
    axes[0].grid(True, alpha=0.3)

    # CIF for default (event=2)
    ajf_default = estimate_cif_aalen_johansen(
        df[duration_col].values,
        df[event_col].values,
        event_of_interest=2
    )
    ajf_default.plot(ax=axes[1], color='indianred')
    axes[1].set_title('Cumulative Incidence: Default')
    axes[1].set_xlabel('Time (months)')
    axes[1].set_ylabel('Cumulative Incidence')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_cif_vs_kaplan_meier(
    df: pd.DataFrame,
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    event_of_interest: int = 1,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Compare CIF with 1-KM to show the difference.

    This demonstrates why 1 - Kaplan-Meier is incorrect for
    cumulative incidence when competing risks exist.

    Parameters
    ----------
    df : pd.DataFrame
        Data
    duration_col : str
        Duration column
    event_col : str
        Event code column
    event_of_interest : int
        Event to compare
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    plt.Axes
    """
    from lifelines import KaplanMeierFitter

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    durations = df[duration_col].values
    events = df[event_col].values

    # CIF (correct)
    ajf = AalenJohansenFitter()
    ajf.fit(durations, events, event_of_interest=event_of_interest)

    # 1 - KM (incorrect, but commonly used)
    kmf = KaplanMeierFitter()
    event_binary = (events == event_of_interest).astype(int)
    kmf.fit(durations, event_binary)

    # Plot
    times = ajf.cumulative_density_.index
    cif = ajf.cumulative_density_.values.flatten()

    km_times = kmf.survival_function_.index
    one_minus_km = 1 - kmf.survival_function_.values.flatten()

    ax.step(times, cif, where='post', label='CIF (Aalen-Johansen)', linewidth=2)
    ax.step(km_times, one_minus_km, where='post', label='1 - KM (incorrect)',
            linestyle='--', linewidth=2)

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Cumulative Incidence')
    ax.set_title('CIF vs 1-Kaplan-Meier Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(
        '1-KM overestimates\nwhen competing risks exist',
        xy=(times[len(times)//2], one_minus_km[len(km_times)//2]),
        xytext=(times[len(times)//3], one_minus_km[len(km_times)//2] + 0.05),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='gray'
    )

    return ax
