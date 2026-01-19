"""
Model evaluation metrics for competing risks analysis.

Following Blumenstock et al. (2022) methodology:
- Time-dependent concordance index at 24, 48, 72 months
- Separate evaluation for prepayment (k=1) and default (k=2)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.metrics import concordance_index_censored


# Default evaluation time points from paper
EVAL_TIMES = [24, 48, 72]


def time_dependent_concordance_index(
    event_times: np.ndarray,
    event_codes: np.ndarray,
    risk_scores: np.ndarray,
    eval_time: float,
    event_of_interest: int = 1,
) -> Tuple[float, int, int]:
    """
    Calculate time-dependent concordance index at a specific time point.

    Following Antolini et al. (2005) as used in Blumenstock et al. (2022):
    C_k^td(t) = P(risk_i > risk_j | T_i <= t, K_i = k, T_j > t)

    Parameters
    ----------
    event_times : np.ndarray
        Observed times
    event_codes : np.ndarray
        Event codes (0=censored, 1=prepay, 2=default)
    risk_scores : np.ndarray
        Predicted risk scores (higher = more risk)
    eval_time : float
        Time point for evaluation (e.g., 24, 48, 72 months)
    event_of_interest : int
        Event code to evaluate

    Returns
    -------
    Tuple[float, int, int]
        (concordance_index, n_concordant, n_comparable)
    """
    event_times = np.asarray(event_times)
    event_codes = np.asarray(event_codes)
    risk_scores = np.asarray(risk_scores)

    n = len(event_times)

    # Cases: experienced event of interest by time t
    cases_mask = (event_times <= eval_time) & (event_codes == event_of_interest)

    # Controls: still at risk at time t (no event of any type)
    controls_mask = event_times > eval_time

    case_indices = np.where(cases_mask)[0]
    control_indices = np.where(controls_mask)[0]

    if len(case_indices) == 0 or len(control_indices) == 0:
        return np.nan, 0, 0

    concordant = 0
    tied = 0
    total = 0

    for i in case_indices:
        for j in control_indices:
            total += 1
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] == risk_scores[j]:
                tied += 0.5
                concordant += 0.5

    c_index = concordant / total if total > 0 else np.nan

    return c_index, int(concordant), total


def concordance_index_competing_risks(
    event_times: np.ndarray,
    event_codes: np.ndarray,
    risk_scores: np.ndarray,
    event_of_interest: int = 1,
) -> Tuple[float, int, int, int, int]:
    """
    Calculate overall concordance index for competing risks.

    Uses scikit-survival's implementation with competing events treated as censored.

    Parameters
    ----------
    event_times : np.ndarray
        Observed times
    event_codes : np.ndarray
        Event codes
    risk_scores : np.ndarray
        Predicted risk scores
    event_of_interest : int
        Event code to evaluate

    Returns
    -------
    Tuple[float, int, int, int, int]
        (c_index, concordant, discordant, tied_risk, tied_time)
    """
    # Treat competing events as censored for this event
    event_indicator = (event_codes == event_of_interest)

    return concordance_index_censored(
        event_indicator,
        event_times,
        risk_scores
    )


def evaluate_model_at_times(
    event_times: np.ndarray,
    event_codes: np.ndarray,
    risk_scores: np.ndarray,
    event_of_interest: int,
    eval_times: List[int] = EVAL_TIMES,
) -> Dict[str, float]:
    """
    Evaluate model at multiple time points.

    Following paper's reporting structure: C(24), C(48), C(72), and mean.

    Parameters
    ----------
    event_times : np.ndarray
        Observed times
    event_codes : np.ndarray
        Event codes
    risk_scores : np.ndarray
        Predicted risk scores
    event_of_interest : int
        Event code (1=prepay, 2=default)
    eval_times : list
        Time points for evaluation

    Returns
    -------
    Dict[str, float]
        Dictionary with C(t) for each time and mean
    """
    results = {}

    c_values = []
    for t in eval_times:
        c_index, _, _ = time_dependent_concordance_index(
            event_times, event_codes, risk_scores, t, event_of_interest
        )
        results[f'C({t})'] = c_index
        if not np.isnan(c_index):
            c_values.append(c_index)

    # Mean across time points
    results['mean_C'] = np.mean(c_values) if c_values else np.nan

    return results


def evaluate_all_events(
    event_times: np.ndarray,
    event_codes: np.ndarray,
    risk_scores_prepay: np.ndarray,
    risk_scores_default: np.ndarray,
    eval_times: List[int] = EVAL_TIMES,
) -> pd.DataFrame:
    """
    Evaluate model for both prepayment and default.

    Returns results in paper's format: C_k(t) for k=1,2 and t=24,48,72.

    Parameters
    ----------
    event_times : np.ndarray
        Observed times
    event_codes : np.ndarray
        Event codes
    risk_scores_prepay : np.ndarray
        Risk scores for prepayment model
    risk_scores_default : np.ndarray
        Risk scores for default model
    eval_times : list
        Time points for evaluation

    Returns
    -------
    pd.DataFrame
        Results table with columns: Metric, Prepay (k=1), Default (k=2), Combined
    """
    results = []

    # Evaluate prepayment (k=1)
    prepay_results = evaluate_model_at_times(
        event_times, event_codes, risk_scores_prepay, 1, eval_times
    )

    # Evaluate default (k=2)
    default_results = evaluate_model_at_times(
        event_times, event_codes, risk_scores_default, 2, eval_times
    )

    # Build results table
    for t in eval_times:
        results.append({
            'Metric': f'C({t})',
            'Prepay (k=1)': prepay_results[f'C({t})'],
            'Default (k=2)': default_results[f'C({t})'],
        })

    # Add means
    results.append({
        'Metric': 'ØC',
        'Prepay (k=1)': prepay_results['mean_C'],
        'Default (k=2)': default_results['mean_C'],
    })

    df = pd.DataFrame(results)

    # Add combined mean
    df['Combined'] = (df['Prepay (k=1)'] + df['Default (k=2)']) / 2

    return df


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    models: Dict[str, object],
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    eval_times: List[int] = EVAL_TIMES,
) -> pd.DataFrame:
    """
    Run a complete experiment with multiple models.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    feature_cols : list
        Feature column names
    models : dict
        Dictionary of model names to model objects
        Each model should have fit() and predict_risk() methods
    duration_col : str
        Duration column name
    event_col : str
        Event code column name
    eval_times : list
        Evaluation time points

    Returns
    -------
    pd.DataFrame
        Results for all models
    """
    all_results = []

    event_times = test_df[duration_col].values
    event_codes = test_df[event_col].values

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")

        # Get predictions (model-specific)
        if hasattr(model, 'predict_risk'):
            risk_prepay = model.predict_risk(test_df[feature_cols], event=1)
            risk_default = model.predict_risk(test_df[feature_cols], event=2)
        elif hasattr(model, 'predict_partial_hazard'):
            # lifelines Cox model
            risk_prepay = model.predict_partial_hazard(test_df[feature_cols]).values.flatten()
            risk_default = risk_prepay  # Same model for both
        else:
            # sklearn-style model
            risk_prepay = model.predict_proba(test_df[feature_cols])[:, 1]
            risk_default = risk_prepay

        # Evaluate
        results = evaluate_all_events(
            event_times, event_codes, risk_prepay, risk_default, eval_times
        )
        results['Model'] = model_name

        all_results.append(results)

    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    cols = ['Model', 'Metric', 'Prepay (k=1)', 'Default (k=2)', 'Combined']
    final_df = final_df[cols]

    return final_df


def format_results_table(
    results_df: pd.DataFrame,
    multiply_by_100: bool = True,
) -> pd.DataFrame:
    """
    Format results table following paper's style.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_experiment
    multiply_by_100 : bool
        Whether to multiply C-index by 100

    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    df = results_df.copy()

    # Multiply by 100 for readability (as in paper)
    if multiply_by_100:
        for col in ['Prepay (k=1)', 'Default (k=2)', 'Combined']:
            df[col] = df[col] * 100

    # Pivot to get models as rows
    df_pivot = df.pivot(index='Model', columns='Metric', values='Combined')

    # Reorder columns
    time_cols = [f'C({t})' for t in EVAL_TIMES]
    cols = time_cols + ['ØC']
    df_pivot = df_pivot[[c for c in cols if c in df_pivot.columns]]

    return df_pivot.round(2)


def brier_score_competing_risks(
    event_times: np.ndarray,
    event_codes: np.ndarray,
    predicted_cif: np.ndarray,
    eval_time: float,
    event_of_interest: int = 1,
) -> float:
    """
    Calculate Brier score for competing risks at a specific time.

    BS(t) = E[(I(T <= t, K = k) - CIF_k(t))^2]

    Parameters
    ----------
    event_times : np.ndarray
        Observed times
    event_codes : np.ndarray
        Event codes
    predicted_cif : np.ndarray
        Predicted CIF at eval_time for each subject
    eval_time : float
        Time point for evaluation
    event_of_interest : int
        Event code to evaluate

    Returns
    -------
    float
        Brier score (lower is better)
    """
    observed = (
        (event_times <= eval_time) &
        (event_codes == event_of_interest)
    ).astype(float)

    return np.mean((observed - predicted_cif) ** 2)


def compare_model_coefficients(
    fine_gray_coefs: pd.DataFrame,
    cause_specific_coefs: pd.DataFrame,
    feature_col: str = 'feature',
    coef_col: str = 'coefficient',
) -> pd.DataFrame:
    """
    Compare coefficients between Fine-Gray and cause-specific models.

    Parameters
    ----------
    fine_gray_coefs : pd.DataFrame
        Fine-Gray model coefficients
    cause_specific_coefs : pd.DataFrame
        Cause-specific model coefficients
    feature_col : str
        Column with feature names
    coef_col : str
        Column with coefficient values

    Returns
    -------
    pd.DataFrame
        Comparison table with both coefficients and hazard ratios
    """
    comparison = fine_gray_coefs[[feature_col, coef_col]].copy()
    comparison.columns = [feature_col, 'coef_fine_gray']

    comparison = comparison.merge(
        cause_specific_coefs[[feature_col, coef_col]].rename(
            columns={coef_col: 'coef_cause_specific'}
        ),
        on=feature_col,
        how='outer'
    )

    # Add hazard ratios
    comparison['hr_fine_gray'] = np.exp(comparison['coef_fine_gray'])
    comparison['hr_cause_specific'] = np.exp(comparison['coef_cause_specific'])

    # Add difference
    comparison['coef_diff'] = (
        comparison['coef_fine_gray'] - comparison['coef_cause_specific']
    )
    comparison['hr_ratio'] = (
        comparison['hr_fine_gray'] / comparison['hr_cause_specific']
    )

    return comparison


def calibration_plot(
    observed_cif: np.ndarray,
    predicted_cif: np.ndarray,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    title: str = 'Calibration Plot',
) -> plt.Axes:
    """
    Create calibration plot comparing predicted vs observed CIF.

    Parameters
    ----------
    observed_cif : np.ndarray
        Observed cumulative incidence (0 or 1)
    predicted_cif : np.ndarray
        Predicted cumulative incidence probabilities
    n_bins : int
        Number of bins for grouping predictions
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Bin predictions
    bins = np.percentile(predicted_cif, np.linspace(0, 100, n_bins + 1))
    bins[0] = 0
    bins[-1] = 1

    bin_indices = np.digitize(predicted_cif, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate observed and predicted means per bin
    pred_means = []
    obs_means = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            pred_means.append(predicted_cif[mask].mean())
            obs_means.append(observed_cif[mask].mean())

    # Plot
    ax.scatter(pred_means, obs_means, s=100, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    ax.set_xlabel('Predicted CIF')
    ax.set_ylabel('Observed CIF')
    ax.set_title(title)
    ax.set_xlim(0, max(max(pred_means) * 1.1, 0.1) if pred_means else 0.1)
    ax.set_ylim(0, max(max(obs_means) * 1.1, 0.1) if obs_means else 0.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add calibration slope
    if len(pred_means) > 1:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            pred_means, obs_means
        )
        ax.annotate(
            f'Slope: {slope:.2f}\nR²: {r_value**2:.3f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    return ax


def plot_concordance_comparison(
    results_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = 'Model Comparison: Time-Dependent Concordance',
) -> plt.Axes:
    """
    Plot concordance index comparison across models and time points.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_experiment
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot to get format for plotting
    df = results_df.copy()
    df = df[df['Metric'].str.startswith('C(')]

    models = df['Model'].unique()
    x = np.arange(len(df['Metric'].unique()))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]['Combined'].values * 100
        ax.bar(x + i * width, model_data, width, label=model, alpha=0.8)

    ax.set_xlabel('Evaluation Time')
    ax.set_ylabel('Concordance Index (×100)')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(df['Metric'].unique())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    return ax
