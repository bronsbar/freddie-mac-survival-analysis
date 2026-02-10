"""
Evaluation functions for Bayesian competing risks model.

Provides metrics for assessing the Bayesian PHM including:
- Time-dependent concordance index (IPCW)
- Brier score for competing risks
- Calibration assessment
- Posterior predictive checks
- Coverage probability

References:
    Bhattacharya, A., Wilson, S.P., & Soyer, R. (2019). A Bayesian approach
    to modeling mortgage default and prepayment. European Journal of
    Operational Research, 274, 1112-1124.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import warnings

try:
    from sksurv.metrics import concordance_index_ipcw, brier_score
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    warnings.warn("scikit-survival not available. Some metrics will be unavailable.")


def compute_time_dependent_cindex(
    model,
    X_train: np.ndarray,
    durations_train: np.ndarray,
    events_train: np.ndarray,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    times: List[int] = [24, 48, 72],
    cause: str = 'default'
) -> Dict[int, float]:
    """
    Compute time-dependent C-index (IPCW) at specified time horizons.

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_train : array of shape (n_train, n_features)
        Training covariates (for censoring distribution estimation)
    durations_train : array of shape (n_train,)
        Training durations
    events_train : array of shape (n_train,)
        Training event codes (0=censored, 1=prepay, 2=default)
    X_test : array of shape (n_test, n_features)
        Test covariates
    durations_test : array of shape (n_test,)
        Test durations
    events_test : array of shape (n_test,)
        Test event codes
    times : list of int
        Time horizons at which to evaluate C-index
    cause : str, {'default', 'prepay'}
        Which cause to evaluate

    Returns
    -------
    results : dict
        Dictionary mapping time horizon to C-index value
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival required. Install with: pip install scikit-survival")

    # Create survival objects for the specific cause
    if cause == 'default':
        event_indicator_train = (events_train == 2)
        event_indicator_test = (events_test == 2)
    else:  # prepay
        event_indicator_train = (events_train == 1)
        event_indicator_test = (events_test == 1)

    y_train = Surv.from_arrays(event_indicator_train, durations_train)
    y_test = Surv.from_arrays(event_indicator_test, durations_test)

    results = {}

    for tau in times:
        try:
            # Get posterior mean CIF at time tau
            cif_mean, _, _ = model.predict_cif(X_test, np.array([tau]), cause=cause)
            risk_score = cif_mean[:, 0]  # CIF at tau as risk score

            # Compute IPCW C-index
            c_index, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
                y_train, y_test, risk_score, tau=tau
            )
            results[tau] = c_index

        except Exception as e:
            warnings.warn(f"Could not compute C-index at tau={tau}: {str(e)}")
            results[tau] = np.nan

    return results


def compute_brier_score(
    model,
    X_train: np.ndarray,
    durations_train: np.ndarray,
    events_train: np.ndarray,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    times: np.ndarray,
    cause: str = 'default'
) -> Tuple[np.ndarray, float]:
    """
    Compute time-dependent Brier score and integrated Brier score.

    The Brier score measures calibration - how well predicted probabilities
    match observed outcomes.

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_train, durations_train, events_train : arrays
        Training data for IPCW weights
    X_test, durations_test, events_test : arrays
        Test data for evaluation
    times : array
        Time points at which to evaluate Brier score
    cause : str, {'default', 'prepay'}
        Which cause to evaluate

    Returns
    -------
    bs : array of shape (len(times),)
        Brier scores at each time point
    ibs : float
        Integrated Brier score
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival required. Install with: pip install scikit-survival")

    # Create survival objects
    if cause == 'default':
        event_indicator_train = (events_train == 2)
        event_indicator_test = (events_test == 2)
    else:
        event_indicator_train = (events_train == 1)
        event_indicator_test = (events_test == 1)

    y_train = Surv.from_arrays(event_indicator_train, durations_train)
    y_test = Surv.from_arrays(event_indicator_test, durations_test)

    # Get CIF predictions at all time points
    cif_mean, _, _ = model.predict_cif(X_test, times, cause=cause)

    # For Brier score, we need survival probabilities (1 - CIF for single cause)
    # In competing risks, we use the cause-specific CIF directly
    # scikit-survival's brier_score expects survival probabilities
    survival_prob = 1 - cif_mean

    try:
        # Compute Brier score
        _, bs = brier_score(y_train, y_test, survival_prob, times)

        # Integrated Brier Score (trapezoidal integration)
        ibs = np.trapz(bs, times) / (times[-1] - times[0])

        return bs, ibs

    except Exception as e:
        warnings.warn(f"Could not compute Brier score: {str(e)}")
        return np.full(len(times), np.nan), np.nan


def compute_calibration(
    model,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    tau: int,
    cause: str = 'default',
    n_groups: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute calibration metrics by decile of predicted risk.

    Compares predicted CIF at time tau with observed event rates
    within groups of predicted risk.

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_test : array of shape (n_test, n_features)
        Test covariates
    durations_test : array of shape (n_test,)
        Test durations
    events_test : array of shape (n_test,)
        Test event codes
    tau : int
        Time horizon for calibration assessment
    cause : str, {'default', 'prepay'}
        Which cause to evaluate
    n_groups : int
        Number of groups (deciles by default)

    Returns
    -------
    calibration : dict
        Dictionary with:
        - 'predicted_mean': mean predicted risk per group
        - 'observed_rate': observed event rate per group
        - 'group_size': number of observations per group
        - 'group_lower': lower bound of predicted risk per group
        - 'group_upper': upper bound of predicted risk per group
    """
    # Get CIF predictions at tau
    cif_mean, cif_lower, cif_upper = model.predict_cif(X_test, np.array([tau]), cause=cause)
    predicted_risk = cif_mean[:, 0]

    # Determine event indicator
    if cause == 'default':
        event_occurred = (events_test == 2) & (durations_test <= tau)
    else:
        event_occurred = (events_test == 1) & (durations_test <= tau)

    # At risk at tau (not censored before tau, or event before tau)
    at_risk = (durations_test >= tau) | event_occurred

    # Create groups based on predicted risk percentiles
    percentiles = np.linspace(0, 100, n_groups + 1)
    thresholds = np.percentile(predicted_risk, percentiles)

    predicted_mean = []
    observed_rate = []
    group_size = []
    group_lower = []
    group_upper = []

    for i in range(n_groups):
        if i == n_groups - 1:
            mask = (predicted_risk >= thresholds[i]) & (predicted_risk <= thresholds[i + 1])
        else:
            mask = (predicted_risk >= thresholds[i]) & (predicted_risk < thresholds[i + 1])

        if mask.sum() > 0:
            # Only count those at risk for observed rate
            mask_at_risk = mask & at_risk
            if mask_at_risk.sum() > 0:
                obs_rate = event_occurred[mask_at_risk].mean()
            else:
                obs_rate = np.nan

            predicted_mean.append(predicted_risk[mask].mean())
            observed_rate.append(obs_rate)
            group_size.append(mask.sum())
            group_lower.append(thresholds[i])
            group_upper.append(thresholds[i + 1])

    return {
        'predicted_mean': np.array(predicted_mean),
        'observed_rate': np.array(observed_rate),
        'group_size': np.array(group_size),
        'group_lower': np.array(group_lower),
        'group_upper': np.array(group_upper),
    }


def compute_coverage_probability(
    model,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    times: np.ndarray,
    cause: str = 'default',
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute coverage probability of (1-alpha)% credible intervals.

    For a well-calibrated Bayesian model, the coverage should be
    approximately (1-alpha).

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_test : array of shape (n_test, n_features)
        Test covariates
    durations_test : array of shape (n_test,)
        Test durations
    events_test : array of shape (n_test,)
        Test event codes
    times : array
        Time points for evaluation
    cause : str, {'default', 'prepay'}
        Which cause to evaluate
    alpha : float
        Significance level (default 0.05 for 95% CI)

    Returns
    -------
    coverage : dict
        Dictionary with:
        - 'overall': overall coverage across all observations and times
        - 'by_time': coverage at each time point
        - 'nominal': nominal coverage level (1-alpha)
    """
    # Get CIF predictions with credible intervals
    cif_mean, cif_lower, cif_upper = model.predict_cif(X_test, times, cause=cause)

    # Determine which observations had the event by each time
    if cause == 'default':
        event_code = 2
    else:
        event_code = 1

    n_test = len(durations_test)
    n_times = len(times)

    # For each observation, compute empirical CIF indicator at each time
    # I(T <= t, event = cause)
    empirical_cif = np.zeros((n_test, n_times))
    for i in range(n_test):
        for t_idx, t in enumerate(times):
            if durations_test[i] <= t and events_test[i] == event_code:
                empirical_cif[i, t_idx] = 1.0

    # Check if empirical falls within credible interval
    # Note: This is a simplification - true coverage would need
    # to account for censoring properly
    within_ci = (empirical_cif >= cif_lower) & (empirical_cif <= cif_upper)

    # For censored observations before time t, we can't assess coverage
    # So we focus on uncensored observations
    valid_mask = np.zeros((n_test, n_times), dtype=bool)
    for i in range(n_test):
        for t_idx, t in enumerate(times):
            # Valid if: event occurred before t, OR observation extends beyond t
            if durations_test[i] <= t or events_test[i] != 0:
                valid_mask[i, t_idx] = True

    # Compute coverage
    coverage_by_time = []
    for t_idx in range(n_times):
        valid = valid_mask[:, t_idx]
        if valid.sum() > 0:
            cov = within_ci[valid, t_idx].mean()
            coverage_by_time.append(cov)
        else:
            coverage_by_time.append(np.nan)

    overall_coverage = within_ci[valid_mask].mean() if valid_mask.sum() > 0 else np.nan

    return {
        'overall': overall_coverage,
        'by_time': np.array(coverage_by_time),
        'nominal': 1 - alpha,
        'times': times,
    }


def compute_posterior_predictive_pvalues(
    model,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    cause: str = 'default',
    n_posterior_samples: int = 100
) -> np.ndarray:
    """
    Compute posterior predictive p-values for model checking.

    For each observation, compute P(T_rep < T_obs | data).
    For a well-specified model, these should be approximately Uniform(0,1).

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_test : array of shape (n_test, n_features)
        Test covariates
    durations_test : array of shape (n_test,)
        Test durations
    events_test : array of shape (n_test,)
        Test event codes
    cause : str, {'default', 'prepay'}
        Which cause to evaluate
    n_posterior_samples : int
        Number of posterior samples to use

    Returns
    -------
    pvalues : array of shape (n_test,)
        Posterior predictive p-values for observations with the specified event
    """
    if cause == 'default':
        event_mask = events_test == 2
    else:
        event_mask = events_test == 1

    # Get CIF at observed event times
    observed_times = durations_test[event_mask]
    X_events = X_test[event_mask]

    if len(observed_times) == 0:
        return np.array([])

    # For each observation, CIF(t_obs) gives P(T <= t_obs | cause)
    # This is approximately the posterior predictive p-value
    cif_at_obs, _, _ = model.predict_cif(X_events, observed_times, cause=cause)

    # Extract diagonal (CIF for each obs at its own observed time)
    pvalues = np.array([cif_at_obs[i, i] for i in range(len(observed_times))])

    return pvalues


def compute_standardized_residuals(
    model,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    cause: str = 'default'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute standardized residuals for model assessment.

    r_i = (t_i - E[T_i | posterior]) / sd[T_i | posterior]

    For a well-calibrated model, residuals should be approximately N(0,1).

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_test : array of shape (n_test, n_features)
        Test covariates
    durations_test : array of shape (n_test,)
        Test durations
    events_test : array of shape (n_test,)
        Test event codes
    cause : str, {'default', 'prepay'}
        Which cause to evaluate

    Returns
    -------
    residuals : array
        Standardized residuals for observations with the specified event
    observed_times : array
        Observed event times for these observations
    """
    if cause == 'default':
        event_mask = events_test == 2
    else:
        event_mask = events_test == 1

    observed_times = durations_test[event_mask]
    X_events = X_test[event_mask]

    if len(observed_times) == 0:
        return np.array([]), np.array([])

    # Compute expected time from survival function
    # E[T] = integral_0^inf S(t) dt
    # Approximate with numerical integration
    t_grid = np.linspace(0.5, 360, 500)  # Up to 30 years

    surv_mean, surv_lower, surv_upper = model.predict_survival(X_events, t_grid)

    # Expected time: integral of survival function
    expected_times = np.trapz(surv_mean, t_grid, axis=1)

    # Variance approximation from credible interval width
    # Using interquartile range as robust estimate
    surv_25, _, _ = model.predict_survival(X_events, t_grid)  # Already have mean
    # Approximate std from CI width (assuming normality)
    # CI width ≈ 3.92 * std for 95% CI
    ci_width_survival = surv_upper - surv_lower
    # Propagate to time: very rough approximation
    std_times = np.trapz(ci_width_survival / 3.92, t_grid, axis=1)
    std_times = np.maximum(std_times, 1.0)  # Avoid division by zero

    # Standardized residuals
    residuals = (observed_times - expected_times) / std_times

    return residuals, observed_times


def evaluate_bayesian_model(
    model,
    X_train: np.ndarray,
    durations_train: np.ndarray,
    events_train: np.ndarray,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    time_horizons: List[int] = [24, 48, 72],
    causes: List[str] = ['default', 'prepay']
) -> Dict:
    """
    Comprehensive evaluation of Bayesian competing risks model.

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted Bayesian model
    X_train, durations_train, events_train : arrays
        Training data
    X_test, durations_test, events_test : arrays
        Test data
    time_horizons : list of int
        Time horizons for evaluation
    causes : list of str
        Causes to evaluate ('default', 'prepay')

    Returns
    -------
    results : dict
        Comprehensive evaluation results including:
        - C-index by cause and time horizon
        - Brier scores
        - Calibration metrics
        - Coverage probabilities
    """
    results = {
        'cindex': {},
        'brier': {},
        'calibration': {},
        'coverage': {},
    }

    for cause in causes:
        # Time-dependent C-index
        results['cindex'][cause] = compute_time_dependent_cindex(
            model,
            X_train, durations_train, events_train,
            X_test, durations_test, events_test,
            times=time_horizons,
            cause=cause
        )

        # Brier score
        brier_times = np.linspace(1, max(time_horizons), 50)
        try:
            bs, ibs = compute_brier_score(
                model,
                X_train, durations_train, events_train,
                X_test, durations_test, events_test,
                times=brier_times,
                cause=cause
            )
            results['brier'][cause] = {
                'times': brier_times,
                'scores': bs,
                'integrated': ibs
            }
        except Exception as e:
            results['brier'][cause] = {'error': str(e)}

        # Calibration at each time horizon
        results['calibration'][cause] = {}
        for tau in time_horizons:
            try:
                results['calibration'][cause][tau] = compute_calibration(
                    model, X_test, durations_test, events_test,
                    tau=tau, cause=cause
                )
            except Exception as e:
                results['calibration'][cause][tau] = {'error': str(e)}

        # Coverage probability
        try:
            results['coverage'][cause] = compute_coverage_probability(
                model, X_test, durations_test, events_test,
                times=np.array(time_horizons, dtype=float),
                cause=cause
            )
        except Exception as e:
            results['coverage'][cause] = {'error': str(e)}

    return results


def format_evaluation_results(results: Dict) -> str:
    """
    Format evaluation results as a readable string.

    Parameters
    ----------
    results : dict
        Results from evaluate_bayesian_model()

    Returns
    -------
    formatted : str
        Formatted results string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BAYESIAN COMPETING RISKS MODEL - EVALUATION RESULTS")
    lines.append("=" * 70)

    # C-index
    lines.append("\n## Time-Dependent Concordance Index (IPCW)")
    lines.append("-" * 50)
    for cause, cindex_dict in results['cindex'].items():
        lines.append(f"\n{cause.upper()}:")
        for tau, cindex in cindex_dict.items():
            if not np.isnan(cindex):
                lines.append(f"  tau = {tau:3d} months: C-index = {cindex:.4f}")
            else:
                lines.append(f"  tau = {tau:3d} months: C-index = N/A")

    # Integrated Brier Score
    lines.append("\n## Integrated Brier Score")
    lines.append("-" * 50)
    for cause, brier_dict in results['brier'].items():
        if 'error' not in brier_dict:
            lines.append(f"{cause.upper()}: IBS = {brier_dict['integrated']:.4f}")
        else:
            lines.append(f"{cause.upper()}: Error - {brier_dict['error']}")

    # Coverage
    lines.append("\n## Coverage Probability (95% CI)")
    lines.append("-" * 50)
    for cause, cov_dict in results['coverage'].items():
        if 'error' not in cov_dict:
            lines.append(f"{cause.upper()}: {cov_dict['overall']:.1%} (nominal: {cov_dict['nominal']:.1%})")
        else:
            lines.append(f"{cause.upper()}: Error - {cov_dict['error']}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)
