"""
Competing Risks Analysis Module for Mortgage Prepayment Modeling.

Replicating Blumenstock et al. (2022) methodology for survival analysis
with competing risks (prepayment and default).

Models:
-------
- Cause-Specific Cox (CSC): Semi-parametric Cox model
- Fine-Gray (FGR): Subdistribution hazard model
- Random Survival Forest (RSF): ML ensemble approach
- DeepHit: Deep learning approach (Lee et al., 2018)
- Dynamic-DeepHit: Dynamic deep learning with longitudinal data (Lee et al., 2020)
- Bayesian PHM: Bayesian competing risks (Bhattacharya et al., 2019)
- Breeden-Crook: Multihorizon discrete-time survival (Breeden & Crook, 2022)

Modules:
--------
data_prep : Time-varying covariate dataset creation
fine_gray : Discrete-time Fine-Gray model implementation
cause_specific : Cause-specific Cox model wrappers
random_forest : Random Survival Forest for competing risks
deephit : DeepHit deep learning model (pycox/PyTorch)
dynamic_deephit : Dynamic-DeepHit with GRU + temporal attention (PyTorch)
bayesian_phm : Bayesian competing risks PHM (Pyro/PyTorch)
bayesian_evaluation : Evaluation metrics for Bayesian models
breeden_crook : Breeden & Crook (2022) multihorizon discrete-time survival
cumulative_incidence : CIF estimation functions
evaluation : Model comparison and validation metrics
"""

from .data_prep import (
    create_loan_month_panel,
    create_fine_gray_dataset,
    create_cause_specific_dataset,
)

from .fine_gray import (
    DiscreteTimeFineGray,
    fit_discrete_time_competing_risks,
)

from .cause_specific import (
    fit_cause_specific_cox,
    CauseSpecificCox,
)

from .random_forest import (
    CompetingRisksRSF,
    fit_rsf_competing_risks,
)

from .deephit import (
    CompetingRisksDeepHit,
    fit_deephit_competing_risks,
)

from .dynamic_deephit import (
    DynamicDeepHitNetwork,
    DynamicDeepHitLoss,
    MortgageSequenceDataset,
    collate_mortgage_sequences,
    preprocess_panel_to_sequences,
)

# Bayesian model (optional - requires Pyro/PyTorch)
try:
    from .bayesian_phm import (
        BayesianCompetingRisksPHM,
        lognormal_log_hazard,
        lognormal_cumulative_hazard,
    )
    from .bayesian_evaluation import (
        compute_time_dependent_cindex,
        compute_brier_score,
        compute_calibration,
        compute_coverage_probability,
        compute_posterior_predictive_pvalues,
        compute_standardized_residuals,
        evaluate_bayesian_model,
        format_evaluation_results,
    )
    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False
    BayesianCompetingRisksPHM = None
    compute_time_dependent_cindex = None
    compute_brier_score = None
    compute_calibration = None
    compute_coverage_probability = None
    compute_posterior_predictive_pvalues = None
    compute_standardized_residuals = None
    evaluate_bayesian_model = None
    format_evaluation_results = None

from .breeden_crook import (
    BreedenCrookMultihorizon,
    enrich_panel_with_delinquency,
    create_delinquency_indicators,
    create_lagged_delinquency,
    build_feature_matrix_apc,
    plot_delinquency_coefficients,
    plot_origination_coefficients,
    plot_pseudo_r2_by_horizon,
)

from .apc_decomposition import (
    BreedenAPC,
    aggregate_portfolio_rates,
)

from .cumulative_incidence import (
    estimate_cif_aalen_johansen,
    estimate_cif_from_model,
    plot_cumulative_incidence,
)

from .evaluation import (
    concordance_index_competing_risks,
    time_dependent_concordance_index,
    evaluate_model_at_times,
    evaluate_all_events,
    run_experiment,
    format_results_table,
    brier_score_competing_risks,
    compare_model_coefficients,
    calibration_plot,
    plot_concordance_comparison,
    EVAL_TIMES,
)

__all__ = [
    # Data preparation
    'create_loan_month_panel',
    'create_fine_gray_dataset',
    'create_cause_specific_dataset',
    # Fine-Gray
    'DiscreteTimeFineGray',
    'fit_discrete_time_competing_risks',
    # Cause-specific
    'fit_cause_specific_cox',
    'CauseSpecificCox',
    # Random Survival Forest
    'CompetingRisksRSF',
    'fit_rsf_competing_risks',
    # DeepHit
    'CompetingRisksDeepHit',
    'fit_deephit_competing_risks',
    # Dynamic-DeepHit
    'DynamicDeepHitNetwork',
    'DynamicDeepHitLoss',
    'MortgageSequenceDataset',
    'collate_mortgage_sequences',
    'preprocess_panel_to_sequences',
    # Bayesian PHM
    'BayesianCompetingRisksPHM',
    'lognormal_log_hazard',
    'lognormal_cumulative_hazard',
    # Bayesian evaluation
    'compute_time_dependent_cindex',
    'compute_brier_score',
    'compute_calibration',
    'compute_coverage_probability',
    'compute_posterior_predictive_pvalues',
    'compute_standardized_residuals',
    'evaluate_bayesian_model',
    'format_evaluation_results',
    # Breeden-Crook
    'BreedenCrookMultihorizon',
    'enrich_panel_with_delinquency',
    'create_delinquency_indicators',
    'create_lagged_delinquency',
    'build_feature_matrix_apc',
    'plot_delinquency_coefficients',
    'plot_origination_coefficients',
    'plot_pseudo_r2_by_horizon',
    # APC decomposition
    'BreedenAPC',
    'aggregate_portfolio_rates',
    # Cumulative incidence
    'estimate_cif_aalen_johansen',
    'estimate_cif_from_model',
    'plot_cumulative_incidence',
    # Evaluation
    'concordance_index_competing_risks',
    'time_dependent_concordance_index',
    'evaluate_model_at_times',
    'evaluate_all_events',
    'run_experiment',
    'format_results_table',
    'brier_score_competing_risks',
    'compare_model_coefficients',
    'calibration_plot',
    'plot_concordance_comparison',
    'EVAL_TIMES',
]

__version__ = '0.5.0'
