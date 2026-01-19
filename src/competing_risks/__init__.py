"""
Competing Risks Analysis Module for Mortgage Prepayment Modeling.

Replicating Blumenstock et al. (2022) methodology for survival analysis
with competing risks (prepayment and default).

Models:
-------
- Cause-Specific Cox (CSC): Semi-parametric Cox model
- Fine-Gray (FGR): Subdistribution hazard model
- Random Survival Forest (RSF): ML ensemble approach

Modules:
--------
data_prep : Time-varying covariate dataset creation
fine_gray : Discrete-time Fine-Gray model implementation
cause_specific : Cause-specific Cox model wrappers
random_forest : Random Survival Forest for competing risks
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

__version__ = '0.2.0'
