"""
Asset-Liability Management (ALM) Cash Flow Projections.

Converts cause-specific Cox hazard model outputs into projected mortgage
cash flows for ALM analysis, including scenario analysis and interest
rate risk metrics.

Modules:
--------
baseline_hazard : Extract baseline hazards from CoxTimeVaryingFitter
scenarios : Macro scenario definitions and covariate path generation
cash_flow_engine : Core cash flow projection engine
risk_metrics : NPV, duration, convexity, WAL
"""

from .baseline_hazard import (
    extract_baseline_hazard,
    extract_baseline_hazards_both,
)

from .scenarios import (
    MacroScenario,
    create_base_scenario,
    apply_rate_shock,
    apply_hpi_shock,
    apply_unemployment_shock,
    scenario_to_covariate_matrix,
)

from .cash_flow_engine import (
    CashFlowConfig,
    MortgageCashFlowEngine,
)

from .risk_metrics import (
    compute_npv,
    compute_modified_duration,
    compute_effective_duration,
    compute_modified_convexity,
    compute_effective_convexity,
    compute_wal,
    compute_all_risk_metrics,
)

__all__ = [
    'extract_baseline_hazard',
    'extract_baseline_hazards_both',
    'MacroScenario',
    'create_base_scenario',
    'apply_rate_shock',
    'apply_hpi_shock',
    'apply_unemployment_shock',
    'scenario_to_covariate_matrix',
    'CashFlowConfig',
    'MortgageCashFlowEngine',
    'compute_npv',
    'compute_modified_duration',
    'compute_effective_duration',
    'compute_modified_convexity',
    'compute_effective_convexity',
    'compute_wal',
    'compute_all_risk_metrics',
]
