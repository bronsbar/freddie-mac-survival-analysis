"""
Column definitions for Freddie Mac Single-Family Loan-Level Dataset.

Based on the Single-Family Loan-Level Dataset General User Guide (October 2025).
"""

# Origination file column names (32 columns)
ORIGINATION_COLUMNS = [
    'credit_score',
    'first_payment_date',
    'first_time_homebuyer',
    'maturity_date',
    'msa',
    'mi_pct',
    'num_units',
    'occupancy_status',
    'orig_cltv',
    'orig_dti',
    'orig_upb',
    'orig_ltv',
    'orig_interest_rate',
    'channel',
    'ppm_flag',
    'amortization_type',
    'property_state',
    'property_type',
    'postal_code',
    'loan_sequence_number',
    'loan_purpose',
    'orig_loan_term',
    'num_borrowers',
    'seller_name',
    'servicer_name',
    'super_conforming_flag',
    'pre_relief_refi_loan_seq',
    'special_eligibility_program',
    'relief_refinance_indicator',
    'property_valuation_method',
    'interest_only_indicator',
    'mi_cancellation_indicator',
]

# Performance file column names (32 columns)
PERFORMANCE_COLUMNS = [
    'loan_sequence_number',
    'monthly_reporting_period',
    'current_actual_upb',
    'current_loan_delinquency_status',
    'loan_age',
    'remaining_months_to_maturity',
    'defect_settlement_date',
    'modification_flag',
    'zero_balance_code',
    'zero_balance_effective_date',
    'current_interest_rate',
    'current_non_interest_bearing_upb',
    'ddlpi',
    'mi_recoveries',
    'net_sale_proceeds',
    'non_mi_recoveries',
    'total_expenses',
    'legal_costs',
    'maintenance_costs',
    'taxes_insurance',
    'misc_expenses',
    'actual_loss',
    'cumulative_mod_cost',
    'interest_rate_step_indicator',
    'payment_deferral_flag',
    'eltv',
    'zero_balance_removal_upb',
    'delinquent_accrued_interest',
    'delinquency_due_to_disaster',
    'borrower_assistance_status',
    'current_month_mod_cost',
    'interest_bearing_upb',
]

# Data types for origination columns
ORIGINATION_DTYPES = {
    'credit_score': 'Int64',
    'first_payment_date': 'str',
    'first_time_homebuyer': 'str',
    'maturity_date': 'str',
    'msa': 'str',
    'mi_pct': 'Int64',
    'num_units': 'Int64',
    'occupancy_status': 'str',
    'orig_cltv': 'Int64',
    'orig_dti': 'Int64',
    'orig_upb': 'Int64',
    'orig_ltv': 'Int64',
    'orig_interest_rate': 'float64',
    'channel': 'str',
    'ppm_flag': 'str',
    'amortization_type': 'str',
    'property_state': 'str',
    'property_type': 'str',
    'postal_code': 'str',
    'loan_sequence_number': 'str',
    'loan_purpose': 'str',
    'orig_loan_term': 'Int64',
    'num_borrowers': 'Int64',
    'seller_name': 'str',
    'servicer_name': 'str',
    'super_conforming_flag': 'str',
    'pre_relief_refi_loan_seq': 'str',
    'special_eligibility_program': 'str',
    'relief_refinance_indicator': 'str',
    'property_valuation_method': 'str',
    'interest_only_indicator': 'str',
    'mi_cancellation_indicator': 'str',
}

# Data types for performance columns
PERFORMANCE_DTYPES = {
    'loan_sequence_number': 'str',
    'monthly_reporting_period': 'str',
    'current_actual_upb': 'float64',
    'current_loan_delinquency_status': 'str',
    'loan_age': 'Int64',
    'remaining_months_to_maturity': 'Int64',
    'defect_settlement_date': 'str',
    'modification_flag': 'str',
    'zero_balance_code': 'str',
    'zero_balance_effective_date': 'str',
    'current_interest_rate': 'float64',
    'current_non_interest_bearing_upb': 'float64',
    'ddlpi': 'str',
    'mi_recoveries': 'float64',
    'net_sale_proceeds': 'str',
    'non_mi_recoveries': 'float64',
    'total_expenses': 'float64',
    'legal_costs': 'float64',
    'maintenance_costs': 'float64',
    'taxes_insurance': 'float64',
    'misc_expenses': 'float64',
    'actual_loss': 'float64',
    'cumulative_mod_cost': 'float64',
    'interest_rate_step_indicator': 'str',
    'payment_deferral_flag': 'str',
    'eltv': 'str',
    'zero_balance_removal_upb': 'float64',
    'delinquent_accrued_interest': 'float64',
    'delinquency_due_to_disaster': 'str',
    'borrower_assistance_status': 'str',
    'current_month_mod_cost': 'float64',
    'interest_bearing_upb': 'float64',
}

# Zero balance code mapping for event types
# Note: Code '01' (Prepaid or Matured) requires additional logic to distinguish
# between prepayment and maturity - see map_event_type_with_maturity() in utils.py
ZERO_BALANCE_CODE_MAP = {
    '01': 'prepay',      # Prepaid or Matured - needs loan_age check for maturity
    '02': 'default',     # Third Party Sale (Foreclosure)
    '03': 'default',     # Short Sale or Charge Off
    '09': 'default',     # REO Disposition
    '15': 'other',       # Whole Loan Sale
    '16': 'other',       # Reperforming Loan Securitization
    '96': 'defect',      # Defect prior to termination
}

# Maturity threshold: if loan_age >= orig_loan_term - MATURITY_THRESHOLD_MONTHS,
# and zero_balance_code is '01', the loan is considered matured (not prepaid)
MATURITY_THRESHOLD_MONTHS = 3

# Missing value codes
MISSING_VALUES = {
    'credit_score': [9999],
    'orig_ltv': [999],
    'orig_cltv': [999],
    'orig_dti': [999],
    'mi_pct': [999],
    'num_units': [99],
    'num_borrowers': [99],
    'eltv': ['999'],
}

# Categorical value mappings
OCCUPANCY_STATUS_MAP = {
    'P': 'primary_residence',
    'S': 'second_home',
    'I': 'investment',
    '9': 'unknown',
}

LOAN_PURPOSE_MAP = {
    'P': 'purchase',
    'C': 'cash_out_refi',
    'N': 'no_cash_out_refi',
    'R': 'refi_not_specified',
    '9': 'unknown',
}

PROPERTY_TYPE_MAP = {
    'SF': 'single_family',
    'CO': 'condo',
    'PU': 'pud',
    'MH': 'manufactured',
    'CP': 'coop',
    '99': 'unknown',
}

CHANNEL_MAP = {
    'R': 'retail',
    'B': 'broker',
    'C': 'correspondent',
    'T': 'tpo_not_specified',
    '9': 'unknown',
}

# FICO score bands for binning
FICO_BANDS = [0, 620, 680, 740, 780, 850]
FICO_LABELS = ['<620', '620-679', '680-739', '740-779', '780+']

# LTV bands for binning
LTV_BANDS = [0, 60, 70, 80, 90, 95, 200]
LTV_LABELS = ['<=60', '61-70', '71-80', '81-90', '91-95', '>95']

# DTI bands for binning
DTI_BANDS = [0, 20, 30, 40, 50, 100]
DTI_LABELS = ['<=20', '21-30', '31-40', '41-50', '>50']
