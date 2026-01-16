"""
Utility functions for data preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

from .columns import (
    ZERO_BALANCE_CODE_MAP,
    MISSING_VALUES,
    FICO_BANDS, FICO_LABELS,
    LTV_BANDS, LTV_LABELS,
    DTI_BANDS, DTI_LABELS,
    MATURITY_THRESHOLD_MONTHS,
)


def parse_date(date_str: str) -> Optional[pd.Timestamp]:
    """
    Parse YYYYMM date string to pandas Timestamp.

    Args:
        date_str: Date string in YYYYMM format

    Returns:
        Pandas Timestamp or None if invalid
    """
    if pd.isna(date_str) or date_str == '' or len(str(date_str)) != 6:
        return None
    try:
        return pd.to_datetime(str(date_str), format='%Y%m')
    except ValueError:
        return None


def extract_vintage_year(loan_sequence_number: str) -> Optional[int]:
    """
    Extract vintage year from loan sequence number.

    The loan_sequence_number format is FYYQ#xxxxxx where:
    - F = Freddie Mac identifier
    - YY = Two-digit origination year
    - Q# = Origination quarter (1-4)

    Args:
        loan_sequence_number: Loan sequence number string

    Returns:
        Year as integer or None
    """
    if pd.isna(loan_sequence_number) or len(str(loan_sequence_number)) < 4:
        return None
    try:
        year_digits = str(loan_sequence_number)[1:3]
        year = int(year_digits)
        # Handle Y2K: years > 90 are 1990s, otherwise 2000s
        return 1900 + year if year > 90 else 2000 + year
    except (ValueError, TypeError):
        return None


def map_event_type(zero_balance_code: str) -> str:
    """
    Map zero balance code to event type for survival analysis.

    Note: This function does NOT distinguish between prepay and matured.
    Use map_event_type_with_maturity() for that distinction.

    Args:
        zero_balance_code: Zero balance code from performance data

    Returns:
        Event type string: 'prepay', 'default', 'other', 'defect', or 'censored'
    """
    if pd.isna(zero_balance_code) or zero_balance_code == '':
        return 'censored'

    code = str(zero_balance_code).strip().zfill(2)
    return ZERO_BALANCE_CODE_MAP.get(code, 'other')


def map_event_type_with_maturity(
    zero_balance_code: str,
    loan_age: int,
    orig_loan_term: int
) -> str:
    """
    Map zero balance code to event type, distinguishing matured loans from prepayments.

    A loan with code '01' (Prepaid or Matured) is classified as:
    - 'matured' if loan_age >= orig_loan_term - MATURITY_THRESHOLD_MONTHS
    - 'prepay' otherwise

    Args:
        zero_balance_code: Zero balance code from performance data
        loan_age: Current loan age in months
        orig_loan_term: Original loan term in months

    Returns:
        Event type string: 'prepay', 'matured', 'default', 'other', 'defect', or 'censored'
    """
    if pd.isna(zero_balance_code) or zero_balance_code == '':
        return 'censored'

    code = str(zero_balance_code).strip().zfill(2)

    # Check for maturity: code 01 with loan age near original term
    if code == '01':
        if (pd.notna(loan_age) and pd.notna(orig_loan_term) and
                loan_age >= orig_loan_term - MATURITY_THRESHOLD_MONTHS):
            return 'matured'
        return 'prepay'

    return ZERO_BALANCE_CODE_MAP.get(code, 'other')


def create_event_indicator(zero_balance_code: str) -> int:
    """
    Create binary event indicator for survival analysis.

    Args:
        zero_balance_code: Zero balance code from performance data

    Returns:
        1 if event occurred, 0 if censored
    """
    if pd.isna(zero_balance_code) or zero_balance_code == '':
        return 0
    return 1


def clean_numeric(value, missing_codes: List = None) -> Optional[float]:
    """
    Clean numeric value, replacing missing codes with NaN.

    Args:
        value: Value to clean
        missing_codes: List of codes that represent missing values

    Returns:
        Cleaned numeric value or None
    """
    if pd.isna(value):
        return None

    if missing_codes and value in missing_codes:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def bin_fico(credit_score: float) -> Optional[str]:
    """
    Bin FICO score into categorical bands.

    Args:
        credit_score: FICO score (300-850)

    Returns:
        FICO band label or None
    """
    if pd.isna(credit_score) or credit_score < 300 or credit_score > 850:
        return None

    for i, (lower, upper) in enumerate(zip(FICO_BANDS[:-1], FICO_BANDS[1:])):
        if lower <= credit_score < upper:
            return FICO_LABELS[i]

    return FICO_LABELS[-1]


def bin_ltv(ltv: float) -> Optional[str]:
    """
    Bin LTV into categorical bands.

    Args:
        ltv: Loan-to-value ratio

    Returns:
        LTV band label or None
    """
    if pd.isna(ltv) or ltv <= 0:
        return None

    for i, (lower, upper) in enumerate(zip(LTV_BANDS[:-1], LTV_BANDS[1:])):
        if lower < ltv <= upper:
            return LTV_LABELS[i]

    return LTV_LABELS[-1]


def bin_dti(dti: float) -> Optional[str]:
    """
    Bin DTI into categorical bands.

    Args:
        dti: Debt-to-income ratio

    Returns:
        DTI band label or None
    """
    if pd.isna(dti) or dti <= 0:
        return None

    for i, (lower, upper) in enumerate(zip(DTI_BANDS[:-1], DTI_BANDS[1:])):
        if lower < dti <= upper:
            return DTI_LABELS[i]

    return DTI_LABELS[-1]


def get_max_delinquency(delinquency_history: pd.Series) -> int:
    """
    Get maximum delinquency status from loan history.

    Args:
        delinquency_history: Series of delinquency status values

    Returns:
        Maximum delinquency status (0 if always current)
    """
    numeric_values = []
    for val in delinquency_history:
        if pd.isna(val) or val == '':
            continue
        if val == 'RA':  # REO Acquisition
            numeric_values.append(99)
        else:
            try:
                numeric_values.append(int(val))
            except ValueError:
                continue

    return max(numeric_values) if numeric_values else 0


def ever_delinquent(delinquency_history: pd.Series, threshold: int = 2) -> bool:
    """
    Check if loan was ever delinquent above threshold.

    Args:
        delinquency_history: Series of delinquency status values
        threshold: Delinquency threshold (2 = 60+ days)

    Returns:
        True if ever delinquent above threshold
    """
    return get_max_delinquency(delinquency_history) >= threshold


def calculate_loan_term_years(orig_loan_term: int) -> Optional[float]:
    """
    Calculate loan term in years.

    Args:
        orig_loan_term: Original loan term in months

    Returns:
        Loan term in years or None
    """
    if pd.isna(orig_loan_term) or orig_loan_term <= 0:
        return None
    return orig_loan_term / 12


def has_mortgage_insurance(mi_pct: float) -> int:
    """
    Check if loan has mortgage insurance.

    Args:
        mi_pct: Mortgage insurance percentage

    Returns:
        1 if has MI, 0 otherwise
    """
    if pd.isna(mi_pct) or mi_pct <= 0:
        return 0
    return 1


def is_high_ltv(ltv: float, threshold: float = 80) -> int:
    """
    Check if loan has high LTV.

    Args:
        ltv: Loan-to-value ratio
        threshold: LTV threshold (default 80)

    Returns:
        1 if high LTV, 0 otherwise
    """
    if pd.isna(ltv):
        return 0
    return 1 if ltv > threshold else 0


def find_data_files(raw_data_path: Path) -> List[Tuple[int, Path, Path]]:
    """
    Find all origination and performance file pairs in raw data directory.

    Args:
        raw_data_path: Path to raw data directory

    Returns:
        List of tuples: (year, origination_file, performance_file)
    """
    file_pairs = []

    for year_dir in sorted(raw_data_path.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.startswith('sample_'):
            continue

        try:
            year = int(year_dir.name.split('_')[1])
        except (ValueError, IndexError):
            continue

        orig_file = year_dir / f'sample_orig_{year}.txt'
        perf_file = year_dir / f'sample_svcg_{year}.txt'

        if orig_file.exists() and perf_file.exists():
            file_pairs.append((year, orig_file, perf_file))

    return file_pairs


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that dataframe has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def print_summary_stats(df: pd.DataFrame, name: str = "Dataset"):
    """
    Print summary statistics for a survival dataset.

    Args:
        df: Survival dataset
        name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Total loans: {len(df):,}")

    if 'event' in df.columns:
        print(f"\nEvent distribution:")
        print(f"  Censored: {(df['event'] == 0).sum():,} ({(df['event'] == 0).mean()*100:.1f}%)")
        print(f"  Events:   {(df['event'] == 1).sum():,} ({(df['event'] == 1).mean()*100:.1f}%)")

    if 'event_type' in df.columns:
        print(f"\nEvent type distribution:")
        for event_type, count in df['event_type'].value_counts().items():
            print(f"  {event_type}: {count:,} ({count/len(df)*100:.1f}%)")

    if 'duration' in df.columns:
        print(f"\nDuration (months):")
        print(f"  Mean:   {df['duration'].mean():.1f}")
        print(f"  Median: {df['duration'].median():.1f}")
        print(f"  Min:    {df['duration'].min():.0f}")
        print(f"  Max:    {df['duration'].max():.0f}")

    if 'vintage_year' in df.columns:
        print(f"\nVintage year range: {df['vintage_year'].min()} - {df['vintage_year'].max()}")

    print(f"{'='*60}\n")
