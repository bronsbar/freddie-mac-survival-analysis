"""
Preprocess Freddie Mac loan data for survival analysis.

This script transforms raw Freddie Mac origination and performance data
into a survival analysis-ready format with:
- duration: time from origination to event (in months)
- event: binary indicator (1 = event occurred, 0 = censored)
- event_type: 'default', 'prepay', or 'censored'
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# Freddie Mac origination file columns
ORIGINATION_COLS = [
    'credit_score', 'first_payment_date', 'first_time_homebuyer_flag',
    'maturity_date', 'msa', 'mi_percentage', 'num_units',
    'occupancy_status', 'cltv', 'dti', 'upb', 'ltv', 'orig_interest_rate',
    'channel', 'prepay_penalty_flag', 'amortization_type', 'property_state',
    'property_type', 'postal_code', 'loan_sequence_number', 'loan_purpose',
    'orig_loan_term', 'num_borrowers', 'seller_name', 'servicer_name',
    'super_conforming_flag', 'pre_harp_loan_seq_num', 'program_indicator',
    'harp_indicator', 'valuation_method', 'io_indicator'
]

# Freddie Mac performance file columns
PERFORMANCE_COLS = [
    'loan_sequence_number', 'monthly_reporting_period', 'current_upb',
    'current_loan_delinquency_status', 'loan_age', 'remaining_months_to_maturity',
    'repurchase_flag', 'modification_flag', 'zero_balance_code',
    'zero_balance_effective_date', 'current_interest_rate', 'current_deferred_upb',
    'ddlpi', 'mi_recoveries', 'net_sale_proceeds', 'non_mi_recoveries',
    'expenses', 'legal_costs', 'maintenance_costs', 'taxes_insurance',
    'misc_expenses', 'actual_loss', 'modification_cost', 'step_modification_flag',
    'deferred_payment_modification', 'eltv', 'zero_balance_removal_upb',
    'delinquent_accrued_interest', 'delinquency_due_to_disaster',
    'borrower_assistance_plan'
]

# Zero balance codes meaning
ZERO_BALANCE_CODES = {
    '01': 'prepay',      # Prepaid or Matured
    '02': 'prepay',      # Third Party Sale
    '03': 'default',     # Short Sale
    '06': 'default',     # Repurchased
    '09': 'default',     # REO Disposition
    '15': 'default',     # Note Sale
    '16': 'default',     # Reperforming Loan Sale
}


def load_origination_data(filepath: Path) -> pd.DataFrame:
    """Load origination data from pipe-delimited file."""
    df = pd.read_csv(
        filepath,
        sep='|',
        header=None,
        names=ORIGINATION_COLS,
        low_memory=False
    )
    return df


def load_performance_data(filepath: Path) -> pd.DataFrame:
    """Load performance data from pipe-delimited file."""
    df = pd.read_csv(
        filepath,
        sep='|',
        header=None,
        names=PERFORMANCE_COLS,
        low_memory=False
    )
    return df


def determine_event_type(zero_balance_code: str) -> str:
    """Map zero balance code to event type."""
    if pd.isna(zero_balance_code):
        return 'censored'
    return ZERO_BALANCE_CODES.get(str(int(zero_balance_code)).zfill(2), 'censored')


def create_survival_dataset(
    origination_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    default_threshold: int = 3
) -> pd.DataFrame:
    """
    Create survival analysis dataset from raw Freddie Mac data.

    Parameters
    ----------
    origination_df : pd.DataFrame
        Loan origination characteristics
    performance_df : pd.DataFrame
        Monthly loan performance data
    default_threshold : int
        Number of months delinquent to consider as default (default=3 for 90+ days)

    Returns
    -------
    pd.DataFrame
        Survival dataset with columns:
        - loan_id: unique loan identifier
        - duration: time to event in months
        - event: 1 if event occurred, 0 if censored
        - event_type: 'default', 'prepay', or 'censored'
        - [covariates from origination data]
    """

    # Get the last observation for each loan
    last_obs = performance_df.sort_values('loan_age').groupby('loan_sequence_number').last()

    # Determine event type and timing
    survival_data = []

    for loan_id, row in tqdm(last_obs.iterrows(), total=len(last_obs), desc="Processing loans"):
        duration = row['loan_age'] if pd.notna(row['loan_age']) else 0

        # Determine event type
        event_type = determine_event_type(row.get('zero_balance_code'))

        # Alternative: check delinquency status for default
        if event_type == 'censored':
            delinq = row.get('current_loan_delinquency_status', 0)
            if pd.notna(delinq) and str(delinq) not in ['0', 'XX', '']:
                try:
                    if int(delinq) >= default_threshold:
                        event_type = 'default'
                except ValueError:
                    pass

        event = 0 if event_type == 'censored' else 1

        survival_data.append({
            'loan_id': loan_id,
            'duration': duration,
            'event': event,
            'event_type': event_type
        })

    survival_df = pd.DataFrame(survival_data)

    # Merge with origination covariates
    origination_df = origination_df.set_index('loan_sequence_number')
    survival_df = survival_df.set_index('loan_id').join(origination_df)
    survival_df = survival_df.reset_index().rename(columns={'index': 'loan_id'})

    return survival_df


def clean_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode covariates for modeling."""

    # Select key covariates for survival analysis
    key_covariates = [
        'loan_id', 'duration', 'event', 'event_type',
        'credit_score', 'ltv', 'cltv', 'dti', 'upb',
        'orig_interest_rate', 'orig_loan_term', 'num_borrowers',
        'first_time_homebuyer_flag', 'occupancy_status',
        'channel', 'property_type', 'loan_purpose', 'property_state'
    ]

    # Keep only available columns
    available_cols = [c for c in key_covariates if c in df.columns]
    df = df[available_cols].copy()

    # Handle missing values in numeric columns
    numeric_cols = ['credit_score', 'ltv', 'cltv', 'dti', 'orig_interest_rate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def main():
    parser = argparse.ArgumentParser(description='Preprocess Freddie Mac data for survival analysis')
    parser.add_argument('--input', type=str, required=True, help='Path to raw data directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--year', type=str, default=None, help='Specific year to process (e.g., 2015)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing data from {input_path}")

    # Find origination and performance files
    # Freddie Mac naming convention: historical_data_YYYY.txt, historical_data_time_YYYY.txt
    orig_files = list(input_path.glob('**/historical_data_20*.txt'))
    perf_files = list(input_path.glob('**/historical_data_time_20*.txt'))

    if not orig_files or not perf_files:
        # Try sample data naming convention
        orig_files = list(input_path.glob('**/sample_orig*.txt'))
        perf_files = list(input_path.glob('**/sample_svcg*.txt'))

    if not orig_files:
        print("No origination files found. Please check the input directory structure.")
        print("Expected files: historical_data_YYYY.txt or sample_orig*.txt")
        return

    print(f"Found {len(orig_files)} origination files and {len(perf_files)} performance files")

    # Process each file pair
    all_survival_data = []

    for orig_file in tqdm(orig_files, desc="Processing files"):
        year = orig_file.stem.split('_')[-1]

        if args.year and year != args.year:
            continue

        # Find matching performance file
        perf_file = None
        for pf in perf_files:
            if year in pf.stem:
                perf_file = pf
                break

        if perf_file is None:
            print(f"Warning: No performance file found for {orig_file}")
            continue

        print(f"\nProcessing {year}...")

        # Load data
        orig_df = load_origination_data(orig_file)
        perf_df = load_performance_data(perf_file)

        # Create survival dataset
        survival_df = create_survival_dataset(orig_df, perf_df)
        survival_df = clean_covariates(survival_df)
        survival_df['vintage'] = year

        all_survival_data.append(survival_df)

    if all_survival_data:
        # Combine all years
        final_df = pd.concat(all_survival_data, ignore_index=True)

        # Save to parquet for efficient storage
        output_file = output_path / 'survival_data.parquet'
        final_df.to_parquet(output_file, index=False)
        print(f"\nSaved {len(final_df)} loans to {output_file}")

        # Also save as CSV for convenience
        csv_file = output_path / 'survival_data.csv'
        final_df.to_csv(csv_file, index=False)
        print(f"Saved CSV to {csv_file}")

        # Print summary statistics
        print("\n=== Dataset Summary ===")
        print(f"Total loans: {len(final_df)}")
        print(f"\nEvent distribution:")
        print(final_df['event_type'].value_counts())
        print(f"\nDuration statistics (months):")
        print(final_df['duration'].describe())
    else:
        print("No data processed. Please check input files.")


if __name__ == '__main__':
    main()
