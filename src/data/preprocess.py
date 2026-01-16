"""
Preprocess Freddie Mac loan data for survival analysis.

This script transforms raw Freddie Mac origination and performance data
into a survival analysis-ready format with:
- duration: time from origination to event (in months)
- event: binary indicator (1 = event occurred, 0 = censored)
- event_type: 'default', 'prepay', 'matured', 'other', or 'censored'

Usage:
    python -m src.data.preprocess --input data/raw --output data/processed
    python -m src.data.preprocess --input data/raw --output data/processed --year 2020
    python -m src.data.preprocess --input data/raw --output data/processed --by-vintage
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .columns import (
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS,
    ORIGINATION_DTYPES,
    PERFORMANCE_DTYPES,
    MISSING_VALUES,
    OCCUPANCY_STATUS_MAP,
    LOAN_PURPOSE_MAP,
    PROPERTY_TYPE_MAP,
    CHANNEL_MAP,
)
from .utils import (
    extract_vintage_year,
    map_event_type_with_maturity,
    bin_fico,
    bin_ltv,
    bin_dti,
    has_mortgage_insurance,
    is_high_ltv,
    calculate_loan_term_years,
    find_data_files,
    print_summary_stats,
    get_max_delinquency,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('preprocessing.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_origination_data(filepath: Path) -> pd.DataFrame:
    """
    Load origination data from pipe-delimited file.

    Args:
        filepath: Path to origination file

    Returns:
        DataFrame with origination data
    """
    logger.info(f"Loading origination data from {filepath}")

    df = pd.read_csv(
        filepath,
        sep='|',
        header=None,
        names=ORIGINATION_COLUMNS,
        dtype=ORIGINATION_DTYPES,
        low_memory=False,
        na_values=['', ' '],
    )

    logger.info(f"Loaded {len(df):,} origination records")
    return df


def load_performance_data(filepath: Path) -> pd.DataFrame:
    """
    Load performance data from pipe-delimited file.

    Args:
        filepath: Path to performance file

    Returns:
        DataFrame with performance data
    """
    logger.info(f"Loading performance data from {filepath}")

    df = pd.read_csv(
        filepath,
        sep='|',
        header=None,
        names=PERFORMANCE_COLUMNS,
        dtype=PERFORMANCE_DTYPES,
        low_memory=False,
        na_values=['', ' '],
    )

    logger.info(f"Loaded {len(df):,} performance records")
    return df


def aggregate_performance_data(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate performance data to get last observation per loan.

    Args:
        perf_df: Performance DataFrame

    Returns:
        DataFrame with one row per loan containing survival information
    """
    logger.info("Aggregating performance data...")

    # Sort by loan_age to ensure we get the last observation
    perf_df = perf_df.sort_values(['loan_sequence_number', 'loan_age'])

    # Get last record for each loan
    last_obs = perf_df.groupby('loan_sequence_number').agg({
        'loan_age': 'last',
        'zero_balance_code': 'last',
        'zero_balance_effective_date': 'last',
        'current_loan_delinquency_status': 'last',
        'current_actual_upb': 'last',
        'modification_flag': lambda x: 'Y' in x.values or 'P' in x.values,
    }).reset_index()

    # Rename columns
    last_obs = last_obs.rename(columns={
        'loan_age': 'duration',
        'modification_flag': 'ever_modified',
    })

    # Calculate max delinquency per loan
    max_delinq = perf_df.groupby('loan_sequence_number')['current_loan_delinquency_status'].apply(
        get_max_delinquency
    ).reset_index()
    max_delinq.columns = ['loan_sequence_number', 'max_delinquency']

    last_obs = last_obs.merge(max_delinq, on='loan_sequence_number', how='left')

    logger.info(f"Aggregated to {len(last_obs):,} unique loans")
    return last_obs


def create_survival_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create survival analysis variables from aggregated performance data.

    Note: event_type and event indicator are created later in
    merge_and_create_survival_dataset after merging with origination data
    (to distinguish matured from prepaid, and treat matured as censored).

    Args:
        df: Aggregated performance DataFrame

    Returns:
        DataFrame with preliminary survival variables
    """
    logger.info("Creating survival variables...")

    # Handle missing duration (set to 0 if missing)
    df['duration'] = df['duration'].fillna(0).astype(int)

    # Create ever_60_plus_delinquent indicator
    df['ever_60_plus_delinquent'] = (df['max_delinquency'] >= 2).astype(int)

    # Create ever_90_plus_delinquent indicator
    df['ever_90_plus_delinquent'] = (df['max_delinquency'] >= 3).astype(int)

    return df


def clean_origination_data(orig_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean origination data: handle missing values and create derived features.

    Args:
        orig_df: Raw origination DataFrame

    Returns:
        Cleaned origination DataFrame
    """
    logger.info("Cleaning origination data...")

    df = orig_df.copy()

    # Handle missing values for numeric columns
    for col, missing_codes in MISSING_VALUES.items():
        if col in df.columns:
            df[col] = df[col].replace(missing_codes, np.nan)

    # Convert credit_score to numeric, handling missing
    if 'credit_score' in df.columns:
        df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
        df.loc[df['credit_score'] == 9999, 'credit_score'] = np.nan

    # Convert LTV/CLTV/DTI to numeric
    for col in ['orig_ltv', 'orig_cltv', 'orig_dti']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == 999, col] = np.nan

    # Convert MI percentage
    if 'mi_pct' in df.columns:
        df['mi_pct'] = pd.to_numeric(df['mi_pct'], errors='coerce')
        df.loc[df['mi_pct'] == 999, 'mi_pct'] = np.nan

    # Extract vintage year from loan sequence number
    df['vintage_year'] = df['loan_sequence_number'].apply(extract_vintage_year)

    # Create FICO bands
    df['fico_band'] = df['credit_score'].apply(bin_fico)

    # Create LTV bands
    df['ltv_band'] = df['orig_ltv'].apply(bin_ltv)

    # Create DTI bands
    df['dti_band'] = df['orig_dti'].apply(bin_dti)

    # Create binary indicators
    df['has_mi'] = df['mi_pct'].apply(has_mortgage_insurance)
    df['is_high_ltv'] = df['orig_ltv'].apply(is_high_ltv)
    df['loan_term_years'] = df['orig_loan_term'].apply(calculate_loan_term_years)

    # Map categorical variables
    df['occupancy_status_desc'] = df['occupancy_status'].map(OCCUPANCY_STATUS_MAP)
    df['loan_purpose_desc'] = df['loan_purpose'].map(LOAN_PURPOSE_MAP)
    df['property_type_desc'] = df['property_type'].map(PROPERTY_TYPE_MAP)
    df['channel_desc'] = df['channel'].map(CHANNEL_MAP)

    # Create first time homebuyer indicator
    df['is_first_time_homebuyer'] = (df['first_time_homebuyer'] == 'Y').astype(int)

    return df


def merge_and_create_survival_dataset(
    orig_df: pd.DataFrame,
    perf_agg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge origination and aggregated performance data to create final survival dataset.

    Args:
        orig_df: Cleaned origination DataFrame
        perf_agg_df: Aggregated performance DataFrame with survival variables

    Returns:
        Final survival analysis dataset
    """
    logger.info("Merging origination and performance data...")

    # Merge on loan_sequence_number
    df = perf_agg_df.merge(
        orig_df,
        on='loan_sequence_number',
        how='inner'
    )

    logger.info(f"Merged dataset has {len(df):,} loans")

    # Create event_type after merge (need both duration and orig_loan_term to distinguish matured from prepaid)
    df['event_type'] = df.apply(
        lambda row: map_event_type_with_maturity(
            row['zero_balance_code'],
            row['duration'],
            row['orig_loan_term']
        ),
        axis=1
    )

    # Create binary event indicator (matured treated as censored)
    # Events: prepay, default, other, defect
    # Censored: censored, matured
    df['event'] = df['event_type'].apply(
        lambda x: 0 if x in ['censored', 'matured'] else 1
    )

    # Select and order final columns
    survival_columns = [
        # Identifiers
        'loan_sequence_number',

        # Survival variables
        'duration',
        'event',
        'event_type',

        # Time variables
        'vintage_year',

        # Core covariates
        'credit_score',
        'orig_ltv',
        'orig_cltv',
        'orig_dti',
        'orig_upb',
        'orig_interest_rate',
        'orig_loan_term',

        # Categorical covariates
        'occupancy_status',
        'loan_purpose',
        'property_type',
        'property_state',
        'channel',
        'num_borrowers',
        'num_units',

        # Binary indicators
        'is_first_time_homebuyer',
        'has_mi',
        'is_high_ltv',
        'ever_modified',
        'ever_60_plus_delinquent',
        'ever_90_plus_delinquent',

        # Binned variables
        'fico_band',
        'ltv_band',
        'dti_band',

        # Descriptive mappings
        'occupancy_status_desc',
        'loan_purpose_desc',
        'property_type_desc',
        'channel_desc',

        # Additional info
        'loan_term_years',
        'mi_pct',
        'max_delinquency',
    ]

    # Keep only columns that exist
    available_columns = [col for col in survival_columns if col in df.columns]
    df = df[available_columns]

    return df


def process_single_year(
    orig_file: Path,
    perf_file: Path,
    year: int
) -> pd.DataFrame:
    """
    Process a single year of data.

    Args:
        orig_file: Path to origination file
        perf_file: Path to performance file
        year: Vintage year

    Returns:
        Survival dataset for the year
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing year {year}")
    logger.info(f"{'='*60}")

    # Load data
    orig_df = load_origination_data(orig_file)
    perf_df = load_performance_data(perf_file)

    # Aggregate performance data
    perf_agg = aggregate_performance_data(perf_df)

    # Create survival variables
    perf_agg = create_survival_variables(perf_agg)

    # Clean origination data
    orig_clean = clean_origination_data(orig_df)

    # Merge and create final dataset
    survival_df = merge_and_create_survival_dataset(orig_clean, perf_agg)

    logger.info(f"Year {year}: {len(survival_df):,} loans processed")

    return survival_df


def save_datasets(
    df: pd.DataFrame,
    output_path: Path,
    save_by_event_type: bool = True,
    save_by_vintage: bool = False
) -> None:
    """
    Save survival datasets to disk.

    Args:
        df: Full survival dataset
        output_path: Output directory path
        save_by_event_type: Whether to save separate files by event type
        save_by_vintage: Whether to save separate files per vintage year
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    csv_file = output_path / 'survival_data.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved {len(df):,} loans to {csv_file}")

    # Try to save parquet (optional, requires pyarrow or fastparquet)
    try:
        parquet_file = output_path / 'survival_data.parquet'
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved parquet to {parquet_file}")
    except ImportError:
        logger.warning("Parquet not saved (pyarrow/fastparquet not installed)")

    if save_by_event_type:
        # Save default-only dataset (for cause-specific hazard models)
        default_df = df[df['event_type'].isin(['default', 'censored'])].copy()
        default_df['event'] = (default_df['event_type'] == 'default').astype(int)
        default_file = output_path / 'survival_data_default.csv'
        default_df.to_csv(default_file, index=False)
        logger.info(f"Saved default dataset ({len(default_df):,} loans) to {default_file}")

        # Save prepayment-only dataset
        prepay_df = df[df['event_type'].isin(['prepay', 'censored'])].copy()
        prepay_df['event'] = (prepay_df['event_type'] == 'prepay').astype(int)
        prepay_file = output_path / 'survival_data_prepay.csv'
        prepay_df.to_csv(prepay_file, index=False)
        logger.info(f"Saved prepayment dataset ({len(prepay_df):,} loans) to {prepay_file}")

    if save_by_vintage:
        save_datasets_by_vintage(df, output_path, save_by_event_type)


def save_datasets_by_vintage(
    df: pd.DataFrame,
    output_path: Path,
    save_by_event_type: bool = True
) -> None:
    """
    Save separate survival datasets for each vintage year.

    Args:
        df: Full survival dataset
        output_path: Output directory path
        save_by_event_type: Whether to save separate files by event type for each vintage
    """
    vintage_dir = output_path / 'by_vintage'
    vintage_dir.mkdir(parents=True, exist_ok=True)

    vintages = sorted(df['vintage_year'].dropna().unique())
    logger.info(f"\nSaving datasets by vintage for {len(vintages)} vintages...")

    for vintage in vintages:
        vintage_int = int(vintage)
        vintage_path = vintage_dir / f'vintage_{vintage_int}'
        vintage_path.mkdir(parents=True, exist_ok=True)

        vintage_df = df[df['vintage_year'] == vintage].copy()

        # Save full vintage dataset
        csv_file = vintage_path / f'survival_data_{vintage_int}.csv'
        vintage_df.to_csv(csv_file, index=False)
        logger.info(f"Vintage {vintage_int}: saved {len(vintage_df):,} loans to {vintage_path}")

        # Try to save parquet (optional)
        try:
            parquet_file = vintage_path / f'survival_data_{vintage_int}.parquet'
            vintage_df.to_parquet(parquet_file, index=False)
        except ImportError:
            pass  # Skip parquet if not available

        if save_by_event_type:
            # Save default-only dataset for this vintage
            default_df = vintage_df[vintage_df['event_type'].isin(['default', 'censored'])].copy()
            default_df['event'] = (default_df['event_type'] == 'default').astype(int)
            default_file = vintage_path / f'survival_data_{vintage_int}_default.csv'
            default_df.to_csv(default_file, index=False)

            # Save prepayment-only dataset for this vintage
            prepay_df = vintage_df[vintage_df['event_type'].isin(['prepay', 'censored'])].copy()
            prepay_df['event'] = (prepay_df['event_type'] == 'prepay').astype(int)
            prepay_file = vintage_path / f'survival_data_{vintage_int}_prepay.csv'
            prepay_df.to_csv(prepay_file, index=False)

    logger.info(f"Saved {len(vintages)} vintage-specific datasets to {vintage_dir}")


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess Freddie Mac data for survival analysis'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output directory'
    )
    parser.add_argument(
        '--year', '-y',
        type=int,
        default=None,
        help='Specific year to process (e.g., 2020). If not specified, process all years.'
    )
    parser.add_argument(
        '--years',
        type=str,
        default=None,
        help='Range of years to process (e.g., "2015-2020")'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Do not create separate files by event type'
    )
    parser.add_argument(
        '--by-vintage',
        action='store_true',
        help='Create separate survival files for each vintage year'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    # Find all data file pairs
    file_pairs = find_data_files(input_path)

    if not file_pairs:
        logger.error(f"No data files found in {input_path}")
        logger.error("Expected structure: data/raw/sample_YYYY/sample_orig_YYYY.txt")
        return

    logger.info(f"Found {len(file_pairs)} year(s) of data")

    # Filter by year if specified
    if args.year:
        file_pairs = [(y, o, p) for y, o, p in file_pairs if y == args.year]
        if not file_pairs:
            logger.error(f"No data found for year {args.year}")
            return
    elif args.years:
        start_year, end_year = map(int, args.years.split('-'))
        file_pairs = [(y, o, p) for y, o, p in file_pairs if start_year <= y <= end_year]

    # Process each year
    all_data = []

    for year, orig_file, perf_file in file_pairs:
        try:
            survival_df = process_single_year(orig_file, perf_file, year)
            all_data.append(survival_df)
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            continue

    if not all_data:
        logger.error("No data was successfully processed")
        return

    # Combine all years
    logger.info("\nCombining all years...")
    final_df = pd.concat(all_data, ignore_index=True)

    # Save datasets
    save_datasets(
        final_df,
        output_path,
        save_by_event_type=not args.no_split,
        save_by_vintage=args.by_vintage
    )

    # Print summary statistics
    print_summary_stats(final_df, "Final Survival Dataset")


if __name__ == '__main__':
    main()
