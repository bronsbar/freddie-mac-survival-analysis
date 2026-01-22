"""
Test script to validate the content of survival_data_blumenstock.parquet
against the original Freddie Mac CSV files.

This script:
1. Loads the processed survival dataset
2. For a sample of records, traces back to original vintage files
3. Verifies loan-level variables match origination data
4. Verifies behavioral variables are correctly calculated

Usage:
    python -m tests.test_survival_data_blumenstock
    pytest tests/test_survival_data_blumenstock.py -v
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.columns import (
    ORIGINATION_COLUMNS, ORIGINATION_DTYPES,
    PERFORMANCE_COLUMNS, PERFORMANCE_DTYPES,
)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
TOLERANCE = 1e-6
NUM_RANDOM_RECORDS = 20
NUM_VINTAGES_TO_TEST = 3
RANDOM_SEED = 42


def load_survival_data() -> Optional[pd.DataFrame]:
    """Load the processed survival dataset."""
    path = PROCESSED_DATA_DIR / 'survival_data_blumenstock.parquet'
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_origination_data(vintage: int) -> Optional[pd.DataFrame]:
    """Load origination data for a vintage."""
    pattern = f'sample_{vintage}/sample_orig_{vintage}.txt'
    files = list(RAW_DATA_DIR.glob(f'**/{pattern}'))

    if not files:
        return None

    df = pd.read_csv(
        files[0], sep='|', names=ORIGINATION_COLUMNS,
        dtype=ORIGINATION_DTYPES, na_values=['', ' ']
    )
    return df


def load_performance_data(vintage: int) -> Optional[pd.DataFrame]:
    """Load performance data for a vintage."""
    pattern = f'sample_{vintage}/sample_svcg_{vintage}.txt'
    files = list(RAW_DATA_DIR.glob(f'**/{pattern}'))

    if not files:
        return None

    df = pd.read_csv(
        files[0], sep='|', names=PERFORMANCE_COLUMNS,
        dtype=PERFORMANCE_DTYPES, na_values=['', ' ']
    )
    return df


def values_match(actual, expected, tolerance: float = TOLERANCE) -> bool:
    """Check if two values match within tolerance."""
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


class TestResult:
    """Container for test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[str] = []

    def add_pass(self):
        self.passed += 1

    def add_fail(self, message: str):
        self.failed += 1
        self.failures.append(message)

    def add_skip(self):
        self.skipped += 1

    def summary(self) -> str:
        total = self.passed + self.failed + self.skipped
        status = "PASSED" if self.failed == 0 else "FAILED"
        return (f"\n{'='*60}\n"
                f"TEST SUMMARY: {status}\n"
                f"{'='*60}\n"
                f"  Passed:  {self.passed}\n"
                f"  Failed:  {self.failed}\n"
                f"  Skipped: {self.skipped}\n"
                f"  Total:   {total}\n")


def test_dataset_integrity(survival_df: pd.DataFrame, results: TestResult) -> None:
    """Test basic integrity of the survival dataset."""
    print("\n--- Testing Dataset Integrity ---")

    # Check required columns exist
    required_cols = ['loan_sequence_number', 'vintage_year', 'duration', 'event_code']
    missing = [c for c in required_cols if c not in survival_df.columns]
    if not missing:
        print(f"  PASS: All required columns present")
        results.add_pass()
    else:
        results.add_fail(f"Missing required columns: {missing}")
        print(f"  FAIL: Missing columns: {missing}")

    # Check event codes are valid (0, 1, 2)
    valid_events = {0, 1, 2}
    actual_events = set(survival_df['event_code'].unique())
    if actual_events.issubset(valid_events):
        print(f"  PASS: Event codes valid: {sorted(actual_events)}")
        results.add_pass()
    else:
        invalid = actual_events - valid_events
        results.add_fail(f"Invalid event codes: {invalid}")
        print(f"  FAIL: Invalid event codes: {invalid}")

    # Check duration is non-negative
    if (survival_df['duration'] >= 0).all():
        print(f"  PASS: All durations non-negative")
        results.add_pass()
    else:
        neg_count = (survival_df['duration'] < 0).sum()
        results.add_fail(f"{neg_count} records have negative duration")
        print(f"  FAIL: {neg_count} negative durations")

    # Check no duplicate loan_sequence_numbers
    n_unique = survival_df['loan_sequence_number'].nunique()
    n_total = len(survival_df)
    if n_unique == n_total:
        print(f"  PASS: No duplicate loans ({n_unique:,} unique)")
        results.add_pass()
    else:
        results.add_fail(f"Duplicate loans: {n_total - n_unique:,}")
        print(f"  FAIL: {n_total - n_unique:,} duplicate loans")


def test_origination_variables(survival_df: pd.DataFrame, results: TestResult) -> None:
    """Test that origination variables match source files."""
    print("\n--- Testing Origination Variables ---")

    # Variable mapping: survival_df column -> origination column
    var_mapping = {
        'int_rate': 'orig_interest_rate',
        'orig_upb': 'orig_upb',
        'fico_score': 'credit_score',
        'dti_r': 'orig_dti',
        'ltv_r': 'orig_ltv',
    }

    # Get unique vintages and sample some
    vintages = survival_df['vintage_year'].unique()
    np.random.seed(RANDOM_SEED)
    test_vintages = np.random.choice(vintages, size=min(NUM_VINTAGES_TO_TEST, len(vintages)), replace=False)

    print(f"  Testing vintages: {sorted(test_vintages)}")

    for vintage in test_vintages:
        orig_df = load_origination_data(vintage)
        if orig_df is None:
            print(f"  SKIP: Vintage {vintage} origination file not found")
            results.add_skip()
            continue

        # Get sample loans from this vintage
        vintage_loans = survival_df[survival_df['vintage_year'] == vintage]
        sample_loans = vintage_loans.sample(
            n=min(NUM_RANDOM_RECORDS, len(vintage_loans)),
            random_state=RANDOM_SEED
        )

        vintage_passed = 0
        vintage_failed = 0
        vintage_skipped = 0

        for _, loan in sample_loans.iterrows():
            loan_id = loan['loan_sequence_number']
            orig_row = orig_df[orig_df['loan_sequence_number'] == loan_id]

            if orig_row.empty:
                vintage_skipped += 1
                continue

            orig_row = orig_row.iloc[0]

            for surv_col, orig_col in var_mapping.items():
                if surv_col not in survival_df.columns or orig_col not in orig_df.columns:
                    vintage_skipped += 1
                    continue

                actual = loan[surv_col]
                expected = orig_row[orig_col]

                if values_match(actual, expected, tolerance=0.01):
                    vintage_passed += 1
                    results.add_pass()
                else:
                    vintage_failed += 1
                    msg = (f"Vintage {vintage}, Loan {loan_id}, {surv_col}: "
                           f"actual={actual}, expected={expected}")
                    results.add_fail(msg)

        status = "PASS" if vintage_failed == 0 else "FAIL"
        print(f"  {status}: Vintage {vintage} ({vintage_passed} passed, {vintage_failed} failed, {vintage_skipped} skipped)")


def test_behavioral_variables(survival_df: pd.DataFrame, results: TestResult) -> None:
    """Test that behavioral variables are within expected ranges."""
    print("\n--- Testing Behavioral Variables ---")

    behavioral_vars = {
        't_act_12m': (0, 12),      # Times current in last 12 months
        't_del_30d_12m': (0, 12),  # Times 30d delinquent in last 12 months
        't_del_60d_12m': (0, 12),  # Times 60d delinquent in last 12 months
        'bal_repaid': (0, 100),    # Balance repaid percentage
    }

    for var, (min_val, max_val) in behavioral_vars.items():
        if var not in survival_df.columns:
            print(f"  SKIP: {var} not in dataset")
            results.add_skip()
            continue

        values = survival_df[var].dropna()

        in_range = ((values >= min_val) & (values <= max_val)).all()
        actual_min = values.min()
        actual_max = values.max()

        if in_range:
            print(f"  PASS: {var} in range [{min_val}, {max_val}] (actual: [{actual_min:.2f}, {actual_max:.2f}])")
            results.add_pass()
        else:
            out_of_range = ((values < min_val) | (values > max_val)).sum()
            results.add_fail(f"{var}: {out_of_range} values outside [{min_val}, {max_val}]")
            print(f"  FAIL: {var} has {out_of_range} values outside range (actual: [{actual_min:.2f}, {actual_max:.2f}])")


def test_macro_variables(survival_df: pd.DataFrame, results: TestResult) -> None:
    """Test that macro variables are present and have reasonable values."""
    print("\n--- Testing Macro Variables (Coverage) ---")

    macro_vars = [
        'hpi_st_d_t_o',
        'ppi_c_FRMA',
        'TB10Y_d_t_o',
        'FRMA30Y_d_t_o',
        'ppi_o_FRMA',
        'hpi_st_log12m',
        'hpi_r_st_us',
        'st_unemp_r12m',
        'st_unemp_r3m',
        'TB10Y_r12m',
        'T10Y3MM',
        'T10Y3MM_r12m',
    ]

    for var in macro_vars:
        if var not in survival_df.columns:
            print(f"  SKIP: {var} not in dataset")
            results.add_skip()
            continue

        coverage = survival_df[var].notna().mean()

        if coverage >= 0.9:
            mean_val = survival_df[var].mean()
            std_val = survival_df[var].std()
            print(f"  PASS: {var} coverage {coverage:.1%}, mean={mean_val:.3f}, std={std_val:.3f}")
            results.add_pass()
        else:
            results.add_fail(f"{var}: only {coverage:.1%} coverage")


def test_macro_variables_against_fred(survival_df: pd.DataFrame, results: TestResult) -> None:
    """Test that macro variables match the FRED source files."""
    print("\n--- Testing Macro Variables Against FRED Sources ---")

    EXTERNAL_DATA_DIR = PROJECT_ROOT / 'data' / 'external'

    # Load FRED source data
    fred_panel_path = EXTERNAL_DATA_DIR / 'fred_monthly_panel.parquet'
    state_unemp_path = EXTERNAL_DATA_DIR / 'state_unemployment.parquet'
    state_hpi_path = EXTERNAL_DATA_DIR / 'state_hpi.parquet'

    # Load national macro data
    if not fred_panel_path.exists():
        print("  SKIP: fred_monthly_panel.parquet not found")
        results.add_skip()
        return

    fred_panel = pd.read_parquet(fred_panel_path)
    fred_panel.index.name = 'date'
    fred_panel = fred_panel.reset_index()
    fred_panel['date'] = pd.to_datetime(fred_panel['date'])
    fred_panel['year_month'] = fred_panel['date'].dt.to_period('M')

    # Load state unemployment
    state_unemp = None
    if state_unemp_path.exists():
        state_unemp = pd.read_parquet(state_unemp_path)
        state_unemp.index.name = 'date'
        state_unemp = state_unemp.reset_index()
        state_unemp['date'] = pd.to_datetime(state_unemp['date'])
        state_unemp['year_month'] = state_unemp['date'].dt.to_period('M')

    # Load state HPI
    state_hpi = None
    if state_hpi_path.exists():
        state_hpi = pd.read_parquet(state_hpi_path)
        state_hpi.index.name = 'date'
        state_hpi = state_hpi.reset_index()
        state_hpi['date'] = pd.to_datetime(state_hpi['date'])
        state_hpi['year_month'] = state_hpi['date'].dt.to_period('M')

    # Sample records to test
    np.random.seed(RANDOM_SEED)
    sample_df = survival_df.sample(n=min(NUM_RANDOM_RECORDS * 5, len(survival_df)), random_state=RANDOM_SEED)

    # === Test 1: National macro variables ===
    # Test T10Y3MM which is calculated as DGS10 - DGS3MO
    print("  Testing national macro variables...")
    national_passed = 0
    national_failed = 0
    national_skipped = 0

    # Calculate T10Y3MM from FRED data for comparison
    if 'DGS10' in fred_panel.columns and 'DGS3MO' in fred_panel.columns:
        fred_panel['T10Y3MM_calc'] = fred_panel['DGS10'] - fred_panel['DGS3MO']

    for _, row in sample_df.head(NUM_RANDOM_RECORDS).iterrows():
        year_month = row.get('year_month')
        if pd.isna(year_month):
            national_skipped += 1
            continue

        # Match by period
        fred_row = fred_panel[fred_panel['year_month'] == year_month]
        if fred_row.empty:
            national_skipped += 1
            continue
        fred_row = fred_row.iloc[0]

        # Test T10Y3MM (calculated from DGS10 - DGS3MO)
        if 'T10Y3MM' in survival_df.columns and 'T10Y3MM_calc' in fred_panel.columns:
            actual = row['T10Y3MM']
            expected = fred_row['T10Y3MM_calc']
            if pd.notna(actual) and pd.notna(expected):
                if values_match(actual, expected, tolerance=0.01):
                    national_passed += 1
                    results.add_pass()
                else:
                    national_failed += 1
                    results.add_fail(f"T10Y3MM @ {year_month}: actual={actual:.4f}, expected={expected:.4f}")
            else:
                national_skipped += 1
        else:
            national_skipped += 1

    status = "PASS" if national_failed == 0 else "FAIL"
    print(f"    {status}: National vars ({national_passed} passed, {national_failed} failed, {national_skipped} skipped)")

    # === Test 2: State unemployment variables ===
    if state_unemp is not None:
        print("  Testing state unemployment variables...")
        unemp_passed = 0
        unemp_failed = 0
        unemp_skipped = 0

        for _, row in sample_df.head(NUM_RANDOM_RECORDS).iterrows():
            year_month = row.get('year_month')
            state = row.get('property_state')

            if pd.isna(year_month) or pd.isna(state):
                unemp_skipped += 1
                continue

            # Get state unemployment for this month
            unemp_col = f'{state}_unemployment'
            if unemp_col not in state_unemp.columns:
                unemp_skipped += 1
                continue

            unemp_row = state_unemp[state_unemp['year_month'] == year_month]
            if unemp_row.empty:
                unemp_skipped += 1
                continue

            expected_unemp = unemp_row[unemp_col].iloc[0]

            # The survival data has derived variables (log returns), not raw unemployment
            # We can at least check that state_unemployment column exists and is reasonable
            if 'state_unemployment' in survival_df.columns:
                actual = row.get('state_unemployment')
                if pd.notna(actual) and pd.notna(expected_unemp):
                    if values_match(actual, expected_unemp, tolerance=0.1):
                        unemp_passed += 1
                        results.add_pass()
                    else:
                        unemp_failed += 1
                        results.add_fail(f"state_unemployment @ {year_month}, {state}: actual={actual:.2f}, expected={expected_unemp:.2f}")
                else:
                    unemp_skipped += 1
            else:
                unemp_skipped += 1

        status = "PASS" if unemp_failed == 0 else "FAIL"
        print(f"    {status}: State unemployment ({unemp_passed} passed, {unemp_failed} failed, {unemp_skipped} skipped)")
    else:
        print("  SKIP: State unemployment data not loaded")
        results.add_skip()

    # === Test 3: State HPI variables ===
    if state_hpi is not None:
        print("  Testing state HPI variables...")
        hpi_passed = 0
        hpi_failed = 0
        hpi_skipped = 0

        for _, row in sample_df.head(NUM_RANDOM_RECORDS).iterrows():
            year_month = row.get('year_month')
            state = row.get('property_state')

            if pd.isna(year_month) or pd.isna(state):
                hpi_skipped += 1
                continue

            # Get state HPI for this month
            hpi_col = f'{state}_hpi'
            if hpi_col not in state_hpi.columns:
                hpi_skipped += 1
                continue

            hpi_row = state_hpi[state_hpi['year_month'] == year_month]
            if hpi_row.empty:
                hpi_skipped += 1
                continue

            expected_hpi = hpi_row[hpi_col].iloc[0]

            # Check state_hpi column
            if 'state_hpi' in survival_df.columns:
                actual = row.get('state_hpi')
                if pd.notna(actual) and pd.notna(expected_hpi):
                    if values_match(actual, expected_hpi, tolerance=0.1):
                        hpi_passed += 1
                        results.add_pass()
                    else:
                        hpi_failed += 1
                        results.add_fail(f"state_hpi @ {year_month}, {state}: actual={actual:.2f}, expected={expected_hpi:.2f}")
                else:
                    hpi_skipped += 1
            else:
                hpi_skipped += 1

        status = "PASS" if hpi_failed == 0 else "FAIL"
        print(f"    {status}: State HPI ({hpi_passed} passed, {hpi_failed} failed, {hpi_skipped} skipped)")
    else:
        print("  SKIP: State HPI data not loaded")
        results.add_skip()


def test_event_duration_consistency(survival_df: pd.DataFrame, results: TestResult) -> None:
    """Test that event codes and durations are consistent."""
    print("\n--- Testing Event-Duration Consistency ---")

    # Defaulted loans should have duration > 0 (takes time to become delinquent)
    defaults = survival_df[survival_df['event_code'] == 2]
    if len(defaults) > 0:
        defaults_with_duration = (defaults['duration'] > 0).mean()
        if defaults_with_duration >= 0.99:
            print(f"  PASS: {defaults_with_duration:.1%} of defaults have duration > 0")
            results.add_pass()
        else:
            results.add_fail(f"Only {defaults_with_duration:.1%} of defaults have duration > 0")
            print(f"  FAIL: {defaults_with_duration:.1%} defaults have duration > 0")
    else:
        print(f"  SKIP: No defaults in dataset")
        results.add_skip()

    # Check duration statistics by event type
    print("\n  Duration statistics by event type:")
    for event_code, event_name in [(0, 'Censored'), (1, 'Prepay'), (2, 'Default')]:
        subset = survival_df[survival_df['event_code'] == event_code]
        if len(subset) > 0:
            mean_dur = subset['duration'].mean()
            median_dur = subset['duration'].median()
            print(f"    {event_name}: n={len(subset):,}, mean={mean_dur:.1f}, median={median_dur:.1f}")


def run_tests(num_records: int = NUM_RANDOM_RECORDS,
              num_vintages: int = NUM_VINTAGES_TO_TEST,
              random_seed: int = RANDOM_SEED,
              verbose: bool = True) -> TestResult:
    """
    Run all tests on survival_data_blumenstock.parquet.

    Parameters
    ----------
    num_records : int
        Number of random records to test per vintage
    num_vintages : int
        Number of vintages to test
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print detailed output

    Returns
    -------
    TestResult
        Test results
    """
    global NUM_RANDOM_RECORDS, NUM_VINTAGES_TO_TEST, RANDOM_SEED
    NUM_RANDOM_RECORDS = num_records
    NUM_VINTAGES_TO_TEST = num_vintages
    RANDOM_SEED = random_seed

    results = TestResult()

    print("="*60)
    print("SURVIVAL DATA BLUMENSTOCK VALIDATION TEST")
    print("="*60)

    # Load data
    survival_df = load_survival_data()
    if survival_df is None:
        print("\nERROR: survival_data_blumenstock.parquet not found")
        print(f"  Expected at: {PROCESSED_DATA_DIR / 'survival_data_blumenstock.parquet'}")
        results.add_fail("Dataset file not found")
        return results

    print(f"\nLoaded dataset: {len(survival_df):,} rows, {len(survival_df.columns)} columns")
    print(f"Vintages: {survival_df['vintage_year'].min()} - {survival_df['vintage_year'].max()}")
    print(f"Events: Censored={sum(survival_df['event_code']==0):,}, "
          f"Prepay={sum(survival_df['event_code']==1):,}, "
          f"Default={sum(survival_df['event_code']==2):,}")

    # Run tests
    test_dataset_integrity(survival_df, results)
    test_origination_variables(survival_df, results)
    test_behavioral_variables(survival_df, results)
    test_macro_variables(survival_df, results)
    test_macro_variables_against_fred(survival_df, results)
    test_event_duration_consistency(survival_df, results)

    # Print summary
    print(results.summary())

    if results.failures and verbose:
        print("FAILURES:")
        for f in results.failures[:20]:
            print(f"  - {f}")
        if len(results.failures) > 20:
            print(f"  ... and {len(results.failures) - 20} more")

    return results


# Pytest-compatible test functions
def test_survival_data_integrity():
    """Pytest: Test dataset integrity."""
    survival_df = load_survival_data()
    assert survival_df is not None, "Dataset not found"
    assert 'loan_sequence_number' in survival_df.columns
    assert 'duration' in survival_df.columns
    assert 'event_code' in survival_df.columns
    assert survival_df['event_code'].isin([0, 1, 2]).all()


def test_survival_data_origination():
    """Pytest: Test origination variables."""
    results = TestResult()
    survival_df = load_survival_data()
    assert survival_df is not None
    test_origination_variables(survival_df, results)
    assert results.failed == 0, f"Origination test failed: {results.failures}"


def test_survival_data_behavioral():
    """Pytest: Test behavioral variables."""
    results = TestResult()
    survival_df = load_survival_data()
    assert survival_df is not None
    test_behavioral_variables(survival_df, results)
    assert results.failed == 0, f"Behavioral test failed: {results.failures}"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test survival data Blumenstock')
    parser.add_argument('--num-records', type=int, default=NUM_RANDOM_RECORDS,
                        help=f'Number of random records to test per vintage (default: {NUM_RANDOM_RECORDS})')
    parser.add_argument('--num-vintages', type=int, default=NUM_VINTAGES_TO_TEST,
                        help=f'Number of vintages to test (default: {NUM_VINTAGES_TO_TEST})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed failure output')
    args = parser.parse_args()

    results = run_tests(
        num_records=args.num_records,
        num_vintages=args.num_vintages,
        random_seed=args.seed,
        verbose=not args.quiet
    )

    sys.exit(0 if results.failed == 0 else 1)
