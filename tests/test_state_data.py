"""
Test script to validate the content of state_hpi.parquet and
state_unemployment.parquet against data downloaded from FRED.

This script:
1. Loads the state-level parquet files
2. Downloads sample state data directly from FRED for comparison
3. Verifies that values match (accounting for resampling transformations)

Usage:
    python -m tests.test_state_data
    pytest tests/test_state_data.py -v
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from io import StringIO
from datetime import datetime
from typing import List, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
EXTERNAL_DATA_DIR = Path(__file__).parent.parent / 'data' / 'external'
TOLERANCE = 1e-6  # Tolerance for floating-point comparisons
NUM_RANDOM_RECORDS = 10  # Number of random records to test per state
NUM_SAMPLE_STATES = 5  # Number of states to sample for testing
RANDOM_SEED = 42

# Sample states to test (geographically diverse)
SAMPLE_STATES = ['CA', 'TX', 'NY', 'FL', 'IL']


def download_fred_csv(series_id: str, start_date: str = '1998-01-01',
                      end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Download FRED series via direct CSV URL.

    Parameters
    ----------
    series_id : str
        FRED series identifier
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame or None
        DataFrame with date index and series values
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}"
        f"&cosd={start_date}"
        f"&coed={end_date}"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), parse_dates=['observation_date'],
                         index_col='observation_date')
        df.index.name = 'DATE'
        df.columns = [series_id]

        # Replace '.' with NaN
        df = df.replace('.', pd.NA)
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')

        return df
    except Exception as e:
        print(f"  Warning: Could not download {series_id}: {e}")
        return None


def load_state_unemployment() -> Optional[pd.DataFrame]:
    """Load state unemployment parquet file."""
    path = EXTERNAL_DATA_DIR / 'state_unemployment.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def load_state_hpi() -> Optional[pd.DataFrame]:
    """Load state HPI parquet file."""
    path = EXTERNAL_DATA_DIR / 'state_hpi.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def compute_expected_unemployment(source_df: pd.DataFrame, target_date: pd.Timestamp,
                                   col: str) -> Optional[float]:
    """
    Compute expected unemployment value for a date.

    State unemployment is monthly, resampled with .first()
    """
    if source_df is None or source_df.empty:
        return None

    # Resample to month start
    resampled = source_df.resample('MS').first()

    if target_date in resampled.index:
        val = resampled.loc[target_date, col]
        return val if pd.notna(val) else None
    return None


def compute_expected_hpi(source_df: pd.DataFrame, target_date: pd.Timestamp,
                         col: str) -> Optional[float]:
    """
    Compute expected HPI value for a date.

    State HPI is quarterly, forward filled to monthly.
    """
    if source_df is None or source_df.empty:
        return None

    # Forward fill to monthly
    resampled = source_df.resample('MS').ffill()

    if target_date in resampled.index:
        val = resampled.loc[target_date, col]
        return val if pd.notna(val) else None
    return None


def values_match(actual: float, expected: float, tolerance: float = TOLERANCE) -> bool:
    """Check if two values match within tolerance."""
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False
    return abs(actual - expected) <= tolerance


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


def test_state_unemployment_integrity(unemp_df: pd.DataFrame, results: TestResult) -> None:
    """Test basic integrity of state unemployment data."""
    print("\n--- Testing State Unemployment Integrity ---")

    # Check index is sorted
    if unemp_df.index.is_monotonic_increasing:
        print("  PASS: Index is sorted")
        results.add_pass()
    else:
        results.add_fail("Unemployment index is not sorted")
        print("  FAIL: Index is not sorted")

    # Check index frequency
    freq = pd.infer_freq(unemp_df.index)
    if freq in ('MS', 'M'):
        print(f"  PASS: Index frequency is monthly ({freq})")
        results.add_pass()
    else:
        results.add_fail(f"Unemployment index frequency is {freq}, expected MS")
        print(f"  FAIL: Index frequency is {freq}")

    # Check column naming convention
    valid_cols = [c for c in unemp_df.columns if c.endswith('_unemployment')]
    if len(valid_cols) == len(unemp_df.columns):
        print(f"  PASS: All {len(valid_cols)} columns follow naming convention")
        results.add_pass()
    else:
        invalid = [c for c in unemp_df.columns if not c.endswith('_unemployment')]
        results.add_fail(f"Invalid column names: {invalid}")
        print(f"  FAIL: Invalid columns: {invalid}")

    # Check we have reasonable number of states
    if len(unemp_df.columns) >= 50:
        print(f"  PASS: Contains {len(unemp_df.columns)} states")
        results.add_pass()
    else:
        results.add_fail(f"Only {len(unemp_df.columns)} states, expected 50+")
        print(f"  FAIL: Only {len(unemp_df.columns)} states")


def test_state_hpi_integrity(hpi_df: pd.DataFrame, results: TestResult) -> None:
    """Test basic integrity of state HPI data."""
    print("\n--- Testing State HPI Integrity ---")

    # Check index is sorted
    if hpi_df.index.is_monotonic_increasing:
        print("  PASS: Index is sorted")
        results.add_pass()
    else:
        results.add_fail("HPI index is not sorted")
        print("  FAIL: Index is not sorted")

    # Check index frequency
    freq = pd.infer_freq(hpi_df.index)
    if freq in ('MS', 'M'):
        print(f"  PASS: Index frequency is monthly ({freq})")
        results.add_pass()
    else:
        results.add_fail(f"HPI index frequency is {freq}, expected MS")
        print(f"  FAIL: Index frequency is {freq}")

    # Check column naming convention
    valid_cols = [c for c in hpi_df.columns if c.endswith('_hpi')]
    if len(valid_cols) == len(hpi_df.columns):
        print(f"  PASS: All {len(valid_cols)} columns follow naming convention")
        results.add_pass()
    else:
        invalid = [c for c in hpi_df.columns if not c.endswith('_hpi')]
        results.add_fail(f"Invalid HPI column names: {invalid}")
        print(f"  FAIL: Invalid columns: {invalid}")

    # Check we have reasonable number of states
    if len(hpi_df.columns) >= 50:
        print(f"  PASS: Contains {len(hpi_df.columns)} states")
        results.add_pass()
    else:
        results.add_fail(f"Only {len(hpi_df.columns)} HPI states, expected 50+")
        print(f"  FAIL: Only {len(hpi_df.columns)} states")


def test_state_unemployment_values(unemp_df: pd.DataFrame, test_dates: List[pd.Timestamp],
                                    sample_states: List[str], results: TestResult) -> None:
    """Test state unemployment values against FRED source."""
    print("\n--- Testing State Unemployment Values ---")
    print(f"  Downloading and comparing {len(sample_states)} sample states...")

    for state in sample_states:
        col_name = f'{state}_unemployment'
        if col_name not in unemp_df.columns:
            print(f"  SKIP: {state} not in data")
            results.add_skip()
            continue

        # Download source data
        series_id = f'{state}UR'
        source_df = download_fred_csv(series_id)
        if source_df is None:
            print(f"  SKIP: Could not download {series_id}")
            results.add_skip()
            continue

        state_passed = 0
        state_failed = 0
        state_skipped = 0

        for date in test_dates:
            if date not in unemp_df.index:
                state_skipped += 1
                continue

            actual = unemp_df.loc[date, col_name]
            expected = compute_expected_unemployment(source_df, date, series_id)

            if expected is None:
                state_skipped += 1
                continue

            if values_match(actual, expected, tolerance=0.01):
                state_passed += 1
                results.add_pass()
            else:
                state_failed += 1
                msg = (f"{col_name} @ {date.strftime('%Y-%m-%d')}: "
                       f"actual={actual:.4f}, expected={expected:.4f}")
                results.add_fail(msg)

        status = "PASS" if state_failed == 0 else "FAIL"
        print(f"  {status}: {state} ({state_passed} passed, {state_failed} failed, {state_skipped} skipped)")


def test_state_hpi_values(hpi_df: pd.DataFrame, test_dates: List[pd.Timestamp],
                          sample_states: List[str], results: TestResult) -> None:
    """Test state HPI values against FRED source."""
    print("\n--- Testing State HPI Values ---")
    print(f"  Downloading and comparing {len(sample_states)} sample states...")

    for state in sample_states:
        col_name = f'{state}_hpi'
        if col_name not in hpi_df.columns:
            print(f"  SKIP: {state} not in data")
            results.add_skip()
            continue

        # Download source data
        series_id = f'{state}STHPI'
        source_df = download_fred_csv(series_id)
        if source_df is None:
            print(f"  SKIP: Could not download {series_id}")
            results.add_skip()
            continue

        state_passed = 0
        state_failed = 0
        state_skipped = 0

        for date in test_dates:
            if date not in hpi_df.index:
                state_skipped += 1
                continue

            actual = hpi_df.loc[date, col_name]
            expected = compute_expected_hpi(source_df, date, series_id)

            if expected is None:
                state_skipped += 1
                continue

            if values_match(actual, expected, tolerance=0.01):
                state_passed += 1
                results.add_pass()
            else:
                state_failed += 1
                msg = (f"{col_name} @ {date.strftime('%Y-%m-%d')}: "
                       f"actual={actual:.4f}, expected={expected:.4f}")
                results.add_fail(msg)

        status = "PASS" if state_failed == 0 else "FAIL"
        print(f"  {status}: {state} ({state_passed} passed, {state_failed} failed, {state_skipped} skipped)")


def test_hpi_forward_fill(hpi_df: pd.DataFrame, results: TestResult) -> None:
    """Test that HPI is properly forward-filled (quarterly -> monthly)."""
    print("\n--- Testing HPI Forward Fill ---")

    # For a sample state, check that values repeat for 3 months (quarterly data)
    sample_col = [c for c in hpi_df.columns if c.endswith('_hpi')][0]

    # Find sequences of repeated values
    values = hpi_df[sample_col].dropna()
    if len(values) < 12:
        print(f"  SKIP: Not enough data to test forward fill")
        results.add_skip()
        return

    # Check that we see some repeated consecutive values (from forward fill)
    consecutive_same = 0
    for i in range(1, len(values)):
        if values.iloc[i] == values.iloc[i-1]:
            consecutive_same += 1

    # With quarterly data forward-filled, we expect ~2/3 of values to be repeats
    repeat_ratio = consecutive_same / (len(values) - 1)
    if repeat_ratio > 0.5:  # Should be around 0.66 for quarterly->monthly
        print(f"  PASS: Forward fill detected (repeat ratio: {repeat_ratio:.2%})")
        results.add_pass()
    else:
        results.add_fail(f"Forward fill ratio too low: {repeat_ratio:.2%}")
        print(f"  FAIL: Forward fill ratio: {repeat_ratio:.2%}")


def test_unemployment_no_forward_fill(unemp_df: pd.DataFrame, results: TestResult) -> None:
    """Test that unemployment is NOT forward-filled (monthly data)."""
    print("\n--- Testing Unemployment Is Monthly (No Forward Fill) ---")

    sample_col = [c for c in unemp_df.columns if c.endswith('_unemployment')][0]

    values = unemp_df[sample_col].dropna()
    if len(values) < 12:
        print(f"  SKIP: Not enough data")
        results.add_skip()
        return

    # Check consecutive repeats - should be low for monthly data
    consecutive_same = 0
    for i in range(1, len(values)):
        if values.iloc[i] == values.iloc[i-1]:
            consecutive_same += 1

    repeat_ratio = consecutive_same / (len(values) - 1)
    # Unemployment rates often stay the same month-over-month, so ratio can be ~50%
    # But should be less than HPI's ~66% (quarterly forward-filled)
    if repeat_ratio < 0.55:
        print(f"  PASS: Monthly frequency confirmed (repeat ratio: {repeat_ratio:.2%})")
        results.add_pass()
    else:
        results.add_fail(f"Unexpected repeat ratio for monthly data: {repeat_ratio:.2%}")
        print(f"  FAIL: High repeat ratio: {repeat_ratio:.2%}")


def test_state_coverage(unemp_df: pd.DataFrame, hpi_df: pd.DataFrame,
                        results: TestResult) -> None:
    """Test that both datasets have the same states."""
    print("\n--- Testing State Coverage ---")

    unemp_states = {c.replace('_unemployment', '') for c in unemp_df.columns}
    hpi_states = {c.replace('_hpi', '') for c in hpi_df.columns}

    common = unemp_states & hpi_states
    only_unemp = unemp_states - hpi_states
    only_hpi = hpi_states - unemp_states

    print(f"  States in both: {len(common)}")
    if only_unemp:
        print(f"  Only in unemployment: {sorted(only_unemp)}")
    if only_hpi:
        print(f"  Only in HPI: {sorted(only_hpi)}")

    # At least 50 states should be in both
    if len(common) >= 50:
        print(f"  PASS: {len(common)} states in both datasets")
        results.add_pass()
    else:
        results.add_fail(f"Only {len(common)} states in both datasets")
        print(f"  FAIL: Only {len(common)} common states")


def run_tests(num_records: int = NUM_RANDOM_RECORDS,
              sample_states: List[str] = SAMPLE_STATES,
              random_seed: int = RANDOM_SEED,
              verbose: bool = True) -> TestResult:
    """
    Run all tests on state data files.

    Parameters
    ----------
    num_records : int
        Number of random records to test per state
    sample_states : list
        State abbreviations to test
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print detailed output

    Returns
    -------
    TestResult
        Test results
    """
    results = TestResult()

    print("="*60)
    print("STATE DATA VALIDATION TEST")
    print("="*60)

    # Load data
    unemp_df = load_state_unemployment()
    hpi_df = load_state_hpi()

    if unemp_df is None:
        print("\nERROR: state_unemployment.parquet not found")
        results.add_fail("state_unemployment.parquet not found")
        return results

    if hpi_df is None:
        print("\nERROR: state_hpi.parquet not found")
        results.add_fail("state_hpi.parquet not found")
        return results

    print(f"\nLoaded state_unemployment: {unemp_df.shape[0]} rows, {unemp_df.shape[1]} columns")
    print(f"  Date range: {unemp_df.index.min()} to {unemp_df.index.max()}")
    print(f"\nLoaded state_hpi: {hpi_df.shape[0]} rows, {hpi_df.shape[1]} columns")
    print(f"  Date range: {hpi_df.index.min()} to {hpi_df.index.max()}")

    # Select random test dates
    np.random.seed(random_seed)
    common_dates = unemp_df.index.intersection(hpi_df.index)
    if len(common_dates) == 0:
        results.add_fail("No common dates between datasets")
        return results

    test_indices = np.random.choice(len(common_dates),
                                     size=min(num_records, len(common_dates)),
                                     replace=False)
    test_dates = [common_dates[i] for i in sorted(test_indices)]

    print(f"\nTesting {len(test_dates)} random dates:")
    for d in test_dates[:5]:
        print(f"  - {d.strftime('%Y-%m-%d')}")
    if len(test_dates) > 5:
        print(f"  ... and {len(test_dates) - 5} more")

    print(f"\nSample states to test: {sample_states}")

    # Run integrity tests
    test_state_unemployment_integrity(unemp_df, results)
    test_state_hpi_integrity(hpi_df, results)
    test_state_coverage(unemp_df, hpi_df, results)

    # Run frequency tests
    test_unemployment_no_forward_fill(unemp_df, results)
    test_hpi_forward_fill(hpi_df, results)

    # Run value comparison tests (downloads from FRED)
    test_state_unemployment_values(unemp_df, test_dates, sample_states, results)
    test_state_hpi_values(hpi_df, test_dates, sample_states, results)

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
def test_state_data_integrity():
    """Pytest: Test state data integrity."""
    unemp_df = load_state_unemployment()
    hpi_df = load_state_hpi()

    assert unemp_df is not None, "state_unemployment.parquet not found"
    assert hpi_df is not None, "state_hpi.parquet not found"
    assert unemp_df.index.is_monotonic_increasing, "Unemployment index not sorted"
    assert hpi_df.index.is_monotonic_increasing, "HPI index not sorted"


def test_state_unemployment_values_pytest():
    """Pytest: Test state unemployment values."""
    results = TestResult()
    unemp_df = load_state_unemployment()
    assert unemp_df is not None

    np.random.seed(RANDOM_SEED)
    test_dates = list(np.random.choice(unemp_df.index, size=5, replace=False))

    test_state_unemployment_values(unemp_df, test_dates, SAMPLE_STATES[:2], results)
    assert results.failed == 0, f"Unemployment test failed: {results.failures}"


def test_state_hpi_values_pytest():
    """Pytest: Test state HPI values."""
    results = TestResult()
    hpi_df = load_state_hpi()
    assert hpi_df is not None

    np.random.seed(RANDOM_SEED)
    test_dates = list(np.random.choice(hpi_df.index, size=5, replace=False))

    test_state_hpi_values(hpi_df, test_dates, SAMPLE_STATES[:2], results)
    assert results.failed == 0, f"HPI test failed: {results.failures}"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test state-level FRED data')
    parser.add_argument('--num-records', type=int, default=NUM_RANDOM_RECORDS,
                        help=f'Number of random records to test (default: {NUM_RANDOM_RECORDS})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--states', type=str, nargs='+', default=SAMPLE_STATES,
                        help=f'States to test (default: {SAMPLE_STATES})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed failure output')
    args = parser.parse_args()

    results = run_tests(
        num_records=args.num_records,
        sample_states=args.states,
        random_seed=args.seed,
        verbose=not args.quiet
    )

    sys.exit(0 if results.failed == 0 else 1)
