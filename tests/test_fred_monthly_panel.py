"""
Test script to validate the content of fred_monthly_panel.parquet
against the original downloaded CSV files.

This script:
1. Loads the combined monthly panel
2. Loads individual series CSV files
3. For multiple random records, verifies that:
   - Raw series values match (accounting for resampling transformations)
   - Derived columns are calculated correctly

Usage:
    python -m tests.test_fred_monthly_panel
    pytest tests/test_fred_monthly_panel.py -v
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
EXTERNAL_DATA_DIR = Path(__file__).parent.parent / 'data' / 'external'
TOLERANCE = 1e-6  # Tolerance for floating-point comparisons
NUM_RANDOM_RECORDS = 50  # Number of random records to test
RANDOM_SEED = 42

# Series metadata (from download_fred.py)
FRED_SERIES = {
    'UNRATE': {'frequency': 'monthly'},
    'MORTGAGE30US': {'frequency': 'weekly'},
    'CSUSHPINSA': {'frequency': 'monthly'},
    'USSTHPI': {'frequency': 'quarterly'},
    'FEDFUNDS': {'frequency': 'monthly'},
    'T10Y2Y': {'frequency': 'daily'},
    'DGS3MO': {'frequency': 'daily'},
    'DGS10': {'frequency': 'daily'},
    'DGS5': {'frequency': 'daily'},
    'DGS2': {'frequency': 'daily'},
    'MORTGAGE15US': {'frequency': 'weekly'},
    'DPRIME': {'frequency': 'monthly'},
    'BAA10Y': {'frequency': 'daily'},
    'HOUST': {'frequency': 'monthly'},
    'HSN1F': {'frequency': 'monthly'},
    'EXHOSLUSM495S': {'frequency': 'monthly'},
    'PERMIT': {'frequency': 'monthly'},
    'MSACSR': {'frequency': 'monthly'},
    'A191RL1Q225SBEA': {'frequency': 'quarterly'},
    'UMCSENT': {'frequency': 'monthly'},
    'CPIAUCSL': {'frequency': 'monthly'},
    'PCEPI': {'frequency': 'monthly'},
    'DSPIC96': {'frequency': 'monthly'},
}


def load_monthly_panel() -> pd.DataFrame:
    """Load the combined monthly panel."""
    panel_path = EXTERNAL_DATA_DIR / 'fred_monthly_panel.parquet'
    if not panel_path.exists():
        raise FileNotFoundError(f"Monthly panel not found: {panel_path}")

    df = pd.read_parquet(panel_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def load_series_csv(series_id: str) -> Optional[pd.DataFrame]:
    """Load an individual series CSV file."""
    csv_path = EXTERNAL_DATA_DIR / f'{series_id.lower()}.csv'
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = 'DATE'
    return df


def compute_monthly_value(series_df: pd.DataFrame, target_date: pd.Timestamp,
                          frequency: str) -> Optional[float]:
    """
    Compute what the monthly panel value should be for a given date.

    Parameters
    ----------
    series_df : pd.DataFrame
        The original series data
    series_id : str
        Series identifier
    target_date : pd.Timestamp
        Target month start date
    frequency : str
        Original frequency: 'daily', 'weekly', 'monthly', 'quarterly'

    Returns
    -------
    float or None
        Expected value in the monthly panel
    """
    if series_df is None or series_df.empty:
        return None

    # Ensure datetime index
    if not isinstance(series_df.index, pd.DatetimeIndex):
        series_df.index = pd.to_datetime(series_df.index)

    col = series_df.columns[0]

    if frequency in ('daily', 'weekly'):
        # Monthly average of all observations in the month
        month_start = target_date
        month_end = (target_date + pd.offsets.MonthEnd(1))
        mask = (series_df.index >= month_start) & (series_df.index <= month_end)
        month_data = series_df.loc[mask, col]
        if month_data.empty or month_data.isna().all():
            return None
        return month_data.mean()

    elif frequency == 'quarterly':
        # Forward fill: find the most recent quarterly value
        quarterly_resampled = series_df.resample('MS').ffill()
        if target_date in quarterly_resampled.index:
            val = quarterly_resampled.loc[target_date, col]
            return val if pd.notna(val) else None
        return None

    else:  # monthly
        # Resample to month start and take first value
        monthly_resampled = series_df.resample('MS').first()
        if target_date in monthly_resampled.index:
            val = monthly_resampled.loc[target_date, col]
            return val if pd.notna(val) else None
        return None


def compute_derived_value(panel_df: pd.DataFrame, column: str,
                          target_date: pd.Timestamp) -> Optional[float]:
    """
    Compute what a derived column value should be.

    Parameters
    ----------
    panel_df : pd.DataFrame
        The monthly panel data
    column : str
        Derived column name
    target_date : pd.Timestamp
        Target date

    Returns
    -------
    float or None
        Expected derived value
    """
    idx = panel_df.index.get_loc(target_date)

    if column == 'hpi_yoy_change':
        # YoY percent change of CSUSHPINSA
        if 'CSUSHPINSA' not in panel_df.columns:
            return None
        if idx < 12:
            return None
        current = panel_df['CSUSHPINSA'].iloc[idx]
        past = panel_df['CSUSHPINSA'].iloc[idx - 12]
        if pd.isna(current) or pd.isna(past) or past == 0:
            return None
        return ((current - past) / past) * 100

    elif column == 'hpi_mom_change':
        # MoM percent change of CSUSHPINSA
        if 'CSUSHPINSA' not in panel_df.columns:
            return None
        if idx < 1:
            return None
        current = panel_df['CSUSHPINSA'].iloc[idx]
        past = panel_df['CSUSHPINSA'].iloc[idx - 1]
        if pd.isna(current) or pd.isna(past) or past == 0:
            return None
        return ((current - past) / past) * 100

    elif column == 'fhfa_hpi_yoy_change':
        # YoY percent change of USSTHPI (quarterly, so 4 periods)
        if 'USSTHPI' not in panel_df.columns:
            return None
        if idx < 4:
            return None
        current = panel_df['USSTHPI'].iloc[idx]
        past = panel_df['USSTHPI'].iloc[idx - 4]
        if pd.isna(current) or pd.isna(past) or past == 0:
            return None
        return ((current - past) / past) * 100

    elif column == 'unemp_yoy_change':
        # YoY difference in UNRATE
        if 'UNRATE' not in panel_df.columns:
            return None
        if idx < 12:
            return None
        current = panel_df['UNRATE'].iloc[idx]
        past = panel_df['UNRATE'].iloc[idx - 12]
        if pd.isna(current) or pd.isna(past):
            return None
        return current - past

    elif column == 'inflation_yoy':
        # YoY percent change of CPIAUCSL
        if 'CPIAUCSL' not in panel_df.columns:
            return None
        if idx < 12:
            return None
        current = panel_df['CPIAUCSL'].iloc[idx]
        past = panel_df['CPIAUCSL'].iloc[idx - 12]
        if pd.isna(current) or pd.isna(past) or past == 0:
            return None
        return ((current - past) / past) * 100

    elif column == 'mortgage_rate_mom_change':
        # MoM difference in MORTGAGE30US
        if 'MORTGAGE30US' not in panel_df.columns:
            return None
        if idx < 1:
            return None
        current = panel_df['MORTGAGE30US'].iloc[idx]
        past = panel_df['MORTGAGE30US'].iloc[idx - 1]
        if pd.isna(current) or pd.isna(past):
            return None
        return current - past

    elif column == 'mortgage_rate_yoy_change':
        # YoY difference in MORTGAGE30US
        if 'MORTGAGE30US' not in panel_df.columns:
            return None
        if idx < 12:
            return None
        current = panel_df['MORTGAGE30US'].iloc[idx]
        past = panel_df['MORTGAGE30US'].iloc[idx - 12]
        if pd.isna(current) or pd.isna(past):
            return None
        return current - past

    elif column == 'mortgage_spread':
        # MORTGAGE30US - DGS10
        if 'MORTGAGE30US' not in panel_df.columns or 'DGS10' not in panel_df.columns:
            return None
        m30 = panel_df['MORTGAGE30US'].iloc[idx]
        t10 = panel_df['DGS10'].iloc[idx]
        if pd.isna(m30) or pd.isna(t10):
            return None
        return m30 - t10

    elif column == 'yield_curve_slope':
        # DGS10 - DGS2
        if 'DGS10' not in panel_df.columns or 'DGS2' not in panel_df.columns:
            return None
        t10 = panel_df['DGS10'].iloc[idx]
        t2 = panel_df['DGS2'].iloc[idx]
        if pd.isna(t10) or pd.isna(t2):
            return None
        return t10 - t2

    elif column == 'gdp_growth':
        # Just a rename of A191RL1Q225SBEA
        if 'A191RL1Q225SBEA' not in panel_df.columns:
            return None
        return panel_df['A191RL1Q225SBEA'].iloc[idx]

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


def test_raw_series(panel_df: pd.DataFrame, test_dates: List[pd.Timestamp],
                    results: TestResult) -> None:
    """Test that raw series values in the panel match the source CSVs."""
    print("\n--- Testing Raw Series Values ---")

    for series_id, meta in FRED_SERIES.items():
        if series_id not in panel_df.columns:
            print(f"  SKIP: {series_id} not in panel")
            results.add_skip()
            continue

        series_df = load_series_csv(series_id)
        if series_df is None:
            print(f"  SKIP: {series_id} CSV not found")
            results.add_skip()
            continue

        frequency = meta['frequency']
        series_passed = 0
        series_failed = 0
        series_skipped = 0

        for date in test_dates:
            if date not in panel_df.index:
                series_skipped += 1
                continue

            actual = panel_df.loc[date, series_id]
            expected = compute_monthly_value(series_df, date, frequency)

            if expected is None:
                series_skipped += 1
                continue

            # Use larger tolerance for averaged values
            tol = TOLERANCE if frequency == 'monthly' else 0.01

            if values_match(actual, expected, tolerance=tol):
                series_passed += 1
                results.add_pass()
            else:
                series_failed += 1
                msg = (f"{series_id} @ {date.strftime('%Y-%m-%d')}: "
                       f"actual={actual:.6f}, expected={expected:.6f}, "
                       f"diff={abs(actual - expected):.6f}")
                results.add_fail(msg)

        status = "PASS" if series_failed == 0 else "FAIL"
        print(f"  {status}: {series_id} ({series_passed} passed, {series_failed} failed, {series_skipped} skipped)")


def test_derived_columns(panel_df: pd.DataFrame, test_dates: List[pd.Timestamp],
                         results: TestResult) -> None:
    """Test that derived columns are calculated correctly."""
    print("\n--- Testing Derived Columns ---")

    derived_columns = [
        'hpi_yoy_change',
        'hpi_mom_change',
        'fhfa_hpi_yoy_change',
        'unemp_yoy_change',
        'inflation_yoy',
        'mortgage_rate_mom_change',
        'mortgage_rate_yoy_change',
        'mortgage_spread',
        'yield_curve_slope',
        'gdp_growth',
    ]

    for column in derived_columns:
        if column not in panel_df.columns:
            print(f"  SKIP: {column} not in panel")
            results.add_skip()
            continue

        col_passed = 0
        col_failed = 0
        col_skipped = 0

        for date in test_dates:
            if date not in panel_df.index:
                col_skipped += 1
                continue

            actual = panel_df.loc[date, column]
            expected = compute_derived_value(panel_df, column, date)

            if expected is None:
                col_skipped += 1
                continue

            # Use larger tolerance for derived values (chained calculations)
            tol = 0.001

            if values_match(actual, expected, tolerance=tol):
                col_passed += 1
                results.add_pass()
            else:
                col_failed += 1
                msg = (f"{column} @ {date.strftime('%Y-%m-%d')}: "
                       f"actual={actual:.6f}, expected={expected:.6f}, "
                       f"diff={abs(actual - expected):.6f}")
                results.add_fail(msg)

        status = "PASS" if col_failed == 0 else "FAIL"
        print(f"  {status}: {column} ({col_passed} passed, {col_failed} failed, {col_skipped} skipped)")


def test_panel_integrity(panel_df: pd.DataFrame, results: TestResult) -> None:
    """Test basic integrity of the panel."""
    print("\n--- Testing Panel Integrity ---")

    # Check index is sorted
    if panel_df.index.is_monotonic_increasing:
        print("  PASS: Index is sorted")
        results.add_pass()
    else:
        results.add_fail("Index is not sorted")
        print("  FAIL: Index is not sorted")

    # Check index is monthly frequency
    freq_check = pd.infer_freq(panel_df.index)
    if freq_check in ('MS', 'M'):
        print(f"  PASS: Index frequency is monthly ({freq_check})")
        results.add_pass()
    else:
        results.add_fail(f"Index frequency is {freq_check}, expected MS or M")
        print(f"  FAIL: Index frequency is {freq_check}")

    # Check no duplicate dates
    if not panel_df.index.has_duplicates:
        print("  PASS: No duplicate dates")
        results.add_pass()
    else:
        results.add_fail("Duplicate dates found in index")
        print("  FAIL: Duplicate dates found")

    # Check expected columns exist
    expected_cols = ['UNRATE', 'MORTGAGE30US', 'DGS10']
    missing_cols = [c for c in expected_cols if c not in panel_df.columns]
    if not missing_cols:
        print(f"  PASS: All core columns present")
        results.add_pass()
    else:
        results.add_fail(f"Missing core columns: {missing_cols}")
        print(f"  FAIL: Missing columns: {missing_cols}")


def run_tests(num_records: int = NUM_RANDOM_RECORDS,
              random_seed: int = RANDOM_SEED,
              verbose: bool = True) -> TestResult:
    """
    Run all tests on the fred_monthly_panel.

    Parameters
    ----------
    num_records : int
        Number of random records to test
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
    print("FRED MONTHLY PANEL VALIDATION TEST")
    print("="*60)

    # Load panel
    try:
        panel_df = load_monthly_panel()
        print(f"\nLoaded panel: {panel_df.shape[0]} rows, {panel_df.shape[1]} columns")
        print(f"Date range: {panel_df.index.min()} to {panel_df.index.max()}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        results.add_fail(str(e))
        return results

    # Select random test dates (avoiding first 12 months for derived columns)
    np.random.seed(random_seed)
    valid_dates = panel_df.index[12:]  # Skip first 12 months
    test_indices = np.random.choice(len(valid_dates),
                                     size=min(num_records, len(valid_dates)),
                                     replace=False)
    test_dates = [valid_dates[i] for i in sorted(test_indices)]

    print(f"\nTesting {len(test_dates)} random records:")
    for d in test_dates:
        print(f"  - {d.strftime('%Y-%m-%d')}")

    # Run tests
    test_panel_integrity(panel_df, results)
    test_raw_series(panel_df, test_dates, results)
    test_derived_columns(panel_df, test_dates, results)

    # Print summary
    print(results.summary())

    if results.failures and verbose:
        print("FAILURES:")
        for f in results.failures[:20]:  # Show first 20 failures
            print(f"  - {f}")
        if len(results.failures) > 20:
            print(f"  ... and {len(results.failures) - 20} more")

    return results


# Pytest-compatible test functions
def test_fred_panel_integrity():
    """Pytest: Test panel integrity."""
    panel_df = load_monthly_panel()
    assert panel_df.index.is_monotonic_increasing, "Index not sorted"
    assert not panel_df.index.has_duplicates, "Duplicate dates found"


def test_fred_raw_series():
    """Pytest: Test raw series values."""
    results = TestResult()
    panel_df = load_monthly_panel()

    np.random.seed(RANDOM_SEED)
    test_dates = list(np.random.choice(panel_df.index[12:], size=5, replace=False))

    test_raw_series(panel_df, test_dates, results)
    assert results.failed == 0, f"Raw series test failed: {results.failures}"


def test_fred_derived_columns():
    """Pytest: Test derived column calculations."""
    results = TestResult()
    panel_df = load_monthly_panel()

    np.random.seed(RANDOM_SEED)
    test_dates = list(np.random.choice(panel_df.index[12:], size=5, replace=False))

    test_derived_columns(panel_df, test_dates, results)
    assert results.failed == 0, f"Derived columns test failed: {results.failures}"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test FRED monthly panel data')
    parser.add_argument('--num-records', type=int, default=NUM_RANDOM_RECORDS,
                        help=f'Number of random records to test (default: {NUM_RANDOM_RECORDS})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed (default: {RANDOM_SEED})')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed failure output')
    args = parser.parse_args()

    results = run_tests(
        num_records=args.num_records,
        random_seed=args.seed,
        verbose=not args.quiet
    )

    # Exit with appropriate code
    sys.exit(0 if results.failed == 0 else 1)
