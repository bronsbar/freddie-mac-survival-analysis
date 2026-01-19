"""
Download macroeconomic data from FRED (Federal Reserve Economic Data).

This script downloads the following series:
- UNRATE: Unemployment Rate (monthly)
- MORTGAGE30US: 30-Year Fixed Rate Mortgage Average (weekly -> monthly avg)
- CSUSHPINSA: S&P/Case-Shiller U.S. National Home Price Index (monthly)
- USSTHPI: All-Transactions House Price Index for the United States (quarterly)

Usage:
    python -m src.data.download_fred [--api-key YOUR_API_KEY]

If no API key is provided, data is downloaded via direct CSV URLs from FRED.
"""

import argparse
import pandas as pd
import requests
from pathlib import Path
from io import StringIO
from datetime import datetime

# FRED series to download - National level
FRED_SERIES = {
    # === Original Series ===
    'UNRATE': {
        'name': 'Unemployment Rate',
        'frequency': 'monthly',
        'description': 'Civilian Unemployment Rate, Seasonally Adjusted'
    },
    'MORTGAGE30US': {
        'name': '30-Year Mortgage Rate',
        'frequency': 'weekly',
        'description': '30-Year Fixed Rate Mortgage Average in the United States'
    },
    'CSUSHPINSA': {
        'name': 'Case-Shiller Home Price Index',
        'frequency': 'monthly',
        'description': 'S&P/Case-Shiller U.S. National Home Price Index, Not Seasonally Adjusted'
    },
    'USSTHPI': {
        'name': 'FHFA House Price Index',
        'frequency': 'quarterly',
        'description': 'All-Transactions House Price Index for the United States'
    },
    'FEDFUNDS': {
        'name': 'Federal Funds Rate',
        'frequency': 'monthly',
        'description': 'Federal Funds Effective Rate'
    },
    'T10Y2Y': {
        'name': 'Treasury Spread 10Y-2Y',
        'frequency': 'daily',
        'description': '10-Year Treasury Constant Maturity Minus 2-Year Treasury'
    },
    # === Interest Rates ===
    'DGS10': {
        'name': '10-Year Treasury Rate',
        'frequency': 'daily',
        'description': '10-Year Treasury Constant Maturity Rate'
    },
    'DGS5': {
        'name': '5-Year Treasury Rate',
        'frequency': 'daily',
        'description': '5-Year Treasury Constant Maturity Rate'
    },
    'DGS2': {
        'name': '2-Year Treasury Rate',
        'frequency': 'daily',
        'description': '2-Year Treasury Constant Maturity Rate'
    },
    'MORTGAGE15US': {
        'name': '15-Year Mortgage Rate',
        'frequency': 'weekly',
        'description': '15-Year Fixed Rate Mortgage Average in the United States'
    },
    'DPRIME': {
        'name': 'Prime Rate',
        'frequency': 'monthly',
        'description': 'Bank Prime Loan Rate'
    },
    'BAA10Y': {
        'name': 'Corporate Bond Spread',
        'frequency': 'daily',
        'description': "Moody's Baa Corporate Bond Yield Relative to 10-Year Treasury"
    },
    # === Housing Market ===
    'HOUST': {
        'name': 'Housing Starts',
        'frequency': 'monthly',
        'description': 'Housing Starts: Total: New Privately Owned Housing Units Started'
    },
    'HSN1F': {
        'name': 'New Home Sales',
        'frequency': 'monthly',
        'description': 'New One Family Houses Sold: United States'
    },
    'EXHOSLUSM495S': {
        'name': 'Existing Home Sales',
        'frequency': 'monthly',
        'description': 'Existing Home Sales'
    },
    'PERMIT': {
        'name': 'Building Permits',
        'frequency': 'monthly',
        'description': 'New Private Housing Units Authorized by Building Permits'
    },
    'MSACSR': {
        'name': 'Monthly Supply of Houses',
        'frequency': 'monthly',
        'description': 'Monthly Supply of New Houses in the United States'
    },
    # === Economic Indicators ===
    'A191RL1Q225SBEA': {
        'name': 'Real GDP Growth',
        'frequency': 'quarterly',
        'description': 'Real Gross Domestic Product, Percent Change from Preceding Period'
    },
    'UMCSENT': {
        'name': 'Consumer Sentiment',
        'frequency': 'monthly',
        'description': 'University of Michigan: Consumer Sentiment'
    },
    'CPIAUCSL': {
        'name': 'CPI All Items',
        'frequency': 'monthly',
        'description': 'Consumer Price Index for All Urban Consumers: All Items'
    },
    'PCEPI': {
        'name': 'PCE Price Index',
        'frequency': 'monthly',
        'description': 'Personal Consumption Expenditures: Chain-type Price Index'
    },
    'DSPIC96': {
        'name': 'Real Disposable Income',
        'frequency': 'monthly',
        'description': 'Real Disposable Personal Income'
    },
}

# State FIPS codes and abbreviations for state-level data
STATE_CODES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}


def download_fred_csv(series_id: str, start_date: str = '1998-01-01',
                      end_date: str = None) -> pd.DataFrame:
    """
    Download FRED series via direct CSV URL (no API key required).

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'UNRATE')
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.

    Returns
    -------
    pd.DataFrame
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

    print(f"Downloading {series_id}...")
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), parse_dates=['observation_date'], index_col='observation_date')
    df.index.name = 'DATE'
    df.columns = [series_id]

    # Replace '.' with NaN (FRED uses '.' for missing values)
    df = df.replace('.', pd.NA)
    df[series_id] = pd.to_numeric(df[series_id], errors='coerce')

    print(f"  Downloaded {len(df)} observations from {df.index.min()} to {df.index.max()}")
    return df


def download_fred_api(series_id: str, api_key: str, start_date: str = '1998-01-01',
                      end_date: str = None) -> pd.DataFrame:
    """
    Download FRED series via API (requires API key).

    Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    from fredapi import Fred

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    fred = Fred(api_key=api_key)
    print(f"Downloading {series_id} via API...")

    series = fred.get_series(series_id, observation_start=start_date,
                             observation_end=end_date)
    df = series.to_frame(name=series_id)

    print(f"  Downloaded {len(df)} observations from {df.index.min()} to {df.index.max()}")
    return df


def resample_to_monthly(df: pd.DataFrame, column: str, method: str = 'mean') -> pd.DataFrame:
    """
    Resample time series to monthly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    column : str
        Column name to resample
    method : str
        Aggregation method: 'mean', 'last', 'first'

    Returns
    -------
    pd.DataFrame
        Monthly resampled data
    """
    if method == 'mean':
        return df.resample('MS').mean()
    elif method == 'last':
        return df.resample('MS').last()
    elif method == 'first':
        return df.resample('MS').first()
    else:
        raise ValueError(f"Unknown method: {method}")


def download_state_series(state_abbrev: str, series_type: str,
                          start_date: str = '1998-01-01',
                          end_date: str = None) -> pd.DataFrame:
    """
    Download state-level FRED series.

    Parameters
    ----------
    state_abbrev : str
        Two-letter state abbreviation (e.g., 'CA')
    series_type : str
        Type of series: 'unemployment' or 'hpi'
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    pd.DataFrame
        State-level data
    """
    if series_type == 'unemployment':
        # State unemployment rate: {STATE}UR (e.g., CAUR)
        series_id = f'{state_abbrev}UR'
    elif series_type == 'hpi':
        # State HPI: {STATE}STHPI (e.g., CASTHPI)
        series_id = f'{state_abbrev}STHPI'
    else:
        raise ValueError(f"Unknown series_type: {series_type}")

    try:
        df = download_fred_csv(series_id, start_date, end_date)
        df.columns = [f'{state_abbrev}_{series_type}']
        return df
    except Exception as e:
        print(f"  Warning: Could not download {series_id}: {e}")
        return None


def download_all_state_data(series_type: str, start_date: str = '1998-01-01',
                            end_date: str = None) -> pd.DataFrame:
    """
    Download state-level data for all states.

    Parameters
    ----------
    series_type : str
        'unemployment' or 'hpi'

    Returns
    -------
    pd.DataFrame
        Panel with columns for each state
    """
    print(f"\nDownloading state-level {series_type} data...")
    state_data = {}

    for state_abbrev in STATE_CODES.keys():
        df = download_state_series(state_abbrev, series_type, start_date, end_date)
        if df is not None:
            col_name = df.columns[0]
            state_data[col_name] = df[col_name]

    if state_data:
        combined = pd.DataFrame(state_data)
        print(f"  Downloaded {len(combined.columns)} states, {len(combined)} observations")
        return combined
    else:
        return pd.DataFrame()


def create_monthly_macro_panel(data_dict: dict) -> pd.DataFrame:
    """
    Combine all FRED series into a single monthly panel.

    Parameters
    ----------
    data_dict : dict
        Dictionary of DataFrames keyed by series ID

    Returns
    -------
    pd.DataFrame
        Combined monthly panel with all series
    """
    monthly_data = {}

    for series_id, df in data_dict.items():
        freq = FRED_SERIES[series_id]['frequency']

        if freq == 'weekly' or freq == 'daily':
            # Resample to monthly average
            monthly = resample_to_monthly(df, series_id, method='mean')
        elif freq == 'quarterly':
            # Forward fill quarterly to monthly
            monthly = df.resample('MS').ffill()
        else:
            # Already monthly, just ensure month start index
            monthly = df.resample('MS').first()

        monthly_data[series_id] = monthly[series_id]

    # Combine all series
    combined = pd.DataFrame(monthly_data)

    # Add derived features
    if 'CSUSHPINSA' in combined.columns:
        # YoY HPI change (%)
        combined['hpi_yoy_change'] = combined['CSUSHPINSA'].pct_change(12, fill_method=None) * 100
        # MoM HPI change (%)
        combined['hpi_mom_change'] = combined['CSUSHPINSA'].pct_change(1, fill_method=None) * 100

    if 'USSTHPI' in combined.columns:
        # YoY FHFA HPI change (%)
        combined['fhfa_hpi_yoy_change'] = combined['USSTHPI'].pct_change(4, fill_method=None) * 100

    if 'UNRATE' in combined.columns:
        # YoY change in unemployment
        combined['unemp_yoy_change'] = combined['UNRATE'].diff(12)

    if 'CPIAUCSL' in combined.columns:
        # YoY inflation rate (%)
        combined['inflation_yoy'] = combined['CPIAUCSL'].pct_change(12, fill_method=None) * 100

    if 'MORTGAGE30US' in combined.columns:
        # MoM change in mortgage rates
        combined['mortgage_rate_mom_change'] = combined['MORTGAGE30US'].diff(1)
        # YoY change in mortgage rates
        combined['mortgage_rate_yoy_change'] = combined['MORTGAGE30US'].diff(12)

    if 'MORTGAGE30US' in combined.columns and 'DGS10' in combined.columns:
        # Mortgage spread over 10Y Treasury
        combined['mortgage_spread'] = combined['MORTGAGE30US'] - combined['DGS10']

    if 'DGS10' in combined.columns and 'DGS2' in combined.columns:
        # Yield curve slope (10Y - 2Y)
        combined['yield_curve_slope'] = combined['DGS10'] - combined['DGS2']

    if 'A191RL1Q225SBEA' in combined.columns:
        # Rename GDP growth for clarity
        combined['gdp_growth'] = combined['A191RL1Q225SBEA']

    return combined


def main():
    parser = argparse.ArgumentParser(description='Download FRED macroeconomic data')
    parser.add_argument('--api-key', type=str, default=None,
                        help='FRED API key (optional, uses direct CSV if not provided)')
    parser.add_argument('--start-date', type=str, default='1998-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--output-dir', type=str, default='data/external',
                        help='Output directory for downloaded data')
    parser.add_argument('--include-states', action='store_true',
                        help='Download state-level unemployment and HPI data')
    parser.add_argument('--national-only', action='store_true',
                        help='Only download national-level data (skip state data)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download national series
    print("=== Downloading National-Level Series ===")
    data_dict = {}
    for series_id in FRED_SERIES.keys():
        try:
            if args.api_key:
                df = download_fred_api(series_id, args.api_key,
                                       args.start_date, args.end_date)
            else:
                df = download_fred_csv(series_id, args.start_date, args.end_date)

            data_dict[series_id] = df

            # Save individual series
            output_path = output_dir / f'{series_id.lower()}.csv'
            df.to_csv(output_path)

        except Exception as e:
            print(f"  ERROR downloading {series_id}: {e}")

    # Create combined monthly panel
    print("\nCreating combined national monthly panel...")
    monthly_panel = create_monthly_macro_panel(data_dict)

    # Save combined panel
    output_path = output_dir / 'fred_monthly_panel.csv'
    monthly_panel.to_csv(output_path)
    print(f"Saved combined panel to {output_path}")
    print(f"  Shape: {monthly_panel.shape}")
    print(f"  Date range: {monthly_panel.index.min()} to {monthly_panel.index.max()}")

    # Also save as parquet for faster loading
    output_path_parquet = output_dir / 'fred_monthly_panel.parquet'
    monthly_panel.to_parquet(output_path_parquet)
    print(f"Saved parquet to {output_path_parquet}")

    # Download state-level data if requested
    if args.include_states and not args.national_only:
        print("\n=== Downloading State-Level Series ===")

        # State unemployment rates
        state_unemp = download_all_state_data('unemployment', args.start_date, args.end_date)
        if not state_unemp.empty:
            # Resample to monthly
            state_unemp = state_unemp.resample('MS').first()
            output_path = output_dir / 'state_unemployment.parquet'
            state_unemp.to_parquet(output_path)
            print(f"Saved state unemployment to {output_path}")
            print(f"  Shape: {state_unemp.shape}")

        # State HPI
        state_hpi = download_all_state_data('hpi', args.start_date, args.end_date)
        if not state_hpi.empty:
            # HPI is quarterly, forward fill to monthly
            state_hpi = state_hpi.resample('MS').ffill()

            # Add YoY changes for each state
            state_hpi_yoy = state_hpi.pct_change(12, fill_method=None) * 100
            state_hpi_yoy.columns = [col.replace('_hpi', '_hpi_yoy') for col in state_hpi_yoy.columns]

            output_path = output_dir / 'state_hpi.parquet'
            state_hpi.to_parquet(output_path)
            print(f"Saved state HPI to {output_path}")
            print(f"  Shape: {state_hpi.shape}")

            output_path = output_dir / 'state_hpi_yoy.parquet'
            state_hpi_yoy.to_parquet(output_path)
            print(f"Saved state HPI YoY changes to {output_path}")

    # Print summary statistics
    print("\n=== National Panel Summary ===")
    print(f"Columns: {monthly_panel.columns.tolist()}")
    print(f"\nStatistics:")
    # Only show stats for a subset of key columns
    key_cols = ['UNRATE', 'MORTGAGE30US', 'CSUSHPINSA', 'FEDFUNDS', 'hpi_yoy_change', 'inflation_yoy']
    available_cols = [c for c in key_cols if c in monthly_panel.columns]
    if available_cols:
        print(monthly_panel[available_cols].describe().round(2).to_string())

    return monthly_panel


if __name__ == '__main__':
    main()
