"""
Yearly panel construction for the Kundig & Sigrist (2025) replication.

Builds one row per (loan, calendar year) where the loan is active at the start
of that year, with start-of-year covariate snapshots and during-year competing-
risks outcomes (prepay, default).

Key references in the paper:
  Section 3.1: data and default definition
  Section 3.2: predictor variables (Table 1)
  Section 3.4: expanding-window training/test split

Departures from the paper (intentional, repo-wide consistency):
  * spatial coordinates: 3-digit zip centroids derived from the pgeocode US
    database (paper uses zip3 centroids too -- the underlying data product
    differs slightly but the resolution matches).
  * predictor set: canonical 13 + lat/lon as used by the other notebooks
    (Cox, RSF, DeepHit, Sadhwani, NN-DTSM, Deep-PTCM); not the paper's full
    28-feature set.
  * competing risks: prepay + default, vs the paper's single-event default.
  * vintage range: 2010-2024 (limited by our processed panel) vs paper's
    1999-2022.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Raw orig file schema (Freddie Mac Single-Family Loan-Level User Guide)
# ------------------------------------------------------------------
ORIG_COLUMNS = [
    'credit_score', 'first_payment_date', 'first_time_homebuyer',
    'maturity_date', 'msa', 'mi_pct', 'nr_units', 'occupancy', 'cltv',
    'dti', 'orig_upb', 'ltv', 'orig_interest_rate', 'channel',
    'prepay_penalty', 'product_type', 'property_state', 'property_type',
    'postal_code', 'loan_sequence_number', 'loan_purpose',
    'orig_loan_term', 'nr_borrowers', 'seller_name', 'servicer_name',
    'super_conforming', 'pre_harp_lsn', 'program_indicator',
    'harp_indicator', 'valuation_method', 'io_indicator',
    'mi_cancel_indicator',
]


def build_zip3_centroids() -> pd.DataFrame:
    """3-digit zip → (latitude, longitude) lookup using pgeocode US data.

    Returns
    -------
    DataFrame with columns ['zip3', 'latitude', 'longitude'].
    """
    import pgeocode

    nomi = pgeocode.Nominatim('us')
    raw = nomi._data.copy()
    raw['zip3'] = raw['postal_code'].astype(str).str.zfill(5).str[:3]
    raw = raw.dropna(subset=['latitude', 'longitude'])
    centroids = (
        raw.groupby('zip3')
        .agg(latitude=('latitude', 'mean'), longitude=('longitude', 'mean'))
        .reset_index()
    )
    return centroids


def build_loan_zip3_lookup(raw_root: Path, restrict_loans: Optional[set] = None) -> pd.DataFrame:
    """Read raw sample_orig_*.txt files and extract loan_sequence_number → zip3.

    Parameters
    ----------
    raw_root : Path
        Directory containing sample_YYYY/sample_orig_YYYY.txt files.
    restrict_loans : set, optional
        If given, only emit rows whose loan_sequence_number is in this set.
        Useful when the processed panel covers a subset of the raw data.

    Returns
    -------
    DataFrame with columns ['loan_sequence_number', 'zip3'].
    """
    raw_root = Path(raw_root)
    paths = sorted(raw_root.glob('sample_*/sample_orig_*.txt'))
    if not paths:
        raise FileNotFoundError(f'No sample_orig files found under {raw_root}')

    pieces = []
    for p in paths:
        df = pd.read_csv(
            p, sep='|', header=None, names=ORIG_COLUMNS, dtype=str,
            usecols=['loan_sequence_number', 'postal_code'],
        )
        df['zip3'] = df['postal_code'].astype(str).str.zfill(5).str[:3]
        df = df[['loan_sequence_number', 'zip3']]
        if restrict_loans is not None:
            df = df[df['loan_sequence_number'].isin(restrict_loans)]
        pieces.append(df)
    out = pd.concat(pieces, ignore_index=True).drop_duplicates('loan_sequence_number')
    return out


def _attach_ir_spread(yearly: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Compute ir_spread = int_rate - MORTGAGE30US[Jan of year].

    `fred` is expected to be a DatetimeIndex DataFrame with a MORTGAGE30US column.
    """
    if 'MORTGAGE30US' not in fred.columns:
        raise KeyError('FRED panel must contain MORTGAGE30US')
    jan_rates = (
        fred.loc[fred.index.month == 1, ['MORTGAGE30US']]
        .copy()
        .reset_index()
        .rename(columns={fred.index.name or 'index': 'date',
                         'MORTGAGE30US': 'mortgage30us_jan'})
    )
    jan_rates['year'] = jan_rates['date'].dt.year
    jan_rates = jan_rates[['year', 'mortgage30us_jan']]
    out = yearly.merge(jan_rates, on='year', how='left')
    out['ir_spread'] = out['int_rate'] - out['mortgage30us_jan']
    return out.drop(columns='mortgage30us_jan')


# ------------------------------------------------------------------
# Main yearly-panel builder
# ------------------------------------------------------------------
PREPAY_CODE = 1
DEFAULT_CODE = 2

# The canonical 13 features used by every other model in this repo.
NUMERIC_STATIC = ['fico_score', 'dti_r', 'ltv_r', 'int_rate']
NUMERIC_TV_LOAN = ['log_upb', 'bal_repaid',
                   't_act_12m', 't_del_30d_12m', 't_del_60d_12m']
NUMERIC_TV_MACRO = ['hpi_st_d_t_o', 'ppi_c_FRMA',
                    'TB10Y_d_t_o', 'FRMA30Y_d_t_o']
SNAPSHOT_FEATURES = NUMERIC_STATIC + NUMERIC_TV_LOAN + NUMERIC_TV_MACRO


def build_yearly_panel(
    monthly_panel: pd.DataFrame,
    loan_zip3: pd.DataFrame,
    zip3_centroids: pd.DataFrame,
    fred: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the (loan, year) panel for the Kundig-Sigrist replication.

    Parameters
    ----------
    monthly_panel : DataFrame
        loan_month_panel.parquet content. Must have columns:
        loan_sequence_number, year_month (period[M]), event_code,
        plus all SNAPSHOT_FEATURES, vintage_year, fold.
    loan_zip3 : DataFrame [loan_sequence_number, zip3]
    zip3_centroids : DataFrame [zip3, latitude, longitude]
    fred : DataFrame
        DatetimeIndex with at least MORTGAGE30US column.

    Returns
    -------
    DataFrame with columns:
      loan_sequence_number, year, vintage_year, fold,
      zip3, lat, lon,
      <SNAPSHOT_FEATURES>, ir_spread, n_months,
      prepay_in_year, default_in_year
    """
    df = monthly_panel.copy()
    if not isinstance(df['year_month'].dtype, pd.PeriodDtype):
        df['year_month'] = pd.PeriodIndex(df['year_month'], freq='M')
    df['year'] = df['year_month'].dt.year.astype(int)
    df['log_upb'] = np.log(df['orig_upb'].clip(lower=1).astype(float))

    # ----- per-loan summary: origination year, terminal event, last year -----
    loan_first = (
        df.groupby('loan_sequence_number', sort=False)
        .agg(first_year=('year', 'min'),
             last_year=('year', 'max'),
             vintage_year=('vintage_year', 'first'),
             fold=('fold', 'first'))
    )
    terminal = (
        df[df['event_code'] > 0]
        .groupby('loan_sequence_number', sort=False)
        .agg(terminal_year=('year', 'first'),
             terminal_event=('event_code', 'first'))
    )
    loans = loan_first.join(terminal, how='left')
    loans['terminal_event'] = loans['terminal_event'].fillna(0).astype(int)
    loans['terminal_year'] = loans['terminal_year'].astype('Int64')
    # active_until: terminal_year if terminated else last_year
    loans['active_until'] = np.where(
        loans['terminal_event'] > 0,
        loans['terminal_year'].astype(float),
        loans['last_year'].astype(float),
    ).astype(int)

    # ----- expand to (loan, year) rows -----
    # A loan is active at start of year Y iff origination_year < Y AND
    # active_until >= Y. So Y ranges over [first_year+1, active_until].
    loans = loans[loans['active_until'] > loans['first_year']]  # at least one such Y

    rows = []
    for lsn, r in loans.iterrows():
        for Y in range(int(r['first_year']) + 1, int(r['active_until']) + 1):
            rows.append((lsn, Y))
    yearly = pd.DataFrame(rows, columns=['loan_sequence_number', 'year'])
    yearly = yearly.merge(
        loans[['vintage_year', 'fold', 'terminal_year', 'terminal_event']]
        .reset_index(),
        on='loan_sequence_number', how='left',
    )

    # ----- outcomes -----
    yearly['default_in_year'] = (
        (yearly['terminal_event'] == DEFAULT_CODE) &
        (yearly['year'] == yearly['terminal_year'])
    ).astype(int)
    yearly['prepay_in_year'] = (
        (yearly['terminal_event'] == PREPAY_CODE) &
        (yearly['year'] == yearly['terminal_year'])
    ).astype(int)
    yearly = yearly.drop(columns=['terminal_year', 'terminal_event'])

    # ----- start-of-year snapshot covariates -----
    df_jan = df[df['year_month'].dt.month == 1].copy()
    snapshot_cols = (['loan_sequence_number', 'year', 'loan_age'] + SNAPSHOT_FEATURES)
    df_jan = df_jan[snapshot_cols]
    yearly = yearly.merge(df_jan, on=['loan_sequence_number', 'year'], how='left')
    # Some loans have no January row in their first calendar year (e.g.,
    # originated mid-year). For them, use the earliest available row in
    # that year as a fallback.
    missing = yearly[SNAPSHOT_FEATURES].isna().any(axis=1)
    if missing.any():
        df_first_in_year = (
            df.sort_values(['loan_sequence_number', 'year_month'])
            .drop_duplicates(['loan_sequence_number', 'year'])
            [snapshot_cols]
        )
        fallback = (
            yearly.loc[missing, ['loan_sequence_number', 'year']]
            .merge(df_first_in_year, on=['loan_sequence_number', 'year'], how='left')
        )
        for col in ['loan_age'] + SNAPSHOT_FEATURES:
            yearly.loc[missing, col] = fallback[col].values

    # n_months = loan age in months at start of year
    yearly['n_months'] = yearly['loan_age']
    yearly = yearly.drop(columns='loan_age')

    # ----- spatial join -----
    yearly = yearly.merge(loan_zip3, on='loan_sequence_number', how='left')
    yearly = yearly.merge(zip3_centroids, on='zip3', how='left')
    yearly = yearly.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    # ----- ir_spread from FRED MORTGAGE30US (January of year) -----
    yearly = _attach_ir_spread(yearly, fred)

    return yearly


# ------------------------------------------------------------------
# Filters (paper §3.1 restrictions)
# ------------------------------------------------------------------

# Conterminous US bounding box (degrees). Excludes AK, HI, PR, GU and other
# Pacific territories which Freddie Mac includes via 3-digit zips that
# resolve to extreme lat/lon outside the contiguous US.
CONUS_LAT_BOUNDS = (24.0, 50.0)
CONUS_LON_BOUNDS = (-125.0, -66.5)


def apply_paper_filters(
    yearly: pd.DataFrame,
    drop_incomplete_final_year: bool = True,
    contiguous_us: bool = True,
    drop_missing_geo: bool = True,
) -> pd.DataFrame:
    """Apply Kundig-Sigrist §3.1 sample restrictions.

    Parameters
    ----------
    drop_incomplete_final_year : bool
        Drop the latest year if outcomes are right-censored within it (the
        loan-month panel doesn't span the full calendar year).
    contiguous_us : bool
        Restrict to lat/lon within :data:`CONUS_LAT_BOUNDS` /
        :data:`CONUS_LON_BOUNDS` (paper restricts to contiguous US).
    drop_missing_geo : bool
        Drop rows whose lat or lon is NaN (zip3 codes pgeocode could not
        resolve). These are typically rare territories.
    """
    out = yearly
    if drop_missing_geo:
        out = out.dropna(subset=['lat', 'lon'])
    if contiguous_us:
        lat_lo, lat_hi = CONUS_LAT_BOUNDS
        lon_lo, lon_hi = CONUS_LON_BOUNDS
        out = out[
            (out['lat'].between(lat_lo, lat_hi)) &
            (out['lon'].between(lon_lo, lon_hi))
        ]
    if drop_incomplete_final_year:
        # The latest year present in the source panel is incomplete iff the
        # source panel doesn't fully cover Dec of that year. We approximate
        # this by dropping the last year unconditionally; the caller can
        # override.
        max_year = int(out['year'].max())
        out = out[out['year'] < max_year]
    return out.reset_index(drop=True)
