"""
Data preparation functions for competing risks analysis.

This module provides functions to transform loan-level data into formats
suitable for Fine-Gray and cause-specific Cox models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union


def create_loan_month_panel(
    origination_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    state_unemp_df: Optional[pd.DataFrame] = None,
    state_hpi_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create a loan-month panel from origination and performance data.

    Parameters
    ----------
    origination_df : pd.DataFrame
        Loan-level origination data with static features
    performance_df : pd.DataFrame
        Monthly performance data with time-varying features
    macro_df : pd.DataFrame, optional
        National-level macroeconomic data indexed by date
    state_unemp_df : pd.DataFrame, optional
        State-level unemployment data in long format
    state_hpi_df : pd.DataFrame, optional
        State-level HPI data in long format

    Returns
    -------
    pd.DataFrame
        Loan-month panel with all features merged
    """
    # Merge origination with performance
    panel = performance_df.merge(
        origination_df,
        on='loan_sequence_number',
        how='left'
    )

    # Merge national macro data if provided
    if macro_df is not None:
        panel = panel.merge(
            macro_df,
            on='year_month',
            how='left'
        )

    # Merge state unemployment if provided
    if state_unemp_df is not None:
        panel = panel.merge(
            state_unemp_df,
            on=['year_month', 'property_state'],
            how='left'
        )

    # Merge state HPI if provided
    if state_hpi_df is not None:
        panel = panel.merge(
            state_hpi_df,
            on=['year_month', 'property_state'],
            how='left'
        )

    return panel


def create_fine_gray_dataset(
    df: pd.DataFrame,
    primary_event: int = 1,
    competing_events: List[int] = [2],
    time_col: str = 'loan_age',
    event_col: str = 'event_code',
    id_col: str = 'loan_sequence_number',
) -> pd.DataFrame:
    """
    Create dataset for discrete-time Fine-Gray model.

    In Fine-Gray, subjects who experience competing events remain in the
    risk set (with event=0) for the primary event. This is the key
    difference from cause-specific hazard models.

    Parameters
    ----------
    df : pd.DataFrame
        Loan-month panel data
    primary_event : int
        Event code for primary event of interest (default: 1 = prepay)
    competing_events : List[int]
        Event codes for competing events (default: [2] = default)
    time_col : str
        Column name for time variable
    event_col : str
        Column name for event code
    id_col : str
        Column name for subject identifier

    Returns
    -------
    pd.DataFrame
        Dataset formatted for Fine-Gray modeling with columns:
        - All original columns
        - 'event_fg': 1 if primary event, 0 otherwise
        - 'weight_fg': Fine-Gray weight (for IPCW if needed)
    """
    df_fg = df.copy()

    # Binary indicator for primary event
    df_fg['event_fg'] = (df_fg[event_col] == primary_event).astype(int)

    # For Fine-Gray, competing events stay in risk set
    # They contribute to the risk set with event=0
    # This is handled by NOT removing them at their event time

    # Identify terminal records
    df_fg['is_terminal'] = df_fg[event_col].isin(
        [primary_event] + competing_events + [0]
    ).astype(int)

    # Calculate Fine-Gray weights (simplified - equal weights)
    # For proper IPCW weights, need censoring distribution
    df_fg['weight_fg'] = 1.0

    return df_fg


def create_cause_specific_dataset(
    df: pd.DataFrame,
    event_of_interest: int,
    time_col: str = 'loan_age',
    event_col: str = 'event_code',
    id_col: str = 'loan_sequence_number',
) -> pd.DataFrame:
    """
    Create dataset for cause-specific Cox model.

    In cause-specific analysis, competing events are treated as censored.

    Parameters
    ----------
    df : pd.DataFrame
        Loan-month panel data
    event_of_interest : int
        Event code for the event to model
    time_col : str
        Column name for time variable
    event_col : str
        Column name for event code
    id_col : str
        Column name for subject identifier

    Returns
    -------
    pd.DataFrame
        Dataset with competing events recoded as censored
    """
    df_cs = df.copy()

    # Recode: event of interest = 1, everything else = 0 (censored)
    df_cs['event_cs'] = (df_cs[event_col] == event_of_interest).astype(int)

    return df_cs


def calculate_derived_features(
    df: pd.DataFrame,
    orig_rate_col: str = 'orig_interest_rate',
    market_rate_col: str = 'MORTGAGE30US',
    orig_ltv_col: str = 'orig_ltv',
    orig_hpi_col: str = 'orig_state_hpi',
    current_hpi_col: str = 'state_hpi',
) -> pd.DataFrame:
    """
    Calculate derived features for prepayment modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Loan-month panel
    orig_rate_col : str
        Column with original interest rate
    market_rate_col : str
        Column with current market rate
    orig_ltv_col : str
        Column with original LTV
    orig_hpi_col : str
        Column with HPI at origination
    current_hpi_col : str
        Column with current HPI

    Returns
    -------
    pd.DataFrame
        DataFrame with additional derived features
    """
    df = df.copy()

    # Refinance incentive (positive = incentive to refinance)
    if orig_rate_col in df.columns and market_rate_col in df.columns:
        df['refinance_incentive'] = df[orig_rate_col] - df[market_rate_col]

        # Bucket the incentive
        df['refi_incentive_bucket'] = pd.cut(
            df['refinance_incentive'],
            bins=[-np.inf, -1, 0, 0.5, 1, 1.5, 2, np.inf],
            labels=['<-1%', '-1-0%', '0-0.5%', '0.5-1%', '1-1.5%', '1.5-2%', '>2%']
        )

    # Current LTV approximation
    if all(col in df.columns for col in [orig_ltv_col, orig_hpi_col, current_hpi_col]):
        hpi_ratio = df[orig_hpi_col] / df[current_hpi_col]
        df['current_ltv_approx'] = df[orig_ltv_col] * hpi_ratio
        df['has_equity'] = (df['current_ltv_approx'] < 80).astype(int)
        df['is_underwater'] = (df['current_ltv_approx'] > 100).astype(int)

    # Loan age squared (for non-linear seasoning)
    if 'loan_age' in df.columns:
        df['loan_age_squared'] = df['loan_age'] ** 2

    return df


def split_by_vintage(
    df: pd.DataFrame,
    train_vintages: List[int],
    test_vintages: List[int],
    vintage_col: str = 'vintage_year',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by vintage year for out-of-time validation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    train_vintages : List[int]
        Vintage years for training
    test_vintages : List[int]
        Vintage years for testing
    vintage_col : str
        Column containing vintage year

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_df = df[df[vintage_col].isin(train_vintages)].copy()
    test_df = df[df[vintage_col].isin(test_vintages)].copy()

    return train_df, test_df


def get_last_record_per_loan(
    df: pd.DataFrame,
    id_col: str = 'loan_sequence_number',
    time_col: str = 'loan_age',
) -> pd.DataFrame:
    """
    Get the last (terminal) record for each loan.

    Useful for survival analysis that requires one record per subject.

    Parameters
    ----------
    df : pd.DataFrame
        Loan-month panel
    id_col : str
        Loan identifier column
    time_col : str
        Time column to sort by

    Returns
    -------
    pd.DataFrame
        One row per loan (the terminal record)
    """
    return df.sort_values(time_col).groupby(id_col).last().reset_index()
