# Implementation Plan: Loan-Month Panel Data for Cox Regression

> **Status**: Planning
> **Target**: Notebook 05 - Cause-Specific Cox with Time-Varying Covariates
> **Vintages**: 2010-2025

---

## 1. Problem Statement

### Current Implementation

The existing `05_cause_specific_cox.ipynb` uses **loan-level data** (one row per loan):

```
loan_id | duration | event | credit_score | orig_ltv | ...
L001    | 36       | 1     | 720          | 80       | ...
L002    | 48       | 2     | 650          | 95       | ...
```

**Issue**: Time-varying covariates (behavioral, macro) are only captured at the terminal observation. This is incorrect for Cox regression - covariates should be measured throughout the observation period.

### Required: Loan-Month Panel Data

For proper handling of time-varying covariates, we need **loan-month panel data**:

```
loan_id | month | start | stop | event | static_vars... | time_varying_vars...
L001    | 1     | 0     | 1    | 0     | fico=720       | ppi_c=-0.5, hpi_d=2.1
L001    | 2     | 1     | 2    | 0     | fico=720       | ppi_c=-0.3, hpi_d=2.4
L001    | 3     | 2     | 3    | 1     | fico=720       | ppi_c=-0.1, hpi_d=2.8
```

Each loan contributes multiple rows, with time-varying covariates updated each month.

---

## 2. Variables (Blumenstock et al. 2022)

### Static Covariates (fixed at origination)

| Variable | Description | Source |
|----------|-------------|--------|
| `int_rate` | Initial interest rate | Origination file |
| `orig_upb` | Original unpaid balance | Origination file |
| `fico_score` | Initial FICO score | Origination file |
| `dti_r` | Initial debt-to-income ratio | Origination file |
| `ltv_r` | Initial loan-to-value ratio | Origination file |

### Time-Varying Covariates (updated each month)

#### Behavioral Variables (from performance data)

| Variable | Description | Calculation |
|----------|-------------|-------------|
| `bal_repaid` | Current repaid balance (%) | `(orig_upb - current_upb) / orig_upb * 100` |
| `t_act_12m` | Times current in last 12 months | Rolling sum of `delinquency == 0` |
| `t_del_30d_12m` | Times 30d delinquent in last 12 months | Rolling sum of `delinquency == 1` |
| `t_del_60d_12m` | Times 60d delinquent in last 12 months | Rolling sum of `delinquency == 2` |

#### Macroeconomic Variables (merged by year-month)

| Variable | Description | Source |
|----------|-------------|--------|
| `hpi_st_d_t_o` | HPI difference (today vs origination, state) | State HPI from FHFA |
| `ppi_c_FRMA` | Prepayment incentive (int_rate - current_mortgage_rate) | FRED MORTGAGE30US |
| `TB10Y_d_t_o` | Treasury rate difference (today vs origination) | FRED DGS10 |
| `FRMA30Y_d_t_o` | 30Y FRM difference (today vs origination) | FRED MORTGAGE30US |
| `ppi_o_FRMA` | Prepayment incentive at origination | Calculated at orig |
| `hpi_st_log12m` | HPI 12-month log return (state) | State HPI from FHFA |
| `hpi_r_st_us` | State HPI / National HPI ratio | FHFA |
| `st_unemp_r12m` | Unemployment 12-month log return (state) | BLS via FRED |
| `st_unemp_r3m` | Unemployment 3-month log return (state) | BLS via FRED |
| `TB10Y_r12m` | Treasury rate 12-month return | FRED DGS10 |
| `T10Y3MM` | Yield spread (10Y - 3M) | FRED |
| `T10Y3MM_r12m` | Yield spread 12-month return | FRED |

---

## 3. Data Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 0: Load terminal records (survival_data_blumenstock.parquet)   │
│              ↓                                                       │
│  Step 1: Sample loans (100 defaults + 9,900 non-defaults × 11 folds)│
│              ↓                                                       │
│  Step 2: For each vintage, load performance data                     │
│              ↓                                                       │
│  Step 3: Filter to sampled loans EARLY (memory efficiency)           │
│              ↓                                                       │
│  Step 4: Calculate behavioral variables (rolling 12-month counts)    │
│              ↓                                                       │
│  Step 5: Determine events (prepay/default/censored)                  │
│              ↓                                                       │
│  Step 6: Create interval format (start, stop)                        │
│              ↓                                                       │
│  Step 7: Merge origination data (static covariates)                  │
│              ↓                                                       │
│  Step 8: Merge macro data (time-varying, by year-month + state)      │
│              ↓                                                       │
│  Step 9: Calculate origination-relative differences                  │
│              ↓                                                       │
│  Step 10: Save loan_month_panel.parquet                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 0: Load Terminal Records & Sample Loans

```python
# Load terminal records to identify loan outcomes
terminal_df = pd.read_parquet('data/processed/survival_data_blumenstock.parquet')

# Separate by terminal event type
defaulted_loans = terminal_df[terminal_df['event_code'] == 2]['loan_sequence_number'].unique()
non_defaulted_loans = terminal_df[terminal_df['event_code'] != 2]['loan_sequence_number'].unique()

# Sample WITHOUT replacement: 100 defaults + 9,900 non-defaults per fold
# (See Section 5 for detailed sampling code)
sampled_loan_ids, fold_assignments = create_stratified_sample(
    defaulted_loans, non_defaulted_loans,
    n_folds=11, defaults_per_fold=100, non_defaults_per_fold=9900
)
```

### Step 1: Load Raw Performance Data (filtered)

For each vintage (2010-2025), load the monthly performance file:

```python
def load_performance_data(vintage: int) -> pd.DataFrame:
    """Load performance (loan-month) data for a vintage."""
    pattern = f'sample_{vintage}/sample_svcg_{vintage}.txt'
    files = list(RAW_DATA_DIR.glob(f'**/{pattern}'))

    df = pd.read_csv(
        files[0], sep='|', names=PERFORMANCE_COLUMNS,
        dtype=PERFORMANCE_DTYPES, na_values=['', ' ']
    )
    return df
```

### Step 2: Calculate Behavioral Variables (per loan-month)

```python
def calculate_behavioral_variables(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-varying behavioral variables."""

    # Sort by loan and time
    perf_df = perf_df.sort_values(['loan_sequence_number', 'loan_age'])

    # Parse delinquency status
    perf_df['delinquency_status'] = pd.to_numeric(
        perf_df['current_loan_delinquency_status'].replace({'X': '0', 'XX': '0'}),
        errors='coerce'
    ).fillna(0).astype(int)

    # Binary indicators
    perf_df['is_current'] = (perf_df['delinquency_status'] == 0).astype(int)
    perf_df['is_30d_del'] = (perf_df['delinquency_status'] == 1).astype(int)
    perf_df['is_60d_del'] = (perf_df['delinquency_status'] == 2).astype(int)

    # Rolling 12-month counts (per loan)
    grouped = perf_df.groupby('loan_sequence_number')
    perf_df['t_act_12m'] = grouped['is_current'].transform(
        lambda x: x.rolling(12, min_periods=1).sum()
    )
    perf_df['t_del_30d_12m'] = grouped['is_30d_del'].transform(
        lambda x: x.rolling(12, min_periods=1).sum()
    )
    perf_df['t_del_60d_12m'] = grouped['is_60d_del'].transform(
        lambda x: x.rolling(12, min_periods=1).sum()
    )

    return perf_df
```

### Step 3: Determine Event Status (per loan-month)

```python
def determine_events(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine event status for each loan-month.

    Event coding (from Blumenstock):
    - 0: Censored (no event)
    - 1: Prepayment (loan repaid early)
    - 2: Default (first 90+ day delinquency)
    """

    grouped = perf_df.groupby('loan_sequence_number')

    # Default: first time reaching 90+ days delinquent
    perf_df['is_default'] = (perf_df['delinquency_status'] >= 3).astype(int)
    perf_df['first_default'] = grouped['is_default'].transform(
        lambda x: (x.cumsum() == 1) & (x == 1)
    ).astype(int)

    # Prepayment: zero balance code = 01
    perf_df['is_prepay'] = (perf_df['zero_balance_code'] == '01').astype(int)

    # Event code for terminal months
    perf_df['event_code'] = 0  # Default: censored
    perf_df.loc[perf_df['first_default'] == 1, 'event_code'] = 2  # Default
    perf_df.loc[perf_df['is_prepay'] == 1, 'event_code'] = 1  # Prepay

    return perf_df
```

### Step 4: Create Interval Format (start, stop)

For Cox regression with time-varying covariates, we need interval format:

```python
def create_interval_format(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create (start, stop) interval format for time-varying Cox.

    Each row represents the interval [start, stop) where:
    - start: beginning of interval (loan_age - 1)
    - stop: end of interval (loan_age)
    - event: 1 if event occurred at stop, 0 otherwise
    """

    perf_df['start'] = perf_df['loan_age'] - 1
    perf_df['stop'] = perf_df['loan_age']

    # Event only on terminal month (prepay or default)
    perf_df['event'] = 0

    # Mark terminal observations
    grouped = perf_df.groupby('loan_sequence_number')

    # For defaults: event at first 90+ delinquency
    default_mask = perf_df['first_default'] == 1
    perf_df.loc[default_mask, 'event'] = 1

    # For prepays: event at prepayment month
    prepay_mask = perf_df['is_prepay'] == 1
    perf_df.loc[prepay_mask, 'event'] = 1

    # Remove observations after event
    # (loan should not contribute risk after event)
    perf_df['cumulative_event'] = grouped['event'].transform('cumsum')
    perf_df = perf_df[perf_df['cumulative_event'] <= 1]  # Keep up to and including first event

    return perf_df
```

### Step 5: Merge Macro Data (by year-month and state)

```python
def merge_macro_data(perf_df: pd.DataFrame,
                      orig_df: pd.DataFrame,
                      macro_national: pd.DataFrame,
                      state_hpi: pd.DataFrame,
                      state_unemp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge time-varying macro data to each loan-month observation.
    """

    # Parse reporting period
    perf_df['reporting_date'] = pd.to_datetime(
        perf_df['monthly_reporting_period'].astype(str), format='%Y%m'
    )
    perf_df['year_month'] = perf_df['reporting_date'].dt.to_period('M')

    # Merge property state from origination
    perf_df = perf_df.merge(
        orig_df[['loan_sequence_number', 'property_state', 'first_payment_date']],
        on='loan_sequence_number',
        how='left'
    )

    # Merge national macro (by year_month)
    perf_df = perf_df.merge(
        macro_national[['year_month', 'MORTGAGE30US', 'DGS10', 'TB10Y_r12m', 'T10Y3MM', 'T10Y3MM_r12m']],
        on='year_month',
        how='left'
    )

    # Merge state unemployment (by year_month and state)
    perf_df = perf_df.merge(
        state_unemp[['year_month', 'property_state', 'st_unemp_r12m', 'st_unemp_r3m']],
        on=['year_month', 'property_state'],
        how='left'
    )

    # Merge state HPI (by year_month and state)
    perf_df = perf_df.merge(
        state_hpi[['year_month', 'property_state', 'state_hpi', 'hpi_st_log12m', 'hpi_r_st_us']],
        on=['year_month', 'property_state'],
        how='left'
    )

    return perf_df
```

### Step 6: Calculate Origination-Relative Variables

```python
def calculate_origination_differences(panel_df: pd.DataFrame,
                                       macro_national: pd.DataFrame,
                                       state_hpi: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate differences from origination time:
    - hpi_st_d_t_o: HPI difference (state)
    - ppi_c_FRMA: Prepayment incentive (current)
    - TB10Y_d_t_o: Treasury difference
    - FRMA30Y_d_t_o: Mortgage rate difference
    """

    # Get origination year_month for each loan
    panel_df['orig_year_month'] = pd.to_datetime(
        panel_df['first_payment_date'].astype(str), format='%Y%m'
    ).dt.to_period('M')

    # Merge origination-time macro values
    orig_macro = macro_national[['year_month', 'MORTGAGE30US', 'DGS10']].rename(
        columns={'year_month': 'orig_year_month',
                 'MORTGAGE30US': 'orig_MORTGAGE30US',
                 'DGS10': 'orig_DGS10'}
    )
    panel_df = panel_df.merge(orig_macro, on='orig_year_month', how='left')

    # Merge origination-time state HPI
    orig_hpi = state_hpi[['year_month', 'property_state', 'state_hpi']].rename(
        columns={'year_month': 'orig_year_month', 'state_hpi': 'orig_state_hpi'}
    )
    panel_df = panel_df.merge(orig_hpi, on=['orig_year_month', 'property_state'], how='left')

    # Calculate differences
    panel_df['hpi_st_d_t_o'] = panel_df['state_hpi'] - panel_df['orig_state_hpi']
    panel_df['ppi_c_FRMA'] = panel_df['int_rate'] - panel_df['MORTGAGE30US']
    panel_df['ppi_o_FRMA'] = panel_df['int_rate'] - panel_df['orig_MORTGAGE30US']
    panel_df['TB10Y_d_t_o'] = panel_df['DGS10'] - panel_df['orig_DGS10']
    panel_df['FRMA30Y_d_t_o'] = panel_df['MORTGAGE30US'] - panel_df['orig_MORTGAGE30US']

    return panel_df
```

---

## 4. Notebook 05 Changes

### Current Structure (loan-level)

```python
# Current: One row per loan
df = pd.read_parquet('survival_data.parquet')

cph = CoxPHFitter(penalizer=0.01)
cph.fit(df, duration_col='duration', event_col='event')
```

### New Structure (loan-month panel with time-varying covariates)

```python
# New: Multiple rows per loan (loan-month panel)
panel_df = pd.read_parquet('loan_month_panel.parquet')

# Filter to event type (cause-specific)
# For prepayment model: default treated as censored
panel_prepay = panel_df.copy()
panel_prepay['event'] = (panel_prepay['event_code'] == 1).astype(int)

# Fit Cox with interval format (start, stop)
cph_prepay = CoxPHFitter(penalizer=0.01)
cph_prepay.fit(
    panel_prepay,
    duration_col='stop',      # End of interval
    event_col='event',        # Event indicator
    start_col='start',        # Start of interval (for time-varying)
    cluster_col='loan_sequence_number'  # Cluster standard errors by loan
)
```

### Experiments (Blumenstock et al.)

```python
# Experiment 4.1: Loan-level variables only
LOAN_VARS = ['int_rate', 'orig_upb', 'fico_score', 'dti_r', 'ltv_r',
             'bal_repaid', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m']

# Experiment 4.2: Macro variables only
MACRO_VARS = ['hpi_st_d_t_o', 'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o',
              'ppi_o_FRMA', 'hpi_st_log12m', 'hpi_r_st_us', 'st_unemp_r12m',
              'st_unemp_r3m', 'TB10Y_r12m', 'T10Y3MM', 'T10Y3MM_r12m']

# Experiment 4.3: All variables
ALL_VARS = LOAN_VARS + MACRO_VARS
```

---

## 5. Sampling Strategy (Blumenstock et al.)

### Overview

Following Blumenstock et al. (2022), we use a **stratified sampling strategy** that oversamples defaults to ensure sufficient events for model estimation:

| Parameter | Value |
|-----------|-------|
| Number of folds | 11 |
| Loans per fold | 10,000 |
| Defaults per fold | 100 (1%) |
| Non-defaults per fold | 9,900 (99%) |
| Sampling method | **Without replacement** |

### Why This Strategy?

1. **Default rarity**: In the full dataset, defaults are ~3% of loans. Oversampling ensures enough default events per fold.
2. **Cross-validation**: 10 folds for CV, 1 fold reserved for hyperparameter tuning.
3. **Reproducibility**: Matches the paper's experimental design.

### Estimated Panel Size (After Sampling)

| Metric | Estimate |
|--------|----------|
| Total loans sampled | 11 × 10,000 = **110,000** |
| Average loan duration | ~50 months |
| Total loan-months | ~**5.5 million** |
| Memory (with all features) | ~**2-3 GB** |

This is much more manageable than the full 38M+ loan-month records!

### Implementation: Two-Stage Sampling

**Stage 1: Sample Loans (at loan level)**

First, identify terminal events and sample loans:

```python
# Configuration
N_FOLDS = 11
DEFAULTS_PER_FOLD = 100
NON_DEFAULTS_PER_FOLD = 9_900
TOTAL_DEFAULTS_NEEDED = N_FOLDS * DEFAULTS_PER_FOLD      # 1,100
TOTAL_NON_DEFAULTS_NEEDED = N_FOLDS * NON_DEFAULTS_PER_FOLD  # 108,900

# Load terminal records to identify event types
terminal_df = pd.read_parquet('data/processed/survival_data_blumenstock.parquet')

# Separate by terminal event
defaulted_loans = terminal_df[terminal_df['event_code'] == 2]['loan_sequence_number'].unique()
non_defaulted_loans = terminal_df[terminal_df['event_code'] != 2]['loan_sequence_number'].unique()

print(f"Defaulted loans available: {len(defaulted_loans):,}")
print(f"Non-defaulted loans available: {len(non_defaulted_loans):,}")

# Verify sufficient defaults
if len(defaulted_loans) < TOTAL_DEFAULTS_NEEDED:
    raise ValueError(f"Need {TOTAL_DEFAULTS_NEEDED} defaults, only {len(defaulted_loans)} available")

# Shuffle and sample WITHOUT replacement
np.random.seed(42)
defaulted_shuffled = np.random.permutation(defaulted_loans)
non_defaulted_shuffled = np.random.permutation(non_defaulted_loans)

# Create folds by taking sequential chunks
fold_assignments = {}
for fold in range(N_FOLDS):
    # Defaults for this fold
    d_start = fold * DEFAULTS_PER_FOLD
    d_end = d_start + DEFAULTS_PER_FOLD
    fold_defaults = defaulted_shuffled[d_start:d_end]

    # Non-defaults for this fold
    nd_start = fold * NON_DEFAULTS_PER_FOLD
    nd_end = nd_start + NON_DEFAULTS_PER_FOLD
    fold_non_defaults = non_defaulted_shuffled[nd_start:nd_end]

    # Combine
    fold_loans = np.concatenate([fold_defaults, fold_non_defaults])
    for loan_id in fold_loans:
        fold_assignments[loan_id] = fold

# All sampled loans
sampled_loan_ids = set(fold_assignments.keys())
print(f"Total loans sampled: {len(sampled_loan_ids):,}")
```

**Stage 2: Create Loan-Month Panel (only for sampled loans)**

```python
def create_panel_for_sampled_loans(
    vintage: int,
    sampled_loan_ids: set,
    fold_assignments: dict
) -> pd.DataFrame:
    """
    Create loan-month panel ONLY for loans in sampled_loan_ids.
    """
    # Load performance data
    perf_df = load_performance_data(vintage)

    # Filter to sampled loans EARLY (reduces memory)
    perf_df = perf_df[perf_df['loan_sequence_number'].isin(sampled_loan_ids)]

    if len(perf_df) == 0:
        return pd.DataFrame()

    # Add fold assignment
    perf_df['fold'] = perf_df['loan_sequence_number'].map(fold_assignments)

    # Calculate behavioral variables
    perf_df = calculate_behavioral_variables(perf_df)

    # Determine events
    perf_df = determine_events(perf_df)

    # Create interval format
    perf_df = create_interval_format(perf_df)

    return perf_df

# Process all vintages
panel_dfs = []
for vintage in range(2010, 2026):
    panel_vintage = create_panel_for_sampled_loans(vintage, sampled_loan_ids, fold_assignments)
    if not panel_vintage.empty:
        panel_dfs.append(panel_vintage)
        print(f"Vintage {vintage}: {len(panel_vintage):,} loan-months")

# Combine
panel_df = pd.concat(panel_dfs, ignore_index=True)
print(f"Total loan-months: {len(panel_df):,}")
```

### Fold Structure

After sampling, each fold contains:

```
Fold 0:  10,000 loans → ~500,000 loan-months
Fold 1:  10,000 loans → ~500,000 loan-months
...
Fold 10: 10,000 loans → ~500,000 loan-months
─────────────────────────────────────────────
Total:   110,000 loans → ~5.5M loan-months
```

### Cross-Validation Usage

```python
# Folds 0-9: Cross-validation (train on 9, test on 1)
# Fold 10: Hyperparameter tuning (held out)

CV_FOLDS = list(range(10))
TUNING_FOLD = 10

# Example: Train on folds 1-9, test on fold 0
train_mask = panel_df['fold'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])
test_mask = panel_df['fold'] == 0

train_df = panel_df[train_mask]
test_df = panel_df[test_mask]
```

---

## 6. Implementation Checklist

### Phase 1: Loan Sampling

- [ ] Create sampling logic in `notebooks/03b_create_loan_month_panel.ipynb`
  - [ ] Load terminal records from `survival_data_blumenstock.parquet`
  - [ ] Separate defaulted vs non-defaulted loans
  - [ ] Sample 100 defaults + 9,900 non-defaults per fold (without replacement)
  - [ ] Create fold assignments dictionary
  - [ ] Verify: 11 folds × 10,000 loans = 110,000 total

### Phase 2: Panel Data Creation

- [ ] Create `src/data/create_loan_month_panel.py`
  - [ ] `load_performance_data(vintage)` - Load raw monthly data
  - [ ] `filter_to_sampled_loans(perf_df, sampled_loan_ids)` - Early filtering
  - [ ] `calculate_behavioral_variables(perf_df)` - Rolling 12-month counts
  - [ ] `determine_events(perf_df)` - Event coding per Blumenstock
  - [ ] `create_interval_format(perf_df)` - (start, stop) for Cox
  - [ ] `merge_macro_data(perf_df, ...)` - Time-varying macro variables
  - [ ] `calculate_origination_differences(panel_df, ...)` - Origination-relative features

- [ ] Complete `notebooks/03b_create_loan_month_panel.ipynb`
  - [ ] Process vintages 2010-2025 (only sampled loans)
  - [ ] Merge origination data (static covariates)
  - [ ] Merge macro data (time-varying)
  - [ ] Save to `data/processed/loan_month_panel.parquet`
  - [ ] Verify: ~5.5M loan-month records with fold assignments

### Phase 3: Cox Model Update

- [ ] Rewrite `notebooks/05_cause_specific_cox.ipynb`
  - [ ] Load loan-month panel data
  - [ ] Implement cross-validation loop (folds 0-9)
  - [ ] Fit cause-specific Cox with `start_col` parameter (time-varying)
  - [ ] Run experiments:
    - [ ] Exp 4.1: Loan-level variables only
    - [ ] Exp 4.2: Macro variables only
    - [ ] Exp 4.3: All variables
  - [ ] Calculate time-dependent C-index at 24, 48, 72 months
  - [ ] Aggregate results across CV folds

### Phase 4: Validation & Reporting

- [ ] Test proportional hazards assumption (Schoenfeld residuals)
- [ ] Check for collinearity among time-varying covariates
- [ ] Create results table (matching Blumenstock Table 4 format)
- [ ] Plot hazard ratios with confidence intervals
- [ ] Document differences from Blumenstock results

---

## 7. Expected Output

### Panel Data Schema

```
loan_month_panel.parquet (~5.5 million rows, ~30 columns)
│
├── Identifiers & Fold:
│   ├── loan_sequence_number (str): Loan identifier
│   ├── fold (int): Cross-validation fold (0-10)
│   ├── vintage_year (int): Origination year
│   └── property_state (str): State code
│
├── Time indices:
│   ├── loan_age (int): Month since origination (1, 2, 3, ...)
│   ├── start (int): Interval start (loan_age - 1)
│   ├── stop (int): Interval end (loan_age)
│   └── year_month (period): Calendar month of observation
│
├── Event indicators:
│   ├── event (int): Event in this interval (0/1) - for Cox
│   └── event_code (int): Terminal event type (0=censored, 1=prepay, 2=default)
│
├── Static covariates (fixed at origination):
│   ├── int_rate (float): Initial interest rate
│   ├── orig_upb (float): Original unpaid balance
│   ├── fico_score (float): Initial FICO score
│   ├── dti_r (float): Initial debt-to-income ratio
│   └── ltv_r (float): Initial loan-to-value ratio
│
├── Time-varying behavioral (updated each month):
│   ├── bal_repaid (float): Percent of balance repaid
│   ├── t_act_12m (int): Times current in last 12 months (0-12)
│   ├── t_del_30d_12m (int): Times 30d delinquent in last 12 months (0-12)
│   └── t_del_60d_12m (int): Times 60d delinquent in last 12 months (0-12)
│
└── Time-varying macro (updated each month):
    ├── hpi_st_d_t_o (float): HPI change since origination (state)
    ├── ppi_c_FRMA (float): Current prepayment incentive
    ├── TB10Y_d_t_o (float): Treasury rate change since origination
    ├── FRMA30Y_d_t_o (float): 30Y FRM change since origination
    ├── ppi_o_FRMA (float): Prepayment incentive at origination
    ├── hpi_st_log12m (float): HPI 12-month log return (state)
    ├── hpi_r_st_us (float): State/National HPI ratio
    ├── st_unemp_r12m (float): Unemployment 12-month log return (state)
    ├── st_unemp_r3m (float): Unemployment 3-month log return (state)
    ├── TB10Y_r12m (float): Treasury 12-month return
    ├── T10Y3MM (float): Yield spread (10Y - 3M)
    └── T10Y3MM_r12m (float): Yield spread 12-month return
```

### Expected Data Statistics

| Metric | Expected Value |
|--------|----------------|
| Total loans | 110,000 |
| Total loan-months | ~5.5 million |
| Loans per fold | 10,000 |
| Defaults per fold | 100 (1%) |
| Non-defaults per fold | 9,900 (99%) |
| Average loan duration | ~50 months |
| File size | ~2-3 GB |

### Model Output

For each experiment (4.1, 4.2, 4.3) and each event type (prepay, default):

1. **Coefficient estimates** with standard errors
2. **Hazard ratios** with 95% CI
3. **Time-dependent C-index** at 24, 48, 72 months
4. **Proportional hazards test** results

---

## 8. References

1. **Blumenstock, G., Lessmann, S., & Seow, H-V. (2022)**. Deep learning for survival and competing risk modelling. *Journal of the Operational Research Society*, 73(1), 26-38.

2. **Therneau, T.M. & Grambsch, P.M. (2000)**. Modeling Survival Data: Extending the Cox Model. Springer. *Chapter 5: Time-varying covariates*.

3. **lifelines documentation**: [Cox regression with time-varying covariates](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#time-varying-covariates)

---

## 9. Summary

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sampling | Blumenstock stratified | 100 defaults + 9,900 non-defaults per fold |
| Number of folds | 11 | 10 for CV, 1 for hyperparameter tuning |
| Sampling method | Without replacement | Ensures independent folds |
| Data format | Loan-month panel | Required for time-varying covariates |
| Cox format | Interval (start, stop) | Proper handling of time-varying covariates |

### Data Flow Summary

```
775,000 loans (2010-2025 vintages)
        ↓ Stratified sampling
110,000 loans (11 folds × 10,000)
        ↓ Expand to loan-month
~5.5M loan-month records
        ↓ Cox regression with time-varying covariates
Cause-specific hazard estimates
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/data/create_loan_month_panel.py` | Create | Panel data creation functions |
| `notebooks/03b_create_loan_month_panel.ipynb` | Create | Data preparation notebook |
| `data/processed/loan_month_panel.parquet` | Output | ~5.5M loan-month records |
| `notebooks/05_cause_specific_cox.ipynb` | Rewrite | Cox with time-varying covariates |
