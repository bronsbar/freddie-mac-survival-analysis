# Data Preparation for Survival Analysis

## Data Overview

The Freddie Mac Single-Family Loan-Level Dataset contains:
- **Origination files** (`sample_orig_YYYY.txt`): Static loan characteristics at origination
- **Performance files** (`sample_svcg_YYYY.txt`): Monthly loan status tracking

Both files are pipe-delimited (`|`) with no header row.

---

## Step 1: Load & Parse Raw Data

### Origination File Columns (32 columns)
| Position | Field Name | Description |
|----------|------------|-------------|
| 1 | credit_score | Borrower credit score (300-850, 9999=Not Available) |
| 2 | first_payment_date | First scheduled payment date (YYYYMM) |
| 3 | first_time_homebuyer | Y/N/9 |
| 4 | maturity_date | Final payment date (YYYYMM) |
| 5 | msa | Metropolitan Statistical Area code |
| 6 | mi_pct | Mortgage insurance percentage (0-55, 999=NA) |
| 7 | num_units | Number of units (1-4) |
| 8 | occupancy_status | P=Primary, S=Second, I=Investment |
| 9 | orig_cltv | Original combined loan-to-value |
| 10 | orig_dti | Original debt-to-income ratio |
| 11 | orig_upb | Original unpaid principal balance |
| 12 | orig_ltv | Original loan-to-value |
| 13 | orig_interest_rate | Original interest rate |
| 14 | channel | R=Retail, B=Broker, C=Correspondent |
| 15 | ppm_flag | Prepayment penalty flag (Y/N) |
| 16 | amortization_type | FRM or ARM |
| 17 | property_state | Two-letter state code |
| 18 | property_type | SF/CO/PU/MH/CP |
| 19 | postal_code | First 3 digits + 00 |
| 20 | loan_sequence_number | Unique loan identifier |
| 21 | loan_purpose | P=Purchase, C=Cash-out Refi, N=No Cash-out Refi |
| 22 | orig_loan_term | Loan term in months |
| 23 | num_borrowers | Number of borrowers |
| 24 | seller_name | Seller name |
| 25 | servicer_name | Servicer name |
| 26 | super_conforming_flag | Y or blank |
| 27 | pre_relief_refi_loan_seq | Pre-relief refinance loan sequence |
| 28 | special_eligibility_program | H/F/R/9 |
| 29 | relief_refinance_indicator | Y or blank |
| 30 | property_valuation_method | 1-4, 9=NA |
| 31 | interest_only_indicator | Y/N |
| 32 | mi_cancellation_indicator | Y/N/7/9 |

### Performance File Columns (32 columns)
| Position | Field Name | Description |
|----------|------------|-------------|
| 1 | loan_sequence_number | Unique loan identifier |
| 2 | monthly_reporting_period | As-of month (YYYYMM) |
| 3 | current_actual_upb | Current unpaid principal balance |
| 4 | current_loan_delinquency_status | 0=Current, 1=30 days, 2=60 days, etc., RA=REO |
| 5 | loan_age | Months since origination |
| 6 | remaining_months_to_maturity | Months remaining |
| 7 | defect_settlement_date | Date of defect settlement |
| 8 | modification_flag | Y=Current period, P=Prior period |
| 9 | zero_balance_code | Termination reason code |
| 10 | zero_balance_effective_date | Date of termination |
| 11 | current_interest_rate | Current interest rate |
| 12 | current_non_interest_bearing_upb | Non-interest bearing UPB |
| 13 | ddlpi | Due date of last paid installment |
| 14 | mi_recoveries | Mortgage insurance recoveries |
| 15 | net_sale_proceeds | Net proceeds from sale |
| 16 | non_mi_recoveries | Non-MI recoveries |
| 17 | total_expenses | Total expenses |
| 18 | legal_costs | Legal costs |
| 19 | maintenance_costs | Maintenance and preservation costs |
| 20 | taxes_insurance | Taxes and insurance |
| 21 | misc_expenses | Miscellaneous expenses |
| 22 | actual_loss | Actual loss calculation |
| 23 | cumulative_mod_cost | Cumulative modification cost |
| 24 | interest_rate_step_indicator | Y/N for step modifications |
| 25 | payment_deferral_flag | Y/P for payment deferrals |
| 26 | eltv | Estimated current LTV |
| 27 | zero_balance_removal_upb | UPB at termination |
| 28 | delinquent_accrued_interest | Delinquent interest owed |
| 29 | delinquency_due_to_disaster | Y if disaster-related |
| 30 | borrower_assistance_status | F/R/T for workout plans |
| 31 | current_month_mod_cost | Current month modification cost |
| 32 | interest_bearing_upb | Interest-bearing UPB |

---

## Step 2: Create Survival Variables

For each loan, extract from the **last performance record**:

| Variable | Source | Logic |
|----------|--------|-------|
| `duration` | `loan_age` | Last observed loan age (months since origination) |
| `event` | `event_type` | 1 if prepay/default/other/defect, 0 if censored/matured |
| `event_type` | `zero_balance_code` + `loan_age` | Categorical: prepay, matured, default, other, censored |

**Note:** Matured loans are treated as **censored** (`event=0`) for survival modeling because maturity is deterministic (known at origination) and represents successful loan completion, not a failure event.

### Zero Balance Code Mapping

| Code | Description | Event Type |
|------|-------------|------------|
| 01 | Prepaid or Matured (Voluntary Payoff) | **Prepayment** or **Matured** (see below) |
| 02 | Third Party Sale (Foreclosure) | **Default** |
| 03 | Short Sale or Charge Off | **Default** |
| 09 | REO Disposition | **Default** |
| 15 | Whole Loan Sale | Other (exclude or censor) |
| 16 | Reperforming Loan Securitization | Other (exclude or censor) |
| 96 | Defect prior to termination | Defect (exclude) |
| (blank) | Loan still active | **Censored** |

### Distinguishing Matured from Prepaid Loans

Code 01 represents both voluntary prepayments and loans that reached maturity. To distinguish:

- **Matured**: `zero_balance_code = 01` AND `loan_age >= orig_loan_term - 3 months`
- **Prepay**: `zero_balance_code = 01` AND `loan_age < orig_loan_term - 3 months`

The 3-month threshold (`MATURITY_THRESHOLD_MONTHS`) accounts for minor timing differences in reporting.

### Competing Risks Framework

For competing risks analysis:
- **Event 1 (Prepayment)**: `event_type = 'prepay'` → `event = 1`
- **Event 2 (Default)**: `event_type = 'default'` → `event = 1`
- **Censored**: `event_type IN ('censored', 'matured')` → `event = 0`

Matured loans are grouped with censored because:
1. Maturity timing is deterministic (known at origination)
2. The loan "survived" to its natural end without prepay/default
3. There's no remaining risk to model after maturity

---

## Step 3: Merge Origination & Performance

1. **Load origination data** for all vintage years (1999-2025)
2. **Load performance data** and aggregate per loan:
   - Keep last record per `loan_sequence_number`
   - Extract `duration`, `event`, `event_type`
   - Optionally track delinquency history (ever 60+ days delinquent, etc.)
3. **Join** origination and performance on `loan_sequence_number`

---

## Step 4: Feature Engineering

### Numeric Variables
| Variable | Transformation |
|----------|---------------|
| `credit_score` | Keep as-is, set 9999 → NaN |
| `orig_ltv` | Keep as-is, set 999 → NaN |
| `orig_cltv` | Keep as-is, set 999 → NaN |
| `orig_dti` | Keep as-is, set 999 → NaN |
| `orig_upb` | Keep as-is (already rounded to $1,000) |
| `orig_interest_rate` | Keep as-is |

### Categorical Binning
| Variable | Bins |
|----------|------|
| `fico_band` | <620, 620-679, 680-739, 740-779, 780+ |
| `ltv_band` | ≤60, 61-70, 71-80, 81-90, 91-95, >95 |
| `dti_band` | ≤20, 21-30, 31-40, 41-50, >50 |

### Categorical Encoding
| Variable | Encoding |
|----------|----------|
| `occupancy_status` | One-hot: primary, second_home, investment |
| `loan_purpose` | One-hot: purchase, cash_out_refi, no_cash_out_refi |
| `property_type` | One-hot: single_family, condo, pud, manufactured, coop |
| `channel` | One-hot: retail, broker, correspondent |
| `first_time_homebuyer` | Binary: 1=Yes, 0=No |

### Derived Features
| Variable | Calculation |
|----------|-------------|
| `vintage_year` | Extract from `loan_sequence_number` (see below) |
| `loan_term_years` | `orig_loan_term / 12` |
| `is_high_ltv` | 1 if `orig_ltv > 80`, else 0 |
| `has_mi` | 1 if `mi_pct > 0`, else 0 |

### Vintage Year Extraction

The `vintage_year` is extracted from the `loan_sequence_number` which encodes the origination period:

```
Format: FYYQ#xxxxxx

F   = Freddie Mac identifier (always 'F')
YY  = Two-digit origination year
Q#  = Origination quarter (1, 2, 3, or 4)
xxx = Sequence number
```

**Examples:**
| loan_sequence_number | Parsed Year | Quarter |
|---------------------|-------------|---------|
| F99Q10012345        | 1999        | Q1      |
| F05Q32098765        | 2005        | Q3      |
| F20Q41234567        | 2020        | Q4      |

**Y2K Handling:**
- Years 91-99 → 1991-1999
- Years 00-90 → 2000-2090

This is more accurate than using `first_payment_date` because:
1. It reflects the actual origination date, not when payments began
2. The first payment date is typically 1 month after origination

---

## Step 5: Handle Missing Values

| Code | Meaning | Action |
|------|---------|--------|
| 9999 | Credit score not available | Set to NaN, consider imputation or exclusion |
| 999 | LTV/CLTV/DTI not available | Set to NaN |
| 9 | Categorical not available | Create "Unknown" category |
| (blank) | Field not populated | Handle per field |

**Strategy options:**
1. **Exclude** loans with missing critical variables (FICO, LTV)
2. **Impute** using median/mode within vintage year
3. **Create indicator** variables for missingness

---

## Step 6: Output Survival Dataset

### Final Schema
```
loan_id          | string   | Unique loan identifier
duration         | int      | Time to event in months
event            | int      | 0=censored, 1=event occurred
event_type       | string   | prepay, default, censored
vintage_year     | int      | Year of origination
credit_score     | float    | Borrower FICO score
orig_ltv         | float    | Original LTV ratio
orig_cltv        | float    | Original CLTV ratio
orig_dti         | float    | Original DTI ratio
orig_upb         | float    | Original loan amount
orig_interest_rate | float  | Original interest rate
loan_purpose     | string   | Purchase/Refi type
occupancy_status | string   | Primary/Second/Investment
property_type    | string   | Property type
property_state   | string   | State code
num_borrowers    | int      | Number of borrowers
first_time_homebuyer | int  | 1=Yes, 0=No
channel          | string   | Origination channel
has_mi           | int      | 1=Has mortgage insurance
```

### Output Files

**Combined datasets:**
- `data/processed/survival_data.csv` - Full survival dataset
- `data/processed/survival_data_default.csv` - Default events only (for cause-specific analysis)
- `data/processed/survival_data_prepay.csv` - Prepayment events only

**Per-vintage datasets (with `--by-vintage` flag):**
```
data/processed/by_vintage/
├── vintage_1999/
│   ├── survival_data_1999.csv
│   ├── survival_data_1999_default.csv
│   └── survival_data_1999_prepay.csv
├── vintage_2000/
│   └── ...
└── vintage_2025/
    └── ...
```

Each vintage directory contains:
- `survival_data_YYYY.csv` - All loans for that vintage year
- `survival_data_YYYY_default.csv` - Default events only
- `survival_data_YYYY_prepay.csv` - Prepayment events only

---

## Implementation Notes

### Memory Considerations
- Process one vintage year at a time
- Use chunked reading for large files
- Consider using Parquet format for intermediate storage

### Validation Checks
1. Verify loan counts match between origination and performance
2. Check for duplicate loan IDs
3. Validate date ranges and event timing
4. Summary statistics by vintage year

### Code Location
- Main preprocessing script: `src/data/preprocess.py`
- Column definitions: `src/data/columns.py`
- Utility functions: `src/data/utils.py`

### Usage Examples

```bash
# Process all years, combined output only
python -m src.data.preprocess --input data/raw --output data/processed

# Process a specific year
python -m src.data.preprocess --input data/raw --output data/processed --year 2020

# Process a range of years
python -m src.data.preprocess --input data/raw --output data/processed --years 2015-2020

# Process all years with per-vintage output files
python -m src.data.preprocess --input data/raw --output data/processed --by-vintage

# Combine options: specific range with vintage split, no event-type splits
python -m src.data.preprocess -i data/raw -o data/processed --years 2015-2020 --by-vintage --no-split
```
