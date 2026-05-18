# Plan: Convert Cause-Specific Cox Hazards to ALM Cash Flow Projections

## Context

The project has trained cause-specific Cox models for prepayment and default (notebook 05). The next step is to convert these hazard outputs into projected mortgage cash flows for Asset-Liability Management (ALM). This requires: (1) extracting baseline hazards from the fitted models, (2) projecting monthly cash flows per loan under a given macro scenario, (3) aggregating to portfolio level, and (4) computing interest rate risk metrics (NPV, duration, convexity) with scenario analysis capability.

## Verified Facts

- **Baseline hazard is available**: Both `ctv_prepay.baseline_cumulative_hazard_` (184 months) and `ctv_default.baseline_cumulative_hazard_` (127 of 170 months, 43 gaps) exist as DataFrames
- **Feature names**: Models use `orig_upb` (raw, not log-transformed) — 21 features total
- **`orig_loan_term`**: Available in `data/processed/survival_data_blumenstock.parquet` (76% are 360m, 17% are 180m)
- **Derived macro features**: Computed in `notebooks/03b_create_loan_month_panel.ipynb` — scenario engine must replicate this logic exactly

## New Files

### Source modules: `src/alm/`

1. **`src/alm/__init__.py`** — Package exports
2. **`src/alm/baseline_hazard.py`** — Extract and prepare baseline hazards from CoxTimeVaryingFitter
3. **`src/alm/cash_flow_engine.py`** — Core cash flow projection engine
4. **`src/alm/scenarios.py`** — Macro scenario definitions and covariate path generation
5. **`src/alm/risk_metrics.py`** — NPV, duration, convexity, WAL

### Notebook

6. **`notebooks/14_alm_cash_flows.ipynb`** — End-to-end demonstration

## Step-by-step Implementation

### Step 1: `src/alm/baseline_hazard.py`

Extract discrete baseline hazard h_0(t) from `baseline_cumulative_hazard_`:

```
H_0(t) = model.baseline_cumulative_hazard_["baseline hazard"]
h_0(t) = H_0(t) - H_0(t-1)   (for t > 1)
h_0(1) = H_0(1)               (for t = 1)
```

Reindex to complete monthly grid [1..360]:
- Fill gaps (default model has 43 missing months) with 0
- For months beyond observed data (>184 for prepay, >170 for default): hold last observed h_0 constant (configurable)

### Step 2: `src/alm/scenarios.py`

**`MacroScenario` dataclass** holding forward paths for: `MORTGAGE30US`, `DGS10`, `DGS3MO`, state HPI indices, state unemployment rates.

**`create_base_scenario()`**: Hold last observed macro values constant.

**`apply_rate_shock()`**: Parallel shift to all interest rates by +/- N bps.

**`apply_hpi_shock()`**: Gradual HPI decline over N months, then flat.

**`apply_unemployment_shock()`**: Unemployment increase over N months.

**`scenario_to_covariate_matrix()`** — Critical function. Translates a MacroScenario + loan static attributes into the 21-feature covariate matrix (N_loans x T_months x 21_features). Must replicate the exact feature derivation logic from `03b_create_loan_month_panel.ipynb`:

| Feature | Derivation |
|---------|-----------|
| `int_rate` | Static: loan's original rate |
| `orig_upb` | Static: loan's original UPB |
| `fico_score` | Static |
| `dti_r` | Static |
| `ltv_r` | Static |
| `bal_repaid` | `(orig_upb - UPB_sched(t)) / orig_upb * 100` from amortization schedule |
| `t_act_12m` | `min(12, loan_age)` (assume performing) |
| `t_del_30d_12m` | 0 (assume performing) |
| `t_del_60d_12m` | 0 (assume performing) |
| `ppi_c_FRMA` | `int_rate - MORTGAGE30US(t)` |
| `ppi_o_FRMA` | `int_rate - MORTGAGE30US(origination)` — static, already known |
| `hpi_st_d_t_o` | `hpi_state(t) - hpi_state(origination)` |
| `TB10Y_d_t_o` | `DGS10(t) - DGS10(origination)` |
| `FRMA30Y_d_t_o` | `MORTGAGE30US(t) - MORTGAGE30US(origination)` |
| `hpi_st_log12m` | `log(hpi_state(t) / hpi_state(t-12))` |
| `hpi_r_st_us` | `hpi_state(t) / hpi_national(t)` |
| `st_unemp_r12m` | `log(unemp(t) / unemp(t-12))` |
| `st_unemp_r3m` | `log(unemp(t) / unemp(t-3))` |
| `TB10Y_r12m` | `log(DGS10(t) / DGS10(t-12))` |
| `T10Y3MM` | `DGS10(t) - DGS3MO(t)` |
| `T10Y3MM_r12m` | `pct_change(T10Y3MM, 12)` |

### Step 3: `src/alm/cash_flow_engine.py`

**`MortgageCashFlowEngine` class** with config dataclass (`lgd`, `discount_rate`, `projection_horizon`).

**Algorithm per loan** (vectorized across all loans):

```
1. AMORTIZATION SCHEDULE
   monthly_rate = int_rate / 100 / 12
   payment = orig_upb * monthly_rate / (1 - (1+monthly_rate)^(-term))
   For t = 1..term:
     interest(t) = UPB(t-1) * monthly_rate
     principal(t) = payment - interest(t)
     UPB(t) = UPB(t-1) - principal(t)

2. CAUSE-SPECIFIC HAZARDS (at each month t, loan_age = current_age + t)
   risk_prepay(t) = exp(X(t) . beta_prepay)
   risk_default(t) = exp(X(t) . beta_default)
   h_prepay(t) = h0_prepay(loan_age) * risk_prepay(t)
   h_default(t) = h0_default(loan_age) * risk_default(t)
   Clip: h_total(t) = min(h_prepay(t) + h_default(t), 0.999)

3. SURVIVAL AND SUB-DENSITIES
   S(0) = 1.0
   S(t) = S(t-1) * (1 - h_total(t))
   f_prepay(t) = h_prepay(t) * S(t-1)    # unconditional prob of prepay at t
   f_default(t) = h_default(t) * S(t-1)   # unconditional prob of default at t

4. EXPECTED CASH FLOWS
   expected_interest(t) = S(t-1) * interest(t)
   expected_principal(t) = S(t-1) * principal(t)
   expected_prepay(t) = f_prepay(t) * UPB(t)          # full payoff
   expected_recovery(t) = f_default(t) * UPB(t) * (1-LGD)
   expected_loss(t) = f_default(t) * UPB(t) * LGD
   total_cf(t) = expected_interest + expected_principal + expected_prepay + expected_recovery
```

**Vectorized implementation**: Process in batches of 10k loans. Arrays are (N, T) float32 (~150 MB each for 109k x 360). Compute amortization, hazards, survival, and cash flows as matrix operations.

### Step 4: `src/alm/risk_metrics.py`

- **NPV**: `sum(total_cf(t) / (1 + r_monthly)^t)`
- **Modified duration**: `-(NPV(r+dr) - NPV(r-dr)) / (2*dr*NPV(r))` — discount rate shock only
- **Effective duration**: Same formula but also shock macro covariates (mortgage rate, treasury) and recompute hazards — captures prepayment optionality
- **Modified/Effective convexity**: `(NPV(r+dr) + NPV(r-dr) - 2*NPV(r)) / (dr^2 * NPV(r))`
- **WAL**: `sum(t * expected_principal_return(t)) / total_principal`

### Step 5: Notebook `14_alm_cash_flows.ipynb`

1. **Setup**: Load models, panel data, join `orig_loan_term`
2. **Baseline hazard validation**: Extract, visualize, sanity-check against notebook 04 CIF
3. **Single loan walkthrough**: Full algorithm on one loan, visualize each step
4. **Portfolio projection** (base scenario): Aggregate monthly CFs, compute risk metrics
5. **Scenario analysis**: 8 scenarios (base, +/-100bp, +/-200bp, HPI stress, recession, severe recession)
6. **Risk metrics comparison**: Table + charts of NPV/duration/convexity across scenarios
7. **Portfolio segmentation**: Risk metrics by vintage, FICO band, LTV band

## Key Files to Reference During Implementation

| File | Why |
|------|-----|
| `notebooks/05_cause_specific_cox.ipynb` | Model training, feature list, how predictions are computed |
| `notebooks/03b_create_loan_month_panel.ipynb` | Exact macro feature derivation logic to replicate in scenarios |
| `src/competing_risks/cumulative_incidence.py` | CIF computation for validation |
| `models/cox_prepay_tv.pkl`, `models/cox_default_tv.pkl` | Fitted models |
| `data/processed/loan_month_panel.parquet` | Panel data with last observed covariate values |
| `data/processed/survival_data_blumenstock.parquet` | Source for `orig_loan_term` |

## Verification

1. **Baseline hazard**: Plot extracted h_0(t) for both models; CIF from baseline alone should roughly match non-parametric Aalen-Johansen from notebook 04
2. **Single loan**: Verify amortization schedule sums to orig_upb; verify S(T) + CIF_prepay(T) + CIF_default(T) = 1
3. **Cash flow accounting identity**: At each t, total expected CF should decrease over time as survival decreases
4. **Risk metrics sanity**: Modified duration should be close to WAL; effective duration < modified duration (negative convexity from prepayment); NPV under rate-up scenario should be > NPV under rate-down (prepayments slow when rates rise)
5. **Scenario ordering**: Prepayment should accelerate when rates fall, slow when rates rise
