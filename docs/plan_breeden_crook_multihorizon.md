# Implementation Plan: Breeden & Crook (2022) Multihorizon Discrete-Time Survival Model

## Paper Reference

**Breeden, J.L. and Crook, J.N. (2022).** *Multihorizon discrete time survival models.* Journal of the Operational Research Society.

---

## Paper Summary

The paper proposes a **multihorizon discrete-time survival model** that fits **separate logistic regressions for each forecast horizon** $L = 1, \ldots, 12$. The key innovation is that model $L$ uses delinquency lagged by $L$ months, revealing how:

- **Short horizons**: Delinquency dominates predictions (strong roll-rate/state-transition effect)
- **Long horizons**: Origination variables (LTV, DTI, FICO) take over as delinquency's predictive power decays

This bridges the gap between:
- **Roll-rate/state-transition models** (accurate short-term via delinquency)
- **Vintage/APC models** (accurate long-term via origination quality)

### Competing Risks Adaptation

We extend the model for two competing events:
- **k = 1**: Prepayment
- **k = 2**: Default (90+ DPD)

Total models: 2 risks × (12 horizons + 1 origination) = **26 logistic regressions**

---

## Model Components

| Component | Standard (NB 12) | APC Two-Stage (NB 13) |
|-----------|-----------------|----------------------|
| Lifecycle $F(a)$ | B-spline basis (5 knots, 7 cols) | 1 smooth scalar from backfitting |
| Vintage $G(v)$ | Year dummies (15+ cols) | 1 smooth scalar from backfitting |
| Environment $H(t)$ | 12 raw macro variables | 1 smooth scalar from backfitting |
| Static origination | FICO, DTI, LTV, rate, log(UPB) | Same |
| Behavioral | `bal_repaid_lag1`, `t_act_12m`, rolling counts | Same |
| **Delinquency (lag $L$)** | $D_{1m}^{lag L}$, ..., $D_{5m+}^{lag L}$ | Same |

### Feature Details

**Static origination features**: `fico_score`, `dti_r`, `ltv_r`, `int_rate`, `log_orig_upb`

**Behavioral features**: `bal_repaid_lag1`, `t_act_12m`, `t_del_30d_12m`, `t_del_60d_12m`

**Macro/environment features** (12 variables):
`hpi_st_d_t_o`, `ppi_c_FRMA`, `TB10Y_d_t_o`, `FRMA30Y_d_t_o`, `ppi_o_FRMA`, `hpi_st_log12m`, `hpi_r_st_us`, `st_unemp_r12m`, `st_unemp_r3m`, `TB10Y_r12m`, `T10Y3MM`, `T10Y3MM_r12m`

**Delinquency indicators**: `D_1m`, `D_2m`, `D_3m`, `D_4m`, `D_5m_plus` (binary indicators for current delinquency status buckets, lagged by $L$ months for horizon model $L$)

---

## Step-by-Step Implementation Plan

### **Step 1: Data Preparation**

**Goal**: Enrich the existing loan-month panel with delinquency status from raw performance files.

**Tasks**:
1. Load `loan_month_panel.parquet` (existing processed panel)
2. Extract `current_loan_delinquency_status` from raw Freddie Mac performance files
3. Merge delinquency status into panel (cache result as `loan_month_panel_with_delinq.parquet`)
4. Create binary delinquency state indicators: `D_1m`, `D_2m`, `D_3m`, `D_4m`, `D_5m_plus`
5. Create lagged delinquency indicators for each horizon $L = 1, \ldots, 12$
6. Create additional derived features (`log_orig_upb`, `bal_repaid_lag1`)

**Output**: Enriched panel with columns `D_Xm_lagL` for all delinquency states × lag horizons.

**Implementation**: `enrich_panel_with_delinquency()`, `create_delinquency_indicators()`, `create_lagged_delinquency()` in `src/competing_risks/breeden_crook.py`

---

### **Step 2: APC Exploration**

**Goal**: Visualize Age-Period-Cohort structure before fitting the model.

**Tasks**:
1. **Age effect $F(a)$**: Plot empirical hazard rate by loan age (months) — maturation pattern
2. **Vintage effect $G(v)$**: Plot hazard by origination year — cohort quality
3. **Calendar time effect $H(t)$**: Plot hazard by calendar time — economic environment

Generate plots for both default and prepayment risks.

**Output**: `reports/figures/breeden_crook_apc_exploration.png`

---

### **Step 3: Model Training**

**Goal**: Fit 26 logistic regressions (2 risks × 13 horizons).

#### Model Structure

For each risk $k \in \{1, 2\}$ and horizon $L \in \{0, 1, \ldots, 12\}$:

$$P(Y_{t+L} = k \mid X_t) = \text{logistic}\left(\beta_0 + F(a) + G(v) + H(t) + \beta_{\text{static}}' X_{\text{static}} + \beta_{\text{behav}}' X_{\text{behav}} + \beta_{\text{delinq}}' D^{\text{lag}L}\right)$$

#### Horizon Types

- **Origination model** (horizon 0): No delinquency features; used for loans < 6 months old
- **Horizon models** ($L = 1, \ldots, 12$): Include $D_{Xm}^{\text{lag}L}$ delinquency indicators

#### Feature Construction

- **Lifecycle $F(a)$**: B-spline basis on loan age with 5 knots (`SplineTransformer`)
- **Vintage $G(v)$**: Year dummies for origination year
- **Environment $H(t)$**: 12 standardized macro variables
- **Static**: FICO, DTI, LTV, interest rate, log(UPB)
- **Behavioral**: Balance repaid (lag 1), activity/delinquency rolling counts
- **Delinquency**: 5 binary indicators, lagged by $L$ months

#### Regularization

For each horizon and risk, tune regularization parameter $C \in \{0.01, 0.1, 1.0, 10.0\}$ using validation fold AUC.

#### Train/Validation/Test Split

- **Train**: Folds 0-8
- **Validation**: Fold 9 (for $C$ tuning)
- **Test**: Fold 10

**Implementation**: `BreedenCrookMultihorizon.fit()` in `src/competing_risks/breeden_crook.py`

---

### **Step 4: Coefficient Analysis Across Horizons**

**Goal**: Reproduce the paper's central findings (Figures 5-9).

#### 4.1 Delinquency Coefficients vs. Horizon (Paper Fig. 5)

Show how delinquency's predictive power **decays** with forecast horizon:
- At $L = 1$: 3-month delinquent loan has very high default probability
- At $L = 12$: Same delinquency status is much less predictive

**Output**: `reports/figures/breeden_crook_coef_delinquency.png`

#### 4.2 Origination Variable Coefficients vs. Horizon (Paper Fig. 6)

Show how origination variables' predictive power **increases** with horizon as delinquency loses dominance.

**Output**: `reports/figures/breeden_crook_coef_origination.png`

#### 4.3 Model Fit by Horizon (Paper Figs. 8-9)

Gini coefficient (2×AUC − 1) shows discriminatory power declining with horizon.

**Output**: `reports/figures/breeden_crook_pseudo_r2.png`

**Implementation**: `plot_delinquency_coefficients()`, `plot_origination_coefficients()`, `plot_pseudo_r2_by_horizon()` in `src/competing_risks/breeden_crook.py`

---

### **Step 5: CIF Prediction**

**Goal**: Generate cumulative incidence function predictions for test loans.

#### Method

For each test loan at forecast origin $t_0$:
1. Use horizon-$L$ model ($L = 1, \ldots, 12$) to predict conditional hazard $h_k(t_0 + L)$
2. For months beyond $L = 12$, extrapolate using the $L = 12$ model
3. Convert conditional hazards to CIF:

$$\text{CIF}_k(t) = \sum_{s=1}^{t} h_k(s) \cdot S(s-1)$$

where $S(t) = \prod_{s=1}^{t} \left[1 - \sum_k h_k(s)\right]$ is the overall survival.

#### Validity Check

Verify $\text{CIF}_{\text{default}}(t) + \text{CIF}_{\text{prepay}}(t) + S(t) = 1$ for all $t$.

**Implementation**: `BreedenCrookMultihorizon.predict_cif()` in `src/competing_risks/breeden_crook.py`

---

### **Step 6: Performance Evaluation**

**Goal**: Evaluate using metrics consistent with other notebooks for comparison.

#### 6.1 Time-Dependent C-Index

Compute at time horizons $\tau = 24, 48, 72$ months for both risks.

#### 6.2 Brier Scores

Time-dependent Brier score at $\tau = 24, 48, 72$.

#### 6.3 Calibration Plots

Predicted vs observed event rates at $\tau = 48$.

#### 6.4 Model Comparison

Compare C-index results with:
- Cause-Specific Cox
- Fine-Gray
- DeepHit
- Dynamic-DeepHit

**Implementation**: Reuse `src/competing_risks/evaluation.py` (`time_dependent_concordance_index`, `brier_score_competing_risks`, `calibration_plot`)

---

### **Step 7: Diagnostic Plots**

**Goal**: Visualize model predictions and calibration.

1. **Sample CIF curves**: Default CIF, Prepay CIF, and Survival for 5 random test loans
2. **Calibration plots**: Predicted vs observed at $\tau = 48$ for both risks
3. **Average CIF by horizon**: Mean CIF and survival with IQR bands across test set
4. **C-index bar chart**: Side-by-side comparison at each horizon

**Output**: `reports/figures/breeden_crook_cif_curves.png`, `reports/figures/breeden_crook_cindex_comparison.png`

---

## Code Structure

### Source Module: `src/competing_risks/breeden_crook.py`

```
Functions:
  enrich_panel_with_delinquency()  - Extract delinq status from raw files
  create_delinquency_indicators()  - Create D_1m, ..., D_5m_plus indicators
  create_lagged_delinquency()      - Create lag-L versions for each horizon
  build_feature_matrix()           - Standard feature matrix (~34 APC cols)
  build_feature_matrix_apc()       - APC feature matrix (3 scalar cols)
  plot_delinquency_coefficients()  - Paper Fig. 5 reproduction
  plot_origination_coefficients()  - Paper Fig. 6 reproduction
  plot_pseudo_r2_by_horizon()      - Paper Figs. 8-9 reproduction

Classes:
  BreedenCrookMultihorizon
    .fit(train_panel, val_panel)    - Fit 26 logistic regressions
    .predict_cif(df, max_months, future_macro) - Generate CIF predictions
    .get_coefficients()             - Extract coefficient DataFrame
    .get_pseudo_r2()                - Get validation metrics by horizon
    # APC mode: pass use_apc=True, apc_default, apc_prepay to __init__
```

### Notebook: `notebooks/12_breeden_crook_multihorizon.ipynb`

```
1. Overview & imports
2. Configuration
3. Data Preparation (load, enrich, lag, split)
4. APC Exploration (Age, Vintage, Calendar time plots)
5. Multihorizon Model Training (26 logistic regressions)
6. Coefficient Analysis (delinquency decay, origination increase, Gini)
7. CIF Prediction & Evaluation (C-index, Brier score, calibration)
8. Diagnostic Plots (sample CIF curves, average CIF, comparison chart)
9. Summary
```

### Evaluation: `src/competing_risks/evaluation.py` (existing, reused)

```
  time_dependent_concordance_index()
  brier_score_competing_risks()
  calibration_plot()
  evaluate_all_events()
```

---

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_HORIZON` | 12 | Forecast horizons $L = 1, \ldots, 12$ |
| `CIF_HORIZON` | 72 | Maximum CIF prediction horizon (months) |
| `N_AGE_KNOTS` | 5 | B-spline knots for lifecycle $F(a)$ |
| `C_VALUES` | [0.01, 0.1, 1.0, 10.0] | Regularization grid for logistic regression |
| `SOLVER` | lbfgs | Logistic regression solver |
| `TRAIN_FOLDS` | 0-8 | Training folds |
| `VAL_FOLDS` | [9] | Validation fold (for $C$ tuning) |
| `TEST_FOLD` | 10 | Held-out test fold |

---

## Key Differences from Other Models in This Project

| Aspect | Cox / Fine-Gray / DeepHit | Breeden-Crook Multihorizon |
|--------|---------------------------|----------------------------|
| **Model type** | Single model for all horizons | Separate model per horizon |
| **Delinquency** | Current state or not used | Lagged by horizon $L$ |
| **Key insight** | — | Delinquency decays, origination grows with horizon |
| **Number of models** | 1 per risk | 13 per risk (26 total) |
| **Inference** | Varies (MLE, neural net) | Logistic regression (fast, interpretable) |
| **APC decomposition** | Not explicit | Explicit via splines + dummies + macro vars |

---

## Expected Outputs

1. **Notebook**: `notebooks/12_breeden_crook_multihorizon.ipynb`
2. **Source module**: `src/competing_risks/breeden_crook.py`
3. **Figures**:
   - `reports/figures/breeden_crook_apc_exploration.png`
   - `reports/figures/breeden_crook_coef_delinquency.png`
   - `reports/figures/breeden_crook_coef_origination.png`
   - `reports/figures/breeden_crook_pseudo_r2.png`
   - `reports/figures/breeden_crook_cif_curves.png`
   - `reports/figures/breeden_crook_cindex_comparison.png`

---

## Bayesian APC Decomposition (Stage 1)

The original implementation in notebook 12 treats APC components as raw features concatenated into the logistic regression (B-splines for age, year dummies for vintage, 12 macro variables for calendar time). Breeden's paper actually uses a **two-stage approach**:

1. **Stage 1**: Extract smooth $F(a)$, $G(v)$, $H(t)$ at the portfolio level via iterative backfitting with penalized smoothing splines (GAM-style; the roughness penalty is equivalent to an RW2 Bayesian prior — hence "Bayesian APC"). This is **not** full MCMC inference.
2. **Stage 2**: Use the extracted APC values as 3 scalar features in the horizon-specific logistic regressions, replacing ~34 raw APC columns.

### Source Module: `src/competing_risks/apc_decomposition.py`

```
Functions:
  aggregate_portfolio_rates()      - Group panel to (age, vintage, caltime) cells
  _weighted_group_mean()           - Weighted mean within group levels
  _smooth_1d()                     - Fit UnivariateSpline with weights (GCV smoothing)

Classes:
  BreedenAPC
    .fit(panel, event_code)        - Iterative backfitting on logit(rate)
    .transform(panel)              - Add F_age, G_vintage, H_caltime columns
    .fit_macro_regression(panel, macro_cols) - OLS of H(t) on macro variables
    .predict_H_from_macro(macro_values)     - Out-of-sample H(t) prediction
    .plot_components(axes)         - Visualize F(a), G(v), H(t) curves

Stored attributes after fit:
  intercept_, F_spline_, G_spline_, H_spline_,
  F_values_, G_values_, H_values_,
  cal_time_map_, convergence_history_, n_iterations_,
  cells_, macro_regression_
```

### Backfitting Algorithm

Operates on aggregated (age, vintage, caltime) cells (~thousands, not millions of loan-months):

1. Aggregate panel to cells, compute `rate = n_events / n_at_risk`
2. Work on logit scale: $y = \text{logit}(\text{clip}(\text{rate}, \epsilon, 1-\epsilon))$
3. Initialize: $\hat{\mu} = \text{weighted\_mean}(y)$, $F = G = H = 0$
4. Iterate until convergence ($\max(|\Delta F|, |\Delta G|, |\Delta H|) < \text{tol}$):
   - $F(a) \leftarrow \text{smooth}(y - \hat{\mu} - G - H \text{ by age})$
   - $G(v) \leftarrow \text{smooth}(y - \hat{\mu} - F - H \text{ by vintage})$
   - $H(t) \leftarrow \text{smooth}(y - \hat{\mu} - F - G \text{ by caltime})$
   - Zero-mean constraints: absorb means of $F$, $G$, $H$ into $\hat{\mu}$

### Modifications to `breeden_crook.py`

- **`build_feature_matrix_apc()`**: Replaces ~34 APC columns with 3 scalar features (`F_age`, `G_vintage`, `H_caltime`). Static, behavioral, and lagged delinquency features remain unchanged.
- **`BreedenCrookMultihorizon.__init__`**: Added `apc_default`, `apc_prepay`, `use_apc` parameters.
- **`_fit_horizon`**: Branches to `_fit_horizon_standard` (original) or `_fit_horizon_apc` (new). APC mode uses risk-specific APC objects for feature construction.
- **`_predict_hazard_at_horizon`**: Branches to `_predict_hazard_at_horizon_apc` when `use_apc=True`.
- **`predict_cif`**: Added `future_macro` parameter for out-of-sample H(t) prediction.
- **`get_coefficients`**: Handles APC feature names.

### Modifications to `__init__.py`

Added imports: `BreedenAPC`, `aggregate_portfolio_rates`, `build_feature_matrix_apc`

### Notebook: `notebooks/13_breeden_apc_decomposition.ipynb`

```
1. Data loading & enrichment (reuse from notebook 12)
2. Fit BreedenAPC for default and prepay separately
3. Visualize F(a), G(v), H(t) curves + convergence history
4. Identity check: F + G + H + intercept ≈ logit(observed rate) (R²)
5. Zero-mean check on components
6. Macro regression for H(t) — R², coefficients, residual plot
7. Fit BreedenCrookMultihorizon with use_apc=True
8. Coefficient analysis: APC vs standard model comparison
9. CIF prediction & evaluation (C-index, Brier scores)
10. Comparison table: APC vs non-APC at tau=24, 48, 72
```

### Key Design Decisions

1. **Feature approach, not offset**: $F$, $G$, $H$ are features (not fixed offsets), letting each horizon adjust APC effects slightly.
2. **Aggregation for efficiency**: Backfitting operates on ~thousands of cells, not millions of loan-months. Each iteration takes milliseconds.
3. **Separate APC per risk**: Default and prepay have different lifecycle/vintage/environment profiles.
4. **Scipy UnivariateSpline with GCV**: No new dependencies. Smoothing can be manually overridden.
5. **Future H(t) via macro regression**: WLS regression of H(t) on macro variables for out-of-sample calendar times.

### Verification Criteria

1. Convergence in <50 iterations
2. Identity: $R^2 > 0.8$ at cell level
3. Zero-mean: weighted mean of $F$, $G$, $H$ ≈ 0
4. Visual: $F(a)$ shows maturation hump, $G(v)$ shows vintage quality, $H(t)$ tracks cycles
5. Macro regression $R^2 > 0.5$
6. C-index: APC version matches or improves on standard at $\tau = 24, 48, 72$
7. CIF validity: $\text{CIF}_{\text{def}} + \text{CIF}_{\text{pre}} + S = 1$

### Expected Additional Outputs

- `reports/figures/breeden_apc_components.png`
- `reports/figures/breeden_apc_macro_regression.png`
- `reports/figures/breeden_apc_coefficients.png`
- `reports/figures/breeden_apc_vs_standard_cindex.png`

---

## Status

**Status**: ✅ Implemented

- Source module (`src/competing_risks/breeden_crook.py`) — complete
- APC decomposition module (`src/competing_risks/apc_decomposition.py`) — complete
- Notebook 12 (`notebooks/12_breeden_crook_multihorizon.ipynb`) — complete (standard model)
- Notebook 13 (`notebooks/13_breeden_apc_decomposition.ipynb`) — complete (APC two-stage model)
- APC exploration figures generated

---

## References

1. Breeden, J.L. and Crook, J.N. (2022). Multihorizon discrete time survival models. *Journal of the Operational Research Society*.
2. Allison, P.D. (1982). Discrete-time methods for the analysis of event histories. *Sociological Methodology*, 13, 61-98.
3. Fine, J.P. and Gray, R.J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA*, 94(446), 496-509.
