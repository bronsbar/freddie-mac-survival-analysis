# Competing Risks Survival Analysis for Mortgage Prepayment and Default

> **Status**: 🔄 Active Implementation
> **Last Updated**: January 2026

## Overview

This project replicates the methodology from **Blumenstock, Lessmann & Seow (2022)** "Deep learning for survival and competing risk modelling" (Journal of the Operational Research Society) using **Dataset 2 (post-crisis period: 2010-2025)**.

### Objective

Compare statistical and machine learning survival models for predicting mortgage prepayment and default as competing risks:

| Model | Type | Description |
|-------|------|-------------|
| **Cause-Specific Cox (CSC)** | Statistical | Semi-parametric Cox model, treats competing events as censored |
| **Fine-Gray (FGR)** | Statistical | Subdistribution hazard model, properly handles competing risks |
| **Random Survival Forest (RSF)** | ML | Ensemble method using Gray's log-rank splitting for competing risks |

**Note**: DeepHit (deep learning model) is excluded from this implementation phase.

---

## Dataset: Post-Crisis Period (2010-2025)

### Why Dataset 2?

Per the paper, the US government launched loan modification programs after 2009 that changed the default definition and borrower behavior. This creates a natural data partition:

- **Dataset 1 (1999-2009)**: Pre-crisis and crisis period
- **Dataset 2 (2010-2025)**: Post-crisis period with government intervention programs

We focus on **Dataset 2** which has:
- Lower default rates due to renegotiation options
- Different censoring patterns (more recent loans)
- More relevant for current mortgage risk modeling

### Event Definitions (from paper)

| Event | Code | Definition |
|-------|------|------------|
| **Censored** | 0 | Active loan not experiencing event until observation time |
| **Prepayment** | 1 | Loan repaid completely and unexpectedly |
| **Default** | 2 | Loan turning 3-month delinquent for the first time |

---

## Variables (Table 2 from Paper)

### Loan-Level Variables (9 features)

| Variable | Description | Type |
|----------|-------------|------|
| `int_rate` | Initial interest rate | Static |
| `orig_upb` | Original unpaid balance | Static |
| `fico_score` | Initial FICO score | Static |
| `dti_r` | Initial debt-to-income ratio | Static |
| `ltv_r` | Initial loan-to-value ratio | Static |
| `bal_repaid` | Current repaid balance in percent | Time-varying |
| `t_act_12m` | Number of times not being delinquent in last 12 months | Time-varying |
| `t_del_30d_12m` | Number of times being 30 days delinquent in last 12 months | Time-varying |
| `t_del_60d_12m` | Number of times being 60 days delinquent in last 12 months | Time-varying |

### Macroeconomic Variables (13 features for Dataset 2)

Note: Some variables from the paper (hpi.zip.o, hpi.zip.d.t.o, equity.est, hpi.r.zip.st) are not available for Dataset 2.

| Variable | Description | Source |
|----------|-------------|--------|
| `hpi_st_d_t_o` | Difference of HPI between origination and today (state-level) | FHFA |
| `ppi_c_FRMA` | Current prepayment incentive (int_rate - current FRM average) | FRED |
| `TB10Y_d_t_o` | Difference of 10-year treasury rate between origination and today | FRED |
| `FRMA30Y_d_t_o` | Difference of 30-year FRM average between origination and today | FRED |
| `ppi_o_FRMA` | Prepayment incentive at origination | FRED |
| `hpi_st_log12m` | HPI 12-month log return (state-level) | FHFA |
| `hpi_r_st_us` | Ratio of HPI between state-level and US-level today | FHFA |
| `st_unemp_r12m` | Unemployment rate 12-month log return (state-level) | BLS |
| `st_unemp_r3m` | Unemployment rate 3-month log return (state-level) | BLS |
| `TB10Y_r12m` | Current 10-year treasury rate 12-month return | FRED |
| `T10Y3MM` | Yield between 3-month and 10-year treasury rates today | FRED |
| `T10Y3MM_r12m` | Yield curve 12-month return | FRED |

---

## Experimental Design

Following the paper's structure for Dataset 2:

### Experiments

| Experiment | Variables | Purpose |
|------------|-----------|---------|
| **Exp 4.1** | Loan-level only (9 vars) | Assess predictive power of loan characteristics |
| **Exp 4.2** | Macroeconomic only (13 vars) | Assess predictive power of macro conditions |
| **Exp 4.3** | All variables (22 vars) | Combined model performance |

### Data Sampling Strategy

Following the paper:
- **Sample size**: ~10,000 observations per fold
- **Train/Test split**: 80%/20%
- **Validation**: Cross-validation with one fold reserved for hyperparameter tuning
- **Subsampling**: 11 random subsamples from the dataset

### Train/Test Split Design

```
Dataset 2: 2010-2025
├── 11 random subsamples (size: 10,000 each)
│   ├── Sample 1 → Train (8,000) / Test (2,000)
│   ├── Sample 2 → Train (8,000) / Test (2,000)
│   ├── ...
│   └── Sample 11 → Hyperparameter tuning
```

---

## Evaluation Metrics

### Time-Dependent Concordance Index

The primary metric from the paper. Evaluates model discrimination at specific time points.

**Evaluation time points**: 24, 48, 72 months after loan issuance

**Formula**:
```
C_k^td(t) = Σ A_{k,i,j} * 1(F_k(t|x_i) > F_k(t|x_j)) / Σ A_{k,i,j}
```

Where:
- `A_{k,i,j} = 1(k_i = k, t_i < t_j)` filters comparable pairs
- `F_k(t|x)` is the predicted cumulative incidence

### Reporting Structure

For each experiment, report:

| Metric | Prepay (k=1) | Default (k=2) | Combined |
|--------|--------------|---------------|----------|
| C(24) | C₁(24) | C₂(24) | - |
| C(48) | C₁(48) | C₂(48) | - |
| C(72) | C₁(72) | C₂(72) | - |
| ØC | ØC₁ | ØC₂ | **ØC** |

- ØC₁, ØC₂: Mean across time points for each event
- ØC: Total mean across both events

---

## Model Specifications

### 1. Cause-Specific Cox (CSC)

```python
from lifelines import CoxPHFitter

# For prepayment: treat default as censored
# For default: treat prepayment as censored
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df, duration_col='duration', event_col='event')
```

### 2. Fine-Gray Model (FGR)

Using discrete-time approximation with logistic regression:

```python
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Subdistribution hazard: competing events stay in risk set
fg_model = LogisticRegression(penalty='l2', C=1.0)
fg_model.fit(X_train, y_train)
```

### 3. Random Survival Forest (RSF)

Using scikit-survival or custom implementation:

```python
from sksurv.ensemble import RandomSurvivalForest

# For competing risks, use separate models or custom implementation
rsf = RandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
```

**Note**: For proper competing risks RSF, implement Gray's log-rank splitting rule.

---

## Implementation Roadmap

### Phase 1: Data Preparation ✅

- [x] Filter Freddie Mac data for 2010-2025 originations
- [x] Create loan-month panel with time-varying covariates
- [x] Calculate behavioral variables (delinquency counts, balance repaid)
- [x] Merge macroeconomic data (state HPI, unemployment, treasury rates)
- [x] Calculate derived features (prepayment incentive, HPI changes)
- [x] Define event coding (prepay=1, default=2, censored=0)
- [x] Create train/test samples following paper's design

**Implemented in**: `notebooks/03_data_preparation_blumenstock.ipynb`

### Phase 2: Implement Models ✅

- [x] Cause-Specific Cox for prepayment and default
- [x] Fine-Gray discrete-time model
- [x] Random Survival Forest for competing risks
- [x] Implement time-dependent concordance index

**Implemented in**:
- `src/competing_risks/cause_specific.py` - CSC wrapper
- `src/competing_risks/fine_gray.py` - Discrete-time Fine-Gray
- `src/competing_risks/random_forest.py` - RSF for competing risks
- `src/competing_risks/evaluation.py` - Time-dependent C-index at 24, 48, 72 months

### Phase 3: Run Experiments ✅

- [x] Experiment 4.1: Loan-level variables only
- [x] Experiment 4.2: Macroeconomic variables only
- [x] Experiment 4.3: All variables combined

**Implemented in**: `notebooks/08_model_comparison.ipynb`

### Phase 4: Evaluation & Comparison 🔄

- [x] Calculate C-index at 24, 48, 72 months
- [ ] Compare model rankings across experiments
- [ ] Statistical significance testing (pairwise t-test)
- [ ] Feature importance analysis

### Phase 5: Documentation

- [ ] Document methodology differences from paper
- [ ] Interpret coefficient differences (FGR vs CSC)
- [ ] Create visualization of cumulative incidence functions
- [ ] Write findings summary

---

## Code Structure

```
notebooks/
├── 03_data_preparation_blumenstock.ipynb   # Data prep following paper
├── 04_nonparametric_cif.ipynb              # Aalen-Johansen CIF estimation
├── 05_cause_specific_cox.ipynb             # CSC models
├── 06_fine_gray_model.ipynb                # FGR model
├── 07_random_survival_forest.ipynb         # RSF model (NEW)
├── 08_model_comparison.ipynb               # Compare all models

src/
├── competing_risks/
│   ├── __init__.py
│   ├── data_prep.py                        # Paper-style data preparation
│   ├── fine_gray.py                        # Discrete-time FG
│   ├── cause_specific.py                   # Cause-specific Cox
│   ├── random_forest.py                    # RSF for competing risks (NEW)
│   ├── cumulative_incidence.py             # CIF calculations
│   └── evaluation.py                       # Time-dependent C-index
```

---

## Key Differences from Original Paper

| Aspect | Paper (Blumenstock 2022) | This Implementation |
|--------|--------------------------|---------------------|
| Period | 2010-2017 | 2010-2025 (extended) |
| DeepHit | Included | Excluded (for now) |
| Sample size | 600,000 total | Based on available Freddie Mac data |
| Fine-Gray | Continuous time | Discrete-time approximation via logistic regression |
| RSF | Custom competing risks | Cause-specific RSF using scikit-survival |
| Some macro vars | Zip-level HPI | State-level HPI (proxy) |

---

## Next Steps

1. **Run the model comparison notebook** (`08_model_comparison.ipynb`) to generate results
2. **Compare model rankings** across experiments 4.1, 4.2, 4.3
3. **Statistical significance testing** using pairwise t-tests on concordance indices
4. **Feature importance analysis** for RSF and coefficient comparison for CSC/FGR
5. **Documentation** of findings and methodology differences

---

## References

1. **Blumenstock, G., Lessmann, S., & Seow, H-V. (2022)**. Deep learning for survival and competing risk modelling. *Journal of the Operational Research Society*, 73(1), 26-38.

2. **Fine, J.P. and Gray, R.J. (1999)**. A Proportional Hazards Model for the Subdistribution of a Competing Risk. *JASA*, 94(446), 496-509.

3. **Ishwaran, H., et al. (2014)**. Random survival forests for competing risks. *Biostatistics*, 15(4), 757-773.

4. **Antolini, L., et al. (2005)**. A time-dependent discrimination index for survival data. *Statistics in Medicine*, 24(24), 3927-3944.
