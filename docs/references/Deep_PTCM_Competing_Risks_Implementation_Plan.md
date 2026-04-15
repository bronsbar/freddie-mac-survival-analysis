# Plan: Deep Promotion Time Cure Model (Deep-PTCM) for Competing Risks

## Context

The project implements 20 mortgage survival analysis models. Medina-Olivares et al. (2024) propose the Deep-PTCM -- a cure model that captures the fraction of borrowers who will *never* experience an event (cured), while flexibly modeling time-to-event for the rest via a DNN. The paper applies it to the same Freddie Mac dataset but only models a single event (default). We will extend it to competing risks (prepayment + default) and benchmark against the existing models.

**Key insight**: The PTCM is a loan-level model (one row per loan, static covariates), not a monthly-panel model. This makes it structurally different from Sadhwani/NN-DTSM but similar to how DeepHit uses terminal observations.

---

## Step 1: Create `src/competing_risks/deep_ptcm.py`

The core module implementing the model in PyTorch. Contains:

### 1a. `PiecewiseExponentialBaseline(nn.Module)`
Parameterizes the latent activation time CDF F_k(t) per cause.
- Divides [0, T_max] into J intervals with breakpoints (e.g., every 12 months: 0, 12, 24, ..., 180, 184+)
- Learnable parameters: `log_lambdas` (J values), mapped to positive via softplus
- `forward(t)` returns `(F_t, f_t)` -- the CDF and density at observed times

### 1b. `DeepPTCMNetwork(nn.Module)`
Shared DNN + cause-specific heads producing theta_k(x) > 0.
- Shared layers: `[128, 64]` with BatchNorm + ReLU + Dropout
- Per-cause heads: `[32]` outputting scalar g_k(x)
- `theta_k(x) = exp(g_k(x))`
- Optional orthogonalization: decompose g_k = w_k^T x + b_k + ort_k(x), where ort_k is the DNN residual projected orthogonal to the column space of X

### 1c. `PTCMLoss(nn.Module)`
Negative log-likelihood for competing risks PTCM:
```
ell_i = sum_k delta_ik * [log(theta_k) + log(f_k(t_i))]  -  sum_k theta_k * F_k(t_i)
```
where delta_ik = 1 if subject i has event k, and the second term applies to all subjects (survival contribution).

### 1d. `CompetingRisksDeepPTCM(BaseEstimator)`
scikit-learn-compatible wrapper matching the project's interface pattern (cf. `deephit.py`):
- `fit(X, duration, event_code, event_types=[1,2], val_data=None)`
- `predict_risk(X, event, time=None)` -- CIF at time t as risk score
- `predict_cumulative_incidence(X, event, times)` -- full CIF curve via trapezoidal integration
- `predict_survival(X, times)` -- S(t) = exp(-sum_k theta_k F_k(t))
- `predict_cure_fraction(X)` -- pi_k(x) = exp(-theta_k(x)) per cause
- `get_linear_coefficients()` -- extract w_k from orthogonalized model
- Training: Adam optimizer, ReduceLROnPlateau scheduler, early stopping on validation NLL

### 1e. `fit_deep_ptcm_competing_risks(df, feature_cols, ...)` convenience function

**Key files to reference for patterns**:
- `src/competing_risks/deephit.py` -- API pattern (fit/predict_risk/predict_cif/predict_survival)
- `src/competing_risks/sadhwani_net.py` -- feature constants, `get_device()`
- `src/competing_risks/evaluation.py` -- metrics to reuse

---

## Step 2: Create `notebooks/21_deep_ptcm.ipynb`

**Data**: `data/processed/blumenstock_dataset2.parquet` (110K loans, 11 folds, event_code 0/1/2). This is the same loan-level dataset used by the DeepHit notebook (08) for direct comparability.

**Features**: `['int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r']` (5 static origination features, consistent with all other notebooks). Derive `log_upb = log(orig_upb)`.

**Split**: folds 0-8 train, fold 9 validation, fold 10 test (matching DeepHit/Sadhwani).

### Notebook sections:

1. **Introduction & Methodology** -- Paper citation, PTCM theory, competing risks extension
2. **Configuration** -- Hyperparameters, paths, feature lists, fold constants
3. **Data Loading** -- Load blumenstock_dataset2, derive log_upb, event distribution summary
4. **Train/Val/Test Split** -- Fold-based split, StandardScaler
5. **Piecewise Exponential Intervals** -- Define breakpoints (every 12 months)
6. **Model Training (Deep-PTCM)** -- Train base model, plot loss curves
7. **Cure Fraction Analysis** -- Compute pi_k(x), visualize distributions, AUC_cure metric
8. **Model Evaluation** -- Time-dependent C-index at 24/48/72, Brier score, IBS
9. **Survival & CIF Curves** -- Plot predicted CIF/survival for example loans
10. **Orthogonalized Model (Deep-PTCM-Ort)** -- Train variant, extract linear coefficients, compare
11. **Feature Importance** -- Permutation importance on test set
12. **Model Comparison** -- Compare with DeepHit, Cox, RSF using same evaluation metrics
13. **Save Models** -- Save state dicts + metadata to `models/`
14. **Summary** -- Results table

---

## Step 3: Add new metrics to `src/competing_risks/evaluation.py`

Two new functions specific to cure models:

- `auc_cure(event_codes, durations, predicted_cure_prob, min_followup=120)` -- AUC for distinguishing cured (censored with long follow-up) vs uncured (experienced event) borrowers
- `integrated_brier_score(event_times, event_codes, predicted_cif_func, event_of_interest, t_max, n_points)` -- IBS = (1/t_max) integral BS(t) dt

---

## Step 4: Update `src/competing_risks/__init__.py`

Add Deep-PTCM exports.

---

## Mathematical Detail: CIF Computation

The CIF cannot be written in closed form for the multi-cause PTCM. Compute numerically:

```
CIF_k(t) = integral_0^t theta_k * f_k(s) * S(s) ds
```

where `S(s) = exp(-sum_j theta_j * F_j(s))`. Use trapezoidal rule on a fine grid (~200 points). The subdensity `theta_k * f_k(s) * S(s)` is smooth within each piecewise interval, so this is accurate.

---

## Verification

1. **Sanity checks in notebook**: F(0)=0, F(T_max) near 1, theta>0, 0<pi<1, CIF_1+CIF_2 <= 1
2. **Overfit test**: Train on 100 loans, verify near-zero loss
3. **Metric comparison**: C-index and Brier score vs DeepHit/Cox on the same test fold
4. **Cure fraction plausibility**: Compare predicted cure fractions to observed censoring rates by vintage

---

## Dependencies

No new packages needed. Uses PyTorch (already installed), numpy, pandas, scikit-learn.

---

## Implementation Order

1. `PiecewiseExponentialBaseline` + `PTCMLoss` (testable independently)
2. `DeepPTCMNetwork` (base, no orthogonalization)
3. `CompetingRisksDeepPTCM` wrapper with fit/predict
4. Notebook: data loading through training and evaluation
5. Add orthogonalization option
6. Add cure fraction analysis + AUC_cure
7. Complete notebook with comparisons and visualizations
