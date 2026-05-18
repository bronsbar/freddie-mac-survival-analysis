# Implementation Plan: Sadhwani et al. (2021) Deep Learning for Mortgage Risk

## Paper Reference
Giesecke, Sirignano & Sadhwani (2021). "Deep Learning for Mortgage Risk."
*Journal of Financial Econometrics*, 19(2), 313–368.

---

## 1. Model Overview

The paper models **monthly mortgage state transitions** using a deep neural network with softmax output. Given the current state and covariates, the network predicts the probability distribution over all possible next-month states.

### States (adapted to our data)
The paper uses 7 states: Current, 30dd, 60dd, 90+dd, Foreclosure, REO, Paid Off.

**Our adaptation**: We use 3 terminal outcomes:
- **Current** (state 0) — loan is active/performing
- **Prepaid** (state 1) — terminal absorbing state
- **Defaulted** (state 2) — terminal absorbing state

This gives us a **3-state model** with transitions: Current → {Current, Prepaid, Defaulted} each month. Once a loan enters Prepaid or Defaulted, it stays there (absorbing states). This is consistent with our panel data where `event_code ∈ {0, 1, 2}`.

### Model Architecture
- Input: current state (one-hot or indicator) + covariates X_{t-1}
- Neural network with hidden layers → softmax output
- Output: P[state_t | state_{t-1}, X_{t-1}] — probability vector over 3 states
- Since we only have Current → {Current, Prepay, Default} transitions (absorbing states), the network only needs to predict the 3-class transition probabilities for loans currently in state "Current"

### Training
- **Maximum likelihood estimation** via cross-entropy loss
- **L2 regularization** + **dropout** for overfitting control
- **Ensemble** of multiple independently trained networks (bootstrapped data, different random seeds)
- **Mini-batch SGD** with learning rate decay

---

## 2. Adaptation Decisions

### 2a. State Space Simplification
The paper's 7-state model includes intermediate delinquency states. Our data already encodes delinquency history as **features** (`t_del_30d_12m`, `t_del_60d_12m`) rather than states. We therefore model only the 3 terminal states (Current, Prepay, Default), which is the relevant competing-risks formulation matching all other notebooks.

### 2b. Covariates
Use the **same 13 features** as the other notebooks:

| Category | Features |
|----------|----------|
| Static | `int_rate`, `log_upb`, `fico_score`, `dti_r`, `ltv_r` |
| Behavioral | `bal_repaid_lag1`, `t_act_12m`, `t_del_30d_12m`, `t_del_60d_12m` |
| Macro | `hpi_st_d_t_o`, `ppi_c_FRMA`, `TB10Y_d_t_o`, `FRMA30Y_d_t_o` |

Plus **current state indicator** (always = Current in our case, so this becomes implicit).

### 2c. Architecture
Following the paper's cross-validation results:
- **5 hidden layers**: 200 units (first layer), 140 units (layers 2–5)
- **Activation**: ReLU (`max(0, x)`)
- **Output**: Softmax over 3 classes (Current, Prepay, Default)
- **Dropout**: Applied to all hidden layers (rate to be cross-validated, paper uses it throughout)
- **L2 penalty**: Weight decay parameter

### 2d. Ensemble
- Train **8 independently initialized** 5-layer networks
- Each on bootstrapped training data
- Average predicted probabilities across ensemble members

---

## 3. Implementation Steps

### Step 1: Data Preparation (notebook cell)
```
- Load loan_month_panel.parquet
- Use existing train/val/test split (fold-based)
- For each loan-month observation where loan is still active:
    - Input: 13 covariates (standardized)
    - Target: next-month outcome (0=still current, 1=prepay, 2=default)
- Last observation per loan gets the terminal event code
- All earlier observations get target=0 (stayed current)
- Standardize features using training set mean/std
```

### Step 2: Model Implementation (`src/competing_risks/sadhwani_net.py`)

```python
class SadhwaniNet(nn.Module):
    """
    Deep neural network for mortgage state transitions.
    Sadhwani et al. (2021), Journal of Financial Econometrics.

    Predicts P[state_t | X_{t-1}] via softmax over 3 outcomes:
    0=current, 1=prepay, 2=default.
    """
    def __init__(self, n_features=13, hidden_sizes=[200,140,140,140,140],
                 n_states=3, dropout=0.5):
        # 5 hidden layers with ReLU + dropout
        # Final softmax output layer

    def forward(self, x):
        # Feed-forward with ReLU + dropout
        # Return log-softmax for NLLLoss
```

**Training function:**
```python
def train_sadhwani(model, train_loader, val_loader,
                   lr=0.1, weight_decay=1e-4, n_epochs=100,
                   lr_decay_halflife=800):
    """
    Mini-batch SGD with learning rate schedule:
    lr_t = lr_0 / (1 + t/halflife)

    Cross-entropy loss (= negative log-likelihood of transitions).
    Early stopping on validation loss.
    """
```

**Ensemble wrapper:**
```python
class SadhwaniEnsemble:
    """Ensemble of 8 independently trained SadhwaniNet models."""

    def fit(self, X_train, y_train, X_val, y_val, n_models=8):
        # Train each with different seed + bootstrapped data

    def predict_proba(self, X):
        # Average predictions across all ensemble members
        # Returns (n_samples, 3) probability matrix
```

### Step 3: CIF Computation for Evaluation

To compute cumulative incidence functions (needed for Brier score and C-index), we chain monthly transition probabilities:

```
CIF_k(t) = CIF_k(t-1) + S(t-1) * p_k(t)
S(t)     = S(t-1) * p_0(t)
```

**Key issue**: chaining requires features at every future month, but future features are not available at prediction time. Three approaches are implemented:

| Method | Function | Macro features | Future info? |
|--------|----------|---------------|-------------|
| **Observed** | `compute_cif()` | From panel (future data) | Yes — diagnostic only |
| **Frozen** | `compute_cif_frozen()` | Held at time-zero value | No |
| **AR-simulated** | `compute_cif_ar()` | AR(p) forward simulation | No |

**Frozen features** (`compute_cif_frozen`): Uses each loan's first observation for all months. Behavioural features (`bal_repaid_lag1`, `t_act_12m`) are updated mechanically; macro features are frozen. Analogous to how the Blumenstock DeepHit computes CIF from a single feature vector.

**AR-simulated** (`compute_cif_ar`): Fits univariate AR(p) models (BIC-selected lag ≤ 4) to each time-varying feature using the training panel. Generates `n_simulations` (default 50) Monte Carlo forward paths. CIF is averaged across simulations. This follows Sadhwani et al. (2021) Section 5.4, where they use AR(4) for the national mortgage rate.

**Observed** (`compute_cif`): Uses actual panel features at each month. Only valid as a diagnostic (upper bound on model performance) since it uses future information.

### Step 4: Evaluation Metrics

Using existing `src/competing_risks/evaluation.py`:

1. **Time-dependent C-index** at τ = 24, 48, 72 months
   - Risk score = CIF_k(τ) for cause k
   - Use `time_dependent_concordance_index()` from evaluation.py

2. **Brier score** at τ = 24, 48, 72 months
   - Use `brier_score_competing_risks()` from evaluation.py
   - Predicted CIF vs observed indicator

3. **Cross-entropy loss** (negative average log-likelihood)
   - Primary goodness-of-fit metric from the paper
   - Computed on monthly transitions

### Step 5: Sensitivity Analysis (Variable Importance)

Following Equation (7) from the paper:
```
Sensitivity(j) = E[|∂h_θ(v, X) / ∂x_j|]
```
Approximate via finite differences for each feature.

---

## 4. Notebook Structure (`18_sadhwani_deep_learning.ipynb`)

| Cell | Section | Content |
|------|---------|---------|
| 1 | Title | Markdown: "Sadhwani et al. (2021): Deep Learning for Mortgage Risk" |
| 2 | Imports | torch, numpy, pandas, evaluation functions |
| 3 | Config | Hyperparameters, device, random seed |
| 4 | Data Loading | Load panel, define features, train/val/test split |
| 5 | Data Prep | Create monthly transition targets, standardize features |
| 6 | Model Definition | SadhwaniNet class |
| 7 | Training Loop | Single model training with LR decay + early stopping |
| 8 | Train Single Model | Train one 5-layer network, plot loss curves |
| 9 | Ensemble Training | Train 8 models with bootstrap + different seeds |
| 10 | Monthly Predictions | Predict transition probabilities on test set |
| 11 | CIF Computation | Chain monthly probabilities → cumulative incidence |
| 12 | C-index Evaluation | Time-dependent C-index at 24/48/72 months |
| 13 | Brier Score | Brier score at 24/48/72 months |
| 14 | Cross-entropy Loss | In-sample and out-of-sample loss comparison |
| 15 | Depth Comparison | Compare 0, 1, 3, 5 hidden layers (Table 11 replication) |
| 16 | Sensitivity Analysis | Variable importance via finite differences |
| 17 | Nonlinear Relationships | Partial dependence plots for top features |
| 18 | Results Summary | Comparison table with other models |

---

## 5. Key Design Choices

| Choice | Decision | Rationale |
|--------|----------|-----------|
| State space | 3 states (C, P, D) | Matches our panel structure; delinquency encoded as features |
| Architecture | 5 layers (200-140-140-140-140) | Paper's cross-validated optimum |
| Ensemble size | 8 models | Paper shows diminishing returns beyond 8 |
| Activation | ReLU | Paper found better than sigmoid |
| Loss | Cross-entropy (NLL) | Exact match to paper's MLE |
| LR schedule | lr_0/(1+t/800) | Paper's Eq. (9) |
| Batch size | 4096 | Adapted from paper's 4000; power of 2 for GPU |
| Dropout | 0.5 (tunable) | Paper's regularization approach |
| L2 penalty | 1e-4 (tunable) | Standard weight decay |
| Features | Same 13 as other notebooks | Comparability across models |
| Evaluation | C-index + Brier at 24/48/72 | Matches all other notebooks |

---

## 6. Files Created

| File | Purpose |
|------|---------|
| `src/competing_risks/sadhwani_net.py` | Model class, training, ensemble, CIF computation, sensitivity analysis |
| `notebooks/18_sadhwani_deep_learning.ipynb` | Interactive notebook with full pipeline |
| `scripts/run_sadhwani_train.py` | CLI training script (local + cluster) with device auto-detection |
| `scripts/slurm_sadhwani.sh` | SLURM submission script for GPU supercomputer clusters |

---

## 7. Running

### Local (MPS / CPU)
```bash
python scripts/run_sadhwani_train.py
# Auto-detects MPS on Apple Silicon, CUDA on GPU machines, else CPU
```

### GPU Cluster (SLURM)
```bash
sbatch scripts/slurm_sadhwani.sh
# Or with overrides:
sbatch --export=EPOCHS=200,ENSEMBLE=8 scripts/slurm_sadhwani.sh
```

### With depth comparison (Table 11)
```bash
python scripts/run_sadhwani_train.py --depth-comparison --n-epochs 200
```

---

## 8. Dependencies

- PyTorch (already available — used by DeepHit and Bayesian models)
- NumPy, Pandas, Matplotlib (standard)
- `src/competing_risks/evaluation.py` (existing C-index, Brier)

No new dependencies needed.
