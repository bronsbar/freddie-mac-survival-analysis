# Implementation Plan: Neural Network Discrete-Time Survival Model with APC Decomposition

## Paper Reference
Wang, H., Bellotti, A., Qu, R. & Bai, R. (2024). "Discrete-Time Survival Models with Neural Networks for Age-Period-Cohort Analysis of Credit Risk."
*Risks*, 12(2), 31.

---

## 1. What This Model Does

The paper proposes a two-stage framework:

1. **Stage 1 (NN-DTSM)**: Train a suite of vintage-level neural networks — one separate MLP per origination quarter — that predict the monthly probability of an event. The non-linearity of the NN naturally captures interaction terms and non-linear covariate effects, while the vintage-level architecture allows each cohort to exhibit different risk behaviour.

2. **Stage 2 (APC Decomposition)**: Use the NN-DTSM predictions to construct Lexis graphs and decompose the predicted risk into three time components via ridge regression:
   - **Age effect** (loan_age): how risk evolves as a loan matures
   - **Period/Calendar-time effect** (year_month): environmental impact (macro economy, regulation) affecting all loans simultaneously
   - **Cohort/Vintage effect** (vintage_quarter): origination-time conditions (underwriting standards, risk appetite) affecting all loans from the same cohort

The calendar-time effect is then fitted against macroeconomic variables (HPI, unemployment, treasury rates) to solve the APC identification problem and provide economic interpretability.

### Extension to Competing Risks

The paper models default only (binary: default vs. non-default). We extend to **competing risks** (prepayment and default) by replacing the single sigmoid output with a 3-class softmax:

$$P_{it} = (P_{it,\text{current}}, P_{it,\text{prepay}}, P_{it,\text{default}})$$

This is the same multinomial-logit output structure as the Sadhwani NN and the joint model, but with vintage-level subnetworks rather than a single shared network.

The APC decomposition is then performed **separately for each cause**, yielding:
- Age, Vintage, Calendar-time effects for **prepayment**
- Age, Vintage, Calendar-time effects for **default**

Comparing these decompositions reveals how the time-related risk drivers differ between the two competing events.

### What This Adds to the Model Comparison

| Model | Non-linear | Vintage-specific | APC interpretability | Macro projection | Competing risks |
|-------|-----------|-----------------|---------------------|-----------------|----------------|
| CSC | No | No | No | No | Yes |
| FGR | No | No | No | No | Yes |
| RSF | Yes | No | No | No | Yes |
| DeepHit | Yes | No | No | No | Yes |
| Sadhwani NN | Yes | No | No | AR (into NN, mismatched) | Yes |
| Breeden & Crook | No | Via APC splines | Yes (linear) | No | Yes |
| Joint Model | No | No | No | AR (into longitudinal) | Yes |
| **NN-DTSM + APC** | **Yes** | **Yes (per-quarter NN)** | **Yes (non-linear)** | **AR (into linear regression, clean)** | **Yes** |

The NN-DTSM + APC is the only model that combines non-linear covariate effects, vintage-specific estimation, and interpretable APC decomposition. The key comparison is against:
- **Breeden & Crook**: both do APC, but Breeden uses linear models. Does the NN discover non-linear patterns in the Lexis graph that the linear model misses?
- **Sadhwani NN**: both are neural networks, but Sadhwani is a single aggregate model. Do vintage-level subnetworks improve over a single model?
- **DeepHit**: both are non-linear, but NN-DTSM adds interpretability via APC. Does the vintage structure help?

---

## 2. Data Summary

### Available Data

Two data sources are available:

1. **Full vintage data** (`data/processed/by_vintage/`): 50,000 loans per vintage year, 27 vintages (1999-2025), **1.325 million loans** total. This is the primary dataset for the NN-DTSM + APC model.

2. **Sampled panel** (`data/processed/loan_month_panel.parquet`): 109K loans (2010-2025 only), used for all other notebooks. This serves as a **development/testing subset** for code validation before running on the full data.

### Full Dataset Structure
- **1,324,950 loans** across 27 yearly vintages (1999–2025)
- At quarterly granularity: **~108 vintage quarters**, ~12,500 loans per quarter
- **Calendar span**: ~1999 to 2025 (~312 calendar months)
- **Max loan age**: ~300 months (25 years for 1999 vintages)

### Event Distribution (Full Data)

| Event | Count | % |
|-------|-------|---|
| Prepaid | ~1,000,000 | ~75% |
| Censored | ~300,000 | ~23% |
| Defaulted | **19,728** | **1.5%** |

Default rate varies strongly by vintage — this is exactly what the APC decomposition should capture:

| Vintage period | Default rate | Interpretation |
|----------------|-------------|----------------|
| 1999–2003 | 0.9–1.4% | Pre-bubble, moderate risk |
| 2004–2005 | 2.9–5.4% | Deteriorating underwriting |
| 2006–2007 | **7.7–8.5%** | Peak crisis vintages |
| 2008–2009 | 1.0–4.0% | Crisis originations but tighter standards |
| 2010–2015 | 0.2–0.9% | Post-crisis, strict underwriting |
| 2016–2025 | 0.0–0.2% | Minimal defaults (recent, still maturing) |

With ~12,500 loans per quarter and crisis-era quarters having hundreds of defaults, the pure vintage-level subnetwork approach (one NN per quarter, as in Wang et al.) is fully viable — no shared-trunk workaround needed.

### Development vs. Production Pipeline

| | Development | Production |
|---|---|---|
| Data source | `loan_month_panel.parquet` (109K loans) | Full vintage data (1.3M loans) |
| Purpose | Code validation, quick iteration | Final results, APC analysis |
| Vintage range | 2010–2025 (62 quarters) | 1999–2025 (~108 quarters) |
| Defaults | 44 | 19,728 |
| Run time | Minutes (laptop) | Hours (supercomputer) |

### Feature Groups

**Static features** (input to NN subnetworks):

| Variable | Description |
|----------|-------------|
| `fico_score` | Credit score at origination |
| `ltv_r` | Loan-to-value ratio at origination |
| `dti_r` | Debt-to-income ratio |
| `int_rate` | Original interest rate |
| `log_upb` | Log original unpaid balance |

**Behavioural features** (input to NN subnetworks):

| Variable | Description |
|----------|-------------|
| `bal_repaid_lag1` | Balance repaid (lagged 1 month) |
| `t_act_12m` | Months active in last 12 months |
| `t_del_30d_12m` | Months 30+ days delinquent in last 12 months |
| `t_del_60d_12m` | Months 60+ days delinquent in last 12 months |

**Macro features** (excluded from NN, used in APC calendar-time regression):

| Variable | Description | For APC regression |
|----------|-------------|-------------------|
| `hpi_st_d_t_o` | State HPI relative to origination | Yes |
| `hpi_st_log12m` | 12-month log HPI change | Yes |
| `st_unemp_r12m` | 12-month state unemployment rate change | Yes |
| `st_unemp_r3m` | 3-month state unemployment rate change | Candidate |
| `ppi_c_FRMA` | Prepayment incentive (rate diff) | Yes |
| `TB10Y_d_t_o` | 10-year treasury diff from origination | Candidate |
| `FRMA30Y_d_t_o` | 30-year FRM rate diff from origination | Candidate |
| `T10Y3MM` | 10Y-3M treasury spread | Candidate |

Macro features are **not** input to the NN subnetworks. Following Wang et al., they are used only in the post-hoc calendar-time regression to solve the APC identification problem and provide economic interpretability.

---

## 3. Model Specification

### 3a. Vintage-Level Neural Network (NN-DTSM)

For each vintage quarter $v \in \{1, \ldots, N_v\}$, build a separate MLP subnetwork $s_v$.

**Input** for loan $i$ at loan age $t$:

$$\mathbf{x}_{it} = [\mathbf{w}_i, \mathbf{b}_{it}]$$

where $\mathbf{w}_i$ = static features (5 variables) and $\mathbf{b}_{it}$ = behavioural features (4 variables). Total input dimension: 9.

**Architecture** of each subnetwork $s_v$:

```
Input (9) → Dropout(d) → Dense(n_h, ReLU) → Dense(n_h, ReLU) → ... → Dense(3, Softmax)
```

- Hidden layers: $n_l$ layers of $n_h$ neurons each
- Activation: ReLU for hidden layers
- Output: **Softmax** over 3 outcomes (current, prepay, default) — this is the competing risks extension
- Dropout: applied after input layer (regularisation)

**Output** at each month:

$$P_{it} = \text{softmax}(s_{v_i}(\mathbf{x}_{it})) = (P_{it,0}, P_{it,1}, P_{it,2})$$

where $v_i$ is the vintage quarter of loan $i$, and:
- $P_{it,0}$ = P(current at month $t$ | survived to $t$)
- $P_{it,1}$ = P(prepay at month $t$ | survived to $t$)
- $P_{it,2}$ = P(default at month $t$ | survived to $t$)

### 3b. Likelihood (Discrete-Time Competing Risks)

The likelihood for loan $i$ with last observed month $t_i^*$ and event type $k_i$ (0=censored, 1=prepay, 2=default):

$$L_i = \left(\prod_{t=1}^{t_i^* - 1} P_{it,0}\right) \times \begin{cases} P_{it_i^*,k_i} & \text{if } k_i \in \{1, 2\} \\ P_{it_i^*,0} & \text{if } k_i = 0 \text{ (censored)} \end{cases}$$

Negative log-likelihood (cross-entropy loss):

$$\mathcal{L} = -\sum_{i=1}^{m} \sum_{t=1}^{t_i^*} \left[ d_{it,1} \log P_{it,1} + d_{it,2} \log P_{it,2} + (1 - d_{it,1} - d_{it,2}) \log P_{it,0} \right]$$

where $d_{it,k} = 1$ if loan $i$ experiences event $k$ at month $t$, and 0 otherwise. This is standard multinomial cross-entropy applied to the panel data — same loss structure as the Sadhwani NN.

### 3c. CIF Prediction

The cumulative incidence function for cause $k$:

$$\text{CIF}_k(t) = \sum_{s=1}^{t} P_{is,k} \prod_{u=1}^{s-1} P_{iu,0}$$

which chains the monthly transition probabilities. This is identical to the CIF computation in the Sadhwani and joint model notebooks.

---

## 4. APC Decomposition

### 4a. Constructing the Lexis Graph

After training the NN-DTSM, generate predictions $\hat{P}_{it,k}$ for each loan-month. Aggregate to cell-level means:

$$D_{vt,k} = \frac{1}{|S|} \sum_{i \in S} \hat{P}_{it,k} \quad \text{where } S = \{i : t \leq t_i^*, \, v_i = v\}$$

This produces a matrix indexed by (loan_age $t$, vintage_quarter $v$) for each cause $k$, which is the Lexis graph.

The **calendar time** is deterministic: $c = v + t$ (vintage quarter + loan age gives the calendar date of the observation).

### 4b. Ridge Regression APC Model

For each cause $k$, decompose the cell-level predictions into three time effects:

$$D_{vt,k} = \sum_{s=1}^{N_T} \alpha_{s,k} \, \delta_s^{[T]}(t) + \sum_{u=1}^{N_V} \beta_{u,k} \, \delta_u^{[V]}(v) + \sum_{b=1}^{N_C} \gamma_{b,k} \, \delta_b^{[C]}(c) + \varepsilon_{vt,k}$$

where $\delta^{[T]}, \delta^{[V]}, \delta^{[C]}$ are indicator variables for each loan age, vintage, and calendar time value.

**Identification problem**: Since $c = v + t$, the three sets of indicators are linearly dependent. Wang et al. solve this with:

1. **Ridge regularisation**: Add penalty $\lambda(\sum \alpha_s^2 + \sum \beta_u^2 + \sum \gamma_b^2)$ to the loss. This shrinks coefficients and yields a unique solution. The ridge parameter $\lambda$ is tuned via cross-validation.

2. **Calendar-time macro regression**: Fit the estimated calendar-time coefficients $\gamma_{b,k}$ against macroeconomic variables to validate the decomposition and solve the identification slope $\sigma$:

$$\gamma_{c,k} = \beta_0' + \sum_{j=1}^{M} \beta_j' \, m_{j(c - l_j)} + \sigma c + \varepsilon_c$$

where $m_j$ are macro variables at their optimal lag $l_j$.

### 4c. Macro Variable Lag Selection

For each macroeconomic variable, fit univariate regressions of the calendar-time coefficients against the macro variable at different lags (0 to 12 quarters / 36 months). Select the lag that maximises $R^2$. This follows Wang et al. Section 3.7.

### 4e. AR Macro Projection for Out-of-Sample Prediction

The APC decomposition decomposes predicted hazard into three additive components. For in-sample periods, all three are directly estimated. For **out-of-sample prediction** (future calendar months beyond the training window), the age and vintage effects are known but the calendar-time effect $\gamma_c$ is not — it depends on macro conditions that haven't been observed yet.

The calendar-time macro regression (Section 4b) creates a natural hook for forward projection:

**Step 1: Fit AR(p) models on each macro variable's historical series.**

For each macro variable $m_j$ (HPI, unemployment, FRM30, etc.), fit an autoregressive model on the training-period time series:

$$m_j(c) = \phi_{j,0} + \sum_{l=1}^{p_j} \phi_{j,l} \, m_j(c - l) + \varepsilon_j(c)$$

The lag order $p_j$ is selected via AIC/BIC. This reuses the same AR projection infrastructure already implemented for the Sadhwani model's macro simulation.

**Step 2: Project macro variables forward with Monte Carlo simulation.**

For each future calendar month $c^* > c_{\max}$, draw $R$ paths from the AR model (propagating noise forward). This yields a distribution of future macro paths:

$$\tilde{m}_j^{(r)}(c^*), \quad r = 1, \ldots, R$$

**Step 3: Project the calendar-time effect via the macro regression.**

Using the fitted macro regression coefficients $\hat{\beta}_j'$ from Section 4b:

$$\hat{\gamma}_{k}^{(r)}(c^*) = \hat{\beta}_0' + \sum_{j=1}^{M} \hat{\beta}_j' \, \tilde{m}_j^{(r)}(c^* - l_j) + \hat{\sigma} \, c^*$$

Each Monte Carlo macro path produces a projected calendar-time effect. Averaging over $R$ paths gives the point forecast; the spread gives confidence bands.

**Step 4: Reconstruct the full out-of-sample hazard.**

$$\hat{h}_k(t, v, c^*) = \hat{\alpha}_{k,t} + \hat{\beta}_{k,v} + \hat{\gamma}_k(c^*)$$

where $\hat{\alpha}_{k,t}$ and $\hat{\beta}_{k,v}$ are the in-sample age and vintage effects (which depend only on loan age and origination quarter, not on the future), and $\hat{\gamma}_k(c^*)$ comes from the AR projection.

**Why this is cleaner than in other models:**

| Aspect | Sadhwani NN (AR) | NN-DTSM + APC (AR) |
|--------|-----------------|---------------------|
| Where macro enters | Directly into the NN input layer | Only in the **linear** calendar-time regression |
| Training mismatch | NN trained on observed macro, evaluated on projected macro — coefficient mismatch | NN is macro-free; macro only enters the linear post-hoc regression — no mismatch |
| Uncertainty propagation | Opaque (through non-linear NN) | Transparent (linear regression + AR simulation) |
| Interpretability | Black box | Full decomposition: which macro variables drive the projected risk change |

The AR projection step adds no new model parameters — it simply extends the existing calendar-time macro regression forward in time using standard time-series forecasting.

### 4d. Separate Decomposition per Cause

Run the full APC pipeline independently for prepayment and default:

| Component | Prepayment interpretation | Default interpretation |
|-----------|--------------------------|----------------------|
| **Age** $\alpha_s$ | How prepayment risk evolves with loan maturity (burnout, seasoning) | How default risk evolves (delinquency accumulation, seasoning) |
| **Vintage** $\beta_u$ | Origination-time prepayment propensity (rate environment, borrower quality) | Origination-time default risk (underwriting standards, vintage quality) |
| **Calendar** $\gamma_b$ | Environmental prepayment drivers (rate changes, refi market) | Environmental default drivers (recession, HPI decline, unemployment) |

Comparing these across causes is a novel contribution: e.g., the 2008 crisis should show up strongly in the default calendar-time effect but may have a different signature in the prepayment calendar-time effect (refi activity collapsed).

---

## 5. Hyperparameter Tuning

Following Wang et al., use **grid search with 5-fold cross-validation** within each vintage's training data. The grid:

| Hyperparameter | Symbol | Values |
|----------------|--------|--------|
| Dropout rate | $d$ | 0, 0.1, 0.2, 0.3, 0.5 |
| Number of hidden layers | $n_l$ | 2, 4 |
| Neurons per hidden layer | $n_h$ | 4, 8, 16 |
| Training epochs | $ti$ | 10, 20, 30, 50 |

**Loss function**: multinomial cross-entropy (Eq. 5 extended to 3 classes).

**Notes**:
- Wang et al. used a larger grid (Table 1) because their subnetworks had a single binary output. With 3-class softmax, we use a comparable grid.
- **Class weights** should be used in the loss function to handle the within-vintage class imbalance (see Section 6).
- Grid search is done on a few representative vintages (one pre-crisis, one post-crisis), then the best hyperparameters are applied to all vintages to save computation.

**Ridge regularisation** ($\lambda$) for the APC regression: tuned separately via leave-one-out cross-validation on the Lexis graph cells.

---

## 6. Handling the Class Imbalance

### Data Strategy: Full Vintage Data

The full dataset (`data/processed/by_vintage/`) contains **1.325 million loans** with **19,728 defaults**. At quarterly granularity (~12,500 loans per quarter), pre-crisis quarters have hundreds of defaults — sufficient for per-vintage NN training.

The smaller sampled panel (`loan_month_panel.parquet`, 109K loans, 44 defaults) is used for **development and code testing only**. Production runs use the full vintage data.

### Building the Panel

The by-vintage files contain loan-level survival data (one row per loan). To train the NN-DTSM, we need to expand these into a **loan-month panel** with time-varying features. The pipeline:

1. Load all vintage parquet files and concatenate
2. Expand each loan into monthly observations (1 row per month survived)
3. Merge macro/behavioural features from external sources (same pipeline as the existing `loan_month_panel.parquet`)
4. Extract `vintage_quarter` from `loan_sequence_number` (format: `F{YY}Q{Q}xxxxxx`)
5. Split into train/test by vintage quarter (out-of-time validation)

### Handling Class Imbalance: Per-Vintage Class-Weighted Cross-Entropy

Wang et al. address the class imbalance (0.1% default rate) by undersampling non-default accounts to 10%, raising the effective default rate to ~1%. However, undersampling is problematic in our competing risks setting: discarding non-default accounts removes both censored *and* prepaid loans, degrading the prepayment signal. Since we model three outcomes simultaneously (current, prepay, default), we need all events to remain in the training set.

Instead, we use **per-vintage class-weighted cross-entropy**. Hussin Adam Khatir & Bee (2022) — cited by Wang et al. themselves — compare data balancing techniques for credit scoring and show that cost-sensitive methods (class weighting) perform comparably to resampling methods while retaining the full information content of the training data. The theoretical equivalence between loss reweighting and resampling was established by Elkan (2001): in expectation, weighting the loss by inverse class frequency produces the same gradient updates as resampling to balance the classes.

The class weights are computed **per vintage**, since the class balance differs dramatically across origination cohorts:

$$w_{k,v} = \frac{N_v}{3 \times N_{k,v}}$$

where $N_v$ is the total loan-months in vintage $v$ and $N_{k,v}$ is the count of event $k$ in vintage $v$. This yields:
- **2007 subnetwork** (8.5% default rate): moderate default weight (~4x)
- **2020 subnetwork** (0.02% default rate): high default weight (~1500x)
- **Prepayment weights**: close to 1.0 for most vintages (prepayment is the dominant event)

This is more principled than a global weight because each vintage subnetwork is trained independently and faces its own class distribution.

### APC Granularity

With sufficient data at quarterly level for both causes:
- **Prepayment APC**: quarterly vintage granularity
- **Default APC**: quarterly vintage granularity for pre-crisis period (1999–2012, where defaults are plentiful); yearly for post-2015 vintages where defaults are sparse. Alternatively, use quarterly throughout and flag low-count cells.

---

## 7. Implementation Architecture

### Python Module: `src/competing_risks/nn_dtsm.py`

```python
class VintageNNDTSM:
    """
    Neural Network Discrete-Time Survival Model with vintage-level subnetworks.
    
    Extension of Wang et al. (2024) to competing risks.
    One separate MLP per vintage quarter, each with a 3-class softmax output
    (current, prepay, default).
    """
    
    def __init__(
        self,
        n_hidden_layers: int = 4,
        n_neurons: int = 8,
        dropout: float = 0.0,
        n_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        use_class_weights: bool = True,
        device: str = 'auto',
        random_seed: int = 42,
    ):
        ...

    def fit(self, panel_df: pd.DataFrame, static_cols, behavioral_cols,
            vintage_col='vintage_quarter') -> 'VintageNNDTSM':
        """
        Train one separate MLP per vintage quarter (Wang et al. approach).
        
        For each vintage:
          1. Extract vintage's loan-month panel
          2. Compute per-vintage class weights
          3. Optionally undersample non-events
          4. Train MLP with cross-entropy loss
          5. Store trained subnetwork
        """
        ...

    def predict_monthly(self, panel_df) -> pd.DataFrame:
        """
        Predict P(current), P(prepay), P(default) for each loan-month.
        Routes each observation to its vintage's subnetwork.
        """
        ...

    def predict_cif(self, panel_df, horizons=[24, 48, 72]) -> Dict:
        """
        Compute CIF by chaining monthly transition probabilities.
        """
        ...

    def build_lexis_data(self, panel_df) -> Dict[int, pd.DataFrame]:
        """
        Construct Lexis graph data: cell-level (age, vintage) means
        of predicted probabilities, separately for each cause.
        Returns {cause_code: DataFrame with columns [loan_age, vintage, cal_time, mean_prob]}.
        """
        ...


class APCDecomposition:
    """
    Ridge regression APC decomposition of Lexis graph data.
    """
    
    def __init__(self, ridge_alpha: float = 1.0):
        ...

    def fit(self, lexis_df: pd.DataFrame) -> 'APCDecomposition':
        """
        Fit ridge regression: D_vt = sum(alpha_t) + sum(beta_v) + sum(gamma_c) + eps
        with cross-validated ridge parameter.
        """
        ...

    def get_effects(self) -> Dict[str, pd.Series]:
        """
        Return {'age': Series, 'vintage': Series, 'calendar': Series}.
        """
        ...

    def fit_macro_regression(
        self, calendar_effects: pd.Series,
        macro_df: pd.DataFrame, max_lag_months: int = 36,
    ) -> Dict:
        """
        Fit calendar-time coefficients against macro variables.
        1. Univariate lag selection per macro variable (maximise R²)
        2. Multivariate regression with best lags + time trend
        Returns regression results, R², selected lags.
        """
        ...

    def project_calendar_effect(
        self, macro_df: pd.DataFrame,
        n_months_ahead: int = 72,
        n_mc_paths: int = 500,
    ) -> pd.DataFrame:
        """
        Project the calendar-time effect forward using AR-simulated macro paths.
        
        1. Fit AR(p) on each macro variable (lag order via AIC)
        2. Simulate n_mc_paths forward trajectories per macro variable
        3. Feed each path through the fitted macro regression to get
           projected gamma_c values
        4. Return DataFrame with columns [calendar_month, gamma_mean, 
           gamma_lo, gamma_hi] (point forecast + confidence bands)
        
        Reuses the same AR projection infrastructure as the Sadhwani model.
        """
        ...

    def predict_hazard_oos(
        self, age: int, vintage: str, calendar_month: str,
        n_mc_paths: int = 500,
    ) -> Dict:
        """
        Out-of-sample hazard prediction combining in-sample age/vintage
        effects with AR-projected calendar-time effect.
        
        h_k(t, v, c*) = alpha_k(t) + beta_k(v) + gamma_k(c*)
        
        Returns point estimate and uncertainty bands from MC simulation.
        """
        ...
```

### Neural Network Architecture (PyTorch)

```python
class _VintageSubnet(nn.Module):
    """Single vintage subnetwork: MLP with 3-class softmax output."""
    
    def __init__(self, input_dim, n_hidden, n_neurons, dropout):
        super().__init__()
        layers = [nn.Dropout(dropout)]
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(in_dim, n_neurons), nn.ReLU()])
            in_dim = n_neurons
        layers.append(nn.Linear(n_neurons, 3))  # 3 classes
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)
```

One `_VintageSubnet` is instantiated and trained per vintage quarter. This follows Wang et al. directly: each vintage learns its own covariate effects and risk patterns independently.

---

## 8. Evaluation

### Primary Metrics (matching other notebooks)

| Metric | Function | Time Points |
|--------|----------|-------------|
| Time-dependent C-index | `time_dependent_concordance_index()` | 24, 48, 72 months |
| Brier score | `brier_score_competing_risks()` | 24, 48, 72 months |

Both evaluated separately for prepayment ($k=1$) and default ($k=2$).

### Additional Metrics (from the paper)

| Metric | Purpose |
|--------|---------|
| McFadden pseudo-$R^2$ | Comparison with Wang et al. and the linear DTSM baseline |
| Per-vintage pseudo-$R^2$ | Assess whether the NN outperforms the linear DTSM per vintage |

### APC-Specific Evaluation

| Analysis | Purpose |
|----------|---------|
| Lexis graphs (heatmaps) for prepay and default | Visualise risk patterns across age × calendar time |
| Lexis graphs by risk segment (high LTV, low FICO) | Segment-specific risk patterns |
| APC effect plots (3 panels per cause) | Visualise decomposed age, vintage, calendar effects |
| Calendar-time vs macro $R^2$ | Validate that calendar-time effect reflects macro conditions |
| Comparison of APC effects across causes | How do prepay vs default time drivers differ? |

### Out-of-Sample Evaluation with AR Macro Projection

The APC decomposition enables a unique out-of-sample evaluation strategy. For loans whose event horizon extends beyond the training window, the calendar-time effect must be projected forward. This is evaluated at two levels:

**Level 1: Macro projection quality**
- Backtest the AR models on a held-out period (e.g., train AR on 1999–2017, project 2018–2020, compare with realised macro values)
- Report RMSE and coverage of AR confidence intervals for each macro variable

**Level 2: Risk prediction quality with projected calendar-time**
- Compute CIF using three variants of the calendar-time effect:
  1. **Oracle**: use the actual (future) macro values in the regression — this is the upper bound on what AR projection can achieve
  2. **AR-projected**: use AR-projected macro values with Monte Carlo simulation — this is the realistic out-of-sample scenario
  3. **Frozen**: hold the calendar-time effect at its last observed value — this is the naive baseline
- Compare C-index and Brier score across these three variants to quantify how much the AR macro projection adds over naive forecasting, and how much room remains compared to the oracle

---

## 9. Implementation Steps

### Step 1: Data Preparation

- Load all vintage parquet files from `data/processed/by_vintage/` and concatenate
- Expand loan-level data into loan-month panel (monthly observations per loan)
- Merge time-varying features (behavioural + macro) — reuse existing preprocessing pipeline
- Extract `vintage_quarter` from `loan_sequence_number` (format: `F{YY}Q{Q}xxxxxx`)
- Create `calendar_quarter` from `year_month` (for APC regression at quarterly frequency)
- Compute `log_upb` from `orig_upb`
- Standardise features within train set; apply same scaling to test
- Map vintage quarters to contiguous integer indices
- Split: use held-out vintage quarters for testing (e.g., train on 1999-2019, test on 2020+; or use within-vintage 75/25 splits matching Wang et al.)

### Step 2: Python Module

Create `src/competing_risks/nn_dtsm.py` with:
- `VintageNNDTSM` class (per-vintage subnetworks, training, prediction, CIF)
- `APCDecomposition` class (ridge regression, macro fitting)
- Utility functions for Lexis graph construction

### Step 3: Hyperparameter Tuning (optional separate script)

Grid search on a representative vintage to find optimal $(d, n_l, n_h, ti)$. Apply to all vintages.

### Step 4: Notebook

Create `notebooks/20_nn_dtsm_apc.ipynb`:

| Section | Content |
|---------|---------|
| 1. Setup | Imports, config, device selection |
| 2. Data | Load panel, extract vintage_quarter, train/test split |
| 3. EDA | Loans per vintage, event rates per vintage, Lexis graph of raw data |
| 4. NN-DTSM Training | Fit vintage subnetworks, training curves |
| 5. NN-DTSM vs Linear DTSM | Per-vintage pseudo-$R^2$ comparison |
| 6. CIF Prediction | Compute CIF on test set |
| 7. Evaluation | C-index and Brier score at 24/48/72 months |
| 8. Lexis Graphs | Heatmaps for full population and risk segments, both causes |
| 9. APC Decomposition (Prepay) | Ridge regression, age/vintage/calendar effect plots |
| 10. APC Decomposition (Default) | Same, at coarser granularity if needed |
| 11. Calendar-Time vs Macro | Lag selection, multivariate regression, $R^2$ |
| 12. AR Macro Projection | Fit AR models on macro variables, project forward, reconstruct out-of-sample calendar-time effect with confidence bands |
| 13. Out-of-Sample Evaluation | Compare CIF under oracle / AR-projected / frozen calendar-time; C-index and Brier score across variants |
| 14. Cross-Cause Comparison | Side-by-side APC effects: prepay vs default |
| 15. Model Comparison | Combined results table with all other notebooks |

### Step 5: SLURM Script

Create `scripts/slurm_nn_dtsm.sh` and `scripts/run_nn_dtsm.py` for supercomputer execution. The NN training benefits from GPU acceleration (unlike the MCMC models), so request a GPU partition.

---

## 10. Comparison with Existing APC Implementation

The project already has `src/competing_risks/apc_decomposition.py` (Breeden & Crook). Key differences:

| Aspect | Existing APC (Breeden) | New APC (Wang) |
|--------|----------------------|----------------|
| Stage 1 model | Linear logistic regression | Neural network (vintage-level) |
| APC extraction | Iterative backfitting with splines | Ridge regression on NN predictions |
| Identification | Smoothing penalty (RW2 prior) | Ridge + macro variable regression |
| Granularity | Portfolio-level aggregation | Vintage-level + segment-level |
| Non-linear effects | No | Yes (NN captures interactions) |
| Segment-specific Lexis | Not possible (linear model) | Yes (feed different segments into NN) |

The existing `apc_decomposition.py` can be **reused** for parts of the ridge regression and macro fitting logic. The new module extends it with the NN-DTSM stage.

---

## 11. Computational Considerations

### Training Time

~108 vintage subnetworks, each with ~12,500 loans × ~50-100 months of panel data ≈ 0.6-1.2M rows per vintage. With small networks (4 layers × 8 neurons), training one subnetwork takes ~10-30 seconds on GPU. The full pipeline:

- **Training only** (108 subnetworks): ~30-60 minutes on GPU
- **With grid search** (5-fold CV × ~120 hyperparameter combos × a few representative vintages): ~2-4 hours on GPU
- **Development** (using sampled panel, 62 vintages × ~1,700 loans): ~5 minutes on laptop

### Device

- **GPU (CUDA/MPS)**: Preferred for NN training. float32 is fine (unlike MCMC).
- **CPU**: Falls back gracefully for smaller datasets.

### SLURM Configuration

```
#SBATCH --gpus-per-node="1"
#SBATCH --partition="gpu_p100"
#SBATCH --mem="44G"
#SBATCH --time="04:00:00"     # Much shorter than the joint model
```

---

## 12. Expected Results

Based on Wang et al.'s findings and our data characteristics:

### Predictive Performance
- **NN-DTSM should outperform linear DTSM** on pseudo-$R^2$ for most vintages, especially pre-crisis vintages (2004-2008) where non-linear interaction effects (e.g., high LTV × high DTI) are strongest.
- **Prepayment C-index**: Should be competitive with Sadhwani NN since both are neural networks with similar features. The vintage structure may help if prepayment behaviour differs meaningfully across origination cohorts.
- **Default C-index**: With 19,728 defaults and high default rates in the 2005-2008 vintages, the model should produce meaningful discrimination. However, post-2015 vintages have almost no defaults, so aggregate C-index will be driven by the earlier period.

### APC Decomposition
- **Prepayment age effect**: Risk should increase with loan age, peak around 24-36 months (burnout period), then stabilise.
- **Prepayment vintage effect**: Vintages from low-rate environments (2020-2021) should show different prepayment patterns than high-rate vintages (2018-2019, 2023+).
- **Prepayment calendar-time effect**: Should correlate strongly with interest rate changes (ppi_c_FRMA, FRMA30Y). When rates drop, prepayment spikes across all vintages.
- **Default age effect**: Risk should rise with loan age, peaking around 36-60 months (the "seasoning curve" well-documented in mortgage risk literature).
- **Default vintage effect**: Should peak sharply for 2006-2007 vintages (worst underwriting standards) and drop dramatically after 2009 (post-crisis tightening). This is the most interpretable component.
- **Default calendar-time effect**: Should spike in 2008-2010 (financial crisis), directly reflecting macro deterioration. This is the component that gets fitted against macro variables.

### Macro Variable Fit
- **Default**: HPI and unemployment rate should explain the calendar-time effect strongly ($R^2 > 0.8$), matching Wang et al.'s finding ($R^2$ = 0.938). The 2008 crisis provides a dramatic natural experiment.
- **Prepayment**: Rate-related macro variables (ppi_c_FRMA, FRMA30Y, treasury rates) should dominate the calendar-time effect, since refinancing waves are driven by interest rate movements.

### AR Macro Projection
- **Short horizons (12–24 months)**: AR-projected calendar-time effect should be close to the oracle, since macro variables are relatively predictable at short horizons. The gap between AR-projected and frozen CIF should be meaningful — frozen predictions miss trend continuation.
- **Long horizons (48–72 months)**: AR projections degrade as forecast uncertainty widens. Confidence bands on the calendar-time effect should widen substantially. The frozen baseline may be competitive at very long horizons where AR forecasts revert to the unconditional mean.
- **Default vs prepayment**: Default calendar-time projection should be more informative, since HPI and unemployment have stronger AR structure (persistent trends) than interest rate changes (more volatile). Prepayment projection may be harder because rate movements are less forecastable.
- **Compared to Sadhwani AR**: The NN-DTSM + APC approach should produce better-calibrated out-of-sample predictions because the macro effect flows through a linear regression (no train/predict mismatch), and uncertainty is propagated transparently through the Monte Carlo simulation.

---

## 13. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Panel expansion is memory-intensive | Medium | 1.3M loans × 50-100 months = 65-130M rows. Process per-vintage, don't load all at once. Use the supercomputer's 44G RAM |
| Post-2015 vintages have very few defaults | Medium | Class-weighted loss per vintage; flag low-default vintages in APC; consider pooling recent vintages |
| Overfitting per-vintage subnetworks | Medium | Dropout, early stopping, grid search with CV, small network architecture |
| APC identification not resolved | Low | Ridge regularisation + macro regression; validate with existing Breeden APC; rich macro data available |
| 108 subnetworks slow to train/tune | Low | GPU acceleration; tune on representative vintages, apply to all |
| Macro regression overfits small sample | Medium | ~108 quarterly calendar-time coefficients (more than Wang's 40); use few macro regressors, report adjusted $R^2$ |
| Feature mismatch between vintage files and sampled panel | Low | Verify column names match; development pipeline standardises both formats |

---

## 14. Dependencies

All already installed:

| Package | Purpose |
|---------|---------|
| `torch` | Neural network training |
| `numpy`, `pandas` | Data manipulation |
| `sklearn` | Ridge regression, StandardScaler, cross-validation |
| `matplotlib`, `seaborn` | Lexis graphs, APC plots |
| `scipy` | Spline utilities (if needed) |

---

## 15. Files to Create

| File | Purpose |
|------|---------|
| `src/competing_risks/nn_dtsm.py` | VintageNNDTSM class, APCDecomposition class |
| `notebooks/20_nn_dtsm_apc.ipynb` | Analysis notebook |
| `scripts/run_nn_dtsm.py` | Supercomputer training script |
| `scripts/slurm_nn_dtsm.sh` | SLURM submission for VSC |

---

## 16. References

1. Wang, H., Bellotti, A., Qu, R. & Bai, R. (2024). Discrete-Time Survival Models with Neural Networks for Age-Period-Cohort Analysis of Credit Risk. *Risks*, 12(2), 31.
2. Hussin Adam Khatir, A. & Bee, M. (2022). Machine Learning Models and Data-Balancing Techniques for Credit Scoring: What Is the Best Combination? *Risks*, 10, 169.
3. Breeden, J. (2016). Incorporating lifecycle and environment in loan-level forecasts and stress tests. *EJOR*, 255, 649-658.
4. Breeden, J. & Crook, J. (2022). Multihorizon discrete time survival models. *JORS*, 73, 56-69.
5. Bellotti, A. & Crook, J. (2013). Forecasting and stress testing credit card default using dynamic models. *IJF*, 29, 563-574.
6. Ohno-Machado, L. (1996). Medical Applications of Artificial Neural Networks: Connectionist Models of Survival Analysis. PhD dissertation, Stanford.
7. Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proceedings of the 17th IJCAI*, 973-978.
