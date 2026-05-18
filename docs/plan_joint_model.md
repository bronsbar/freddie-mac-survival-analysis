# Implementation Plan: Joint Model for Longitudinal and Discrete Survival Data

## Paper Reference
Medina-Olivares, V., Calabrese, R., Crook, J. & Lindgren, F. (2022). "Joint models for longitudinal and discrete survival data in credit scoring."
*European Journal of Operational Research*, 307(3), 1457-1473.

---

## 1. Why a Joint Model?

All models implemented so far (CSC, FGR, RSF, DeepHit, Sadhwani NN) treat time-varying covariates as exogenous inputs: they plug the observed TVC value directly into the hazard at each time step. This creates two problems when the TVC is **endogenous** (its observation is tied to loan survival):

1. **Tautological conditioning** -- P(T > t | Y(t) observed) = 1, because measuring the TVC at time t already implies the loan survived to t. Standard models condition on information that presupposes the outcome.
2. **Unavailable future values** -- At prediction time, future TVC values don't exist. Workarounds (frozen features, AR simulation) are bolted on after the fact rather than integrated into the estimation.

The joint model solves both problems by modelling the TVC trajectory and the survival process **simultaneously**, linked through shared latent variables (random effects). At prediction time, the model projects the TVC forward using the longitudinal submodel -- no future observations needed, and the survival coefficients are calibrated for projected (not observed) TVC values.

### What this adds to the model comparison

| Model | TVC handling | Competing risks | Estimation |
|-------|-------------|----------------|------------|
| CSC | Exogenous | Cause-specific censoring | Semi-parametric MLE |
| FGR | Exogenous | Subdistribution hazard | MLE |
| RSF | Exogenous | Gray's log-rank splitting | Ensemble trees |
| DeepHit | Static snapshot | Native (multi-output PMF) | NN + ranking loss |
| Sadhwani NN | AR projection (two-stage) | 3-state softmax | NN cross-entropy |
| **Joint model** | **Endogenous (modelled)** | **Extended to competing risks** | **Bayesian MCMC** |

The joint model is the only approach that acknowledges endogeneity and models it structurally. Comparing its discrimination and calibration against the others will reveal whether properly handling endogenous TVCs matters in practice for mortgage risk.

---

## 2. Model Specification

### 2a. Extension to Competing Risks

The paper models default only. We extend to **competing risks** (prepayment and default) by replacing the single discrete-time survival submodel with a cause-specific formulation:

$$p_{ik}(t) = P(T_i = t, K_i = k \mid T_i \geq t, b_i), \quad k \in \{1 \text{ (prepay)}, 2 \text{ (default)}\}$$

The discrete-time hazard for each cause becomes:

$$p_{i,\text{prepay}}(t) = \frac{\exp(\eta_{i1}(t))}{1 + \exp(\eta_{i1}(t)) + \exp(\eta_{i2}(t))}$$

$$p_{i,\text{default}}(t) = \frac{\exp(\eta_{i2}(t))}{1 + \exp(\eta_{i1}(t)) + \exp(\eta_{i2}(t))}$$

$$p_{i,\text{current}}(t) = \frac{1}{1 + \exp(\eta_{i1}(t)) + \exp(\eta_{i2}(t))}$$

This is a **multinomial logit** for 3 outcomes at each month -- structurally identical to the Sadhwani NN's softmax output, but with a linear predictor instead of a neural network, and with the TVC modelled jointly rather than taken as given.

Each cause-specific linear predictor:

$$\eta_{ik}(t) = a_{k}^{(t)} + \gamma_k' z_i + \lambda_k \, m_i(t-1)$$

where:
- $a_k^{(t)}$ = cause-specific baseline hazard at time $t$ (cubic B-splines, 3 knots + 1 intercept)
- $z_i$ = time-invariant covariates (FICO, LTV, DTI, interest rate, log UPB)
- $\gamma_k$ = cause-specific covariate effects
- $\lambda_k$ = cause-specific association parameters linking the TVC to each hazard
- $m_i(t-1)$ = model-implied longitudinal predictor (see below)

Note: $\lambda_{\text{prepay}}$ and $\lambda_{\text{default}}$ can differ in sign and magnitude. A rising rate difference (market rates exceeding the loan's fixed rate) makes prepayment less attractive (negative $\lambda_{\text{prepay}}$) but may signal economic stress that raises default risk (positive $\lambda_{\text{default}}$). This is a key advantage over single-event models.

### 2b. Longitudinal Submodel (M5 specification)

Following the paper's best model (M5), the interest rate difference $Y_i(t)$ is modelled as:

$$Y_i(t) = \alpha_0 + U_{0i} + U_{1i} \cdot t + \phi \, Y_i(t-1) + \epsilon_i(t)$$

where:
- $\alpha_0$ = population intercept (fixed effect)
- $U_{0i}$ = random intercept for loan $i$ -- loan-specific level of rate difference
- $U_{1i}$ = random slope for loan $i$ -- loan-specific trend
- $\phi$ = AR(1) coefficient -- captures momentum/serial correlation
- $\epsilon_i(t) \sim \mathcal{N}(0, \sigma^2)$ = measurement noise

Random effects:

$$(U_{0i}, U_{1i})^T \sim \mathcal{N}_2\left(\mathbf{0}, \boldsymbol{\Sigma}\right), \quad \boldsymbol{\Sigma} = \begin{pmatrix} \sigma^2_{U_0} & \rho \sigma_{U_0}\sigma_{U_1} \\ \rho \sigma_{U_0}\sigma_{U_1} & \sigma^2_{U_1}\end{pmatrix}$$

The **model-implied true trajectory** (noise-free) used in the survival submodel:

$$m_i(t) = \alpha_0 + U_{0i} + U_{1i} \cdot t + \phi \, Y_i(t-1)$$

This is $Y_i(t)$ minus the noise $\epsilon_i(t)$, filtering out measurement error.

### 2c. Conditional Independence Assumption

Given the random effects $b_i = (U_{0i}, U_{1i})$:

$$P(T_i, K_i, \bar{Y}_i \mid b_i) = P(T_i, K_i \mid b_i) \times P(\bar{Y}_i \mid b_i)$$

The longitudinal and survival processes are independent conditional on the shared latent structure. All dependence flows through $b_i$ and the AR term.

### 2d. Joint Likelihood

For loan $i$ with event at time $t_i$ of type $k_i$ (0=censored, 1=prepay, 2=default):

$$L_i = \int \underbrace{P(\bar{Y}_i \mid b_i)}_{\text{longitudinal}} \times \underbrace{P(T_i, K_i \mid b_i)}_{\text{survival}} \times \underbrace{P(b_i)}_{\text{random effects}} \, db_i$$

**Longitudinal contribution**:

$$P(\bar{Y}_i \mid b_i) = \prod_{s=1}^{n_i} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(Y_i(s) - m_i(s))^2}{2\sigma^2}\right)$$

**Survival contribution** (competing risks):

$$P(T_i, K_i \mid b_i) = p_{i,k_i}(t_i) \times \prod_{s=1}^{t_i - 1} p_{i,0}(s)$$

where $p_{i,0}(s) = 1 - p_{i,1}(s) - p_{i,2}(s)$ is the probability of remaining current at month $s$, and $p_{i,k_i}(t_i)$ is the transition probability for the observed event at the event time (with $p_{i,0}(t_i)$ for censored loans at the last observed month).

---

## 3. Choice of TVC

### Primary: Prepayment Incentive (ppi_c_FRMA)

The paper uses the interest rate difference (implicit rate minus fixed rate at origination). Our closest analogue is `ppi_c_FRMA` = `int_rate - current_30yr_FRM_average`.

This is the strongest candidate because:
- It is **endogenous**: it reflects both loan-specific characteristics and market conditions, and it can only be observed while the loan is active
- It directly captures **prepayment incentive**: when market rates drop below the loan's rate, refinancing becomes attractive
- It carries information about **default risk**: a borrower locked into a high rate while market rates are low may be under financial stress (or may have been denied refinancing due to credit deterioration)
- It has the strongest AR dynamics among our features, making the longitudinal submodel meaningful

### Extension: Multiple TVCs

If the single-TVC model works well, extend to a multivariate longitudinal submodel:

| TVC | Rationale |
|-----|-----------|
| `ppi_c_FRMA` | Primary prepayment incentive |
| `hpi_st_d_t_o` | Equity position (drives both prepay and default) |

Each TVC gets its own longitudinal equation with shared random effects. This increases model complexity and MCMC computation time substantially, so start with a single TVC.

---

## 4. Covariates

### Time-Invariant Covariates in Survival Submodel ($z_i$)

Use the same 5 static features as all other notebooks:

| Variable | Description |
|----------|-------------|
| `fico_score` | Credit score at origination |
| `ltv_r` | Loan-to-value ratio at origination |
| `dti_r` | Debt-to-income ratio |
| `int_rate` | Original interest rate |
| `log_upb` | Log original unpaid balance |

### Time-Varying Features Not Modelled Jointly

The 4 behavioural features (`bal_repaid_lag1`, `t_act_12m`, `t_del_30d_12m`, `t_del_60d_12m`) could be included as additional time-varying covariates in the survival submodel. Two options:

1. **Exclude them** -- Keep the model pure and comparable to the paper. The joint model's advantage is in handling the endogenous TVC; adding exogenous TVCs directly to the hazard is standard practice.
2. **Include as exogenous TVCs** -- Add them directly to $\eta_{ik}(t)$ as standard time-varying covariates (not modelled jointly). This is valid because these behavioural features are observed deterministically as long as the loan is active.

**Recommendation**: Start with option 1 for a clean comparison, then test option 2 to see if behavioural features add discriminatory power beyond what the joint TVC captures.

---

## 5. Estimation: Bayesian MCMC via Pyro

### Why Bayesian?

The joint model requires integrating over the random effects $b_i$ for every loan. With 10,000+ loans and a 2D random effect per loan, this integral is intractable analytically. Bayesian MCMC (specifically Hamiltonian Monte Carlo / NUTS) handles this naturally by sampling from the joint posterior of all parameters and random effects simultaneously.

### Why Pyro?

The project already uses **Pyro + NUTS + ArviZ** for the Bayesian competing risks model in `bayesian_phm.py` (notebook 09). Reusing the same stack:

1. **No new dependencies** -- Pyro, PyTorch, and ArviZ are already installed and tested
2. **Consistent patterns** -- The existing `BayesianCompetingRisksPHM` class provides a proven template for model specification, MCMC execution, posterior extraction, ArviZ diagnostics, and CIF prediction
3. **Device flexibility** -- Pyro already handles CPU/MPS/CUDA transparently in our codebase
4. **PyTorch autodiff** -- Handles the complex joint likelihood gradient automatically

### Pyro Model Specification

```python
def joint_model(Y, Y_lag, loan_id, time_idx, Z, B_spline,
                event_time, event_type, N_loans, n_basis, P):
    """Pyro model for joint longitudinal + competing risks survival."""
    dtype = Y.dtype
    device = Y.device

    # ── Longitudinal submodel parameters ──
    alpha_0 = pyro.sample('alpha_0', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(5., dtype=dtype, device=device)))
    phi = pyro.sample('phi', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(0.5, dtype=dtype, device=device)))
    sigma = pyro.sample('sigma', dist.Exponential(
        torch.tensor(1., dtype=dtype, device=device)))

    # ── Random effects (intercept + slope per loan) ──
    # Non-centered parameterisation for better sampling
    tau = pyro.sample('tau', dist.Exponential(
        torch.ones(2, dtype=dtype, device=device)).to_event(1))
    L_Omega = pyro.sample('L_Omega', dist.LKJCholesky(
        2, concentration=torch.tensor(2., dtype=dtype, device=device)))
    L_sigma = torch.mm(torch.diag(tau), L_Omega)

    # Standard normal draws, then transform (non-centered)
    z_RE = pyro.sample('z_RE', dist.Normal(
        torch.zeros(N_loans, 2, dtype=dtype, device=device),
        torch.ones(N_loans, 2, dtype=dtype, device=device)).to_event(2))
    b = torch.mm(z_RE, L_sigma.T)  # (N_loans, 2): b[:,0]=U_0i, b[:,1]=U_1i

    # ── Longitudinal likelihood ──
    mu = alpha_0 + b[loan_id, 0] + b[loan_id, 1] * time_idx + phi * Y_lag
    pyro.sample('Y_obs', dist.Normal(mu, sigma).to_event(1), obs=Y)

    # ── Survival submodel parameters (cause-specific) ──
    a_prepay = pyro.sample('a_prepay', dist.Normal(
        torch.zeros(n_basis, dtype=dtype, device=device),
        2. * torch.ones(n_basis, dtype=dtype, device=device)).to_event(1))
    a_default = pyro.sample('a_default', dist.Normal(
        torch.zeros(n_basis, dtype=dtype, device=device),
        2. * torch.ones(n_basis, dtype=dtype, device=device)).to_event(1))
    gamma_prepay = pyro.sample('gamma_prepay', dist.Normal(
        torch.zeros(P, dtype=dtype, device=device),
        torch.ones(P, dtype=dtype, device=device)).to_event(1))
    gamma_default = pyro.sample('gamma_default', dist.Normal(
        torch.zeros(P, dtype=dtype, device=device),
        torch.ones(P, dtype=dtype, device=device)).to_event(1))
    lambda_prepay = pyro.sample('lambda_prepay', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(1., dtype=dtype, device=device)))
    lambda_default = pyro.sample('lambda_default', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(1., dtype=dtype, device=device)))

    # ── Survival likelihood (discrete-time competing risks) ──
    # For each loan, loop over months 1..event_time[i] and accumulate
    # the multinomial-logit log-likelihood.
    # m_i(t) = alpha_0 + U_0i + U_1i*t + phi*Y_i(t-1)  [noise-free]
    # eta_k(t) = B(t) @ a_k + Z_i @ gamma_k + lambda_k * m_i(t)
    #
    # Vectorised implementation groups observations by (loan, month)
    # using pre-built index tensors for efficiency.

    log_lik = compute_survival_log_lik(
        b, alpha_0, phi, Y_obs_by_loan, B_spline,
        a_prepay, a_default, Z, gamma_prepay, gamma_default,
        lambda_prepay, lambda_default, event_time, event_type)

    pyro.factor('survival_log_lik', log_lik)
```

Key design choices following `bayesian_phm.py` patterns:
- **float64 throughout** for numerical stability (matching the existing model)
- **Non-centered parameterisation** for random effects ($b_i = L_\Sigma \cdot z_i$ where $z_i \sim \mathcal{N}(0, I)$) -- critical for NUTS efficiency with hierarchical models
- **LKJ prior** on the correlation matrix of random effects (standard choice)
- **`pyro.factor`** for the survival contribution (same pattern as the existing model's `log_likelihood`)

### Python Wrapper Class

```python
class JointCompetingRisksModel:
    """
    Joint model for longitudinal and discrete survival data.
    Extends Medina-Olivares et al. (2022) to competing risks.

    Uses Pyro NUTS (same as BayesianCompetingRisksPHM in notebook 09).
    """

    def __init__(
        self,
        tvc_col: str = 'ppi_c_FRMA',
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        target_accept_prob: float = 0.90,
        random_seed: int = 42,
        device: str = 'cpu',
    ):
        ...

    def _model(self, Y, Y_lag, loan_id, time_idx, Z, B_spline,
               event_time, event_type, N_loans, n_basis, P):
        """Pyro model specification (see above)."""
        ...

    def fit(self, panel_df, static_cols, tvc_col):
        """
        Prepare data and run MCMC.

        Follows the same pattern as BayesianCompetingRisksPHM.fit():
        - Convert to float64 tensors
        - Set up NUTS kernel
        - Run MCMC
        - Store posterior_samples_ and inference_data_
        """
        pyro.set_rng_seed(self.random_seed)
        pyro.clear_param_store()

        nuts_kernel = NUTS(
            self._model,
            target_accept_prob=self.target_accept_prob,
            jit_compile=False,
        )
        self.mcmc_ = MCMC(
            nuts_kernel,
            num_samples=self.num_samples,
            warmup_steps=self.num_warmup,
            num_chains=self.num_chains,
        )
        self.mcmc_.run(...)
        self.posterior_samples_ = {
            k: v.cpu().numpy() for k, v in self.mcmc_.get_samples().items()
        }
        self.inference_data_ = az.from_pyro(self.mcmc_)
        return self

    def predict_cif(self, new_panel_df, horizons=[24, 48, 72],
                    n_mc_samples=200):
        """Compute predictive CIF for new loans."""
        ...

    def get_diagnostics(self):
        """R-hat and ESS via ArviZ (same as existing model)."""
        rhat = az.rhat(self.inference_data_)
        ess = az.ess(self.inference_data_)
        return {'rhat': rhat, 'ess': ess}

    def print_summary(self):
        """Print MCMC summary."""
        self.mcmc_.summary()
```

### MCMC Configuration

Following the paper, adapted to Pyro conventions:
- **Chains**: 4 (matching `bayesian_phm.py` default)
- **Warmup**: 1,000 iterations per chain
- **Sampling**: 2,000 iterations per chain
- **Target accept probability**: 0.90 (higher than the 0.8 default due to complex posterior geometry with random effects)

Start with these conservative settings. If diagnostics show issues (divergences, low ESS), increase warmup and raise target_accept_prob to 0.95.

Monitor convergence via:
- $\hat{R} < 1.05$ for all parameters (via `az.rhat()`)
- Effective sample size $n_{\text{eff}} > 400$ for all parameters (via `az.ess()`)
- No divergent transitions (reported by Pyro NUTS)
- Visual trace plots (via `az.plot_trace()`)

---

## 6. Prediction and CIF Computation

### For a new loan at time $s$ (with TVC history up to $s$)

**Step 1: Estimate random effects**

Given the observed TVC history $\bar{Y}_i(s)$, compute the posterior of $(U_{0i}, U_{1i})$ via empirical Bayes (mode of the conditional posterior) or MCMC sampling.

**Step 2: Project TVC forward**

For each future month $t > s$:

$$\hat{m}_i(t) = \alpha_0 + \hat{U}_{0i} + \hat{U}_{1i} \cdot t + \phi \, \hat{Y}_i(t-1)$$

where $\hat{Y}_i(t) = \hat{m}_i(t) + \epsilon$ with $\epsilon$ drawn from $\mathcal{N}(0, \hat{\sigma}^2)$. Run multiple Monte Carlo draws to propagate uncertainty.

**Step 3: Compute CIF**

Chain the monthly transition probabilities:

$$\text{CIF}_k(t) = \text{CIF}_k(t-1) + S(t-1) \cdot p_{ik}(t)$$
$$S(t) = S(t-1) \cdot (1 - p_{i1}(t) - p_{i2}(t))$$

Average over Monte Carlo draws (over both parameter posterior samples and TVC projections) to get the predictive CIF with uncertainty bands.

### Comparison with Sadhwani CIF Methods

| Aspect | Sadhwani Frozen | Sadhwani AR | Joint Model |
|--------|----------------|------------|-------------|
| Macro projection | Held at t=0 value | AR(p), fitted separately | AR(1), fitted jointly |
| Coefficients | Trained on observed TVCs | Trained on observed TVCs | Trained on projected TVCs |
| Random effects | None | None | Per-loan $(U_{0i}, U_{1i})$ |
| Uncertainty | None (point estimate) | Monte Carlo over macro paths | Full posterior (parameters + random effects + TVC) |

---

## 7. Evaluation

Use the **same evaluation framework** as all other notebooks for direct comparability.

### Primary Metrics

| Metric | Function | Time Points |
|--------|----------|-------------|
| Time-dependent C-index | `time_dependent_concordance_index()` | 24, 48, 72 months |
| Brier score | `brier_score_competing_risks()` | 24, 48, 72 months |

Both evaluated separately for prepayment ($k=1$) and default ($k=2$).

Risk score for the C-index = predicted CIF at the evaluation time point, computed via the forward projection procedure in Section 6.

### Comparison Table Structure

```
                    Prepayment C-index          Default C-index
Model           C(24)   C(48)   C(72)  OC    C(24)   C(48)   C(72)  OC
-----------     -----   -----   -----  --    -----   -----   -----  --
CSC              ...     ...     ...   ...    ...     ...     ...   ...
FGR              ...     ...     ...   ...    ...     ...     ...   ...
RSF              ...     ...     ...   ...    ...     ...     ...   ...
DeepHit          ...     ...     ...   ...    ...     ...     ...   ...
Sadhwani (froz)  ...     ...     ...   ...    ...     ...     ...   ...
Sadhwani (AR)    ...     ...     ...   ...    ...     ...     ...   ...
Joint (M5)       ...     ...     ...   ...    ...     ...     ...   ...
```

### What We Expect to See

Based on the paper's findings:
- **Short horizons (t=24)**: Joint model may not outperform simpler models. The endogeneity problem is mild when most of the TVC trajectory is observed, leaving little to project.
- **Long horizons (t=72)**: Joint model should show the largest improvements. The TVC projection advantage compounds over time, and the random effects better separate heterogeneous loans.
- **Default**: Largest improvement expected here. Default is rarer and more sensitive to TVC dynamics than prepayment.
- **Prepayment**: Improvement depends on how informative `ppi_c_FRMA` is for prepayment after controlling for static covariates.

---

## 8. Implementation Steps

### Step 1: Data Preparation

Create the data structures needed for Pyro from the existing `loan_month_panel.parquet`:

```python
def prepare_joint_model_data(panel_df, tvc_col='ppi_c_FRMA'):
    """
    Transform loan-month panel into PyTorch tensors for Pyro.
    
    Returns dict with:
    - N, N_loans, T_max, n_basis
    - loan_id, time_idx, Y, Y_lag (longitudinal)
    - event_time, event_type (survival)
    - Z (time-invariant covariates, standardised)
    - B (B-spline basis matrix for baseline hazard)
    """
```

Key transformations:
- Standardise all covariates (zero mean, unit variance) on training set
- Create lagged TVC column ($Y_{i,t-1}$) per loan
- Build B-spline basis matrix with 3 interior knots at the 25th, 50th, 75th percentiles of event times
- Map loan IDs to contiguous integer indices [1, N_loans]

### Step 2: Python Module

Create `src/competing_risks/joint_model.py` with the `JointCompetingRisksModel` class following the same patterns as `bayesian_phm.py`:

- Pyro model function with `pyro.sample` / `pyro.factor`
- `fit()` using `NUTS` + `MCMC` from Pyro
- `predict_cif()` using posterior samples
- `get_diagnostics()` / `print_summary()` via ArviZ

### Step 4: Notebook

Create `notebooks/19_joint_model.ipynb`:

| Section | Content |
|---------|---------|
| 1. Setup | Imports, config, device |
| 2. Data | Load panel, prepare tensors, train/test split |
| 3. Model | Fit Pyro model via NUTS |
| 4. Diagnostics | Trace plots, R-hat, effective sample size, divergences |
| 5. Parameter Estimates | Posterior summaries, comparison with paper Table 3 |
| 6. Longitudinal Fit | Plot fitted vs observed TVC trajectories for sample loans |
| 7. CIF Prediction | Compute CIF on test set |
| 8. Evaluation | C-index and Brier score at 24/48/72 months |
| 9. Comparison | Side-by-side with all other models |
| 10. Association | Interpret $\lambda_{\text{prepay}}$ vs $\lambda_{\text{default}}$ (sign, magnitude, credible intervals) |

### Step 5: Model Variants

Implement and compare (mirroring the paper's M0-M5 progression):

| Variant | Longitudinal | Random Effects | AR | Purpose |
|---------|-------------|---------------|-----|---------|
| JM0 | -- | -- | -- | Baseline: discrete-time competing risks (no TVC) |
| JM1 | Yes | Intercept only | No | Does the TVC add value? |
| JM3 | Yes | Intercept + slope | No | Does loan-specific trend matter? |
| JM5 | Yes | Intercept + slope | Yes | Full model -- does AR improve? |

---

## 9. Computational Considerations

### Sample Size

The paper uses 10,399 loans (285K observations). Our dataset is similar in size, so run times should be comparable. If the dataset is larger, subsample to ~10,000 loans for initial development and scale up once the model is validated.

### Expected Run Time

Based on the paper's setup (3 chains x 8,000 iterations):
- **N = 1,000 loans**: ~30 minutes
- **N = 5,000 loans**: ~2-3 hours
- **N = 10,000 loans**: ~5-8 hours

The bottleneck is the per-loan random effects -- each loan adds 2 parameters to the posterior, so 10,000 loans means 20,000+ parameters for HMC to navigate.

### Strategies for Scaling

1. **Start small**: Develop and debug with 1,000 loans, validate against paper results
2. **Within-chain parallelism**: Pyro supports `pyro.plate` for vectorised likelihood evaluation across loans
3. **Variational inference**: Use Pyro's SVI for quick approximate posteriors during development, switch to full MCMC for final results
4. **GPU acceleration**: Pyro/PyTorch supports MPS (Apple Silicon) and CUDA natively for GPU-accelerated sampling

### Device Selection: MPS vs CPU

The model defaults to CPU, matching `bayesian_phm.py`. While MPS (Apple Silicon) works well for the neural network models (Sadhwani, DeepHit) that use float32, NUTS-based MCMC requires float64 throughout for numerical stability. PyTorch's MPS backend has incomplete float64 support -- operations either silently fall back to float32 or raise errors. With 20,000+ parameters (per-loan random effects), float32 precision can cause divergences and poor chain mixing. Use CPU for MCMC; reserve MPS/CUDA for the SVI prototyping path where float32 is acceptable.

---

## 10. Dependencies

**No new dependencies required.** Everything is already installed for the Bayesian competing risks notebook (09):

| Package | Purpose | Status |
|---------|---------|--------|
| `pyro-ppl` | Probabilistic programming, NUTS sampler | Already installed |
| `torch` | Tensor computation, autodiff | Already installed |
| `arviz` | MCMC diagnostics (R-hat, ESS, trace plots) | Already installed |
| `patsy` or `scipy` | B-spline basis construction | `scipy.interpolate.BSpline` already available |
| `numpy`, `pandas`, `matplotlib` | Standard stack | Already installed |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MCMC convergence issues | Medium | Start with simpler variants (JM1), raise target_accept_prob, use non-centered parameterisation for random effects |
| Slow run time on full dataset | High | Subsample for development, vectorise survival likelihood, use Pyro's SVI (variational inference) for prototyping |
| Competing risks extension adds complexity | Medium | Validate single-event version against paper first, then add second cause |
| Single TVC may not capture enough variation | Medium | Start with ppi_c_FRMA, extend to multivariate if discrimination is poor |

---

## 12. Validation Strategy

1. **Reproduce the paper's single-event results** (default only, single TVC) on our data. Compare parameter estimates ($\phi$, $\lambda$, random effect variances) with Table 3 of the paper. They should be in the same ballpark given similar data.

2. **Extend to competing risks** and verify:
   - $\lambda_{\text{prepay}}$ and $\lambda_{\text{default}}$ have plausible signs
   - CIF curves sum to less than 1 at all horizons
   - Model produces calibrated probabilities (Brier score)

3. **Compare against existing models** using the standard evaluation framework to answer: does properly modelling TVC endogeneity improve mortgage risk prediction in practice?

---

## 13. References

1. Medina-Olivares, V., Calabrese, R., Crook, J. & Lindgren, F. (2022). Joint models for longitudinal and discrete survival data in credit scoring. *EJOR*, 307(3), 1457-1473.
2. Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event Data*. Chapman & Hall/CRC.
3. Blumenstock, G., Lessmann, S. & Seow, H-V. (2022). Deep learning for survival and competing risk modelling. *JORS*, 73(1), 26-38.
4. Giesecke, S., Sirignano, J. & Sadhwani, K. (2021). Deep Learning for Mortgage Risk. *JFE*, 19(2), 313-368.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/competing_risks/joint_model.py` | Pyro model, wrapper class (data prep, fitting, prediction, CIF) -- follows `bayesian_phm.py` patterns |
| `notebooks/19_joint_model.ipynb` | Analysis notebook |
