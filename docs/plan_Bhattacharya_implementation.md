# Implementation Plan: Bhattacharya et al. (2018) Bayesian Competing Risks Model

## Paper Reference

**Bhattacharya, A., Wilson, S.P., & Soyer, R. (2019).** *A Bayesian approach to modeling mortgage default and prepayment.* European Journal of Operational Research, 274, 1112–1124.

---

## Paper Summary

The paper presents a **Bayesian competing risks proportional hazards model** for mortgage default and prepayment with:
- **Lognormal baseline hazards** for both default and prepayment (non-monotonic hazard rates)
- **Separate regression coefficients** (θ_D, θ_P) for each competing risk
- **MCMC inference** via Metropolis-within-Gibbs sampling
- **Posterior predictive distributions** for default/prepayment times

### Key Advantages of Bayesian Approach
1. **Full uncertainty quantification** via posterior distributions
2. **Probabilistic predictions** with credible intervals
3. **Natural handling of prior information**
4. **Coherent predictive inference** for new loans

---

## Step-by-Step Implementation Plan

### **Step 1: Data Preparation**

**Goal**: Transform existing loan-month panel data into the format required by the model.

**Tasks**:
1. Load `loan_month_panel.parquet` (already have ~109K loans, ~5.4M loan-months)
2. Create terminal observations dataset:
   - For each loan, identify final status: Default (event_code=2), Prepay (event_code=1), or Censored (event_code=0)
   - Extract time to event (loan_age in months)
3. Map covariates to paper's specification:

   | Paper Covariate | Our Data Variable | Notes |
   |-----------------|-------------------|-------|
   | credit_score | fico_score | Standardize |
   | mortgage_insurance_% | mi_pct | If available |
   | num_units | num_units | Categorical |
   | CLTV | orig_cltv or ltv_r | Standardize |
   | DTI | dti_r | Standardize |
   | UPB | orig_upb → log_upb | Log-transform, standardize |
   | orig_interest_rate | int_rate | Standardize |
   | num_borrowers | num_borrowers | Categorical |
   | first_homebuyer | first_time_homebuyer | Binary indicator |
   | occupancy_status | occupancy_status | Binary (owner-occupied vs other) |
   | property_type | property_type | Binary (single-family vs other) |
   | foreclosure_state | property_state | Binary (judicial vs non-judicial) |

4. Standardize continuous covariates (zero mean, unit variance)
5. Convert categorical variables to indicator (0/1) variables
6. Handle time-varying covariates: Use piecewise-constant assumption with monthly observations

**Output**:
- `X`: (N × m) covariate matrix
- `t_D`, `t_P`, `t_C`: Event times for defaulted, prepaid, and censored loans
- `event_indicators`: Classification of each loan

---

### **Step 2: Model Specification**

**Goal**: Implement the competing risks PHM with lognormal baseline hazards.

#### Mathematical Model

**Mortgage lifetime**:
$$L = \min(T_D, T_P, T_M)$$

where $T_D$ = time to default, $T_P$ = time to prepayment, $T_M$ = maturity date.

**Default hazard** (Eq. 1 in paper):
$$\lambda_D(t | X_D(t)) = r_D(t | \mu_D, \sigma_D) \exp(\theta_D' X_D(t))$$

**Prepayment hazard** (Eq. 2 in paper):
$$\lambda_P(t | X_P(t)) = r_P(t | \mu_P, \sigma_P) \exp(\theta_P' X_P(t))$$

**Lognormal baseline hazard** (Eq. 3 & 4 in paper):
$$r(t | \mu, \sigma) = \frac{(2\pi\sigma^2)^{-1/2} t^{-1} \exp\left(-\frac{(\log(t) - \mu)^2}{2\sigma^2}\right)}{1 - \Phi\left(\frac{\log(t) - \mu}{\sigma}\right)}$$

where $\Phi$ is the standard normal CDF.

**Parameters to estimate**:
- Baseline parameters: $\psi = (\mu_D, \sigma_D, \mu_P, \sigma_P)$ — 4 parameters
- Default regression coefficients: $\theta_D$ — m parameters
- Prepayment regression coefficients: $\theta_P$ — m parameters
- **Total**: 4 + 2m parameters

**Conditional Independence Assumption**:
$$P(T_D > t_D, T_P > t_P | \text{params}, X) = P(T_D > t_D | \text{params}, X) \times P(T_P > t_P | \text{params}, X)$$

---

### **Step 3: Likelihood Function**

**Goal**: Implement the competing risks likelihood (Eq. 7 in paper).

#### Likelihood Structure

For N mortgages with $n_D$ defaults, $n_P$ prepayments, and $n_C = N - n_D - n_P$ censored:

$$\mathcal{L} = \underbrace{\prod_{i=1}^{n_D} \lambda_D(t_i^D) \exp\left(-\int_0^{t_i^D} [\lambda_D(w) + \lambda_P(w)] dw\right)}_{\text{Defaulted loans}}$$

$$\times \underbrace{\prod_{i=n_D+1}^{n_D+n_P} \lambda_P(t_i^P) \exp\left(-\int_0^{t_i^P} [\lambda_D(w) + \lambda_P(w)] dw\right)}_{\text{Prepaid loans}}$$

$$\times \underbrace{\prod_{i=n_D+n_P+1}^{N} \exp\left(-\int_0^{t_i^C} [\lambda_D(w) + \lambda_P(w)] dw\right)}_{\text{Censored loans}}$$

#### Cumulative Hazard Integral (Closed Form)

The integral of the lognormal hazard has a closed-form expression (Eq. A.3-A.4 in Appendix):

$$\int_{t_a}^{t_b} r(s | \mu, \sigma) ds = -\log\left(1 - \Phi\left[\frac{\log(t_b) - \mu}{\sigma}\right]\right) + \log\left(1 - \Phi\left[\frac{\log(t_a) - \mu}{\sigma}\right]\right)$$

For **piecewise-constant covariates** observed at times $\tau_1 < \tau_2 < \cdots < \tau_m$:

$$\int_0^{t} \lambda(w | X(w)) dw = \sum_{j=1}^{m'} \exp(\theta' X(\tau_j)) \times \left[\int_{s_{j-1}}^{s_j} r(w) dw\right] + \exp(\theta' X(\tau_{m'})) \times \left[\int_{s_{m'}}^{t} r(w) dw\right]$$

where $m' = \max\{j | \tau_j < t\}$ and $s_j = 0.5(\tau_j + \tau_{j+1})$.

---

### **Step 4: Prior Specification**

**Goal**: Implement priors matching the paper (Section 5).

| Parameter | Prior Distribution | Hyperparameters |
|-----------|-------------------|-----------------|
| $\theta_D$ (each component) | Normal | mean=0, sd=100 |
| $\theta_P$ (each component) | Normal | mean=0, sd=100 |
| $\mu_D$ | Normal | mean=0, sd=10 |
| $\mu_P$ | Normal | mean=0, sd=10 |
| $\sigma_D$ | Exponential | mean=100 |
| $\sigma_P$ | Exponential | mean=100 |

**Note**: These are weakly informative priors that allow the data to dominate inference.

---

### **Step 5: MCMC Sampler Implementation**

**Goal**: Implement Metropolis-within-Gibbs sampler (detailed in paper's Appendix).

#### Sampling Blocks

**Block 1: Sample $\theta_D$** (m-dimensional block update)
- Proposal: $\theta_D^* \sim \mathcal{N}(\theta_D, s^2_{\theta,D} \times I_{m \times m})$
- Accept with probability:
$$\min\left(1, \frac{p(\theta_D^*) \prod_{i=1}^{n_D} \lambda_D^*(t_i^D)}{p(\theta_D) \prod_{i=1}^{n_D} \lambda_D(t_i^D)} \times \frac{\exp(-A^* - B^* - C^*)}{\exp(-A - B - C)}\right)$$

where $A, B, C$ are cumulative hazard contributions from defaulted, prepaid, and censored loans.

**Block 2: Sample $\theta_P$** (m-dimensional block update)
- Same structure as $\theta_D$, using prepayment hazard

**Block 3: Sample $\mu_D$** (scalar)
- Proposal: $\mu_D^* \sim \mathcal{N}(\mu_D, s^2_{\mu,D})$
- Standard MH acceptance

**Block 4: Sample $\mu_P$** (scalar)
- Same structure as $\mu_D$

**Block 5: Sample $\sigma_D$** (scalar)
- Proposal: $\sigma_D^{2,*} \sim \text{Uniform}(a \cdot \sigma_D^2, \sigma_D^2 / a)$ where $a \in (0, 1)$
- **Note**: Asymmetric proposal requires Jacobian correction in MH ratio

**Block 6: Sample $\sigma_P$** (scalar)
- Same structure as $\sigma_D$

#### Implementation Options

| Option | Library | Pros | Cons |
|--------|---------|------|------|
| **Pyro + NUTS** | Pyro/PyTorch | PyTorch ecosystem, GPU support (CUDA/MPS), good integration with deep learning | Slightly slower than NumPyro |
| **NumPyro + NUTS** | NumPyro/JAX | Fastest, automatic tuning | Requires JAX learning curve |
| **PyMC + NUTS** | PyMC | Mature, good diagnostics | Slower than NumPyro |
| **Custom MH** | NumPy/Numba | Exact replication of paper | Manual tuning, slower |

**Chosen**: Pyro with NUTS sampler for PyTorch ecosystem integration and broad device support (CPU, CUDA, MPS).

#### MCMC Settings (from paper)
- Chains: 50 (can reduce to 4-8 for practical purposes)
- Iterations per chain: 75,000
- Burn-in (warmup): 60,000
- Thinning: every 50th sample
- Final posterior samples: ~15,000

**Practical settings for our implementation**:
- Chains: 4
- Warmup: 2,000-5,000
- Samples: 5,000-10,000
- Thinning: 1 (NUTS is less autocorrelated)

---

### **Step 6: Posterior Predictive Inference**

**Goal**: Compute predictive quantities for model assessment and prediction.

#### Posterior Predictive Survival Functions (Eq. 8 & 9)

$$P(T_D > t | \text{data}, X) \approx \frac{1}{G} \sum_{g=1}^G \exp\left(-\int_0^t \lambda_D^{(g)}(w) dw\right)$$

$$P(T_P > t | \text{data}, X) \approx \frac{1}{G} \sum_{g=1}^G \exp\left(-\int_0^t \lambda_P^{(g)}(w) dw\right)$$

where $G$ is the number of posterior samples.

#### Posterior Predictive Density

$$f_D(t) \approx \frac{1}{G} \sum_{g=1}^G \lambda_D^{(g)}(t) \exp\left(-\int_0^t \lambda_D^{(g)}(w) dw\right)$$

#### Event Probability Prediction

For a loan with covariates $X$ and maturity $T_M$:

1. For each posterior sample $g = 1, \ldots, G$:
   - Simulate $t_D^{(g)}$ from posterior predictive of $T_D$
   - Simulate $t_P^{(g)}$ from posterior predictive of $T_P$

2. Classify each simulation:
   - **Default**: $t_D < T_M$ and $t_D < t_P$
   - **Prepay**: $t_P < T_M$ and $t_P < t_D$
   - **Mature**: $t_D \geq T_M$ and $t_P \geq T_M$

3. Estimate probabilities as proportions:
$$\hat{P}(\text{Default}) = \frac{1}{G} \sum_{g=1}^G \mathbf{1}[t_D^{(g)} < T_M \text{ and } t_D^{(g)} < t_P^{(g)}]$$

#### Cumulative Incidence Function (CIF)

For competing risks, the cause-specific CIF:
$$F_k(t) = P(T \leq t, \text{cause} = k) = \int_0^t \lambda_k(u) S(u) du$$

where $S(u) = \exp(-\int_0^u [\lambda_D(w) + \lambda_P(w)] dw)$ is the overall survival.

---

### **Step 7: Performance Evaluation with Time-Dependent Concordance Index**

**Goal**: Evaluate model performance using metrics consistent with other notebooks for comparison.

#### 7.1 Time-Dependent C-Index (IPCW)

Compute **Inverse Probability of Censoring Weighted (IPCW) C-index** at time horizons τ = 24, 48, 72 months, matching the evaluation in notebooks 05-08.

**For each cause k (default, prepay)**:

$$C^{\text{td}}(\tau) = P(\hat{F}_k(τ | X_i) > \hat{F}_k(\tau | X_j) | T_i < T_j, T_i \leq \tau, \delta_i = k)$$

where $\hat{F}_k(\tau | X)$ is the predicted CIF at time $\tau$.

**Implementation**:
```python
from sksurv.metrics import concordance_index_ipcw

# For each time horizon
for tau in [24, 48, 72]:
    # Get posterior mean CIF at tau
    cif_at_tau = posterior_mean_cif[:, tau_index]

    # Compute IPCW C-index
    c_index = concordance_index_ipcw(
        survival_train, survival_test,
        estimate=cif_at_tau, tau=tau
    )
```

#### 7.2 Overall Concordance Index (Harrell's C)

$$C = \frac{\sum_{i,j} \mathbf{1}[\hat{r}_i > \hat{r}_j] \cdot \mathbf{1}[T_i < T_j] \cdot \delta_i}{\sum_{i,j} \mathbf{1}[T_i < T_j] \cdot \delta_i}$$

Use mean CIF across time as risk score.

#### 7.3 Brier Score (Time-Dependent)

$$BS(t) = \frac{1}{N} \sum_{i=1}^N \hat{W}_i(t) \left[\mathbf{1}(T_i \leq t, \delta_i = k) - \hat{F}_k(t | X_i)\right]^2$$

where $\hat{W}_i(t)$ are IPCW weights.

**Integrated Brier Score (IBS)**:
$$IBS = \frac{1}{t_{max}} \int_0^{t_{max}} BS(t) dt$$

#### 7.4 Calibration Assessment

1. **Calibration plots**: Predicted vs observed event rates by decile
2. **Hosmer-Lemeshow type test** for grouped predictions

#### 7.5 Posterior Predictive Checks

Bayesian-specific diagnostics:

1. **Coverage probability**: % of observed times within 95% posterior credible interval
   - Target: ~95% for well-calibrated model

2. **Posterior predictive p-values**:
   $$p_i = P(T_i^{rep} < T_i^{obs} | \text{data})$$
   Should be approximately Uniform(0,1) if model is well-specified.

3. **Standardized residuals**:
   $$r_i = \frac{t_i^{obs} - E[T_i | \text{data}]}{\text{sd}[T_i | \text{data}]}$$
   Should be approximately N(0,1).

#### 7.6 Comparison Table Structure

| Model | C-index (τ=24) | C-index (τ=48) | C-index (τ=72) | C-index (Overall) | IBS |
|-------|----------------|----------------|----------------|-------------------|-----|
| **Prepayment** |
| Cause-Specific Cox | - | - | - | - | - |
| Fine-Gray | - | - | - | - | - |
| Random Survival Forest | - | - | - | - | - |
| DeepHit | 0.977 | 0.941 | 0.904 | 0.958 | - |
| **Bayesian PHM** | TBD | TBD | TBD | TBD | TBD |
| **Default** |
| Cause-Specific Cox | - | - | - | - | - |
| Fine-Gray | - | - | - | - | - |
| Random Survival Forest | - | - | - | - | - |
| DeepHit | 0.998 | 0.999 | 0.892 | 0.899 | - |
| **Bayesian PHM** | TBD | TBD | TBD | TBD | TBD |

---

### **Step 8: Model Assessment (Paper-Specific)**

**Goal**: Replicate assessment methods from the paper.

#### 8.1 Standardized Residuals (Fig. 5, 6 in paper)

For observed events:
$$r_i = \frac{t_i - E[T_i | \text{posterior}]}{\text{sd}[T_i | \text{posterior}]}$$

**Expected**: Centered around 0 for well-calibrated model.

#### 8.2 Coverage Probability

From paper (Section 6):
- Overall coverage: 93%
- Prepay coverage: 95%
- Default coverage: 50% (poor due to data imbalance)

#### 8.3 Posterior Reliability at Observed Time (Fig. 7 in paper)

$$R_i = P(T > t_i^{obs} | \text{posterior}, X_i)$$

For well-calibrated model, $R_i$ should be approximately Uniform(0,1).

---

### **Step 9: Notebook Structure**

Create `notebooks/09_bayesian_competing_risks.ipynb` with sections:

```markdown
# Bayesian Competing Risks Model (Bhattacharya et al. 2018)

## 1. Introduction & Methodology
   - Paper overview and motivation
   - Model equations and assumptions
   - Comparison with frequentist approaches

## 2. Setup & Imports
   - Library imports (Pyro, PyTorch, ArviZ)
   - Configuration constants
   - Random seed setting

## 3. Data Preparation
   - Load loan-month panel data
   - Create terminal observations
   - Feature engineering & standardization
   - Train/validation/test split (folds 0-8 / 9 / 10)

## 4. Model Implementation
   - Lognormal hazard functions (PyTorch)
   - Likelihood function
   - Prior specification
   - Pyro model definition

## 5. MCMC Inference
   - Run MCMC sampler
   - Convergence diagnostics (trace plots, R-hat, ESS)
   - Posterior summaries (Table 1 & 2 equivalents)

## 6. Results & Interpretation
   - Posterior distributions of baseline parameters
   - Posterior distributions of regression coefficients
   - Covariate effects interpretation
   - Comparison with paper's results

## 7. Posterior Predictive Analysis
   - Predictive survival functions
   - Predictive CIF curves
   - Event probability predictions
   - Sample loan predictions (Fig. 4 equivalent)

## 8. Performance Evaluation
   ### 8.1 Time-Dependent Concordance Index
   - C-index at τ = 24, 48, 72 months (IPCW)
   - Overall Harrell's C-index

   ### 8.2 Brier Score
   - Time-dependent Brier score
   - Integrated Brier Score

   ### 8.3 Calibration
   - Calibration plots by decile
   - Coverage probabilities

   ### 8.4 Model Comparison
   - Comparison table with Cox, RSF, DeepHit
   - Statistical significance of differences

## 9. Bayesian-Specific Diagnostics
   - Posterior predictive checks
   - Residual analysis (Fig. 5, 6 equivalent)
   - Reliability function assessment (Fig. 7 equivalent)

## 10. Save Results
   - Save posterior samples
   - Save predictions
   - Save performance metrics

## 11. Summary & Conclusions
   - Key findings
   - Advantages of Bayesian approach
   - Limitations and future work
```

---

### **Step 10: Code Structure**

#### New Module: `src/competing_risks/bayesian_phm.py`

```python
"""
Bayesian Competing Risks Proportional Hazards Model
Based on Bhattacharya, Wilson & Soyer (2019)

Uses Pyro (PyTorch-based probabilistic programming) for MCMC inference.
"""

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from typing import Dict, Tuple, Optional
import arviz as az


class BayesianCompetingRisksPHM:
    """
    Bayesian competing risks proportional hazards model with
    lognormal baseline hazards.

    Uses Pyro for MCMC inference via the NUTS sampler.
    Supports CPU, CUDA (NVIDIA GPU), and MPS (Apple Silicon) devices.

    Reference:
        Bhattacharya, A., Wilson, S.P., & Soyer, R. (2019).
        A Bayesian approach to modeling mortgage default and prepayment.
        European Journal of Operational Research, 274, 1112-1124.

    Parameters
    ----------
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Number of posterior samples per chain
    n_warmup : int
        Number of warmup (burn-in) samples
    random_seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_chains: int = 4,
        n_samples: int = 5000,
        n_warmup: int = 2000,
        random_seed: int = 42
    ):
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.random_seed = random_seed
        self.posterior_samples_ = None
        self.mcmc_ = None

    def _lognormal_hazard(
        self,
        t: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute lognormal baseline hazard rate.

        r(t) = φ((log(t) - μ)/σ) / (σ * t * (1 - Φ((log(t) - μ)/σ)))

        where φ is standard normal PDF and Φ is standard normal CDF.
        """
        z = (torch.log(t) - mu) / sigma
        # Standard normal CDF using error function
        Phi_z = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        # Standard normal PDF
        phi_z = torch.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        denominator = sigma * t * (1 - Phi_z)
        return phi_z / (denominator + 1e-10)

    def _cumulative_baseline_hazard(
        self,
        t: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cumulative baseline hazard H_0(t) = -log(S_0(t)).

        For lognormal: H_0(t) = -log(1 - Φ((log(t) - μ)/σ))
        """
        z = (torch.log(t) - mu) / sigma
        Phi_z = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        return -torch.log(1 - Phi_z + 1e-10)

    def _model(
        self,
        X: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,  # 0=censored, 1=prepay, 2=default
        n_features: int
    ):
        """Pyro model specification."""
        device = X.device

        # Priors for baseline hazard parameters
        mu_D = pyro.sample('mu_D', dist.Normal(0, 10))
        mu_P = pyro.sample('mu_P', dist.Normal(0, 10))
        sigma_D = pyro.sample('sigma_D', dist.Exponential(0.01))  # mean=100
        sigma_P = pyro.sample('sigma_P', dist.Exponential(0.01))

        # Priors for regression coefficients
        theta_D = pyro.sample(
            'theta_D',
            dist.Normal(torch.zeros(n_features, device=device),
                       100 * torch.ones(n_features, device=device)).to_event(1)
        )
        theta_P = pyro.sample(
            'theta_P',
            dist.Normal(torch.zeros(n_features, device=device),
                       100 * torch.ones(n_features, device=device)).to_event(1)
        )

        # Compute hazards and cumulative hazards
        linear_pred_D = torch.matmul(X, theta_D)
        linear_pred_P = torch.matmul(X, theta_P)

        # Baseline cumulative hazards at observed times
        H0_D = self._cumulative_baseline_hazard(durations, mu_D, sigma_D)
        H0_P = self._cumulative_baseline_hazard(durations, mu_P, sigma_P)

        # Full cumulative hazards
        H_D = H0_D * torch.exp(linear_pred_D)
        H_P = H0_P * torch.exp(linear_pred_P)

        # Baseline hazards at observed times
        h0_D = self._lognormal_hazard(durations, mu_D, sigma_D)
        h0_P = self._lognormal_hazard(durations, mu_P, sigma_P)

        # Full hazards
        h_D = h0_D * torch.exp(linear_pred_D)
        h_P = h0_P * torch.exp(linear_pred_P)

        # Log-likelihood contributions
        is_default = (events == 2).float()
        is_prepay = (events == 1).float()

        log_lik = (
            is_default * torch.log(h_D + 1e-10) +
            is_prepay * torch.log(h_P + 1e-10) -
            H_D - H_P
        )

        pyro.factor('log_likelihood', torch.sum(log_lik))

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        device: str = 'cpu'
    ) -> 'BayesianCompetingRisksPHM':
        """
        Fit the model using MCMC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix
        durations : array-like of shape (n_samples,)
            Observed durations (time to event or censoring)
        events : array-like of shape (n_samples,)
            Event indicators: 0=censored, 1=prepay, 2=default
        device : str
            Device to use ('cpu', 'cuda', or 'mps')

        Returns
        -------
        self
        """
        pyro.clear_param_store()
        pyro.set_rng_seed(self.random_seed)

        X = torch.tensor(X, dtype=torch.float32, device=device)
        durations = torch.tensor(durations, dtype=torch.float32, device=device)
        events = torch.tensor(events, dtype=torch.int64, device=device)
        n_features = X.shape[1]

        # Set up MCMC
        kernel = NUTS(self._model, jit_compile=False)
        self.mcmc_ = MCMC(
            kernel,
            warmup_steps=self.n_warmup,
            num_samples=self.n_samples,
            num_chains=self.n_chains
        )

        # Run MCMC
        self.mcmc_.run(X, durations, events, n_features)

        # Store posterior samples
        self.posterior_samples_ = {k: v.cpu().numpy()
                                   for k, v in self.mcmc_.get_samples().items()}

        return self

    def predict_survival(
        self,
        X_new: np.ndarray,
        times: np.ndarray,
        cause: str = 'overall'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict survival function with credible intervals.

        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features)
            Covariate matrix for new observations
        times : array-like of shape (n_times,)
            Times at which to evaluate survival
        cause : str
            'default', 'prepay', or 'overall'

        Returns
        -------
        mean : array of shape (n_samples, n_times)
            Posterior mean survival
        lower : array of shape (n_samples, n_times)
            2.5% quantile
        upper : array of shape (n_samples, n_times)
            97.5% quantile
        """
        # Implementation details...
        pass

    def predict_cif(
        self,
        X_new: np.ndarray,
        times: np.ndarray,
        cause: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict cumulative incidence function with credible intervals.

        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features)
            Covariate matrix for new observations
        times : array-like of shape (n_times,)
            Times at which to evaluate CIF
        cause : str
            'default' or 'prepay'

        Returns
        -------
        mean, lower, upper : arrays of shape (n_samples, n_times)
        """
        # Implementation details...
        pass

    def predict_event_probabilities(
        self,
        X_new: np.ndarray,
        maturity: float = 360
    ) -> Dict[str, np.ndarray]:
        """
        Predict probabilities of default, prepay, and maturity.

        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features)
        maturity : float
            Loan maturity in months (default 360 = 30 years)

        Returns
        -------
        dict with keys 'default', 'prepay', 'mature', each containing
        arrays of shape (n_samples,) with posterior mean probabilities
        """
        # Implementation via simulation from posterior predictive
        pass

    def get_posterior_summary(self) -> Dict:
        """
        Get summary statistics for all parameters.

        Returns
        -------
        dict with parameter names as keys, containing:
            - mean, median, std
            - 95% credible interval (2.5%, 97.5%)
        """
        if self.posterior_samples_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        summary = {}
        for param, samples in self.posterior_samples_.items():
            samples = np.array(samples)
            if samples.ndim == 1:
                summary[param] = {
                    'mean': np.mean(samples),
                    'median': np.median(samples),
                    'std': np.std(samples),
                    'ci_lower': np.percentile(samples, 2.5),
                    'ci_upper': np.percentile(samples, 97.5)
                }
            else:
                # For vector parameters (theta_D, theta_P)
                for i in range(samples.shape[1]):
                    summary[f'{param}[{i}]'] = {
                        'mean': np.mean(samples[:, i]),
                        'median': np.median(samples[:, i]),
                        'std': np.std(samples[:, i]),
                        'ci_lower': np.percentile(samples[:, i], 2.5),
                        'ci_upper': np.percentile(samples[:, i], 97.5)
                    }
        return summary

    def to_arviz(self) -> az.InferenceData:
        """Convert MCMC output to ArviZ InferenceData for diagnostics."""
        if self.mcmc_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return az.from_pyro(self.mcmc_)
```

#### Evaluation Functions: `src/competing_risks/bayesian_evaluation.py`

```python
"""
Evaluation functions for Bayesian competing risks model.
"""

import numpy as np
from typing import Dict, Tuple
from sksurv.metrics import concordance_index_ipcw, brier_score


def compute_time_dependent_cindex(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    times: list = [24, 48, 72],
    cause: str = 'default'
) -> Dict[int, float]:
    """
    Compute time-dependent C-index at specified time horizons.

    Parameters
    ----------
    model : BayesianCompetingRisksPHM
        Fitted model
    X_train, y_train : training data (for censoring distribution)
    X_test, y_test : test data
    times : list of time horizons
    cause : 'default' or 'prepay'

    Returns
    -------
    dict mapping time horizon to C-index value
    """
    results = {}

    for tau in times:
        # Get posterior mean CIF at time tau
        cif_mean, _, _ = model.predict_cif(X_test, np.array([tau]), cause)
        risk_score = cif_mean[:, 0]  # CIF at tau

        # Compute IPCW C-index
        c_index, _, _, _, _ = concordance_index_ipcw(
            y_train, y_test, risk_score, tau=tau
        )
        results[tau] = c_index

    return results


def compute_brier_score(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    times: np.ndarray,
    cause: str = 'default'
) -> Tuple[np.ndarray, float]:
    """
    Compute time-dependent Brier score and integrated Brier score.

    Returns
    -------
    bs : array of Brier scores at each time point
    ibs : integrated Brier score
    """
    # Implementation...
    pass


def compute_calibration(
    model,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    n_groups: int = 10
) -> Dict:
    """
    Compute calibration metrics by decile of predicted risk.
    """
    # Implementation...
    pass


def compute_coverage_probability(
    model,
    X_test: np.ndarray,
    durations_test: np.ndarray,
    events_test: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute coverage probability of (1-alpha)% credible intervals.
    """
    # Implementation...
    pass
```

---

### **Step 11: Dependencies**

Add to `requirements.txt`:

```
# Bayesian inference (Pyro - PyTorch based)
pyro-ppl>=1.8.0

# MCMC diagnostics
arviz>=0.15.0

# Existing dependencies (already present)
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
scikit-survival>=0.22.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0  # Already present for DeepHit
```

**Alternative implementations (not used)**:
```
# NumPyro/JAX (faster but different ecosystem)
numpyro>=0.12.0
jax>=0.4.0

# PyMC (mature but slower)
pymc>=5.0.0
pytensor>=2.0.0
```

---

## Implementation Timeline

| Step | Description | Estimated Effort | Dependencies |
|------|-------------|------------------|--------------|
| 1 | Data preparation | 2-3 hours | Existing data pipeline |
| 2-4 | Model specification | 3-4 hours | Pyro/PyTorch |
| 5 | MCMC sampler | 4-6 hours | Core implementation |
| 6 | Posterior predictive | 3-4 hours | Step 5 |
| 7 | Performance evaluation | 3-4 hours | sksurv metrics |
| 8 | Model assessment | 2-3 hours | ArviZ |
| 9-10 | Notebook & module | 3-4 hours | Integration |
| **Total** | | **20-28 hours** | |

**Status**: ✅ Implemented using Pyro (PyTorch-based)

---

## Key Differences from Existing Models

| Aspect | Existing (Cox, RSF, DeepHit) | Bhattacharya Bayesian |
|--------|------------------------------|----------------------|
| **Inference** | Point estimates (MLE) | Full posterior distributions |
| **Uncertainty** | Bootstrap/asymptotic SE | Exact posterior credible intervals |
| **Baseline hazard** | Unspecified (Cox) or flexible | Parametric (lognormal) |
| **Predictions** | Point predictions | Predictive distributions |
| **Interpretability** | Hazard ratios | Posterior distributions of effects |
| **Prior information** | Not incorporated | Naturally incorporated |
| **Small samples** | Can be unstable | Regularization via priors |

---

## Expected Outputs

1. **Notebook**: `notebooks/09_bayesian_competing_risks.ipynb`
2. **Module**: `src/competing_risks/bayesian_phm.py`
3. **Evaluation**: `src/competing_risks/bayesian_evaluation.py`
4. **Saved artifacts**:
   - `models/bayesian_phm_posterior.nc` (NetCDF via ArviZ)
   - `models/bayesian_phm_summary.csv`
   - `reports/figures/bayesian_*.png`

---

## References

1. Bhattacharya, A., Wilson, S.P., & Soyer, R. (2019). A Bayesian approach to modeling mortgage default and prepayment. *European Journal of Operational Research*, 274, 1112-1124.

2. Deng, Y. (1997). Mortgage termination: An empirical hazard model with a stochastic term structure. *The Journal of Real Estate Finance and Economics*, 14(3), 309-331.

3. Cox, D.R. (1972). Regression models and life tables. *Journal of the Royal Statistical Society, Series B*, 34(2), 187-220.

4. Gelfand, A.E., & Mallick, B.K. (1995). Bayesian analysis of proportional hazards models built from monotone functions. *Biometrics*, 51(3), 843-852.
