"""
Joint Model for Longitudinal and Discrete Survival Data with Competing Risks.

Extension of Medina-Olivares, Calabrese, Crook & Lindgren (2022) to competing
risks (prepayment and default).  The model jointly estimates:

    1. A longitudinal submodel for an endogenous time-varying covariate (TVC)
       with random intercept + slope + AR(1) term.
    2. A discrete-time competing-risks survival submodel linked to the
       longitudinal submodel through the model-implied TVC trajectory.

Estimation is Bayesian via Pyro NUTS, following the same patterns as
``bayesian_phm.py``.

Reference
---------
Medina-Olivares, V., Calabrese, R., Crook, J. & Lindgren, F. (2022).
Joint models for longitudinal and discrete survival data in credit scoring.
*European Journal of Operational Research*, 307(3), 1457-1473.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.interpolate import BSpline
import warnings

try:
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS
    import arviz as az
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    warnings.warn(
        "Pyro not available. Install with: pip install pyro-ppl arviz"
    )


# ── Feature constants (shared with sadhwani_net.py) ──────────────────────────

STATIC_FEATURES = ['int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r']
TVC_DEFAULT = 'ppi_c_FRMA'


# =============================================================================
# B-spline baseline hazard
# =============================================================================

def _build_bspline_basis(T_max: int, n_interior_knots: int = 3,
                         degree: int = 3) -> np.ndarray:
    """
    Build a cubic B-spline design matrix for months 1..T_max.

    Knots are placed at equally spaced quantiles of 1..T_max.

    Parameters
    ----------
    T_max : int
        Maximum loan age (months).
    n_interior_knots : int
        Number of interior knots.
    degree : int
        B-spline degree (3 = cubic).

    Returns
    -------
    np.ndarray of shape (T_max, n_basis)
        Design matrix where row t is the basis evaluated at month t+1.
    """
    t_grid = np.arange(1, T_max + 1, dtype=float)

    # Interior knots at evenly-spaced quantiles
    quantiles = np.linspace(0, 100, n_interior_knots + 2)[1:-1]
    interior_knots = np.percentile(t_grid, quantiles)

    # Full knot vector with boundary knots repeated (degree+1) times
    lo, hi = t_grid[0], t_grid[-1]
    knots = np.concatenate([
        np.repeat(lo, degree + 1),
        interior_knots,
        np.repeat(hi, degree + 1),
    ])

    n_basis = len(knots) - degree - 1
    B = np.zeros((T_max, n_basis))
    for j in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[j] = 1.0
        spline = BSpline(knots, coeffs, degree, extrapolate=True)
        B[:, j] = spline(t_grid)

    return B


# =============================================================================
# Data preparation
# =============================================================================

def prepare_joint_model_data(
    panel_df: pd.DataFrame,
    tvc_col: str = TVC_DEFAULT,
    static_cols: Optional[List[str]] = None,
    n_interior_knots: int = 3,
) -> Dict:
    """
    Transform loan-month panel into tensors for the Pyro joint model.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Loan-month panel with columns: loan_sequence_number, loan_age,
        event_code, tvc_col, and the static feature columns.
    tvc_col : str
        Name of the time-varying covariate to model longitudinally.
    static_cols : list[str], optional
        Static covariate column names (default: STATIC_FEATURES).
    n_interior_knots : int
        Number of interior knots for B-spline baseline hazard.

    Returns
    -------
    dict
        Tensors and metadata needed by the Pyro model.
    """
    if static_cols is None:
        static_cols = [c for c in STATIC_FEATURES if c in panel_df.columns]

    df = panel_df.sort_values(['loan_sequence_number', 'loan_age']).copy()

    # Ensure log_upb exists
    if 'log_upb' not in df.columns and 'orig_upb' in df.columns:
        df['log_upb'] = np.log(df['orig_upb'].clip(lower=1).astype(float))

    # Drop rows with missing TVC or static features
    required = [tvc_col] + static_cols
    df = df.dropna(subset=required)

    # ── Loan-level survival data ────────────────────────────────────────────
    loan_summary = df.groupby('loan_sequence_number').agg(
        event_time=('loan_age', 'max'),
        event_type=('event_code', 'last'),
    )
    loan_ids_ordered = loan_summary.index.values
    N_loans = len(loan_ids_ordered)
    loan_id_map = {lid: idx for idx, lid in enumerate(loan_ids_ordered)}

    event_time = loan_summary['event_time'].values.astype(np.int64)
    event_type = loan_summary['event_type'].values.astype(np.int64)

    # ── Longitudinal data (all loan-month observations) ─────────────────────
    df['_loan_idx'] = df['loan_sequence_number'].map(loan_id_map)
    loan_id_arr = df['_loan_idx'].values.astype(np.int64)
    time_idx_arr = df['loan_age'].values.astype(np.float64)

    Y = df[tvc_col].values.astype(np.float64)

    # Lagged TVC (within-loan)
    df['_tvc_lag'] = df.groupby('loan_sequence_number')[tvc_col].shift(1)
    df['_tvc_lag'] = df['_tvc_lag'].fillna(0.0)
    Y_lag = df['_tvc_lag'].values.astype(np.float64)

    # ── Static covariates (one row per loan, standardised) ──────────────────
    static_per_loan = df.groupby('_loan_idx')[static_cols].first()
    static_per_loan = static_per_loan.sort_index()
    Z = static_per_loan.values.astype(np.float64)

    # Standardise
    z_mean = Z.mean(axis=0)
    z_std = Z.std(axis=0)
    z_std[z_std == 0] = 1.0
    Z = (Z - z_mean) / z_std

    # ── B-spline basis for baseline hazard ──────────────────────────────────
    T_max = int(event_time.max())
    B = _build_bspline_basis(T_max, n_interior_knots=n_interior_knots)
    n_basis = B.shape[1]

    # ── Pre-build per-loan observation index for survival likelihood ────────
    # For each loan i, we need the observed Y values at months 1..event_time[i]
    # to compute m_i(t) = alpha_0 + U_0i + U_1i*t + phi*Y_i(t-1).
    # We store a padded matrix Y_by_loan of shape (N_loans, T_max) with NaN
    # padding, plus a mask.
    Y_by_loan = np.full((N_loans, T_max), np.nan, dtype=np.float64)
    for idx, row in df.iterrows():
        li = int(row['_loan_idx'])
        t = int(row['loan_age']) - 1  # 0-indexed
        if t < T_max:
            Y_by_loan[li, t] = row[tvc_col]

    # Lagged Y by loan: Y_lag_by_loan[i, t] = Y_by_loan[i, t-1] (0 for t=0)
    Y_lag_by_loan = np.zeros_like(Y_by_loan)
    Y_lag_by_loan[:, 1:] = Y_by_loan[:, :-1]
    # Replace NaN with 0 for padding (masked out in likelihood)
    Y_lag_by_loan = np.nan_to_num(Y_lag_by_loan, nan=0.0)

    # Mask: True where loan i is observed at month t+1
    obs_mask = ~np.isnan(Y_by_loan)

    P = len(static_cols)

    return {
        # Longitudinal
        'Y': Y,
        'Y_lag': Y_lag,
        'loan_id': loan_id_arr,
        'time_idx': time_idx_arr,
        # Survival
        'event_time': event_time,
        'event_type': event_type,
        'Z': Z,
        'z_mean': z_mean,
        'z_std': z_std,
        'B_spline': B,
        # Per-loan matrices for survival likelihood
        'Y_by_loan': Y_by_loan,
        'Y_lag_by_loan': Y_lag_by_loan,
        'obs_mask': obs_mask,
        # Dimensions
        'N': len(Y),
        'N_loans': N_loans,
        'T_max': T_max,
        'n_basis': n_basis,
        'P': P,
        # Metadata
        'loan_ids': loan_ids_ordered,
        'static_cols': static_cols,
        'tvc_col': tvc_col,
    }


# =============================================================================
# Pyro model
# =============================================================================

def _joint_model(
    Y, Y_lag, loan_id, time_idx,
    Z, B_spline,
    Y_lag_by_loan, event_time, event_type,
    N_loans, T_max, n_basis, P,
):
    """
    Pyro model: joint longitudinal + discrete-time competing risks.

    Longitudinal submodel (M5):
        Y_i(t) = alpha_0 + U_0i + U_1i * t + phi * Y_i(t-1) + eps_i(t)

    Survival submodel (multinomial logit, competing risks):
        eta_k(t) = B(t) @ a_k + Z_i @ gamma_k + lambda_k * m_i(t-1)
        P(current)  = 1 / (1 + exp(eta_1) + exp(eta_2))
        P(prepay)   = exp(eta_1) / (1 + exp(eta_1) + exp(eta_2))
        P(default)  = exp(eta_2) / (1 + exp(eta_1) + exp(eta_2))
    """
    dtype = Y.dtype
    device = Y.device

    # ── Longitudinal parameters ─────────────────────────────────────────────
    alpha_0 = pyro.sample('alpha_0', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(5., dtype=dtype, device=device)))

    phi = pyro.sample('phi', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(0.5, dtype=dtype, device=device)))

    sigma = pyro.sample('sigma', dist.Exponential(
        torch.tensor(1., dtype=dtype, device=device)))

    # ── Random effects (non-centered parameterisation) ──────────────────────
    tau = pyro.sample('tau', dist.Exponential(
        torch.ones(2, dtype=dtype, device=device)).to_event(1))

    L_Omega = pyro.sample('L_Omega', dist.LKJCholesky(
        2, concentration=torch.tensor(2., dtype=dtype, device=device)))

    L_sigma = torch.mm(torch.diag(tau), L_Omega)  # (2, 2)

    # Standard normal draws per loan, then transform
    z_RE = pyro.sample('z_RE', dist.Normal(
        torch.zeros(N_loans, 2, dtype=dtype, device=device),
        torch.ones(N_loans, 2, dtype=dtype, device=device),
    ).to_event(2))

    b = torch.mm(z_RE, L_sigma.T)  # (N_loans, 2): col 0 = U_0i, col 1 = U_1i

    # ── Longitudinal likelihood ─────────────────────────────────────────────
    mu = alpha_0 + b[loan_id, 0] + b[loan_id, 1] * time_idx + phi * Y_lag
    pyro.sample('Y_obs', dist.Normal(mu, sigma).to_event(1), obs=Y)

    # ── Survival parameters (cause-specific) ────────────────────────────────
    a_prepay = pyro.sample('a_prepay', dist.Normal(
        torch.zeros(n_basis, dtype=dtype, device=device),
        2. * torch.ones(n_basis, dtype=dtype, device=device),
    ).to_event(1))

    a_default = pyro.sample('a_default', dist.Normal(
        torch.zeros(n_basis, dtype=dtype, device=device),
        2. * torch.ones(n_basis, dtype=dtype, device=device),
    ).to_event(1))

    gamma_prepay = pyro.sample('gamma_prepay', dist.Normal(
        torch.zeros(P, dtype=dtype, device=device),
        torch.ones(P, dtype=dtype, device=device),
    ).to_event(1))

    gamma_default = pyro.sample('gamma_default', dist.Normal(
        torch.zeros(P, dtype=dtype, device=device),
        torch.ones(P, dtype=dtype, device=device),
    ).to_event(1))

    lambda_prepay = pyro.sample('lambda_prepay', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(1., dtype=dtype, device=device)))

    lambda_default = pyro.sample('lambda_default', dist.Normal(
        torch.tensor(0., dtype=dtype, device=device),
        torch.tensor(1., dtype=dtype, device=device)))

    # ── Survival likelihood (vectorised over loans x months) ────────────────
    # B_spline:      (T_max, n_basis)
    # Y_lag_by_loan: (N_loans, T_max)  — lagged TVC per loan per month
    # event_time:    (N_loans,)        — last observed month
    # event_type:    (N_loans,)        — 0=cens, 1=prepay, 2=default

    # Baseline hazard at every month: (T_max, n_basis) @ (n_basis,) -> (T_max,)
    h0_prepay = torch.mv(B_spline, a_prepay)   # (T_max,)
    h0_default = torch.mv(B_spline, a_default)  # (T_max,)

    # Static covariate contributions: (N_loans, P) @ (P,) -> (N_loans,)
    Zg_prepay = torch.mv(Z, gamma_prepay)    # (N_loans,)
    Zg_default = torch.mv(Z, gamma_default)  # (N_loans,)

    # Model-implied TVC: m_i(t) = alpha_0 + U_0i + U_1i*t + phi*Y_i(t-1)
    # Time grid: (T_max,)
    t_grid = torch.arange(1, T_max + 1, dtype=dtype, device=device)

    # m_i(t) for all loans and months: (N_loans, T_max)
    # alpha_0 + U_0i: (N_loans, 1)
    # U_1i * t: (N_loans, 1) * (1, T_max) = (N_loans, T_max)
    # phi * Y_lag_by_loan: (N_loans, T_max)
    m = (alpha_0 + b[:, 0:1]
         + b[:, 1:2] * t_grid.unsqueeze(0)
         + phi * Y_lag_by_loan)  # (N_loans, T_max)

    # Linear predictors: eta_k(i, t) = h0_k(t) + Zg_k(i) + lambda_k * m(i, t)
    # h0_k(t): (T_max,) -> broadcast (1, T_max)
    # Zg_k(i): (N_loans,) -> broadcast (N_loans, 1)
    eta_prepay = (h0_prepay.unsqueeze(0)
                  + Zg_prepay.unsqueeze(1)
                  + lambda_prepay * m)   # (N_loans, T_max)

    eta_default = (h0_default.unsqueeze(0)
                   + Zg_default.unsqueeze(1)
                   + lambda_default * m)  # (N_loans, T_max)

    # Clamp to prevent exp() overflow during NUTS warmup exploration
    # (same pattern as bayesian_phm.py)
    eta_prepay = torch.clamp(eta_prepay, -20, 20)
    eta_default = torch.clamp(eta_default, -20, 20)

    # Log-softmax for multinomial logit (numerically stable via logsumexp):
    # logits = [0, eta_prepay, eta_default]  (0 for the "current" baseline)
    # log P(k) = logit_k - logsumexp(logits)
    #
    # Stable logsumexp: M = max(0, eta_1, eta_2)
    #   log(exp(0) + exp(eta_1) + exp(eta_2))
    #     = M + log(exp(-M) + exp(eta_1 - M) + exp(eta_2 - M))
    zero = torch.zeros_like(eta_prepay)
    M = torch.max(torch.max(zero, eta_prepay), eta_default)
    log_denom = M + torch.log(
        torch.exp(-M) + torch.exp(eta_prepay - M) + torch.exp(eta_default - M))
    log_p_current = -log_denom              # (N_loans, T_max)
    log_p_prepay = eta_prepay - log_denom   # (N_loans, T_max)
    log_p_default = eta_default - log_denom  # (N_loans, T_max)

    # Build month mask: month_mask[i, t] = 1 if t+1 <= event_time[i]
    # (i.e., loan i is still in the risk set at month t+1)
    month_idx = torch.arange(T_max, dtype=torch.int64, device=device).unsqueeze(0)
    event_time_expanded = event_time.unsqueeze(1)  # (N_loans, 1)

    # Months before the event: contribute log P(current)
    before_event = (month_idx < (event_time_expanded - 1))  # (N_loans, T_max)

    # Survival contribution: sum log P(current) for all months before event
    log_lik_surv = (log_p_current * before_event.to(dtype)).sum()

    # Event contribution at the terminal month
    is_prepay = (event_type == 1).to(dtype)        # (N_loans,)
    is_default = (event_type == 2).to(dtype)       # (N_loans,)
    is_censored = (event_type == 0).to(dtype)      # (N_loans,)

    # Extract log-prob at the event month for each loan
    event_month_idx = (event_time - 1).clamp(min=0)  # 0-indexed
    log_p_prepay_event = log_p_prepay[
        torch.arange(N_loans, device=device), event_month_idx]
    log_p_default_event = log_p_default[
        torch.arange(N_loans, device=device), event_month_idx]
    log_p_current_event = log_p_current[
        torch.arange(N_loans, device=device), event_month_idx]

    log_lik_event = (
        is_prepay * log_p_prepay_event
        + is_default * log_p_default_event
        + is_censored * log_p_current_event
    ).sum()

    # Guard against NaN from extreme proposals — NUTS will reject these
    total_log_lik = log_lik_surv + log_lik_event
    total_log_lik = torch.where(
        torch.isfinite(total_log_lik), total_log_lik,
        torch.tensor(-1e10, dtype=dtype, device=device))

    pyro.factor('survival_log_lik', total_log_lik)


# =============================================================================
# Model wrapper
# =============================================================================

class JointCompetingRisksModel:
    """
    Joint model for longitudinal and discrete survival data,
    extended to competing risks (prepayment and default).

    Uses Pyro NUTS for Bayesian estimation, following the same patterns
    as BayesianCompetingRisksPHM.

    Parameters
    ----------
    tvc_col : str
        Time-varying covariate column to model longitudinally.
    static_cols : list[str], optional
        Static covariate columns.
    n_interior_knots : int
        Number of interior knots for B-spline baseline hazard.
    num_warmup : int
        MCMC warmup iterations per chain.
    num_samples : int
        MCMC sampling iterations per chain.
    num_chains : int
        Number of MCMC chains.
    target_accept_prob : float
        Target acceptance probability for NUTS.
    random_seed : int
        Random seed for reproducibility.
    device : str
        PyTorch device ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        tvc_col: str = TVC_DEFAULT,
        static_cols: Optional[List[str]] = None,
        n_interior_knots: int = 3,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        target_accept_prob: float = 0.90,
        random_seed: int = 42,
        device: str = 'cpu',
    ):
        if not PYRO_AVAILABLE:
            raise ImportError(
                "Pyro is required. "
                "Install with: pip install pyro-ppl arviz"
            )

        self.tvc_col = tvc_col
        self.static_cols = static_cols or STATIC_FEATURES
        self.n_interior_knots = n_interior_knots
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept_prob = target_accept_prob
        self.random_seed = random_seed
        self.device = device

        self.posterior_samples_ = None
        self.mcmc_ = None
        self.inference_data_ = None
        self.data_ = None

    def fit(
        self,
        panel_df: pd.DataFrame,
    ) -> 'JointCompetingRisksModel':
        """
        Prepare data and run MCMC.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Loan-month panel (training data).

        Returns
        -------
        self
        """
        pyro.set_rng_seed(self.random_seed)
        pyro.clear_param_store()

        # Prepare data
        data = prepare_joint_model_data(
            panel_df,
            tvc_col=self.tvc_col,
            static_cols=self.static_cols,
            n_interior_knots=self.n_interior_knots,
        )
        self.data_ = data

        # Convert to float64 tensors
        dev = self.device
        dtype = torch.float64

        Y = torch.tensor(data['Y'], dtype=dtype, device=dev)
        Y_lag = torch.tensor(data['Y_lag'], dtype=dtype, device=dev)
        loan_id = torch.tensor(data['loan_id'], dtype=torch.int64, device=dev)
        time_idx = torch.tensor(data['time_idx'], dtype=dtype, device=dev)
        Z = torch.tensor(data['Z'], dtype=dtype, device=dev)
        B_spline = torch.tensor(data['B_spline'], dtype=dtype, device=dev)
        Y_lag_by_loan = torch.tensor(
            data['Y_lag_by_loan'], dtype=dtype, device=dev)
        event_time = torch.tensor(
            data['event_time'], dtype=torch.int64, device=dev)
        event_type = torch.tensor(
            data['event_type'], dtype=torch.int64, device=dev)

        N_loans = data['N_loans']
        T_max = data['T_max']
        n_basis = data['n_basis']
        P = data['P']

        # Set up NUTS sampler
        nuts_kernel = NUTS(
            _joint_model,
            target_accept_prob=self.target_accept_prob,
            jit_compile=False,
        )

        self.mcmc_ = MCMC(
            nuts_kernel,
            num_samples=self.num_samples,
            warmup_steps=self.num_warmup,
            num_chains=self.num_chains,
        )

        # Run MCMC
        self.mcmc_.run(
            Y, Y_lag, loan_id, time_idx,
            Z, B_spline,
            Y_lag_by_loan, event_time, event_type,
            N_loans, T_max, n_basis, P,
        )

        # Store results
        self.posterior_samples_ = {
            k: v.cpu().numpy()
            for k, v in self.mcmc_.get_samples().items()
        }

        # Convert to ArviZ InferenceData (exclude large random effects)
        try:
            self.inference_data_ = az.from_pyro(self.mcmc_)
        except Exception:
            self.inference_data_ = None
            warnings.warn("Could not create ArviZ InferenceData.")

        return self

    def predict_cif(
        self,
        panel_df: pd.DataFrame,
        horizons: List[int] = None,
        n_posterior_samples: int = 200,
    ) -> Dict[str, np.ndarray]:
        """
        Compute predictive CIF for loans in panel_df.

        For each loan:
        1. Estimate random effects from observed TVC history
        2. Project TVC forward using the longitudinal submodel
        3. Chain monthly transition probabilities into CIF

        Parameters
        ----------
        panel_df : pd.DataFrame
            Panel data for prediction loans.
        horizons : list[int]
            Months at which to evaluate CIF.
        n_posterior_samples : int
            Number of posterior draws to average over.

        Returns
        -------
        dict with keys:
            'loan_sequence_number': array of loan IDs
            'duration': observed event/censor times
            'event_code': observed event codes
            'cif_prepay_{t}': CIF for prepayment at each horizon t
            'cif_default_{t}': CIF for default at each horizon t
        """
        if self.posterior_samples_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if horizons is None:
            horizons = [24, 48, 72]

        max_horizon = max(horizons)
        data = self.data_

        # Prepare prediction data
        pred_data = prepare_joint_model_data(
            panel_df,
            tvc_col=self.tvc_col,
            static_cols=self.static_cols,
            n_interior_knots=self.n_interior_knots,
        )

        N_loans = pred_data['N_loans']
        T_max_pred = max_horizon
        P = pred_data['P']

        # Use training-set standardisation for Z
        Z_raw = (pred_data['Z'] * pred_data['z_std'] + pred_data['z_mean'])
        Z_pred = (Z_raw - data['z_mean']) / data['z_std']

        # B-spline basis for prediction horizon
        B_pred = _build_bspline_basis(T_max_pred, self.n_interior_knots)
        n_basis = B_pred.shape[1]

        # Thin posterior to n_posterior_samples
        total_samples = len(self.posterior_samples_['alpha_0'])
        if total_samples > n_posterior_samples:
            idx = np.linspace(0, total_samples - 1,
                              n_posterior_samples, dtype=int)
        else:
            idx = np.arange(total_samples)

        # Get initial TVC values per loan (last observed values)
        Y_init = np.zeros(N_loans, dtype=np.float64)
        for i in range(N_loans):
            mask = pred_data['obs_mask'][i]
            if mask.any():
                last_t = np.where(mask)[0][-1]
                Y_init[i] = pred_data['Y_by_loan'][i, last_t]

        # Accumulate CIF across posterior samples
        cif_prepay_acc = {t: np.zeros(N_loans) for t in horizons}
        cif_default_acc = {t: np.zeros(N_loans) for t in horizons}

        for s_idx in idx:
            alpha_0 = self.posterior_samples_['alpha_0'][s_idx]
            phi = self.posterior_samples_['phi'][s_idx]
            sigma = self.posterior_samples_['sigma'][s_idx]

            a_prep = self.posterior_samples_['a_prepay'][s_idx]  # (n_basis,)
            a_def = self.posterior_samples_['a_default'][s_idx]
            g_prep = self.posterior_samples_['gamma_prepay'][s_idx]  # (P,)
            g_def = self.posterior_samples_['gamma_default'][s_idx]
            lam_prep = self.posterior_samples_['lambda_prepay'][s_idx]
            lam_def = self.posterior_samples_['lambda_default'][s_idx]

            # Estimate random effects for prediction loans via empirical Bayes
            # Simplified: fit U_0i, U_1i from observed TVC trajectory per loan
            b_pred = _estimate_random_effects(
                pred_data, alpha_0, phi, sigma, s_idx, self.posterior_samples_)

            # Static covariate contribution
            Zg_prep = Z_pred @ g_prep  # (N_loans,)
            Zg_def = Z_pred @ g_def

            # Forward simulation of TVC and CIF chaining
            surv = np.ones(N_loans)
            cif_p = np.zeros(N_loans)
            cif_d = np.zeros(N_loans)
            y_prev = Y_init.copy()

            rng = np.random.RandomState(self.random_seed + s_idx)

            for t in range(max_horizon):
                month = t + 1  # 1-indexed

                # Model-implied TVC: m_i(t) = alpha_0 + U_0i + U_1i*t + phi*y_prev
                m_t = alpha_0 + b_pred[:, 0] + b_pred[:, 1] * month + phi * y_prev

                # Baseline hazard at this month
                if month <= B_pred.shape[0]:
                    b_row = B_pred[month - 1]  # (n_basis,)
                else:
                    b_row = B_pred[-1]

                h0_p = b_row @ a_prep
                h0_d = b_row @ a_def

                # Linear predictors
                eta_p = h0_p + Zg_prep + lam_prep * m_t
                eta_d = h0_d + Zg_def + lam_def * m_t

                # Multinomial logit probabilities
                exp_p = np.exp(np.clip(eta_p, -20, 20))
                exp_d = np.exp(np.clip(eta_d, -20, 20))
                denom = 1.0 + exp_p + exp_d
                p_prep = exp_p / denom
                p_def = exp_d / denom

                # Update CIF
                cif_p += surv * p_prep
                cif_d += surv * p_def
                surv *= (1.0 - p_prep - p_def)

                if month in horizons:
                    cif_prepay_acc[month] += cif_p
                    cif_default_acc[month] += cif_d

                # Simulate next TVC value (for next iteration)
                noise = rng.normal(0, sigma, size=N_loans)
                y_prev = m_t + noise

        # Average over posterior samples
        n_samples = len(idx)
        result = {
            'loan_sequence_number': pred_data['loan_ids'],
            'duration': pred_data['event_time'],
            'event_code': pred_data['event_type'],
        }
        for t in horizons:
            result[f'cif_prepay_{t}'] = cif_prepay_acc[t] / n_samples
            result[f'cif_default_{t}'] = cif_default_acc[t] / n_samples

        return result

    def get_posterior_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for model parameters (excluding random effects).

        Returns
        -------
        pd.DataFrame
            Summary with columns: mean, std, 5%, 95%.
        """
        if self.posterior_samples_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        rows = []
        scalar_params = ['alpha_0', 'phi', 'sigma',
                         'lambda_prepay', 'lambda_default']

        for param in scalar_params:
            if param in self.posterior_samples_:
                samples = self.posterior_samples_[param]
                rows.append({
                    'parameter': param,
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    '5%': np.percentile(samples, 5),
                    '95%': np.percentile(samples, 95),
                })

        # Random effect SDs
        if 'tau' in self.posterior_samples_:
            tau = self.posterior_samples_['tau']
            for j, name in enumerate(['tau_intercept', 'tau_slope']):
                rows.append({
                    'parameter': name,
                    'mean': np.mean(tau[:, j]),
                    'std': np.std(tau[:, j]),
                    '5%': np.percentile(tau[:, j], 5),
                    '95%': np.percentile(tau[:, j], 95),
                })

        # Covariate effects
        static_cols = self.data_['static_cols'] if self.data_ else STATIC_FEATURES
        for param_name, prefix in [('gamma_prepay', 'prepay'),
                                   ('gamma_default', 'default')]:
            if param_name in self.posterior_samples_:
                samples = self.posterior_samples_[param_name]
                for j, feat in enumerate(static_cols):
                    rows.append({
                        'parameter': f'{prefix}_{feat}',
                        'mean': np.mean(samples[:, j]),
                        'std': np.std(samples[:, j]),
                        '5%': np.percentile(samples[:, j], 5),
                        '95%': np.percentile(samples[:, j], 95),
                    })

        # Baseline hazard coefficients
        for param_name in ['a_prepay', 'a_default']:
            if param_name in self.posterior_samples_:
                samples = self.posterior_samples_[param_name]
                for j in range(samples.shape[1]):
                    rows.append({
                        'parameter': f'{param_name}[{j}]',
                        'mean': np.mean(samples[:, j]),
                        'std': np.std(samples[:, j]),
                        '5%': np.percentile(samples[:, j], 5),
                        '95%': np.percentile(samples[:, j], 95),
                    })

        return pd.DataFrame(rows)

    def get_diagnostics(self) -> Dict:
        """
        Get MCMC convergence diagnostics.

        Returns
        -------
        dict with 'rhat' and 'ess' entries.
        """
        if self.inference_data_ is None:
            raise ValueError("No ArviZ inference data available.")

        return {
            'rhat': az.rhat(self.inference_data_),
            'ess': az.ess(self.inference_data_),
        }

    def print_summary(self):
        """Print MCMC summary."""
        if self.mcmc_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        self.mcmc_.summary()


# =============================================================================
# Empirical Bayes random effects estimation
# =============================================================================

def _estimate_random_effects(
    pred_data: Dict,
    alpha_0: float,
    phi: float,
    sigma: float,
    sample_idx: int,
    posterior_samples: Dict,
) -> np.ndarray:
    """
    Estimate per-loan random effects (U_0i, U_1i) from observed TVC history.

    Uses a simple OLS approach: for each loan, regress
        Y_i(t) - alpha_0 - phi * Y_i(t-1)  on  [1, t]
    to get (U_0i, U_1i).

    Parameters
    ----------
    pred_data : dict
        From prepare_joint_model_data.
    alpha_0, phi, sigma : float
        Longitudinal model parameters.
    sample_idx : int
        Index into posterior samples (for tau/L_Omega if using MAP).
    posterior_samples : dict
        Full posterior samples.

    Returns
    -------
    np.ndarray of shape (N_loans, 2)
        Estimated [U_0i, U_1i] per loan.
    """
    N_loans = pred_data['N_loans']
    Y_by_loan = pred_data['Y_by_loan']
    Y_lag_by_loan = pred_data['Y_lag_by_loan']
    obs_mask = pred_data['obs_mask']
    T_max = pred_data['T_max']

    # Prior precision from tau
    tau = posterior_samples['tau'][sample_idx]  # (2,)
    prior_prec = np.diag(1.0 / (tau**2 + 1e-10))

    b = np.zeros((N_loans, 2))
    t_grid = np.arange(1, T_max + 1, dtype=np.float64)

    for i in range(N_loans):
        mask = obs_mask[i]
        if mask.sum() < 2:
            # Not enough observations — use prior mean (0, 0)
            continue

        y_obs = Y_by_loan[i, mask]
        y_lag = Y_lag_by_loan[i, mask]
        t_obs = t_grid[mask]

        # Residual after removing fixed effects and AR term
        resid = y_obs - alpha_0 - phi * y_lag

        # Design matrix: [1, t]
        X = np.column_stack([np.ones(len(t_obs)), t_obs])

        # Ridge regression with prior precision (empirical Bayes MAP)
        data_prec = (1.0 / (sigma**2 + 1e-10)) * X.T @ X
        combined_prec = data_prec + prior_prec
        data_target = (1.0 / (sigma**2 + 1e-10)) * X.T @ resid

        try:
            b[i] = np.linalg.solve(combined_prec, data_target)
        except np.linalg.LinAlgError:
            b[i] = 0.0

    return b
