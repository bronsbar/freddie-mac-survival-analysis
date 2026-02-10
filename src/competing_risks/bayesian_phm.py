"""
Bayesian Competing Risks Proportional Hazards Model

Implementation of Bhattacharya, Wilson & Soyer (2019):
"A Bayesian approach to modeling mortgage default and prepayment"
European Journal of Operational Research, 274, 1112-1124.

This module provides a scikit-learn compatible interface for the Bayesian
competing risks model with lognormal baseline hazards.

Uses Pyro (PyTorch-based probabilistic programming) for MCMC inference.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
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


def lognormal_log_hazard(t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute log of lognormal baseline hazard.

    log(r(t)) = log(phi(z)) - log(sigma) - log(t) - log(1 - Phi(z))
    where z = (log(t) - mu) / sigma

    Parameters
    ----------
    t : torch.Tensor
        Time points (must be positive)
    mu : torch.Tensor
        Location parameter of lognormal
    sigma : torch.Tensor
        Scale parameter of lognormal (must be positive)

    Returns
    -------
    torch.Tensor
        Log hazard values
    """
    z = (torch.log(t) - mu) / sigma
    # Log of standard normal PDF: -0.5*z^2 - 0.5*log(2*pi)
    log_phi = -0.5 * z**2 - 0.5 * torch.log(torch.tensor(2 * np.pi, device=t.device))
    # Standard normal CDF
    Phi_z = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    # Log survival of standard normal: log(1 - Phi(z))
    log_survival = torch.log(1 - Phi_z + 1e-10)
    return log_phi - torch.log(sigma) - torch.log(t) - log_survival


def lognormal_cumulative_hazard(t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute cumulative baseline hazard for lognormal.

    H_0(t) = -log(1 - Phi(z)) where z = (log(t) - mu) / sigma

    Parameters
    ----------
    t : torch.Tensor
        Time points
    mu : torch.Tensor
        Location parameter
    sigma : torch.Tensor
        Scale parameter

    Returns
    -------
    torch.Tensor
        Cumulative hazard values
    """
    z = (torch.log(t) - mu) / sigma
    # Standard normal CDF
    Phi_z = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    return -torch.log(1 - Phi_z + 1e-10)


class BayesianCompetingRisksPHM:
    """
    Bayesian competing risks proportional hazards model with lognormal
    baseline hazards.

    This implements the model from Bhattacharya et al. (2019) using
    Pyro for MCMC inference via the NUTS sampler.

    Parameters
    ----------
    num_warmup : int, default=1000
        Number of warmup (burn-in) samples for MCMC
    num_samples : int, default=2000
        Number of posterior samples to draw
    num_chains : int, default=4
        Number of MCMC chains to run
    target_accept_prob : float, default=0.8
        Target acceptance probability for NUTS
    random_seed : int, default=42
        Random seed for reproducibility
    prior_theta_sd : float, default=100.0
        Prior standard deviation for regression coefficients
    prior_mu_sd : float, default=10.0
        Prior standard deviation for baseline location parameters
    prior_sigma_rate : float, default=0.01
        Prior rate for exponential on baseline scale parameters
    device : str, default='cpu'
        Device to use for computation ('cpu', 'cuda', or 'mps')

    Attributes
    ----------
    posterior_samples_ : dict
        Dictionary of posterior samples after fitting
    mcmc_ : MCMC
        Pyro MCMC object
    inference_data_ : InferenceData
        ArviZ InferenceData object for diagnostics
    feature_names_ : list
        Names of features used in fitting

    Examples
    --------
    >>> model = BayesianCompetingRisksPHM(num_warmup=500, num_samples=1000)
    >>> model.fit(X_train, durations_train, events_train)
    >>> cif_mean, cif_lower, cif_upper = model.predict_cif(X_test, times, cause='default')

    References
    ----------
    Bhattacharya, A., Wilson, S.P., & Soyer, R. (2019). A Bayesian approach
    to modeling mortgage default and prepayment. European Journal of
    Operational Research, 274, 1112-1124.
    """

    def __init__(
        self,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        target_accept_prob: float = 0.8,
        random_seed: int = 42,
        prior_theta_sd: float = 100.0,
        prior_mu_sd: float = 10.0,
        prior_sigma_rate: float = 0.01,
        device: str = 'cpu',
    ):
        if not PYRO_AVAILABLE:
            raise ImportError(
                "Pyro is required. "
                "Install with: pip install pyro-ppl arviz"
            )

        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.target_accept_prob = target_accept_prob
        self.random_seed = random_seed
        self.prior_theta_sd = prior_theta_sd
        self.prior_mu_sd = prior_mu_sd
        self.prior_sigma_rate = prior_sigma_rate
        self.device = device

        self.posterior_samples_ = None
        self.mcmc_ = None
        self.inference_data_ = None
        self.feature_names_ = None
        self.n_features_ = None

    def _model(self, X, durations, events, n_features):
        """Pyro model specification."""
        N = X.shape[0]

        # Priors for baseline hazard parameters
        mu_D = pyro.sample('mu_D', dist.Normal(3.0, self.prior_mu_sd))
        sigma_D = pyro.sample('sigma_D', dist.Exponential(self.prior_sigma_rate))
        mu_P = pyro.sample('mu_P', dist.Normal(3.0, self.prior_mu_sd))
        sigma_P = pyro.sample('sigma_P', dist.Exponential(self.prior_sigma_rate))

        # Priors for regression coefficients
        theta_D = pyro.sample(
            'theta_D',
            dist.Normal(torch.zeros(n_features, device=X.device),
                       self.prior_theta_sd * torch.ones(n_features, device=X.device)).to_event(1)
        )
        theta_P = pyro.sample(
            'theta_P',
            dist.Normal(torch.zeros(n_features, device=X.device),
                       self.prior_theta_sd * torch.ones(n_features, device=X.device)).to_event(1)
        )

        # Linear predictors
        eta_D = torch.matmul(X, theta_D)
        eta_P = torch.matmul(X, theta_P)

        # Log-hazards at observed times
        log_h0_D = lognormal_log_hazard(durations, mu_D, sigma_D)
        log_h0_P = lognormal_log_hazard(durations, mu_P, sigma_P)
        log_h_D = log_h0_D + eta_D
        log_h_P = log_h0_P + eta_P

        # Cumulative hazards
        H0_D = lognormal_cumulative_hazard(durations, mu_D, sigma_D)
        H0_P = lognormal_cumulative_hazard(durations, mu_P, sigma_P)
        H_D = H0_D * torch.exp(eta_D)
        H_P = H0_P * torch.exp(eta_P)

        # Log-likelihood
        is_default = (events == 2).float()
        is_prepay = (events == 1).float()

        log_lik = (
            is_default * log_h_D +
            is_prepay * log_h_P -
            H_D - H_P
        )

        pyro.factor('log_likelihood', torch.sum(log_lik))

    def fit(
        self,
        X: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'BayesianCompetingRisksPHM':
        """
        Fit the Bayesian competing risks model using MCMC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix (should be standardized)
        durations : array-like of shape (n_samples,)
            Observed durations (time to event or censoring)
        events : array-like of shape (n_samples,)
            Event indicators: 0=censored, 1=prepay, 2=default
        feature_names : list of str, optional
            Names of features for interpretation

        Returns
        -------
        self : BayesianCompetingRisksPHM
            Fitted model
        """
        # Set random seed
        pyro.set_rng_seed(self.random_seed)
        pyro.clear_param_store()

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        durations = torch.tensor(durations, dtype=torch.float32, device=self.device)
        events = torch.tensor(events, dtype=torch.int64, device=self.device)

        # Ensure durations are positive
        durations = torch.clamp(durations, min=0.5)

        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names or [f'x{i}' for i in range(self.n_features_)]

        # Set up NUTS sampler
        nuts_kernel = NUTS(
            self._model,
            target_accept_prob=self.target_accept_prob,
            jit_compile=False,  # Disable JIT for compatibility
        )

        self.mcmc_ = MCMC(
            nuts_kernel,
            num_samples=self.num_samples,
            warmup_steps=self.num_warmup,
            num_chains=self.num_chains,
        )

        # Run MCMC
        self.mcmc_.run(X, durations, events, self.n_features_)

        # Store results
        self.posterior_samples_ = {k: v.cpu().numpy() for k, v in self.mcmc_.get_samples().items()}

        # Convert to ArviZ InferenceData
        self.inference_data_ = az.from_pyro(self.mcmc_)

        return self

    def predict_cif(
        self,
        X: np.ndarray,
        times: np.ndarray,
        cause: str = 'default',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict cumulative incidence function with credible intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix for new observations
        times : array-like of shape (n_times,)
            Times at which to evaluate CIF
        cause : str, {'default', 'prepay'}
            Which cause to compute CIF for

        Returns
        -------
        cif_mean : array of shape (n_samples, n_times)
            Posterior mean CIF
        cif_lower : array of shape (n_samples, n_times)
            2.5% quantile of CIF
        cif_upper : array of shape (n_samples, n_times)
            97.5% quantile of CIF
        """
        if self.posterior_samples_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        times = torch.tensor(times, dtype=torch.float32, device=self.device)
        N = X.shape[0]
        T = len(times)

        mu_D = torch.tensor(self.posterior_samples_['mu_D'], device=self.device)
        sigma_D = torch.tensor(self.posterior_samples_['sigma_D'], device=self.device)
        mu_P = torch.tensor(self.posterior_samples_['mu_P'], device=self.device)
        sigma_P = torch.tensor(self.posterior_samples_['sigma_P'], device=self.device)
        theta_D = torch.tensor(self.posterior_samples_['theta_D'], device=self.device)
        theta_P = torch.tensor(self.posterior_samples_['theta_P'], device=self.device)

        n_samples = len(mu_D)
        cif_samples = np.zeros((n_samples, N, T))

        for s in range(n_samples):
            eta_D = torch.matmul(X, theta_D[s])
            eta_P = torch.matmul(X, theta_P[s])

            for t_idx, t in enumerate(times):
                H0_D = lognormal_cumulative_hazard(t, mu_D[s], sigma_D[s])
                H0_P = lognormal_cumulative_hazard(t, mu_P[s], sigma_P[s])

                H_D = H0_D * torch.exp(eta_D)
                H_P = H0_P * torch.exp(eta_P)

                S_t = torch.exp(-H_D - H_P)
                total_H = H_D + H_P + 1e-10

                if cause == 'default':
                    cif_samples[s, :, t_idx] = ((H_D / total_H) * (1 - S_t)).cpu().numpy()
                else:
                    cif_samples[s, :, t_idx] = ((H_P / total_H) * (1 - S_t)).cpu().numpy()

        cif_mean = np.mean(cif_samples, axis=0)
        cif_lower = np.percentile(cif_samples, 2.5, axis=0)
        cif_upper = np.percentile(cif_samples, 97.5, axis=0)

        return cif_mean, cif_lower, cif_upper

    def predict_survival(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict overall survival function with credible intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix
        times : array-like of shape (n_times,)
            Times at which to evaluate survival

        Returns
        -------
        surv_mean : array of shape (n_samples, n_times)
            Posterior mean survival
        surv_lower : array of shape (n_samples, n_times)
            2.5% quantile
        surv_upper : array of shape (n_samples, n_times)
            97.5% quantile
        """
        if self.posterior_samples_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        times = torch.tensor(times, dtype=torch.float32, device=self.device)
        N = X.shape[0]
        T = len(times)

        mu_D = torch.tensor(self.posterior_samples_['mu_D'], device=self.device)
        sigma_D = torch.tensor(self.posterior_samples_['sigma_D'], device=self.device)
        mu_P = torch.tensor(self.posterior_samples_['mu_P'], device=self.device)
        sigma_P = torch.tensor(self.posterior_samples_['sigma_P'], device=self.device)
        theta_D = torch.tensor(self.posterior_samples_['theta_D'], device=self.device)
        theta_P = torch.tensor(self.posterior_samples_['theta_P'], device=self.device)

        n_samples = len(mu_D)
        surv_samples = np.zeros((n_samples, N, T))

        for s in range(n_samples):
            eta_D = torch.matmul(X, theta_D[s])
            eta_P = torch.matmul(X, theta_P[s])

            for t_idx, t in enumerate(times):
                H0_D = lognormal_cumulative_hazard(t, mu_D[s], sigma_D[s])
                H0_P = lognormal_cumulative_hazard(t, mu_P[s], sigma_P[s])

                H_D = H0_D * torch.exp(eta_D)
                H_P = H0_P * torch.exp(eta_P)

                surv_samples[s, :, t_idx] = torch.exp(-H_D - H_P).cpu().numpy()

        surv_mean = np.mean(surv_samples, axis=0)
        surv_lower = np.percentile(surv_samples, 2.5, axis=0)
        surv_upper = np.percentile(surv_samples, 97.5, axis=0)

        return surv_mean, surv_lower, surv_upper

    def get_posterior_summary(self) -> Dict:
        """
        Get summary statistics for all model parameters.

        Returns
        -------
        summary : dict
            Dictionary with parameter summaries including mean, median,
            std, and 95% credible intervals
        """
        if self.posterior_samples_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        summary = {}

        # Baseline parameters
        for param in ['mu_D', 'sigma_D', 'mu_P', 'sigma_P']:
            samples = np.array(self.posterior_samples_[param])
            summary[param] = {
                'mean': np.mean(samples),
                'median': np.median(samples),
                'std': np.std(samples),
                'ci_lower': np.percentile(samples, 2.5),
                'ci_upper': np.percentile(samples, 97.5),
            }

        # Regression coefficients
        for param, prefix in [('theta_D', 'default'), ('theta_P', 'prepay')]:
            samples = np.array(self.posterior_samples_[param])
            for i, feat in enumerate(self.feature_names_):
                key = f'{prefix}_{feat}'
                summary[key] = {
                    'mean': np.mean(samples[:, i]),
                    'median': np.median(samples[:, i]),
                    'std': np.std(samples[:, i]),
                    'ci_lower': np.percentile(samples[:, i], 2.5),
                    'ci_upper': np.percentile(samples[:, i], 97.5),
                }

        return summary

    def print_summary(self):
        """Print MCMC summary."""
        if self.mcmc_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        self.mcmc_.summary()

    def get_diagnostics(self) -> Dict:
        """
        Get MCMC convergence diagnostics.

        Returns
        -------
        diagnostics : dict
            Dictionary with R-hat and effective sample size
        """
        if self.inference_data_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        rhat = az.rhat(self.inference_data_)
        ess = az.ess(self.inference_data_)

        return {
            'rhat': {k: float(v.values) for k, v in rhat.items() if v.size == 1},
            'ess': {k: float(v.values) for k, v in ess.items() if v.size == 1},
        }
