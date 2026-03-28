#!/usr/bin/env python3
"""
Bayesian Competing Risks PHM - Supercomputer Script

Implements Bhattacharya, Wilson & Soyer (2019) Bayesian competing risks
proportional hazards model for mortgage default and prepayment.

Usage:
    python run_bayesian_phm.py [--num-chains 50] [--num-samples 15000] [--num-warmup 60000] [--thinning 50]

Output:
    - results/bayesian_phm_results.txt (summary report)
    - models/bayesian_phm_posterior.npz (posterior samples)
    - models/bayesian_phm_inference.nc (ArviZ inference data)
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Bayesian Competing Risks PHM')
    parser.add_argument('--num-chains', type=int, default=50, help='Number of MCMC chains')
    parser.add_argument('--num-samples', type=int, default=15000, help='Number of post-warmup samples per chain')
    parser.add_argument('--num-warmup', type=int, default=60000, help='Number of warmup (burn-in) steps')
    parser.add_argument('--thinning', type=int, default=50, help='Keep every Nth sample')
    parser.add_argument('--target-accept', type=float, default=0.8, help='Target acceptance probability')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    return parser.parse_args()


# Prior hyperparameters (Bhattacharya et al. 2019)
PRIOR_PARAMS = {
    'theta_sd': 100.0,
    'mu_sd': 10.0,
    'sigma_rate': 0.01,
}

# Cross-validation folds (Blumenstock methodology)
TRAIN_FOLDS = list(range(9))
VAL_FOLDS = [9]
TEST_FOLD = 10

# Features
STATIC_FEATURES = ['int_rate', 'orig_upb', 'fico_score', 'dti_r', 'ltv_r']
BEHAVIORAL_FEATURES = ['bal_repaid', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m']
MACRO_FEATURES = ['hpi_st_d_t_o', 'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o']
ALL_FEATURES = STATIC_FEATURES + BEHAVIORAL_FEATURES + MACRO_FEATURES

# Evaluation time horizons
TIME_HORIZONS = [24, 48, 72]


# ==============================================================================
# Model Functions
# ==============================================================================

_LOG_2PI = math.log(2 * math.pi)
_SQRT2 = math.sqrt(2)


def _log_standard_normal_survival(z):
    """
    log(1 - Phi(z)) computed via erfc for numerical stability.

    1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
    log(1 - Phi(z)) = log(0.5) + log(erfc(z / sqrt(2)))

    erfc is numerically stable for large z (unlike 1 - erf),
    and with float64 handles |z| up to ~26 before underflow.
    """
    return torch.log(torch.tensor(0.5, dtype=z.dtype, device=z.device)) + \
           torch.log(torch.erfc(z / _SQRT2).clamp(min=1e-30))


def lognormal_log_hazard(t, mu, sigma):
    """Log hazard for lognormal baseline (numerically stable)."""
    z = (torch.log(t) - mu) / sigma
    z = torch.clamp(z, -25, 25)
    log_phi = -0.5 * z**2 - 0.5 * _LOG_2PI
    log_surv = _log_standard_normal_survival(z)
    return log_phi - torch.log(sigma) - torch.log(t) - log_surv


def lognormal_cumulative_hazard(t, mu, sigma):
    """Cumulative hazard for lognormal baseline (numerically stable)."""
    z = (torch.log(t) - mu) / sigma
    z = torch.clamp(z, -25, 25)
    return -_log_standard_normal_survival(z)


def bayesian_competing_risks_model(X, durations, events, n_features):
    """Pyro model for Bayesian competing risks PHM (float64-safe)."""
    device = X.device
    dtype = X.dtype  # Inherit dtype from input (float64 for stability)

    # Baseline hazard parameters
    mu_D = pyro.sample('mu_D', dist.Normal(
        torch.tensor(3.0, dtype=dtype, device=device),
        torch.tensor(PRIOR_PARAMS['mu_sd'], dtype=dtype, device=device)))
    sigma_D = pyro.sample('sigma_D', dist.Exponential(
        torch.tensor(PRIOR_PARAMS['sigma_rate'], dtype=dtype, device=device)))
    mu_P = pyro.sample('mu_P', dist.Normal(
        torch.tensor(3.0, dtype=dtype, device=device),
        torch.tensor(PRIOR_PARAMS['mu_sd'], dtype=dtype, device=device)))
    sigma_P = pyro.sample('sigma_P', dist.Exponential(
        torch.tensor(PRIOR_PARAMS['sigma_rate'], dtype=dtype, device=device)))

    # Regression coefficients
    theta_D = pyro.sample('theta_D', dist.Normal(
        torch.zeros(n_features, dtype=dtype, device=device),
        PRIOR_PARAMS['theta_sd'] * torch.ones(n_features, dtype=dtype, device=device)).to_event(1))
    theta_P = pyro.sample('theta_P', dist.Normal(
        torch.zeros(n_features, dtype=dtype, device=device),
        PRIOR_PARAMS['theta_sd'] * torch.ones(n_features, dtype=dtype, device=device)).to_event(1))

    # Linear predictors
    eta_D = torch.matmul(X, theta_D)
    eta_P = torch.matmul(X, theta_P)

    # Hazards and cumulative hazards
    log_h_D = lognormal_log_hazard(durations, mu_D, sigma_D) + eta_D
    log_h_P = lognormal_log_hazard(durations, mu_P, sigma_P) + eta_P
    H_D = lognormal_cumulative_hazard(durations, mu_D, sigma_D) * torch.exp(eta_D)
    H_P = lognormal_cumulative_hazard(durations, mu_P, sigma_P) * torch.exp(eta_P)

    # Log-likelihood (competing risks)
    log_lik = (events == 2).to(dtype) * log_h_D + (events == 1).to(dtype) * log_h_P - H_D - H_P
    pyro.factor('log_likelihood', torch.sum(log_lik))


# ==============================================================================
# Model Wrapper for Predictions
# ==============================================================================

class BayesianModelWrapper:
    """Wrapper for posterior predictions."""

    def __init__(self, posterior_samples, device='cpu'):
        self.posterior_samples_ = posterior_samples
        self.device = device

    def predict_cif(self, X, times, cause='default'):
        """Predict cumulative incidence function."""
        X = torch.tensor(X, dtype=torch.float64, device=self.device)
        times = torch.tensor(times, dtype=torch.float64, device=self.device)
        N, T = X.shape[0], len(times)

        ps = {k: torch.tensor(v, device=self.device) for k, v in self.posterior_samples_.items()}
        n_samples = len(ps['mu_D'])
        cif_samples = np.zeros((n_samples, N, T))

        for s in range(n_samples):
            eta_D = torch.matmul(X, ps['theta_D'][s])
            eta_P = torch.matmul(X, ps['theta_P'][s])
            for t_idx, t in enumerate(times):
                H_D = lognormal_cumulative_hazard(t, ps['mu_D'][s], ps['sigma_D'][s]) * torch.exp(eta_D)
                H_P = lognormal_cumulative_hazard(t, ps['mu_P'][s], ps['sigma_P'][s]) * torch.exp(eta_P)
                S_t = torch.exp(-H_D - H_P)
                cif = (H_D if cause == 'default' else H_P) / (H_D + H_P + 1e-10) * (1 - S_t)
                cif_samples[s, :, t_idx] = cif.cpu().numpy()

        return np.mean(cif_samples, 0), np.percentile(cif_samples, 2.5, 0), np.percentile(cif_samples, 97.5, 0)

    def predict_survival(self, X, times):
        """Predict survival function."""
        X = torch.tensor(X, dtype=torch.float64, device=self.device)
        times = torch.tensor(times, dtype=torch.float64, device=self.device)
        N, T = X.shape[0], len(times)

        ps = {k: torch.tensor(v, device=self.device) for k, v in self.posterior_samples_.items()}
        n_samples = len(ps['mu_D'])
        surv_samples = np.zeros((n_samples, N, T))

        for s in range(n_samples):
            eta_D = torch.matmul(X, ps['theta_D'][s])
            eta_P = torch.matmul(X, ps['theta_P'][s])
            for t_idx, t in enumerate(times):
                H_D = lognormal_cumulative_hazard(t, ps['mu_D'][s], ps['sigma_D'][s]) * torch.exp(eta_D)
                H_P = lognormal_cumulative_hazard(t, ps['mu_P'][s], ps['sigma_P'][s]) * torch.exp(eta_P)
                surv_samples[s, :, t_idx] = torch.exp(-H_D - H_P).cpu().numpy()

        return np.mean(surv_samples, 0), np.percentile(surv_samples, 2.5, 0), np.percentile(surv_samples, 97.5, 0)


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def compute_time_dependent_cindex(model, X_train, durations_train, events_train,
                                   X_test, durations_test, events_test,
                                   times, cause='default'):
    """Compute IPCW C-index at specified time horizons."""
    cause_code = 2 if cause == 'default' else 1

    # Create survival objects
    train_event = (events_train == cause_code)
    test_event = (events_test == cause_code)

    y_train = Surv.from_arrays(train_event, durations_train)
    y_test = Surv.from_arrays(test_event, durations_test)

    # Get risk scores (use CIF at max time as risk)
    cif_mean, _, _ = model.predict_cif(X_test, np.array([max(times)]), cause)
    risk_scores = cif_mean[:, 0]

    results = {}
    for tau in times:
        try:
            c_index, _, _, _, _ = concordance_index_ipcw(
                y_train, y_test, risk_scores, tau=tau
            )
            results[tau] = c_index
        except Exception as e:
            print(f"  Warning: C-index at τ={tau} failed: {e}")
            results[tau] = np.nan

    return results


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    args = parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    models_dir = base_dir / args.models_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'bayesian_phm_results.txt'

    # Start logging
    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("BAYESIAN COMPETING RISKS PHM")
    log("Bhattacharya, Wilson & Soyer (2019)")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"Pyro version: {pyro.__version__}")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    log(f"\nConfiguration (Bhattacharya et al. 2019):")
    log(f"  num_chains: {args.num_chains}")
    log(f"  num_samples: {args.num_samples}")
    log(f"  num_warmup: {args.num_warmup}")
    log(f"  thinning: {args.thinning}")
    log(f"  target_accept: {args.target_accept}")
    log(f"  seed: {args.seed}")
    log(f"  effective samples per chain: {args.num_samples // args.thinning}")
    log(f"  total effective samples: {args.num_samples // args.thinning * args.num_chains}")

    # Load data
    log("\n" + "-" * 70)
    log("Loading data...")
    panel_df = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    log(f"  Loaded {len(panel_df):,} loan-months")
    log(f"  Unique loans: {panel_df['loan_sequence_number'].nunique():,}")

    # Prepare features
    feature_cols = [f for f in ALL_FEATURES if f in panel_df.columns]
    log(f"  Available features: {len(feature_cols)}/{len(ALL_FEATURES)}")

    # Terminal observations
    time_col, event_col = 'loan_age', 'event_code'
    panel_df = panel_df.sort_values(['loan_sequence_number', time_col])
    terminal_df = panel_df.groupby('loan_sequence_number').last().reset_index()

    # Feature engineering
    if 'bal_repaid' in feature_cols:
        bal_repaid_lag = panel_df.groupby('loan_sequence_number').apply(
            lambda g: g['bal_repaid'].iloc[-2] if len(g) >= 2 else g['bal_repaid'].iloc[-1])
        terminal_df['bal_repaid_lag1'] = terminal_df['loan_sequence_number'].map(bal_repaid_lag)
        feature_cols = [f if f != 'bal_repaid' else 'bal_repaid_lag1' for f in feature_cols]

    if 'orig_upb' in terminal_df.columns:
        terminal_df['log_upb'] = np.log(terminal_df['orig_upb'])
        feature_cols = [f if f != 'orig_upb' else 'log_upb' for f in feature_cols]

    terminal_df = terminal_df.dropna(subset=feature_cols)
    log(f"  Terminal observations: {len(terminal_df):,}")

    # Split data
    train_df = terminal_df[terminal_df['fold'].isin(TRAIN_FOLDS)].copy()
    val_df = terminal_df[terminal_df['fold'].isin(VAL_FOLDS)].copy()
    test_df = terminal_df[terminal_df['fold'] == TEST_FOLD].copy()

    log(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Event distribution
    log("\n  Event distribution (train):")
    event_names = {0: 'Censored', 1: 'Prepay', 2: 'Default'}
    for code, count in train_df[event_col].value_counts().sort_index().items():
        log(f"    {event_names.get(code, 'Other')} (k={code}): {count:,}")

    # Standardize (float64 for MCMC numerical stability; float32 caused
    # lognormal survival tail underflow and stuck NUTS chains)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype('float64')
    X_val = scaler.transform(val_df[feature_cols]).astype('float64')
    X_test = scaler.transform(test_df[feature_cols]).astype('float64')

    duration_train = np.maximum(train_df[time_col].values.astype('float64'), 0.5)
    duration_val = np.maximum(val_df[time_col].values.astype('float64'), 0.5)
    duration_test = np.maximum(test_df[time_col].values.astype('float64'), 0.5)

    event_train = train_df[event_col].values.astype('int64')
    event_val = val_df[event_col].values.astype('int64')
    event_test = test_df[event_col].values.astype('int64')

    # MCMC Inference
    log("\n" + "-" * 70)
    log("Running MCMC inference...")
    log(f"  Device: cpu (multi-chain requires CPU)")
    log(f"  Chains: {args.num_chains}")
    log(f"  Warmup (burn-in): {args.num_warmup}")
    log(f"  Samples per chain: {args.num_samples}")
    log(f"  Thinning: every {args.thinning}th sample")
    log(f"  Iterations per chain: {args.num_warmup + args.num_samples}")

    pyro.clear_param_store()

    X_train_torch = torch.tensor(X_train, dtype=torch.float64, device='cpu')
    duration_train_torch = torch.tensor(duration_train, dtype=torch.float64, device='cpu')
    event_train_torch = torch.tensor(event_train, dtype=torch.int64, device='cpu')

    mcmc_start = time.time()

    mcmc = MCMC(
        NUTS(bayesian_competing_risks_model, target_accept_prob=args.target_accept, jit_compile=False),
        num_samples=args.num_samples,
        warmup_steps=args.num_warmup,
        num_chains=args.num_chains,
    )

    mcmc.run(X=X_train_torch, durations=duration_train_torch,
             events=event_train_torch, n_features=X_train.shape[1])

    mcmc_time = time.time() - mcmc_start
    log(f"  MCMC completed in {mcmc_time/60:.1f} minutes")

    # Get posterior samples and apply thinning (Bhattacharya: every 50th draw)
    raw_samples = {k: v.cpu().numpy() for k, v in mcmc.get_samples().items()}
    posterior_samples = {k: v[::args.thinning] for k, v in raw_samples.items()}
    inference_data = az.from_pyro(mcmc)

    n_raw = next(iter(raw_samples.values())).shape[0]
    n_thinned = next(iter(posterior_samples.values())).shape[0]
    log(f"\n  Raw samples: {n_raw}")
    log(f"  After thinning (every {args.thinning}th): {n_thinned}")
    log("\n  Posterior sample shapes (thinned):")
    for k, v in posterior_samples.items():
        log(f"    {k}: {v.shape}")

    # MCMC diagnostics
    log("\n  MCMC diagnostics:")
    summary = az.summary(inference_data, var_names=['mu_D', 'sigma_D', 'mu_P', 'sigma_P'])
    for param in ['mu_D', 'sigma_D', 'mu_P', 'sigma_P']:
        if param in summary.index:
            rhat = summary.loc[param, 'r_hat']
            ess = summary.loc[param, 'ess_bulk']
            log(f"    {param}: R-hat={rhat:.3f}, ESS={ess:.0f}")

    # Baseline parameter summary
    log("\n" + "-" * 70)
    log("Posterior summaries:")
    log("\n  Baseline hazard parameters:")
    log(f"  {'Parameter':<12} {'Mean':>10} {'Median':>10} {'CI 2.5%':>10} {'CI 97.5%':>10}")
    log("  " + "-" * 52)
    for param in ['mu_D', 'sigma_D', 'mu_P', 'sigma_P']:
        samples = posterior_samples[param]
        log(f"  {param:<12} {np.mean(samples):>10.4f} {np.median(samples):>10.4f} "
            f"{np.percentile(samples, 2.5):>10.4f} {np.percentile(samples, 97.5):>10.4f}")

    # Coefficient summaries
    log("\n  Default coefficients (theta_D):")
    log(f"  {'Feature':<20} {'Mean':>10} {'CI 2.5%':>10} {'CI 97.5%':>10} {'Significant':>12}")
    log("  " + "-" * 64)
    coef_summary_D = []
    for i, f in enumerate(feature_cols):
        samples = posterior_samples['theta_D'][:, i]
        ci_low, ci_high = np.percentile(samples, 2.5), np.percentile(samples, 97.5)
        sig = "Yes" if not (ci_low < 0 < ci_high) else "No"
        log(f"  {f:<20} {np.mean(samples):>10.4f} {ci_low:>10.4f} {ci_high:>10.4f} {sig:>12}")
        coef_summary_D.append({
            'Feature': f, 'Mean': np.mean(samples), 'CI_2.5%': ci_low, 'CI_97.5%': ci_high, 'Significant': sig == "Yes"
        })

    log("\n  Prepayment coefficients (theta_P):")
    log(f"  {'Feature':<20} {'Mean':>10} {'CI 2.5%':>10} {'CI 97.5%':>10} {'Significant':>12}")
    log("  " + "-" * 64)
    coef_summary_P = []
    for i, f in enumerate(feature_cols):
        samples = posterior_samples['theta_P'][:, i]
        ci_low, ci_high = np.percentile(samples, 2.5), np.percentile(samples, 97.5)
        sig = "Yes" if not (ci_low < 0 < ci_high) else "No"
        log(f"  {f:<20} {np.mean(samples):>10.4f} {ci_low:>10.4f} {ci_high:>10.4f} {sig:>12}")
        coef_summary_P.append({
            'Feature': f, 'Mean': np.mean(samples), 'CI_2.5%': ci_low, 'CI_97.5%': ci_high, 'Significant': sig == "Yes"
        })

    # Model evaluation
    log("\n" + "-" * 70)
    log("Model evaluation...")

    model = BayesianModelWrapper(posterior_samples, device='cpu')

    # C-index evaluation
    log("\n  Time-dependent C-index (IPCW):")
    log(f"  {'Cause':<12} " + " ".join([f"τ={t:<6}" for t in TIME_HORIZONS]))
    log("  " + "-" * 40)

    cindex_results = {}
    for cause in ['default', 'prepay']:
        cindex = compute_time_dependent_cindex(
            model, X_train, duration_train, event_train,
            X_test, duration_test, event_test,
            TIME_HORIZONS, cause
        )
        cindex_results[cause] = cindex
        vals = " ".join([f"{cindex.get(t, np.nan):.4f} " for t in TIME_HORIZONS])
        log(f"  {cause.capitalize():<12} {vals}")

    # Summary statistics
    log("\n" + "-" * 70)
    log("Summary:")
    log(f"  Total raw samples: {n_raw}")
    log(f"  Total thinned samples: {n_thinned}")
    log(f"  Effective samples (min): {summary['ess_bulk'].min():.0f}")
    log(f"  Max R-hat: {summary['r_hat'].max():.3f}")

    sig_D = [c['Feature'] for c in coef_summary_D if c['Significant']]
    sig_P = [c['Feature'] for c in coef_summary_P if c['Significant']]
    log(f"\n  Significant covariates:")
    log(f"    Default: {', '.join(sig_D) if sig_D else 'None'}")
    log(f"    Prepay: {', '.join(sig_P) if sig_P else 'None'}")

    # Timing
    total_time = time.time() - start_time
    log(f"\nTotal runtime: {total_time/60:.1f} minutes")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Save results
    log(f"\nSaving results...")

    # Write text report
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"  Report: {results_file}")

    # Save posterior samples
    np.savez(models_dir / 'bayesian_phm_posterior.npz', **posterior_samples)
    log(f"  Posterior: {models_dir / 'bayesian_phm_posterior.npz'}")

    # Save inference data
    inference_data.to_netcdf(models_dir / 'bayesian_phm_inference.nc')
    log(f"  Inference: {models_dir / 'bayesian_phm_inference.nc'}")

    # Save coefficients
    pd.DataFrame(coef_summary_D).to_csv(models_dir / 'bayesian_phm_coef_default.csv', index=False)
    pd.DataFrame(coef_summary_P).to_csv(models_dir / 'bayesian_phm_coef_prepay.csv', index=False)
    log(f"  Coefficients: {models_dir / 'bayesian_phm_coef_*.csv'}")

    # Save C-index results
    pd.DataFrame(cindex_results).to_csv(models_dir / 'bayesian_phm_cindex.csv')
    log(f"  C-index: {models_dir / 'bayesian_phm_cindex.csv'}")

    # Save scaler and features
    with open(models_dir / 'bayesian_phm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(models_dir / 'bayesian_phm_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    log(f"  Scaler/features: {models_dir / 'bayesian_phm_*.pkl'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
