#!/usr/bin/env python3
"""
Bayesian Single-Risk PHM - Prepayment Only

Lognormal baseline proportional hazards model for mortgage prepayment.
Defaults and censored observations are both treated as right-censored.

Usage:
    python run_bayesian_single_risk.py [--num-chains 4] [--num-samples 1000] [--num-warmup 1000]

Output:
    - results/bayesian_single_risk_results.txt (summary report)
    - models/bayesian_single_risk_posterior.npz (posterior samples)
    - models/bayesian_single_risk_inference.nc (ArviZ inference data)
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
    parser = argparse.ArgumentParser(description='Bayesian Single-Risk PHM (Prepayment)')
    parser.add_argument('--num-chains', type=int, default=4, help='Number of MCMC chains')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of post-warmup samples per chain')
    parser.add_argument('--num-warmup', type=int, default=1000, help='Number of NUTS warmup steps')
    parser.add_argument('--target-accept', type=float, default=0.8, help='Target acceptance probability')
    parser.add_argument('--min-ess-per-chain', type=int, default=400, help='Minimum ESS per chain for convergence')
    parser.add_argument('--max-rhat', type=float, default=1.01, help='Maximum R-hat for convergence')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    return parser.parse_args()


# Prior hyperparameters
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


def bayesian_single_risk_model(X, durations, events, n_features):
    """Pyro model for Bayesian single-risk PHM (prepayment).

    events: 1 = prepay (event), 0 = censored (includes original censored + defaults)
    """
    device = X.device
    dtype = X.dtype

    # Baseline hazard parameters
    mu = pyro.sample('mu', dist.Normal(
        torch.tensor(3.0, dtype=dtype, device=device),
        torch.tensor(PRIOR_PARAMS['mu_sd'], dtype=dtype, device=device)))
    sigma = pyro.sample('sigma', dist.Exponential(
        torch.tensor(PRIOR_PARAMS['sigma_rate'], dtype=dtype, device=device)))

    # Regression coefficients
    theta = pyro.sample('theta', dist.Normal(
        torch.zeros(n_features, dtype=dtype, device=device),
        PRIOR_PARAMS['theta_sd'] * torch.ones(n_features, dtype=dtype, device=device)).to_event(1))

    # Linear predictor
    eta = torch.clamp(torch.matmul(X, theta), -20, 20)

    # Hazard and cumulative hazard
    log_h = lognormal_log_hazard(durations, mu, sigma) + eta
    H = lognormal_cumulative_hazard(durations, mu, sigma) * torch.exp(eta)
    H = torch.clamp(H, max=30.0)

    # Log-likelihood: event * log_h - H
    log_lik = events.to(dtype) * log_h - H
    log_lik = torch.where(torch.isfinite(log_lik), log_lik, torch.tensor(-1e10, dtype=dtype, device=device))
    pyro.factor('log_likelihood', torch.sum(log_lik))


# ==============================================================================
# Model Wrapper for Predictions
# ==============================================================================

class BayesianModelWrapper:
    """Wrapper for posterior survival predictions."""

    def __init__(self, posterior_samples, device='cpu'):
        self.posterior_samples_ = posterior_samples
        self.device = device

    def predict_survival(self, X, times):
        """Predict survival function S(t) = exp(-H(t))."""
        X = torch.tensor(X, dtype=torch.float64, device=self.device)
        times = torch.tensor(times, dtype=torch.float64, device=self.device)
        N, T = X.shape[0], len(times)

        ps = {k: torch.tensor(v, device=self.device) for k, v in self.posterior_samples_.items()}
        n_samples = len(ps['mu'])
        surv_samples = np.zeros((n_samples, N, T))

        for s in range(n_samples):
            eta = torch.matmul(X, ps['theta'][s])
            for t_idx, t in enumerate(times):
                H = lognormal_cumulative_hazard(t, ps['mu'][s], ps['sigma'][s]) * torch.exp(eta)
                surv_samples[s, :, t_idx] = torch.exp(-H).cpu().numpy()

        return np.mean(surv_samples, 0), np.percentile(surv_samples, 2.5, 0), np.percentile(surv_samples, 97.5, 0)


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def compute_time_dependent_cindex(posterior_samples, X_train, durations_train, events_train,
                                   X_test, durations_test, events_test, times):
    """Compute IPCW C-index at specified time horizons.

    Uses the posterior mean of the regression coefficients as a linear
    risk score (eta = X @ theta_mean).  This is a monotone function of
    the hazard, so it preserves concordance ranking.
    """
    y_train = Surv.from_arrays(events_train.astype(bool), durations_train)
    y_test = Surv.from_arrays(events_test.astype(bool), durations_test)

    # Posterior mean linear predictor as risk score
    theta_mean = posterior_samples['theta'].mean(axis=0)
    risk_scores = X_test @ theta_mean

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

    results_file = output_dir / 'bayesian_single_risk_results.txt'

    # Start logging
    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("BAYESIAN SINGLE-RISK PHM (PREPAYMENT)")
    log("Lognormal baseline, proportional hazards")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"Pyro version: {pyro.__version__}")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    log(f"\nConfiguration (NUTS — no thinning needed):")
    log(f"  num_chains: {args.num_chains}")
    log(f"  num_samples: {args.num_samples}")
    log(f"  num_warmup: {args.num_warmup}")
    log(f"  target_accept: {args.target_accept}")
    log(f"  seed: {args.seed}")
    log(f"  total posterior samples: {args.num_samples * args.num_chains}")
    log(f"\nConvergence thresholds:")
    log(f"  min_ess_per_chain: {args.min_ess_per_chain}")
    log(f"  max_rhat: {args.max_rhat}")

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

    # Recode events: prepay (1) -> 1, everything else -> 0 (censored)
    log("\n  Event recoding: prepay=1 (event), default+censored=0 (censored)")
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        n_prepay = (split_df[event_col] == 1).sum()
        n_cens = (split_df[event_col] != 1).sum()
        log(f"    {split_name}: {n_prepay:,} prepay events, {n_cens:,} censored")

    # Standardize (float64 for MCMC numerical stability)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype('float64')
    X_val = scaler.transform(val_df[feature_cols]).astype('float64')
    X_test = scaler.transform(test_df[feature_cols]).astype('float64')

    duration_train = np.maximum(train_df[time_col].values.astype('float64'), 0.5)
    duration_val = np.maximum(val_df[time_col].values.astype('float64'), 0.5)
    duration_test = np.maximum(test_df[time_col].values.astype('float64'), 0.5)

    # Binary event indicator: prepay = 1, else = 0
    event_train = (train_df[event_col].values == 1).astype('int64')
    event_val = (val_df[event_col].values == 1).astype('int64')
    event_test = (test_df[event_col].values == 1).astype('int64')

    # MCMC Inference
    log("\n" + "-" * 70)
    log("Running MCMC inference...")
    log(f"  Device: cpu (multi-chain requires CPU)")
    log(f"  Chains: {args.num_chains}")
    log(f"  Warmup: {args.num_warmup}")
    log(f"  Samples per chain: {args.num_samples}")
    log(f"  Total posterior samples: {args.num_samples * args.num_chains}")
    log(f"  Parameters: 2 baseline + {len(feature_cols)} coefficients = {2 + len(feature_cols)}")

    pyro.clear_param_store()

    X_train_torch = torch.tensor(X_train, dtype=torch.float64, device='cpu')
    duration_train_torch = torch.tensor(duration_train, dtype=torch.float64, device='cpu')
    event_train_torch = torch.tensor(event_train, dtype=torch.int64, device='cpu')

    mcmc_start = time.time()

    mcmc = MCMC(
        NUTS(bayesian_single_risk_model, target_accept_prob=args.target_accept, jit_compile=False),
        num_samples=args.num_samples,
        warmup_steps=args.num_warmup,
        num_chains=args.num_chains,
    )

    mcmc.run(X=X_train_torch, durations=duration_train_torch,
             events=event_train_torch, n_features=X_train.shape[1])

    mcmc_time = time.time() - mcmc_start
    log(f"  MCMC completed in {mcmc_time/60:.1f} minutes")

    # Get posterior samples (no thinning — NUTS produces near-independent draws)
    posterior_samples = {k: v.cpu().numpy() for k, v in mcmc.get_samples().items()}
    inference_data = az.from_pyro(mcmc)

    n_samples = next(iter(posterior_samples.values())).shape[0]
    log(f"\n  Total posterior samples: {n_samples}")
    log("\n  Posterior sample shapes:")
    for k, v in posterior_samples.items():
        log(f"    {k}: {v.shape}")

    # Convergence diagnostics
    log("\n  Convergence diagnostics:")
    summary = az.summary(inference_data, var_names=['mu', 'sigma', 'theta'])
    min_ess = summary['ess_bulk'].min()
    max_rhat = summary['r_hat'].max()
    min_ess_threshold = args.min_ess_per_chain * args.num_chains

    for param in ['mu', 'sigma']:
        if param in summary.index:
            rhat = summary.loc[param, 'r_hat']
            ess = summary.loc[param, 'ess_bulk']
            log(f"    {param}: R-hat={rhat:.4f}, ESS={ess:.0f}")

    log(f"\n    Min bulk ESS:  {min_ess:.0f}  (threshold: {min_ess_threshold})")
    log(f"    Max R-hat:     {max_rhat:.4f}  (threshold: {args.max_rhat})")

    ess_ok = min_ess >= min_ess_threshold
    rhat_ok = max_rhat <= args.max_rhat
    if ess_ok and rhat_ok:
        log("    CONVERGED")
    else:
        if not ess_ok:
            worst = summary['ess_bulk'].idxmin()
            log(f"    WARNING: Low ESS — worst: {worst} (ESS={summary.loc[worst, 'ess_bulk']:.0f})")
        if not rhat_ok:
            worst = summary['r_hat'].idxmax()
            log(f"    WARNING: High R-hat — worst: {worst} (R-hat={summary.loc[worst, 'r_hat']:.4f})")
        log("    Consider increasing --num-samples or --num-warmup.")

    # Baseline parameter summary
    log("\n" + "-" * 70)
    log("Posterior summaries:")
    log("\n  Baseline hazard parameters:")
    log(f"  {'Parameter':<12} {'Mean':>10} {'SD':>10} {'Median':>10} {'CI 2.5%':>10} {'CI 97.5%':>10}")
    log("  " + "-" * 64)
    for param in ['mu', 'sigma']:
        samples = posterior_samples[param]
        log(f"  {param:<12} {np.mean(samples):>10.4f} {np.std(samples):>10.4f} {np.median(samples):>10.4f} "
            f"{np.percentile(samples, 2.5):>10.4f} {np.percentile(samples, 97.5):>10.4f}")

    # Coefficient summaries
    log("\n  Prepayment coefficients (theta):")
    log(f"  {'Feature':<20} {'Mean':>10} {'SD':>10} {'CI 2.5%':>10} {'CI 97.5%':>10} {'Significant':>12}")
    log("  " + "-" * 76)
    coef_summary = []
    for i, f in enumerate(feature_cols):
        samples = posterior_samples['theta'][:, i]
        ci_low, ci_high = np.percentile(samples, 2.5), np.percentile(samples, 97.5)
        sig = "Yes" if not (ci_low < 0 < ci_high) else "No"
        log(f"  {f:<20} {np.mean(samples):>10.4f} {np.std(samples):>10.4f} {ci_low:>10.4f} {ci_high:>10.4f} {sig:>12}")
        coef_summary.append({
            'Feature': f, 'Mean': np.mean(samples), 'SD': np.std(samples),
            'CI_2.5%': ci_low, 'CI_97.5%': ci_high, 'Significant': sig == "Yes"
        })

    # Model evaluation
    log("\n" + "-" * 70)
    log("Model evaluation...")

    # C-index evaluation
    log("\n  Time-dependent C-index (IPCW):")
    log(f"  " + " ".join([f"τ={t:<6}" for t in TIME_HORIZONS]))
    log("  " + "-" * 30)

    cindex_results = compute_time_dependent_cindex(
        posterior_samples, X_train, duration_train, event_train,
        X_test, duration_test, event_test, TIME_HORIZONS
    )
    vals = " ".join([f"{cindex_results.get(t, np.nan):.4f} " for t in TIME_HORIZONS])
    log(f"  {vals}")

    # Summary statistics
    log("\n" + "-" * 70)
    log("Summary:")
    log(f"  Total posterior samples: {n_samples}")
    log(f"  Converged: {'Yes' if ess_ok and rhat_ok else 'No'}")
    log(f"  Effective samples (min): {summary['ess_bulk'].min():.0f}")
    log(f"  Max R-hat: {summary['r_hat'].max():.3f}")

    sig_feats = [c['Feature'] for c in coef_summary if c['Significant']]
    log(f"\n  Significant covariates:")
    log(f"    Prepay: {', '.join(sig_feats) if sig_feats else 'None'}")

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
    np.savez(models_dir / 'bayesian_single_risk_posterior.npz', **posterior_samples)
    log(f"  Posterior: {models_dir / 'bayesian_single_risk_posterior.npz'}")

    # Save inference data
    inference_data.to_netcdf(models_dir / 'bayesian_single_risk_inference.nc')
    log(f"  Inference: {models_dir / 'bayesian_single_risk_inference.nc'}")

    # Save coefficients
    pd.DataFrame(coef_summary).to_csv(models_dir / 'bayesian_single_risk_coef.csv', index=False)
    log(f"  Coefficients: {models_dir / 'bayesian_single_risk_coef.csv'}")

    # Save C-index results
    pd.DataFrame({'prepay': cindex_results}, index=cindex_results.keys()).to_csv(
        models_dir / 'bayesian_single_risk_cindex.csv')
    log(f"  C-index: {models_dir / 'bayesian_single_risk_cindex.csv'}")

    # Save scaler and features
    with open(models_dir / 'bayesian_single_risk_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(models_dir / 'bayesian_single_risk_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    log(f"  Scaler/features: {models_dir / 'bayesian_single_risk_*.pkl'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
