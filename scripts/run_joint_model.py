#!/usr/bin/env python3
"""
Joint Model for Longitudinal and Discrete Survival Data — Supercomputer Script

Extension of Medina-Olivares et al. (2022) to competing risks (prepayment
and default).  Bayesian MCMC estimation via Pyro NUTS.

Usage:
    python run_joint_model.py [--num-chains 4] [--num-samples 2000] [--max-loans 10000]

Output:
    - results/joint_model_results.txt       (summary report)
    - models/joint_model_posterior.npz       (posterior samples)
    - models/joint_model_inference.nc        (ArviZ inference data)
    - models/joint_model_cindex.csv          (C-index at 24/48/72 months)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyro
import arviz as az
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from competing_risks.joint_model import (
    JointCompetingRisksModel,
    prepare_joint_model_data,
    _estimate_random_effects,
    STATIC_FEATURES,
    TVC_DEFAULT,
)
from competing_risks.evaluation import (
    time_dependent_concordance_index,
    brier_score_competing_risks,
    EVAL_TIMES,
)


# =============================================================================
# Configuration
# =============================================================================

TRAIN_FOLDS = list(range(9))
VAL_FOLDS = [9]
TEST_FOLD = 10

STATIC_COLS = ['int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r']
TVC_COL = 'ppi_c_FRMA'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Joint Model — Longitudinal + Competing Risks (Pyro NUTS)')

    # MCMC settings
    parser.add_argument('--num-chains', type=int, default=4,
                        help='Number of MCMC chains (default: 4)')
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Post-warmup samples per chain (default: 2000)')
    parser.add_argument('--num-warmup', type=int, default=1000,
                        help='NUTS warmup steps per chain (default: 1000)')
    parser.add_argument('--target-accept', type=float, default=0.90,
                        help='Target acceptance probability (default: 0.90)')

    # Data settings
    parser.add_argument('--max-loans', type=int, default=None,
                        help='Max training loans (None = full dataset)')
    parser.add_argument('--max-test-loans', type=int, default=None,
                        help='Max test loans (None = full dataset)')
    parser.add_argument('--n-interior-knots', type=int, default=3,
                        help='B-spline interior knots (default: 3)')

    # Prediction settings
    parser.add_argument('--n-posterior-cif', type=int, default=200,
                        help='Posterior samples for CIF prediction (default: 200)')
    parser.add_argument('--eval-times', type=int, nargs='+', default=[24, 48, 72],
                        help='Evaluation horizons in months (default: 24 48 72)')

    # Paths
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Models directory')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    models_dir = base_dir / args.models_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'joint_model_results.txt'

    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("JOINT MODEL — LONGITUDINAL + COMPETING RISKS")
    log("Medina-Olivares et al. (2022), extended to competing risks")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch:    {torch.__version__}")
    log(f"Pyro:       {pyro.__version__}")
    log(f"Device:     cpu (float64 required for NUTS stability)")

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    log(f"\nMCMC configuration:")
    log(f"  num_chains:        {args.num_chains}")
    log(f"  num_samples:       {args.num_samples}")
    log(f"  num_warmup:        {args.num_warmup}")
    log(f"  target_accept:     {args.target_accept}")
    log(f"  total posterior:   {args.num_samples * args.num_chains}")
    log(f"\nData configuration:")
    log(f"  max_loans (train): {args.max_loans or 'all'}")
    log(f"  max_loans (test):  {args.max_test_loans or 'all'}")
    log(f"  n_interior_knots:  {args.n_interior_knots}")
    log(f"  eval_times:        {args.eval_times}")
    log(f"  seed:              {args.seed}")

    # ── Load data ──────────────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Loading data...")
    panel = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    log(f"  Full panel: {panel.shape[0]:,} rows, "
        f"{panel['loan_sequence_number'].nunique():,} loans")

    if 'log_upb' not in panel.columns and 'orig_upb' in panel.columns:
        panel['log_upb'] = np.log(panel['orig_upb'].clip(lower=1).astype(float))

    required_cols = STATIC_COLS + [TVC_COL]
    panel = panel.dropna(subset=required_cols)
    log(f"  After dropna: {panel.shape[0]:,} rows, "
        f"{panel['loan_sequence_number'].nunique():,} loans")

    # ── Train / test split ─────────────────────────────────────────────────
    train_panel = panel[panel['fold'].isin(TRAIN_FOLDS)].copy()
    test_panel = panel[panel['fold'] == TEST_FOLD].copy()

    log(f"\n  Train: {train_panel['loan_sequence_number'].nunique():,} loans, "
        f"{train_panel.shape[0]:,} rows")
    log(f"  Test:  {test_panel['loan_sequence_number'].nunique():,} loans, "
        f"{test_panel.shape[0]:,} rows")

    # ── Subsample if requested ─────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)

    if args.max_loans is not None:
        train_loan_ids = train_panel['loan_sequence_number'].unique()
        if len(train_loan_ids) > args.max_loans:
            # Stratified subsample: keep proportional event mix
            loan_events = train_panel.groupby(
                'loan_sequence_number')['event_code'].last()
            sampled_ids = []
            for evt in [0, 1, 2]:
                pool = loan_events[loan_events == evt].index.values
                n_take = max(1, int(
                    args.max_loans * len(pool) / len(train_loan_ids)))
                sampled_ids.extend(
                    rng.choice(pool, size=min(n_take, len(pool)), replace=False))
            sampled_ids = np.array(sampled_ids)[:args.max_loans]
            train_panel = train_panel[
                train_panel['loan_sequence_number'].isin(sampled_ids)]
            log(f"\n  Subsampled train: "
                f"{train_panel['loan_sequence_number'].nunique():,} loans, "
                f"{train_panel.shape[0]:,} rows")

    if args.max_test_loans is not None:
        test_loan_ids = test_panel['loan_sequence_number'].unique()
        if len(test_loan_ids) > args.max_test_loans:
            sampled_test = rng.choice(
                test_loan_ids, size=args.max_test_loans, replace=False)
            test_panel = test_panel[
                test_panel['loan_sequence_number'].isin(sampled_test)]
            log(f"  Subsampled test:  "
                f"{test_panel['loan_sequence_number'].nunique():,} loans, "
                f"{test_panel.shape[0]:,} rows")

    # Event distribution
    log("\n  Train event distribution:")
    evt_dist = (train_panel.groupby('loan_sequence_number')['event_code']
                .last().value_counts().sort_index())
    evt_names = {0: 'Censored', 1: 'Prepay', 2: 'Default'}
    for code, count in evt_dist.items():
        log(f"    {evt_names.get(code, 'Other')} (k={code}): {count:,}")

    # ── Fit joint model ────────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Fitting joint model (NUTS MCMC)...")
    log(f"  This may take several hours for large datasets.")

    mcmc_start = time.time()

    model = JointCompetingRisksModel(
        tvc_col=TVC_COL,
        static_cols=STATIC_COLS,
        n_interior_knots=args.n_interior_knots,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept_prob=args.target_accept,
        random_seed=args.seed,
        device='cpu',
    )

    model.fit(train_panel)

    mcmc_time = time.time() - mcmc_start
    log(f"\n  MCMC completed in {mcmc_time / 60:.1f} minutes "
        f"({mcmc_time / 3600:.2f} hours)")

    # ── Convergence diagnostics ────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Convergence diagnostics:")

    summary_df = model.get_posterior_summary()

    # Key scalar parameters
    scalar_params = ['alpha_0', 'phi', 'sigma',
                     'lambda_prepay', 'lambda_default',
                     'tau_intercept', 'tau_slope']
    key_summary = summary_df[summary_df['parameter'].isin(scalar_params)]

    log(f"\n  {'Parameter':<20} {'Mean':>10} {'Std':>10} {'5%':>10} {'95%':>10}")
    log("  " + "-" * 64)
    for _, row in key_summary.iterrows():
        log(f"  {row['parameter']:<20} {row['mean']:>10.4f} {row['std']:>10.4f} "
            f"{row['5%']:>10.4f} {row['95%']:>10.4f}")

    # ArviZ diagnostics (R-hat, ESS) — exclude random effects (z_RE)
    if model.inference_data_ is not None:
        try:
            var_names = [v for v in model.inference_data_.posterior.data_vars
                         if v not in ('z_RE',)]
            az_summary = az.summary(
                model.inference_data_, var_names=var_names)
            min_ess = az_summary['ess_bulk'].min()
            max_rhat = az_summary['r_hat'].max()

            log(f"\n  Min bulk ESS:  {min_ess:.0f}")
            log(f"  Max R-hat:     {max_rhat:.4f}")

            if max_rhat < 1.05 and min_ess > 400:
                log("  CONVERGED")
            else:
                if max_rhat >= 1.05:
                    worst = az_summary['r_hat'].idxmax()
                    log(f"  WARNING: High R-hat — worst: {worst} "
                        f"(R-hat={az_summary.loc[worst, 'r_hat']:.4f})")
                if min_ess < 400:
                    worst = az_summary['ess_bulk'].idxmin()
                    log(f"  WARNING: Low ESS — worst: {worst} "
                        f"(ESS={az_summary.loc[worst, 'ess_bulk']:.0f})")
                log("  Consider increasing --num-samples or --num-warmup.")
        except Exception as e:
            log(f"  ArviZ diagnostics failed: {e}")
    else:
        log("  ArviZ inference data not available.")

    # ── Interpret association parameters ───────────────────────────────────
    log("\n" + "-" * 70)
    log("Association parameters (TVC -> hazard):")
    log(f"  TVC: {TVC_COL} = int_rate - current_30yr_FRM_average\n")

    for param in ['lambda_prepay', 'lambda_default']:
        samples = model.posterior_samples_[param]
        mean = np.mean(samples)
        ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])
        cause = param.split('_')[1]
        direction = "INCREASES" if mean > 0 else "DECREASES"
        sig = "Yes" if not (ci_lo < 0 < ci_hi) else "No"
        log(f"  {param}:")
        log(f"    Mean: {mean:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"Significant: {sig}")
        log(f"    -> Higher prepayment incentive {direction} {cause} risk\n")

    # Longitudinal parameters
    log("  Longitudinal parameters:")
    for param in ['alpha_0', 'phi', 'sigma']:
        samples = model.posterior_samples_[param]
        log(f"    {param}: {np.mean(samples):.4f} "
            f"(std={np.std(samples):.4f})")

    # ── CIF prediction on test set ─────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Computing CIF predictions on test set...")

    cif_start = time.time()
    cif_result = model.predict_cif(
        test_panel,
        horizons=args.eval_times,
        n_posterior_samples=args.n_posterior_cif,
    )
    cif_time = time.time() - cif_start
    log(f"  CIF prediction completed in {cif_time / 60:.1f} minutes")
    log(f"  Test loans: {len(cif_result['loan_sequence_number']):,}")

    for t in args.eval_times:
        cif_p = cif_result[f'cif_prepay_{t}']
        cif_d = cif_result[f'cif_default_{t}']
        log(f"  t={t}: CIF_prepay={cif_p.mean():.4f}, "
            f"CIF_default={cif_d.mean():.4f}, "
            f"CIF_total={(cif_p + cif_d).mean():.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Evaluation: Time-dependent C-index and Brier Score")

    event_times = cif_result['duration']
    event_codes = cif_result['event_code']

    results_rows = []
    log(f"\n  {'Horizon':<10} {'C-Prepay':>10} {'C-Default':>10} "
        f"{'BS-Prepay':>10} {'BS-Default':>10}")
    log("  " + "-" * 54)

    for t in args.eval_times:
        c_prepay, _, _ = time_dependent_concordance_index(
            event_times, event_codes,
            cif_result[f'cif_prepay_{t}'],
            eval_time=t, event_of_interest=1)

        c_default, _, _ = time_dependent_concordance_index(
            event_times, event_codes,
            cif_result[f'cif_default_{t}'],
            eval_time=t, event_of_interest=2)

        bs_prepay = brier_score_competing_risks(
            event_times, event_codes,
            cif_result[f'cif_prepay_{t}'],
            eval_time=t, event_of_interest=1)

        bs_default = brier_score_competing_risks(
            event_times, event_codes,
            cif_result[f'cif_default_{t}'],
            eval_time=t, event_of_interest=2)

        results_rows.append({
            'Horizon': t,
            'C_prepay': c_prepay,
            'C_default': c_default,
            'BS_prepay': bs_prepay,
            'BS_default': bs_default,
        })

        log(f"  t={t:<7} {c_prepay:>10.4f} {c_default:>10.4f} "
            f"{bs_prepay:>10.4f} {bs_default:>10.4f}")

    # Mean row
    results_df = pd.DataFrame(results_rows)
    means = results_df[['C_prepay', 'C_default', 'BS_prepay', 'BS_default']].mean()
    log(f"  {'Mean':<10} {means['C_prepay']:>10.4f} {means['C_default']:>10.4f} "
        f"{means['BS_prepay']:>10.4f} {means['BS_default']:>10.4f}")

    # ── Timing summary ─────────────────────────────────────────────────────
    total_time = time.time() - start_time
    log("\n" + "-" * 70)
    log("Timing summary:")
    log(f"  MCMC fitting:     {mcmc_time / 60:>8.1f} min")
    log(f"  CIF prediction:   {cif_time / 60:>8.1f} min")
    log(f"  Total runtime:    {total_time / 60:>8.1f} min "
        f"({total_time / 3600:.2f} hours)")
    log(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ── Save results ───────────────────────────────────────────────────────
    log("\nSaving results...")

    # Text report
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"  Report:     {results_file}")

    # Posterior samples (exclude z_RE to keep file size manageable)
    posterior_to_save = {
        k: v for k, v in model.posterior_samples_.items()
        if k != 'z_RE'
    }
    np.savez(models_dir / 'joint_model_posterior.npz', **posterior_to_save)
    log(f"  Posterior:   {models_dir / 'joint_model_posterior.npz'}")

    # ArviZ inference data
    if model.inference_data_ is not None:
        try:
            model.inference_data_.to_netcdf(
                models_dir / 'joint_model_inference.nc')
            log(f"  Inference:   {models_dir / 'joint_model_inference.nc'}")
        except Exception as e:
            log(f"  Inference save failed: {e}")

    # Evaluation results
    results_df.to_csv(models_dir / 'joint_model_cindex.csv', index=False)
    log(f"  C-index:     {models_dir / 'joint_model_cindex.csv'}")

    # Parameter summary
    summary_df.to_csv(models_dir / 'joint_model_params.csv', index=False)
    log(f"  Parameters:  {models_dir / 'joint_model_params.csv'}")

    # Data standardisation (needed for later prediction)
    scaler_info = {
        'z_mean': model.data_['z_mean'],
        'z_std': model.data_['z_std'],
        'static_cols': model.data_['static_cols'],
        'tvc_col': model.data_['tvc_col'],
    }
    with open(models_dir / 'joint_model_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_info, f)
    log(f"  Scaler:      {models_dir / 'joint_model_scaler.pkl'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
