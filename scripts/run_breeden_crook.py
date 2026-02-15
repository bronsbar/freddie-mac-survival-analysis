#!/usr/bin/env python3
"""
Breeden & Crook (2022) Multihorizon Discrete-Time Survival Model - Supercomputer Script

Implements multihorizon logistic regressions for competing risks
(prepayment and default), where each horizon-L model uses delinquency
lagged by L months.

Usage:
    python run_breeden_crook.py [--max-horizon 12] [--cif-horizon 72] [--C 1.0]

Output:
    - models/breeden_crook_models.pkl          (26 logistic regressions)
    - models/breeden_crook_scaler.pkl          (per-horizon scalers)
    - models/breeden_crook_coefficients.csv    (all coefficients)
    - results/breeden_crook_cindex.csv         (C-index results)
    - results/breeden_crook_results.txt        (summary report)
    - reports/figures/breeden_crook_coef_delinquency.png
    - reports/figures/breeden_crook_coef_origination.png
    - reports/figures/breeden_crook_pseudo_r2.png
    - reports/figures/breeden_crook_cindex_comparison.png
    - reports/figures/breeden_crook_cif_curves.png
"""

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.competing_risks.breeden_crook import (
    BreedenCrookMultihorizon,
    enrich_panel_with_delinquency,
    create_delinquency_indicators,
    create_lagged_delinquency,
    plot_delinquency_coefficients,
    plot_origination_coefficients,
    plot_pseudo_r2_by_horizon,
    EVENT_NAMES,
)
from src.competing_risks.evaluation import (
    time_dependent_concordance_index,
    evaluate_model_at_times,
    evaluate_all_events,
    brier_score_competing_risks,
    EVAL_TIMES,
)


# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Breeden & Crook (2022) Multihorizon Discrete-Time Survival Model')

    # Model
    parser.add_argument('--max-horizon', type=int, default=12,
                        help='Maximum forecast horizon L')
    parser.add_argument('--cif-horizon', type=int, default=72,
                        help='Maximum months for CIF prediction')
    parser.add_argument('--C', type=float, default=None,
                        help='Regularization strength (None = tune on validation)')
    parser.add_argument('--solver', type=str, default='lbfgs',
                        help='Logistic regression solver')
    parser.add_argument('--n-age-knots', type=int, default=5,
                        help='Number of knots for age spline')

    # General
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Raw data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Models directory')
    parser.add_argument('--figures-dir', type=str, default='reports/figures',
                        help='Figures directory')

    return parser.parse_args()


# Cross-validation folds
TRAIN_FOLDS = list(range(10))
VAL_FOLDS = [9]
TEST_FOLD = 10
TIME_HORIZONS = [24, 48, 72]


# ==============================================================================
# Plotting
# ==============================================================================

def plot_cindex_comparison(results_df, figures_dir):
    """Bar chart of C-index by event type and horizon."""
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = TIME_HORIZONS
    prepay_vals = []
    default_vals = []

    for t in horizons:
        row = results_df[results_df['Metric'] == f'C({t})']
        prepay_vals.append(row['Prepay (k=1)'].values[0] if len(row) > 0 else 0)
        default_vals.append(row['Default (k=2)'].values[0] if len(row) > 0 else 0)

    x = np.arange(len(horizons))
    width = 0.35
    bars1 = ax.bar(x - width/2, prepay_vals, width, label='Prepayment',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, default_vals, width, label='Default',
                   color='indianred', alpha=0.8)

    for bar, val in zip(bars1, prepay_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, default_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Time Horizon (months)')
    ax.set_ylabel('Time-Dependent C-index')
    ax.set_title('Breeden-Crook: Time-Dependent Concordance Index')
    ax.set_xticks(x)
    ax.set_xticklabels([f'tau = {h}' for h in horizons])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figures_dir / 'breeden_crook_cindex_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_sample_cif_curves(CIF_def, CIF_pre, S, figures_dir, n_samples=5):
    """Plot sample CIF curves for selected loans."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)
    sample_idx = np.random.choice(len(CIF_def), size=n_samples, replace=False)
    months = np.arange(CIF_def.shape[1])

    for ax, data, title, ylabel in zip(axes,
        [CIF_def[sample_idx], CIF_pre[sample_idx], S[sample_idx]],
        ['Default CIF', 'Prepayment CIF', 'Survival S(t)'],
        ['Cumulative Incidence', 'Cumulative Incidence', 'Survival Probability'],
    ):
        for i, idx in enumerate(sample_idx):
            ax.plot(months, data[i], label=f'Loan {idx}', alpha=0.7)
        ax.set_xlabel('Months from Forecast Origin')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Breeden-Crook: {title}')
        ax.legend(loc='lower right' if 'CIF' in title else 'lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(figures_dir / 'breeden_crook_cif_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    raw_dir = base_dir / args.raw_dir
    output_dir = base_dir / args.output_dir
    models_dir = base_dir / args.models_dir
    figures_dir = base_dir / args.figures_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'breeden_crook_results.txt'

    # Logging
    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("BREEDEN & CROOK (2022) MULTIHORIZON DISCRETE-TIME SURVIVAL MODEL")
    log("Competing risks: prepayment (k=1) and default (k=2)")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Seeds
    np.random.seed(args.seed)

    log(f"\nConfiguration:")
    log(f"  max_horizon: {args.max_horizon}")
    log(f"  cif_horizon: {args.cif_horizon}")
    log(f"  C (regularization): {args.C or 'auto-tune'}")
    log(f"  solver: {args.solver}")
    log(f"  n_age_knots: {args.n_age_knots}")
    log(f"  seed: {args.seed}")

    # ==================== Load & Enrich Data ====================
    log("\n" + "-" * 70)
    log("Step 1: Loading and enriching panel data...")

    panel_df = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    log(f"  Loaded panel: {len(panel_df):,} rows, "
        f"{panel_df['loan_sequence_number'].nunique():,} loans")

    # Enrich with delinquency status
    cache_path = data_dir / 'loan_month_panel_with_delinq.parquet'
    panel_df = enrich_panel_with_delinquency(
        panel_df, raw_dir, cache_path=cache_path,
    )

    log(f"\n  Delinquency status distribution:")
    for status, count in panel_df['delinq_status'].value_counts().sort_index().items():
        pct = 100 * count / len(panel_df)
        log(f"    Status {status}: {count:>10,} ({pct:5.2f}%)")

    # ==================== Create Delinquency Features ====================
    log("\n" + "-" * 70)
    log("Step 2: Creating delinquency indicators and lags...")

    panel_df = create_delinquency_indicators(panel_df)
    panel_df = create_lagged_delinquency(panel_df, max_lag=args.max_horizon)

    log(f"  Panel shape after enrichment: {panel_df.shape}")

    # Create log_orig_upb
    if 'orig_upb' in panel_df.columns:
        panel_df['log_orig_upb'] = np.log(
            panel_df['orig_upb'].astype(float).clip(lower=1)
        )

    # Create bal_repaid_lag1
    if 'bal_repaid' in panel_df.columns:
        panel_df['bal_repaid_lag1'] = panel_df.groupby(
            'loan_sequence_number'
        )['bal_repaid'].shift(1)

    # ==================== Train/Val/Test Split ====================
    log("\n" + "-" * 70)
    log("Step 3: Splitting data...")

    train_folds_actual = [f for f in TRAIN_FOLDS if f not in VAL_FOLDS]
    train_panel = panel_df[panel_df['fold'].isin(train_folds_actual)].copy()
    val_panel = panel_df[panel_df['fold'].isin(VAL_FOLDS)].copy()
    test_panel = panel_df[panel_df['fold'] == TEST_FOLD].copy()

    log(f"  Train (folds {train_folds_actual}): {len(train_panel):,} rows, "
        f"{train_panel['loan_sequence_number'].nunique():,} loans")
    log(f"  Val (fold {VAL_FOLDS}): {len(val_panel):,} rows, "
        f"{val_panel['loan_sequence_number'].nunique():,} loans")
    log(f"  Test (fold {TEST_FOLD}): {len(test_panel):,} rows, "
        f"{test_panel['loan_sequence_number'].nunique():,} loans")

    log("\n  Event distribution (train terminal):")
    train_terminal = train_panel.groupby('loan_sequence_number').last().reset_index()
    for code, count in train_terminal['event_code'].value_counts().sort_index().items():
        log(f"    {EVENT_NAMES.get(code, 'Other')} (k={code}): {count:,}")

    # ==================== Fit Models ====================
    log("\n" + "-" * 70)
    log("Step 4: Fitting multihorizon logistic regressions...")
    log(f"  Total models: 2 risks x ({args.max_horizon} horizons + 1 origination) "
        f"= {2 * (args.max_horizon + 1)} models")

    model = BreedenCrookMultihorizon(
        max_horizon=args.max_horizon,
        solver=args.solver,
        n_age_knots=args.n_age_knots,
        seed=args.seed,
    )

    train_start = time.time()
    model.fit(train_panel, val_panel, C=args.C, log_fn=log)
    train_time = time.time() - train_start
    log(f"\nTraining completed in {train_time/60:.1f} minutes")

    # ==================== Coefficient Analysis ====================
    log("\n" + "-" * 70)
    log("Step 5: Coefficient analysis...")

    coef_df = model.get_coefficients()
    r2_df = model.get_pseudo_r2()

    log(f"  Total coefficients: {len(coef_df):,}")
    log(f"\n  Validation AUC by horizon:")
    for risk in ['default', 'prepay']:
        risk_r2 = r2_df[r2_df['risk'] == risk].sort_values('horizon')
        log(f"    {risk.title()}:")
        for _, row in risk_r2.iterrows():
            log(f"      L={int(row['horizon']):2d}: AUC={row['auc_val']:.4f}, "
                f"Gini={row['gini']:.4f}, events={int(row['n_events']):,}")

    # Plot coefficients
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_delinquency_coefficients(coef_df, risk='default', ax=axes[0])
    plot_delinquency_coefficients(coef_df, risk='prepay', ax=axes[1])
    axes[1].set_title('Delinquency Coefficients vs. Horizon (Prepay)')
    plt.tight_layout()
    plt.savefig(figures_dir / 'breeden_crook_coef_delinquency.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {figures_dir / 'breeden_crook_coef_delinquency.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_origination_coefficients(coef_df, risk='default', ax=axes[0])
    plot_origination_coefficients(coef_df, risk='prepay', ax=axes[1])
    axes[1].set_title('Origination Variable Coefficients vs. Horizon (Prepay)')
    plt.tight_layout()
    plt.savefig(figures_dir / 'breeden_crook_coef_origination.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {figures_dir / 'breeden_crook_coef_origination.png'}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pseudo_r2_by_horizon(r2_df, ax=ax)
    plt.tight_layout()
    plt.savefig(figures_dir / 'breeden_crook_pseudo_r2.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {figures_dir / 'breeden_crook_pseudo_r2.png'}")

    # ==================== CIF Prediction ====================
    log("\n" + "-" * 70)
    log("Step 6: Generating CIF predictions for test set...")

    # Get terminal observations for test loans
    test_terminal = test_panel.groupby('loan_sequence_number').last().reset_index()
    log(f"  Test loans: {len(test_terminal):,}")

    cif_start = time.time()
    CIF_def, CIF_pre, S = model.predict_cif(test_terminal, max_months=args.cif_horizon)
    cif_time = time.time() - cif_start
    log(f"  CIF computation: {cif_time:.1f} seconds")

    # Verify CIF validity: CIF_def + CIF_pre + S = 1
    check = CIF_def + CIF_pre + S
    log(f"  CIF validity check (CIF_def + CIF_pre + S):")
    log(f"    Min: {check.min():.6f}, Max: {check.max():.6f}, "
        f"Mean: {check.mean():.6f}")

    # Plot sample CIF curves
    plot_sample_cif_curves(CIF_def, CIF_pre, S, figures_dir)
    log(f"  Saved: {figures_dir / 'breeden_crook_cif_curves.png'}")

    # ==================== Evaluation ====================
    log("\n" + "-" * 70)
    log("Step 7: Evaluation...")

    event_times = test_terminal['loan_age'].values.astype(float)
    event_codes = test_terminal['event_code'].values.astype(int)

    # Use CIF at evaluation horizons as risk scores
    results_rows = []
    for tau in TIME_HORIZONS:
        tau_idx = min(tau, args.cif_horizon)
        risk_prepay = CIF_pre[:, tau_idx]
        risk_default = CIF_def[:, tau_idx]

        # C-index for prepay
        c_prepay, _, _ = time_dependent_concordance_index(
            event_times, event_codes, risk_prepay, tau, event_of_interest=1
        )
        # C-index for default
        c_default, _, _ = time_dependent_concordance_index(
            event_times, event_codes, risk_default, tau, event_of_interest=2
        )

        results_rows.append({
            'Metric': f'C({tau})',
            'Prepay (k=1)': c_prepay,
            'Default (k=2)': c_default,
        })

        log(f"  C({tau}): Prepay={c_prepay:.4f}, Default={c_default:.4f}")

    results_df = pd.DataFrame(results_rows)

    # Add means
    mean_prepay = results_df['Prepay (k=1)'].mean()
    mean_default = results_df['Default (k=2)'].mean()
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Metric': 'mean_C',
        'Prepay (k=1)': mean_prepay,
        'Default (k=2)': mean_default,
    }])], ignore_index=True)
    results_df['Combined'] = (results_df['Prepay (k=1)'] + results_df['Default (k=2)']) / 2

    log(f"\n  Mean C-index: Prepay={mean_prepay:.4f}, Default={mean_default:.4f}, "
        f"Combined={(mean_prepay + mean_default)/2:.4f}")

    # Brier scores
    log("\n  Brier Scores:")
    for tau in TIME_HORIZONS:
        tau_idx = min(tau, args.cif_horizon)
        bs_prepay = brier_score_competing_risks(
            event_times, event_codes, CIF_pre[:, tau_idx], tau, event_of_interest=1
        )
        bs_default = brier_score_competing_risks(
            event_times, event_codes, CIF_def[:, tau_idx], tau, event_of_interest=2
        )
        log(f"    BS({tau}): Prepay={bs_prepay:.6f}, Default={bs_default:.6f}")

    # Plot C-index comparison
    plot_cindex_comparison(results_df, figures_dir)
    log(f"  Saved: {figures_dir / 'breeden_crook_cindex_comparison.png'}")

    # ==================== Summary ====================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\nModel: Breeden & Crook (2022) Multihorizon Logistic Regression")
    log(f"  Horizons: 0 (origination) + 1-{args.max_horizon}")
    log(f"  Total models: {2 * (args.max_horizon + 1)}")
    log(f"  Solver: {args.solver}")
    log(f"  Age representation: B-spline ({args.n_age_knots} knots)")

    log(f"\nData:")
    log(f"  Train: {len(train_panel):,} rows ({train_panel['loan_sequence_number'].nunique():,} loans)")
    log(f"  Val: {len(val_panel):,} rows ({val_panel['loan_sequence_number'].nunique():,} loans)")
    log(f"  Test: {len(test_terminal):,} loans")

    log(f"\nResults (Test Set):")
    for _, row in results_df.iterrows():
        log(f"  {row['Metric']}: Prepay={row['Prepay (k=1)']:.4f}, "
            f"Default={row['Default (k=2)']:.4f}, "
            f"Combined={row['Combined']:.4f}")

    total_time = time.time() - start_time
    log(f"\nTotal runtime: {total_time/60:.1f} minutes")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ==================== Save Artifacts ====================
    log("\nSaving artifacts...")

    # Models
    with open(models_dir / 'breeden_crook_models.pkl', 'wb') as f:
        pickle.dump(model, f)
    log(f"  Models: {models_dir / 'breeden_crook_models.pkl'}")

    # Scalers
    with open(models_dir / 'breeden_crook_scaler.pkl', 'wb') as f:
        pickle.dump(model.scalers, f)
    log(f"  Scalers: {models_dir / 'breeden_crook_scaler.pkl'}")

    # Coefficients
    coef_df.to_csv(models_dir / 'breeden_crook_coefficients.csv', index=False)
    log(f"  Coefficients: {models_dir / 'breeden_crook_coefficients.csv'}")

    # C-index results
    results_df.to_csv(output_dir / 'breeden_crook_cindex.csv', index=False)
    log(f"  C-index: {output_dir / 'breeden_crook_cindex.csv'}")

    # Text report
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"  Report: {results_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
