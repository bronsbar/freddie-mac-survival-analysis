#!/usr/bin/env python3
"""
Kundig & Sigrist (2025) — competing-risks adaptation, expanding-window training.

Trains five model variants (linear_independent, linear_spatial,
linear_spatio_temporal, lagaboost_spatial, lagaboost_spatio_temporal) on the
yearly panel built by notebook 22a, doing one-year-ahead default + prepay
predictions in an expanding window.

Per test year and per cause we report AUC, H-measure (approximated by AUC for
balanced classes), log-loss, Brier score and ECE, plus the GP covariance
parameters for the frailty variants.

Usage
-----
    python scripts/run_lagaboost_competing_risks.py \
        --first-test-year 2015 --last-test-year 2024 \
        --variants linear_independent linear_spatial lagaboost_spatio_temporal

    python scripts/run_lagaboost_competing_risks.py --quick   # short demo run
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Ensure single OMP thread before gpboost native lib loads
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

from src.competing_risks.spacetime_ml import CompetingRisksLaGaBoost, VARIANTS
from src.data.kundig_sigrist_panel import SNAPSHOT_FEATURES


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS),
                   choices=list(VARIANTS), help='Variants to train.')
    p.add_argument('--first-test-year', type=int, default=2015)
    p.add_argument('--last-test-year', type=int, default=2024)
    p.add_argument('--num-boost-round', type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=0.1)
    p.add_argument('--max-depth', type=int, default=5)
    p.add_argument('--min-data-in-leaf', type=int, default=100)
    p.add_argument('--lambda-l2', type=float, default=1.0)
    p.add_argument('--num-neighbors', type=int, default=20)
    p.add_argument('--max-iter', type=int, default=100,
                   help='GPModel optimisation iterations (linear variants)')
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--output-dir', default='results/kundig_sigrist')
    p.add_argument('--models-dir', default='models/kundig_sigrist')
    p.add_argument('--figures-dir', default='reports/figures/kundig_sigrist')
    p.add_argument('--quick', action='store_true',
                   help='Tiny grid for smoke testing.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ============================================================
# Metrics
# ============================================================
def expected_calibration_error(y_true, p, n_bins=20):
    """Equally-spaced-quantile ECE matching paper's definition."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9)
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    bins = np.digitize(p, quantiles) - 1
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        avg_p = p[mask].mean()
        avg_y = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_p - avg_y)
    return ece


def evaluate(y_true, p):
    """Returns dict with AUC, log-loss, Brier, ECE."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {'AUC': float('nan'), 'log_loss': float('nan'),
                'Brier': float('nan'), 'ECE': float('nan'), 'n_events': int(y_true.sum())}
    return {
        'AUC': roc_auc_score(y_true, p),
        'log_loss': log_loss(y_true, p, labels=[0, 1]),
        'Brier': brier_score_loss(y_true, p),
        'ECE': expected_calibration_error(y_true, p),
        'n_events': int(y_true.sum()),
    }


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    if args.quick:
        args.num_boost_round = 30
        args.max_iter = 30

    base = Path(__file__).parent.parent
    data_dir = base / args.data_dir
    out_dir = base / args.output_dir
    models_dir = base / args.models_dir
    figures_dir = base / args.figures_dir
    for d in (out_dir, models_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_lines = []
    def log(msg=''):
        print(msg, flush=True)
        log_lines.append(msg)

    np.random.seed(args.seed)
    t_start = time.time()

    log('=' * 78)
    log('KUNDIG-SIGRIST COMPETING RISKS')
    log(f'  start: {datetime.now():%Y-%m-%d %H:%M:%S}')
    log(f'  variants: {args.variants}')
    log(f'  test years: {args.first_test_year}..{args.last_test_year}')
    log(f'  hyperparams: lr={args.learning_rate}, depth={args.max_depth}, '
        f'min_leaf={args.min_data_in_leaf}, l2={args.lambda_l2}, '
        f'rounds={args.num_boost_round}, neighbors={args.num_neighbors}')

    # Load and scale
    log('\nLoading panel...')
    panel = pd.read_parquet(data_dir / 'kundig_sigrist_yearly_panel.parquet')
    features = SNAPSHOT_FEATURES + ['n_months', 'ir_spread', 'lat', 'lon']
    panel = panel.dropna(subset=features + ['default_in_year', 'prepay_in_year']).copy()
    for c in features:
        panel[c] = panel[c].astype(float)
    log(f'  rows: {len(panel):,}, loans: {panel["loan_sequence_number"].nunique():,}, '
        f'years: {panel["year"].min()}..{panel["year"].max()}')

    scale_cols = [c for c in features if c not in ('lat', 'lon')]

    # Expanding-window loop
    rows = []
    cov_rows = []

    for test_year in range(args.first_test_year, args.last_test_year + 1):
        log(f'\n--- Test year {test_year} ---')
        train = panel[panel['year'] < test_year].copy()
        test = panel[panel['year'] == test_year].copy()
        if len(test) == 0:
            log(f'  (no rows in test year {test_year}, skipping)')
            continue
        log(f'  train: n={len(train):,}, prepay={int(train["prepay_in_year"].sum()):,}, '
            f'default={int(train["default_in_year"].sum()):,}')
        log(f'  test : n={len(test):,}, prepay={int(test["prepay_in_year"].sum()):,}, '
            f'default={int(test["default_in_year"].sum()):,}')

        # Scale on train, apply to test
        sc = StandardScaler()
        train[scale_cols] = sc.fit_transform(train[scale_cols])
        test[scale_cols] = sc.transform(test[scale_cols])

        for variant in args.variants:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = CompetingRisksLaGaBoost(
                    variant=variant,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    min_data_in_leaf=args.min_data_in_leaf,
                    lambda_l2=args.lambda_l2,
                    num_boost_round=args.num_boost_round,
                    num_neighbors=args.num_neighbors,
                    max_iter=args.max_iter,
                    random_state=args.seed,
                    verbose=False,
                )
                model.fit(train, features)
                pred = model.predict_proba(test)
            elapsed = time.time() - t0

            for cause, label in [('prepay', 'prepay_in_year'),
                                  ('default', 'default_in_year')]:
                metrics = evaluate(test[label].values, pred[cause])
                row = {
                    'variant': variant,
                    'test_year': test_year,
                    'cause': cause,
                    'fit_seconds': elapsed,
                    **metrics,
                }
                rows.append(row)

            log(f'    {variant:30s}  fit={elapsed:6.1f}s  '
                f'AUC_p={rows[-2]["AUC"]:.4f}  AUC_d={rows[-1]["AUC"]:.4f}  '
                f'logloss_d={rows[-1]["log_loss"]:.5f}')

            # GP covariance parameters
            cov = model.get_cov_pars()
            for cause, df in cov.items():
                if df is None:
                    continue
                rec = df.iloc[0].to_dict()
                rec.update({'variant': variant, 'test_year': test_year, 'cause': cause})
                cov_rows.append(rec)

    metrics_df = pd.DataFrame(rows)
    cov_df = pd.DataFrame(cov_rows) if cov_rows else pd.DataFrame()

    metrics_csv = out_dir / 'lagaboost_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    log(f'\nSaved metrics: {metrics_csv}')
    if len(cov_df):
        cov_csv = out_dir / 'lagaboost_cov_pars.csv'
        cov_df.to_csv(cov_csv, index=False)
        log(f'Saved cov pars: {cov_csv}')

    # ==================== Plots ====================
    log('\nPlotting...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cause in zip(axes, ['prepay', 'default']):
        sub = metrics_df[metrics_df['cause'] == cause]
        for v in args.variants:
            s = sub[sub['variant'] == v]
            ax.plot(s['test_year'], s['AUC'], marker='o', label=v)
        ax.set_title(f'{cause.capitalize()}: AUC by test year')
        ax.set_xlabel('Test year')
        ax.set_ylabel('AUC')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    fig_path = figures_dir / 'lagaboost_auc_by_year.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  {fig_path}')

    # Mean-by-variant summary
    summary = (
        metrics_df.groupby(['variant', 'cause'])
        [['AUC', 'log_loss', 'Brier', 'ECE']]
        .mean()
        .round(4)
    )
    log('\n=== Mean metrics across test years ===')
    log(summary.to_string())

    summary.to_csv(out_dir / 'lagaboost_summary.csv')

    log(f'\nTotal runtime: {(time.time() - t_start) / 60:.1f} min')

    # Save text report
    (out_dir / 'lagaboost_results.txt').write_text('\n'.join(log_lines))


if __name__ == '__main__':
    main()
