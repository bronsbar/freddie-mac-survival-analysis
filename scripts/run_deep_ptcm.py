#!/usr/bin/env python3
"""
Deep-PTCM Competing Risks - Supercomputer Script

Implements the Deep Promotion Time Cure Model (Medina-Olivares et al., 2024)
extended to competing risks (prepayment + default).

Trains four variants:
  1. Deep-PTCM            (head_layers=[], paper architecture)
  2. Deep-PTCM-Ort        (orthogonalized, head_layers=[])
  3. Deep-PTCM-H32        (head_layers=[32], deeper cause-specific heads)
  4. Deep-PTCM-Ort-H32    (orthogonalized, head_layers=[32])

Usage:
    python run_deep_ptcm.py [--epochs 200] [--batch-size 512] [--lr 0.01]

Output:
    - results/deep_ptcm_results.txt              (summary report)
    - models/deep_ptcm.pt                         (model checkpoint)
    - models/deep_ptcm_ort.pt
    - models/deep_ptcm_h32.pt
    - models/deep_ptcm_ort_h32.pt
    - reports/figures/deep_ptcm_training_curves.png
    - reports/figures/deep_ptcm_cure_fractions.png
    - reports/figures/deep_ptcm_cindex_comparison.png
    - reports/figures/deep_ptcm_survival_curves.png
    - reports/figures/deep_ptcm_feature_importance.png
    - reports/figures/deep_ptcm_ort_coefficients.png
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.competing_risks.deep_ptcm import (
    CompetingRisksDeepPTCM,
    fit_deep_ptcm_competing_risks,
    STATIC_FEATURES,
    TRAIN_FOLDS,
    VAL_FOLDS,
    TEST_FOLD,
    get_device,
)
from src.competing_risks.evaluation import (
    time_dependent_concordance_index,
    brier_score_competing_risks,
    auc_cure,
    integrated_brier_score,
    EVAL_TIMES,
)


# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep-PTCM Competing Risks (Medina-Olivares et al. 2024)')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate (SGD)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--lr-decay-rate', type=float, default=0.75,
                        help='Inverse time decay rate')
    parser.add_argument('--lr-decay-steps', type=int, default=100,
                        help='Inverse time decay steps')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')

    # Architecture
    parser.add_argument('--shared-layers', type=int, nargs='+', default=[512, 512],
                        help='Shared hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--num-intervals', type=int, default=15,
                        help='Piecewise exponential intervals per cause')

    # Evaluation
    parser.add_argument('--importance-repeats', type=int, default=5,
                        help='Repeats for permutation importance')
    parser.add_argument('--cure-min-followup', type=int, default=120,
                        help='Min follow-up months for AUC_cure')

    # Variants to skip
    parser.add_argument('--skip-ort', action='store_true',
                        help='Skip orthogonalized variants')
    parser.add_argument('--skip-h32', action='store_true',
                        help='Skip head_layers=[32] variants')

    # General
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Models directory')
    parser.add_argument('--figures-dir', type=str, default='reports/figures',
                        help='Figures directory')

    return parser.parse_args()


FEATURE_COLS = ['int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r']
EVENT_NAMES = {0: 'Censored', 1: 'Prepay', 2: 'Default'}


# ==============================================================================
# Evaluation helpers
# ==============================================================================

def evaluate_model(model, X_test, test_durations, test_events, log_fn=print):
    """Compute C-index, Brier score, IBS, and AUC_cure for a model."""
    results_rows = []
    for event_code, event_name in [(1, 'Prepay'), (2, 'Default')]:
        for t in EVAL_TIMES:
            risk = model.predict_risk(X_test, event=event_code, time=t)
            c_idx, _, n_comp = time_dependent_concordance_index(
                test_durations, test_events, risk, t, event_code
            )
            cif_t = model.predict_cumulative_incidence(
                X_test, event=event_code, times=np.array([t])
            )[:, 0]
            bs = brier_score_competing_risks(
                test_durations, test_events, cif_t, t, event_code
            )
            log_fn(f"    {event_name} t={t:3d}: C={c_idx:.4f}  BS={bs:.6f}  (n={n_comp:,})")
            results_rows.append({
                'Event': event_name, 'Horizon': t,
                'C-index': c_idx, 'Brier Score': bs,
            })

    # IBS
    ibs_grid = np.arange(1, 73, dtype='float32')
    for event_code, event_name in [(1, 'Prepay'), (2, 'Default')]:
        cif_at_grid = model.predict_cumulative_incidence(
            X_test, event=event_code, times=ibs_grid
        )
        ibs = integrated_brier_score(
            test_durations, test_events,
            cif_at_grid, ibs_grid,
            event_of_interest=event_code,
        )
        log_fn(f"    IBS ({event_name}, t=1..72): {ibs:.6f}")
        results_rows.append({
            'Event': event_name, 'Horizon': 'IBS',
            'C-index': np.nan, 'Brier Score': ibs,
        })

    # AUC_cure
    cure = model.predict_cure_fraction(X_test)
    auc_val = auc_cure(
        test_events, test_durations, cure['overall'], min_followup=120
    )
    log_fn(f"    AUC_cure (overall): {auc_val:.4f}")

    return pd.DataFrame(results_rows), auc_val, cure


def permutation_importance(model, X_df, durations, events, feature_cols,
                           event_code, eval_time=72, n_repeats=5, seed=42):
    """Compute permutation importance for each feature."""
    rng = np.random.RandomState(seed)

    base_risk = model.predict_risk(X_df, event=event_code, time=eval_time)
    base_c, _, _ = time_dependent_concordance_index(
        durations, events, base_risk, eval_time, event_code
    )

    importances = {}
    for feat in feature_cols:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_df.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            perm_risk = model.predict_risk(X_perm, event=event_code, time=eval_time)
            perm_c, _, _ = time_dependent_concordance_index(
                durations, events, perm_risk, eval_time, event_code
            )
            drops.append(base_c - perm_c)
        importances[feat] = (np.mean(drops), np.std(drops))

    return pd.DataFrame([
        {'feature': f, 'importance': v[0], 'std': v[1]}
        for f, v in importances.items()
    ]).sort_values('importance', ascending=False)


# ==============================================================================
# Plotting
# ==============================================================================

def plot_training_curves(all_histories, figures_dir):
    """Plot training curves for all model variants."""
    n = len(all_histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, history) in zip(axes, all_histories.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label='Train')
        ax.plot(epochs, history['val_loss'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NLL')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'deep_ptcm_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cure_fractions(cure_dict, figures_dir):
    """Plot cure fraction distributions for the base model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, values) in zip(axes, cure_dict.items()):
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'Cure fraction: {name}')
        ax.set_xlabel(f'pi_{name}(x)')
        ax.set_ylabel('Count')
        ax.axvline(values.mean(), color='red', linestyle='--',
                   label=f'mean={values.mean():.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / 'deep_ptcm_cure_fractions.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cindex_comparison(all_results, figures_dir):
    """Bar chart comparing C-index across all variants."""
    # Filter to numeric horizons only
    rows = []
    for model_name, (res_df, _, _) in all_results.items():
        for _, row in res_df.iterrows():
            if row['Horizon'] != 'IBS':
                rows.append({
                    'Model': model_name,
                    'Event': row['Event'],
                    'Horizon': int(row['Horizon']),
                    'C-index': row['C-index'],
                })
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, event_name in zip(axes, ['Prepay', 'Default']):
        subset = plot_df[plot_df['Event'] == event_name]
        pivot = subset.pivot(index='Horizon', columns='Model', values='C-index')
        pivot.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title(f'{event_name}: C-index by Horizon')
        ax.set_xlabel('Horizon (months)')
        ax.set_ylabel('C-index')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(figures_dir / 'deep_ptcm_cindex_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_survival_curves(model, test_df, feature_cols, figures_dir):
    """Plot survival and CIF curves for example loans."""
    test_sorted = test_df.sort_values('fico_score')
    idx_low = test_sorted.index[len(test_sorted) // 10]
    idx_med = test_sorted.index[len(test_sorted) // 2]
    idx_high = test_sorted.index[9 * len(test_sorted) // 10]
    example_indices = [idx_low, idx_med, idx_high]
    example_labels = ['Low FICO', 'Mid FICO', 'High FICO']

    times_grid = np.linspace(1, 180, 180).astype('float32')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for idx, label in zip(example_indices, example_labels):
        X_ex = test_df.loc[[idx], feature_cols]
        surv = model.predict_survival(X_ex, times=times_grid)
        fico = test_df.loc[idx, 'fico_score']
        ax.plot(times_grid, surv[0], label=f'{label} (FICO={fico:.0f})')
    ax.set_xlabel('Months')
    ax.set_ylabel('S(t)')
    ax.set_title('Overall Survival')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax, event_code, title in [(axes[1], 1, 'Prepayment CIF'),
                                   (axes[2], 2, 'Default CIF')]:
        for idx, label in zip(example_indices, example_labels):
            X_ex = test_df.loc[[idx], feature_cols]
            cif = model.predict_cumulative_incidence(X_ex, event=event_code, times=times_grid)
            ax.plot(times_grid, cif[0], label=label)
        ax.set_xlabel('Months')
        ax.set_ylabel(f'CIF(t)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'deep_ptcm_survival_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance(imp_prepay_df, imp_default_df, figures_dir):
    """Plot permutation importance for both causes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, color, title in zip(axes,
        [imp_prepay_df, imp_default_df],
        ['steelblue', 'indianred'],
        ['Prepayment', 'Default'],
    ):
        ax.barh(df['feature'], df['importance'],
                xerr=df['std'], color=color, alpha=0.7, capsize=3)
        ax.set_xlabel('C-index drop (permutation)')
        ax.set_title(f'Deep-PTCM Feature Importance: {title}')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'deep_ptcm_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ort_coefficients(model_ort, figures_dir):
    """Plot orthogonalized linear coefficients."""
    coefs = model_ort.get_linear_coefficients()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (event_name, coef_df) in zip(axes, coefs.items()):
        plot_df = coef_df[coef_df['feature'] != '(intercept)'].copy()
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in plot_df['coefficient']]
        ax.barh(plot_df['feature'], plot_df['coefficient'], color=colors)
        ax.set_title(f'Linear coefficients: {event_name}')
        ax.set_xlabel('Coefficient (on log-theta scale)')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'deep_ptcm_ort_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    models_dir = base_dir / args.models_dir
    figures_dir = base_dir / args.figures_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'deep_ptcm_results.txt'

    # Logging
    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("DEEP-PTCM COMPETING RISKS")
    log("Medina-Olivares et al. (2024) extended to competing risks")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"Device: {get_device()}")

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Base hyperparameters
    base_params = dict(
        num_intervals=args.num_intervals,
        shared_layers=args.shared_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_steps=args.lr_decay_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
        random_state=args.seed,
    )

    log(f"\nConfiguration:")
    for k, v in base_params.items():
        log(f"  {k}: {v}")
    log(f"  skip_ort: {args.skip_ort}")
    log(f"  skip_h32: {args.skip_h32}")

    # ==================== Load Data ====================
    log("\n" + "-" * 70)
    log("Loading data...")

    df = pd.read_parquet(data_dir / 'blumenstock_dataset2.parquet')
    df['log_upb'] = np.log(df['orig_upb'].clip(lower=1).astype(float))

    log(f"  Loaded {len(df):,} loans")
    log(f"  Vintages: {df['vintage_year'].min()} - {df['vintage_year'].max()}")
    log(f"  Duration range: {df['duration'].min()} - {df['duration'].max()} months")

    log("\n  Event distribution:")
    for code, name in EVENT_NAMES.items():
        n = (df['event_code'] == code).sum()
        log(f"    {name} (k={code}): {n:,} ({100 * n / len(df):.1f}%)")

    # Drop NaN and split
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    log(f"\n  After dropping NaN: {len(df_clean):,} (dropped {len(df) - len(df_clean):,})")

    train_df = df_clean[df_clean['fold'].isin(TRAIN_FOLDS)].copy()
    val_df = df_clean[df_clean['fold'].isin(VAL_FOLDS)].copy()
    test_df = df_clean[df_clean['fold'] == TEST_FOLD].copy()

    log(f"  Train (folds {TRAIN_FOLDS}): {len(train_df):,}")
    log(f"  Val (fold {VAL_FOLDS}): {len(val_df):,}")
    log(f"  Test (fold {TEST_FOLD}): {len(test_df):,}")

    X_test = test_df[FEATURE_COLS]
    test_durations = test_df['duration'].values
    test_events = test_df['event_code'].values

    # ==================== Define Variants ====================
    variants = {
        'Deep-PTCM': {**base_params, 'head_layers': [], 'batch_norm': False,
                       'orthogonalize': False},
    }
    if not args.skip_ort:
        variants['Deep-PTCM-Ort'] = {
            **base_params, 'head_layers': [], 'batch_norm': False,
            'orthogonalize': True,
        }
    if not args.skip_h32:
        variants['Deep-PTCM-H32'] = {
            **base_params, 'head_layers': [32], 'batch_norm': False,
            'orthogonalize': False,
        }
    if not args.skip_ort and not args.skip_h32:
        variants['Deep-PTCM-Ort-H32'] = {
            **base_params, 'head_layers': [32], 'batch_norm': False,
            'orthogonalize': True,
        }

    # ==================== Train & Evaluate ====================
    trained_models = {}
    all_histories = {}
    all_results = {}  # model_name -> (results_df, auc_cure, cure_dict)

    for variant_name, params in variants.items():
        log("\n" + "-" * 70)
        log(f"Training {variant_name}...")
        log(f"  head_layers={params['head_layers']}, "
            f"orthogonalize={params['orthogonalize']}")

        t0 = time.time()
        model = fit_deep_ptcm_competing_risks(
            df=train_df,
            feature_cols=FEATURE_COLS,
            duration_col='duration',
            event_col='event_code',
            event_types=[1, 2],
            val_df=val_df,
            **params,
        )
        elapsed = time.time() - t0

        n_params = sum(p.numel() for p in model.network_.parameters())
        n_params += sum(p.numel() for p in model.baselines_.parameters())
        log(f"  Training time: {elapsed / 60:.1f} minutes")
        log(f"  Epochs: {len(model.history_['train_loss'])}")
        log(f"  Best val loss: {min(model.history_['val_loss']):.4f}")
        log(f"  Parameters: {n_params:,}")

        trained_models[variant_name] = model
        all_histories[variant_name] = model.history_

        # Evaluate
        log(f"\n  Evaluation ({variant_name}):")
        res_df, auc_val, cure = evaluate_model(
            model, X_test, test_durations, test_events, log_fn=log
        )
        all_results[variant_name] = (res_df, auc_val, cure)

    # ==================== Orthogonalized Coefficients ====================
    if not args.skip_ort:
        log("\n" + "-" * 70)
        log("Orthogonalized linear coefficients:")
        ort_model_name = 'Deep-PTCM-Ort'
        if ort_model_name in trained_models:
            coefs = trained_models[ort_model_name].get_linear_coefficients()
            for event_name, coef_df in coefs.items():
                log(f"\n  {event_name}:")
                for _, row in coef_df.iterrows():
                    log(f"    {row['feature']:>15s}: {row['coefficient']:+.4f}")

    # ==================== Permutation Importance ====================
    log("\n" + "-" * 70)
    log("Computing permutation importance (base Deep-PTCM)...")

    imp_start = time.time()
    imp_dfs = {}
    for event_code, event_name in [(1, 'Prepay'), (2, 'Default')]:
        imp_df = permutation_importance(
            trained_models['Deep-PTCM'], X_test, test_durations, test_events,
            FEATURE_COLS, event_code=event_code, n_repeats=args.importance_repeats,
        )
        imp_dfs[event_name] = imp_df
        log(f"\n  {event_name}:")
        for _, row in imp_df.iterrows():
            log(f"    {row['feature']:>15s}: {row['importance']:+.4f} +/- {row['std']:.4f}")

    log(f"  Importance computed in {(time.time() - imp_start) / 60:.1f} minutes")

    # ==================== Plots ====================
    log("\n" + "-" * 70)
    log("Generating plots...")

    plot_training_curves(all_histories, figures_dir)
    log(f"  {figures_dir / 'deep_ptcm_training_curves.png'}")

    # Cure fractions from base model
    _, _, base_cure = all_results['Deep-PTCM']
    plot_cure_fractions(base_cure, figures_dir)
    log(f"  {figures_dir / 'deep_ptcm_cure_fractions.png'}")

    plot_cindex_comparison(all_results, figures_dir)
    log(f"  {figures_dir / 'deep_ptcm_cindex_comparison.png'}")

    plot_survival_curves(trained_models['Deep-PTCM'], test_df, FEATURE_COLS, figures_dir)
    log(f"  {figures_dir / 'deep_ptcm_survival_curves.png'}")

    plot_importance(imp_dfs['Prepay'], imp_dfs['Default'], figures_dir)
    log(f"  {figures_dir / 'deep_ptcm_feature_importance.png'}")

    if not args.skip_ort and 'Deep-PTCM-Ort' in trained_models:
        plot_ort_coefficients(trained_models['Deep-PTCM-Ort'], figures_dir)
        log(f"  {figures_dir / 'deep_ptcm_ort_coefficients.png'}")

    # ==================== Summary ====================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\nArchitecture (paper defaults):")
    log(f"  Shared: {args.shared_layers}")
    log(f"  Dropout: {args.dropout}")
    log(f"  Optimizer: SGD (lr={args.lr}, inverse time decay "
        f"rate={args.lr_decay_rate}, steps={args.lr_decay_steps})")
    log(f"  Intervals: {args.num_intervals}")

    log(f"\nData:")
    log(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    log(f"  Features: {FEATURE_COLS}")

    log(f"\nResults (Test Set):")
    log(f"{'Model':<25s} | {'Prepay C':>9s} | {'Default C':>9s} | {'AUC_cure':>9s}")
    log("-" * 60)
    for model_name, (res_df, auc_val, _) in all_results.items():
        # Mean C-index across horizons
        c_df = res_df[res_df['Horizon'] != 'IBS']
        c_prepay = c_df[c_df['Event'] == 'Prepay']['C-index'].mean()
        c_default = c_df[c_df['Event'] == 'Default']['C-index'].mean()
        log(f"{model_name:<25s} | {c_prepay:>9.4f} | {c_default:>9.4f} | {auc_val:>9.4f}")

    total_time = time.time() - start_time
    log(f"\nTotal runtime: {total_time / 60:.1f} minutes")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ==================== Save ====================
    log("\nSaving artifacts...")

    # Model checkpoints
    save_names = {
        'Deep-PTCM': 'deep_ptcm.pt',
        'Deep-PTCM-Ort': 'deep_ptcm_ort.pt',
        'Deep-PTCM-H32': 'deep_ptcm_h32.pt',
        'Deep-PTCM-Ort-H32': 'deep_ptcm_ort_h32.pt',
    }
    for model_name, model_obj in trained_models.items():
        fname = save_names.get(model_name, f'deep_ptcm_{model_name}.pt')
        model_obj.save(str(models_dir / fname))
        log(f"  Model: {models_dir / fname}")

    # Importance CSVs
    for event_name, imp_df in imp_dfs.items():
        fname = f'deep_ptcm_importance_{event_name.lower()}.csv'
        imp_df.to_csv(models_dir / fname, index=False)
        log(f"  Importance: {models_dir / fname}")

    # Combined results CSV
    all_rows = []
    for model_name, (res_df, auc_val, _) in all_results.items():
        res_copy = res_df.copy()
        res_copy['Model'] = model_name
        res_copy['AUC_cure'] = auc_val
        all_rows.append(res_copy)
    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_df.to_csv(models_dir / 'deep_ptcm_all_results.csv', index=False)
    log(f"  Results: {models_dir / 'deep_ptcm_all_results.csv'}")

    # Text report
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"  Report: {results_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
