#!/usr/bin/env python3
"""
NN-DTSM + APC Decomposition — Supercomputer Training Script.

Extension of Wang et al. (2024) to competing risks (prepayment and default).
Trains per-vintage neural network subnetworks, computes CIF, performs APC
decomposition, and evaluates on held-out test data.

Usage (local, development):
    python scripts/run_nn_dtsm.py --use-sampled-panel --n-epochs 10

Usage (supercomputer, full data):
    python scripts/run_nn_dtsm.py --n-epochs 30 --n-neurons 8 --n-hidden 4

Output:
    - results/nn_dtsm_results.txt          (summary report)
    - models/nn_dtsm_model.pt              (all vintage subnetworks)
    - models/nn_dtsm_cindex.csv            (C-index at eval horizons)
    - models/nn_dtsm_brier.csv             (Brier scores)
    - reports/figures/nn_dtsm_*.png         (APC plots, Lexis graphs)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from competing_risks.nn_dtsm import (
    VintageNNDTSM,
    APCDecomposition,
    prepare_panel_features,
    extract_vintage_quarter,
    get_device,
    STATIC_FEATURES,
    BEHAVIORAL_FEATURES,
    MACRO_FEATURES,
    DEFAULT_INPUT_FEATURES,
    TRAIN_FOLDS,
    VAL_FOLDS,
    TEST_FOLD,
)
from competing_risks.evaluation import (
    time_dependent_concordance_index,
    brier_score_competing_risks,
    EVAL_TIMES,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='NN-DTSM + APC (Wang et al. 2024) — Competing Risks')

    # Data source
    p.add_argument('--use-sampled-panel', action='store_true',
                   help='Use sampled panel (109K loans) for development')
    p.add_argument('--data-dir', type=str, default='data/processed')
    p.add_argument('--output-dir', type=str, default='results')
    p.add_argument('--models-dir', type=str, default='models')
    p.add_argument('--figures-dir', type=str, default='reports/figures')

    # Architecture
    p.add_argument('--n-hidden', type=int, default=4,
                   help='Hidden layers per subnetwork (default: 4)')
    p.add_argument('--n-neurons', type=int, default=8,
                   help='Neurons per hidden layer (default: 8)')
    p.add_argument('--dropout', type=float, default=0.0,
                   help='Dropout rate (default: 0.0)')

    # Training
    p.add_argument('--n-epochs', type=int, default=20,
                   help='Epochs per subnetwork (default: 20)')
    p.add_argument('--batch-size', type=int, default=256,
                   help='Mini-batch size (default: 256)')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate (default: 1e-3)')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='L2 regularisation (default: 1e-4)')
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience (default: 10)')
    p.add_argument('--no-class-weights', action='store_true',
                   help='Disable class-weighted loss')

    # Evaluation
    p.add_argument('--eval-times', type=int, nargs='+', default=[24, 48, 72],
                   help='CIF evaluation horizons (default: 24 48 72)')

    # APC
    p.add_argument('--skip-apc', action='store_true',
                   help='Skip APC decomposition')
    p.add_argument('--ar-max-lag', type=int, default=4,
                   help='Max AR lag for macro projection (default: 4)')
    p.add_argument('--n-mc-paths', type=int, default=500,
                   help='Monte Carlo paths for AR projection (default: 500)')

    # Device
    p.add_argument('--device', type=str, default='auto',
                   choices=['auto', 'cuda', 'mps', 'cpu'])

    # Misc
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


# =============================================================================
# Data loading
# =============================================================================

def load_full_vintage_data(data_dir: Path, log_fn=print) -> pd.DataFrame:
    """
    Load all vintage parquet files and concatenate into a single DataFrame.

    The by-vintage files have loan-level survival data (one row per loan).
    This function loads them all and adds vintage_quarter.
    """
    vintage_dir = data_dir / 'by_vintage'
    if not vintage_dir.exists():
        raise FileNotFoundError(f"Vintage directory not found: {vintage_dir}")

    dfs = []
    for vdir in sorted(vintage_dir.iterdir()):
        if not vdir.is_dir():
            continue
        parquet_files = list(vdir.glob('*.parquet'))
        if not parquet_files:
            continue
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No parquet files found in {vintage_dir}")

    full = pd.concat(dfs, ignore_index=True)
    log_fn(f"  Loaded {len(full):,} loans from {len(dfs)} vintage files")

    # Extract vintage quarter
    if 'loan_sequence_number' in full.columns:
        full['vintage_quarter'] = extract_vintage_quarter(
            full['loan_sequence_number'])

    return full


def load_sampled_panel(data_dir: Path, log_fn=print) -> pd.DataFrame:
    """Load the sampled loan-month panel for development."""
    panel = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    panel = prepare_panel_features(panel)
    log_fn(f"  Loaded sampled panel: {panel.shape[0]:,} rows, "
           f"{panel['loan_sequence_number'].nunique():,} loans")
    return panel


# =============================================================================
# Plotting
# =============================================================================

def plot_lexis_heatmap(lexis_df, cause_name, save_path, log_fn=print):
    """Plot Lexis heatmap (age x vintage) for one cause."""
    pivot = lexis_df.pivot_table(
        values='mean_prob', index='loan_age',
        columns='vintage_quarter', aggfunc='mean',
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.pcolormesh(
        range(pivot.shape[1]), pivot.index, pivot.values,
        shading='auto', cmap='YlOrRd',
    )
    plt.colorbar(im, ax=ax, label=f'P({cause_name})')

    n_cols = pivot.shape[1]
    step = max(1, n_cols // 15)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_xticklabels([str(pivot.columns[i]) for i in range(0, n_cols, step)],
                       rotation=45, ha='right')
    ax.set_xlabel('Vintage Quarter')
    ax.set_ylabel('Loan Age (months)')
    ax.set_title(f'Lexis Graph: {cause_name} Hazard Rate')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_fn(f"  Saved {save_path}")


def plot_apc_effects(apc_prepay, apc_default, save_path, log_fn=print):
    """Plot side-by-side APC effects for prepayment and default."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    apc_prepay.plot_effects(title_prefix='Prepay', axes=axes[0])
    apc_default.plot_effects(title_prefix='Default', axes=axes[1])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_fn(f"  Saved {save_path}")


def plot_ar_projection(proj_df, cause_name, save_path, log_fn=print):
    """Plot AR-projected calendar-time effect with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 5))

    months = proj_df['month_ahead']
    ax.plot(months, proj_df['gamma_mean'], 'b-', linewidth=2, label='Mean')
    ax.fill_between(months, proj_df['gamma_lo'], proj_df['gamma_hi'],
                    alpha=0.3, color='steelblue', label='90% CI')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Months Ahead')
    ax.set_ylabel('γ(c*)')
    ax.set_title(f'{cause_name}: AR-Projected Calendar-Time Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_fn(f"  Saved {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    models_dir = base_dir / args.models_dir
    figures_dir = base_dir / args.figures_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("NN-DTSM + APC DECOMPOSITION — COMPETING RISKS")
    log("Wang et al. (2024), extended to prepayment + default")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch:    {torch.__version__}")
    log(f"Device:     {get_device(args.device)}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log(f"\nArchitecture:")
    log(f"  n_hidden:     {args.n_hidden}")
    log(f"  n_neurons:    {args.n_neurons}")
    log(f"  dropout:      {args.dropout}")
    log(f"  n_epochs:     {args.n_epochs}")
    log(f"  batch_size:   {args.batch_size}")
    log(f"  lr:           {args.lr}")
    log(f"  class_weights: {not args.no_class_weights}")
    log(f"  eval_times:   {args.eval_times}")
    log(f"  seed:         {args.seed}")

    # ── Load data ──────────────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Loading data...")

    if args.use_sampled_panel:
        panel = load_sampled_panel(data_dir, log_fn=log)
    else:
        # Full vintage data — for now, use the sampled panel
        # TODO: implement full vintage panel expansion
        log("  Full vintage data loading not yet implemented — using sampled panel")
        panel = load_sampled_panel(data_dir, log_fn=log)

    # ── Train / test split ─────────────────────────────────────────────────
    train_panel = panel[panel['fold'].isin(TRAIN_FOLDS)].copy()
    test_panel = panel[panel['fold'] == TEST_FOLD].copy()

    log(f"\n  Train: {train_panel['loan_sequence_number'].nunique():,} loans, "
        f"{train_panel.shape[0]:,} rows")
    log(f"  Test:  {test_panel['loan_sequence_number'].nunique():,} loans, "
        f"{test_panel.shape[0]:,} rows")

    # Event distribution
    evt_names = {0: 'Censored', 1: 'Prepay', 2: 'Default'}
    for subset_name, subset in [('Train', train_panel), ('Test', test_panel)]:
        evt_dist = (subset.groupby('loan_sequence_number')['event_code']
                    .last().value_counts().sort_index())
        log(f"\n  {subset_name} event distribution:")
        for code, count in evt_dist.items():
            log(f"    {evt_names.get(code, f'k={code}')}: {count:,}")

    # Determine available input features
    input_features = [c for c in DEFAULT_INPUT_FEATURES if c in train_panel.columns]
    log(f"\n  Input features ({len(input_features)}): {input_features}")

    # ── Train NN-DTSM ──────────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Training vintage subnetworks...")

    train_start = time.time()

    model = VintageNNDTSM(
        n_hidden_layers=args.n_hidden,
        n_neurons=args.n_neurons,
        dropout=args.dropout,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_class_weights=not args.no_class_weights,
        device=args.device,
        random_seed=args.seed,
    )

    model.fit(train_panel, input_features=input_features, log_fn=log)

    train_time = time.time() - train_start
    log(f"\n  Training completed in {train_time / 60:.1f} minutes")
    log(f"  Trained {len(model.subnets_)} subnetworks")

    # Save model
    model_path = models_dir / 'nn_dtsm_model.pt'
    model.save(str(model_path))
    log(f"  Saved model to {model_path}")

    # ── McFadden pseudo-R² ─────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("McFadden pseudo-R² per vintage:")

    r2_dict = model.mcfadden_pseudo_r2(test_panel)
    r2_df = pd.DataFrame([
        {'vintage': vq, 'pseudo_r2': r2}
        for vq, r2 in sorted(r2_dict.items())
    ])

    if len(r2_df) > 0:
        log(f"  Mean R²: {r2_df['pseudo_r2'].mean():.4f}")
        log(f"  Range:   [{r2_df['pseudo_r2'].min():.4f}, "
            f"{r2_df['pseudo_r2'].max():.4f}]")
        for _, row in r2_df.iterrows():
            log(f"    {row['vintage']}: {row['pseudo_r2']:.4f}")

    # ── CIF prediction on test set ─────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Computing CIF predictions on test set...")

    cif_start = time.time()
    cif_result = model.predict_cif(
        test_panel, eval_times=args.eval_times)
    cif_time = time.time() - cif_start

    log(f"  CIF prediction completed in {cif_time / 60:.1f} minutes")
    log(f"  Test loans: {len(cif_result['loan_sequence_number']):,}")

    for t in args.eval_times:
        cif_p = cif_result[f'cif_prepay_{t}']
        cif_d = cif_result[f'cif_default_{t}']
        log(f"  t={t}: CIF_prepay={cif_p.mean():.4f}, "
            f"CIF_default={cif_d.mean():.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    log("\n" + "-" * 70)
    log("Evaluation: Time-dependent C-index and Brier Score")

    event_times = cif_result['duration']
    event_codes = cif_result['event_code']

    cindex_rows = []
    brier_rows = []

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

        cindex_rows.append({
            'Horizon': t, 'C_prepay': c_prepay, 'C_default': c_default})
        brier_rows.append({
            'Horizon': t, 'BS_prepay': bs_prepay, 'BS_default': bs_default})

        log(f"  t={t:<7} {c_prepay:>10.4f} {c_default:>10.4f} "
            f"{bs_prepay:>10.4f} {bs_default:>10.4f}")

    # Save evaluation tables
    cindex_df = pd.DataFrame(cindex_rows)
    brier_df = pd.DataFrame(brier_rows)
    cindex_df.to_csv(models_dir / 'nn_dtsm_cindex.csv', index=False)
    brier_df.to_csv(models_dir / 'nn_dtsm_brier.csv', index=False)
    log(f"\n  Saved C-index to {models_dir / 'nn_dtsm_cindex.csv'}")
    log(f"  Saved Brier scores to {models_dir / 'nn_dtsm_brier.csv'}")

    # ── APC Decomposition ──────────────────────────────────────────────────
    if not args.skip_apc:
        log("\n" + "-" * 70)
        log("APC Decomposition...")

        # Build Lexis data from training predictions
        lexis = model.build_lexis_data(train_panel)

        apc_results = {}
        for cause, cause_name in [(1, 'Prepay'), (2, 'Default')]:
            log(f"\n  --- {cause_name} ---")
            lexis_df = lexis[cause]

            if len(lexis_df) < 10:
                log(f"  Skipping {cause_name}: too few Lexis cells ({len(lexis_df)})")
                continue

            # Lexis heatmap
            plot_lexis_heatmap(
                lexis_df, cause_name,
                figures_dir / f'nn_dtsm_lexis_{cause_name.lower()}.png',
                log_fn=log)

            # Ridge regression APC
            apc = APCDecomposition()
            apc.fit(lexis_df, log_fn=log)
            apc_results[cause] = apc

            effects = apc.get_effects()
            log(f"  Age effect range: [{effects['age'].min():.6f}, "
                f"{effects['age'].max():.6f}]")
            log(f"  Vintage effect range: [{effects['vintage'].min():.6f}, "
                f"{effects['vintage'].max():.6f}]")
            log(f"  Calendar effect range: [{effects['calendar'].min():.6f}, "
                f"{effects['calendar'].max():.6f}]")

            # Macro regression
            macro_avail = [c for c in MACRO_FEATURES if c in train_panel.columns]
            if macro_avail:
                try:
                    apc.fit_macro_regression(
                        train_panel, macro_cols=macro_avail, log_fn=log)
                except Exception as e:
                    log(f"  Macro regression failed: {e}")

            # AR models for projection
            if macro_avail and apc.macro_regression_ is not None:
                try:
                    apc.fit_ar_models(
                        train_panel, macro_cols=macro_avail,
                        max_lag=args.ar_max_lag, log_fn=log)

                    # Project forward
                    proj = apc.project_calendar_effect(
                        n_months_ahead=max(args.eval_times),
                        n_mc_paths=args.n_mc_paths,
                        seed=args.seed)

                    plot_ar_projection(
                        proj, cause_name,
                        figures_dir / f'nn_dtsm_ar_proj_{cause_name.lower()}.png',
                        log_fn=log)
                except Exception as e:
                    log(f"  AR projection failed: {e}")

        # Side-by-side APC plots
        if 1 in apc_results and 2 in apc_results:
            plot_apc_effects(
                apc_results[1], apc_results[2],
                figures_dir / 'nn_dtsm_apc_effects.png',
                log_fn=log)

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    log("\n" + "=" * 70)
    log(f"COMPLETED in {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Write results file
    results_file = output_dir / 'nn_dtsm_results.txt'
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"\nResults written to {results_file}")


if __name__ == '__main__':
    main()
