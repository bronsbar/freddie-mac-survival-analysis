#!/usr/bin/env python3
"""
Sadhwani et al. (2021) Deep Learning for Mortgage Risk — Training Script.

Trains the neural network ensemble on loan-month panel data.
Designed for both local (MPS/CPU) and GPU cluster (CUDA) execution.

Usage (local):
    python scripts/run_sadhwani_train.py

Usage (GPU cluster / SLURM):
    sbatch scripts/slurm_sadhwani.sh
    # or directly:
    python scripts/run_sadhwani_train.py --device cuda --batch-size 8192 \
        --n-epochs 200 --n-ensemble 8 --output-dir results

Output:
    - results/sadhwani_results.txt          (full report)
    - models/sadhwani_ensemble.pt           (ensemble state dicts)
    - models/sadhwani_scaler.pkl            (fitted StandardScaler)
    - models/sadhwani_cindex.csv            (C-index table)
    - models/sadhwani_brier.csv             (Brier score table)
"""

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from competing_risks.sadhwani_net import (
    SadhwaniNet,
    SadhwaniEnsemble,
    get_device,
    train_single_model,
    compute_cif,
    compute_cif_frozen,
    compute_cif_ar,
    fit_ar_models,
    variable_sensitivity,
    prepare_monthly_targets,
    prepare_features,
    ALL_FEATURES,
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
        description='Train Sadhwani et al. (2021) deep learning mortgage model')

    # Paths
    p.add_argument('--data-dir', type=str, default='data/processed')
    p.add_argument('--output-dir', type=str, default='results')
    p.add_argument('--models-dir', type=str, default='models')

    # Device
    p.add_argument('--device', type=str, default='auto',
                   choices=['auto', 'cuda', 'mps', 'cpu'],
                   help='Compute device (auto detects CUDA > MPS > CPU)')

    # Architecture
    p.add_argument('--hidden-sizes', type=int, nargs='+',
                   default=[200, 140, 140, 140, 140],
                   help='Neurons per hidden layer')
    p.add_argument('--dropout', type=float, default=0.5)

    # Training
    p.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='L2 penalty')
    p.add_argument('--lr-halflife', type=int, default=800,
                   help='Epochs until LR halves (Eq. 9)')
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--n-epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience')

    # Ensemble
    p.add_argument('--n-ensemble', type=int, default=8,
                   help='Number of ensemble members')
    p.add_argument('--seed', type=int, default=42)

    # Evaluation
    p.add_argument('--eval-times', type=int, nargs='+', default=[24, 48, 72])
    p.add_argument('--skip-cif', action='store_true',
                   help='Skip CIF computation (faster, no Brier/C-index)')

    # Depth comparison experiment
    p.add_argument('--depth-comparison', action='store_true',
                   help='Also train 0,1,3-layer models for comparison (Table 11)')

    return p.parse_args()


# =============================================================================
# Logging
# =============================================================================

_log_lines = []


def log(msg: str = ''):
    print(msg)
    _log_lines.append(msg)


def save_log(path: Path):
    path.write_text('\n'.join(_log_lines))


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    models_dir = project_root / args.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    log("=" * 70)
    log("SADHWANI ET AL. (2021) — DEEP LEARNING FOR MORTGAGE RISK")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"Device: {device}")
    if device.type == 'cuda':
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    log(f"\nArchitecture: {args.hidden_sizes}")
    log(f"Dropout: {args.dropout}")
    log(f"Ensemble size: {args.n_ensemble}")
    log(f"Batch size: {args.batch_size}")
    log(f"LR: {args.lr}, decay halflife: {args.lr_halflife}")
    log(f"Weight decay: {args.weight_decay}")
    log(f"Max epochs: {args.n_epochs}, patience: {args.patience}")
    log(f"Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("Loading data...")

    panel_df = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    log(f"  Loaded {len(panel_df):,} loan-months")
    log(f"  Unique loans: {panel_df['loan_sequence_number'].nunique():,}")

    # Prepare features
    feature_cols, panel_df = prepare_features(panel_df)
    log(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Monthly transition targets
    panel_df = prepare_monthly_targets(panel_df)
    log(f"  Target distribution (full panel):")
    for val, name in [(0, 'Current'), (1, 'Prepay'), (2, 'Default')]:
        n = (panel_df['target'] == val).sum()
        log(f"    {name} (target={val}): {n:,} ({100*n/len(panel_df):.2f}%)")

    # Drop rows with missing features
    panel_df = panel_df.dropna(subset=feature_cols)
    log(f"  After dropping NaN: {len(panel_df):,} loan-months")

    # Split by fold (loan-level folds → loan-month inherits)
    train_panel = panel_df[panel_df['fold'].isin(TRAIN_FOLDS)].copy()
    val_panel = panel_df[panel_df['fold'].isin(VAL_FOLDS)].copy()
    test_panel = panel_df[panel_df['fold'] == TEST_FOLD].copy()

    log(f"\n  Train: {len(train_panel):,} loan-months "
        f"({train_panel['loan_sequence_number'].nunique():,} loans)")
    log(f"  Val:   {len(val_panel):,} loan-months "
        f"({val_panel['loan_sequence_number'].nunique():,} loans)")
    log(f"  Test:  {len(test_panel):,} loan-months "
        f"({test_panel['loan_sequence_number'].nunique():,} loans)")

    # Event distribution per split
    for name, split in [('Train', train_panel), ('Val', val_panel), ('Test', test_panel)]:
        terminal = split.groupby('loan_sequence_number').last()
        n_cens = (terminal['event_code'] == 0).sum()
        n_prep = (terminal['event_code'] == 1).sum()
        n_def = (terminal['event_code'] == 2).sum()
        log(f"  {name} events: censored={n_cens:,}, prepay={n_prep:,}, default={n_def:,}")

    # ------------------------------------------------------------------
    # Standardize
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("Standardizing features...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_panel[feature_cols].values).astype(np.float32)
    y_train = train_panel['target'].values.astype(np.int64)

    X_val = scaler.transform(val_panel[feature_cols].values).astype(np.float32)
    y_val = val_panel['target'].values.astype(np.int64)

    log(f"  X_train shape: {X_train.shape}")
    log(f"  X_val shape:   {X_val.shape}")
    log(f"  Class weights (train): "
        f"current={np.mean(y_train==0):.4f}, "
        f"prepay={np.mean(y_train==1):.4f}, "
        f"default={np.mean(y_train==2):.4f}")

    # Save scaler
    scaler_path = models_dir / 'sadhwani_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    log(f"  Scaler saved to {scaler_path}")

    # ------------------------------------------------------------------
    # Train ensemble
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log(f"Training ensemble ({args.n_ensemble} members)...")
    train_start = time.time()

    ensemble = SadhwaniEnsemble(
        n_models=args.n_ensemble,
        n_features=len(feature_cols),
        hidden_sizes=args.hidden_sizes,
        n_states=3,
        dropout=args.dropout,
    )

    histories = ensemble.fit(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_halflife=args.lr_halflife,
        patience=args.patience,
        device=device,
        base_seed=args.seed,
        verbose=True,
    )

    train_time = time.time() - train_start
    log(f"\n  Ensemble training completed in {train_time/60:.1f} minutes")

    # Report per-member final val loss
    for i, hist in enumerate(histories):
        best_val = min(hist['val_loss'])
        log(f"  Member {i+1}: best val_loss = {best_val:.6f} "
            f"(epoch {hist['val_loss'].index(best_val)+1})")

    # In-sample and out-of-sample cross-entropy
    train_probs = ensemble.predict_proba(X_train, device=device)
    val_probs = ensemble.predict_proba(X_val, device=device)
    train_ce = -np.mean(np.log(train_probs[np.arange(len(y_train)), y_train].clip(1e-10)))
    val_ce = -np.mean(np.log(val_probs[np.arange(len(y_val)), y_val].clip(1e-10)))
    log(f"\n  Cross-entropy (negative avg log-likelihood):")
    log(f"    Train: {train_ce:.6f}")
    log(f"    Val:   {val_ce:.6f}")

    # Save ensemble
    ensemble_path = models_dir / 'sadhwani_ensemble.pt'
    ensemble.save(str(ensemble_path))
    log(f"  Ensemble saved to {ensemble_path}")

    # ------------------------------------------------------------------
    # Depth comparison (optional, replicates Table 11)
    # ------------------------------------------------------------------
    if args.depth_comparison:
        log("\n" + "-" * 70)
        log("Depth comparison (Table 11 replication)...")

        depth_configs = {
            '0-Hidden (logit)': [],
            '1-Hidden': [200],
            '3-Hidden': [200, 140, 140],
            '5-Hidden': [200, 140, 140, 140, 140],
        }

        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        depth_results = []
        for name, hsizes in depth_configs.items():
            log(f"\n  Training {name}...")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            net = SadhwaniNet(
                n_features=len(feature_cols),
                hidden_sizes=hsizes,
                n_states=3,
                dropout=args.dropout if hsizes else 0.0,
            )
            train_ds = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True)
            hist = train_single_model(
                net, train_loader, val_loader,
                lr=args.lr, weight_decay=args.weight_decay,
                n_epochs=args.n_epochs, lr_halflife=args.lr_halflife,
                patience=args.patience, device=device, verbose=False,
            )

            # w/o dropout version
            net_nodrop = SadhwaniNet(
                n_features=len(feature_cols),
                hidden_sizes=hsizes,
                n_states=3,
                dropout=0.0,
            )
            torch.manual_seed(args.seed)
            train_ds2 = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
            train_loader2 = DataLoader(train_ds2, batch_size=args.batch_size,
                                       shuffle=True)
            hist_nd = train_single_model(
                net_nodrop, train_loader2, val_loader,
                lr=args.lr, weight_decay=args.weight_decay,
                n_epochs=args.n_epochs, lr_halflife=args.lr_halflife,
                patience=args.patience, device=device, verbose=False,
            )

            depth_results.append({
                'Model': name,
                'In-sample loss (w/o dropout)': min(hist_nd['train_loss']),
                'OOS loss (w/o dropout)': min(hist_nd['val_loss']),
                'OOS loss (with dropout)': min(hist['val_loss']),
            })

        depth_df = pd.DataFrame(depth_results)
        log("\n  Table 11 — Cross-entropy by depth:")
        log(depth_df.to_string(index=False))

    # ------------------------------------------------------------------
    # CIF + Evaluation (three methods)
    # ------------------------------------------------------------------
    if not args.skip_cif:
        log("\n" + "-" * 70)
        log("Computing CIF on test set (three methods)...")

        def eval_cif(cif_result, method_name):
            """Log C-index and Brier for a CIF result."""
            log(f"\n  --- {method_name} ---")
            log(f"  {'Cause':<12} " + "  ".join(f"C({t})" for t in args.eval_times)
                + "   " + "  ".join(f"BS({t})" for t in args.eval_times))
            log("  " + "-" * 70)
            rows = []
            for cause, code in [('Prepay', 1), ('Default', 2)]:
                row = {'Method': method_name, 'Cause': cause}
                c_vals, b_vals = [], []
                for t in args.eval_times:
                    cif_key = f'cif_{cause.lower()}_{t}'
                    c_idx, _, _ = time_dependent_concordance_index(
                        cif_result['duration'], cif_result['event_code'],
                        cif_result[cif_key], t, event_of_interest=code,
                    )
                    bs = brier_score_competing_risks(
                        cif_result['duration'], cif_result['event_code'],
                        cif_result[cif_key], t, event_of_interest=code,
                    )
                    row[f'C({t})'] = c_idx
                    row[f'BS({t})'] = bs
                    c_vals.append(f"{c_idx:.4f}" if not np.isnan(c_idx) else "NaN   ")
                    b_vals.append(f"{bs:.6f}")
                rows.append(row)
                log(f"  {cause:<12} " + "  ".join(c_vals) + "   " + "  ".join(b_vals))
            return rows

        all_rows = []

        # Method A: Observed features (diagnostic)
        log("\n  Computing CIF with observed features (diagnostic)...")
        t0 = time.time()
        cif_obs = compute_cif(
            test_panel, feature_cols, ensemble, scaler,
            eval_times=args.eval_times, device=device, batch_size=args.batch_size,
        )
        log(f"  Done in {(time.time()-t0)/60:.1f} min")
        all_rows.extend(eval_cif(cif_obs, 'Observed'))

        # Method B: Frozen features
        log("\n  Computing CIF with frozen features...")
        t0 = time.time()
        cif_frz = compute_cif_frozen(
            test_panel, feature_cols, ensemble, scaler,
            eval_times=args.eval_times, device=device, batch_size=args.batch_size,
        )
        log(f"  Done in {(time.time()-t0)/60:.1f} min")
        all_rows.extend(eval_cif(cif_frz, 'Frozen'))

        # Method C: AR-simulated macro paths
        log("\n  Fitting AR models on training panel...")
        ar_models = fit_ar_models(train_panel, feature_cols, max_lag=4)
        for col, m in ar_models.items():
            p = len(m.params) - 1
            log(f"    {col}: AR({p}), sigma={np.sqrt(m.sigma2):.4f}")

        log("  Simulating CIF with AR paths (50 simulations)...")
        t0 = time.time()
        cif_ar_result = compute_cif_ar(
            test_panel, feature_cols, ensemble, scaler,
            eval_times=args.eval_times, ar_models=ar_models,
            n_simulations=50, device=device, batch_size=args.batch_size,
            seed=args.seed,
        )
        log(f"  Done in {(time.time()-t0)/60:.1f} min")
        all_rows.extend(eval_cif(cif_ar_result, 'AR-simulated'))

        # Save combined results
        results_df = pd.DataFrame(all_rows)
        cindex_path = models_dir / 'sadhwani_cindex.csv'
        results_df.to_csv(cindex_path, index=False)
        log(f"\n  Results saved to {cindex_path}")
    else:
        log("\n  Skipping CIF/evaluation (--skip-cif)")

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("Variable sensitivity analysis...")

    # Use a subsample for speed
    n_sens = min(50000, len(X_val))
    idx_sens = np.random.RandomState(args.seed).choice(len(X_val), n_sens, replace=False)
    X_sens = X_val[idx_sens]

    for cause, to_state in [('Prepay', 1), ('Default', 2)]:
        sens = variable_sensitivity(
            ensemble, X_sens, feature_cols,
            to_state=to_state, device=device,
        )
        log(f"\n  Sensitivity: Current -> {cause}")
        log(f"  {'Feature':<25} {'Gradient':>10}")
        log("  " + "-" * 37)
        for _, r in sens.iterrows():
            log(f"  {r['feature']:<25} {r['sensitivity']:>10.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    log("\n" + "-" * 70)
    log("Summary:")
    log(f"  Ensemble members: {args.n_ensemble}")
    log(f"  Architecture: {args.hidden_sizes}")
    log(f"  Cross-entropy (train): {train_ce:.6f}")
    log(f"  Cross-entropy (val):   {val_ce:.6f}")
    log(f"  Total runtime: {total_time/60:.1f} minutes")
    log(f"  Device: {device}")
    log(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Save report
    report_path = output_dir / 'sadhwani_results.txt'
    save_log(report_path)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
