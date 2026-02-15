#!/usr/bin/env python3
"""
Dynamic-DeepHit Competing Risks - Supercomputer Script

Implements Dynamic-DeepHit (Lee et al., 2020) for competing risks (prepayment
and default) using longitudinal loan-month panel data with GRU + temporal
attention.

Usage:
    python run_dynamic_deephit.py [--epochs 100] [--batch-size 64] [--lr 0.001]

Output:
    - results/dynamic_deephit_results.txt
    - models/dynamic_deephit_joint.pt
    - models/dynamic_deephit_scaler.pkl
    - models/dynamic_deephit_time_bins.npy
    - models/dynamic_deephit_feature_cols.pkl
    - models/dynamic_deephit_history.pkl
    - models/dynamic_deephit_cindex_static.csv
    - models/dynamic_deephit_cindex_dynamic.csv
    - reports/figures/dynamic_deephit_training_curves.png
    - reports/figures/dynamic_deephit_time_dependent_cindex.png
    - reports/figures/dynamic_deephit_dynamic_cindex_heatmap.png
    - reports/figures/dynamic_deephit_attention.png
    - reports/figures/dynamic_deephit_survival_curves.png
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.competing_risks.dynamic_deephit import (
    DynamicDeepHitNetwork,
    DynamicDeepHitLoss,
    MortgageSequenceDataset,
    collate_mortgage_sequences,
    preprocess_panel_to_sequences,
    ALL_FEATURES,
    TIME_VARYING_FEATURES,
)


# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dynamic-DeepHit Competing Risks (Lee et al. 2020)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--grad-accum-steps', type=int, default=4,
                        help='Gradient accumulation steps')

    # Loss function
    parser.add_argument('--alpha-prepay', type=float, default=0.2,
                        help='Ranking loss alpha for prepayment')
    parser.add_argument('--alpha-default', type=float, default=1.0,
                        help='Ranking loss alpha for default')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Smoothing parameter for ranking loss')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Weight for L3 next-step prediction loss')
    parser.add_argument('--default-event-weight', type=float, default=50.0,
                        help='NLL weight for default events')

    # Architecture
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='Input embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='GRU hidden dimension')
    parser.add_argument('--num-rnn-layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--head-hidden1', type=int, default=128,
                        help='First hidden layer in cause heads')
    parser.add_argument('--head-hidden2', type=int, default=64,
                        help='Second hidden layer in cause heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Sequence / time
    parser.add_argument('--max-seq-len', type=int, default=120,
                        help='Maximum sequence length')
    parser.add_argument('--num-time-bins', type=int, default=120,
                        help='Number of discrete time bins')

    # Dynamic evaluation
    parser.add_argument('--landmark-times', type=int, nargs='+',
                        default=[12, 24, 36, 48, 60],
                        help='Landmark ages for dynamic evaluation')
    parser.add_argument('--prediction-horizons', type=int, nargs='+',
                        default=[12, 24, 36],
                        help='Prediction horizons (months ahead)')

    # General
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Models directory')
    parser.add_argument('--figures-dir', type=str, default='reports/figures',
                        help='Figures directory')

    return parser.parse_args()


# Cross-validation folds (Blumenstock methodology)
TRAIN_FOLDS = list(range(10))
VAL_FOLDS = [9]
TEST_FOLD = 10

EVENT_NAMES = {0: 'Censored', 1: 'Prepay', 2: 'Default'}
STATIC_TIME_HORIZONS = [24, 48, 72]


# ==============================================================================
# Training
# ==============================================================================

def train_dynamic_deephit(
    model, criterion, train_loader, val_loader, time_bins,
    epochs=100, learning_rate=0.001, patience=10,
    grad_accum_steps=4, device=torch.device('cpu'), log_fn=print,
):
    """Train Dynamic-DeepHit with early stopping and gradient accumulation."""
    model = model.to(device)
    time_bins = time_bins.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Mixed precision on CUDA
    use_amp = device.type == 'cuda'
    scaler_amp = torch.amp.GradScaler('cuda') if use_amp else None

    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_l1', 'train_l2', 'train_l3',
        'val_l1', 'val_l2', 'val_l3']}

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    log_fn(f"Training on {device}")
    log_fn(f"{'Epoch':>6} | {'Train':>10} | {'Val':>10} | "
           f"{'L1':>8} | {'L2':>8} | {'L3':>8} | {'LR':>10}")
    log_fn("-" * 75)

    for epoch in range(epochs):
        model.train()
        train_metrics = {k: [] for k in ['loss', 'l1', 'l2', 'l3']}

        optimizer.zero_grad()
        for step, (x_padded, lengths, durations, events) in enumerate(train_loader):
            x_padded = x_padded.to(device, non_blocking=True)
            lengths = lengths  # keep on CPU for pack_padded_sequence
            durations = durations.to(device, non_blocking=True)
            events = events.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    pmf, x_pred = model(x_padded, lengths)
                    loss, l1, l2, l3 = criterion(
                        pmf, x_pred, x_padded, lengths, durations, events, time_bins)
                    loss = loss / grad_accum_steps
                scaler_amp.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    optimizer.zero_grad()
            else:
                pmf, x_pred = model(x_padded, lengths)
                loss, l1, l2, l3 = criterion(
                    pmf, x_pred, x_padded, lengths, durations, events, time_bins)
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()
                if (step + 1) % grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_metrics['loss'].append(loss.item() * grad_accum_steps)
            train_metrics['l1'].append(l1.item())
            train_metrics['l2'].append(l2.item())
            train_metrics['l3'].append(l3.item())

        # Flush remaining gradients
        if (step + 1) % grad_accum_steps != 0:
            if use_amp:
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        avg_train = {k: np.mean(v) for k, v in train_metrics.items()}

        # Validation
        model.eval()
        val_metrics = {k: [] for k in ['loss', 'l1', 'l2', 'l3']}
        with torch.no_grad():
            for x_padded, lengths, durations, events in val_loader:
                x_padded = x_padded.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                durations = durations.to(device, non_blocking=True)
                events = events.to(device, non_blocking=True)

                pmf, x_pred = model(x_padded, lengths)
                loss, l1, l2, l3 = criterion(
                    pmf, x_pred, x_padded, lengths, durations, events, time_bins)

                val_metrics['loss'].append(loss.item())
                val_metrics['l1'].append(l1.item())
                val_metrics['l2'].append(l2.item())
                val_metrics['l3'].append(l3.item())

        avg_val = {k: np.mean(v) for k, v in val_metrics.items()}

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val['loss'])

        history['train_loss'].append(avg_train['loss'])
        history['val_loss'].append(avg_val['loss'])
        history['train_l1'].append(avg_train['l1'])
        history['train_l2'].append(avg_train['l2'])
        history['train_l3'].append(avg_train['l3'])
        history['val_l1'].append(avg_val['l1'])
        history['val_l2'].append(avg_val['l2'])
        history['val_l3'].append(avg_val['l3'])

        if avg_val['loss'] < best_val_loss:
            best_val_loss = avg_val['loss']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == epochs - 1 or epochs_no_improve >= patience:
            log_fn(f"{epoch:>6} | {avg_train['loss']:>10.4f} | {avg_val['loss']:>10.4f} | "
                   f"{avg_val['l1']:>8.4f} | {avg_val['l2']:>8.4f} | "
                   f"{avg_val['l3']:>8.4f} | {current_lr:>10.6f}")

        if epochs_no_improve >= patience:
            log_fn(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    log_fn(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    return history


# ==============================================================================
# Evaluation Helpers
# ==============================================================================

def predict_test_set(model, test_loader, device):
    """Run inference on test set, return CIF and survival arrays."""
    model.eval()
    all_cif = []
    all_surv = []
    all_attn = []

    with torch.no_grad():
        for x_padded, lengths, durations, events in test_loader:
            x_padded = x_padded.to(device)
            lengths = lengths  # keep on CPU for pack_padded_sequence

            # Single forward pass → compute CIF and survival from PMF
            pmf, _ = model(x_padded, lengths)
            cif = torch.cumsum(pmf, dim=-1)
            surv = 1.0 - cif.sum(dim=1)
            attn = model.get_attention_weights()

            all_cif.append(cif.cpu().numpy())
            all_surv.append(surv.cpu().numpy())
            if attn is not None:
                all_attn.append(attn.cpu().numpy())

    cif_np = np.concatenate(all_cif, axis=0)
    surv_np = np.concatenate(all_surv, axis=0)

    # Attention weights may have different seq_len per batch due to padding;
    # pad to a common max length before concatenation
    if all_attn:
        max_t = max(a.shape[1] for a in all_attn)
        padded_attn = []
        for a in all_attn:
            if a.shape[1] < max_t:
                pad_width = max_t - a.shape[1]
                a = np.pad(a, ((0, 0), (0, pad_width)), constant_values=0.0)
            padded_attn.append(a)
        attn_np = np.concatenate(padded_attn, axis=0)
    else:
        attn_np = None

    return cif_np, surv_np, attn_np


def get_risk_at_horizon(cif, time_points, tau):
    """Get CIF value at time horizon tau."""
    idx = min(np.searchsorted(time_points, tau), len(time_points) - 1)
    return cif[:, idx]


def compute_static_cindex(cif_np, surv_np, time_points,
                          duration_test, event_test,
                          duration_train, event_train,
                          log_fn=print):
    """Compute IPCW C-index at static time horizons (24, 48, 72)."""
    cif_prepay = cif_np[:, 0, :]
    cif_default = cif_np[:, 1, :]

    results = {}
    for cause_name, cause_code, cif_cause in [
        ('Prepay', 1, cif_prepay), ('Default', 2, cif_default)
    ]:
        log_fn(f"\n  {cause_name.upper()} C-index:")

        event_binary = (event_test == cause_code).astype(bool)
        y_train_sk = Surv.from_arrays(
            (event_train == cause_code).astype(bool), duration_train)
        y_test_sk = Surv.from_arrays(event_binary, duration_test)

        ipcw = {}
        for tau in STATIC_TIME_HORIZONS:
            try:
                risk = get_risk_at_horizon(cif_cause, time_points, tau)
                c_tau = concordance_index_ipcw(y_train_sk, y_test_sk, risk, tau=tau)
                ipcw[tau] = c_tau[0]
                log_fn(f"    tau={tau:3d} months: C-index (IPCW) = {c_tau[0]:.4f}")
            except Exception as e:
                log_fn(f"    tau={tau:3d} months: Error - {str(e)[:60]}")
                ipcw[tau] = np.nan

        risk_overall = cif_cause.mean(axis=1)
        c_harrell = concordance_index_censored(
            event_binary, duration_test, risk_overall)[0]
        log_fn(f"    Overall (Harrell): {c_harrell:.4f}")

        results[cause_name] = {'ipcw': ipcw, 'harrell': c_harrell}

    return results


def compute_dynamic_cindex(
    model, panel_df, feature_cols, scaler, time_bins_tensor,
    landmark_times, prediction_horizons, max_seq_len,
    device, log_fn=print,
):
    """
    Compute dynamic C-index at landmark times x prediction horizons.

    For each landmark age t_L, take loans that survived past t_L,
    truncate their history to t_L months, predict from there,
    and evaluate C-index for events within horizon.
    """
    model.eval()
    time_col = 'loan_age'
    event_col = 'event_code'
    loan_id_col = 'loan_sequence_number'

    # Work with test fold only
    test_df = panel_df[panel_df['fold'] == TEST_FOLD].copy()

    results = {}
    log_fn("\n  Dynamic C-index (landmark x horizon):")

    for t_L in landmark_times:
        results[t_L] = {}

        # Loans that survived past t_L
        loan_durations = test_df.groupby(loan_id_col)[time_col].max()
        eligible_loans = loan_durations[loan_durations > t_L].index

        if len(eligible_loans) < 50:
            log_fn(f"    t_L={t_L}: too few eligible loans ({len(eligible_loans)})")
            for h in prediction_horizons:
                results[t_L][h] = np.nan
            continue

        # Build truncated sequences at landmark time
        sequences = []
        durations_from_landmark = []
        events_at_end = []

        for loan_id in eligible_loans:
            loan_data = test_df[test_df[loan_id_col] == loan_id].sort_values(time_col)

            # Truncate history to t_L
            truncated = loan_data[loan_data[time_col] <= t_L]
            if len(truncated) == 0:
                continue

            x = truncated[feature_cols].values.astype('float32')
            if scaler is not None:
                x = scaler.transform(x).astype('float32')

            if len(x) > max_seq_len:
                x = x[-max_seq_len:]

            # Residual time from landmark
            final_duration = float(loan_data[time_col].iloc[-1])
            final_event = int(loan_data[event_col].iloc[-1])

            sequences.append(torch.tensor(x, dtype=torch.float32))
            durations_from_landmark.append(final_duration - t_L)
            events_at_end.append(final_event)

        if len(sequences) < 50:
            for h in prediction_horizons:
                results[t_L][h] = np.nan
            continue

        # Pad and predict
        x_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0).to(device)
        seq_lengths = torch.tensor(
            [s.shape[0] for s in sequences], dtype=torch.long).to(device)

        with torch.no_grad():
            cif = model.predict_cif(x_padded, seq_lengths).cpu().numpy()

        residual_dur = np.array(durations_from_landmark)
        residual_ev = np.array(events_at_end)

        time_bins_np = time_bins_tensor.cpu().numpy()
        time_points = (time_bins_np[:-1] + time_bins_np[1:]) / 2

        for h in prediction_horizons:
            for cause_name, cause_code, cause_idx in [
                ('Prepay', 1, 0), ('Default', 2, 1)
            ]:
                key = f"{cause_name}_{h}"
                event_binary = (residual_ev == cause_code).astype(bool)

                # Only evaluate if there are events within horizon
                events_in_window = (
                    (residual_dur <= h) & (residual_ev == cause_code)
                ).sum()
                if events_in_window < 3:
                    results[t_L][key] = np.nan
                    continue

                try:
                    risk = get_risk_at_horizon(cif[:, cause_idx, :], time_points, h)
                    c_idx = concordance_index_censored(
                        event_binary & (residual_dur <= h),
                        np.minimum(residual_dur, h),
                        risk,
                    )[0]
                    results[t_L][key] = c_idx
                except Exception:
                    results[t_L][key] = np.nan

        # Log summary for this landmark
        prepay_vals = [results[t_L].get(f"Prepay_{h}", np.nan) for h in prediction_horizons]
        default_vals = [results[t_L].get(f"Default_{h}", np.nan) for h in prediction_horizons]
        log_fn(f"    t_L={t_L:3d}: Prepay=[{', '.join(f'{v:.3f}' if not np.isnan(v) else 'NaN' for v in prepay_vals)}] "
               f"Default=[{', '.join(f'{v:.3f}' if not np.isnan(v) else 'NaN' for v in default_vals)}]")

    return results


# ==============================================================================
# Plotting
# ==============================================================================

def plot_training_curves(history, figures_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ax, (train_key, val_key, title) in zip(axes, [
        ('train_loss', 'val_loss', 'Total Loss'),
        ('train_l1', 'val_l1', 'L1: NLL'),
        ('train_l2', 'val_l2', 'L2: Ranking'),
        ('train_l3', 'val_l3', 'L3: Next-Step'),
    ]):
        ax.plot(history[train_key], label='Train')
        ax.plot(history[val_key], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Dynamic-DeepHit: {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'dynamic_deephit_training_curves.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_static_cindex(cindex_results, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = STATIC_TIME_HORIZONS
    prepay_vals = [cindex_results['Prepay']['ipcw'].get(h, 0) for h in horizons]
    default_vals = [cindex_results['Default']['ipcw'].get(h, 0) for h in horizons]

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
    ax.set_ylabel('C-index (IPCW)')
    ax.set_title('Dynamic-DeepHit: Time-Dependent Concordance Index')
    ax.set_xticks(x)
    ax.set_xticklabels([f'tau = {h}' for h in horizons])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figures_dir / 'dynamic_deephit_time_dependent_cindex.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_dynamic_cindex_heatmap(dynamic_results, landmark_times,
                                prediction_horizons, figures_dir):
    """Plot dynamic C-index as heatmap (landmark x horizon)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cause_name, cmap in zip(axes, ['Prepay', 'Default'],
                                     ['Blues', 'Reds']):
        matrix = np.full((len(landmark_times), len(prediction_horizons)), np.nan)
        for i, t_L in enumerate(landmark_times):
            for j, h in enumerate(prediction_horizons):
                key = f"{cause_name}_{h}"
                if t_L in dynamic_results and key in dynamic_results[t_L]:
                    matrix[i, j] = dynamic_results[t_L][key]

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(prediction_horizons)))
        ax.set_xticklabels([f'{h}m' for h in prediction_horizons])
        ax.set_yticks(range(len(landmark_times)))
        ax.set_yticklabels([f'{t}m' for t in landmark_times])
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('Landmark Age')
        ax.set_title(f'{cause_name}: Dynamic C-index')

        for i in range(len(landmark_times)):
            for j in range(len(prediction_horizons)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=9, color='white' if val > 0.75 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Dynamic-DeepHit: Dynamic C-index (Landmark x Horizon)', fontsize=13)
    plt.tight_layout()
    plt.savefig(figures_dir / 'dynamic_deephit_dynamic_cindex_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_weights(attn_np, lengths_test, figures_dir, n_samples=6):
    """Plot attention weight distributions for sample loans."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    np.random.seed(42)
    # Pick loans with reasonable length
    valid_idx = np.where(lengths_test > 10)[0]
    if len(valid_idx) < n_samples:
        valid_idx = np.arange(min(n_samples, len(attn_np)))
    sample_idx = np.random.choice(valid_idx, size=min(n_samples, len(valid_idx)),
                                  replace=False)

    for ax, idx in zip(axes, sample_idx):
        length = int(lengths_test[idx])
        weights = attn_np[idx, :length]
        ax.bar(range(length), weights, color='steelblue', alpha=0.7)
        ax.set_xlabel('Month')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Loan {idx} (len={length})')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Dynamic-DeepHit: Temporal Attention Weights', fontsize=13)
    plt.tight_layout()
    plt.savefig(figures_dir / 'dynamic_deephit_attention.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_survival_curves(cif_np, surv_np, time_points, figures_dir):
    """Plot CIF and survival curves for sample loans."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    np.random.seed(42)
    sample_idx = np.random.choice(len(surv_np), size=5, replace=False)

    cif_prepay = cif_np[:, 0, :]
    cif_default = cif_np[:, 1, :]

    for ax, data, title, ylabel in zip(axes,
        [cif_prepay[sample_idx], cif_default[sample_idx], surv_np[sample_idx]],
        ['Prepayment CIF', 'Default CIF', 'Overall Survival S(t)'],
        ['Cumulative Incidence', 'Cumulative Incidence', 'Survival Probability'],
    ):
        for i, idx in enumerate(sample_idx):
            ax.plot(time_points, data[i], label=f'Loan {idx}', alpha=0.7)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Dynamic-DeepHit: {title}')
        ax.legend(loc='lower right' if 'CIF' in title else 'lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(figures_dir / 'dynamic_deephit_survival_curves.png',
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
    output_dir = base_dir / args.output_dir
    models_dir = base_dir / args.models_dir
    figures_dir = base_dir / args.figures_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'dynamic_deephit_results.txt'

    # Logging
    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("DYNAMIC-DEEPHIT COMPETING RISKS")
    log("Lee et al. (2020) — GRU + Temporal Attention")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch version: {torch.__version__}")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        log(f"  CUDA version: {torch.version.cuda}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        log("Device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        log("Device: CPU")

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log(f"\nConfiguration:")
    log(f"  epochs: {args.epochs}")
    log(f"  batch_size: {args.batch_size}")
    log(f"  learning_rate: {args.lr}")
    log(f"  patience: {args.patience}")
    log(f"  grad_accum_steps: {args.grad_accum_steps}")
    log(f"  alpha_prepay: {args.alpha_prepay}")
    log(f"  alpha_default: {args.alpha_default}")
    log(f"  sigma: {args.sigma}")
    log(f"  beta (L3): {args.beta}")
    log(f"  default_event_weight: {args.default_event_weight}")
    log(f"  architecture: GRU({args.embed_dim}->{args.hidden_dim}, "
        f"layers={args.num_rnn_layers}) + Attention + Heads({args.head_hidden1},{args.head_hidden2})")
    log(f"  dropout: {args.dropout}")
    log(f"  max_seq_len: {args.max_seq_len}")
    log(f"  num_time_bins: {args.num_time_bins}")
    log(f"  seed: {args.seed}")

    # ==================== Load Data ====================
    log("\n" + "-" * 70)
    log("Loading data...")

    panel_df = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    log(f"  Loaded {len(panel_df):,} loan-months")
    log(f"  Unique loans: {panel_df['loan_sequence_number'].nunique():,}")

    feature_cols = [f for f in ALL_FEATURES if f in panel_df.columns]
    missing = [f for f in ALL_FEATURES if f not in panel_df.columns]
    log(f"  Available features: {len(feature_cols)}/{len(ALL_FEATURES)}")
    if missing:
        log(f"  Missing: {missing}")

    # ==================== Preprocess Sequences ====================
    log("\n" + "-" * 70)
    log("Preprocessing sequences...")

    train_folds_actual = [f for f in TRAIN_FOLDS if f not in VAL_FOLDS]

    # Split panel by fold
    train_panel = panel_df[panel_df['fold'].isin(train_folds_actual)].copy()
    val_panel = panel_df[panel_df['fold'].isin(VAL_FOLDS)].copy()
    test_panel = panel_df[panel_df['fold'] == TEST_FOLD].copy()

    log(f"  Train panel: {len(train_panel):,} loan-months")
    log(f"  Val panel: {len(val_panel):,} loan-months")
    log(f"  Test panel: {len(test_panel):,} loan-months")

    # Preprocess train (fits scaler)
    train_seqs, scaler, final_feature_cols = preprocess_panel_to_sequences(
        train_panel, feature_cols, max_seq_len=args.max_seq_len,
        fit_scaler=True,
    )
    log(f"  Train sequences: {len(train_seqs):,}")

    # Preprocess val/test (uses fitted scaler)
    val_seqs, _, _ = preprocess_panel_to_sequences(
        val_panel, feature_cols, max_seq_len=args.max_seq_len,
        scaler=scaler,
    )
    test_seqs, _, _ = preprocess_panel_to_sequences(
        test_panel, feature_cols, max_seq_len=args.max_seq_len,
        scaler=scaler,
    )
    log(f"  Val sequences: {len(val_seqs):,}")
    log(f"  Test sequences: {len(test_seqs):,}")

    # Sequence length statistics
    train_lengths = [s['length'] for s in train_seqs]
    log(f"  Sequence lengths: mean={np.mean(train_lengths):.1f}, "
        f"median={np.median(train_lengths):.0f}, "
        f"max={np.max(train_lengths)}, "
        f"min={np.min(train_lengths)}")

    # Event distribution
    train_events = [s['event'] for s in train_seqs]
    log("\n  Event distribution (train):")
    for code in sorted(set(train_events)):
        count = sum(1 for e in train_events if e == code)
        log(f"    {EVENT_NAMES.get(code, 'Other')} (k={code}): {count:,}")

    # Create datasets and loaders
    train_dataset = MortgageSequenceDataset(train_seqs, max_seq_len=args.max_seq_len)
    val_dataset = MortgageSequenceDataset(val_seqs, max_seq_len=args.max_seq_len)
    test_dataset = MortgageSequenceDataset(test_seqs, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_mortgage_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_mortgage_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_mortgage_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ==================== Time Bins ====================
    all_durations = np.array([s['duration'] for s in train_seqs + val_seqs + test_seqs])
    time_bins = np.linspace(all_durations.min(), all_durations.max(),
                            args.num_time_bins + 1)
    time_bins_tensor = torch.tensor(time_bins, dtype=torch.float32)
    time_points = (time_bins[:-1] + time_bins[1:]) / 2

    log(f"\n  Time bins: {args.num_time_bins}, "
        f"width={np.diff(time_bins).mean():.2f} months")

    # ==================== Compute TV feature indices ====================
    tv_feature_indices = []
    for feat in TIME_VARYING_FEATURES:
        # Map to final_feature_cols
        if feat in final_feature_cols:
            tv_feature_indices.append(final_feature_cols.index(feat))
    log(f"  Time-varying features for L3: {len(tv_feature_indices)}")

    # ==================== Build Model ====================
    log("\n" + "-" * 70)
    log("Building model...")

    in_features = len(final_feature_cols)
    model = DynamicDeepHitNetwork(
        in_features=in_features,
        num_time_bins=args.num_time_bins,
        num_causes=2,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_rnn_layers=args.num_rnn_layers,
        head_hidden1=args.head_hidden1,
        head_hidden2=args.head_hidden2,
        dropout=args.dropout,
        num_tv_features=len(tv_feature_indices),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Parameters: {n_params:,}")
    log(f"  Input features: {in_features}")

    criterion = DynamicDeepHitLoss(
        alpha_prepay=args.alpha_prepay,
        alpha_default=args.alpha_default,
        sigma=args.sigma,
        beta=args.beta,
        default_event_weight=args.default_event_weight,
        num_tv_features=len(tv_feature_indices),
        tv_feature_indices=tv_feature_indices,
    )

    # ==================== Train ====================
    log("\n" + "-" * 70)
    log("Training Dynamic-DeepHit...")

    train_start = time.time()
    history = train_dynamic_deephit(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        time_bins=time_bins_tensor,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        grad_accum_steps=args.grad_accum_steps,
        device=device,
        log_fn=log,
    )
    train_time = time.time() - train_start
    log(f"\nTraining completed in {train_time/60:.1f} minutes")

    # ==================== Static Evaluation ====================
    log("\n" + "-" * 70)
    log("Static evaluation (C-index at 24/48/72 months)...")

    cif_np, surv_np, attn_np = predict_test_set(model, test_loader, device)

    duration_test = np.array([s['duration'] for s in test_seqs])
    event_test = np.array([s['event'] for s in test_seqs])
    duration_train = np.array([s['duration'] for s in train_seqs])
    event_train = np.array([s['event'] for s in train_seqs])

    static_cindex = compute_static_cindex(
        cif_np, surv_np, time_points,
        duration_test, event_test,
        duration_train, event_train,
        log_fn=log,
    )

    # ==================== Dynamic Evaluation ====================
    log("\n" + "-" * 70)
    log("Dynamic evaluation (landmark analysis)...")

    # Apply same feature engineering to panel_df for dynamic eval
    panel_df_fe = panel_df.copy()
    if 'bal_repaid' in panel_df_fe.columns:
        panel_df_fe['bal_repaid'] = (
            panel_df_fe.groupby('loan_sequence_number')['bal_repaid']
            .shift(1).fillna(0.0)
        )
    if 'orig_upb' in panel_df_fe.columns:
        panel_df_fe['log_upb'] = np.log(panel_df_fe['orig_upb'].clip(lower=1.0))

    dynamic_cindex = compute_dynamic_cindex(
        model=model,
        panel_df=panel_df_fe,
        feature_cols=final_feature_cols,
        scaler=scaler,
        time_bins_tensor=time_bins_tensor,
        landmark_times=args.landmark_times,
        prediction_horizons=args.prediction_horizons,
        max_seq_len=args.max_seq_len,
        device=device,
        log_fn=log,
    )

    # ==================== Plots ====================
    log("\n" + "-" * 70)
    log("Generating plots...")

    plot_training_curves(history, figures_dir)
    log(f"  {figures_dir / 'dynamic_deephit_training_curves.png'}")

    plot_static_cindex(static_cindex, figures_dir)
    log(f"  {figures_dir / 'dynamic_deephit_time_dependent_cindex.png'}")

    plot_dynamic_cindex_heatmap(
        dynamic_cindex, args.landmark_times, args.prediction_horizons, figures_dir)
    log(f"  {figures_dir / 'dynamic_deephit_dynamic_cindex_heatmap.png'}")

    if attn_np is not None:
        lengths_test = np.array([s['length'] for s in test_seqs])
        plot_attention_weights(attn_np, lengths_test, figures_dir)
        log(f"  {figures_dir / 'dynamic_deephit_attention.png'}")

    plot_survival_curves(cif_np, surv_np, time_points, figures_dir)
    log(f"  {figures_dir / 'dynamic_deephit_survival_curves.png'}")

    # ==================== Summary ====================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\nArchitecture:")
    log(f"  Input Embedding: {in_features} -> {args.embed_dim}")
    log(f"  GRU: {args.num_rnn_layers} layers x {args.hidden_dim} hidden")
    log(f"  Temporal Attention: score({args.hidden_dim}+{in_features}) -> 1")
    log(f"  Cause heads: {args.head_hidden1} -> {args.head_hidden2} -> {args.num_time_bins}")
    log(f"  Parameters: {n_params:,}")

    log(f"\nData:")
    log(f"  Train: {len(train_seqs):,} sequences")
    log(f"  Val: {len(val_seqs):,} sequences")
    log(f"  Test: {len(test_seqs):,} sequences")
    log(f"  Features: {in_features}")

    log(f"\nStatic Performance (Test Set):")
    for cause_name in ['Prepay', 'Default']:
        cr = static_cindex[cause_name]
        log(f"  {cause_name}:")
        log(f"    Harrell C-index: {cr['harrell']:.4f}")
        for tau, c in cr['ipcw'].items():
            log(f"    IPCW C-index (tau={tau}): {c:.4f}")

    total_time = time.time() - start_time
    log(f"\nTotal runtime: {total_time/60:.1f} minutes")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ==================== Save ====================
    log("\nSaving artifacts...")

    # Model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_features': in_features,
        'num_time_bins': args.num_time_bins,
        'num_causes': 2,
        'embed_dim': args.embed_dim,
        'hidden_dim': args.hidden_dim,
        'num_rnn_layers': args.num_rnn_layers,
        'head_hidden1': args.head_hidden1,
        'head_hidden2': args.head_hidden2,
        'dropout': args.dropout,
        'num_tv_features': len(tv_feature_indices),
        'tv_feature_indices': tv_feature_indices,
    }, models_dir / 'dynamic_deephit_joint.pt')
    log(f"  Model: {models_dir / 'dynamic_deephit_joint.pt'}")

    # Scaler
    with open(models_dir / 'dynamic_deephit_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Time bins
    np.save(models_dir / 'dynamic_deephit_time_bins.npy', time_bins)

    # Feature columns
    with open(models_dir / 'dynamic_deephit_feature_cols.pkl', 'wb') as f:
        pickle.dump(final_feature_cols, f)

    # Training history
    with open(models_dir / 'dynamic_deephit_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Static C-index CSV
    cindex_df = pd.DataFrame({
        cause: {f'ipcw_{tau}': cr['ipcw'].get(tau, np.nan) for tau in STATIC_TIME_HORIZONS}
            | {'harrell': cr['harrell']}
        for cause, cr in static_cindex.items()
    })
    cindex_df.to_csv(models_dir / 'dynamic_deephit_cindex_static.csv')
    log(f"  Static C-index: {models_dir / 'dynamic_deephit_cindex_static.csv'}")

    # Dynamic C-index CSV
    dyn_rows = []
    for t_L, horizons in dynamic_cindex.items():
        for key, val in horizons.items():
            dyn_rows.append({'landmark': t_L, 'metric': key, 'cindex': val})
    dyn_df = pd.DataFrame(dyn_rows)
    dyn_df.to_csv(models_dir / 'dynamic_deephit_cindex_dynamic.csv', index=False)
    log(f"  Dynamic C-index: {models_dir / 'dynamic_deephit_cindex_dynamic.csv'}")

    # Text report
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"  Report: {results_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
