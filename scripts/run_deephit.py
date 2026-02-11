#!/usr/bin/env python3
"""
DeepHit Competing Risks - Supercomputer Script

Implements DeepHit (Lee et al., 2018) with the Blumenstock et al. (2022)
architecture for competing risks (prepayment and default) using pure PyTorch.

Usage:
    python run_deephit.py [--epochs 100] [--batch-size 256] [--lr 0.01]

Output:
    - results/deephit_results.txt          (summary report)
    - models/deephit_joint.pt              (model checkpoint)
    - models/deephit_scaler.pkl            (feature scaler)
    - models/deephit_time_bins.npy         (discretization bins)
    - models/deephit_feature_cols.pkl      (feature column names)
    - models/deephit_history.pkl           (training history)
    - models/deephit_importance_prepay.csv (permutation importance)
    - models/deephit_importance_default.csv
    - reports/figures/deephit_training_curves.png
    - reports/figures/deephit_time_dependent_cindex.png
    - reports/figures/deephit_survival_curves.png
    - reports/figures/deephit_feature_importance.png
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
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================================
# Configuration
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DeepHit Competing Risks (Blumenstock et al. 2022)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Loss function
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Weight for ranking loss')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Smoothing parameter for ranking loss')

    # Architecture
    parser.add_argument('--shared-layers', type=int, default=3,
                        help='Number of shared FFN layers')
    parser.add_argument('--shared-nodes', type=int, default=300,
                        help='Nodes per shared layer')
    parser.add_argument('--head-layers', type=int, default=5,
                        help='Number of cause-specific head layers')
    parser.add_argument('--head-nodes', type=int, default=200,
                        help='Nodes per head layer')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--no-batch-norm', action='store_true',
                        help='Disable batch normalization')

    # Time discretization
    parser.add_argument('--num-durations', type=int, default=200,
                        help='Number of discrete time bins')

    # Evaluation
    parser.add_argument('--importance-repeats', type=int, default=5,
                        help='Repeats for permutation importance')

    # General
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0=main process)')
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

# Features (Blumenstock et al. 2022, Table 2)
STATIC_FEATURES = ['int_rate', 'orig_upb', 'fico_score', 'dti_r', 'ltv_r']
BEHAVIORAL_FEATURES = ['bal_repaid', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m']
MACRO_FEATURES = [
    'hpi_st_d_t_o', 'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o',
    'ppi_o_FRMA', 'hpi_st_log12m', 'hpi_r_st_us', 'st_unemp_r12m',
    'st_unemp_r3m', 'TB10Y_r12m', 'T10Y3MM', 'T10Y3MM_r12m',
]
ALL_FEATURES = STATIC_FEATURES + BEHAVIORAL_FEATURES + MACRO_FEATURES

EVENT_NAMES = {0: 'Censored', 1: 'Prepay', 2: 'Default'}
TIME_HORIZONS = [24, 48, 72]


# ==============================================================================
# DeepHit Network (Blumenstock et al. 2022 architecture)
# ==============================================================================

class DeepHitNetwork(nn.Module):
    """
    DeepHit neural network for competing risks.

    Architecture:
    - Shared FFN with residual connection from input
    - Cause-specific heads (one per event type)
    - Joint softmax over (time, cause) for PMF output
    """

    def __init__(self, in_features, num_time_bins, num_causes=2,
                 shared_layers=3, shared_nodes=300,
                 head_layers=5, head_nodes=200,
                 dropout=0.2, batch_norm=True):
        super().__init__()
        self.in_features = in_features
        self.num_time_bins = num_time_bins
        self.num_causes = num_causes

        # Shared FFN
        shared = []
        prev_dim = in_features
        for _ in range(shared_layers):
            shared.append(nn.Linear(prev_dim, shared_nodes))
            if batch_norm:
                shared.append(nn.BatchNorm1d(shared_nodes))
            shared.append(nn.ReLU())
            shared.append(nn.Dropout(dropout))
            prev_dim = shared_nodes
        self.shared = nn.Sequential(*shared)

        # Residual projection
        self.residual_proj = nn.Linear(in_features, shared_nodes)

        # Cause-specific heads
        self.heads = nn.ModuleList()
        for _ in range(num_causes):
            head = []
            prev_dim = shared_nodes
            for _ in range(head_layers):
                head.append(nn.Linear(prev_dim, head_nodes))
                if batch_norm:
                    head.append(nn.BatchNorm1d(head_nodes))
                head.append(nn.ReLU())
                head.append(nn.Dropout(dropout))
                prev_dim = head_nodes
            head.append(nn.Linear(prev_dim, num_time_bins))
            self.heads.append(nn.Sequential(*head))

    def forward(self, x):
        """Returns joint PMF [batch, num_causes, num_time_bins]."""
        batch_size = x.shape[0]
        shared_out = self.shared(x)
        residual = self.residual_proj(x)
        combined = shared_out + residual

        head_outputs = [head(combined) for head in self.heads]
        logits = torch.stack(head_outputs, dim=1)

        logits_flat = logits.view(batch_size, -1)
        pmf_flat = torch.softmax(logits_flat, dim=-1)
        return pmf_flat.view(batch_size, self.num_causes, self.num_time_bins)

    def predict_cif(self, x):
        """CIF_k(t) = cumsum of PMF over time [batch, num_causes, num_time_bins]."""
        return torch.cumsum(self.forward(x), dim=-1)

    def predict_survival(self, x):
        """S(t) = 1 - sum_k CIF_k(t) [batch, num_time_bins]."""
        return 1 - self.predict_cif(x).sum(dim=1)


# ==============================================================================
# DeepHit Loss (Lee et al., 2018)
# ==============================================================================

class DeepHitLoss(nn.Module):
    """Combined NLL + ranking loss for competing risks."""

    def __init__(self, alpha=0.2, sigma=0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, pmf, durations, events, time_bins):
        batch_size = pmf.shape[0]
        num_causes = pmf.shape[1]
        num_bins = pmf.shape[2]
        device = pmf.device
        eps = 1e-7

        bin_indices = torch.bucketize(durations, time_bins[1:])
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        # CIF and survival
        cif = torch.cumsum(pmf, dim=-1)
        total_cif = cif.sum(dim=1)
        survival = torch.clamp(1 - total_cif, min=0.0)

        survival_at_time = survival[torch.arange(batch_size, device=device), bin_indices]

        cause_indices = torch.clamp((events - 1).long(), 0, num_causes - 1)
        pmf_at_event = pmf[
            torch.arange(batch_size, device=device), cause_indices, bin_indices
        ]

        is_censored = (events == 0).float()
        is_uncensored = (events > 0).float()

        nll_loss = (
            -torch.log(pmf_at_event + eps) * is_uncensored +
            -torch.log(survival_at_time + eps) * is_censored
        ).mean()

        # Ranking loss
        if self.alpha > 0 and is_uncensored.sum() > 0:
            ranking_loss = self._ranking_loss(cif, bin_indices, events, num_causes)
        else:
            ranking_loss = torch.tensor(0.0, device=device)

        return nll_loss + self.alpha * ranking_loss, nll_loss, ranking_loss

    def _ranking_loss(self, cif, bin_indices, events, num_causes):
        device = cif.device
        ranking_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for k in range(num_causes):
            event_code = k + 1
            cause_mask = (events == event_code)
            cause_idx = torch.where(cause_mask)[0]

            if len(cause_idx) < 1:
                continue

            sample_size = min(100, len(cause_idx))
            if len(cause_idx) > sample_size:
                perm = torch.randperm(len(cause_idx), device=device)[:sample_size]
                cause_idx = cause_idx[perm]

            for idx in cause_idx:
                t_i = bin_indices[idx]
                later_idx = torch.where(bin_indices > t_i)[0]

                if len(later_idx) == 0:
                    continue
                if len(later_idx) > 10:
                    perm = torch.randperm(len(later_idx), device=device)[:10]
                    later_idx = later_idx[perm]

                diff = cif[later_idx, k, t_i] - cif[idx, k, t_i]
                ranking_loss = ranking_loss + torch.exp(diff / self.sigma).sum()
                n_pairs += len(later_idx)

        if n_pairs > 0:
            ranking_loss = ranking_loss / n_pairs
        return ranking_loss


# ==============================================================================
# Training
# ==============================================================================

def train_deephit(model, criterion, X_train, y_train, e_train,
                  X_val, y_val, e_val, time_bins,
                  batch_size=256, epochs=100, learning_rate=0.01,
                  patience=10, num_workers=0, device=torch.device('cpu'),
                  log_fn=print):
    """Train DeepHit with early stopping. Returns training history dict."""
    model = model.to(device)
    time_bins = time_bins.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    e_train_t = torch.tensor(e_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    e_val_t = torch.tensor(e_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, e_train_t),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_nll', 'train_rank', 'val_nll', 'val_rank']}

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    log_fn(f"Training on {device}")
    log_fn(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
           f"{'NLL':>8} | {'Rank':>8} | {'LR':>10}")
    log_fn("-" * 70)

    for epoch in range(epochs):
        model.train()
        train_losses, train_nlls, train_ranks = [], [], []

        for batch_X, batch_y, batch_e in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_e = batch_e.to(device, non_blocking=True)

            optimizer.zero_grad()
            pmf = model(batch_X)
            loss, nll, rank = criterion(pmf, batch_y, batch_e, time_bins)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_nlls.append(nll.item())
            train_ranks.append(rank.item())

        avg_train = np.mean(train_losses)
        avg_nll = np.mean(train_nlls)
        avg_rank = np.mean(train_ranks)

        # Validation
        model.eval()
        with torch.no_grad():
            pmf_val = model(X_val_t)
            val_loss, val_nll, val_rank = criterion(pmf_val, y_val_t, e_val_t, time_bins)
            val_loss_v = val_loss.item()
            val_nll_v = val_nll.item()
            val_rank_v = val_rank.item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss_v)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss_v)
        history['train_nll'].append(avg_nll)
        history['train_rank'].append(avg_rank)
        history['val_nll'].append(val_nll_v)
        history['val_rank'].append(val_rank_v)

        if val_loss_v < best_val_loss:
            best_val_loss = val_loss_v
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == epochs - 1 or epochs_no_improve >= patience:
            log_fn(f"{epoch:>6} | {avg_train:>10.4f} | {val_loss_v:>10.4f} | "
                   f"{val_nll_v:>8.4f} | {val_rank_v:>8.4f} | {current_lr:>10.6f}")

        if epochs_no_improve >= patience:
            log_fn(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    log_fn(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    return history


# ==============================================================================
# Evaluation
# ==============================================================================

def get_risk_at_horizon(cif, time_points, tau):
    """Get CIF value at time horizon tau."""
    idx = min(np.searchsorted(time_points, tau), len(time_points) - 1)
    return cif[:, idx]


def compute_cindex(model, X_test, duration_test, event_test,
                   duration_train, event_train, time_points,
                   device, log_fn=print):
    """Compute Harrell and IPCW C-index for both causes."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        cif_np = model.predict_cif(X_t).cpu().numpy()
        surv_np = model.predict_survival(X_t).cpu().numpy()

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

        # IPCW C-index at time horizons
        ipcw = {}
        for tau in TIME_HORIZONS:
            try:
                risk = get_risk_at_horizon(cif_cause, time_points, tau)
                c_tau = concordance_index_ipcw(y_train_sk, y_test_sk, risk, tau=tau)
                ipcw[tau] = c_tau[0]
                log_fn(f"    tau={tau:3d} months: C-index (IPCW) = {c_tau[0]:.4f}")
            except Exception as e:
                log_fn(f"    tau={tau:3d} months: Error - {str(e)[:60]}")
                ipcw[tau] = np.nan

        # Harrell C-index
        risk_overall = cif_cause.mean(axis=1)
        c_harrell = concordance_index_censored(
            event_binary, duration_test, risk_overall)[0]
        log_fn(f"    Overall (Harrell): {c_harrell:.4f}")

        results[cause_name] = {'ipcw': ipcw, 'harrell': c_harrell}

    return results, cif_prepay, cif_default, surv_np


def permutation_importance_deephit(model, X, duration, event, feature_names,
                                   cause_idx, device, n_repeats=5):
    """Permutation importance for a given cause."""
    model.eval()
    event_code = cause_idx + 1
    event_binary = (event == event_code).astype(bool)

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        cif_baseline = model.predict_cif(X_t).cpu().numpy()[:, cause_idx, :]
    risk_baseline = cif_baseline.mean(axis=1)
    baseline_cindex = concordance_index_censored(
        event_binary, duration, risk_baseline)[0]

    importances, importances_std = [], []
    for i, feat in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, i])
            with torch.no_grad():
                X_perm_t = torch.tensor(X_perm, dtype=torch.float32).to(device)
                cif_perm = model.predict_cif(X_perm_t).cpu().numpy()[:, cause_idx, :]
            risk_perm = cif_perm.mean(axis=1)
            perm_cindex = concordance_index_censored(
                event_binary, duration, risk_perm)[0]
            scores.append(baseline_cindex - perm_cindex)

        importances.append(np.mean(scores))
        importances_std.append(np.std(scores))

    return np.array(importances), np.array(importances_std)


# ==============================================================================
# Plotting
# ==============================================================================

def plot_training_curves(history, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (train_key, val_key, title) in zip(axes, [
        ('train_loss', 'val_loss', 'Total Loss'),
        ('train_nll', 'val_nll', 'NLL Component'),
        ('train_rank', 'val_rank', 'Ranking Component'),
    ]):
        ax.plot(history[train_key], label='Train')
        ax.plot(history[val_key], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'DeepHit: {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'deephit_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cindex(cindex_results, figures_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = TIME_HORIZONS
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
    ax.set_title('DeepHit: Time-Dependent Concordance Index by Event Type')
    ax.set_xticks(x)
    ax.set_xticklabels([f'tau = {h}' for h in horizons])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figures_dir / 'deephit_time_dependent_cindex.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_survival_curves(cif_prepay, cif_default, surv_np, time_points, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    np.random.seed(42)
    sample_idx = np.random.choice(len(surv_np), size=5, replace=False)

    for ax, data, title, ylabel in zip(axes,
        [cif_prepay[sample_idx], cif_default[sample_idx], surv_np[sample_idx]],
        ['Prepayment CIF', 'Default CIF', 'Overall Survival S(t)'],
        ['Cumulative Incidence', 'Cumulative Incidence', 'Survival Probability'],
    ):
        for i, idx in enumerate(sample_idx):
            ax.plot(time_points, data[i], label=f'Loan {idx}', alpha=0.7)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'DeepHit: {title}')
        ax.legend(loc='lower right' if 'CIF' in title else 'lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(figures_dir / 'deephit_survival_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance(imp_prepay_df, imp_default_df, figures_dir, top_n=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, df, color, title in zip(axes,
        [imp_prepay_df, imp_default_df],
        ['steelblue', 'indianred'],
        ['Prepayment', 'Default'],
    ):
        plot_df = df.head(top_n).iloc[::-1]
        ax.barh(plot_df['feature'], plot_df['importance'],
                xerr=plot_df['std'], color=color, alpha=0.7, capsize=3)
        ax.set_xlabel('Importance (decrease in C-index)')
        ax.set_title(f'DeepHit Permutation Importance: {title}')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(figures_dir / 'deephit_feature_importance.png', dpi=150, bbox_inches='tight')
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

    results_file = output_dir / 'deephit_results.txt'

    # Logging
    start_time = time.time()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("DEEPHIT COMPETING RISKS")
    log("Lee et al. (2018) / Blumenstock et al. (2022) architecture")
    log("=" * 70)
    log(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch version: {torch.__version__}")

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        log(f"  CUDA version: {torch.version.cuda}")
        log(f"  GPUs available: {torch.cuda.device_count()}")
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
    log(f"  alpha: {args.alpha}")
    log(f"  sigma: {args.sigma}")
    log(f"  architecture: shared={args.shared_layers}x{args.shared_nodes}, "
        f"head={args.head_layers}x{args.head_nodes}")
    log(f"  dropout: {args.dropout}")
    log(f"  batch_norm: {not args.no_batch_norm}")
    log(f"  num_durations: {args.num_durations}")
    log(f"  seed: {args.seed}")

    # ==================== Load Data ====================
    log("\n" + "-" * 70)
    log("Loading data...")

    panel_df = pd.read_parquet(data_dir / 'loan_month_panel.parquet')
    log(f"  Loaded {len(panel_df):,} loan-months")
    log(f"  Unique loans: {panel_df['loan_sequence_number'].nunique():,}")

    # Features
    feature_cols = [f for f in ALL_FEATURES if f in panel_df.columns]
    missing = [f for f in ALL_FEATURES if f not in panel_df.columns]
    log(f"  Available features: {len(feature_cols)}/{len(ALL_FEATURES)}")
    if missing:
        log(f"  Missing: {missing}")

    # Terminal observations
    time_col, event_col = 'loan_age', 'event_code'
    panel_df = panel_df.sort_values(['loan_sequence_number', time_col])
    terminal_df = panel_df.groupby('loan_sequence_number').last().reset_index()
    log(f"  Terminal observations: {len(terminal_df):,}")

    # Feature engineering
    if 'bal_repaid' in feature_cols:
        log("  Lagging bal_repaid...")
        bal_repaid_lag = panel_df.groupby('loan_sequence_number').apply(
            lambda g: g['bal_repaid'].iloc[-2] if len(g) >= 2 else g['bal_repaid'].iloc[-1])
        terminal_df['bal_repaid_lag1'] = terminal_df['loan_sequence_number'].map(bal_repaid_lag)
        feature_cols = [f if f != 'bal_repaid' else 'bal_repaid_lag1' for f in feature_cols]

    if 'orig_upb' in terminal_df.columns:
        terminal_df['log_upb'] = np.log(terminal_df['orig_upb'])
        feature_cols = [f if f != 'orig_upb' else 'log_upb' for f in feature_cols]
        log("  Created log_upb")

    n_before = len(terminal_df)
    terminal_df = terminal_df.dropna(subset=feature_cols)
    log(f"  After dropping NaN: {len(terminal_df):,} (dropped {n_before - len(terminal_df):,})")

    # Split
    train_folds_actual = [f for f in TRAIN_FOLDS if f not in VAL_FOLDS]
    train_df = terminal_df[terminal_df['fold'].isin(train_folds_actual)].copy()
    val_df = terminal_df[terminal_df['fold'].isin(VAL_FOLDS)].copy()
    test_df = terminal_df[terminal_df['fold'] == TEST_FOLD].copy()

    log(f"\n  Train (folds {train_folds_actual}): {len(train_df):,}")
    log(f"  Val (fold {VAL_FOLDS}): {len(val_df):,}")
    log(f"  Test (fold {TEST_FOLD}): {len(test_df):,}")

    log("\n  Event distribution (train):")
    for code, count in train_df[event_col].value_counts().sort_index().items():
        log(f"    {EVENT_NAMES.get(code, 'Other')} (k={code}): {count:,}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype('float32')
    X_val = scaler.transform(val_df[feature_cols]).astype('float32')
    X_test = scaler.transform(test_df[feature_cols]).astype('float32')

    duration_train = train_df[time_col].values.astype('float32')
    duration_val = val_df[time_col].values.astype('float32')
    duration_test = test_df[time_col].values.astype('float32')

    event_train = train_df[event_col].values.astype('float32')
    event_val = val_df[event_col].values.astype('float32')
    event_test = test_df[event_col].values.astype('float32')

    log(f"\n  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    log(f"  Duration range: {duration_train.min():.0f} - {duration_train.max():.0f} months")

    # Time bins
    all_durations = np.concatenate([duration_train, duration_val, duration_test])
    time_bins = np.linspace(all_durations.min(), all_durations.max(), args.num_durations + 1)
    time_bins_tensor = torch.tensor(time_bins, dtype=torch.float32)
    time_points = (time_bins[:-1] + time_bins[1:]) / 2

    log(f"  Time bins: {args.num_durations}, width={np.diff(time_bins).mean():.2f} months")

    # ==================== Train ====================
    log("\n" + "-" * 70)
    log("Training DeepHit...")

    in_features = X_train.shape[1]
    model = DeepHitNetwork(
        in_features=in_features,
        num_time_bins=args.num_durations,
        num_causes=2,
        shared_layers=args.shared_layers,
        shared_nodes=args.shared_nodes,
        head_layers=args.head_layers,
        head_nodes=args.head_nodes,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Parameters: {n_params:,}")

    criterion = DeepHitLoss(alpha=args.alpha, sigma=args.sigma)

    train_start = time.time()
    history = train_deephit(
        model=model,
        criterion=criterion,
        X_train=X_train,
        y_train=duration_train,
        e_train=event_train,
        X_val=X_val,
        y_val=duration_val,
        e_val=event_val,
        time_bins=time_bins_tensor,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        device=device,
        log_fn=log,
    )
    train_time = time.time() - train_start
    log(f"\nTraining completed in {train_time/60:.1f} minutes")

    # ==================== Evaluate ====================
    log("\n" + "-" * 70)
    log("Evaluating model...")

    cindex_results, cif_prepay, cif_default, surv_np = compute_cindex(
        model, X_test, duration_test, event_test,
        duration_train, event_train, time_points, device, log)

    # ==================== Feature Importance ====================
    log("\n" + "-" * 70)
    log("Computing permutation importance...")

    imp_start = time.time()

    log("  Prepayment cause...")
    imp_prepay, imp_prepay_std = permutation_importance_deephit(
        model, X_test, duration_test, event_test, feature_cols,
        cause_idx=0, device=device, n_repeats=args.importance_repeats)

    imp_prepay_df = pd.DataFrame({
        'feature': feature_cols, 'importance': imp_prepay, 'std': imp_prepay_std
    }).sort_values('importance', ascending=False)

    log("  Default cause...")
    imp_default, imp_default_std = permutation_importance_deephit(
        model, X_test, duration_test, event_test, feature_cols,
        cause_idx=1, device=device, n_repeats=args.importance_repeats)

    imp_default_df = pd.DataFrame({
        'feature': feature_cols, 'importance': imp_default, 'std': imp_default_std
    }).sort_values('importance', ascending=False)

    imp_time = time.time() - imp_start
    log(f"  Importance computed in {imp_time/60:.1f} minutes")

    log("\n  Top 5 Features - PREPAYMENT:")
    for _, row in imp_prepay_df.head(5).iterrows():
        log(f"    {row['feature']:<25} {row['importance']:>8.4f} +/- {row['std']:.4f}")

    log("\n  Top 5 Features - DEFAULT:")
    for _, row in imp_default_df.head(5).iterrows():
        log(f"    {row['feature']:<25} {row['importance']:>8.4f} +/- {row['std']:.4f}")

    # ==================== Plots ====================
    log("\n" + "-" * 70)
    log("Generating plots...")

    plot_training_curves(history, figures_dir)
    log(f"  {figures_dir / 'deephit_training_curves.png'}")

    plot_cindex(cindex_results, figures_dir)
    log(f"  {figures_dir / 'deephit_time_dependent_cindex.png'}")

    plot_survival_curves(cif_prepay, cif_default, surv_np, time_points, figures_dir)
    log(f"  {figures_dir / 'deephit_survival_curves.png'}")

    plot_importance(imp_prepay_df, imp_default_df, figures_dir)
    log(f"  {figures_dir / 'deephit_feature_importance.png'}")

    # ==================== Summary ====================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\nArchitecture:")
    log(f"  Shared FFN: {args.shared_layers} layers x {args.shared_nodes} nodes")
    log(f"  Residual: shared_output + proj(input)")
    log(f"  Prepay head: {args.head_layers} layers x {args.head_nodes} nodes "
        f"-> {args.num_durations} time bins")
    log(f"  Default head: {args.head_layers} layers x {args.head_nodes} nodes "
        f"-> {args.num_durations} time bins")
    log(f"  Output: Joint softmax over {2 * args.num_durations} (cause, time)")
    log(f"  Parameters: {n_params:,}")

    log(f"\nData:")
    log(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    log(f"  Features: {len(feature_cols)}")

    log(f"\nPerformance (Test Set):")
    for cause_name in ['Prepay', 'Default']:
        cr = cindex_results[cause_name]
        log(f"  {cause_name}:")
        log(f"    Harrell C-index: {cr['harrell']:.4f}")
        for tau, c in cr['ipcw'].items():
            log(f"    IPCW C-index (tau={tau}): {c:.4f}")

    log(f"\nTop 3 Features:")
    log(f"  Prepay: {', '.join(imp_prepay_df['feature'].head(3).tolist())}")
    log(f"  Default: {', '.join(imp_default_df['feature'].head(3).tolist())}")

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
        'num_time_bins': args.num_durations,
        'num_causes': 2,
        'shared_layers': args.shared_layers,
        'shared_nodes': args.shared_nodes,
        'head_layers': args.head_layers,
        'head_nodes': args.head_nodes,
        'dropout': args.dropout,
        'batch_norm': not args.no_batch_norm,
    }, models_dir / 'deephit_joint.pt')
    log(f"  Model: {models_dir / 'deephit_joint.pt'}")

    # Scaler
    with open(models_dir / 'deephit_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Time bins
    np.save(models_dir / 'deephit_time_bins.npy', time_bins)

    # Feature columns
    with open(models_dir / 'deephit_feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    # Training history
    with open(models_dir / 'deephit_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Importance CSVs
    imp_prepay_df.to_csv(models_dir / 'deephit_importance_prepay.csv', index=False)
    imp_default_df.to_csv(models_dir / 'deephit_importance_default.csv', index=False)

    # Text report
    with open(results_file, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"  Report: {results_file}")

    # C-index results
    cindex_df = pd.DataFrame({
        cause: {f'ipcw_{tau}': cr['ipcw'].get(tau, np.nan) for tau in TIME_HORIZONS}
            | {'harrell': cr['harrell']}
        for cause, cr in cindex_results.items()
    })
    cindex_df.to_csv(models_dir / 'deephit_cindex.csv')
    log(f"  C-index: {models_dir / 'deephit_cindex.csv'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
