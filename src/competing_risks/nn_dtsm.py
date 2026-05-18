"""
Neural Network Discrete-Time Survival Model with APC Decomposition.

Extension of Wang et al. (2024) to competing risks (prepayment and default).
One separate MLP per vintage quarter, each with a 3-class softmax output.

Stage 1: Per-vintage NN-DTSM predicts monthly transition probabilities.
Stage 2: APC decomposition via ridge regression on the Lexis graph.
Stage 3: Calendar-time macro regression + AR projection for out-of-sample.

Reference:
    Wang, H., Bellotti, A., Qu, R. & Bai, R. (2024). "Discrete-Time Survival
    Models with Neural Networks for Age-Period-Cohort Analysis of Credit Risk."
    Risks, 12(2), 31.
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Constants
# =============================================================================

STATIC_FEATURES = ['fico_score', 'ltv_r', 'dti_r', 'int_rate', 'log_upb']

BEHAVIORAL_FEATURES = [
    'bal_repaid_lag1', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m',
]

MACRO_FEATURES = [
    'hpi_st_d_t_o', 'hpi_st_log12m', 'st_unemp_r12m', 'st_unemp_r3m',
    'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o', 'T10Y3MM',
]

DEFAULT_INPUT_FEATURES = STATIC_FEATURES + BEHAVIORAL_FEATURES

TRAIN_FOLDS = list(range(9))
VAL_FOLDS = [9]
TEST_FOLD = 10


# =============================================================================
# Device detection
# =============================================================================

def get_device(prefer: str = 'auto') -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if prefer == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(prefer)


# =============================================================================
# Vintage quarter extraction
# =============================================================================

def extract_vintage_quarter(loan_sequence_number: pd.Series) -> pd.Series:
    """
    Extract vintage quarter from loan sequence number.

    Format: F{YY}Q{Q}xxxxxx  (e.g., F06Q10000059 -> 2006Q1)

    Parameters
    ----------
    loan_sequence_number : pd.Series of str

    Returns
    -------
    pd.Series of str like '2006Q1'
    """
    def _parse(lsn):
        m = re.match(r'F(\d{2})Q(\d)', str(lsn))
        if m:
            yy, q = int(m.group(1)), int(m.group(2))
            year = 2000 + yy if yy < 50 else 1900 + yy
            return f'{year}Q{q}'
        return None

    return loan_sequence_number.map(_parse)


# =============================================================================
# Data preparation
# =============================================================================

def prepare_panel_features(
    panel: pd.DataFrame,
    input_features: List[str] = None,
) -> pd.DataFrame:
    """
    Prepare panel with derived features needed for NN-DTSM.

    Creates log_upb and bal_repaid_lag1 if missing, and extracts
    vintage_quarter from loan_sequence_number.

    Parameters
    ----------
    panel : pd.DataFrame
        Loan-month panel.
    input_features : list[str]
        Features to use as NN inputs.

    Returns
    -------
    pd.DataFrame with added columns.
    """
    df = panel.copy()

    if 'log_upb' not in df.columns and 'orig_upb' in df.columns:
        df['log_upb'] = np.log(df['orig_upb'].clip(lower=1))

    if 'bal_repaid_lag1' not in df.columns and 'bal_repaid' in df.columns:
        df = df.sort_values(['loan_sequence_number', 'loan_age'])
        df['bal_repaid_lag1'] = df.groupby('loan_sequence_number')['bal_repaid'].shift(1)
        df['bal_repaid_lag1'] = df['bal_repaid_lag1'].fillna(0.0)

    if 'vintage_quarter' not in df.columns and 'loan_sequence_number' in df.columns:
        df['vintage_quarter'] = extract_vintage_quarter(df['loan_sequence_number'])

    return df


def compute_class_weights(
    event_codes: np.ndarray,
    n_classes: int = 3,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for cross-entropy loss.

    w_k = N / (n_classes * N_k)

    Following Hussin Adam Khatir & Bee (2022): cost-sensitive weighting
    matches resampling performance while retaining full data.

    Parameters
    ----------
    event_codes : array of int (0=current, 1=prepay, 2=default)
    n_classes : int

    Returns
    -------
    torch.Tensor of shape (n_classes,)
    """
    counts = np.bincount(event_codes, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(event_codes) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# Neural network architecture
# =============================================================================

class _VintageSubnet(nn.Module):
    """Single vintage subnetwork: MLP with 3-class output."""

    def __init__(self, input_dim: int, n_hidden: int, n_neurons: int,
                 dropout: float):
        super().__init__()
        layers = [nn.Dropout(dropout)]
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(in_dim, n_neurons), nn.ReLU()])
            in_dim = n_neurons
        layers.append(nn.Linear(n_neurons, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities for NLLLoss."""
        return F.log_softmax(self.net(x), dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (softmax)."""
        return F.softmax(self.net(x), dim=-1)


# =============================================================================
# Training utilities
# =============================================================================

@torch.no_grad()
def _evaluate_loss(model, loader, criterion, device):
    """Compute average loss over a DataLoader."""
    model.eval()
    total, n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        total += criterion(model(X_b), y_b).item() * X_b.size(0)
        n += X_b.size(0)
    return total / n if n > 0 else float('inf')


def _train_subnet(
    model: _VintageSubnet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 20,
    patience: int = 10,
    class_weights: Optional[torch.Tensor] = None,
    device: torch.device = None,
    verbose: bool = False,
) -> Dict[str, list]:
    """
    Train one vintage subnetwork.

    Returns dict with 'train_loss', 'val_loss' per epoch.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    if class_weights is not None:
        criterion = nn.NLLLoss(weight=class_weights.to(device))
    else:
        criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss, n_samples = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
            n_samples += X_b.size(0)

        train_loss = running_loss / n_samples
        history['train_loss'].append(train_loss)

        val_loss = (_evaluate_loss(model, val_loader, criterion, device)
                    if val_loader else train_loss)
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return history


# =============================================================================
# VintageNNDTSM — main model class
# =============================================================================

class VintageNNDTSM:
    """
    Neural Network Discrete-Time Survival Model with vintage-level subnetworks.

    Extension of Wang et al. (2024) to competing risks.
    One separate MLP per vintage quarter, each with a 3-class softmax output
    (current, prepay, default).

    Parameters
    ----------
    n_hidden_layers : int
        Hidden layers per subnetwork.
    n_neurons : int
        Neurons per hidden layer.
    dropout : float
        Dropout rate after input layer.
    n_epochs : int
        Training epochs per subnetwork.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation.
    patience : int
        Early stopping patience.
    use_class_weights : bool
        Use inverse-frequency class weights (Hussin Adam Khatir & Bee, 2022).
    val_frac : float
        Fraction of vintage data for validation (within-vintage split).
    device : str
        'auto', 'cuda', 'mps', or 'cpu'.
    random_seed : int
    """

    def __init__(
        self,
        n_hidden_layers: int = 4,
        n_neurons: int = 8,
        dropout: float = 0.0,
        n_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        use_class_weights: bool = True,
        val_frac: float = 0.25,
        device: str = 'auto',
        random_seed: int = 42,
    ):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.use_class_weights = use_class_weights
        self.val_frac = val_frac
        self.device_name = device
        self.random_seed = random_seed

        # Fitted attributes
        self.subnets_: Dict[str, _VintageSubnet] = {}
        self.scalers_: Dict[str, StandardScaler] = {}
        self.input_features_: Optional[List[str]] = None
        self.vintage_quarters_: Optional[List[str]] = None
        self.histories_: Dict[str, Dict[str, list]] = {}
        self.device_: Optional[torch.device] = None

    def fit(
        self,
        panel_df: pd.DataFrame,
        input_features: List[str] = None,
        vintage_col: str = 'vintage_quarter',
        log_fn=print,
    ) -> 'VintageNNDTSM':
        """
        Train one separate MLP per vintage quarter.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Loan-month panel with event_code, loan_age, and feature columns.
        input_features : list[str]
            Feature columns to use as NN inputs.
        vintage_col : str
            Column identifying vintage quarter.
        log_fn : callable
            Logging function.

        Returns
        -------
        self
        """
        if input_features is None:
            input_features = DEFAULT_INPUT_FEATURES
        self.input_features_ = input_features
        self.device_ = get_device(self.device_name)

        vintages = sorted(panel_df[vintage_col].dropna().unique())
        self.vintage_quarters_ = vintages

        log_fn(f"Training {len(vintages)} vintage subnetworks on {self.device_}")
        log_fn(f"Architecture: {self.n_hidden_layers} layers × "
               f"{self.n_neurons} neurons, dropout={self.dropout}")

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        for i, vq in enumerate(vintages):
            vdf = panel_df[panel_df[vintage_col] == vq].copy()

            # Check available features
            avail = [c for c in input_features if c in vdf.columns]
            if len(avail) < len(input_features):
                missing = set(input_features) - set(avail)
                log_fn(f"  Warning: vintage {vq} missing features: {missing}")
            if not avail:
                log_fn(f"  Skipping vintage {vq}: no features available")
                continue

            X = vdf[avail].values.astype(np.float32)
            y = vdf['event_code'].values.astype(np.int64)

            # Handle NaN
            nan_mask = np.isnan(X).any(axis=1)
            if nan_mask.any():
                X = X[~nan_mask]
                y = y[~nan_mask]

            if len(X) < 20:
                log_fn(f"  Skipping vintage {vq}: only {len(X)} observations")
                continue

            # Per-vintage standardisation
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scalers_[vq] = scaler

            # Within-vintage train/val split (by loan)
            loan_ids = vdf.loc[~nan_mask.values if nan_mask.any() else
                               slice(None), 'loan_sequence_number'].values \
                       if 'loan_sequence_number' in vdf.columns \
                       else np.arange(len(X))
            unique_loans = np.unique(loan_ids)
            n_val = max(1, int(len(unique_loans) * self.val_frac))
            rng = np.random.RandomState(self.random_seed + i)
            val_loans = set(rng.choice(unique_loans, n_val, replace=False))
            val_mask = np.isin(loan_ids, list(val_loans))

            X_train, y_train = X[~val_mask], y[~val_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Class weights
            cw = compute_class_weights(y_train) if self.use_class_weights else None

            # DataLoaders
            train_ds = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                      shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                    shuffle=False)

            # Build and train subnet
            subnet = _VintageSubnet(
                input_dim=len(avail),
                n_hidden=self.n_hidden_layers,
                n_neurons=self.n_neurons,
                dropout=self.dropout,
            )

            history = _train_subnet(
                subnet, train_loader, val_loader,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                n_epochs=self.n_epochs,
                patience=self.patience,
                class_weights=cw,
                device=self.device_,
                verbose=False,
            )

            self.subnets_[vq] = subnet
            self.histories_[vq] = history

            n_events = np.bincount(y, minlength=3)
            best_val = min(history['val_loss'])
            epochs_run = len(history['val_loss'])
            if (i + 1) % max(1, len(vintages) // 20) == 0 or (i + 1) == len(vintages):
                log_fn(f"  [{i+1}/{len(vintages)}] {vq}: "
                       f"{len(X):,} obs, events=[{n_events[0]},{n_events[1]},{n_events[2]}], "
                       f"val_loss={best_val:.4f}, epochs={epochs_run}")

        log_fn(f"Done. Trained {len(self.subnets_)} subnetworks.")
        return self

    @torch.no_grad()
    def predict_monthly(
        self,
        panel_df: pd.DataFrame,
        vintage_col: str = 'vintage_quarter',
    ) -> pd.DataFrame:
        """
        Predict P(current), P(prepay), P(default) for each loan-month.

        Routes each observation to its vintage's subnetwork.

        Parameters
        ----------
        panel_df : pd.DataFrame
        vintage_col : str

        Returns
        -------
        pd.DataFrame with columns p_current, p_prepay, p_default added.
        """
        result = panel_df.copy()
        result['p_current'] = np.nan
        result['p_prepay'] = np.nan
        result['p_default'] = np.nan

        for vq, subnet in self.subnets_.items():
            mask = result[vintage_col] == vq
            if not mask.any():
                continue

            avail = [c for c in self.input_features_ if c in result.columns]
            X = result.loc[mask, avail].values.astype(np.float32)

            # Handle NaN
            nan_rows = np.isnan(X).any(axis=1)
            if nan_rows.all():
                continue

            X_clean = X.copy()
            X_clean[nan_rows] = 0.0

            if vq in self.scalers_:
                X_clean = self.scalers_[vq].transform(X_clean)

            subnet.eval()
            subnet.to(self.device_)
            X_t = torch.tensor(X_clean, dtype=torch.float32, device=self.device_)
            probs = subnet.predict_proba(X_t).cpu().numpy()

            probs[nan_rows] = np.nan
            idx = result.index[mask]
            result.loc[idx, 'p_current'] = probs[:, 0]
            result.loc[idx, 'p_prepay'] = probs[:, 1]
            result.loc[idx, 'p_default'] = probs[:, 2]

        return result

    @torch.no_grad()
    def predict_cif(
        self,
        panel_df: pd.DataFrame,
        eval_times: List[int] = None,
        vintage_col: str = 'vintage_quarter',
    ) -> Dict[str, np.ndarray]:
        """
        Compute CIF by chaining monthly transition probabilities.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Loan-month panel (must be sorted by loan_sequence_number, loan_age).
        eval_times : list[int]
            Horizons for CIF snapshots (default: [24, 48, 72]).
        vintage_col : str

        Returns
        -------
        dict with 'loan_sequence_number', 'duration', 'event_code',
             'cif_prepay_{t}', 'cif_default_{t}' for each t in eval_times.
        """
        if eval_times is None:
            eval_times = [24, 48, 72]

        # Get monthly predictions
        pred_df = self.predict_monthly(panel_df, vintage_col=vintage_col)

        # Group by loan, chain monthly probs
        loan_groups = pred_df.sort_values(['loan_sequence_number', 'loan_age']) \
                             .groupby('loan_sequence_number')

        loan_ids = []
        durations = []
        event_codes = []
        cif_arrays = {f'cif_prepay_{t}': [] for t in eval_times}
        cif_arrays.update({f'cif_default_{t}': [] for t in eval_times})

        for loan_id, grp in loan_groups:
            p_current = grp['p_current'].values
            p_prepay = grp['p_prepay'].values
            p_default = grp['p_default'].values

            # Skip loans with all NaN
            if np.isnan(p_current).all():
                continue

            # Fill NaN with baseline rates
            nan_mask = np.isnan(p_current)
            if nan_mask.any():
                p_current[nan_mask] = 0.98
                p_prepay[nan_mask] = 0.015
                p_default[nan_mask] = 0.005

            # Chain CIF
            surv = 1.0
            cif_p, cif_d = 0.0, 0.0
            snap_p = {}
            snap_d = {}
            for month_idx in range(len(p_current)):
                cif_p += surv * p_prepay[month_idx]
                cif_d += surv * p_default[month_idx]
                surv *= p_current[month_idx]

                age = month_idx + 1
                if age in eval_times:
                    snap_p[age] = cif_p
                    snap_d[age] = cif_d

            # For horizons beyond observed data, use last value
            for t in eval_times:
                if t not in snap_p:
                    snap_p[t] = cif_p
                    snap_d[t] = cif_d

            loan_ids.append(loan_id)
            durations.append(grp['loan_age'].iloc[-1])
            event_codes.append(grp['event_code'].iloc[-1])

            for t in eval_times:
                cif_arrays[f'cif_prepay_{t}'].append(snap_p[t])
                cif_arrays[f'cif_default_{t}'].append(snap_d[t])

        result = {
            'loan_sequence_number': np.array(loan_ids),
            'duration': np.array(durations),
            'event_code': np.array(event_codes),
        }
        for key, vals in cif_arrays.items():
            result[key] = np.array(vals)

        return result

    def build_lexis_data(
        self,
        panel_df: pd.DataFrame,
        vintage_col: str = 'vintage_quarter',
    ) -> Dict[int, pd.DataFrame]:
        """
        Construct Lexis graph data: cell-level (age, vintage) means
        of predicted probabilities.

        Parameters
        ----------
        panel_df : pd.DataFrame
        vintage_col : str

        Returns
        -------
        dict mapping cause_code -> DataFrame with columns
            [loan_age, vintage_quarter, calendar_month, mean_prob, n_at_risk].
        """
        pred_df = self.predict_monthly(panel_df, vintage_col=vintage_col)

        lexis = {}
        for cause, col in [(1, 'p_prepay'), (2, 'p_default')]:
            cells = (
                pred_df.dropna(subset=[col])
                .groupby(['loan_age', vintage_col])
                .agg(
                    mean_prob=(col, 'mean'),
                    n_at_risk=(col, 'count'),
                    year_month=('year_month', 'first'),
                )
                .reset_index()
                .rename(columns={vintage_col: 'vintage_quarter'})
            )
            cells['calendar_month'] = cells['year_month']
            lexis[cause] = cells

        return lexis

    def mcfadden_pseudo_r2(
        self,
        panel_df: pd.DataFrame,
        vintage_col: str = 'vintage_quarter',
    ) -> Dict[str, float]:
        """
        Compute McFadden pseudo-R² per vintage.

        R² = 1 - LL(model) / LL(null)

        Returns dict mapping vintage -> R².
        """
        pred_df = self.predict_monthly(panel_df, vintage_col=vintage_col)
        results = {}

        for vq in self.subnets_:
            mask = pred_df[vintage_col] == vq
            vdf = pred_df[mask].dropna(subset=['p_current'])
            if len(vdf) == 0:
                continue

            y = vdf['event_code'].values
            p = np.column_stack([
                vdf['p_current'].values,
                vdf['p_prepay'].values,
                vdf['p_default'].values,
            ])
            p = np.clip(p, 1e-15, 1.0)

            # Model log-likelihood
            ll_model = np.sum(np.log(p[np.arange(len(y)), y]))

            # Null model: class frequencies
            freq = np.bincount(y, minlength=3).astype(float)
            freq /= freq.sum()
            freq = np.clip(freq, 1e-15, 1.0)
            ll_null = np.sum(np.log(freq[y]))

            r2 = 1.0 - ll_model / ll_null if ll_null != 0 else 0.0
            results[vq] = r2

        return results

    def save(self, path: str):
        """Save all subnetworks and scalers."""
        state = {
            'config': {
                'n_hidden_layers': self.n_hidden_layers,
                'n_neurons': self.n_neurons,
                'dropout': self.dropout,
                'n_epochs': self.n_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'patience': self.patience,
                'use_class_weights': self.use_class_weights,
                'val_frac': self.val_frac,
                'random_seed': self.random_seed,
            },
            'input_features': self.input_features_,
            'vintage_quarters': self.vintage_quarters_,
            'subnets': {vq: net.state_dict()
                        for vq, net in self.subnets_.items()},
            'scalers': {vq: {
                'mean': sc.mean_.tolist(),
                'scale': sc.scale_.tolist(),
                'var': sc.var_.tolist(),
                'n_features': sc.n_features_in_,
            } for vq, sc in self.scalers_.items()},
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'VintageNNDTSM':
        """Load model from file."""
        dev = get_device(device)
        state = torch.load(path, map_location=dev, weights_only=False)
        cfg = state['config']
        model = cls(**cfg, device=device)
        model.input_features_ = state['input_features']
        model.vintage_quarters_ = state['vintage_quarters']
        model.device_ = dev

        n_features = len(model.input_features_)
        for vq, sd in state['subnets'].items():
            subnet = _VintageSubnet(
                input_dim=n_features,
                n_hidden=cfg['n_hidden_layers'],
                n_neurons=cfg['n_neurons'],
                dropout=cfg['dropout'],
            )
            subnet.load_state_dict(sd)
            subnet.to(dev)
            model.subnets_[vq] = subnet

        for vq, sc_data in state['scalers'].items():
            sc = StandardScaler()
            sc.mean_ = np.array(sc_data['mean'])
            sc.scale_ = np.array(sc_data['scale'])
            sc.var_ = np.array(sc_data['var'])
            sc.n_features_in_ = sc_data['n_features']
            model.scalers_[vq] = sc

        return model


# =============================================================================
# APC Decomposition via ridge regression
# =============================================================================

class APCDecomposition:
    """
    Ridge regression APC decomposition of Lexis graph data.

    Decomposes cell-level NN predictions into:
        D_vt = sum(alpha_t) + sum(beta_v) + sum(gamma_c) + eps

    with ridge regularisation to handle the APC identification problem.

    Parameters
    ----------
    ridge_alphas : list[float]
        Candidate regularisation strengths for RidgeCV.
    """

    def __init__(
        self,
        ridge_alphas: Optional[List[float]] = None,
    ):
        if ridge_alphas is None:
            self.ridge_alphas = np.logspace(-2, 4, 50).tolist()
        else:
            self.ridge_alphas = ridge_alphas

        # Fitted attributes
        self.ridge_model_: Optional[RidgeCV] = None
        self.age_effects_: Optional[pd.Series] = None
        self.vintage_effects_: Optional[pd.Series] = None
        self.calendar_effects_: Optional[pd.Series] = None
        self.age_levels_: Optional[np.ndarray] = None
        self.vintage_levels_: Optional[np.ndarray] = None
        self.calendar_levels_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.cells_: Optional[pd.DataFrame] = None
        self.macro_regression_ = None
        self._macro_cols: Optional[List[str]] = None
        self._macro_lags: Optional[Dict[str, int]] = None
        self._ar_models: Optional[Dict[str, object]] = None

    def fit(
        self,
        lexis_df: pd.DataFrame,
        age_col: str = 'loan_age',
        vintage_col: str = 'vintage_quarter',
        calendar_col: str = 'calendar_month',
        prob_col: str = 'mean_prob',
        weight_col: str = 'n_at_risk',
        log_fn=print,
    ) -> 'APCDecomposition':
        """
        Fit ridge regression APC decomposition.

        Parameters
        ----------
        lexis_df : pd.DataFrame
            Cell-level data from VintageNNDTSM.build_lexis_data().
        age_col, vintage_col, calendar_col : str
            Column names.
        prob_col : str
            Column with cell-level predicted probability.
        weight_col : str
            Column with cell counts (for weighting).
        log_fn : callable

        Returns
        -------
        self
        """
        df = lexis_df.dropna(subset=[prob_col]).copy()
        self.cells_ = df.copy()

        # Get unique levels
        self.age_levels_ = np.sort(df[age_col].unique())
        self.vintage_levels_ = np.sort(df[vintage_col].unique())
        self.calendar_levels_ = np.sort(df[calendar_col].unique())

        n_age = len(self.age_levels_)
        n_vin = len(self.vintage_levels_)
        n_cal = len(self.calendar_levels_)

        log_fn(f"  APC: {n_age} age levels, {n_vin} vintage levels, "
               f"{n_cal} calendar levels, {len(df)} cells")

        # Build indicator matrix
        age_map = {v: i for i, v in enumerate(self.age_levels_)}
        vin_map = {v: i for i, v in enumerate(self.vintage_levels_)}
        cal_map = {v: i for i, v in enumerate(self.calendar_levels_)}

        n_rows = len(df)
        n_cols = n_age + n_vin + n_cal

        X = np.zeros((n_rows, n_cols), dtype=np.float64)
        for i, (_, row) in enumerate(df.iterrows()):
            a_idx = age_map.get(row[age_col])
            v_idx = vin_map.get(row[vintage_col])
            c_idx = cal_map.get(row[calendar_col])
            if a_idx is not None:
                X[i, a_idx] = 1.0
            if v_idx is not None:
                X[i, n_age + v_idx] = 1.0
            if c_idx is not None:
                X[i, n_age + n_vin + c_idx] = 1.0

        y = df[prob_col].values.astype(np.float64)
        weights = df[weight_col].values.astype(np.float64) if weight_col in df.columns \
                  else np.ones(n_rows)

        # Weighted ridge regression
        sqrt_w = np.sqrt(weights)
        X_w = X * sqrt_w[:, None]
        y_w = y * sqrt_w

        ridge = RidgeCV(alphas=self.ridge_alphas, fit_intercept=True)
        ridge.fit(X_w, y_w)
        self.ridge_model_ = ridge
        self.intercept_ = ridge.intercept_

        log_fn(f"  Ridge alpha={ridge.alpha_:.4f}")

        # Extract effects
        coefs = ridge.coef_
        age_coefs = coefs[:n_age]
        vin_coefs = coefs[n_age:n_age + n_vin]
        cal_coefs = coefs[n_age + n_vin:]

        # Zero-mean centering
        age_coefs -= age_coefs.mean()
        vin_coefs -= vin_coefs.mean()
        cal_coefs -= cal_coefs.mean()

        self.age_effects_ = pd.Series(age_coefs, index=self.age_levels_,
                                      name='age_effect')
        self.vintage_effects_ = pd.Series(vin_coefs, index=self.vintage_levels_,
                                          name='vintage_effect')
        self.calendar_effects_ = pd.Series(cal_coefs, index=self.calendar_levels_,
                                           name='calendar_effect')

        return self

    def get_effects(self) -> Dict[str, pd.Series]:
        """Return age, vintage, calendar effect Series."""
        return {
            'age': self.age_effects_,
            'vintage': self.vintage_effects_,
            'calendar': self.calendar_effects_,
        }

    def fit_macro_regression(
        self,
        macro_df: pd.DataFrame,
        macro_cols: List[str] = None,
        max_lag_months: int = 36,
        log_fn=print,
    ) -> 'APCDecomposition':
        """
        Fit calendar-time coefficients against macro variables with lag selection.

        Step 1: For each macro variable, find the lag (0..max_lag_months) that
                maximises univariate R² against the calendar-time effects.
        Step 2: Fit multivariate regression with best lags + time trend.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Must have a time index or 'year_month' column matching
            calendar_effects_ index, plus macro variable columns.
        macro_cols : list[str]
            Macro variable columns. If None, uses MACRO_FEATURES.
        max_lag_months : int
            Maximum lag for univariate lag selection.
        log_fn : callable

        Returns
        -------
        self
        """
        import statsmodels.api as sm

        if macro_cols is None:
            macro_cols = MACRO_FEATURES
        avail_cols = [c for c in macro_cols if c in macro_df.columns]
        if not avail_cols:
            raise ValueError("No macro columns found in macro_df.")

        # Align calendar effects with macro data
        cal = self.calendar_effects_.copy()
        cal.name = 'gamma'
        cal_df = cal.reset_index()
        cal_df.columns = ['calendar_month', 'gamma']

        # Merge macro data
        if 'year_month' in macro_df.columns:
            macro_by_time = macro_df.groupby('year_month')[avail_cols].mean().reset_index()
            merged = cal_df.merge(macro_by_time, left_on='calendar_month',
                                  right_on='year_month', how='inner')
        else:
            merged = cal_df.merge(macro_df[avail_cols].reset_index(),
                                  left_on='calendar_month',
                                  right_on='index', how='inner')

        # Step 1: Univariate lag selection
        best_lags = {}
        log_fn("  Lag selection:")
        for col in avail_cols:
            best_r2, best_lag = -np.inf, 0
            for lag in range(0, min(max_lag_months + 1, len(merged) // 3)):
                if lag > 0:
                    y_shifted = merged['gamma'].values[lag:]
                    x_shifted = merged[col].values[:-lag]
                else:
                    y_shifted = merged['gamma'].values
                    x_shifted = merged[col].values

                valid = ~(np.isnan(y_shifted) | np.isnan(x_shifted))
                if valid.sum() < 5:
                    continue

                X_reg = sm.add_constant(x_shifted[valid])
                try:
                    r2 = sm.OLS(y_shifted[valid], X_reg).fit().rsquared
                except Exception:
                    continue

                if r2 > best_r2:
                    best_r2, best_lag = r2, lag

            best_lags[col] = best_lag
            log_fn(f"    {col}: best lag = {best_lag} months, R² = {best_r2:.4f}")

        self._macro_lags = best_lags

        # Step 2: Multivariate regression with best lags + time trend
        max_lag = max(best_lags.values()) if best_lags else 0
        if max_lag > 0:
            gamma_vals = merged['gamma'].values[max_lag:]
        else:
            gamma_vals = merged['gamma'].values

        X_multi = np.column_stack([
            merged[col].values[max_lag - best_lags[col]:
                               len(merged) - best_lags[col] if best_lags[col] > 0
                               else len(merged)][:len(gamma_vals)]
            for col in avail_cols
        ])

        # Add time trend
        time_trend = np.arange(len(gamma_vals), dtype=np.float64)
        X_multi = np.column_stack([X_multi, time_trend])

        valid = ~(np.isnan(gamma_vals) | np.isnan(X_multi).any(axis=1))
        X_multi = sm.add_constant(X_multi[valid])
        gamma_vals = gamma_vals[valid]

        model = sm.OLS(gamma_vals, X_multi).fit()
        self.macro_regression_ = model
        self._macro_cols = avail_cols

        log_fn(f"  Macro regression R² = {model.rsquared:.4f}, "
               f"Adj R² = {model.rsquared_adj:.4f}")

        return self

    def fit_ar_models(
        self,
        macro_df: pd.DataFrame,
        macro_cols: List[str] = None,
        max_lag: int = 4,
        log_fn=print,
    ) -> 'APCDecomposition':
        """
        Fit AR(p) models on each macro variable for forward projection.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Time-indexed macro data (one row per calendar month).
        macro_cols : list[str]
        max_lag : int
            Maximum AR lag order (selected via BIC).
        log_fn : callable

        Returns
        -------
        self
        """
        from statsmodels.tsa.ar_model import AutoReg, ar_select_order

        if macro_cols is None:
            macro_cols = self._macro_cols or MACRO_FEATURES
        avail = [c for c in macro_cols if c in macro_df.columns]

        if 'year_month' in macro_df.columns:
            ts = macro_df.groupby('year_month')[avail].mean().sort_index()
        else:
            ts = macro_df[avail].sort_index()

        self._ar_models = {}
        for col in avail:
            series = ts[col].dropna().values.astype(np.float64)
            if len(series) < max_lag + 5:
                log_fn(f"  AR skip {col}: too few observations ({len(series)})")
                continue
            if np.std(series) < 1e-10:
                continue

            try:
                sel = ar_select_order(series, maxlag=max_lag, ic='bic',
                                      old_names=False)
                best_p = max(sel.ar_lags) if sel.ar_lags else 1
            except Exception:
                best_p = 1

            ar_fit = AutoReg(series, lags=best_p, old_names=False).fit()
            self._ar_models[col] = ar_fit
            log_fn(f"  AR({best_p}) for {col}: σ={np.sqrt(ar_fit.sigma2):.4f}")

        return self

    def project_calendar_effect(
        self,
        n_months_ahead: int = 72,
        n_mc_paths: int = 500,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Project the calendar-time effect forward using AR-simulated macro paths.

        Parameters
        ----------
        n_months_ahead : int
        n_mc_paths : int
            Number of Monte Carlo macro paths.
        seed : int

        Returns
        -------
        pd.DataFrame with columns [month_ahead, gamma_mean, gamma_lo, gamma_hi]
            where lo/hi are 5th/95th percentiles.
        """
        if self._ar_models is None:
            raise RuntimeError("Call fit_ar_models() first.")
        if self.macro_regression_ is None:
            raise RuntimeError("Call fit_macro_regression() first.")

        import statsmodels.api as sm

        rng = np.random.RandomState(seed)

        # Simulate forward macro paths
        macro_paths = {}  # col -> (n_mc_paths, n_months_ahead)
        for col, ar_fit in self._ar_models.items():
            sigma = np.sqrt(ar_fit.sigma2)
            params = ar_fit.params
            p = len(params) - 1
            last_vals = ar_fit.model.endog[-p:]

            sims = np.zeros((n_mc_paths, n_months_ahead))
            for s in range(n_mc_paths):
                history = list(last_vals.copy())
                for t in range(n_months_ahead):
                    pred = params[0]
                    for lag in range(1, p + 1):
                        pred += params[lag] * history[-lag]
                    pred += rng.normal(0, sigma)
                    history.append(pred)
                    sims[s, t] = pred
            macro_paths[col] = sims

        # For each MC path, compute projected gamma via the macro regression
        avail_cols = self._macro_cols
        n_regressors = len(avail_cols) + 1 + 1  # macro + trend + const
        gamma_projections = np.zeros((n_mc_paths, n_months_ahead))

        # Last observed time index for the trend continuation
        n_train = self.macro_regression_.nobs
        max_lag = max(self._macro_lags.values()) if self._macro_lags else 0

        for s in range(n_mc_paths):
            for t in range(n_months_ahead):
                x_row = []
                for col in avail_cols:
                    lag = self._macro_lags.get(col, 0)
                    effective_t = t - lag
                    if effective_t >= 0 and col in macro_paths:
                        x_row.append(macro_paths[col][s, effective_t])
                    elif col in macro_paths:
                        # Use last observed value for the lagged period
                        ar_fit = self._ar_models[col]
                        x_row.append(ar_fit.model.endog[-1])
                    else:
                        x_row.append(0.0)

                # Time trend continuation
                x_row.append(float(n_train + t))

                X_pred = np.array([1.0] + x_row)  # add constant
                gamma_projections[s, t] = self.macro_regression_.predict(
                    X_pred.reshape(1, -1))[0]

        # Summarise
        results = pd.DataFrame({
            'month_ahead': np.arange(1, n_months_ahead + 1),
            'gamma_mean': gamma_projections.mean(axis=0),
            'gamma_lo': np.percentile(gamma_projections, 5, axis=0),
            'gamma_hi': np.percentile(gamma_projections, 95, axis=0),
            'gamma_std': gamma_projections.std(axis=0),
        })

        return results

    def predict_hazard_oos(
        self,
        ages: np.ndarray,
        vintage: str,
        months_ahead: np.ndarray,
        n_mc_paths: int = 500,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Out-of-sample hazard combining in-sample age/vintage effects
        with AR-projected calendar-time effect.

        h(t, v, c*) = intercept + alpha(t) + beta(v) + gamma(c*)

        Parameters
        ----------
        ages : np.ndarray
            Loan ages at which to evaluate.
        vintage : str
            Vintage quarter identifier.
        months_ahead : np.ndarray
            Months into the future for each age.
        n_mc_paths : int
        seed : int

        Returns
        -------
        dict with 'hazard_mean', 'hazard_lo', 'hazard_hi' arrays.
        """
        # Age effects: lookup or interpolate
        age_vals = np.interp(
            ages.astype(float),
            self.age_levels_.astype(float),
            self.age_effects_.values,
        )

        # Vintage effect: single value
        if vintage in self.vintage_effects_.index:
            vin_val = self.vintage_effects_[vintage]
        else:
            vin_val = 0.0

        # Calendar-time effect: project forward
        gamma_proj = self.project_calendar_effect(
            n_months_ahead=int(months_ahead.max()) + 1,
            n_mc_paths=n_mc_paths,
            seed=seed,
        )

        gamma_mean = np.interp(months_ahead, gamma_proj['month_ahead'].values,
                               gamma_proj['gamma_mean'].values)
        gamma_lo = np.interp(months_ahead, gamma_proj['month_ahead'].values,
                             gamma_proj['gamma_lo'].values)
        gamma_hi = np.interp(months_ahead, gamma_proj['month_ahead'].values,
                             gamma_proj['gamma_hi'].values)

        base = self.intercept_ + age_vals + vin_val

        return {
            'hazard_mean': base + gamma_mean,
            'hazard_lo': base + gamma_lo,
            'hazard_hi': base + gamma_hi,
        }

    def plot_effects(
        self,
        title_prefix: str = '',
        axes=None,
        figsize: Tuple[int, int] = (16, 4),
    ):
        """
        Plot age, vintage, and calendar-time effects.

        Parameters
        ----------
        title_prefix : str
            Prepend to subplot titles (e.g., 'Prepay' or 'Default').
        axes : array of 3 matplotlib Axes, optional
        figsize : tuple

        Returns
        -------
        axes
        """
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=figsize)

        pfx = f'{title_prefix} ' if title_prefix else ''

        # Age effect
        axes[0].plot(self.age_effects_.index, self.age_effects_.values,
                     'b-', linewidth=2)
        axes[0].set_xlabel('Loan Age (months)')
        axes[0].set_ylabel('α(t)')
        axes[0].set_title(f'{pfx}Age Effect')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Vintage effect
        idx = self.vintage_effects_.index
        axes[1].bar(range(len(idx)), self.vintage_effects_.values,
                    alpha=0.7, color='steelblue')
        n_labels = min(len(idx), 20)
        step = max(1, len(idx) // n_labels)
        axes[1].set_xticks(range(0, len(idx), step))
        axes[1].set_xticklabels([str(idx[i]) for i in range(0, len(idx), step)],
                                rotation=45, ha='right')
        axes[1].set_ylabel('β(v)')
        axes[1].set_title(f'{pfx}Vintage Effect')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Calendar-time effect
        axes[2].plot(range(len(self.calendar_effects_)),
                     self.calendar_effects_.values,
                     'b-', linewidth=1.5)
        idx_c = self.calendar_effects_.index
        n_labels_c = min(len(idx_c), 15)
        step_c = max(1, len(idx_c) // n_labels_c)
        axes[2].set_xticks(range(0, len(idx_c), step_c))
        axes[2].set_xticklabels([str(idx_c[i]) for i in range(0, len(idx_c), step_c)],
                                rotation=45, ha='right')
        axes[2].set_ylabel('γ(c)')
        axes[2].set_title(f'{pfx}Calendar-Time Effect')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)

        return axes
