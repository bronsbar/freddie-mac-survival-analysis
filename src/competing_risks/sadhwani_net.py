"""
Deep Learning for Mortgage Risk — Sadhwani, Sirignano & Giesecke (2021).

Neural network for monthly mortgage state transition probabilities.
3-state model: Current (0), Prepay (1), Default (2).

The network predicts P[state_t | X_{t-1}] via softmax, trained by
maximum likelihood (cross-entropy) on loan-month transitions.

Reference:
    Giesecke, Sirignano & Sadhwani (2021). "Deep Learning for Mortgage Risk."
    Journal of Financial Econometrics, 19(2), 313-368.
"""

import math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Device detection
# =============================================================================

def get_device(prefer: str = 'auto') -> torch.device:
    """Select best available device: CUDA > MPS > CPU.

    Parameters
    ----------
    prefer : str
        'auto' (best available), 'cuda', 'mps', or 'cpu'.
    """
    if prefer == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(prefer)


# =============================================================================
# Network
# =============================================================================

class SadhwaniNet(nn.Module):
    """
    Feedforward neural network for mortgage state transitions.

    Architecture follows the paper's cross-validated optimum:
    5 hidden layers (200-140-140-140-140), ReLU, dropout.
    Softmax output over 3 states.

    Parameters
    ----------
    n_features : int
        Number of input covariates.
    hidden_sizes : list[int]
        Neurons per hidden layer.
    n_states : int
        Number of output states (3: current, prepay, default).
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_features: int = 13,
        hidden_sizes: List[int] = None,
        n_states: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [200, 140, 140, 140, 140]

        self.n_features = n_features
        self.n_states = n_states

        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, n_states)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities (for NLLLoss)."""
        h = self.hidden(x)
        return torch.log_softmax(self.output(h), dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (softmax)."""
        h = self.hidden(x)
        return torch.softmax(self.output(h), dim=-1)


# =============================================================================
# Learning-rate schedule  (Eq. 9 of the paper)
# =============================================================================

def _lr_lambda(epoch: int, halflife: int = 800) -> float:
    """lr_t = lr_0 / (1 + t / halflife)."""
    return 1.0 / (1.0 + epoch / halflife)


# =============================================================================
# Training loop
# =============================================================================

def train_single_model(
    model: SadhwaniNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    n_epochs: int = 100,
    lr_halflife: int = 800,
    patience: int = 10,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train one SadhwaniNet via mini-batch SGD with LR decay.

    Parameters
    ----------
    model : SadhwaniNet
    train_loader, val_loader : DataLoader
    lr : float – initial learning rate
    weight_decay : float – L2 penalty
    n_epochs : int
    lr_halflife : int – epochs until LR halves (Eq. 9)
    patience : int – early stopping patience on val loss
    device : torch.device
    verbose : bool

    Returns
    -------
    dict with keys 'train_loss', 'val_loss' (per epoch).
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                          momentum=0.9)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _lr_lambda(e, lr_halflife))

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        # --- train ---
        model.train()
        running_loss = 0.0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            log_probs = model(X_batch)
            loss = criterion(log_probs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)
        train_loss = running_loss / n_samples
        history['train_loss'].append(train_loss)
        scheduler.step()

        # --- validate ---
        val_loss = _evaluate_loss(model, val_loader, criterion, device) if val_loader else train_loss
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={current_lr:.6f}")

        # early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return history


@torch.no_grad()
def _evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        log_probs = model(X_batch)
        total_loss += criterion(log_probs, y_batch).item() * X_batch.size(0)
        n += X_batch.size(0)
    return total_loss / n if n > 0 else float('inf')


# =============================================================================
# Ensemble
# =============================================================================

class SadhwaniEnsemble:
    """
    Ensemble of independently trained SadhwaniNet models.

    Each member is trained on bootstrapped data with a different random seed.
    Predictions are averaged across all members.

    Parameters
    ----------
    n_models : int – ensemble size (paper uses 8)
    n_features : int
    hidden_sizes : list[int]
    n_states : int
    dropout : float
    """

    def __init__(
        self,
        n_models: int = 8,
        n_features: int = 13,
        hidden_sizes: List[int] = None,
        n_states: int = 3,
        dropout: float = 0.5,
    ):
        self.n_models = n_models
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.n_states = n_states
        self.dropout = dropout
        self.models: List[SadhwaniNet] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        *,
        batch_size: int = 4096,
        n_epochs: int = 100,
        lr: float = 0.1,
        weight_decay: float = 1e-4,
        lr_halflife: int = 800,
        patience: int = 10,
        device: torch.device = None,
        base_seed: int = 42,
        verbose: bool = True,
    ) -> List[Dict[str, list]]:
        """Train all ensemble members. Returns list of training histories."""
        if device is None:
            device = get_device()

        val_loader = None
        if X_val is not None:
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.models = []
        histories = []
        n_train = len(X_train)

        for i in range(self.n_models):
            seed = base_seed + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            if verbose:
                print(f"\n--- Ensemble member {i+1}/{self.n_models} (seed={seed}) ---")

            # Bootstrap sample
            idx = np.random.choice(n_train, size=n_train, replace=True)
            X_boot = X_train[idx]
            y_boot = y_train[idx]

            train_ds = TensorDataset(
                torch.tensor(X_boot, dtype=torch.float32),
                torch.tensor(y_boot, dtype=torch.long),
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      drop_last=False)

            net = SadhwaniNet(
                n_features=self.n_features,
                hidden_sizes=self.hidden_sizes,
                n_states=self.n_states,
                dropout=self.dropout,
            )

            hist = train_single_model(
                net, train_loader, val_loader,
                lr=lr, weight_decay=weight_decay,
                n_epochs=n_epochs, lr_halflife=lr_halflife,
                patience=patience, device=device, verbose=verbose,
            )
            self.models.append(net)
            histories.append(hist)

        return histories

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, device: torch.device = None) -> np.ndarray:
        """Average predicted probabilities across ensemble members.

        Returns
        -------
        np.ndarray of shape (n_samples, n_states)
        """
        if device is None:
            device = get_device()

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        probs = np.zeros((len(X), self.n_states), dtype=np.float64)
        for net in self.models:
            net = net.to(device)
            net.eval()
            p = net.predict_proba(X_t).cpu().numpy()
            probs += p
        probs /= len(self.models)
        return probs

    def save(self, path: str):
        """Save ensemble state dicts + config."""
        state = {
            'config': {
                'n_models': self.n_models,
                'n_features': self.n_features,
                'hidden_sizes': self.hidden_sizes,
                'n_states': self.n_states,
                'dropout': self.dropout,
            },
            'models': [m.state_dict() for m in self.models],
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'SadhwaniEnsemble':
        """Load ensemble from file."""
        if device is None:
            device = get_device()
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state['config']
        ens = cls(**cfg)
        for sd in state['models']:
            net = SadhwaniNet(
                n_features=cfg['n_features'],
                hidden_sizes=cfg['hidden_sizes'],
                n_states=cfg['n_states'],
                dropout=cfg['dropout'],
            )
            net.load_state_dict(sd)
            net.to(device)
            ens.models.append(net)
        return ens


# =============================================================================
# CIF computation — chain monthly transition probabilities
# =============================================================================

def _init_cif_result(loan_ids, eval_times, panel_df):
    """Common result dict initialisation."""
    n_loans = len(loan_ids)
    result = {'loan_sequence_number': loan_ids}
    for t in eval_times:
        result[f'cif_prepay_{t}'] = np.zeros(n_loans)
        result[f'cif_default_{t}'] = np.zeros(n_loans)
    terminal = panel_df.groupby('loan_sequence_number').last()
    result['duration'] = terminal.loc[loan_ids, 'loan_age'].values
    result['event_code'] = terminal.loc[loan_ids, 'event_code'].values
    return result


def _chain_cif(probs_sequence, eval_times):
    """Chain monthly probability arrays into CIF.

    Parameters
    ----------
    probs_sequence : list[np.ndarray]
        probs_sequence[t] has shape (n_loans, 3) for month t.
        Entries may be None for loans not present at that month.
    eval_times : list[int]

    Returns
    -------
    dict mapping 'cif_prepay_{t}' / 'cif_default_{t}' to arrays.
    """
    n_loans = probs_sequence[0].shape[0]
    surv = np.ones(n_loans)
    cif_prepay = np.zeros(n_loans)
    cif_default = np.zeros(n_loans)
    snapshots = {}

    for month_idx, probs in enumerate(probs_sequence):
        cif_prepay += surv * probs[:, 1]
        cif_default += surv * probs[:, 2]
        surv *= probs[:, 0]

        if (month_idx + 1) in eval_times:
            snapshots[f'cif_prepay_{month_idx+1}'] = cif_prepay.copy()
            snapshots[f'cif_default_{month_idx+1}'] = cif_default.copy()

    return snapshots


# ── Method 1: Frozen features ──────────────────────────────────────────────

@torch.no_grad()
def compute_cif_frozen(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    model,
    scaler: StandardScaler,
    eval_times: List[int],
    device: torch.device = None,
    batch_size: int = 65536,
) -> Dict[str, np.ndarray]:
    """
    CIF by chaining with **frozen** (time-zero) features.

    Uses each loan's first observation as features for every future month.
    Behavioural features (bal_repaid_lag1, t_act_12m) are updated
    mechanically; macro features are held constant at their time-zero value.

    This avoids using future information and is comparable to how the
    Blumenstock DeepHit computes CIF from a single feature vector.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Loan-month panel (needs loan_sequence_number, loan_age, event_code).
    feature_cols : list[str]
    model : SadhwaniNet or SadhwaniEnsemble
    scaler : fitted StandardScaler
    eval_times : list[int]
    device, batch_size : as usual

    Returns
    -------
    dict  (same schema as compute_cif_ar)
    """
    if device is None:
        device = get_device()

    max_t = max(eval_times)
    panel_sorted = panel_df.sort_values(['loan_sequence_number', 'loan_age'])

    # Take first observation per loan as baseline features
    first_obs = panel_sorted.groupby('loan_sequence_number').first()
    loan_ids = first_obs.index.values
    n_loans = len(loan_ids)

    result = _init_cif_result(loan_ids, eval_times, panel_sorted)

    X_base = first_obs[feature_cols].values.astype(np.float32)

    # Locate mechanical columns
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    has_bal = 'bal_repaid_lag1' in col_idx
    has_act = 't_act_12m' in col_idx

    # Pre-compute starting loan_age
    start_age = first_obs['loan_age'].values.astype(np.float32)

    # Chain forward with frozen features, only updating mechanical cols
    probs_seq = []
    for month in range(max_t):
        X_month = X_base.copy()

        # Mechanical update: t_act_12m = min(12, age)
        if has_act:
            age = start_age + month
            X_month[:, col_idx['t_act_12m']] = np.minimum(12.0, age)

        # Mechanical update: bal_repaid grows linearly (crude approximation)
        # Better than stale value; exact amortisation would need orig_upb + rate
        if has_bal and month > 0:
            # Simple linear extrapolation: delta ~ first-month value / start_age
            base_val = X_base[:, col_idx['bal_repaid_lag1']]
            safe_age = np.maximum(start_age, 1.0)
            X_month[:, col_idx['bal_repaid_lag1']] = base_val + (base_val / safe_age) * month

        X_scaled = scaler.transform(X_month).astype(np.float32)
        probs = _predict_batched(model, X_scaled, device, batch_size)
        probs_seq.append(probs)

    snapshots = _chain_cif(probs_seq, eval_times)
    result.update(snapshots)
    return result


# ── Method 2: AR-simulated macro paths ─────────────────────────────────────

def fit_ar_models(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    max_lag: int = 4,
) -> Dict[str, object]:
    """
    Fit univariate AR(p) models to each time-varying feature.

    Macro features are averaged per calendar month (year_month) to recover
    the actual economic time series.  Behavioural features are averaged per
    loan_age since they describe the loan lifecycle, not the economy.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Loan-month panel (must contain 'year_month' for macro features).
    feature_cols : list[str]
    max_lag : int
        Maximum AR lag order. BIC selects the best p <= max_lag.

    Returns
    -------
    dict mapping feature_name -> fitted AR result (statsmodels).
    Only includes features that are genuinely time-varying.
    """
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.ar_model import ar_select_order

    STATIC = {'int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r'}
    MACRO = set(MACRO_FEATURES)

    # Macro features: average per calendar month (actual economic time series)
    if 'year_month' in panel_df.columns:
        macro_avg = panel_df.groupby('year_month')[feature_cols].mean().sort_index()
    else:
        macro_avg = panel_df.groupby('loan_age')[feature_cols].mean()

    # Behavioural features: average per loan age (lifecycle profile)
    behav_avg = panel_df.groupby('loan_age')[feature_cols].mean()

    ar_models = {}
    for col in feature_cols:
        if col in STATIC:
            continue

        if col in MACRO:
            series = np.asarray(macro_avg[col].dropna(), dtype=np.float64)
        else:
            series = np.asarray(behav_avg[col].dropna(), dtype=np.float64)

        if len(series) < max_lag + 2:
            continue
        if np.std(series) < 1e-10:
            continue

        # Select lag order by BIC
        try:
            sel = ar_select_order(series, maxlag=max_lag, ic='bic', old_names=False)
            best_lag = max(sel.ar_lags) if sel.ar_lags else 1
        except Exception:
            best_lag = 1

        ar_fit = AutoReg(series, lags=best_lag, old_names=False).fit()
        ar_models[col] = ar_fit

    return ar_models


def simulate_ar_paths(
    ar_models: Dict[str, object],
    n_steps: int,
    n_simulations: int = 50,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate forward paths from fitted AR models (single shared path).

    Parameters
    ----------
    ar_models : dict
        feature_name -> fitted statsmodels AutoReg result.
    n_steps : int
        Number of months to simulate forward.
    n_simulations : int
        Number of Monte Carlo paths per feature.
    seed : int

    Returns
    -------
    dict mapping feature_name -> np.ndarray of shape (n_simulations, n_steps).
    """
    rng = np.random.RandomState(seed)
    paths = {}

    for col, ar_fit in ar_models.items():
        sigma = np.sqrt(ar_fit.sigma2)
        params = ar_fit.params          # [const, phi_1, ..., phi_p]
        p = len(params) - 1             # AR order
        last_vals = ar_fit.model.endog[-p:]  # last p observed values

        sims = np.zeros((n_simulations, n_steps))
        for s in range(n_simulations):
            history = list(last_vals.copy())
            for t in range(n_steps):
                pred = params[0]  # intercept
                for lag in range(1, p + 1):
                    pred += params[lag] * history[-lag]
                pred += rng.normal(0, sigma)
                history.append(pred)
                sims[s, t] = pred
            # Trim history
        paths[col] = sims

    return paths


def simulate_ar_paths_per_loan(
    ar_models: Dict[str, object],
    start_values: Dict[str, np.ndarray],
    n_steps: int,
    n_simulations: int = 50,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate per-loan forward paths from fitted AR models.

    Each loan starts from its own initial macro values rather than a single
    shared starting point.  Noise is shared across loans within each
    simulation so that macro shocks are correlated (all loans experience
    the same macro surprise, but from different starting levels).

    Parameters
    ----------
    ar_models : dict
        feature_name -> fitted statsmodels AutoReg result.
    start_values : dict
        feature_name -> np.ndarray of shape (n_loans, p) with each loan's
        last p observed values for that feature.
    n_steps : int
        Number of months to simulate forward.
    n_simulations : int
        Number of Monte Carlo paths per feature.
    seed : int

    Returns
    -------
    dict mapping feature_name -> np.ndarray of shape
        (n_simulations, n_loans, n_steps).
    """
    rng = np.random.RandomState(seed)
    paths = {}

    for col, ar_fit in ar_models.items():
        if col not in start_values:
            continue
        sigma = np.sqrt(ar_fit.sigma2)
        params = ar_fit.params          # [const, phi_1, ..., phi_p]
        p = len(params) - 1             # AR order
        init = start_values[col]        # (n_loans, p)
        n_loans = init.shape[0]

        sims = np.zeros((n_simulations, n_loans, n_steps))
        for s in range(n_simulations):
            # History: (n_loans, p + n_steps) — columns are time steps
            history = np.zeros((n_loans, p + n_steps))
            history[:, :p] = init

            # Shared noise across loans per simulation
            noise = rng.normal(0, sigma, size=n_steps)

            for t in range(n_steps):
                pred = np.full(n_loans, params[0])
                for lag in range(1, p + 1):
                    pred += params[lag] * history[:, p + t - lag]
                pred += noise[t]
                history[:, p + t] = pred
                sims[s, :, t] = pred

        paths[col] = sims

    return paths


@torch.no_grad()
def compute_cif_ar(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    model,
    scaler: StandardScaler,
    eval_times: List[int],
    ar_models: Dict[str, object],
    n_simulations: int = 50,
    device: torch.device = None,
    batch_size: int = 65536,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    CIF by chaining with **per-loan AR-simulated** macro paths.

    Each loan's AR simulation starts from its own initial macro values,
    preserving cross-loan variation in macro conditions.  Noise is shared
    across loans within each simulation so that macro shocks are correlated.

    Static and behavioural features are handled as in compute_cif_frozen.

    This follows Sadhwani et al. (2021) Section 5.4, where future macro
    paths are generated from AR models fitted to historical data.

    Parameters
    ----------
    panel_df : pd.DataFrame
    feature_cols : list[str]
    model : SadhwaniNet or SadhwaniEnsemble
    scaler : fitted StandardScaler
    eval_times : list[int]
    ar_models : dict – fitted AR models from fit_ar_models()
    n_simulations : int – Monte Carlo paths to average
    device, batch_size, seed : as usual

    Returns
    -------
    dict  (same schema as compute_cif_frozen)
    """
    if device is None:
        device = get_device()

    max_t = max(eval_times)
    panel_sorted = panel_df.sort_values(['loan_sequence_number', 'loan_age'])

    # Baseline: first observation per loan
    first_obs = panel_sorted.groupby('loan_sequence_number').first()
    loan_ids = first_obs.index.values
    n_loans = len(loan_ids)

    result = _init_cif_result(loan_ids, eval_times, panel_sorted)

    X_base = first_obs[feature_cols].values.astype(np.float32)
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    has_bal = 'bal_repaid_lag1' in col_idx
    has_act = 't_act_12m' in col_idx
    start_age = first_obs['loan_age'].values.astype(np.float32)

    # Build per-loan starting values for each AR macro feature.
    # Each loan starts from its own initial macro value.  For AR(p) with
    # p > 1 we need p prior values; we approximate by repeating the
    # loan's first observation (the AR dynamics quickly dominate).
    start_values = {}
    for col in ar_models:
        if col not in col_idx:
            continue
        p = len(ar_models[col].params) - 1
        loan_vals = X_base[:, col_idx[col]]  # (n_loans,)
        # Tile to (n_loans, p) — each lag slot initialised to the same value
        start_values[col] = np.tile(loan_vals[:, None], (1, p))

    # Simulate per-loan AR paths
    ar_paths = simulate_ar_paths_per_loan(
        ar_models, start_values, n_steps=max_t,
        n_simulations=n_simulations, seed=seed,
    )

    # Accumulate CIF across simulations
    cif_accum = {}
    for t in eval_times:
        cif_accum[f'cif_prepay_{t}'] = np.zeros(n_loans)
        cif_accum[f'cif_default_{t}'] = np.zeros(n_loans)

    for sim in range(n_simulations):
        probs_seq = []
        for month in range(max_t):
            X_month = X_base.copy()

            # Mechanical updates
            if has_act:
                X_month[:, col_idx['t_act_12m']] = np.minimum(12.0, start_age + month)
            if has_bal and month > 0:
                base_val = X_base[:, col_idx['bal_repaid_lag1']]
                safe_age = np.maximum(start_age, 1.0)
                X_month[:, col_idx['bal_repaid_lag1']] = base_val + (base_val / safe_age) * month

            # Replace macro features with per-loan AR-simulated values
            for col, sim_paths in ar_paths.items():
                if col in col_idx:
                    X_month[:, col_idx[col]] = sim_paths[sim, :, month]

            X_scaled = scaler.transform(X_month).astype(np.float32)
            probs = _predict_batched(model, X_scaled, device, batch_size)
            probs_seq.append(probs)

        snapshots = _chain_cif(probs_seq, eval_times)
        for key in cif_accum:
            cif_accum[key] += snapshots.get(key, np.zeros(n_loans))

    # Average across simulations
    for key in cif_accum:
        result[key] = cif_accum[key] / n_simulations

    return result


# ── Legacy wrapper (uses observed panel features — for training diagnostics) ─

@torch.no_grad()
def compute_cif(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    model,
    scaler: StandardScaler,
    eval_times: List[int],
    device: torch.device = None,
    batch_size: int = 65536,
) -> Dict[str, np.ndarray]:
    """
    CIF using **observed** panel features (uses future information).

    Kept for training diagnostics only — do NOT use for out-of-sample
    evaluation.  Use compute_cif_frozen or compute_cif_ar instead.
    """
    if device is None:
        device = get_device()

    max_t = max(eval_times)
    panel_sorted = panel_df.sort_values(['loan_sequence_number', 'loan_age'])
    loan_ids = panel_sorted['loan_sequence_number'].unique()
    n_loans = len(loan_ids)

    result = _init_cif_result(loan_ids, eval_times, panel_sorted)
    loan_id_to_pos = {lid: i for i, lid in enumerate(loan_ids)}

    surv = np.ones(n_loans)
    cif_prepay = np.zeros(n_loans)
    cif_default = np.zeros(n_loans)

    for age in range(0, max_t):
        age_rows = panel_sorted[panel_sorted['loan_age'] == age]
        if len(age_rows) == 0:
            continue

        positions = np.array([loan_id_to_pos[lid] for lid in age_rows['loan_sequence_number'].values])
        X_scaled = scaler.transform(age_rows[feature_cols].values.astype(np.float32)).astype(np.float32)
        probs = _predict_batched(model, X_scaled, device, batch_size)

        cif_prepay[positions] += surv[positions] * probs[:, 1]
        cif_default[positions] += surv[positions] * probs[:, 2]
        surv[positions] *= probs[:, 0]

        if (age + 1) in eval_times:
            result[f'cif_prepay_{age+1}'] = cif_prepay.copy()
            result[f'cif_default_{age+1}'] = cif_default.copy()

    return result


@torch.no_grad()
def _predict_batched(model, X: np.ndarray, device, batch_size: int) -> np.ndarray:
    """Predict probabilities in batches, returns numpy."""
    n = len(X)
    probs = np.empty((n, 3), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_t = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        if hasattr(model, 'models'):
            # Ensemble
            p = model.predict_proba(X[start:end], device=device)
        else:
            model.eval()
            p = model.predict_proba(X_t).cpu().numpy()
        probs[start:end] = p
    return probs


# =============================================================================
# Sensitivity analysis (Eq. 7 of the paper)
# =============================================================================

@torch.no_grad()
def variable_sensitivity(
    model,
    X: np.ndarray,
    feature_names: List[str],
    from_state: int = 0,
    to_state: int = 1,
    delta: float = 1e-3,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Compute average absolute gradient of transition probability w.r.t. each feature.

    Finite-difference approximation of Eq. (7):
        Sensitivity(j) = E[|∂h(v,X)/∂x_j|]

    Parameters
    ----------
    model : SadhwaniNet or SadhwaniEnsemble
    X : np.ndarray (n_samples, n_features) – standardised
    feature_names : list[str]
    from_state : int – not used (always current in 3-state)
    to_state : int – target state column index (1=prepay, 2=default)
    delta : float – finite difference step size
    device : torch.device

    Returns
    -------
    pd.DataFrame with columns ['feature', 'sensitivity'] sorted descending.
    """
    if device is None:
        device = get_device()

    X_base = torch.tensor(X, dtype=torch.float32, device=device)

    if hasattr(model, 'models'):
        # ensemble – average
        base_probs = model.predict_proba(X, device=device)[:, to_state]
    else:
        model.eval()
        model = model.to(device)
        base_probs = model.predict_proba(X_base).cpu().numpy()[:, to_state]

    sensitivities = []
    for j in range(X.shape[1]):
        X_plus = X.copy()
        X_plus[:, j] += delta

        if hasattr(model, 'models'):
            probs_plus = model.predict_proba(X_plus, device=device)[:, to_state]
        else:
            X_pt = torch.tensor(X_plus, dtype=torch.float32, device=device)
            probs_plus = model.predict_proba(X_pt).cpu().numpy()[:, to_state]

        grad = np.abs(probs_plus - base_probs) / delta
        sensitivities.append(grad.mean())

    df = pd.DataFrame({
        'feature': feature_names,
        'sensitivity': sensitivities,
    }).sort_values('sensitivity', ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# Data preparation helpers
# =============================================================================

# Standard feature sets (same as other notebooks)
STATIC_FEATURES = ['int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r']
BEHAVIORAL_FEATURES = ['bal_repaid_lag1', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m']
MACRO_FEATURES = ['hpi_st_d_t_o', 'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o']
ALL_FEATURES = STATIC_FEATURES + BEHAVIORAL_FEATURES + MACRO_FEATURES

# Folds (Blumenstock methodology)
TRAIN_FOLDS = list(range(9))
VAL_FOLDS = [9]
TEST_FOLD = 10


def prepare_monthly_targets(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'target' column: the next-month state for each loan-month observation.

    For each loan's history:
    - All months before the terminal event → target = 0 (stayed current)
    - The terminal month → target = event_code (1=prepay, 2=default)
    - If event_code == 0 (censored), the last month also gets target = 0

    Returns a copy of the panel with the 'target' column added.
    """
    df = panel_df.sort_values(['loan_sequence_number', 'loan_age']).copy()

    # Default: stayed current
    df['target'] = 0

    # For the last observation per loan, set target = event_code
    last_idx = df.groupby('loan_sequence_number').tail(1).index
    df.loc[last_idx, 'target'] = df.loc[last_idx, 'event_code']

    return df


def prepare_features(panel_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Ensure required feature columns exist (log_upb, bal_repaid_lag1).

    Returns (feature_cols, modified_df).
    """
    df = panel_df.copy()

    # log_upb from orig_upb if needed
    if 'log_upb' not in df.columns and 'orig_upb' in df.columns:
        df['log_upb'] = np.log(df['orig_upb'].clip(lower=1))

    # bal_repaid_lag1 from bal_repaid if needed
    if 'bal_repaid_lag1' not in df.columns and 'bal_repaid' in df.columns:
        df['bal_repaid_lag1'] = df.groupby('loan_sequence_number')['bal_repaid'].shift(1)
        df['bal_repaid_lag1'] = df['bal_repaid_lag1'].fillna(0)

    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    return feature_cols, df
