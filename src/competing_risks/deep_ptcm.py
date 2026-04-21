"""
Deep Promotion Time Cure Model (Deep-PTCM) for Competing Risks.

Extends Medina-Olivares et al. (2024) "The Deep Promotion Time Cure Model"
to competing risks (prepayment and default) for mortgage survival analysis.

The PTCM assumes each subject has K_i ~ Poisson(theta(x)) unobserved latent
risk factors.  Subjects with K_i = 0 are "cured" -- they never experience the
event.  The cure fraction is pi(x) = exp(-theta(x)).

For competing risks we fit independent latent causes:
    S(t; x) = prod_k exp(-theta_k(x) * F_k(t))
where F_k is a piecewise-exponential baseline CDF for cause k.

The Deep-PTCM replaces the linear predictor log(theta) = w'x + b with a DNN,
optionally decomposed via orthogonalization into interpretable linear effects
plus a nonlinear residual.

Reference
---------
Medina-Olivares, V., Lessmann, S. & Klein, N. (2024).
"The Deep Promotion Time Cure Model."
IEEE Trans. Neural Netw. Learn. Syst., 35(12), 18848-18858.
"""

import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Feature constants (matching the project convention)
# ---------------------------------------------------------------------------
STATIC_FEATURES = ['int_rate', 'log_upb', 'fico_score', 'dti_r', 'ltv_r']

TRAIN_FOLDS = list(range(9))
VAL_FOLDS = [9]
TEST_FOLD = 10


def get_device() -> torch.device:
    """Select best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ===================================================================
# 1. Piecewise Exponential Baseline  F_k(t)
# ===================================================================

class PiecewiseExponentialBaseline(nn.Module):
    """Piecewise exponential CDF for the latent activation time.

    The study period [0, T_max] is split into J intervals with breakpoints
    tau_0 = 0 < tau_1 < ... < tau_J = T_max.  Within interval j the hazard
    of the latent activation time is constant lambda_j > 0.

        F(t) = 1 - exp(-Lambda(t))
        f(t) = lambda_{j(t)} * exp(-Lambda(t))

    where Lambda(t) = sum_{j} lambda_j * d_j(t) is the cumulative hazard and
    d_j(t) is the exposure in interval j up to time t.

    Parameters
    ----------
    breakpoints : array-like of shape (J+1,)
        Interval boundaries [tau_0, tau_1, ..., tau_J].
    """

    def __init__(self, breakpoints: np.ndarray):
        super().__init__()
        breakpoints = np.asarray(breakpoints, dtype=np.float32)
        assert breakpoints[0] == 0.0, "First breakpoint must be 0"
        self.register_buffer('breakpoints', torch.from_numpy(breakpoints))
        J = len(breakpoints) - 1
        # Unconstrained parameters; lambda_j = softplus(raw_j)
        self.raw_lambdas = nn.Parameter(torch.zeros(J))

    @property
    def lambdas(self) -> torch.Tensor:
        return F_torch.softplus(self.raw_lambdas)

    def forward(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute F(t) and f(t).

        Parameters
        ----------
        t : Tensor of shape (*)
            Observed times (non-negative).

        Returns
        -------
        F_t : Tensor, same shape as t
            CDF values.
        f_t : Tensor, same shape as t
            Density values.
        """
        lam = self.lambdas                          # (J,)
        bp = self.breakpoints                       # (J+1,)
        widths = bp[1:] - bp[:-1]                   # (J,)

        t_flat = t.reshape(-1, 1)                   # (N, 1)
        # Exposure per interval: min(t - tau_{j-1}, width_j), clipped at 0
        lower = bp[:-1].unsqueeze(0)                # (1, J)
        exposure = (t_flat - lower).clamp(min=0.0)
        exposure = torch.min(exposure, widths.unsqueeze(0))  # (N, J)

        cum_hazard = (exposure * lam.unsqueeze(0)).sum(dim=1)  # (N,)

        # Which interval does t fall in?
        # j(t) = last j s.t. tau_j <= t  (clamped to valid range)
        j_idx = torch.searchsorted(bp, t_flat.squeeze(1), right=True) - 1
        j_idx = j_idx.clamp(0, len(lam) - 1)
        lam_at_t = lam[j_idx]                       # (N,)

        S_0 = torch.exp(-cum_hazard)                # 1 - F
        F_t = 1.0 - S_0
        f_t = lam_at_t * S_0

        return F_t.reshape(t.shape), f_t.reshape(t.shape)


# ===================================================================
# 2. DNN for theta(x)
# ===================================================================

class DeepPTCMNetwork(nn.Module):
    """Neural network producing theta_k(x) > 0 for each competing cause.

    Architecture
    ------------
    - Shared hidden layers (BatchNorm + ReLU + Dropout)
    - Per-cause heads each outputting a scalar g_k(x)
    - theta_k(x) = exp(g_k(x))

    With ``orthogonalize=True`` the predictor is decomposed as:
        g_k(x) = w_k' x + b_k  +  [DNN_k(x) - proj_{col(X)} DNN_k(x)]
    so the linear part is interpretable and the DNN captures nonlinear residuals.

    Parameters
    ----------
    in_features : int
    num_causes : int
    shared_layers : list[int]
    head_layers : list[int]
    dropout : float
    batch_norm : bool
    orthogonalize : bool
    """

    def __init__(
        self,
        in_features: int,
        num_causes: int = 2,
        shared_layers: Optional[List[int]] = None,
        head_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        batch_norm: bool = False,
        orthogonalize: bool = False,
    ):
        super().__init__()
        if shared_layers is None:
            shared_layers = [512, 512]
        if head_layers is None:
            head_layers = []

        self.in_features = in_features
        self.num_causes = num_causes
        self.orthogonalize = orthogonalize

        # --- shared trunk ---
        layers = []
        prev = in_features
        for n in shared_layers:
            layers.append(nn.Linear(prev, n))
            if batch_norm:
                layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = n
        self.shared = nn.Sequential(*layers)

        # --- cause-specific heads ---
        self.heads = nn.ModuleList()
        for _ in range(num_causes):
            head = []
            p = prev
            for n in head_layers:
                head.append(nn.Linear(p, n))
                if batch_norm:
                    head.append(nn.BatchNorm1d(n))
                head.append(nn.ReLU())
                if dropout > 0:
                    head.append(nn.Dropout(dropout))
                p = n
            head.append(nn.Linear(p, 1))
            self.heads.append(nn.Sequential(*head))

        # --- linear predictors for orthogonalization ---
        if orthogonalize:
            self.linears = nn.ModuleList([
                nn.Linear(in_features, 1) for _ in range(num_causes)
            ])
            # Will hold (X'X + eps I)^{-1}  computed once from training data
            self.register_buffer(
                'XtX_inv', torch.eye(in_features, dtype=torch.float32)
            )

    def set_orthogonalization_matrix(self, X_train: torch.Tensor):
        """Precompute (X'X + eps I)^{-1} from the full training feature matrix.

        Call this once before training when ``orthogonalize=True``.
        """
        XtX = X_train.T @ X_train / X_train.shape[0]
        eps = 1e-4 * torch.eye(XtX.shape[0], device=XtX.device)
        self.XtX_inv = torch.linalg.inv(XtX + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return theta of shape (batch, num_causes), all positive."""
        shared_out = self.shared(x)                 # (B, H)

        g_list = []
        for k in range(self.num_causes):
            eta_nn = self.heads[k](shared_out).squeeze(-1)  # (B,)

            if self.orthogonalize:
                eta_lin = self.linears[k](x).squeeze(-1)     # (B,)
                # Project DNN output out of column space of X
                # beta = (X'X)^{-1} (x' eta_nn) / B  ... per-batch approx
                beta = self.XtX_inv @ (x.T @ eta_nn) / x.shape[0]  # (p,)
                eta_ort = eta_nn - x @ beta          # (B,)
                g_k = eta_lin + eta_ort
            else:
                g_k = eta_nn

            g_list.append(g_k)

        g = torch.stack(g_list, dim=1)               # (B, K)
        theta = torch.exp(g.clamp(max=10.0))          # positive, capped to avoid overflow
        return theta


# ===================================================================
# 3. Loss function
# ===================================================================

class PTCMLoss(nn.Module):
    """Negative log-likelihood for competing-risks PTCM.

    For subject i with event delta_ik in {0, 1, ..., K}:

        ell_i = sum_k delta_ik * [log theta_k + log f_k(t_i)]
                - sum_k theta_k * F_k(t_i)

    The second term (overall survival) applies to *all* subjects.
    """

    def forward(
        self,
        theta: torch.Tensor,        # (B, K)
        F_t: torch.Tensor,          # (B, K)
        f_t: torch.Tensor,          # (B, K)
        event_code: torch.Tensor,   # (B,)  0=cens, 1..K=events
        event_types: List[int],     # [1, 2]
    ) -> torch.Tensor:
        eps = 1e-8
        B = theta.shape[0]

        # Survival contribution (all subjects)
        log_surv = -(theta * F_t).sum(dim=1)         # (B,)

        # Event contribution (uncensored only)
        log_event = torch.zeros(B, device=theta.device)
        for k_idx, k_code in enumerate(event_types):
            mask = (event_code == k_code).float()     # (B,)
            log_event = log_event + mask * (
                torch.log(theta[:, k_idx] + eps)
                + torch.log(f_t[:, k_idx] + eps)
            )

        nll = -(log_event + log_surv).mean()
        return nll


# ===================================================================
# 4. Main wrapper
# ===================================================================

class CompetingRisksDeepPTCM(BaseEstimator):
    """Competing Risks Deep Promotion Time Cure Model.

    Parameters
    ----------
    num_intervals : int
        Number of piecewise-exponential intervals for each F_k.
    shared_layers, head_layers : list[int]
        DNN architecture.  Paper defaults: shared=[512, 512], head=[].
    dropout : float
        Paper uses 0.2.
    batch_norm : bool
        Paper does not use batch norm.
    orthogonalize : bool
        Decompose into linear + orthogonal nonlinear component.
    lr : float
        Initial learning rate.  Paper uses 0.01 with SGD.
    weight_decay : float
    batch_size : int
    epochs : int
    patience : int
        Early-stopping patience (epochs without validation improvement).
    lr_decay_rate : float
        Decay rate for inverse time decay schedule.  Paper uses 0.75.
    lr_decay_steps : int
        Decay steps for inverse time decay schedule.  Paper uses 100.
    verbose : bool
    random_state : int
    """

    def __init__(
        self,
        num_intervals: int = 15,
        shared_layers: Optional[List[int]] = None,
        head_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        batch_norm: bool = False,
        orthogonalize: bool = False,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 200,
        patience: int = 15,
        lr_decay_rate: float = 0.75,
        lr_decay_steps: int = 100,
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.num_intervals = num_intervals
        self.shared_layers = shared_layers or [512, 512]
        self.head_layers = head_layers or []
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.orthogonalize = orthogonalize
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.verbose = verbose
        self.random_state = random_state

        # Populated after fit()
        self.network_: Optional[DeepPTCMNetwork] = None
        self.baselines_: Optional[nn.ModuleList] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self.event_types_: List[int] = []
        self.breakpoints_: Optional[np.ndarray] = None
        self.device_: torch.device = get_device()
        self.history_: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        duration: np.ndarray,
        event_code: np.ndarray,
        event_types: List[int] = [1, 2],
        val_data: Optional[Tuple] = None,
        scale_features: bool = True,
    ) -> "CompetingRisksDeepPTCM":
        """Train the Deep-PTCM.

        Parameters
        ----------
        X : (N, p) feature matrix
        duration : (N,) observed times
        event_code : (N,) event codes (0 = censored)
        event_types : list of positive event codes
        val_data : optional (X_val, dur_val, ev_val)
        scale_features : standardise features
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.event_types_ = event_types
        K = len(event_types)

        # --- features ---
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values.astype('float32')
        else:
            self.feature_names_ = [f'x{i}' for i in range(X.shape[1])]
            X = np.asarray(X, dtype='float32')

        duration = np.asarray(duration, dtype='float32')
        event_code = np.asarray(event_code, dtype='int64')

        if scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X).astype('float32')

        # --- breakpoints ---
        t_max = float(duration.max()) + 1.0
        bp = np.linspace(0, t_max, self.num_intervals + 1).astype('float32')
        self.breakpoints_ = bp

        # --- build model components ---
        device = self.device_
        in_features = X.shape[1]

        self.network_ = DeepPTCMNetwork(
            in_features=in_features,
            num_causes=K,
            shared_layers=self.shared_layers,
            head_layers=self.head_layers,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            orthogonalize=self.orthogonalize,
        ).to(device)

        self.baselines_ = nn.ModuleList([
            PiecewiseExponentialBaseline(bp) for _ in range(K)
        ]).to(device)

        if self.orthogonalize:
            X_t = torch.from_numpy(X).to(device)
            self.network_.set_orthogonalization_matrix(X_t)

        # --- data loaders ---
        X_t = torch.from_numpy(X)
        dur_t = torch.from_numpy(duration)
        ev_t = torch.from_numpy(event_code)
        train_ds = TensorDataset(X_t, dur_t, ev_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_data is not None:
            X_v, dur_v, ev_v = val_data
            if isinstance(X_v, pd.DataFrame):
                X_v = X_v.values.astype('float32')
            else:
                X_v = np.asarray(X_v, dtype='float32')
            if self.scaler_ is not None:
                X_v = self.scaler_.transform(X_v).astype('float32')
            dur_v = np.asarray(dur_v, dtype='float32')
            ev_v = np.asarray(ev_v, dtype='int64')
            val_ds = TensorDataset(
                torch.from_numpy(X_v),
                torch.from_numpy(dur_v),
                torch.from_numpy(ev_v),
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size * 2, shuffle=False,
            )

        # --- optimiser (SGD + inverse time decay, following the paper) ---
        params = (
            list(self.network_.parameters())
            + list(self.baselines_.parameters())
        )
        optimizer = torch.optim.SGD(
            params, lr=self.lr, weight_decay=self.weight_decay,
        )
        # Inverse time decay: lr = lr_0 / (1 + decay_rate * step / decay_steps)
        decay_rate = self.lr_decay_rate
        decay_steps = self.lr_decay_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 / (1.0 + decay_rate * step / decay_steps),
        )

        criterion = PTCMLoss()

        # --- training loop ---
        best_val_loss = float('inf')
        best_state = None
        wait = 0
        history: Dict[str, list] = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            # -- train --
            self.network_.train()
            self.baselines_.train()
            epoch_loss = 0.0
            n_samples = 0

            for X_b, dur_b, ev_b in train_loader:
                X_b = X_b.to(device)
                dur_b = dur_b.to(device)
                ev_b = ev_b.to(device)

                theta = self.network_(X_b)           # (B, K)
                F_t_list, f_t_list = [], []
                for k in range(K):
                    Fk, fk = self.baselines_[k](dur_b)
                    F_t_list.append(Fk)
                    f_t_list.append(fk)
                F_t = torch.stack(F_t_list, dim=1)   # (B, K)
                f_t = torch.stack(f_t_list, dim=1)

                loss = criterion(theta, F_t, f_t, ev_b, self.event_types_)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * len(X_b)
                n_samples += len(X_b)

            train_loss = epoch_loss / n_samples
            history['train_loss'].append(train_loss)

            # -- validation --
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, criterion, device, K)
            else:
                val_loss = train_loss
            history['val_loss'].append(val_loss)
            scheduler.step()

            # -- early stopping --
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    'network': copy.deepcopy(self.network_.state_dict()),
                    'baselines': copy.deepcopy(self.baselines_.state_dict()),
                }
                wait = 0
            else:
                wait += 1

            if self.verbose and (epoch % 10 == 0 or wait == 0):
                lr_now = optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch {epoch:3d} | train {train_loss:.4f} | "
                    f"val {val_loss:.4f} | lr {lr_now:.1e} | "
                    f"wait {wait}/{self.patience}"
                )

            if wait >= self.patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best
        if best_state is not None:
            self.network_.load_state_dict(best_state['network'])
            self.baselines_.load_state_dict(best_state['baselines'])

        self.history_ = history
        self.network_.eval()
        self.baselines_.eval()
        return self

    @torch.no_grad()
    def _eval_loss(self, loader, criterion, device, K):
        self.network_.eval()
        self.baselines_.eval()
        total = 0.0
        n = 0
        for X_b, dur_b, ev_b in loader:
            X_b, dur_b, ev_b = X_b.to(device), dur_b.to(device), ev_b.to(device)
            theta = self.network_(X_b)
            F_t_list, f_t_list = [], []
            for k in range(K):
                Fk, fk = self.baselines_[k](dur_b)
                F_t_list.append(Fk)
                f_t_list.append(fk)
            F_t = torch.stack(F_t_list, dim=1)
            f_t = torch.stack(f_t_list, dim=1)
            loss = criterion(theta, F_t, f_t, ev_b, self.event_types_)
            total += loss.item() * len(X_b)
            n += len(X_b)
        return total / n

    # ------------------------------------------------------------------
    # helpers: prepare features
    # ------------------------------------------------------------------
    def _prepare_X(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype('float32')
        else:
            X = np.asarray(X, dtype='float32')
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return torch.from_numpy(X.astype('float32')).to(self.device_)

    # ------------------------------------------------------------------
    # predict_risk
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_risk(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """Risk score for *event* (CIF at *time*, or at T_max)."""
        if event not in self.event_types_:
            raise ValueError(f"Unknown event {event}")
        k_idx = self.event_types_.index(event)

        if time is None:
            time = float(self.breakpoints_[-1])

        cif = self.predict_cumulative_incidence(X, event, np.array([time]))
        return cif[:, 0]

    # ------------------------------------------------------------------
    # predict_cumulative_incidence
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_cumulative_incidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
        times: Optional[np.ndarray] = None,
        n_grid: int = 200,
    ) -> np.ndarray:
        """Compute CIF_k(t) via trapezoidal integration.

        CIF_k(t) = int_0^t  theta_k f_k(s) S(s) ds

        Returns shape (N, len(times)).
        """
        if event not in self.event_types_:
            raise ValueError(f"Unknown event {event}")
        k_idx = self.event_types_.index(event)
        K = len(self.event_types_)

        X_t = self._prepare_X(X)
        self.network_.eval()
        self.baselines_.eval()

        theta = self.network_(X_t)                   # (N, K)
        t_max = float(self.breakpoints_[-1])

        # Dense evaluation grid
        grid = torch.linspace(0, t_max, n_grid, device=self.device_)

        # Precompute F_k(s) and f_k(s) on the grid for each cause
        F_grid = torch.zeros(K, n_grid, device=self.device_)
        f_grid = torch.zeros(K, n_grid, device=self.device_)
        for k in range(K):
            Fk, fk = self.baselines_[k](grid)
            F_grid[k] = Fk
            f_grid[k] = fk

        # Overall survival on grid: S(s) = exp(-sum_k theta_k F_k(s))
        # theta: (N, K), F_grid: (K, T) -> (N, T)
        theta_F = torch.einsum('nk,kt->nt', theta, F_grid)
        S_grid = torch.exp(-theta_F)                 # (N, T)

        # Subdensity for cause k: theta_k * f_k(s) * S(s)
        # theta[:, k_idx]: (N,), f_grid[k_idx]: (T,)
        sub_density = (
            theta[:, k_idx].unsqueeze(1)
            * f_grid[k_idx].unsqueeze(0)
            * S_grid
        )                                             # (N, T)

        # Trapezoidal integration -> CIF on grid
        dt = grid[1:] - grid[:-1]                     # (T-1,)
        avg = 0.5 * (sub_density[:, :-1] + sub_density[:, 1:])
        cif_grid = torch.zeros_like(sub_density)
        cif_grid[:, 1:] = torch.cumsum(avg * dt.unsqueeze(0), dim=1)

        # Interpolate to requested times
        if times is None:
            return cif_grid.cpu().numpy()

        times_t = torch.from_numpy(
            np.asarray(times, dtype='float32')
        ).to(self.device_)
        # Clamp to grid range
        times_t = times_t.clamp(0, t_max)

        # Linear interpolation
        N = X_t.shape[0]
        T_out = len(times)
        result = torch.zeros(N, T_out, device=self.device_)
        for i, t_val in enumerate(times_t):
            idx = torch.searchsorted(grid, t_val).clamp(1, n_grid - 1)
            w = (t_val - grid[idx - 1]) / (grid[idx] - grid[idx - 1] + 1e-12)
            result[:, i] = (
                (1 - w) * cif_grid[:, idx - 1] + w * cif_grid[:, idx]
            )

        return result.cpu().numpy()

    # ------------------------------------------------------------------
    # predict_survival
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_survival(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        times: Optional[np.ndarray] = None,
        n_grid: int = 200,
    ) -> np.ndarray:
        """Overall survival S(t) = exp(-sum_k theta_k F_k(t)).

        Returns shape (N, len(times)).
        """
        K = len(self.event_types_)
        X_t = self._prepare_X(X)
        self.network_.eval()
        self.baselines_.eval()

        theta = self.network_(X_t)
        t_max = float(self.breakpoints_[-1])

        if times is None:
            grid = torch.linspace(0, t_max, n_grid, device=self.device_)
        else:
            grid = torch.from_numpy(
                np.asarray(times, dtype='float32')
            ).to(self.device_)

        F_grid = torch.zeros(K, len(grid), device=self.device_)
        for k in range(K):
            Fk, _ = self.baselines_[k](grid)
            F_grid[k] = Fk

        S = torch.exp(-torch.einsum('nk,kt->nt', theta, F_grid))
        return S.cpu().numpy()

    # ------------------------------------------------------------------
    # predict_cure_fraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_cure_fraction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Cure fractions pi_k(x) = exp(-theta_k(x)).

        Returns dict with keys for each event name plus 'overall'.
        """
        X_t = self._prepare_X(X)
        self.network_.eval()
        theta = self.network_(X_t)                   # (N, K)

        result: Dict[str, np.ndarray] = {}
        event_names = {1: 'prepay', 2: 'default'}
        for k_idx, k_code in enumerate(self.event_types_):
            name = event_names.get(k_code, f'event_{k_code}')
            pi_k = torch.exp(-theta[:, k_idx])
            result[name] = pi_k.cpu().numpy()

        # Overall: never experience any event
        pi_all = torch.exp(-theta.sum(dim=1))
        result['overall'] = pi_all.cpu().numpy()
        return result

    # ------------------------------------------------------------------
    # get_linear_coefficients (orthogonalized model)
    # ------------------------------------------------------------------
    def get_linear_coefficients(self) -> Dict[str, pd.DataFrame]:
        """Extract linear coefficients from orthogonalized model.

        Returns a dict mapping event name to a DataFrame with columns
        ['feature', 'coefficient'].
        """
        if not self.orthogonalize:
            raise ValueError("Only available when orthogonalize=True")
        if self.network_ is None:
            raise ValueError("Model not fitted")

        event_names = {1: 'prepay', 2: 'default'}
        result = {}
        for k_idx, k_code in enumerate(self.event_types_):
            lin = self.network_.linears[k_idx]
            w = lin.weight.detach().cpu().numpy().flatten()
            b = lin.bias.detach().cpu().numpy().item()
            rows = [{'feature': f, 'coefficient': c}
                    for f, c in zip(self.feature_names_, w)]
            rows.append({'feature': '(intercept)', 'coefficient': b})
            name = event_names.get(k_code, f'event_{k_code}')
            result[name] = pd.DataFrame(rows)
        return result

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save model state to disk."""
        state = {
            'network': self.network_.state_dict(),
            'baselines': self.baselines_.state_dict(),
            'breakpoints': self.breakpoints_,
            'feature_names': self.feature_names_,
            'event_types': self.event_types_,
            'scaler_mean': self.scaler_.mean_ if self.scaler_ else None,
            'scaler_scale': self.scaler_.scale_ if self.scaler_ else None,
            'history': self.history_,
            'params': {
                'num_intervals': self.num_intervals,
                'shared_layers': self.shared_layers,
                'head_layers': self.head_layers,
                'dropout': self.dropout,
                'batch_norm': self.batch_norm,
                'orthogonalize': self.orthogonalize,
            },
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Load a saved model."""
        if device is None:
            device = get_device()
        state = torch.load(path, map_location=device, weights_only=False)
        p = state['params']
        model = cls(
            num_intervals=p['num_intervals'],
            shared_layers=p['shared_layers'],
            head_layers=p['head_layers'],
            dropout=p['dropout'],
            batch_norm=p['batch_norm'],
            orthogonalize=p['orthogonalize'],
        )
        model.device_ = device
        model.breakpoints_ = state['breakpoints']
        model.feature_names_ = state['feature_names']
        model.event_types_ = state['event_types']
        model.history_ = state.get('history', {})

        K = len(model.event_types_)
        in_f = len(model.feature_names_)

        model.network_ = DeepPTCMNetwork(
            in_features=in_f,
            num_causes=K,
            shared_layers=p['shared_layers'],
            head_layers=p['head_layers'],
            dropout=p['dropout'],
            batch_norm=p['batch_norm'],
            orthogonalize=p['orthogonalize'],
        ).to(device)
        model.network_.load_state_dict(state['network'])
        model.network_.eval()

        model.baselines_ = nn.ModuleList([
            PiecewiseExponentialBaseline(model.breakpoints_) for _ in range(K)
        ]).to(device)
        model.baselines_.load_state_dict(state['baselines'])
        model.baselines_.eval()

        if state['scaler_mean'] is not None:
            model.scaler_ = StandardScaler()
            model.scaler_.mean_ = state['scaler_mean']
            model.scaler_.scale_ = state['scaler_scale']
            model.scaler_.n_features_in_ = in_f

        return model


# ===================================================================
# 5. Convenience function
# ===================================================================

def fit_deep_ptcm_competing_risks(
    df: pd.DataFrame,
    feature_cols: List[str],
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    event_types: List[int] = [1, 2],
    val_df: Optional[pd.DataFrame] = None,
    **ptcm_params,
) -> CompetingRisksDeepPTCM:
    """Convenience function to fit Deep-PTCM for competing risks.

    Parameters
    ----------
    df : training DataFrame
    feature_cols : column names for features
    duration_col, event_col : column names
    event_types : event codes to model
    val_df : optional validation DataFrame
    **ptcm_params : passed to CompetingRisksDeepPTCM

    Returns
    -------
    CompetingRisksDeepPTCM
    """
    X = df[feature_cols]
    dur = df[duration_col].values
    ev = df[event_col].values

    val_data = None
    if val_df is not None:
        val_data = (
            val_df[feature_cols],
            val_df[duration_col].values,
            val_df[event_col].values,
        )

    model = CompetingRisksDeepPTCM(**ptcm_params)
    model.fit(X, dur, ev, event_types=event_types, val_data=val_data)
    return model
