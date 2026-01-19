"""
DeepHit model for Competing Risks Survival Analysis.

This module wraps the pycox DeepHit implementation for use with
the Blumenstock et al. (2022) replication framework.

DeepHit (Lee et al., 2018) is a deep learning approach that:
- Directly models the probability mass function (PMF) over discrete time
- Uses a combined loss: alpha * NLL + (1-alpha) * ranking_loss
- Handles competing risks with cause-specific output layers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

try:
    import torch
    import torch.nn as nn
    import torchtuples as tt
    from pycox.models import DeepHit
    from pycox.preprocessing.label_transforms import LabTransDiscreteTime
    PYCOX_AVAILABLE = True
except ImportError:
    PYCOX_AVAILABLE = False
    DeepHit = None


class LabTransformCompetingRisks(LabTransDiscreteTime):
    """
    Label transformer for competing risks that discretizes time
    while preserving event type information.
    """

    def transform(self, durations, events):
        """
        Transform continuous durations to discrete time indices.

        Parameters
        ----------
        durations : array-like
            Time to event or censoring
        events : array-like
            Event codes (0=censored, 1+=event types)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (discretized durations, event codes)
        """
        durations, is_event = super().transform(durations, events > 0)
        events = np.asarray(events).copy()
        events[is_event == 0] = 0
        return durations, events.astype('int64')


class CauseSpecificNet(nn.Module):
    """
    Neural network architecture for DeepHit with competing risks.

    Architecture follows the DeepHit paper:
    - Shared representation layers
    - Cause-specific sub-networks for each event type

    Parameters
    ----------
    in_features : int
        Number of input features
    num_nodes_shared : list
        Number of nodes in shared layers
    num_nodes_indiv : list
        Number of nodes in cause-specific layers
    num_risks : int
        Number of competing risks (event types)
    out_features : int
        Number of discrete time points
    batch_norm : bool
        Whether to use batch normalization
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        num_nodes_shared: List[int],
        num_nodes_indiv: List[int],
        num_risks: int,
        out_features: int,
        batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_risks = num_risks
        self.out_features = out_features

        # Shared representation layers
        shared_layers = []
        prev_size = in_features
        for nodes in num_nodes_shared:
            shared_layers.append(nn.Linear(prev_size, nodes))
            if batch_norm:
                shared_layers.append(nn.BatchNorm1d(nodes))
            shared_layers.append(nn.ReLU())
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            prev_size = nodes
        self.shared = nn.Sequential(*shared_layers)

        # Cause-specific sub-networks
        self.risk_nets = nn.ModuleList()
        for _ in range(num_risks):
            risk_layers = []
            prev_size_risk = prev_size
            for nodes in num_nodes_indiv:
                risk_layers.append(nn.Linear(prev_size_risk, nodes))
                if batch_norm:
                    risk_layers.append(nn.BatchNorm1d(nodes))
                risk_layers.append(nn.ReLU())
                if dropout > 0:
                    risk_layers.append(nn.Dropout(dropout))
                prev_size_risk = nodes
            risk_layers.append(nn.Linear(prev_size_risk, out_features))
            self.risk_nets.append(nn.Sequential(*risk_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            Output PMF logits, shape (batch_size, num_risks, out_features)
        """
        shared_repr = self.shared(x)

        # Stack outputs from each cause-specific network
        outputs = []
        for risk_net in self.risk_nets:
            outputs.append(risk_net(shared_repr))

        # Shape: (batch_size, num_risks, out_features)
        return torch.stack(outputs, dim=1)


class CompetingRisksDeepHit(BaseEstimator):
    """
    DeepHit wrapper for competing risks survival analysis.

    Provides a scikit-learn-like interface compatible with the
    evaluation framework used in this project.

    Parameters
    ----------
    num_durations : int
        Number of discrete time points for discretization
    num_nodes_shared : list
        Number of nodes in shared layers
    num_nodes_indiv : list
        Number of nodes in cause-specific layers
    batch_norm : bool
        Whether to use batch normalization
    dropout : float
        Dropout rate
    alpha : float
        Weight between NLL and ranking loss (0-1)
        alpha=1 gives only NLL, alpha=0 gives only ranking loss
    sigma : float
        Parameter for ranking loss
    lr : float
        Learning rate
    weight_decay : float
        L2 regularization weight
    batch_size : int
        Training batch size
    epochs : int
        Maximum training epochs
    patience : int
        Early stopping patience (in cycles for AdamWR)
    verbose : bool
        Whether to print training progress
    random_state : int
        Random seed
    """

    def __init__(
        self,
        num_durations: int = 100,
        num_nodes_shared: List[int] = [64, 64],
        num_nodes_indiv: List[int] = [32],
        batch_norm: bool = True,
        dropout: float = 0.1,
        alpha: float = 0.2,
        sigma: float = 0.1,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        batch_size: int = 256,
        epochs: int = 512,
        patience: int = 10,
        verbose: bool = True,
        random_state: int = 42,
    ):
        if not PYCOX_AVAILABLE:
            raise ImportError(
                "pycox and PyTorch are required for DeepHit. "
                "Install with: pip install pycox torch torchtuples"
            )

        self.num_durations = num_durations
        self.num_nodes_shared = num_nodes_shared
        self.num_nodes_indiv = num_nodes_indiv
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.alpha = alpha
        self.sigma = sigma
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state

        self.model_: Optional[DeepHit] = None
        self.labtrans_: Optional[LabTransformCompetingRisks] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []
        self.event_types_: List[int] = []
        self.duration_index_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        duration: np.ndarray,
        event_code: np.ndarray,
        event_types: List[int] = [1, 2],
        val_data: Optional[Tuple] = None,
        scale_features: bool = True,
    ) -> "CompetingRisksDeepHit":
        """
        Fit the DeepHit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        duration : array-like of shape (n_samples,)
            Observed times
        event_code : array-like of shape (n_samples,)
            Event codes (0=censored, 1=prepay, 2=default, etc.)
        event_types : list
            Event codes to model (excluding censored=0)
        val_data : tuple, optional
            Validation data as (X_val, duration_val, event_code_val)
        scale_features : bool
            Whether to standardize features

        Returns
        -------
        self
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.event_types_ = event_types
        num_risks = len(event_types)

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values.astype('float32')
        else:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]
            X = np.asarray(X).astype('float32')

        duration = np.asarray(duration).astype('float32')
        event_code = np.asarray(event_code).astype('int64')

        # Scale features
        if scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X).astype('float32')

        # Create label transformer
        self.labtrans_ = LabTransformCompetingRisks(self.num_durations)
        y_train = self.labtrans_.fit_transform(duration, event_code)
        self.duration_index_ = self.labtrans_.cuts

        # Build network
        in_features = X.shape[1]
        out_features = self.labtrans_.out_features

        net = CauseSpecificNet(
            in_features=in_features,
            num_nodes_shared=self.num_nodes_shared,
            num_nodes_indiv=self.num_nodes_indiv,
            num_risks=num_risks,
            out_features=out_features,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
        )

        # Create optimizer
        optimizer = tt.optim.AdamWR(
            lr=self.lr,
            decoupled_weight_decay=self.weight_decay,
            cycle_eta_multiplier=0.8,
        )

        # Create DeepHit model
        self.model_ = DeepHit(
            net,
            optimizer,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=self.duration_index_,
        )

        # Prepare validation data if provided
        if val_data is not None:
            X_val, dur_val, ev_val = val_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values.astype('float32')
            else:
                X_val = np.asarray(X_val).astype('float32')

            if self.scaler_ is not None:
                X_val = self.scaler_.transform(X_val).astype('float32')

            y_val = self.labtrans_.transform(
                np.asarray(dur_val).astype('float32'),
                np.asarray(ev_val).astype('int64')
            )
            val = (X_val, y_val)

            callbacks = [tt.callbacks.EarlyStoppingCycle(patience=self.patience)]
        else:
            val = None
            callbacks = []

        # Train
        if self.verbose:
            print(f"  Training DeepHit with {num_risks} risks...")

        self.model_.fit(
            X, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            val_data=val,
            verbose=self.verbose,
        )

        return self

    def predict_risk(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict risk scores for a specific event.

        Uses cumulative incidence at the maximum observed time as risk score.

        Parameters
        ----------
        X : array-like
            Feature matrix
        event : int
            Event code to predict (1 or 2)
        time : float, optional
            Time point for CIF evaluation. If None, uses max time.

        Returns
        -------
        np.ndarray
            Risk scores (higher = more risk)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if event not in self.event_types_:
            raise ValueError(
                f"Event {event} not in trained event types: {self.event_types_}"
            )

        # Get event index (0-indexed for model output)
        event_idx = self.event_types_.index(event)

        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X.values.astype('float32')
        else:
            X = np.asarray(X).astype('float32')

        if self.scaler_ is not None:
            X = self.scaler_.transform(X).astype('float32')

        # Predict CIF
        cif = self.model_.predict_cif(X)  # Shape: (num_risks, n_samples, n_times)

        if time is not None:
            # Find closest time index
            time_idx = np.searchsorted(self.duration_index_, time)
            time_idx = min(time_idx, len(self.duration_index_) - 1)
            return cif[event_idx, :, time_idx]
        else:
            # Use CIF at maximum time as risk score
            return cif[event_idx, :, -1]

    def predict_cumulative_incidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict cumulative incidence function for a specific event.

        Parameters
        ----------
        X : array-like
            Feature matrix
        event : int
            Event code
        times : array-like, optional
            Time points for prediction. If None, uses all discretized times.

        Returns
        -------
        np.ndarray
            Cumulative incidence, shape (n_samples, n_times)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if event not in self.event_types_:
            raise ValueError(
                f"Event {event} not in trained event types: {self.event_types_}"
            )

        event_idx = self.event_types_.index(event)

        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X.values.astype('float32')
        else:
            X = np.asarray(X).astype('float32')

        if self.scaler_ is not None:
            X = self.scaler_.transform(X).astype('float32')

        # Predict CIF
        cif = self.model_.predict_cif(X)  # (num_risks, n_samples, n_times)
        cif_event = cif[event_idx].T  # (n_samples, n_times)

        if times is not None:
            # Interpolate to requested times
            from scipy.interpolate import interp1d
            interpolated = np.zeros((cif_event.shape[0], len(times)))
            for i in range(cif_event.shape[0]):
                f = interp1d(
                    self.duration_index_,
                    cif_event[i],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(0, cif_event[i, -1])
                )
                interpolated[i] = f(times)
            return interpolated

        return cif_event

    def predict_survival(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict overall survival (avoiding all events).

        Parameters
        ----------
        X : array-like
            Feature matrix
        times : array-like, optional
            Time points for prediction

        Returns
        -------
        np.ndarray
            Survival probabilities, shape (n_samples, n_times)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X.values.astype('float32')
        else:
            X = np.asarray(X).astype('float32')

        if self.scaler_ is not None:
            X = self.scaler_.transform(X).astype('float32')

        # Predict survival
        surv = self.model_.predict_surv(X)  # (n_samples, n_times)

        if times is not None:
            from scipy.interpolate import interp1d
            interpolated = np.zeros((surv.shape[0], len(times)))
            for i in range(surv.shape[0]):
                f = interp1d(
                    self.duration_index_,
                    surv[i],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(1, surv[i, -1])
                )
                interpolated[i] = f(times)
            return interpolated

        return surv


def fit_deephit_competing_risks(
    df: pd.DataFrame,
    feature_cols: List[str],
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    event_types: List[int] = [1, 2],
    val_df: Optional[pd.DataFrame] = None,
    **deephit_params,
) -> CompetingRisksDeepHit:
    """
    Convenience function to fit DeepHit for competing risks.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with features, duration, and event code
    feature_cols : list
        Column names for features
    duration_col : str
        Duration column name
    event_col : str
        Event code column name
    event_types : list
        Event codes to model
    val_df : pd.DataFrame, optional
        Validation data
    **deephit_params
        Parameters for CompetingRisksDeepHit

    Returns
    -------
    CompetingRisksDeepHit
        Fitted model
    """
    X = df[feature_cols]
    duration = df[duration_col].values
    event_code = df[event_col].values

    val_data = None
    if val_df is not None:
        val_data = (
            val_df[feature_cols],
            val_df[duration_col].values,
            val_df[event_col].values,
        )

    model = CompetingRisksDeepHit(**deephit_params)
    model.fit(X, duration, event_code, event_types=event_types, val_data=val_data)

    return model
