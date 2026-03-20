"""
Dynamic-DeepHit for Competing Risks Survival Analysis.

Implements Dynamic-DeepHit (Lee et al., 2020) for mortgage competing risks
(prepayment + default) using longitudinal loan-month panel data.

Dynamic-DeepHit extends the original DeepHit by processing the full monthly
history via a GRU + temporal attention mechanism, enabling dynamic predictions
that update as new data arrives.

Reference:
    Lee, C., Yoon, J., & van der Schaar, M. (2020).
    Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis
    with Competing Risks Based on Longitudinal Data. IEEE TBME.

Architecture:
    Input (batch, seq_len, 21) + lengths
      -> Input Embedding: Linear(21->64) + ReLU
      -> GRU (2 layers, hidden=128, dropout=0.6)
      -> Temporal Attention -> context vector (batch, 128)
      -> Concatenate [context; x_J] -> (batch, 149)
      -> Cause-Specific Heads (K=2): 149->128->64->num_time_bins
      -> Joint Softmax -> PMF (batch, 2, num_time_bins)
      -> Next-Step Predictor (for L3 loss): Linear(128->16)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Feature constants
# ============================================================================

STATIC_FEATURES = ['int_rate', 'orig_upb', 'fico_score', 'dti_r', 'ltv_r']
BEHAVIORAL_FEATURES = ['bal_repaid', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m']
MACRO_FEATURES = [
    'hpi_st_d_t_o', 'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o',
    'ppi_o_FRMA', 'hpi_st_log12m', 'hpi_r_st_us', 'st_unemp_r12m',
    'st_unemp_r3m', 'TB10Y_r12m', 'T10Y3MM', 'T10Y3MM_r12m',
]
ALL_FEATURES = STATIC_FEATURES + BEHAVIORAL_FEATURES + MACRO_FEATURES

# Time-varying features used for L3 next-step prediction
# (behavioral + macro, 16 features that change over time)
TIME_VARYING_FEATURES = BEHAVIORAL_FEATURES + MACRO_FEATURES


# ============================================================================
# Data Pipeline
# ============================================================================

def preprocess_panel_to_sequences(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = 'loan_age',
    event_col: str = 'event_code',
    loan_id_col: str = 'loan_sequence_number',
    fold_col: str = 'fold',
    max_seq_len: int = 120,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[List[Dict], Optional[StandardScaler], List[str]]:
    """
    Convert panel DataFrame into a list of per-loan sequence dictionaries.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Loan-month panel data.
    feature_cols : list
        Feature column names (pre-transformation).
    time_col : str
        Column with loan age / duration.
    event_col : str
        Column with event code (0=censored, 1=prepay, 2=default).
    loan_id_col : str
        Column identifying unique loans.
    fold_col : str
        Column with fold assignments.
    max_seq_len : int
        Maximum sequence length; longer sequences are right-truncated
        (keep most recent observations).
    scaler : StandardScaler, optional
        Pre-fitted scaler.  If None and fit_scaler=True, a new one is fitted.
    fit_scaler : bool
        Whether to fit a new scaler on this data.

    Returns
    -------
    sequences : list of dict
        Each dict has keys: 'x' (T, D), 'duration' (float), 'event' (int),
        'length' (int), 'fold' (int), 'loan_id' (str).
    scaler : StandardScaler or None
    final_feature_cols : list of str
        The feature column names after transformations.
    """
    df = panel_df.copy()

    # Feature engineering: lag bal_repaid by 1 within each loan
    if 'bal_repaid' in feature_cols:
        df['bal_repaid'] = df.groupby(loan_id_col)['bal_repaid'].shift(1)
        # First month has no lag; fill with 0
        df['bal_repaid'] = df['bal_repaid'].fillna(0.0)

    # Log-transform orig_upb
    final_feature_cols = list(feature_cols)
    if 'orig_upb' in final_feature_cols:
        df['log_upb'] = np.log(df['orig_upb'].clip(lower=1.0))
        idx = final_feature_cols.index('orig_upb')
        final_feature_cols[idx] = 'log_upb'

    # Drop rows with NaN in features (should be rare)
    df = df.dropna(subset=final_feature_cols)

    # Fit or apply scaler
    if fit_scaler:
        scaler = StandardScaler()
        df[final_feature_cols] = scaler.fit_transform(
            df[final_feature_cols].values
        ).astype('float32')
    elif scaler is not None:
        df[final_feature_cols] = scaler.transform(
            df[final_feature_cols].values
        ).astype('float32')

    # Group by loan and build sequences
    df = df.sort_values([loan_id_col, time_col])
    sequences = []

    for loan_id, group in df.groupby(loan_id_col):
        x = group[final_feature_cols].values.astype('float32')
        duration = float(group[time_col].iloc[-1])
        event = int(group[event_col].iloc[-1])
        fold = int(group[fold_col].iloc[0])

        # Right-truncate: keep last max_seq_len observations
        if len(x) > max_seq_len:
            x = x[-max_seq_len:]

        sequences.append({
            'x': x,               # (T, D) numpy array
            'duration': duration,
            'event': event,
            'length': len(x),
            'fold': fold,
            'loan_id': loan_id,
        })

    return sequences, scaler, final_feature_cols


class MortgageSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed loan sequences.

    Parameters
    ----------
    sequences : list of dict
        Output from ``preprocess_panel_to_sequences``.
    max_seq_len : int
        Maximum sequence length (for right-truncation at access time).
    """

    def __init__(self, sequences: List[Dict], max_seq_len: int = 120):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq['x'], dtype=torch.float32)

        # Right-truncate if needed (safety; should already be done)
        if x.shape[0] > self.max_seq_len:
            x = x[-self.max_seq_len:]

        return {
            'x': x,
            'duration': torch.tensor(seq['duration'], dtype=torch.float32),
            'event': torch.tensor(seq['event'], dtype=torch.long),
            'length': x.shape[0],
        }


def collate_mortgage_sequences(batch: List[Dict]) -> Tuple:
    """
    Custom collate function that pads variable-length sequences.

    Returns
    -------
    x_padded : Tensor (batch, max_len, D)
    lengths : Tensor (batch,)
    durations : Tensor (batch,)
    events : Tensor (batch,)
    """
    xs = [item['x'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    durations = torch.stack([item['duration'] for item in batch])
    events = torch.stack([item['event'] for item in batch])

    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)

    return x_padded, lengths, durations, events


# ============================================================================
# Model Components
# ============================================================================

class SharedRNNSubnetwork(nn.Module):
    """
    GRU-based shared subnetwork that processes longitudinal sequences.

    Input embedding: Linear(in_features -> embed_dim) + ReLU
    GRU: input=embed_dim, hidden=hidden_dim, num_layers, dropout, batch_first
    Uses pack_padded_sequence / pad_packed_sequence for variable-length efficiency.

    Returns all hidden states h_all: (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        in_features: int = 21,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU(),
        )

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self, x_padded: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_padded : (batch, seq_len, in_features)
        lengths : (batch,)

        Returns
        -------
        h_all : (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x_padded.shape

        # Embed
        embedded = self.input_embed(x_padded)  # (batch, seq_len, embed_dim)

        # Run GRU — use pack/unpack on CPU/CUDA, plain GRU on MPS
        # (pack_padded_sequence has known issues with MPS on PyTorch ≤2.4)
        if embedded.device.type in ('cpu', 'cuda'):
            lengths_cpu = lengths.detach().cpu().clamp(min=1)
            packed = pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            h_all, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
        else:
            # MPS fallback: run GRU on full padded sequence
            h_all, _ = self.gru(embedded)
            # Zero out hidden states at padded positions to prevent
            # NaN propagation through downstream BN/attention
            lengths_dev = lengths.to(h_all.device)
            arange = torch.arange(seq_len, device=h_all.device).unsqueeze(0)
            mask = arange < lengths_dev.unsqueeze(1)  # (B, T)
            h_all = h_all * mask.unsqueeze(-1).float()

        return h_all  # (batch, seq_len, hidden_dim)


class TemporalAttention(nn.Module):
    """
    Temporal attention over past hidden states.

    Score: f_a(h_j, x_J) = W2 * tanh(W1 * [h_j; x_J] + b1) + b2
    Masked softmax (ignores padded positions).
    Context: c = sum(a_j * h_j) -> (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 21):
        super().__init__()
        concat_dim = hidden_dim + input_dim
        self.score_net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_all: torch.Tensor,
        x_last: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_all : (batch, seq_len, hidden_dim)
        x_last : (batch, input_dim) — last observed input features
        lengths : (batch,)

        Returns
        -------
        context : (batch, hidden_dim)
        attn_weights : (batch, seq_len)
        """
        batch_size, seq_len, hidden_dim = h_all.shape

        # Expand x_last to match seq_len: (batch, seq_len, input_dim)
        x_last_expanded = x_last.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate [h_j; x_J] for each timestep
        concat = torch.cat([h_all, x_last_expanded], dim=-1)  # (B, T, H+D)

        # Score
        scores = self.score_net(concat).squeeze(-1)  # (batch, seq_len)

        # Create mask: 1 for valid positions, 0 for padded
        # lengths may be on CPU (for pack_padded_sequence compat); move to device
        lengths_dev = lengths.to(h_all.device)
        arange = torch.arange(seq_len, device=h_all.device).unsqueeze(0)  # (1, T)
        mask = arange < lengths_dev.unsqueeze(1)  # (batch, T)

        # Masked softmax
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        # Replace NaN from all-masked rows (shouldn't happen, but safety)
        attn_weights = attn_weights.masked_fill(~mask, 0.0)

        # Context vector
        context = torch.bmm(
            attn_weights.unsqueeze(1), h_all
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attn_weights


class CauseSpecificSubnetwork(nn.Module):
    """
    MLP head for a single cause.

    Input: [context; x_J] -> (batch, hidden_dim + input_dim)
    Architecture: Linear->BN->ReLU->Dropout -> Linear->BN->ReLU->Dropout -> Linear
    Output: (batch, num_time_bins)
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        num_time_bins: int = 120,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_time_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicDeepHitNetwork(nn.Module):
    """
    Dynamic-DeepHit main model (Lee et al., 2020).

    Assembles SharedRNNSubnetwork, TemporalAttention, and K CauseSpecificSubnetworks.

    Parameters
    ----------
    in_features : int
        Number of input features per timestep.
    num_time_bins : int
        Number of discrete time bins for PMF output.
    num_causes : int
        Number of competing risks (default 2: prepay, default).
    embed_dim : int
        Dimension of input embedding.
    hidden_dim : int
        GRU hidden dimension.
    num_rnn_layers : int
        Number of GRU layers.
    head_hidden1 : int
        First hidden layer size in cause-specific heads.
    head_hidden2 : int
        Second hidden layer size in cause-specific heads.
    dropout : float
        Dropout rate.
    num_tv_features : int
        Number of time-varying features for L3 next-step prediction.
    """

    def __init__(
        self,
        in_features: int = 21,
        num_time_bins: int = 120,
        num_causes: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_rnn_layers: int = 2,
        head_hidden1: int = 128,
        head_hidden2: int = 64,
        dropout: float = 0.6,
        num_tv_features: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_time_bins = num_time_bins
        self.num_causes = num_causes
        self.hidden_dim = hidden_dim

        # Shared RNN
        self.shared_rnn = SharedRNNSubnetwork(
            in_features=in_features,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_rnn_layers,
            dropout=dropout,
        )

        # Temporal attention
        self.attention = TemporalAttention(
            hidden_dim=hidden_dim,
            input_dim=in_features,
        )

        # Cause-specific heads
        cs_input_dim = hidden_dim + in_features  # [context; x_J]
        self.cause_heads = nn.ModuleList([
            CauseSpecificSubnetwork(
                input_dim=cs_input_dim,
                hidden1=head_hidden1,
                hidden2=head_hidden2,
                num_time_bins=num_time_bins,
                dropout=dropout,
            )
            for _ in range(num_causes)
        ])

        # Next-step predictor for L3 loss
        self.next_step_predictor = nn.Linear(hidden_dim, num_tv_features)

        # Store last attention weights for interpretability
        self._last_attn_weights = None

    def forward(
        self,
        x_padded: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x_padded : (batch, seq_len, in_features)
        lengths : (batch,)

        Returns
        -------
        pmf : (batch, num_causes, num_time_bins)
            Joint probability mass function after softmax.
        x_pred : (batch, seq_len - 1, num_tv_features)
            Next-step feature predictions for L3 loss.
        """
        batch_size = x_padded.shape[0]

        # 1. Shared RNN
        h_all = self.shared_rnn(x_padded, lengths)  # (B, T, H)

        # 2. Get last valid features x_J for each sample
        # lengths may be on CPU; move to device for indexing
        lengths_dev = lengths.to(x_padded.device)
        last_idx = (lengths_dev - 1).clamp(min=0).long()  # (B,)
        x_last = x_padded[
            torch.arange(batch_size, device=x_padded.device), last_idx
        ]  # (B, D)

        # 3. Temporal attention
        context, attn_weights = self.attention(h_all, x_last, lengths)
        self._last_attn_weights = attn_weights.detach()

        # 4. Concatenate [context; x_J]
        cs_input = torch.cat([context, x_last], dim=-1)  # (B, H+D)

        # 5. Cause-specific heads
        head_logits = [head(cs_input) for head in self.cause_heads]
        logits = torch.stack(head_logits, dim=1)  # (B, K, T_bins)

        # 6. Joint softmax over (cause x time_bins)
        logits_flat = logits.view(batch_size, -1)  # (B, K * T_bins)
        pmf_flat = torch.softmax(logits_flat, dim=-1)
        pmf = pmf_flat.view(
            batch_size, self.num_causes, self.num_time_bins
        )

        # 7. Next-step prediction (for L3)
        x_pred = self.next_step_predictor(h_all[:, :-1, :])  # (B, T-1, D_tv)

        return pmf, x_pred

    def predict_cif(
        self,
        x_padded: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cumulative incidence function: CIF_k(t) = cumsum(pmf_k, dim=time).

        Returns
        -------
        cif : (batch, num_causes, num_time_bins)
        """
        pmf, _ = self.forward(x_padded, lengths)
        return torch.cumsum(pmf, dim=-1)

    def predict_survival(
        self,
        x_padded: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overall survival: S(t) = 1 - sum_k CIF_k(t).

        Returns
        -------
        survival : (batch, num_time_bins)
        """
        cif = self.predict_cif(x_padded, lengths)
        return 1.0 - cif.sum(dim=1)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last forward pass."""
        return self._last_attn_weights


# ============================================================================
# Loss Function
# ============================================================================

class DynamicDeepHitLoss(nn.Module):
    """
    Three-component loss for Dynamic-DeepHit (Lee et al., 2020).

    L = L1 + alpha * L2 + beta * L3

    L1: Conditional NLL — log-likelihood for uncensored + log-survival for censored
    L2: Cause-specific ranking loss with sampled pairs
    L3: Next-step prediction MSE for time-varying covariates

    Parameters
    ----------
    alpha_prepay : float
        Ranking loss alpha for prepayment cause.
    alpha_default : float
        Ranking loss alpha for default cause.
    sigma : float
        Smoothing parameter for ranking loss.
    beta : float
        Weight for L3 next-step prediction loss.
    default_event_weight : float
        NLL weight multiplier for default events (class imbalance).
    num_tv_features : int
        Number of time-varying features for L3.
    tv_feature_indices : list of int, optional
        Indices of time-varying features within the full feature vector.
        If None, uses last ``num_tv_features`` features.
    """

    def __init__(
        self,
        alpha_prepay: float = 0.2,
        alpha_default: float = 1.0,
        sigma: float = 0.1,
        beta: float = 0.1,
        default_event_weight: float = 50.0,
        num_tv_features: int = 16,
        tv_feature_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.alpha_prepay = alpha_prepay
        self.alpha_default = alpha_default
        self.sigma = sigma
        self.beta = beta
        self.default_event_weight = default_event_weight
        self.num_tv_features = num_tv_features
        self.tv_feature_indices = tv_feature_indices

    def forward(
        self,
        pmf: torch.Tensor,
        x_pred: torch.Tensor,
        x_padded: torch.Tensor,
        lengths: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss.

        Parameters
        ----------
        pmf : (batch, num_causes, num_time_bins)
        x_pred : (batch, seq_len-1, num_tv_features) — next-step predictions
        x_padded : (batch, seq_len, in_features) — original padded input
        lengths : (batch,)
        durations : (batch,) — observed loan ages
        events : (batch,) — event codes (0=cens, 1=prepay, 2=default)
        time_bins : (num_time_bins + 1,) — bin edges

        Returns
        -------
        total_loss, l1_loss, l2_loss, l3_loss
        """
        l1 = self._nll_loss(pmf, durations, events, time_bins)
        l2 = self._ranking_loss(pmf, durations, events, time_bins)
        l3 = self._nextstep_loss(x_pred, x_padded, lengths)

        # Guard against NaN (can occur on MPS with edge-case batches)
        l1 = torch.nan_to_num(l1, nan=0.0)
        l2 = torch.nan_to_num(l2, nan=0.0)
        l3 = torch.nan_to_num(l3, nan=0.0)

        total = l1 + l2 + l3
        return total, l1, l2, l3

    def _nll_loss(
        self,
        pmf: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """L1: Conditional NLL with class-imbalance weighting."""
        batch_size = pmf.shape[0]
        num_causes = pmf.shape[1]
        num_bins = pmf.shape[2]
        device = pmf.device
        eps = 1e-7

        # Map durations to bin indices
        bin_indices = torch.bucketize(durations, time_bins[1:])
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        # CIF and survival
        cif = torch.cumsum(pmf, dim=-1)
        total_cif = cif.sum(dim=1)
        survival = torch.clamp(1.0 - total_cif, min=eps)

        # Survival at observed time
        batch_idx = torch.arange(batch_size, device=device)
        survival_at_t = survival[batch_idx, bin_indices]

        # PMF at observed (time, cause) for uncensored
        cause_indices = (events - 1).clamp(min=0).long()
        pmf_at_event = pmf[batch_idx, cause_indices, bin_indices]

        is_censored = (events == 0).float()
        is_prepay = (events == 1).float()
        is_default = (events == 2).float()

        # Weighted NLL
        nll = (
            -torch.log(pmf_at_event + eps) * is_prepay
            - torch.log(pmf_at_event + eps) * is_default * self.default_event_weight
            - torch.log(survival_at_t + eps) * is_censored
        )

        return nll.mean()

    def _ranking_loss(
        self,
        pmf: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """L2: Cause-specific ranking loss with sampled pairs."""
        batch_size = pmf.shape[0]
        num_causes = pmf.shape[1]
        num_bins = pmf.shape[2]
        device = pmf.device

        bin_indices = torch.bucketize(durations, time_bins[1:])
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        cif = torch.cumsum(pmf, dim=-1)

        ranking_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        alphas = [self.alpha_prepay, self.alpha_default]

        for k in range(num_causes):
            alpha_k = alphas[k] if k < len(alphas) else self.alpha_prepay
            if alpha_k == 0:
                continue

            event_code = k + 1
            cause_mask = (events == event_code)
            cause_idx = torch.where(cause_mask)[0]

            if len(cause_idx) < 1:
                continue

            # Sample anchors
            n_anchors = min(100, len(cause_idx))
            if len(cause_idx) > n_anchors:
                perm = torch.randperm(len(cause_idx), device=device)[:n_anchors]
                cause_idx = cause_idx[perm]

            for idx in cause_idx:
                t_i = bin_indices[idx]
                later_mask = bin_indices > t_i
                later_idx = torch.where(later_mask)[0]

                if len(later_idx) == 0:
                    continue

                # Sample comparisons
                n_comp = min(10, len(later_idx))
                if len(later_idx) > n_comp:
                    perm = torch.randperm(len(later_idx), device=device)[:n_comp]
                    later_idx = later_idx[perm]

                diff = cif[later_idx, k, t_i] - cif[idx, k, t_i]
                pair_loss = torch.exp(diff / self.sigma)

                ranking_loss = ranking_loss + alpha_k * pair_loss.sum()
                n_pairs += len(later_idx)

        if n_pairs > 0:
            ranking_loss = ranking_loss / n_pairs

        return ranking_loss

    def _nextstep_loss(
        self,
        x_pred: torch.Tensor,
        x_padded: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """L3: Next-step prediction MSE for time-varying covariates."""
        if self.beta == 0:
            return torch.tensor(0.0, device=x_pred.device)

        batch_size, max_len, in_features = x_padded.shape

        # Extract time-varying features from x_padded
        if self.tv_feature_indices is not None:
            tv_indices = self.tv_feature_indices
        else:
            # Default: last num_tv_features columns
            tv_indices = list(range(
                in_features - self.num_tv_features, in_features
            ))
        tv_indices_t = torch.tensor(tv_indices, device=x_padded.device)

        # Target: x_{t+1} for time-varying features
        # x_padded[:, 1:, tv_indices] is the target (next timestep)
        x_target = x_padded[:, 1:, :][:, :, tv_indices_t]  # (B, T-1, D_tv)

        # Ensure x_pred and x_target have same seq length
        min_len = min(x_pred.shape[1], x_target.shape[1])
        x_pred = x_pred[:, :min_len, :]
        x_target = x_target[:, :min_len, :]

        # Create mask for valid (non-padded) positions
        # Position t is valid if t+1 < length (i.e., both t and t+1 are real)
        arange = torch.arange(min_len, device=x_padded.device).unsqueeze(0)  # (1, T-1)
        lengths_dev = lengths.to(x_padded.device)
        mask = arange < (lengths_dev.unsqueeze(1) - 1)  # (B, T-1)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_pred.device)

        # MSE only over valid positions
        mse = ((x_pred - x_target) ** 2) * mask.unsqueeze(-1).float()
        l3 = self.beta * mse.sum() / (mask.sum() * len(tv_indices))

        return l3
