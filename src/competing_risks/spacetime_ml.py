"""
Spatio-temporal LaGaBoost for competing risks (prepay + default).

Replicates Kundig & Sigrist (2025) — "A spatio-temporal machine learning model
for mortgage credit risk" — and extends it from single-event default to two
cause-specific hazards using independent fits per cause.

Five variants per cause, controlled by ``variant``:

    'linear_independent'      : sklearn LogisticRegression, no GP
    'linear_spatial'          : linear predictor + Matern(s) GP
    'linear_spatio_temporal'  : linear predictor + Matern(t, s) GP
    'lagaboost_spatial'       : tree-boosted predictor + Matern(s) GP
    'lagaboost_spatio_temporal': tree-boosted predictor + Matern(t, s) GP

The hazard model for cause k in year t+1 conditional on being active at year t::

    P(event_k = 1 | active, X, b) = sigmoid( F_k(X) + b_k(t, lat, lon) )

For the spatio-temporal kernel GPBoost expects ``gp_coords`` with the time
coordinate first, followed by the spatial coordinates.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Sequence, Union

# GPBoost ships its own OpenMP runtime which conflicts with sklearn's libomp on
# macOS once both have been initialised in the same process. Forcing a single
# OMP thread before gpboost's native lib loads avoids the pthread_mutex_init
# crash with negligible runtime cost on the model sizes used here.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd

import gpboost as gpb
from sklearn.linear_model import LogisticRegression


VARIANTS = (
    'linear_independent',
    'linear_spatial',
    'linear_spatio_temporal',
    'lagaboost_spatial',
    'lagaboost_spatio_temporal',
)


def _coords_for(variant: str, year, lat, lon) -> Optional[np.ndarray]:
    """Build the GP coordinate array for the variant; None for independent."""
    if variant == 'linear_independent':
        return None
    if variant.endswith('spatio_temporal'):
        return np.column_stack([
            np.asarray(year, dtype=float),
            np.asarray(lat, dtype=float),
            np.asarray(lon, dtype=float),
        ])
    if variant.endswith('spatial'):
        return np.column_stack([
            np.asarray(lat, dtype=float),
            np.asarray(lon, dtype=float),
        ])
    raise ValueError(f'Unknown variant: {variant}')


def _make_gp_model(variant: str, coords: np.ndarray, num_neighbors: int) -> gpb.GPModel:
    cov = ('matern_space_time' if variant.endswith('spatio_temporal')
           else 'matern')
    return gpb.GPModel(
        gp_coords=coords,
        cov_function=cov,
        cov_fct_shape=1.5,
        likelihood='bernoulli_logit',
        gp_approx='vecchia',
        num_neighbors=num_neighbors,
    )


class _SingleCauseModel:
    """One cause's spatio-temporal hazard model."""

    def __init__(
        self,
        variant: str,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_data_in_leaf: int = 100,
        lambda_l2: float = 0.0,
        num_boost_round: int = 100,
        num_neighbors: int = 20,
        max_iter: int = 200,
        verbose: bool = False,
    ):
        if variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}')
        self.variant = variant
        self.tree_params = {
            'objective': 'binary',
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            'lambda_l2': lambda_l2,
            'verbose': -1,
        }
        self.num_boost_round = num_boost_round
        self.num_neighbors = num_neighbors
        self.max_iter = max_iter
        self.verbose = verbose

        # Filled in by fit()
        self.gp_model_ = None
        self.bst_ = None
        self.linear_ = None
        self.feature_cols_: List[str] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        year: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
    ) -> '_SingleCauseModel':
        self.feature_cols_ = list(X.columns)
        X_arr = X.values.astype(float)
        y = np.asarray(y, dtype=int)

        if self.variant == 'linear_independent':
            self.linear_ = LogisticRegression(
                C=1.0, max_iter=self.max_iter, random_state=0, solver='lbfgs',
            )
            self.linear_.fit(X_arr, y)
            return self

        coords = _coords_for(self.variant, year, lat, lon)

        if self.variant.startswith('linear'):
            X_int = np.hstack([np.ones((len(X_arr), 1)), X_arr])
            gp = _make_gp_model(self.variant, coords, self.num_neighbors)
            gp.fit(y=y, X=X_int, params={'maxit': self.max_iter})
            self.gp_model_ = gp
        else:
            gp = _make_gp_model(self.variant, coords, self.num_neighbors)
            ds = gpb.Dataset(X_arr, label=y)
            self.bst_ = gpb.train(
                params=self.tree_params,
                train_set=ds,
                gp_model=gp,
                num_boost_round=self.num_boost_round,
            )
            self.gp_model_ = gp
        return self

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame,
        year: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_arr = X[self.feature_cols_].values.astype(float)

        if self.variant == 'linear_independent':
            return self.linear_.predict_proba(X_arr)[:, 1]

        coords = _coords_for(self.variant, year, lat, lon)

        if self.variant.startswith('linear'):
            X_int = np.hstack([np.ones((len(X_arr), 1)), X_arr])
            pred = self.gp_model_.predict(
                X_pred=X_int, gp_coords_pred=coords, predict_response=True,
            )
            return np.asarray(pred['mu']).flatten()

        pred = self.bst_.predict(
            data=X_arr, gp_coords_pred=coords,
            predict_var=False, pred_latent=False,
        )
        return np.asarray(pred['response_mean']).flatten()


class CompetingRisksLaGaBoost:
    """Cause-specific spatio-temporal LaGaBoost for prepay + default.

    Mirrors the API of CompetingRisksDeepPTCM where practical, but operates on
    the yearly panel rather than time-to-event data.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.
    learning_rate, max_depth, min_data_in_leaf, lambda_l2, num_boost_round
        Tree-boosting hyperparameters (paper Table A3).
    num_neighbors : int
        Vecchia approximation neighbour count (paper uses 20).
    random_state : int
    verbose : bool
    """

    PREPAY_LABEL = 'prepay_in_year'
    DEFAULT_LABEL = 'default_in_year'

    def __init__(
        self,
        variant: str = 'lagaboost_spatio_temporal',
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_data_in_leaf: int = 100,
        lambda_l2: float = 0.0,
        num_boost_round: int = 100,
        num_neighbors: int = 20,
        max_iter: int = 200,
        random_state: int = 42,
        verbose: bool = False,
    ):
        if variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}')
        self.variant = variant
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.num_boost_round = num_boost_round
        self.num_neighbors = num_neighbors
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        # Filled in by fit()
        self.prepay_model_: Optional[_SingleCauseModel] = None
        self.default_model_: Optional[_SingleCauseModel] = None
        self.feature_cols_: List[str] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        panel_df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> 'CompetingRisksLaGaBoost':
        self.feature_cols_ = list(feature_cols)

        X = panel_df[self.feature_cols_]
        y_p = panel_df[self.PREPAY_LABEL].values
        y_d = panel_df[self.DEFAULT_LABEL].values
        year = panel_df['year'].values
        lat = panel_df['lat'].values
        lon = panel_df['lon'].values

        kw = dict(
            variant=self.variant,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_data_in_leaf=self.min_data_in_leaf,
            lambda_l2=self.lambda_l2,
            num_boost_round=self.num_boost_round,
            num_neighbors=self.num_neighbors,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )
        if self.verbose:
            print(f'Fitting {self.variant} (prepay): n={len(panel_df):,}, events={int(y_p.sum()):,}')
        self.prepay_model_ = _SingleCauseModel(**kw).fit(X, y_p, year, lat, lon)
        if self.verbose:
            print(f'Fitting {self.variant} (default): n={len(panel_df):,}, events={int(y_d.sum()):,}')
        self.default_model_ = _SingleCauseModel(**kw).fit(X, y_d, year, lat, lon)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Returns dict with 'prepay' and 'default' keys, each a 1-D array."""
        X = panel_df[self.feature_cols_]
        year = panel_df['year'].values if 'year' in panel_df.columns else None
        lat = panel_df['lat'].values if 'lat' in panel_df.columns else None
        lon = panel_df['lon'].values if 'lon' in panel_df.columns else None
        return {
            'prepay': self.prepay_model_.predict_proba(X, year, lat, lon),
            'default': self.default_model_.predict_proba(X, year, lat, lon),
        }

    # ------------------------------------------------------------------
    def get_cov_pars(self) -> Dict[str, Optional[pd.DataFrame]]:
        """Posterior covariance hyperparameters for each cause (None for independent)."""
        out = {}
        for name, m in [('prepay', self.prepay_model_), ('default', self.default_model_)]:
            if m is None or m.gp_model_ is None:
                out[name] = None
            else:
                out[name] = m.gp_model_.get_cov_pars()
        return out
