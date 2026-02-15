"""
Breeden & Crook (2022) Multihorizon Discrete-Time Survival Model.

Implements the multihorizon approach where separate logistic regressions
are fitted for each forecast horizon L=1,...,12. Model L uses delinquency
lagged by L months, so delinquency dominates short horizons while
origination variables (LTV, DTI, FICO) take over at longer horizons.

Adapted for competing risks (prepayment k=1, default k=2) on the
Freddie Mac dataset.

Reference:
    Breeden, J.L. and Crook, J.N. (2022). "Multihorizon discrete time
    survival models." Journal of the Operational Research Society.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import roc_auc_score, log_loss


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DELINQ_STATE = 5

DELINQ_INDICATOR_COLS = ['D_1m', 'D_2m', 'D_3m', 'D_4m', 'D_5m_plus']

STATIC_FEATURES = ['fico_score', 'dti_r', 'ltv_r', 'int_rate', 'log_orig_upb']

BEHAVIORAL_FEATURES = ['bal_repaid_lag1', 't_act_12m', 't_del_30d_12m', 't_del_60d_12m']

MACRO_FEATURES = [
    'hpi_st_d_t_o', 'ppi_c_FRMA', 'TB10Y_d_t_o', 'FRMA30Y_d_t_o',
    'ppi_o_FRMA', 'hpi_st_log12m', 'hpi_r_st_us', 'st_unemp_r12m',
    'st_unemp_r3m', 'TB10Y_r12m', 'T10Y3MM', 'T10Y3MM_r12m',
]

EVENT_NAMES = {0: 'Censored', 1: 'Prepay', 2: 'Default'}


# ---------------------------------------------------------------------------
# Data enrichment: extract delinquency status from raw performance files
# ---------------------------------------------------------------------------

def enrich_panel_with_delinquency(
    panel: pd.DataFrame,
    raw_dir: Union[str, Path],
    years: Optional[List[int]] = None,
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Enrich loan_month_panel with month-by-month delinquency status from raw
    performance files.

    Parameters
    ----------
    panel : pd.DataFrame
        The loan_month_panel with columns including loan_sequence_number, loan_age.
    raw_dir : str or Path
        Path to data/raw/ directory containing sample_YYYY/ subdirectories.
    years : list of int, optional
        Years to process (default: 1999-2025).
    cache_path : str or Path, optional
        If provided and exists, load from cache instead of reprocessing.

    Returns
    -------
    pd.DataFrame
        Panel with added 'delinq_status' column (int, 0-5).
    """
    from src.data.columns import PERFORMANCE_COLUMNS, PERFORMANCE_DTYPES

    raw_dir = Path(raw_dir)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Loading cached enriched panel from {cache_path}")
            return pd.read_parquet(cache_path)

    if years is None:
        years = list(range(1999, 2026))

    # Get the set of loan IDs in our panel
    panel_loans = set(panel['loan_sequence_number'].unique())
    print(f"Panel has {len(panel_loans):,} unique loans")

    # Extract delinquency status from raw performance files
    delinq_records = []
    for year in years:
        svcg_file = raw_dir / f'sample_{year}' / f'sample_svcg_{year}.txt'
        if not svcg_file.exists():
            continue

        print(f"  Reading {svcg_file.name}...", end=' ')
        chunks = pd.read_csv(
            svcg_file,
            sep='|',
            header=None,
            names=PERFORMANCE_COLUMNS,
            dtype=PERFORMANCE_DTYPES,
            usecols=['loan_sequence_number', 'loan_age',
                     'current_loan_delinquency_status'],
        )

        # Filter to our sampled loans
        chunks = chunks[chunks['loan_sequence_number'].isin(panel_loans)]

        # Parse delinquency status
        delinq = pd.to_numeric(
            chunks['current_loan_delinquency_status']
            .replace({'X': '0', 'XX': '0', ' ': '0', '': '0'}),
            errors='coerce',
        ).fillna(0).clip(upper=MAX_DELINQ_STATE).astype(int)

        chunks = chunks[['loan_sequence_number', 'loan_age']].copy()
        chunks['delinq_status'] = delinq.values
        delinq_records.append(chunks)
        print(f"{len(chunks):,} rows")

    if not delinq_records:
        raise ValueError("No raw performance files found. Check raw_dir path.")

    delinq_df = pd.concat(delinq_records, ignore_index=True)

    # Drop duplicates (same loan-month can appear in multiple vintage files)
    delinq_df = delinq_df.drop_duplicates(
        subset=['loan_sequence_number', 'loan_age'], keep='last'
    )
    print(f"Total delinquency records: {len(delinq_df):,}")

    # Merge into panel
    panel = panel.merge(
        delinq_df,
        on=['loan_sequence_number', 'loan_age'],
        how='left',
    )

    # Fill missing delinquency with 0 (current)
    panel['delinq_status'] = panel['delinq_status'].fillna(0).astype(int)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path, index=False)
        print(f"Cached enriched panel to {cache_path}")

    return panel


# ---------------------------------------------------------------------------
# Create delinquency indicators and their lags
# ---------------------------------------------------------------------------

def create_delinquency_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary delinquency state indicators from delinq_status.

    Creates D_1m, D_2m, D_3m, D_4m, D_5m_plus columns.
    Current (0) is the reference category (all indicators = 0).
    """
    panel = panel.copy()
    panel['D_1m'] = (panel['delinq_status'] == 1).astype(np.int8)
    panel['D_2m'] = (panel['delinq_status'] == 2).astype(np.int8)
    panel['D_3m'] = (panel['delinq_status'] == 3).astype(np.int8)
    panel['D_4m'] = (panel['delinq_status'] == 4).astype(np.int8)
    panel['D_5m_plus'] = (panel['delinq_status'] >= 5).astype(np.int8)
    return panel


def create_lagged_delinquency(
    panel: pd.DataFrame,
    max_lag: int = 12,
) -> pd.DataFrame:
    """
    Create lagged delinquency indicators for each horizon L=1,...,max_lag.

    For each delinquency indicator D_Xm, creates D_Xm_lag1, ..., D_Xm_lag{max_lag}.
    """
    panel = panel.sort_values(['loan_sequence_number', 'loan_age'])

    for L in range(1, max_lag + 1):
        for d_col in DELINQ_INDICATOR_COLS:
            col_name = f'{d_col}_lag{L}'
            panel[col_name] = panel.groupby('loan_sequence_number')[d_col].shift(L)

    return panel


def get_delinq_lag_cols(horizon: int) -> List[str]:
    """Get the lagged delinquency column names for a given horizon L."""
    return [f'{d_col}_lag{horizon}' for d_col in DELINQ_INDICATOR_COLS]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_age_splines(
    ages: np.ndarray,
    n_knots: int = 5,
    degree: int = 3,
    fitted_transformer: Optional[SplineTransformer] = None,
) -> Tuple[np.ndarray, SplineTransformer]:
    """
    Create B-spline basis for loan age F(a).

    Returns the spline-transformed array and the fitted transformer.
    """
    ages_2d = ages.reshape(-1, 1)
    if fitted_transformer is None:
        transformer = SplineTransformer(
            n_knots=n_knots, degree=degree, include_bias=False
        )
        result = transformer.fit_transform(ages_2d)
    else:
        transformer = fitted_transformer
        result = transformer.transform(ages_2d)
    return result, transformer


def get_vintage_dummies(
    vintage_years: pd.Series,
    fitted_categories: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create vintage year dummy variables G(v), dropping the first category.
    """
    if fitted_categories is not None:
        cats = fitted_categories
    else:
        cats = np.sort(vintage_years.unique())

    dummies = pd.DataFrame(index=vintage_years.index)
    # Drop first category as reference
    for cat in cats[1:]:
        dummies[f'vintage_{cat}'] = (vintage_years == cat).astype(np.int8)

    return dummies, cats


def build_feature_matrix(
    panel: pd.DataFrame,
    horizon: int,
    age_transformer: Optional[SplineTransformer] = None,
    vintage_categories: Optional[np.ndarray] = None,
    include_delinq: bool = True,
    n_age_knots: int = 5,
) -> Tuple[pd.DataFrame, SplineTransformer, np.ndarray, List[str]]:
    """
    Build the feature matrix for a specific horizon-L model.

    Parameters
    ----------
    panel : pd.DataFrame
        Enriched panel with delinquency indicators and lags.
    horizon : int
        Forecast horizon L (1-12). Controls which lagged delinquency to use.
    age_transformer : SplineTransformer, optional
        Pre-fitted spline transformer for age. If None, fits a new one.
    vintage_categories : np.ndarray, optional
        Pre-fitted vintage categories. If None, derives from data.
    include_delinq : bool
        Whether to include delinquency indicators (False for origination model).
    n_age_knots : int
        Number of knots for age spline.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    age_transformer : SplineTransformer
        Fitted age spline transformer.
    vintage_categories : np.ndarray
        Fitted vintage categories.
    feature_names : list of str
        Feature column names.
    """
    feature_dfs = []
    feature_names = []

    # 1. Age splines F(a)
    ages = panel['loan_age'].values.astype(float)
    age_splines, age_transformer = prepare_age_splines(
        ages, n_knots=n_age_knots, fitted_transformer=age_transformer
    )
    age_cols = [f'age_spline_{i}' for i in range(age_splines.shape[1])]
    feature_dfs.append(pd.DataFrame(age_splines, index=panel.index, columns=age_cols))
    feature_names.extend(age_cols)

    # 2. Vintage dummies G(v)
    vintage_dummies, vintage_categories = get_vintage_dummies(
        panel['vintage_year'], vintage_categories
    )
    feature_dfs.append(vintage_dummies)
    feature_names.extend(vintage_dummies.columns.tolist())

    # 3. Macro features H(t)
    available_macro = [c for c in MACRO_FEATURES if c in panel.columns]
    if available_macro:
        feature_dfs.append(panel[available_macro].reset_index(drop=True))
        feature_names.extend(available_macro)

    # 4. Static origination features
    static_cols = []
    for col in STATIC_FEATURES:
        if col == 'log_orig_upb' and col not in panel.columns:
            if 'orig_upb' in panel.columns:
                panel = panel.copy()
                panel['log_orig_upb'] = np.log(
                    panel['orig_upb'].astype(float).clip(lower=1)
                )
                static_cols.append('log_orig_upb')
        elif col in panel.columns:
            static_cols.append(col)

    if static_cols:
        feature_dfs.append(panel[static_cols].reset_index(drop=True))
        feature_names.extend(static_cols)

    # 5. Behavioral features
    available_behavioral = [c for c in BEHAVIORAL_FEATURES if c in panel.columns]
    if 'bal_repaid_lag1' not in panel.columns and 'bal_repaid' in panel.columns:
        # Use bal_repaid directly if lag not available
        available_behavioral = [
            'bal_repaid' if c == 'bal_repaid_lag1' else c
            for c in available_behavioral
        ]
    if available_behavioral:
        feature_dfs.append(panel[available_behavioral].reset_index(drop=True))
        feature_names.extend(available_behavioral)

    # 6. Lagged delinquency indicators (the Breeden-Crook innovation)
    if include_delinq and horizon > 0:
        delinq_cols = get_delinq_lag_cols(horizon)
        available_delinq = [c for c in delinq_cols if c in panel.columns]
        if available_delinq:
            feature_dfs.append(panel[available_delinq].reset_index(drop=True))
            feature_names.extend(available_delinq)

    X = pd.concat(feature_dfs, axis=1)
    X.index = panel.index

    return X, age_transformer, vintage_categories, feature_names


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class BreedenCrookMultihorizon:
    """
    Breeden & Crook (2022) Multihorizon Discrete-Time Survival Model
    for competing risks.

    Fits separate logistic regressions for each forecast horizon L=1,...,max_horizon,
    where model L uses delinquency lagged by L months. Also fits an origination model
    (no delinquency features) for loans < 6 months old.

    Parameters
    ----------
    max_horizon : int
        Maximum forecast horizon (default 12).
    C_values : list of float
        Regularization strengths to tune over (default [0.01, 0.1, 1.0, 10.0]).
    solver : str
        Logistic regression solver (default 'lbfgs').
    max_iter : int
        Maximum iterations for logistic regression.
    n_age_knots : int
        Number of knots for age spline basis.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        max_horizon: int = 12,
        C_values: Optional[List[float]] = None,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        n_age_knots: int = 5,
        seed: int = 42,
    ):
        self.max_horizon = max_horizon
        self.C_values = C_values or [0.01, 0.1, 1.0, 10.0]
        self.solver = solver
        self.max_iter = max_iter
        self.n_age_knots = n_age_knots
        self.seed = seed

        # Fitted components
        self.models_default: Dict[int, LogisticRegression] = {}
        self.models_prepay: Dict[int, LogisticRegression] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.age_transformer: Optional[SplineTransformer] = None
        self.vintage_categories: Optional[np.ndarray] = None
        self.feature_names: Dict[int, List[str]] = {}
        self.best_C: Dict[str, Dict[int, float]] = {'default': {}, 'prepay': {}}
        self.training_metrics: Dict[str, Dict[int, dict]] = {
            'default': {}, 'prepay': {}
        }

    def _select_C(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Select best regularization C by validation AUC."""
        best_C = self.C_values[0]
        best_auc = 0.0

        for C in self.C_values:
            model = LogisticRegression(
                C=C,
                class_weight='balanced',
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)[:, 1]
            try:
                auc = roc_auc_score(y_val, proba_val)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_C = C

        return best_C

    def fit(
        self,
        panel_train: pd.DataFrame,
        panel_val: pd.DataFrame,
        C: Optional[float] = None,
        log_fn=print,
    ) -> 'BreedenCrookMultihorizon':
        """
        Fit all multihorizon models for both risks.

        Parameters
        ----------
        panel_train : pd.DataFrame
            Training panel with delinquency indicators and lags.
        panel_val : pd.DataFrame
            Validation panel for C selection.
        C : float, optional
            If provided, skip C tuning and use this value for all models.
        log_fn : callable
            Logging function.

        Returns
        -------
        self
        """
        # Fit origination model (horizon=0, no delinquency)
        log_fn("\nFitting origination model (no delinquency features)...")
        self._fit_horizon(
            panel_train, panel_val, horizon=0,
            include_delinq=False, C_override=C, log_fn=log_fn,
        )

        # Fit horizon-specific models
        for L in range(1, self.max_horizon + 1):
            log_fn(f"\nFitting horizon L={L} models...")
            self._fit_horizon(
                panel_train, panel_val, horizon=L,
                include_delinq=True, C_override=C, log_fn=log_fn,
            )

        return self

    def _fit_horizon(
        self,
        panel_train: pd.DataFrame,
        panel_val: pd.DataFrame,
        horizon: int,
        include_delinq: bool = True,
        C_override: Optional[float] = None,
        log_fn=print,
    ):
        """Fit default and prepay models for a single horizon."""
        # Filter: need enough history for lags
        if horizon > 0:
            train_mask = panel_train['loan_age'] > horizon
            val_mask = panel_val['loan_age'] > horizon
            train_data = panel_train[train_mask].copy()
            val_data = panel_val[val_mask].copy()
        else:
            train_data = panel_train.copy()
            val_data = panel_val.copy()

        # Build features
        X_train_df, self.age_transformer, self.vintage_categories, feat_names = \
            build_feature_matrix(
                train_data, horizon,
                age_transformer=self.age_transformer,
                vintage_categories=self.vintage_categories,
                include_delinq=include_delinq,
                n_age_knots=self.n_age_knots,
            )

        X_val_df, _, _, _ = build_feature_matrix(
            val_data, horizon,
            age_transformer=self.age_transformer,
            vintage_categories=self.vintage_categories,
            include_delinq=include_delinq,
            n_age_knots=self.n_age_knots,
        )

        self.feature_names[horizon] = feat_names

        # Drop rows with NaN
        valid_train = X_train_df.notna().all(axis=1)
        valid_val = X_val_df.notna().all(axis=1)
        X_train_df = X_train_df[valid_train]
        train_data = train_data[valid_train]
        X_val_df = X_val_df[valid_val]
        val_data = val_data[valid_val]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df.values.astype(float))
        X_val = scaler.transform(X_val_df.values.astype(float))
        self.scalers[horizon] = scaler

        log_fn(f"  Horizon {horizon}: {len(X_train):,} train, "
               f"{len(X_val):,} val, {len(feat_names)} features")

        # Fit for each risk
        for risk_name, event_code, model_dict in [
            ('default', 2, self.models_default),
            ('prepay', 1, self.models_prepay),
        ]:
            y_train = (train_data['event_code'] == event_code).astype(int).values
            y_val = (val_data['event_code'] == event_code).astype(int).values

            # Select C
            if C_override is not None:
                best_C = C_override
            else:
                best_C = self._select_C(X_train, y_train, X_val, y_val)
            self.best_C[risk_name][horizon] = best_C

            # Fit final model
            model = LogisticRegression(
                C=best_C,
                class_weight='balanced',
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            model_dict[horizon] = model

            # Compute metrics
            proba_train = model.predict_proba(X_train)[:, 1]
            proba_val = model.predict_proba(X_val)[:, 1]

            try:
                auc_train = roc_auc_score(y_train, proba_train)
                auc_val = roc_auc_score(y_val, proba_val)
            except ValueError:
                auc_train = auc_val = np.nan

            try:
                ll_train = log_loss(y_train, proba_train)
                ll_val = log_loss(y_val, proba_val)
            except ValueError:
                ll_train = ll_val = np.nan

            self.training_metrics[risk_name][horizon] = {
                'C': best_C,
                'auc_train': auc_train,
                'auc_val': auc_val,
                'log_loss_train': ll_train,
                'log_loss_val': ll_val,
                'n_train': len(y_train),
                'n_events': int(y_train.sum()),
            }

            log_fn(f"    {risk_name}: C={best_C}, AUC(val)={auc_val:.4f}, "
                   f"events={y_train.sum():,}/{len(y_train):,}")

    def predict_cif(
        self,
        panel_origin: pd.DataFrame,
        max_months: int = 72,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate CIF predictions from forecast origin observations.

        For each test loan, uses its terminal observation as the forecast
        origin at time t0. All horizon-L models use delinquency at t0
        (since model L was trained with lag L, predicting at t0+L means
        delinquency at (t0+L)-L = t0).

        Parameters
        ----------
        panel_origin : pd.DataFrame
            Terminal (forecast-origin) observations. Must include
            delinq_status (or D_*m indicators), loan_age, vintage_year,
            and all static/behavioral/macro features.
        max_months : int
            Maximum forecast horizon in months.

        Returns
        -------
        CIF_default : np.ndarray, shape (n_loans, max_months+1)
            Cumulative incidence for default at each month.
        CIF_prepay : np.ndarray, shape (n_loans, max_months+1)
            Cumulative incidence for prepayment at each month.
        S : np.ndarray, shape (n_loans, max_months+1)
            Survival function at each month.
        """
        n = len(panel_origin)

        CIF_def = np.zeros((n, max_months + 1))
        CIF_pre = np.zeros((n, max_months + 1))
        S = np.ones((n, max_months + 1))

        # Precompute delinquency indicators at forecast origin
        origin = panel_origin.copy()
        if 'D_1m' not in origin.columns:
            origin = create_delinquency_indicators(origin)

        origin_age = origin['loan_age'].values.astype(float)

        for l in range(1, max_months + 1):
            model_l = min(l, self.max_horizon)

            # Use origination model for loans with insufficient history
            if model_l == 0 or (origin_age + l).min() <= model_l:
                # Separate loans needing origination vs horizon model
                use_orig = (origin_age + l) <= model_l
            else:
                use_orig = np.zeros(n, dtype=bool)

            # Build features for this horizon
            # Key insight: delinquency at forecast origin t0 is used,
            # because model L was trained with lag L
            forecast_data = origin.copy()
            forecast_data['loan_age'] = origin_age + l  # Age at prediction time

            # Set lagged delinquency = current delinquency at origin
            for d_col in DELINQ_INDICATOR_COLS:
                forecast_data[f'{d_col}_lag{model_l}'] = origin[d_col].values

            h_def = np.zeros(n)
            h_pre = np.zeros(n)

            # Horizon model predictions
            horizon_mask = ~use_orig
            if horizon_mask.any():
                h_def_h, h_pre_h = self._predict_hazard_at_horizon(
                    forecast_data[horizon_mask], model_l, include_delinq=True
                )
                h_def[horizon_mask] = h_def_h
                h_pre[horizon_mask] = h_pre_h

            # Origination model predictions
            if use_orig.any():
                h_def_o, h_pre_o = self._predict_hazard_at_horizon(
                    forecast_data[use_orig], 0, include_delinq=False
                )
                h_def[use_orig] = h_def_o
                h_pre[use_orig] = h_pre_o

            # Clip to ensure valid probabilities
            h_total = h_def + h_pre
            excess = np.maximum(h_total - 1.0, 0.0)
            if excess.any():
                scale = np.where(h_total > 1.0, 1.0 / h_total, 1.0)
                h_def = h_def * scale
                h_pre = h_pre * scale

            # Chain hazards into CIF
            S[:, l] = S[:, l - 1] * (1 - h_def - h_pre)
            CIF_def[:, l] = CIF_def[:, l - 1] + S[:, l - 1] * h_def
            CIF_pre[:, l] = CIF_pre[:, l - 1] + S[:, l - 1] * h_pre

        return CIF_def, CIF_pre, S

    def _predict_hazard_at_horizon(
        self,
        data: pd.DataFrame,
        horizon: int,
        include_delinq: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict default and prepay hazards for a given horizon."""
        X_df, _, _, _ = build_feature_matrix(
            data, horizon,
            age_transformer=self.age_transformer,
            vintage_categories=self.vintage_categories,
            include_delinq=include_delinq,
            n_age_knots=self.n_age_knots,
        )

        # Handle NaN by filling with 0
        X = X_df.fillna(0).values.astype(float)

        scaler = self.scalers.get(horizon)
        if scaler is not None:
            X = scaler.transform(X)

        model_def = self.models_default[horizon]
        model_pre = self.models_prepay[horizon]

        h_def = model_def.predict_proba(X)[:, 1]
        h_pre = model_pre.predict_proba(X)[:, 1]

        return h_def, h_pre

    def get_coefficients(self) -> pd.DataFrame:
        """
        Extract coefficients across all horizons for coefficient analysis.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            horizon, risk, feature, coefficient.
        """
        records = []
        for risk_name, model_dict in [
            ('default', self.models_default),
            ('prepay', self.models_prepay),
        ]:
            for horizon, model in model_dict.items():
                feat_names = self.feature_names[horizon]
                coefs = model.coef_[0]
                for fname, coef in zip(feat_names, coefs):
                    records.append({
                        'horizon': horizon,
                        'risk': risk_name,
                        'feature': fname,
                        'coefficient': coef,
                    })

        return pd.DataFrame(records)

    def get_pseudo_r2(self) -> pd.DataFrame:
        """
        Compute McFadden's pseudo-R2 and Gini (2*AUC-1) by horizon.

        Returns
        -------
        pd.DataFrame
            DataFrame with horizon, risk, pseudo_r2, gini, auc columns.
        """
        records = []
        for risk_name in ['default', 'prepay']:
            for horizon, metrics in self.training_metrics[risk_name].items():
                auc = metrics['auc_val']
                gini = 2 * auc - 1 if not np.isnan(auc) else np.nan
                records.append({
                    'horizon': horizon,
                    'risk': risk_name,
                    'auc_val': auc,
                    'gini': gini,
                    'log_loss_val': metrics['log_loss_val'],
                    'n_events': metrics['n_events'],
                    'n_train': metrics['n_train'],
                })

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting functions for coefficient analysis
# ---------------------------------------------------------------------------

def plot_delinquency_coefficients(
    coef_df: pd.DataFrame,
    risk: str = 'default',
    ax=None,
):
    """
    Plot delinquency coefficients vs. forecast horizon (paper's Fig. 5).

    Shows how delinquency's predictive power decays with horizon.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    delinq_features = DELINQ_INDICATOR_COLS
    risk_df = coef_df[coef_df['risk'] == risk]

    for d_feat in delinq_features:
        data = []
        for h in range(1, 13):
            lag_feat = f'{d_feat}_lag{h}'
            row = risk_df[
                (risk_df['horizon'] == h) & (risk_df['feature'] == lag_feat)
            ]
            if not row.empty:
                data.append((h, row['coefficient'].values[0]))

        if data:
            horizons, coefs = zip(*data)
            label = d_feat.replace('D_', '').replace('_plus', '+').replace('m', '-month')
            ax.plot(horizons, coefs, 'o-', label=label, linewidth=2, markersize=6)

    ax.set_xlabel('Forecast Horizon L (months)')
    ax.set_ylabel('Coefficient')
    ax.set_title(f'Delinquency Coefficients vs. Horizon ({risk.title()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))

    return ax


def plot_origination_coefficients(
    coef_df: pd.DataFrame,
    risk: str = 'default',
    features: Optional[List[str]] = None,
    ax=None,
):
    """
    Plot origination variable coefficients vs. horizon (paper's Fig. 6).

    Shows how origination variables' predictive power increases with horizon.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if features is None:
        features = ['fico_score', 'dti_r', 'ltv_r', 'int_rate', 'log_orig_upb']

    risk_df = coef_df[coef_df['risk'] == risk]

    for feat in features:
        data = []
        for h in range(0, 13):
            row = risk_df[
                (risk_df['horizon'] == h) & (risk_df['feature'] == feat)
            ]
            if not row.empty:
                data.append((h, row['coefficient'].values[0]))

        if data:
            horizons, coefs = zip(*data)
            ax.plot(horizons, coefs, 'o-', label=feat, linewidth=2, markersize=6)

    ax.set_xlabel('Forecast Horizon L (months)')
    ax.set_ylabel('Coefficient')
    ax.set_title(f'Origination Variable Coefficients vs. Horizon ({risk.title()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 13))

    return ax


def plot_pseudo_r2_by_horizon(
    r2_df: pd.DataFrame,
    ax=None,
):
    """
    Plot Gini and AUC by horizon (paper's Figs. 8-9).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for risk in ['default', 'prepay']:
        data = r2_df[r2_df['risk'] == risk].sort_values('horizon')
        # Exclude origination model for cleaner plot
        data = data[data['horizon'] > 0]
        ax.plot(data['horizon'], data['gini'], 'o-',
                label=f'{risk.title()} (Gini)', linewidth=2, markersize=6)

    ax.set_xlabel('Forecast Horizon L (months)')
    ax.set_ylabel('Gini Coefficient (2*AUC - 1)')
    ax.set_title('Model Discriminatory Power by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 13))

    return ax
