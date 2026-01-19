"""
Cause-specific Cox proportional hazards models for competing risks.

In cause-specific analysis, competing events are treated as censored.
This gives the hazard of the event among those still at risk, which
differs from the Fine-Gray subdistribution hazard.

Use cause-specific models when interested in:
- Understanding covariate effects on event intensity
- Etiological questions about risk factors
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler


class CauseSpecificCox:
    """
    Cause-specific Cox proportional hazards model wrapper.

    This class provides a unified interface for fitting cause-specific
    Cox models using either lifelines or scikit-survival.

    Parameters
    ----------
    event_of_interest : int
        Event code for the event to model
    penalizer : float
        L2 regularization penalty
    backend : str
        'lifelines' or 'sksurv'

    Attributes
    ----------
    model_ : fitted model
        The fitted Cox model
    feature_names_ : List[str]
        Names of features used in fitting
    """

    def __init__(
        self,
        event_of_interest: int = 1,
        penalizer: float = 0.01,
        backend: str = 'lifelines',
    ):
        self.event_of_interest = event_of_interest
        self.penalizer = penalizer
        self.backend = backend

        self.model_ = None
        self.feature_names_ = None
        self.scaler_ = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        duration_col: str = 'duration',
        event_col: str = 'event_code',
        standardize: bool = True,
    ) -> 'CauseSpecificCox':
        """
        Fit cause-specific Cox model.

        Parameters
        ----------
        df : pd.DataFrame
            Data with one row per subject (terminal record)
        feature_cols : List[str]
            Feature columns to use
        duration_col : str
            Column with survival time
        event_col : str
            Column with event code
        standardize : bool
            Whether to standardize features

        Returns
        -------
        self
        """
        # Create cause-specific event indicator
        df_fit = df.copy()
        df_fit['event_cs'] = (df_fit[event_col] == self.event_of_interest).astype(int)

        # Prepare features
        self.feature_names_ = feature_cols

        if self.backend == 'lifelines':
            # Lifelines expects all columns in one DataFrame
            cols_to_use = feature_cols + [duration_col, 'event_cs']
            df_model = df_fit[cols_to_use].dropna()

            self.model_ = CoxPHFitter(penalizer=self.penalizer)
            self.model_.fit(
                df_model,
                duration_col=duration_col,
                event_col='event_cs'
            )

        elif self.backend == 'sksurv':
            # scikit-survival needs structured array for y
            df_fit = df_fit[feature_cols + [duration_col, 'event_cs']].dropna()

            X = df_fit[feature_cols].values

            if standardize:
                self.scaler_ = StandardScaler()
                X = self.scaler_.fit_transform(X)

            y = np.array(
                list(zip(
                    df_fit['event_cs'].astype(bool),
                    df_fit[duration_col].astype(float)
                )),
                dtype=[('event', bool), ('time', float)]
            )

            self.model_ = CoxPHSurvivalAnalysis(alpha=self.penalizer)
            self.model_.fit(X, y)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self

    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict relative hazard (risk score).

        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on

        Returns
        -------
        np.ndarray
            Predicted risk scores
        """
        X = df[self.feature_names_]

        if self.backend == 'lifelines':
            # Lifelines partial hazard
            return self.model_.predict_partial_hazard(X).values

        elif self.backend == 'sksurv':
            X_vals = X.values
            if self.scaler_ is not None:
                X_vals = self.scaler_.transform(X_vals)
            return self.model_.predict(X_vals)

    def predict_survival_function(
        self,
        df: pd.DataFrame,
        times: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Predict survival function.

        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        times : np.ndarray, optional
            Time points for prediction

        Returns
        -------
        pd.DataFrame
            Survival probabilities at each time point
        """
        if self.backend == 'lifelines':
            return self.model_.predict_survival_function(
                df[self.feature_names_],
                times=times
            )
        elif self.backend == 'sksurv':
            X = df[self.feature_names_].values
            if self.scaler_ is not None:
                X = self.scaler_.transform(X)
            surv_funcs = self.model_.predict_survival_function(X)
            # Convert to DataFrame
            if times is None:
                times = surv_funcs[0].x
            return pd.DataFrame(
                {i: fn(times) for i, fn in enumerate(surv_funcs)},
                index=times
            )

    def get_summary(self) -> pd.DataFrame:
        """
        Get coefficient summary with hazard ratios.

        Returns
        -------
        pd.DataFrame
            Summary table with coefficients, HRs, and p-values
        """
        if self.backend == 'lifelines':
            return self.model_.summary

        elif self.backend == 'sksurv':
            return pd.DataFrame({
                'feature': self.feature_names_,
                'coef': self.model_.coef_,
                'exp(coef)': np.exp(self.model_.coef_),
            })

    def plot_coefficients(self, ax=None):
        """Plot hazard ratios with confidence intervals."""
        if self.backend == 'lifelines':
            self.model_.plot(ax=ax)
        else:
            import matplotlib.pyplot as plt
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))

            coefs = self.model_.coef_
            hrs = np.exp(coefs)

            colors = ['green' if c < 0 else 'red' for c in coefs]
            ax.barh(self.feature_names_, hrs, color=colors, alpha=0.7)
            ax.axvline(x=1, color='black', linestyle='--')
            ax.set_xlabel('Hazard Ratio')
            ax.set_title('Cause-Specific Cox Model Hazard Ratios')

            return ax


def fit_cause_specific_cox(
    df: pd.DataFrame,
    event_of_interest: int,
    feature_cols: List[str],
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    penalizer: float = 0.01,
) -> CoxPHFitter:
    """
    Convenience function to fit cause-specific Cox model using lifelines.

    Parameters
    ----------
    df : pd.DataFrame
        Data with one row per subject
    event_of_interest : int
        Event code to model
    feature_cols : List[str]
        Feature columns
    duration_col : str
        Duration column
    event_col : str
        Event code column
    penalizer : float
        L2 regularization

    Returns
    -------
    CoxPHFitter
        Fitted lifelines model
    """
    # Prepare data
    df_fit = df.copy()
    df_fit['event_cs'] = (df_fit[event_col] == event_of_interest).astype(int)

    cols = feature_cols + [duration_col, 'event_cs']
    df_model = df_fit[cols].dropna()

    # Fit model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_model, duration_col=duration_col, event_col='event_cs')

    return cph


def fit_both_cause_specific(
    df: pd.DataFrame,
    feature_cols: List[str],
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    penalizer: float = 0.01,
) -> Tuple[CoxPHFitter, CoxPHFitter]:
    """
    Fit cause-specific Cox models for both prepay and default.

    Parameters
    ----------
    df : pd.DataFrame
        Data with one row per subject
    feature_cols : List[str]
        Feature columns
    duration_col : str
        Duration column
    event_col : str
        Event code column
    penalizer : float
        L2 regularization

    Returns
    -------
    Tuple[CoxPHFitter, CoxPHFitter]
        (prepay_model, default_model)
    """
    prepay_model = fit_cause_specific_cox(
        df, event_of_interest=1, feature_cols=feature_cols,
        duration_col=duration_col, event_col=event_col,
        penalizer=penalizer
    )

    default_model = fit_cause_specific_cox(
        df, event_of_interest=2, feature_cols=feature_cols,
        duration_col=duration_col, event_col=event_col,
        penalizer=penalizer
    )

    return prepay_model, default_model
