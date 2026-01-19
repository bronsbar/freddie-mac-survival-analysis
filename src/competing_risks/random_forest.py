"""
Random Survival Forest for Competing Risks.

This module implements RSF for competing risks following Ishwaran et al. (2014).
Uses scikit-survival as the base, with extensions for competing risks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


class CompetingRisksRSF(BaseEstimator):
    """
    Random Survival Forest for Competing Risks.

    Fits separate RSF models for each event type (cause-specific approach)
    and combines predictions for cumulative incidence estimation.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples in a leaf node
    max_features : str or int
        Number of features to consider for splits
    n_jobs : int
        Number of parallel jobs
    random_state : int
        Random seed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.models_: Dict[int, RandomSurvivalForest] = {}
        self.event_types_: List[int] = []
        self.feature_names_: List[str] = []
        self.scaler_: Optional[StandardScaler] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        duration: np.ndarray,
        event_code: np.ndarray,
        event_types: List[int] = [1, 2],
        scale_features: bool = True,
    ) -> "CompetingRisksRSF":
        """
        Fit cause-specific RSF models for each event type.

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
        scale_features : bool
            Whether to standardize features

        Returns
        -------
        self
        """
        self.event_types_ = event_types

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]

        # Scale features
        if scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        duration = np.asarray(duration)
        event_code = np.asarray(event_code)

        # Fit cause-specific model for each event
        for event in event_types:
            print(f"  Fitting RSF for event {event}...")

            # Create cause-specific event indicator
            # Event of interest = True, all others = False (censored)
            event_indicator = (event_code == event)

            # Create structured array for scikit-survival
            y = Surv.from_arrays(event_indicator, duration)

            # Fit RSF
            rsf = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            rsf.fit(X, y)

            self.models_[event] = rsf

        return self

    def predict_risk(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
    ) -> np.ndarray:
        """
        Predict risk scores for a specific event.

        Parameters
        ----------
        X : array-like
            Feature matrix
        event : int
            Event code to predict

        Returns
        -------
        np.ndarray
            Risk scores (higher = more risk)
        """
        if event not in self.models_:
            raise ValueError(f"No model for event {event}. Available: {list(self.models_.keys())}")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        # RSF risk score is the negative of the survival function sum
        # Higher risk = lower survival
        rsf = self.models_[event]

        # Use predict method which returns risk scores
        return rsf.predict(X)

    def predict_survival_function(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict cause-specific survival function.

        Parameters
        ----------
        X : array-like
            Feature matrix
        event : int
            Event code
        times : array-like, optional
            Time points for prediction

        Returns
        -------
        np.ndarray
            Survival probabilities, shape (n_samples, n_times)
        """
        if event not in self.models_:
            raise ValueError(f"No model for event {event}")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        rsf = self.models_[event]
        surv_funcs = rsf.predict_survival_function(X)

        if times is None:
            # Return at all unique event times
            return np.array([fn.y for fn in surv_funcs])
        else:
            # Interpolate at requested times
            return np.array([fn(times) for fn in surv_funcs])

    def predict_cumulative_incidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        event: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Predict cumulative incidence function for a specific event.

        For cause-specific models: CIF(t) = 1 - S(t)
        Note: This is an approximation; true CIF requires all cause-specific hazards.

        Parameters
        ----------
        X : array-like
            Feature matrix
        event : int
            Event code
        times : array-like
            Time points for prediction

        Returns
        -------
        np.ndarray
            Cumulative incidence, shape (n_samples, n_times)
        """
        surv = self.predict_survival_function(X, event, times)
        return 1 - surv

    def get_feature_importance(self, event: int) -> pd.DataFrame:
        """
        Get feature importance scores for a specific event model.

        Parameters
        ----------
        event : int
            Event code

        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if event not in self.models_:
            raise ValueError(f"No model for event {event}")

        rsf = self.models_[event]
        importances = rsf.feature_importances_

        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances,
        }).sort_values('importance', ascending=False)


def fit_rsf_competing_risks(
    df: pd.DataFrame,
    feature_cols: List[str],
    duration_col: str = 'duration',
    event_col: str = 'event_code',
    event_types: List[int] = [1, 2],
    **rsf_params,
) -> CompetingRisksRSF:
    """
    Convenience function to fit competing risks RSF.

    Parameters
    ----------
    df : pd.DataFrame
        Data with features, duration, and event code
    feature_cols : list
        Column names for features
    duration_col : str
        Duration column name
    event_col : str
        Event code column name
    event_types : list
        Event codes to model
    **rsf_params
        Parameters for CompetingRisksRSF

    Returns
    -------
    CompetingRisksRSF
        Fitted model
    """
    X = df[feature_cols]
    duration = df[duration_col].values
    event_code = df[event_col].values

    model = CompetingRisksRSF(**rsf_params)
    model.fit(X, duration, event_code, event_types=event_types)

    return model
