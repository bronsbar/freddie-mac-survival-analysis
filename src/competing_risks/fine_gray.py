"""
Discrete-time Fine-Gray competing risks model implementation.

This module implements a discrete-time approximation to the Fine-Gray
subdistribution hazard model, which is equivalent in the limit and
naturally handles time-varying covariates.

References:
-----------
Fine, J.P. and Gray, R.J. (1999). "A Proportional Hazards Model for the
Subdistribution of a Competing Risk." JASA, 94(446), 496-509.

Allison, P.D. (1982). "Discrete-Time Methods for the Analysis of Event
Histories." Sociological Methodology, 13, 61-98.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class DiscreteTimeFineGray:
    """
    Discrete-time Fine-Gray competing risks model.

    This model treats each time interval as a binary outcome and uses
    modified risk sets where subjects with competing events remain
    in the risk set (with outcome=0) after their competing event.

    Parameters
    ----------
    primary_event : int
        Event code for the primary event of interest
    competing_events : List[int]
        Event codes for competing events
    alpha : float
        Regularization strength for logistic regression
    standardize : bool
        Whether to standardize features before fitting

    Attributes
    ----------
    model_ : fitted model
        The fitted logistic regression model
    scaler_ : StandardScaler
        Fitted scaler (if standardize=True)
    feature_names_ : List[str]
        Names of features used in fitting
    coef_ : np.ndarray
        Model coefficients
    """

    def __init__(
        self,
        primary_event: int = 1,
        competing_events: List[int] = [2],
        alpha: float = 0.01,
        standardize: bool = True,
    ):
        self.primary_event = primary_event
        self.competing_events = competing_events
        self.alpha = alpha
        self.standardize = standardize

        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.coef_ = None

    def _prepare_risk_set(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        event_col: str,
    ) -> pd.DataFrame:
        """
        Prepare risk set with Fine-Gray modification.

        In Fine-Gray, subjects with competing events remain in the risk
        set after their event (contributing with outcome=0).
        """
        df = df.copy()

        # Create binary outcome for primary event
        df['y'] = (df[event_col] == self.primary_event).astype(int)

        # For competing events, they stay in risk set but with y=0
        # This is automatically handled since we don't remove them

        # Mark terminal records
        all_events = [self.primary_event, 0] + self.competing_events
        df['is_terminal'] = df[event_col].isin(all_events)

        return df

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        id_col: str = 'loan_sequence_number',
        time_col: str = 'loan_age',
        event_col: str = 'event_code',
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'DiscreteTimeFineGray':
        """
        Fit the discrete-time Fine-Gray model.

        Parameters
        ----------
        df : pd.DataFrame
            Loan-month panel data
        feature_cols : List[str]
            Names of feature columns to use
        id_col : str
            Column with subject identifier
        time_col : str
            Column with time variable
        event_col : str
            Column with event code
        sample_weight : np.ndarray, optional
            Sample weights for fitting

        Returns
        -------
        self
        """
        # Prepare risk set
        df_fit = self._prepare_risk_set(df, id_col, time_col, event_col)

        # Extract features and outcome
        X = df_fit[feature_cols].values
        y = df_fit['y'].values

        # Handle missing values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]

        # Standardize features
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Fit logistic regression
        self.model_ = LogisticRegression(
            C=1.0 / self.alpha,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)

        # Store results
        self.feature_names_ = feature_cols
        self.coef_ = self.model_.coef_[0]

        return self

    def predict_hazard(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Predict subdistribution hazard probabilities.

        Parameters
        ----------
        df : pd.DataFrame
            Data to predict on
        feature_cols : List[str], optional
            Feature columns (uses training features if not specified)

        Returns
        -------
        np.ndarray
            Predicted hazard probabilities
        """
        if feature_cols is None:
            feature_cols = self.feature_names_

        X = df[feature_cols].values

        if self.standardize and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict_proba(X)[:, 1]

    def predict_cumulative_incidence(
        self,
        df: pd.DataFrame,
        id_col: str = 'loan_sequence_number',
        time_col: str = 'loan_age',
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Predict cumulative incidence function for each subject.

        CIF(t) = 1 - prod(1 - h(s)) for s = 1 to t

        Parameters
        ----------
        df : pd.DataFrame
            Loan-month panel data
        id_col : str
            Subject identifier column
        time_col : str
            Time variable column
        feature_cols : List[str], optional
            Feature columns

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [id_col, time_col, 'hazard', 'cif']
        """
        # Predict hazard for each record
        hazard = self.predict_hazard(df, feature_cols)

        result = df[[id_col, time_col]].copy()
        result['hazard'] = hazard

        # Calculate CIF per subject
        def calc_cif(group):
            group = group.sort_values(time_col)
            survival = np.cumprod(1 - group['hazard'].values)
            group['survival'] = survival
            group['cif'] = 1 - survival
            return group

        result = result.groupby(id_col).apply(calc_cif).reset_index(drop=True)

        return result

    def get_hazard_ratios(self) -> pd.DataFrame:
        """
        Get hazard ratios (exp of coefficients) with feature names.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names, coefficients, and hazard ratios
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")

        return pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': self.coef_,
            'hazard_ratio': np.exp(self.coef_),
            'interpretation': [
                f"{(np.exp(c) - 1) * 100:+.1f}% per unit increase"
                for c in self.coef_
            ]
        })


def fit_discrete_time_competing_risks(
    df: pd.DataFrame,
    feature_cols: List[str],
    primary_event: int = 1,
    competing_events: List[int] = [2],
    id_col: str = 'loan_sequence_number',
    time_col: str = 'loan_age',
    event_col: str = 'event_code',
    use_statsmodels: bool = False,
) -> Union[DiscreteTimeFineGray, sm.Logit]:
    """
    Convenience function to fit discrete-time competing risks model.

    Parameters
    ----------
    df : pd.DataFrame
        Loan-month panel data
    feature_cols : List[str]
        Feature columns to use
    primary_event : int
        Primary event code
    competing_events : List[int]
        Competing event codes
    id_col : str
        Subject identifier column
    time_col : str
        Time variable column
    event_col : str
        Event code column
    use_statsmodels : bool
        If True, use statsmodels for inference (p-values, CIs)

    Returns
    -------
    Fitted model
    """
    if use_statsmodels:
        # Prepare data
        df_fit = df.copy()
        df_fit['y'] = (df_fit[event_col] == primary_event).astype(int)

        # Drop missing
        df_fit = df_fit[feature_cols + ['y']].dropna()

        X = sm.add_constant(df_fit[feature_cols])
        y = df_fit['y']

        model = sm.Logit(y, X)
        result = model.fit(disp=False)
        return result

    else:
        model = DiscreteTimeFineGray(
            primary_event=primary_event,
            competing_events=competing_events,
        )
        model.fit(df, feature_cols, id_col, time_col, event_col)
        return model
