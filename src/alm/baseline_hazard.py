"""
Extract and prepare baseline hazards from CoxTimeVaryingFitter models.

The baseline cumulative hazard H_0(t) is stored in the fitted model.
We convert it to discrete baseline hazard h_0(t) and reindex to a
complete monthly grid for projection purposes.
"""

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter


def extract_baseline_hazard(
    model: CoxTimeVaryingFitter,
    max_month: int = 360,
    extrapolation: str = "flat",
) -> np.ndarray:
    """
    Extract discrete baseline hazard h_0(t) from a fitted CoxTimeVaryingFitter.

    Parameters
    ----------
    model : CoxTimeVaryingFitter
        Fitted lifelines model with baseline_cumulative_hazard_ attribute.
    max_month : int
        Maximum month to extend the hazard grid to (e.g. 360 for 30-year).
    extrapolation : str
        How to handle months beyond observed data.
        'flat' = hold last observed h_0 constant.
        'zero' = set to 0 beyond observed range.

    Returns
    -------
    np.ndarray
        Discrete baseline hazard h_0(t) for t = 1..max_month (length max_month).
        Index position i corresponds to month i+1.
    """
    # Extract cumulative hazard H_0(t)
    cum_haz = model.baseline_cumulative_hazard_
    col = cum_haz.columns[0]  # "baseline hazard"
    H0 = cum_haz[col]

    # Reindex to complete grid [1..max_observed] filling gaps with forward fill
    max_observed = int(H0.index.max())
    full_index = np.arange(1, max_observed + 1)
    H0_full = H0.reindex(full_index).ffill().fillna(0.0)

    # Convert cumulative to discrete: h_0(t) = H_0(t) - H_0(t-1)
    H0_vals = H0_full.values
    h0 = np.zeros(max_observed)
    h0[0] = H0_vals[0]  # h_0(1) = H_0(1)
    h0[1:] = np.diff(H0_vals)

    # Ensure non-negative (numerical precision)
    h0 = np.maximum(h0, 0.0)

    # Extend to max_month
    if max_observed < max_month:
        extension = np.zeros(max_month - max_observed)
        if extrapolation == "flat":
            # Use last observed discrete hazard
            last_h0 = h0[max_observed - 1] if max_observed > 0 else 0.0
            extension[:] = last_h0
        # 'zero' leaves extension as zeros
        h0 = np.concatenate([h0, extension])
    else:
        h0 = h0[:max_month]

    return h0


def extract_baseline_hazards_both(
    model_prepay: CoxTimeVaryingFitter,
    model_default: CoxTimeVaryingFitter,
    max_month: int = 360,
    extrapolation: str = "flat",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract baseline hazards for both prepayment and default models.

    Parameters
    ----------
    model_prepay : CoxTimeVaryingFitter
        Fitted prepayment Cox model.
    model_default : CoxTimeVaryingFitter
        Fitted default Cox model.
    max_month : int
        Maximum month for the hazard grid.
    extrapolation : str
        Extrapolation method ('flat' or 'zero').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (h0_prepay, h0_default) each of length max_month.
    """
    h0_prepay = extract_baseline_hazard(model_prepay, max_month, extrapolation)
    h0_default = extract_baseline_hazard(model_default, max_month, extrapolation)
    return h0_prepay, h0_default
