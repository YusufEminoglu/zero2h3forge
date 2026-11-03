# -*- coding: utf-8 -*-
"""Criterion Normalization Engine (0-100 Benefit/Cost & Fuzzy Membership) for 02H3Forge.

Parallels the normalization formulas of ``planx_suitability_lab`` while operating
directly on H3 hexagonal vector attributes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .h3_grid_engine import ensure_layer_field, sanitize_field_slug


NORMALIZATION_METHODS: List[Tuple[str, str]] = [
    ("minmax", "Min-Max Linear (Observed Min..Max)"),
    ("percentile", "Robust Percentile Linear (P2..P98 Outlier-Safe)"),
    ("fuzzy_linear", "Fuzzy Linear Breakpoints [Low..High]"),
    ("fuzzy_sigmoid", "Fuzzy Sigmoid Curve (Midpoint & Spread)"),
    ("fuzzy_gaussian", "Fuzzy Gaussian Bell (Optimum & Spread)"),
    ("direct_clip", "Direct 0-100 Score (Pre-Scored Multi-Ring)"),
]

POLARITY_OPTIONS: List[Tuple[str, str]] = [
    ("benefit", "Benefit (+) : Higher Raw Value → Higher Score (100)"),
    ("cost", "Cost (-)    : Lower Raw Value → Higher Score (100)"),
]


def parse_polarity(polarity: str | int) -> str:
    """Normalize polarity input to ``'benefit'`` or ``'cost'``."""
    if isinstance(polarity, (int, float)):
        return "benefit" if polarity >= 0 else "cost"
    token = str(polarity or "benefit").strip().lower()
    if token.startswith("cost") or token in ("-", "-1", "c", "inverse"):
        return "cost"
    return "benefit"


def normalize_array(
    values: Sequence[float] | np.ndarray,
    polarity: str | int = "benefit",
    method: str = "minmax",
    low: Optional[float] = None,
    high: Optional[float] = None,
    mid: Optional[float] = None,
    spread: Optional[float] = None,
) -> np.ndarray:
    """Transform a 1D numeric array into a [0, 100] standardized score array.

    Parameters
    ----------
    values : array-like of float
        Raw attribute values across H3 hexagons.
    polarity : ``'benefit'`` or ``'cost'``
        ``'benefit'`` means higher raw values map toward 100;
        ``'cost'`` means lower raw values map toward 100.
    method : str
        One of ``'minmax'``, ``'percentile'``, ``'fuzzy_linear'``,
        ``'fuzzy_sigmoid'``, ``'fuzzy_gaussian'``, ``'direct_clip'``.
    low, high : float, optional
        Lower and upper breakpoints for ``'fuzzy_linear'`` (or overrides for ``'minmax'``).
    mid, spread : float, optional
        Center and steepness/width (>0) for ``'fuzzy_sigmoid'`` and ``'fuzzy_gaussian'``.
    """
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(valid):
        return out

    pol = parse_polarity(polarity)
    m = (method or "minmax").strip().lower()
    x_valid = arr[valid]

    if m == "direct_clip":
        unit = np.clip(x_valid, 0.0, 100.0) / 100.0
        if pol == "cost":
            unit = 1.0 - unit
        out[valid] = np.clip(unit * 100.0, 0.0, 100.0)
        return out

    if m == "percentile":
        lo = float(np.percentile(x_valid, 2.0))
        hi = float(np.percentile(x_valid, 98.0))
        if hi <= lo:
            lo = float(np.min(x_valid))
            hi = float(np.max(x_valid))
    elif m == "fuzzy_linear" and low is not None and high is not None:
        lo = float(low)
        hi = float(high)
        if hi <= lo:
            raise ValueError("HIGH breakpoint must be strictly greater than LOW breakpoint.")
    else:
        lo = float(low) if low is not None else float(np.min(x_valid))
        hi = float(high) if high is not None else float(np.max(x_valid))

    if m in ("minmax", "percentile", "fuzzy_linear"):
        span = hi - lo
        if span <= 1e-12:
            # Constant non-zero field gets 50.0 (or 0.0 if all zeros)
            const_score = 50.0 if np.any(x_valid != 0.0) else 0.0
            out[valid] = const_score
            return out
        if pol == "benefit":
            unit = (x_valid - lo) / span
        else:
            unit = (hi - x_valid) / span
        out[valid] = np.clip(unit, 0.0, 1.0) * 100.0
        return out

    # Sigmoid or Gaussian
    c_mid = float(mid) if mid is not None else float(np.median(x_valid))
    if spread is not None:
        c_spread = float(spread)
    else:
        std_v = float(np.std(x_valid))
        c_spread = std_v if std_v > 1e-6 else max(1.0, 0.25 * (hi - lo))
    if c_spread <= 0.0:
        raise ValueError("SPREAD parameter must be strictly positive (> 0).")

    if m == "fuzzy_sigmoid":
        z = np.clip((x_valid - c_mid) / c_spread, -60.0, 60.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        unit = sig if pol == "benefit" else (1.0 - sig)
        out[valid] = np.clip(unit, 0.0, 1.0) * 100.0
        return out

    if m == "fuzzy_gaussian":
        gauss = np.exp(-0.5 * ((x_valid - c_mid) / c_spread) ** 2)
        unit = gauss if pol == "benefit" else (1.0 - gauss)
        out[valid] = np.clip(unit, 0.0, 1.0) * 100.0
        return out

    # Fallback to linear min-max
    span = max(1e-12, hi - lo)
    unit = (x_valid - lo) / span if pol == "benefit" else (hi - x_valid) / span
    out[valid] = np.clip(unit, 0.0, 1.0) * 100.0
    return out


def normalize_h3_field(
    h3_layer,
    source_field: str,
    polarity: str = "benefit",
    method: str = "minmax",
    low: Optional[float] = None,
    high: Optional[float] = None,
    mid: Optional[float] = None,
    spread: Optional[float] = None,
    output_field: Optional[str] = None,
) -> str:
    """Normalize ``source_field`` on ``h3_layer`` into ``norm_<slug>`` in [0, 100]."""
    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    src_idx = h3_layer.fields().indexFromName(source_field)
    if src_idx < 0:
        raise ValueError(f"Field '{source_field}' not found on H3 layer.")

    norm_col = output_field or sanitize_field_slug(source_field, prefix="norm")
    out_idx = ensure_layer_field(h3_layer, norm_col, kind="double")

    fids: List[int] = []
    raw_vals: List[float] = []
    for feat in h3_layer.getFeatures():
        fids.append(int(feat.id()))
        v = feat.attribute(src_idx)
        try:
            fv = float(v) if v is not None else float("nan")
        except (ValueError, TypeError):
            fv = float("nan")
        raw_vals.append(fv)

    norm_arr = normalize_array(
        raw_vals,
        polarity=polarity,
        method=method,
        low=low,
        high=high,
        mid=mid,
        spread=spread,
    )

    updates: Dict[int, Dict[int, object]] = {}
    for i, fid in enumerate(fids):
        updates[fid] = {out_idx: round(float(norm_arr[i]), 2)}
    if updates:
        h3_layer.dataProvider().changeAttributeValues(updates)
    return norm_col
