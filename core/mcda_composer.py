# -*- coding: utf-8 -*-
"""Coupled Weight Balancer, Objective/AHP Weighting, MCDA Composition & XAI for 02H3Forge.

Implements:
- Coupled sum-to-100 auto-balancing weights with per-criterion lock flags
- AHP (Saaty + Alonso & Lamata 2006 RI table), CRITIC, Shannon Entropy, and PCA weights
- WLC, OWA, TOPSIS, and VIKOR multi-criteria overlay engines
- H3 1-ring spatial contagion smoothing
- Local Explainable AI (top criterion driver & share %) and Monte Carlo robustness
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .h3_grid_engine import ensure_layer_field, get_h3_neighbors


# Saaty (n=1..15) + Alonso & Lamata (2006) (n=16..20) Random Consistency Index table.
RI_TABLE: Dict[int, float] = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.54,
    13: 1.56,
    14: 1.57,
    15: 1.59,
    16: 1.5978,
    17: 1.6086,
    18: 1.6181,
    19: 1.6265,
    20: 1.6341,
}

COMPOSE_METHODS: List[Tuple[str, str]] = [
    ("wlc", "WLC — Weighted Linear Combination"),
    ("owa_pessimistic", "OWA (Pessimistic / Precautionary Risk)"),
    ("owa_optimistic", "OWA (Optimistic / Opportunity)"),
    ("topsis", "TOPSIS — Ideal / Anti-Ideal Closeness"),
    ("vikor", "VIKOR — Compromise Utility & Max Regret"),
]

CLASS_LABELS_RISK = ("Very Low", "Low", "Moderate", "High", "Critical")
CLASS_LABELS_SUITABILITY = ("Very Low", "Low", "Moderate", "High", "Very High")


# ---------------------------------------------------------------------------
# 1. Coupled Sum-to-100 Weight Auto-Balancer
# ---------------------------------------------------------------------------

def rebalance_coupled_weights(
    weights: Sequence[float],
    changed_index: int,
    new_value: float,
    locked_flags: Optional[Sequence[bool]] = None,
    active_flags: Optional[Sequence[bool]] = None,
) -> List[float]:
    """Rebalance weights so all active weights sum to 100.0 when one slider moves.

    When criterion ``changed_index`` is set to ``new_value``, all other active
    unlocked criteria are scaled proportionally to absorb the delta.
    """
    n = len(weights)
    if n == 0:
        return []

    w = np.array([max(0.0, float(v)) for v in weights], dtype=np.float64)
    active = (
        [bool(a) for a in active_flags]
        if (active_flags is not None and len(active_flags) == n)
        else [True] * n
    )
    locked = (
        [bool(lk) for lk in locked_flags]
        if (locked_flags is not None and len(locked_flags) == n)
        else [False] * n
    )

    # Zero out inactive criteria
    for k in range(n):
        if not active[k]:
            w[k] = 0.0

    active_indices = [k for k in range(n) if active[k]]
    if not active_indices:
        return [0.0] * n

    if len(active_indices) == 1:
        w[:] = 0.0
        w[active_indices[0]] = 100.0
        return w.tolist()

    idx = int(changed_index)
    if idx < 0 or idx >= n or not active[idx] or locked[idx]:
        # Normalize existing active unlocked weights to fill 100 - locked
        locked_sum = sum(w[k] for k in active_indices if locked[k])
        unlocked_others = [k for k in active_indices if not locked[k]]
        rem = max(0.0, 100.0 - locked_sum)
        if unlocked_others:
            s_u = sum(w[k] for k in unlocked_others)
            if s_u > 1e-9:
                for k in unlocked_others:
                    w[k] = w[k] * (rem / s_u)
            else:
                for k in unlocked_others:
                    w[k] = rem / len(unlocked_others)
        return np.round(w, 4).tolist()

    other_locked = [k for k in active_indices if k != idx and locked[k]]
    other_unlocked = [k for k in active_indices if k != idx and not locked[k]]
    locked_budget = float(sum(w[k] for k in other_locked))
    max_allowed = max(0.0, 100.0 - locked_budget)

    if not other_unlocked:
        # No other unlocked criterion can move; idx must hold max_allowed
        w[idx] = max_allowed
        return np.round(w, 4).tolist()

    clamped_new = float(np.clip(float(new_value), 0.0, max_allowed))
    w[idx] = clamped_new
    remaining = max(0.0, max_allowed - clamped_new)

    sum_other = float(sum(w[k] for k in other_unlocked))
    if sum_other > 1e-9:
        scale = remaining / sum_other
        for k in other_unlocked:
            w[k] = w[k] * scale
    else:
        share = remaining / float(len(other_unlocked))
        for k in other_unlocked:
            w[k] = share

    # Ensure exact 100.0 sum after rounding
    w = np.round(w, 2)
    diff = round(100.0 - float(np.sum(w)), 2)
    if abs(diff) > 0.0 and other_unlocked:
        target_k = max(other_unlocked, key=lambda k: w[k])
        w[target_k] = round(max(0.0, float(w[target_k]) + diff), 2)
    return w.tolist()


def equal_weights_pct(n: int, active_flags: Optional[Sequence[bool]] = None) -> List[float]:
    """Return equal percentage weights summing to 100.0 across active criteria."""
    if n <= 0:
        return []
    active = (
        [bool(a) for a in active_flags]
        if (active_flags is not None and len(active_flags) == n)
        else [True] * n
    )
    cnt = sum(1 for a in active if a)
    if cnt == 0:
        return [0.0] * n
    base = round(100.0 / cnt, 2)
    out = [base if a else 0.0 for a in active]
    diff = round(100.0 - sum(out), 2)
    for k in range(n - 1, -1, -1):
        if active[k]:
            out[k] = round(out[k] + diff, 2)
            break
    return out


def seed_weights_for_fields(
    fields: Sequence[str],
    previous: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[List[float], List[bool], List[bool]]:
    """Initial ``(weights, active, locked)`` when the criterion set changes.

    Criteria already known (by field name) keep their activity, lock and
    relative proportions; newcomers receive a fair ``1/N`` share; user-locked
    weights are honored and the active total is rebalanced to 100%.
    """
    prev = previous or {}
    n = len(fields)
    if n == 0:
        return [], [], []
    active = [bool(prev.get(f, {}).get("active", True)) for f in fields]
    locked = [bool(prev.get(f, {}).get("locked", False)) for f in fields]
    known = [f in prev for f in fields]
    if not any(known):
        return equal_weights_pct(n, active_flags=active), active, locked

    fresh_share = 100.0 / max(1, sum(active))
    weights = [
        float(prev[f].get("weight", 0.0)) if known[k] else fresh_share  # type: ignore[union-attr]
        for k, f in enumerate(fields)
    ]
    pinned = [locked[k] or not known[k] for k in range(n)]
    weights = rebalance_coupled_weights(weights, -1, 0.0, locked_flags=pinned, active_flags=active)
    weights = rebalance_coupled_weights(weights, -1, 0.0, locked_flags=locked, active_flags=active)
    return [round(float(w), 4) for w in weights], active, locked


# ---------------------------------------------------------------------------
# 2. Objective & Expert Weight Engines (CRITIC, Entropy, PCA, AHP)
# ---------------------------------------------------------------------------

def _to_unit_01(matrix_0_100: np.ndarray) -> np.ndarray:
    """Scale each column of an (M, N) matrix to [0, 1]."""
    X = np.asarray(matrix_0_100, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64)
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    span = np.where((maxs - mins) <= 1e-12, 1.0, (maxs - mins))
    Z = (X - mins) / span
    return np.clip(np.nan_to_num(Z, nan=0.0), 0.0, 1.0)


def _normalize_to_100(raw_weights: np.ndarray) -> List[float]:
    n = raw_weights.size
    if n == 0:
        return []
    w = np.where(np.isfinite(raw_weights) & (raw_weights > 0.0), raw_weights, 0.0)
    s = float(np.sum(w))
    if s <= 1e-12:
        return equal_weights_pct(n)
    pct = np.round((w / s) * 100.0, 2)
    diff = round(100.0 - float(np.sum(pct)), 2)
    max_idx = int(np.argmax(pct))
    pct[max_idx] = round(max(0.0, float(pct[max_idx]) + diff), 2)
    return pct.tolist()


def critic_weights_pct(matrix_0_100: np.ndarray) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """Compute CRITIC objective weights (%) from an (M, N) normalized H3 matrix."""
    Z = _to_unit_01(matrix_0_100)
    if Z.size == 0:
        return [], np.array([]), np.array([])
    m, n = Z.shape
    if m < 3 or n == 1:
        return equal_weights_pct(n), np.zeros(n), np.ones(n)

    sigma = np.std(Z, axis=0)
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    conflict = np.sum(1.0 - corr, axis=1)
    scores = sigma * conflict
    return _normalize_to_100(scores), sigma, conflict


def entropy_weights_pct(matrix_0_100: np.ndarray) -> List[float]:
    """Compute Shannon Entropy objective weights (%) from an (M, N) H3 matrix."""
    Z = _to_unit_01(matrix_0_100)
    if Z.size == 0:
        return []
    m, n = Z.shape
    if m < 2 or n == 1:
        return equal_weights_pct(n)
    P = Z + 1e-12
    P = P / np.sum(P, axis=0, keepdims=True)
    entropy = -np.sum(P * np.log(P), axis=0) / np.log(max(2, m))
    diversity = np.maximum(0.0, 1.0 - entropy)
    return _normalize_to_100(diversity)


def pca_weights_pct(matrix_0_100: np.ndarray) -> List[float]:
    """Compute PCA first-component objective weights (%) from an (M, N) H3 matrix."""
    X = np.asarray(matrix_0_100, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        return []
    m, n = X.shape
    if m < 3 or n == 1:
        return equal_weights_pct(n)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    cov = np.atleast_2d(np.cov(Z, rowvar=False))
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    if vals[0] <= 1e-12:
        return equal_weights_pct(n)
    loadings = np.abs(vecs[:, 0])
    return _normalize_to_100(loadings)


def validate_ahp_matrix(matrix: np.ndarray, tolerance: float = 0.02) -> None:
    """Validate reciprocity and positivity of an (N, N) AHP pairwise matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise ValueError("AHP pairwise matrix must be a non-empty square matrix.")
    n = m.shape[0]
    if n > max(RI_TABLE):
        raise ValueError(f"AHP supports at most {max(RI_TABLE)} criteria; got {n}.")
    if not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("All AHP matrix entries must be finite and strictly positive.")
    for i in range(n):
        if abs(m[i, i] - 1.0) > tolerance:
            raise ValueError(f"AHP matrix diagonal [{i}][{i}] must be 1.0.")
        for j in range(i + 1, n):
            if abs(m[i, j] * m[j, i] - 1.0) > tolerance:
                raise ValueError(
                    f"AHP matrix is not reciprocal at [{i}][{j}]={m[i, j]:g} and [{j}][{i}]={m[j, i]:g}."
                )


def ahp_weights_from_matrix(matrix: Sequence[Sequence[float]] | np.ndarray) -> Tuple[List[float], float, float, float]:
    """Compute AHP principal eigenvector weights (%), lambda_max, CI, and CR."""
    m = np.asarray(matrix, dtype=np.float64)
    validate_ahp_matrix(m)
    n = m.shape[0]
    vals, vecs = np.linalg.eig(m)
    max_idx = int(np.argmax(vals.real))
    lam_max = float(vals[max_idx].real)
    w_raw = np.abs(vecs[:, max_idx].real)
    weights_pct = _normalize_to_100(w_raw)

    ri = RI_TABLE.get(n, 0.0)
    ci = max(0.0, (lam_max - float(n)) / max(1.0, float(n - 1)))
    cr = 0.0 if ri <= 0.0 else (ci / ri)
    return weights_pct, round(lam_max, 4), round(ci, 4), round(cr, 4)


# ---------------------------------------------------------------------------
# 3. Multi-Criteria Composition / Overlay Engines (WLC, OWA, TOPSIS, VIKOR)
# ---------------------------------------------------------------------------

def _prepare_weights_01(weights_pct: Sequence[float], n_cols: int) -> np.ndarray:
    w = np.asarray(weights_pct, dtype=np.float64)
    if w.size != n_cols:
        raise ValueError(f"Expected {n_cols} criterion weights, got {w.size}.")
    w = np.maximum(0.0, w)
    s = float(np.sum(w))
    if s <= 1e-12:
        return np.full(n_cols, 1.0 / max(1, n_cols), dtype=np.float64)
    return w / s


def _feasible_rows(constraint_mask, m: int) -> np.ndarray:
    """Boolean (M,) array of hexagons allowed by the hard constraint mask."""
    if constraint_mask is None:
        return np.ones(m, dtype=bool)
    mask = np.asarray(constraint_mask, dtype=np.float64).reshape(-1)
    if mask.size != m:
        return np.ones(m, dtype=bool)
    feas = mask > 0.5
    return feas if np.any(feas) else np.ones(m, dtype=bool)


def compose_wlc(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
) -> np.ndarray:
    """Weighted Linear Combination (WLC) on an (M, N) normalized [0, 100] matrix."""
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    w = _prepare_weights_01(weights_pct, X.shape[1])
    scores = np.sum(X * w[None, :], axis=1)
    scores = np.clip(scores, 0.0, 100.0)
    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        scores = scores * mask
    return scores


OWA_ATTITUDE_ALPHA: Dict[str, float] = {
    "pessimistic": 2.0,   # orness ~0.33: weak criteria dominate (precautionary)
    "neutral": 1.0,       # orness 0.50: identical to WLC
    "optimistic": 0.5,    # orness ~0.67: strong criteria dominate (opportunity)
}


def owa_orness(alpha: float) -> float:
    """Orness of the RIM quantifier ``Q(r) = r**alpha`` (1 = max/OR, 0 = min/AND)."""
    return 1.0 / (1.0 + max(1e-9, float(alpha)))


def compose_owa(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    attitude: str = "pessimistic",
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Weighted OWA (Torra 1997) with a RIM quantifier ``Q(r) = r**alpha``.

    For each hexagon the criteria are sorted from highest to lowest score and
    receive the weights ``Q(S_i) - Q(S_{i-1})``, where ``S_i`` is the cumulative
    *importance* weight of the ``i`` best criteria. This keeps the analyst's
    criterion weights and adds a decision attitude:

    - ``alpha > 1`` (pessimistic / precautionary): low scores dominate, AND-like
    - ``alpha = 1``: exactly WLC
    - ``alpha < 1`` (optimistic / opportunity): high scores dominate, OR-like

    ``alpha`` overrides ``attitude`` (``'pessimistic'``, ``'neutral'``,
    ``'optimistic'``). Orness = ``1 / (1 + alpha)``.
    """
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    m, n = X.shape
    w = _prepare_weights_01(weights_pct, n)
    a = float(alpha) if alpha is not None else OWA_ATTITUDE_ALPHA.get(attitude, 2.0)
    a = float(np.clip(a, 0.05, 20.0))
    if n == 1 or abs(a - 1.0) < 1e-9:
        return compose_wlc(X, weights_pct, constraint_mask=constraint_mask)

    sort_idx = np.argsort(-X, axis=1, kind="stable")
    X_sorted = np.take_along_axis(X, sort_idx, axis=1)
    cum = np.cumsum(w[sort_idx], axis=1)
    cum = np.clip(cum / np.maximum(cum[:, -1:], 1e-12), 0.0, 1.0)
    q_hi = cum ** a
    q_lo = np.concatenate([np.zeros((m, 1)), q_hi[:, :-1]], axis=1)
    wowa_w = q_hi - q_lo

    scores = np.clip(np.sum(X_sorted * wowa_w, axis=1), 0.0, 100.0)
    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        scores = scores * mask
    return scores


def compose_topsis(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
) -> np.ndarray:
    """TOPSIS relative closeness (0-100) on pre-aligned [0, 100] criteria."""
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    w = _prepare_weights_01(weights_pct, X.shape[1])
    # Excluded hexagons must not define the ideal / anti-ideal reference points.
    feas = _feasible_rows(constraint_mask, X.shape[0])

    norms = np.sqrt(np.sum(X[feas] ** 2, axis=0, keepdims=True))
    norms = np.where(norms <= 1e-12, 1.0, norms)
    V = (X / norms) * w[None, :]

    ideal_best = np.max(V[feas], axis=0)
    ideal_worst = np.min(V[feas], axis=0)

    d_pos = np.sqrt(np.sum((V - ideal_best[None, :]) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((V - ideal_worst[None, :]) ** 2, axis=1))
    denom = d_pos + d_neg
    closeness = np.where(denom <= 1e-12, 0.5, d_neg / denom) * 100.0
    scores = np.clip(closeness, 0.0, 100.0)
    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        scores = scores * mask
    return scores


def compose_vikor(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    v: float = 0.5,
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
) -> np.ndarray:
    """VIKOR compromise score (0-100, higher = higher composite target score)."""
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    w = _prepare_weights_01(weights_pct, X.shape[1])
    feas = _feasible_rows(constraint_mask, X.shape[0])

    f_star = np.max(X[feas], axis=0)
    f_minus = np.min(X[feas], axis=0)
    span = np.where((f_star - f_minus) <= 1e-12, 1.0, (f_star - f_minus))

    # Normalized regret distance per cell & criterion (0 = at best/target, 1 = at worst)
    regret_mat = w[None, :] * np.clip((f_star[None, :] - X) / span[None, :], 0.0, None)
    S = np.sum(regret_mat, axis=1)
    R = np.max(regret_mat, axis=1)

    s_min, s_max = float(np.min(S[feas])), float(np.max(S[feas]))
    r_min, r_max = float(np.min(R[feas])), float(np.max(R[feas]))
    s_term = (S - s_min) / max(1e-12, s_max - s_min) if (s_max > s_min + 1e-12) else np.zeros_like(S)
    r_term = (R - r_min) / max(1e-12, r_max - r_min) if (r_max > r_min + 1e-12) else np.zeros_like(R)

    Q = np.clip(float(v) * s_term + (1.0 - float(v)) * r_term, 0.0, 1.0)
    # Invert Q (where 0 = ideal target) to 0..100 score so 100 = highest composite intensity
    scores = np.clip((1.0 - Q) * 100.0, 0.0, 100.0)
    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        scores = scores * mask
    return scores


def compose_scores(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    method: str = "wlc",
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
    vikor_v: float = 0.5,
    owa_alpha: Optional[float] = None,
) -> np.ndarray:
    """Dispatch to the selected multi-criteria composition algorithm.

    ``owa_alpha`` (RIM quantifier exponent) overrides the attitude implied by
    ``owa_pessimistic`` / ``owa_optimistic``.
    """
    m = (method or "wlc").strip().lower()
    if m in ("owa_pessimistic", "owa_optimistic", "owa"):
        attitude = "optimistic" if m == "owa_optimistic" else "pessimistic"
        return compose_owa(
            matrix_0_100, weights_pct, attitude=attitude,
            constraint_mask=constraint_mask, alpha=owa_alpha,
        )
    if m == "topsis":
        return compose_topsis(matrix_0_100, weights_pct, constraint_mask=constraint_mask)
    if m == "vikor":
        return compose_vikor(matrix_0_100, weights_pct, v=vikor_v, constraint_mask=constraint_mask)
    return compose_wlc(matrix_0_100, weights_pct, constraint_mask=constraint_mask)


def apply_h3_spatial_smoothing(
    scores: np.ndarray,
    h3_ids: List[str],
    alpha: float = 0.20,
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
) -> np.ndarray:
    """Blend each hexagon's score with its 1-ring H3 neighbors by factor alpha in [0, 0.5]."""
    a = float(np.clip(alpha, 0.0, 0.5))
    if a <= 1e-6 or len(h3_ids) != len(scores):
        return np.asarray(scores, dtype=np.float64)

    adj = get_h3_neighbors(h3_ids)
    id_to_idx = {cid: i for i, cid in enumerate(h3_ids)}
    out = np.array(scores, dtype=np.float64, copy=True)
    # Excluded neighbors carry a forced 0 score; never let them bleed into
    # feasible hexagons through the contagion kernel.
    feas = (
        (np.asarray(constraint_mask, dtype=np.float64) > 0.5)
        if constraint_mask is not None and len(constraint_mask) == len(h3_ids)
        else np.ones(len(h3_ids), dtype=bool)
    )

    for i, cid in enumerate(h3_ids):
        nbrs = adj.get(cid, [])
        if not nbrs or not feas[i]:
            continue
        nbr_vals = [
            float(scores[id_to_idx[n]]) for n in nbrs
            if n in id_to_idx and feas[id_to_idx[n]]
        ]
        if nbr_vals:
            out[i] = (1.0 - a) * float(scores[i]) + a * float(np.mean(nbr_vals))

    out = np.clip(out, 0.0, 100.0)
    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        out = out * mask
    return out


# ---------------------------------------------------------------------------
# 4. Explainable AI (XAI Local Driver) & Monte Carlo Stability
# ---------------------------------------------------------------------------

def compute_local_xai_drivers(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    criterion_names: Sequence[str],
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    """Identify the dominant contributing criterion and its percentage share per hexagon.

    Hexagons removed by a hard constraint are labelled ``"Excluded"``.
    """
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    m, n = X.shape
    w = _prepare_weights_01(weights_pct, n)
    clean_names = [
        str(c).replace("norm_", "").replace("raw_", "").replace("ring_", "")
        for c in criterion_names
    ]

    weighted_parts = X * w[None, :]
    totals = np.sum(weighted_parts, axis=1)
    top_idx = np.argmax(weighted_parts, axis=1)

    excluded = (
        np.asarray(constraint_mask, dtype=np.float64).reshape(-1) <= 0.5
        if constraint_mask is not None and len(constraint_mask) == m
        else np.zeros(m, dtype=bool)
    )
    drivers: List[str] = []
    shares = np.zeros(m, dtype=np.float64)
    for i in range(m):
        j = int(top_idx[i])
        tot = float(totals[i])
        if excluded[i]:
            drivers.append("Excluded")
            shares[i] = 0.0
        elif tot <= 1e-9:
            drivers.append("None")
            shares[i] = 0.0
        else:
            drivers.append(clean_names[j])
            shares[i] = min(100.0, (float(weighted_parts[i, j]) / tot) * 100.0)
    return drivers, np.round(shares, 1)


def compute_montecarlo_stability(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
    n_runs: int = 50,
    perturbation: float = 0.15,
    risk_aversion: float = 0.5,
    seed: int = 42,
    method: str = "wlc",
    owa_alpha: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Monte Carlo weight perturbation to compute (mc_mean, mc_std, rob_score).

    The perturbed weights are pushed through the *same* composition engine
    (``method``) used for the composite map, so ``mc_std`` measures the
    uncertainty of the map the analyst actually sees (WLC is vectorized; OWA,
    TOPSIS and VIKOR are evaluated run by run).
    """
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    m, n = X.shape
    w0 = _prepare_weights_01(weights_pct, n)
    rng = np.random.default_rng(int(seed))

    runs = max(5, int(n_runs))
    noise = rng.uniform(1.0 - float(perturbation), 1.0 + float(perturbation), size=(runs, n))
    W_sim = w0[None, :] * noise
    W_sums = np.sum(W_sim, axis=1, keepdims=True)
    W_sums = np.where(W_sums <= 1e-12, 1.0, W_sums)
    W_sim = W_sim / W_sums

    m_key = (method or "wlc").strip().lower()
    if m_key == "wlc":
        # (M, N) @ (N, runs) -> (M, runs)
        sim_scores = X @ W_sim.T
    else:
        sim_scores = np.empty((m, runs), dtype=np.float64)
        for k in range(runs):
            sim_scores[:, k] = compose_scores(
                X, W_sim[k] * 100.0, method=m_key, constraint_mask=constraint_mask,
                owa_alpha=owa_alpha,
            )
    mc_mean = np.mean(sim_scores, axis=1)
    mc_std = np.std(sim_scores, axis=1)
    rob_score = np.clip(mc_mean - float(risk_aversion) * mc_std, 0.0, 100.0)

    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        mc_mean = mc_mean * mask
        mc_std = mc_std * mask
        rob_score = rob_score * mask

    return np.round(mc_mean, 2), np.round(mc_std, 2), np.round(rob_score, 2)


def jenks_breaks(values: Sequence[float] | np.ndarray, n_classes: int = 5, max_sample: int = 2000) -> List[float]:
    """Fisher-Jenks natural breaks (exact DP, vectorized with prefix sums).

    Large inputs are reduced to ``max_sample`` evenly spaced order statistics,
    which preserves the distribution shape while keeping the O(k·n²) dynamic
    program fast. Returns ``n_classes + 1`` breaks from min to max.
    """
    x = np.sort(np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64))
    if x.size == 0:
        return [0.0] * (n_classes + 1)
    if x.size > max_sample:
        x = x[np.linspace(0, x.size - 1, max_sample).round().astype(int)]
    n = x.size
    k = int(min(n_classes, np.unique(x).size))
    if k <= 1:
        return [float(x[0])] * n_classes + [float(x[-1])]

    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])
    # cost[j][i]: minimal within-class SSD for the first i values in j classes
    inf = np.inf
    cost = np.full((k + 1, n + 1), inf)
    back = np.zeros((k + 1, n + 1), dtype=np.int64)
    cost[0, 0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            starts = np.arange(j - 1, i)          # class covers x[l:i]
            cnt = i - starts
            ssd = (s2[i] - s2[starts]) - (s1[i] - s1[starts]) ** 2 / cnt
            total = cost[j - 1, starts] + ssd
            best = int(np.argmin(total))
            cost[j, i] = total[best]
            back[j, i] = starts[best]
    cuts: List[int] = []
    i = n
    for j in range(k, 0, -1):
        start = int(back[j, i])
        cuts.append(start)
        i = start
    cuts = sorted(cuts)[1:]  # drop the leading 0
    breaks = [float(x[0])] + [float(x[c - 1]) for c in cuts] + [float(x[-1])]
    while len(breaks) < n_classes + 1:
        breaks.insert(-1, breaks[-1])
    return breaks


def classify_scores_5_bands(
    scores: Sequence[float] | np.ndarray,
    method: str = "equal",
    theme: str = "risk",
) -> Tuple[List[int], List[str], List[float]]:
    """Classify 0..100 composite scores into 5 classes (1..5) and return breaks."""
    arr = np.clip(np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0), 0.0, 100.0)
    labels = CLASS_LABELS_SUITABILITY if theme == "suitability" else CLASS_LABELS_RISK

    if method in ("quantile", "jenks") and arr.size >= 5:
        pos = arr[arr > 0.0] if np.any(arr > 0.0) else arr
        if method == "jenks":
            qs = [float(b) for b in jenks_breaks(pos, 5)[1:5]]
        else:
            qs = [float(np.percentile(pos, q)) for q in (20, 40, 60, 80)]
        # Ensure strictly increasing breaks; fallback to equal interval if degenerate
        if len(set(round(v, 2) for v in qs)) == 4 and qs[0] > 0.0 and qs[-1] < 100.0:
            breaks = [0.0] + qs + [100.0]
        else:
            breaks = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    else:
        breaks = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    class_ids: List[int] = []
    class_names: List[str] = []
    for val in arr:
        cid = 5
        for b_idx in range(4):
            if val <= breaks[b_idx + 1]:
                cid = b_idx + 1
                break
        class_ids.append(cid)
        class_names.append(labels[cid - 1])
    return class_ids, class_names, breaks


def _class_ids_with_breaks(scores: np.ndarray, breaks: Sequence[float]) -> np.ndarray:
    """Vectorized 1..5 class assignment matching :func:`classify_scores_5_bands`."""
    inner = np.asarray(list(breaks)[1:5], dtype=np.float64)
    return np.searchsorted(inner, np.asarray(scores, dtype=np.float64), side="left") + 1


def oat_weight_sensitivity(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    criterion_names: Sequence[str],
    method: str = "wlc",
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
    deltas_pct: Sequence[float] = (-20.0, -10.0, 10.0, 20.0),
    breaks: Optional[Sequence[float]] = None,
    owa_alpha: Optional[float] = None,
) -> List[Dict[str, object]]:
    """One-at-a-time (OAT) weight sensitivity of the composite map.

    Each criterion weight is changed by a *relative* ``delta`` (e.g. +20% of
    its own value); the other weights are rescaled proportionally so the total
    stays 100%. For every perturbation the map is recomposed with the same
    engine and compared to the baseline on feasible hexagons:

    - ``mean_abs_change``: mean |Δscore| (score points, 0-100 scale)
    - ``max_abs_change``: largest |Δscore| of any hexagon
    - ``class_change_pct``: share of hexagons that move to another class band

    Returns one row per criterion (worst case over ``deltas_pct``), sorted from
    most to least influential.
    """
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    m, n = X.shape
    w0 = np.asarray(_prepare_weights_01(weights_pct, n) * 100.0, dtype=np.float64)
    feas = _feasible_rows(constraint_mask, m)
    b_list = list(breaks) if breaks is not None and len(breaks) == 6 else [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]

    base = compose_scores(X, w0, method=method, constraint_mask=constraint_mask, owa_alpha=owa_alpha)
    base_cls = _class_ids_with_breaks(base, b_list)
    names = [
        str(c).replace("norm_", "").replace("raw_", "").replace("ring_", "")
        for c in criterion_names
    ]

    rows: List[Dict[str, object]] = []
    for j in range(n):
        worst = {"mean_abs_change": 0.0, "max_abs_change": 0.0, "class_change_pct": 0.0}
        if w0[j] <= 1e-9 or n == 1:
            rows.append({"criterion": names[j], "weight": round(float(w0[j]), 2), **worst})
            continue
        for d in deltas_pct:
            target = float(np.clip(w0[j] * (1.0 + float(d) / 100.0), 0.0, 100.0))
            w = rebalance_coupled_weights(w0.tolist(), j, target)
            sc = compose_scores(X, w, method=method, constraint_mask=constraint_mask, owa_alpha=owa_alpha)
            diff = np.abs(sc - base)[feas]
            cls_changed = (_class_ids_with_breaks(sc, b_list) != base_cls)[feas]
            worst["mean_abs_change"] = max(worst["mean_abs_change"], float(np.mean(diff)) if diff.size else 0.0)
            worst["max_abs_change"] = max(worst["max_abs_change"], float(np.max(diff)) if diff.size else 0.0)
            pct = float(np.mean(cls_changed) * 100.0) if cls_changed.size else 0.0
            worst["class_change_pct"] = max(worst["class_change_pct"], pct)
        rows.append({
            "criterion": names[j],
            "weight": round(float(w0[j]), 2),
            "mean_abs_change": round(worst["mean_abs_change"], 3),
            "max_abs_change": round(worst["max_abs_change"], 3),
            "class_change_pct": round(worst["class_change_pct"], 2),
        })
    rows.sort(key=lambda r: (-float(r["class_change_pct"]), -float(r["mean_abs_change"])))
    return rows


def run_weight_sensitivity_on_layer(
    h3_layer,
    criterion_fields: Sequence[str],
    weights_pct: Sequence[float],
    compose_method: str = "wlc",
    apply_constraints: bool = True,
    breaks: Optional[Sequence[float]] = None,
    deltas_pct: Sequence[float] = (-20.0, -10.0, 10.0, 20.0),
    owa_alpha: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Run :func:`oat_weight_sensitivity` on ``h3_layer`` and persist the table."""
    _, _, mat, mask_arr = extract_h3_criteria_matrix(h3_layer, criterion_fields)
    if mat.shape[0] == 0:
        raise ValueError("H3 layer contains no features.")
    rows = oat_weight_sensitivity(
        mat,
        weights_pct,
        criterion_fields,
        method=compose_method,
        constraint_mask=mask_arr if apply_constraints else None,
        deltas_pct=deltas_pct,
        breaks=breaks,
        owa_alpha=owa_alpha,
    )
    h3_layer.setCustomProperty("h3forge/sensitivity", json.dumps({
        "method": compose_method,
        "deltas_pct": [float(d) for d in deltas_pct],
        "rows": rows,
    }))
    return rows


# ---------------------------------------------------------------------------
# 5. PyQGIS Layer Runner
# ---------------------------------------------------------------------------

def extract_h3_criteria_matrix(
    h3_layer,
    criterion_fields: Sequence[str],
) -> Tuple[List[int], List[str], np.ndarray, np.ndarray]:
    """Read ``(fids, h3_ids, matrix_0_100, constraint_mask)`` from ``h3_layer``."""
    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if not criterion_fields:
        raise ValueError("At least one normalized criterion field is required.")

    fields = h3_layer.fields()
    col_indices: List[int] = []
    for cname in criterion_fields:
        idx = fields.indexFromName(cname)
        if idx < 0:
            raise ValueError(f"Criterion field '{cname}' not found on H3 layer.")
        col_indices.append(idx)

    idx_h3id = fields.indexFromName("h3_id")
    idx_mask = fields.indexFromName("constraint_mask")

    fids: List[int] = []
    h3_ids: List[str] = []
    rows: List[List[float]] = []
    masks: List[int] = []

    for feat in h3_layer.getFeatures():
        fids.append(int(feat.id()))
        h3_ids.append(str(feat.attribute(idx_h3id)) if idx_h3id >= 0 else str(feat.id()))
        if idx_mask >= 0:
            mv = feat.attribute(idx_mask)
            masks.append(0 if (mv is not None and int(mv) == 0) else 1)
        else:
            masks.append(1)

        row_vals: List[float] = []
        for c_idx in col_indices:
            v = feat.attribute(c_idx)
            try:
                fv = float(v) if v is not None else 0.0
                if not np.isfinite(fv):
                    fv = 0.0
            except (ValueError, TypeError):
                fv = 0.0
            row_vals.append(fv)
        rows.append(row_vals)

    mat = np.asarray(rows, dtype=np.float64)
    mask_arr = np.asarray(masks, dtype=np.int32)
    return fids, h3_ids, mat, mask_arr


def run_h3_overlay_on_layer(
    h3_layer,
    criterion_fields: Sequence[str],
    weights_pct: Sequence[float],
    compose_method: str = "wlc",
    apply_constraints: bool = True,
    spatial_smoothing_alpha: float = 0.0,
    run_montecarlo: bool = True,
    classification_method: str = "equal",
    theme: str = "risk",
    mc_runs: int = 50,
    mc_perturbation: float = 0.15,
    owa_alpha: Optional[float] = None,
) -> Dict[str, object]:
    """Execute full MCDA overlay, XAI attribution, and classification on ``h3_layer``."""
    fids, h3_ids, mat, mask_arr = extract_h3_criteria_matrix(h3_layer, criterion_fields)
    if mat.shape[0] == 0:
        raise ValueError("H3 layer contains no features.")

    active_mask = mask_arr if apply_constraints else None
    raw_scores = compose_scores(
        mat,
        weights_pct,
        method=compose_method,
        constraint_mask=active_mask,
        owa_alpha=owa_alpha,
    )
    if spatial_smoothing_alpha > 1e-6:
        comp_scores = apply_h3_spatial_smoothing(
            raw_scores,
            h3_ids,
            alpha=spatial_smoothing_alpha,
            constraint_mask=active_mask,
        )
    else:
        comp_scores = raw_scores

    drivers, shares = compute_local_xai_drivers(
        mat, weights_pct, criterion_fields, constraint_mask=active_mask
    )
    class_ids, class_labels, breaks = classify_scores_5_bands(
        comp_scores,
        method=classification_method,
        theme=theme,
    )

    if run_montecarlo:
        _, mc_std, rob_score = compute_montecarlo_stability(
            mat,
            weights_pct,
            constraint_mask=active_mask,
            n_runs=mc_runs,
            perturbation=mc_perturbation,
            method=compose_method,
            owa_alpha=owa_alpha,
        )
    else:
        mc_std = np.zeros(len(fids), dtype=np.float64)
        rob_score = np.round(comp_scores, 2)

    idx_score = ensure_layer_field(h3_layer, "comp_score", kind="double")
    idx_cid = ensure_layer_field(h3_layer, "class_id", kind="int")
    idx_clbl = ensure_layer_field(h3_layer, "class_label", kind="string", length=32)
    idx_drv = ensure_layer_field(h3_layer, "top_driver", kind="string", length=48)
    idx_shr = ensure_layer_field(h3_layer, "driver_share", kind="double")
    idx_std = ensure_layer_field(h3_layer, "mc_std", kind="double")
    idx_rob = ensure_layer_field(h3_layer, "rob_score", kind="double")

    updates: Dict[int, Dict[int, object]] = {}
    for i, fid in enumerate(fids):
        updates[fid] = {
            idx_score: round(float(comp_scores[i]), 2),
            idx_cid: int(class_ids[i]),
            idx_clbl: str(class_labels[i]),
            idx_drv: str(drivers[i]),
            idx_shr: round(float(shares[i]), 1),
            idx_std: round(float(mc_std[i]), 2),
            idx_rob: round(float(rob_score[i]), 2),
        }

    if updates:
        h3_layer.dataProvider().changeAttributeValues(updates)

    # Persist run provenance on the layer (read back by the HTML report).
    h3_layer.setCustomProperty("h3forge/last_run", json.dumps({
        "method": compose_method,
        "owa_alpha": (float(owa_alpha) if owa_alpha is not None else None),
        "theme": theme,
        "class_method": classification_method,
        "breaks": [round(float(b), 3) for b in breaks],
        "alpha": float(spatial_smoothing_alpha),
        "constraints": bool(apply_constraints),
        "montecarlo": bool(run_montecarlo),
        "mc_runs": int(mc_runs),
        "mc_perturbation": float(mc_perturbation),
        "weights": {str(f): round(float(w), 4) for f, w in zip(criterion_fields, weights_pct)},
    }))

    feas = (
        np.asarray(mask_arr, dtype=np.int32) > 0
        if apply_constraints
        else np.ones(len(fids), dtype=bool)
    )
    if not np.any(feas):
        feas = np.ones(len(fids), dtype=bool)
    return {
        "feature_count": len(fids),
        "excluded_count": int(np.sum(mask_arr == 0)) if apply_constraints else 0,
        "mean_rob_score": round(float(np.mean(rob_score[feas])), 2),
        "mean_score": round(float(np.mean(comp_scores)), 2),
        "max_score": round(float(np.max(comp_scores)), 2),
        "min_score": round(float(np.min(comp_scores)), 2),
        "mean_mc_std": round(float(np.mean(mc_std)), 2),
        "breaks": breaks,
    }
