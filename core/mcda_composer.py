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


def compose_owa(
    matrix_0_100: np.ndarray,
    weights_pct: Sequence[float],
    attitude: str = "pessimistic",
    constraint_mask: Optional[Sequence[int] | np.ndarray] = None,
) -> np.ndarray:
    """Ordered Weighted Averaging (OWA) combining criterion weights and order weights.

    - ``'pessimistic'`` (precautionary risk): emphasizes worst/highest-hazard rank
    - ``'optimistic'``: emphasizes best/highest rank
    - ``'neutral'``: equivalent to WLC
    """
    X = np.clip(np.nan_to_num(np.asarray(matrix_0_100, dtype=np.float64), nan=0.0), 0.0, 100.0)
    m, n = X.shape
    w = _prepare_weights_01(weights_pct, n)
    if n == 1 or attitude == "neutral":
        return compose_wlc(X, weights_pct, constraint_mask=constraint_mask)

    ranks = np.arange(1, n + 1, dtype=np.float64)
    if attitude == "optimistic":
        # Highest values receive highest order weight
        order_w = (n - ranks + 1.0) ** 1.8
    else:
        # Pessimistic / precautionary: lowest values receive highest order weight
        order_w = ranks ** 1.8
    order_w = order_w / np.sum(order_w)

    # Sort each hexagon's criteria descending
    sort_idx = np.argsort(-X, axis=1)
    X_sorted = np.take_along_axis(X, sort_idx, axis=1)
    w_sorted = w[sort_idx]

    combined_w = w_sorted * order_w[None, :]
    row_sums = np.sum(combined_w, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 1e-12, 1.0, row_sums)
    combined_w = combined_w / row_sums

    scores = np.clip(np.sum(X_sorted * combined_w, axis=1), 0.0, 100.0)
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

    norms = np.sqrt(np.sum(X ** 2, axis=0, keepdims=True))
    norms = np.where(norms <= 1e-12, 1.0, norms)
    V = (X / norms) * w[None, :]

    ideal_best = np.max(V, axis=0)
    ideal_worst = np.min(V, axis=0)

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

    f_star = np.max(X, axis=0)
    f_minus = np.min(X, axis=0)
    span = np.where((f_star - f_minus) <= 1e-12, 1.0, (f_star - f_minus))

    # Normalized regret distance per cell & criterion (0 = at best/target, 1 = at worst)
    regret_mat = w[None, :] * ((f_star[None, :] - X) / span[None, :])
    S = np.sum(regret_mat, axis=1)
    R = np.max(regret_mat, axis=1)

    s_min, s_max = float(np.min(S)), float(np.max(S))
    r_min, r_max = float(np.min(R)), float(np.max(R))
    s_term = (S - s_min) / max(1e-12, s_max - s_min) if (s_max > s_min + 1e-12) else np.zeros_like(S)
    r_term = (R - r_min) / max(1e-12, r_max - r_min) if (r_max > r_min + 1e-12) else np.zeros_like(R)

    Q = float(v) * s_term + (1.0 - float(v)) * r_term
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
) -> np.ndarray:
    """Dispatch to the selected multi-criteria composition algorithm."""
    m = (method or "wlc").strip().lower()
    if m == "owa_pessimistic":
        return compose_owa(matrix_0_100, weights_pct, attitude="pessimistic", constraint_mask=constraint_mask)
    if m == "owa_optimistic":
        return compose_owa(matrix_0_100, weights_pct, attitude="optimistic", constraint_mask=constraint_mask)
    if m == "topsis":
        return compose_topsis(matrix_0_100, weights_pct, constraint_mask=constraint_mask)
    if m == "vikor":
        return compose_vikor(matrix_0_100, weights_pct, constraint_mask=constraint_mask)
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

    for i, cid in enumerate(h3_ids):
        nbrs = adj.get(cid, [])
        if not nbrs:
            continue
        nbr_vals = [float(scores[id_to_idx[n]]) for n in nbrs if n in id_to_idx]
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
) -> Tuple[List[str], np.ndarray]:
    """Identify the dominant contributing criterion and its percentage share per hexagon."""
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

    drivers: List[str] = []
    shares = np.zeros(m, dtype=np.float64)
    for i in range(m):
        j = int(top_idx[i])
        tot = float(totals[i])
        if tot <= 1e-9:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Monte Carlo weight perturbation to compute (mc_mean, mc_std, rob_score)."""
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

    # (M, N) @ (N, runs) -> (M, runs)
    sim_scores = X @ W_sim.T
    mc_mean = np.mean(sim_scores, axis=1)
    mc_std = np.std(sim_scores, axis=1)
    rob_score = np.clip(mc_mean - float(risk_aversion) * mc_std, 0.0, 100.0)

    if constraint_mask is not None:
        mask = (np.asarray(constraint_mask, dtype=np.float64) > 0.5).astype(np.float64)
        mc_mean = mc_mean * mask
        mc_std = mc_std * mask
        rob_score = rob_score * mask

    return np.round(mc_mean, 2), np.round(mc_std, 2), np.round(rob_score, 2)


def classify_scores_5_bands(
    scores: Sequence[float] | np.ndarray,
    method: str = "equal",
    theme: str = "risk",
) -> Tuple[List[int], List[str], List[float]]:
    """Classify 0..100 composite scores into 5 classes (1..5) and return breaks."""
    arr = np.clip(np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0), 0.0, 100.0)
    labels = CLASS_LABELS_SUITABILITY if theme == "suitability" else CLASS_LABELS_RISK

    if method == "quantile" and arr.size >= 5:
        pos = arr[arr > 0.0] if np.any(arr > 0.0) else arr
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

    drivers, shares = compute_local_xai_drivers(mat, weights_pct, criterion_fields)
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

    return {
        "feature_count": len(fids),
        "mean_score": round(float(np.mean(comp_scores)), 2),
        "max_score": round(float(np.max(comp_scores)), 2),
        "min_score": round(float(np.min(comp_scores)), 2),
        "mean_mc_std": round(float(np.mean(mc_std)), 2),
        "breaks": breaks,
    }
