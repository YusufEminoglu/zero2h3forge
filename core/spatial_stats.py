# -*- coding: utf-8 -*-
"""H3 Spatial Statistics: Global Moran's I and Getis-Ord Gi* Hotspots for 02H3Forge.

Both statistics use binary H3 1-ring contiguity weights built from the hexagon
IDs (native ``h3`` or the synthetic metric fallback grid). Hexagons removed by a
hard constraint (``constraint_mask = 0``) are dropped from the analysis so that
their forced zero scores cannot fabricate artificial cold spots.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .h3_grid_engine import ensure_layer_field, get_h3_neighbors


# (code, label, minimum |z|) — two-sided confidence bins used by ArcGIS/PySAL.
HOTSPOT_BINS: List[Tuple[int, str, float]] = [
    (3, "Hot Spot (99%)", 2.576),
    (2, "Hot Spot (95%)", 1.960),
    (1, "Hot Spot (90%)", 1.645),
    (0, "Not Significant", 0.0),
    (-1, "Cold Spot (90%)", 1.645),
    (-2, "Cold Spot (95%)", 1.960),
    (-3, "Cold Spot (99%)", 2.576),
]
HOTSPOT_LABELS: Dict[int, str] = {code: lbl for code, lbl, _ in HOTSPOT_BINS}
EXCLUDED_LABEL = "Excluded"


def normal_two_sided_p(z: float) -> float:
    """Two-sided p-value of a standard normal z-score."""
    if not np.isfinite(z):
        return 1.0
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def build_neighbor_lists(
    h3_ids: Sequence[str],
    valid: Optional[Sequence[bool]] = None,
    k: int = 1,
) -> List[List[int]]:
    """Return k-ring neighbor index lists, restricted to ``valid`` hexagons.

    The k-ring is the set of hexagons within ``k`` grid steps (``k = 1``: the
    6 touching cells; ``k = 2``: 18 cells; ``k = 3``: 36 cells). Grid steps are
    walked across *all* hexagons, so an excluded cell never breaks the
    neighbourhood geometry; it is only dropped from the final lists.
    """
    ids = [str(c) for c in h3_ids]
    n = len(ids)
    ok = [True] * n if valid is None else [bool(v) for v in valid]
    adj = get_h3_neighbors(ids)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    ring1: List[List[int]] = [
        [id_to_idx[nb] for nb in adj.get(cid, []) if nb in id_to_idx and id_to_idx[nb] != i]
        for i, cid in enumerate(ids)
    ]
    depth = max(1, int(k))
    out: List[List[int]] = []
    for i in range(n):
        if not ok[i]:
            out.append([])
            continue
        seen = {i}
        frontier = [i]
        for _ in range(depth):
            nxt = []
            for u in frontier:
                for v in ring1[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
        seen.discard(i)
        out.append(sorted(j for j in seen if ok[j]))
    return out


def _edge_arrays(neighbors: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[int] = []
    cols: List[int] = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            rows.append(i)
            cols.append(int(j))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def global_morans_i(
    values: Sequence[float] | np.ndarray,
    neighbors: Sequence[Sequence[int]],
    valid: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
    """Global Moran's I with binary contiguity weights (normality assumption).

    Returns ``{"I", "expected", "variance", "z", "p", "n", "links"}``.
    """
    x_all = np.asarray(values, dtype=np.float64)
    n_all = x_all.size
    ok = np.ones(n_all, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    ok &= np.isfinite(x_all)

    # Re-index onto the valid subset only
    sub_idx = np.flatnonzero(ok)
    remap = {int(old): new for new, old in enumerate(sub_idx)}
    sub_nbrs: List[List[int]] = []
    for old in sub_idx:
        sub_nbrs.append([remap[j] for j in neighbors[int(old)] if j in remap])

    x = x_all[sub_idx]
    n = int(x.size)
    empty = {"I": float("nan"), "expected": float("nan"), "variance": float("nan"),
             "z": float("nan"), "p": 1.0, "n": float(n), "links": 0.0}
    if n < 3:
        return empty

    r, c = _edge_arrays(sub_nbrs)
    s0 = float(r.size)
    if s0 <= 0.0:
        return empty
    z = x - float(np.mean(x))
    m2 = float(np.sum(z * z))
    if m2 <= 1e-12:
        return {**empty, "links": s0}

    I_val = (n / s0) * float(np.sum(z[r] * z[c])) / m2
    e_i = -1.0 / (n - 1.0)

    # S1 = 1/2 * sum_ij (w_ij + w_ji)^2 ; S2 = sum_i (w_i. + w_.i)^2
    # A reciprocal pair (i->j, j->i) adds 0.5 * (2^2 + 2^2) = 4 (2 per directed
    # edge); a one-way link adds 0.5 * (1^2 + 1^2) = 1.
    edge_set = set(zip(r.tolist(), c.tolist()))
    s1 = sum(2.0 if (j, i) in edge_set else 1.0 for (i, j) in edge_set)
    out_deg = np.bincount(r, minlength=n).astype(np.float64)
    in_deg = np.bincount(c, minlength=n).astype(np.float64)
    s2 = float(np.sum((out_deg + in_deg) ** 2))

    var_n = (n * n * s1 - n * s2 + 3.0 * s0 * s0) / ((n * n - 1.0) * s0 * s0) - e_i * e_i
    var_n = max(var_n, 1e-18)
    z_i = (I_val - e_i) / math.sqrt(var_n)
    return {
        "I": round(I_val, 6),
        "expected": round(e_i, 6),
        "variance": float(var_n),
        "z": round(z_i, 4),
        "p": normal_two_sided_p(z_i),
        "n": float(n),
        "links": s0,
    }


def getis_ord_gi_star(
    values: Sequence[float] | np.ndarray,
    neighbors: Sequence[Sequence[int]],
    valid: Optional[Sequence[bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Getis-Ord Gi* (self-inclusive binary weights) z-scores per hexagon.

    Returns ``(z_scores, p_values, hotspot_codes)`` where codes are in
    ``-3..3`` (see :data:`HOTSPOT_BINS`). Invalid hexagons receive ``z = 0``,
    ``p = 1`` and code ``0``.
    """
    x_all = np.asarray(values, dtype=np.float64)
    n_all = x_all.size
    ok = np.ones(n_all, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    ok &= np.isfinite(x_all)

    z_out = np.zeros(n_all, dtype=np.float64)
    p_out = np.ones(n_all, dtype=np.float64)
    codes = np.zeros(n_all, dtype=np.int32)

    n = int(np.sum(ok))
    if n < 3:
        return z_out, p_out, codes
    xv = x_all[ok]
    x_bar = float(np.mean(xv))
    s = math.sqrt(max(0.0, float(np.mean(xv * xv)) - x_bar * x_bar))
    if s <= 1e-12:
        return z_out, p_out, codes

    for i in np.flatnonzero(ok):
        members = [int(i)] + [int(j) for j in neighbors[int(i)] if ok[int(j)]]
        w_sum = float(len(members))
        local_sum = float(np.sum(x_all[members]))
        denom_sq = (n * w_sum - w_sum * w_sum) / (n - 1.0)
        if denom_sq <= 1e-12:
            continue
        gi = (local_sum - x_bar * w_sum) / (s * math.sqrt(denom_sq))
        z_out[i] = gi
        p_out[i] = normal_two_sided_p(gi)

    codes = classify_hotspots(z_out)
    codes[~ok] = 0
    return np.round(z_out, 4), p_out, codes


def bh_fdr_threshold(p_values: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Benjamini-Hochberg critical p-value for false discovery rate ``alpha``.

    Returns the largest ``p_(i)`` with ``p_(i) <= i / m * alpha`` (0.0 when no
    test survives), i.e. hexagons with ``p <= threshold`` are significant.
    """
    p = np.sort(np.asarray(p_values, dtype=np.float64))
    m = p.size
    if m == 0:
        return 0.0
    crit = (np.arange(1, m + 1) / m) * float(alpha)
    passed = np.flatnonzero(p <= crit)
    return float(p[passed[-1]]) if passed.size else 0.0


def classify_hotspots_fdr(
    z_scores: Sequence[float] | np.ndarray,
    p_values: Sequence[float] | np.ndarray,
    valid: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Hotspot codes ``-3..3`` with Benjamini-Hochberg FDR control per level.

    Confidence levels 90/95/99% use FDR rates 0.10/0.05/0.01 over the valid
    hexagons, which removes the flood of chance "hot spots" that thousands of
    simultaneous local tests otherwise produce.
    """
    z = np.nan_to_num(np.asarray(z_scores, dtype=np.float64), nan=0.0)
    p = np.nan_to_num(np.asarray(p_values, dtype=np.float64), nan=1.0)
    ok = np.ones(z.size, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    codes = np.zeros(z.size, dtype=np.int32)
    for level, alpha in ((1, 0.10), (2, 0.05), (3, 0.01)):
        thr = bh_fdr_threshold(p[ok], alpha)
        if thr <= 0.0:
            continue
        sig = ok & (p <= thr)
        codes[sig & (z > 0)] = level
        codes[sig & (z < 0)] = -level
    return codes


def classify_hotspots(z_scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map Gi* z-scores to hotspot confidence codes ``-3..3``."""
    z = np.nan_to_num(np.asarray(z_scores, dtype=np.float64), nan=0.0)
    codes = np.zeros(z.shape, dtype=np.int32)
    for level, thr in ((1, 1.645), (2, 1.960), (3, 2.576)):
        codes[z >= thr] = level
        codes[z <= -thr] = -level
    return codes


def summarize_hotspots(codes: Sequence[int], valid: Optional[Sequence[bool]] = None) -> Dict[str, int]:
    """Count hexagons per hotspot label (plus ``Excluded``)."""
    c = np.asarray(codes, dtype=np.int32)
    ok = np.ones(c.size, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    out: Dict[str, int] = {lbl: int(np.sum((c == code) & ok)) for code, lbl, _ in HOTSPOT_BINS}
    out[EXCLUDED_LABEL] = int(np.sum(~ok))
    return out


# ---------------------------------------------------------------------------
# PyQGIS layer runner
# ---------------------------------------------------------------------------

def run_hotspot_analysis_on_layer(
    h3_layer,
    value_field: str = "comp_score",
    respect_constraints: bool = True,
    k_ring: int = 1,
    fdr: bool = True,
) -> Dict[str, object]:
    """Compute Gi* hotspots and global Moran's I for ``value_field`` on ``h3_layer``.

    ``k_ring`` sets the neighbourhood (grid steps); ``fdr`` applies
    Benjamini-Hochberg false-discovery-rate control to the confidence bins.
    Writes ``gi_z``, ``gi_p``, ``hotspot`` (-3..3) and ``hotspot_lbl`` columns.
    """
    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    fields = h3_layer.fields()
    idx_val = fields.indexFromName(value_field)
    if idx_val < 0:
        raise ValueError(f"Field '{value_field}' not found. Forge a composite map first.")
    idx_h3 = fields.indexFromName("h3_id")
    idx_mask = fields.indexFromName("constraint_mask")

    fids: List[int] = []
    h3_ids: List[str] = []
    vals: List[float] = []
    valid: List[bool] = []
    for feat in h3_layer.getFeatures():
        fids.append(int(feat.id()))
        h3_ids.append(str(feat.attribute(idx_h3)) if idx_h3 >= 0 else str(feat.id()))
        v = feat.attribute(idx_val)
        try:
            fv = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            fv = float("nan")
        vals.append(fv)
        keep = np.isfinite(fv)
        if respect_constraints and idx_mask >= 0:
            mv = feat.attribute(idx_mask)
            try:
                if mv is not None and int(mv) == 0:
                    keep = False
            except (TypeError, ValueError):
                pass
        valid.append(bool(keep))

    if not fids:
        raise ValueError("H3 layer contains no features.")

    nbrs = build_neighbor_lists(h3_ids, valid=valid, k=k_ring)
    moran = global_morans_i(vals, nbrs, valid=valid)
    z, p, codes = getis_ord_gi_star(vals, nbrs, valid=valid)
    if fdr:
        codes = classify_hotspots_fdr(z, p, valid=valid)

    idx_z = ensure_layer_field(h3_layer, "gi_z", kind="double")
    idx_p = ensure_layer_field(h3_layer, "gi_p", kind="double")
    idx_c = ensure_layer_field(h3_layer, "hotspot", kind="int")
    idx_l = ensure_layer_field(h3_layer, "hotspot_lbl", kind="string", length=24)

    updates: Dict[int, Dict[int, object]] = {}
    for i, fid in enumerate(fids):
        lbl = HOTSPOT_LABELS[int(codes[i])] if valid[i] else EXCLUDED_LABEL
        updates[fid] = {
            idx_z: round(float(z[i]), 4),
            idx_p: round(float(p[i]), 5),
            idx_c: int(codes[i]),
            idx_l: lbl,
        }
    h3_layer.dataProvider().changeAttributeValues(updates)

    result: Dict[str, object] = {
        "value_field": value_field,
        "k_ring": int(k_ring),
        "fdr": bool(fdr),
        "moran": {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                  for k, v in moran.items()},
        "counts": summarize_hotspots(codes, valid=valid),
        "n_valid": int(sum(valid)),
    }
    # Persist for the HTML report (JSON-safe: NaN replaced by null above).
    h3_layer.setCustomProperty("h3forge/hotspot", json.dumps(result))
    return result
