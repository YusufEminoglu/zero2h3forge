# -*- coding: utf-8 -*-
"""Model execution pipeline for 02H3Forge.

Three phases, so heavy work can run in a background ``QgsTask``:

1. :func:`prepare_run` (main thread) — snapshot the H3 layer and every source
   layer the model needs (geometries already in the H3 CRS, provider clones
   for rasters), and decide which raw criterion columns can be reused from
   the cache.
2. :func:`compute_run` (any thread) — raw criteria, value functions, rule-based
   constraints, hierarchical composition, explainability, Monte Carlo,
   classification, hotspots and sensitivity. Pure NumPy + geometry copies.
3. :func:`apply_run` (main thread) — write all columns and provenance to the
   layer in one batch and style it.

The hierarchy maths (:func:`compose_hierarchy` and friends) has no QGIS
dependency and is unit-tested directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compute import (
    Cancelled,
    centroid_within_array,
    distance_array,
    raster_zonal_array,
    snapshot_hex_layer,
    snapshot_raster_source,
    snapshot_vector_source,
    vector_metric_array,
)
from .mcda_composer import (
    _class_ids_with_breaks,
    apply_h3_spatial_smoothing,
    classify_scores_5_bands,
    compose_scores,
)
from .model import RULE_OPERATORS, ModelSpec, ValueFunction
from .normalizer import normalize_array
from .proximity_rings import evaluate_graduated_rings
from .spatial_stats import (
    EXCLUDED_LABEL,
    HOTSPOT_LABELS,
    build_neighbor_lists,
    classify_hotspots_fdr,
    getis_ord_gi_star,
    global_morans_i,
    summarize_hotspots,
)

ProgressFn = Optional[Callable[[float], None]]
CancelFn = Optional[Callable[[], bool]]


# ---------------------------------------------------------------------------
# 1. Value functions
# ---------------------------------------------------------------------------

def apply_value_function(raw: np.ndarray, vf: ValueFunction) -> np.ndarray:
    """Map raw criterion values to 0-100 scores (NaN raw -> 0)."""
    x = np.asarray(raw, dtype=np.float64)
    if vf.method == "rings":
        rings = [(float(r[0]), float(r[1]), float(r[2]), False) for r in vf.rings]
        scores, _ = evaluate_graduated_rings(x, rings, outside_score=vf.outside_score)
        return np.where(np.isfinite(x), scores, 0.0)
    return normalize_array(
        x, polarity=vf.polarity, method=vf.method,
        low=vf.low, high=vf.high, mid=vf.mid, spread=vf.spread,
    )


# ---------------------------------------------------------------------------
# 2. Hierarchy maths (pure)
# ---------------------------------------------------------------------------

@dataclass
class GroupPlan:
    gid: str
    label: str
    weight: float
    method: str
    owa_alpha: float
    crit_ids: List[str]
    crit_labels: List[str]
    crit_weights: List[float]


def plans_from_model(model: ModelSpec) -> List[GroupPlan]:
    plans: List[GroupPlan] = []
    for g in model.active_groups():
        crits = model.criteria_in(g.id, active_only=True)
        plans.append(GroupPlan(
            gid=g.id, label=g.label, weight=max(0.0, float(g.weight)), method=g.method,
            owa_alpha=float(g.owa_alpha), crit_ids=[c.id for c in crits],
            crit_labels=[c.label for c in crits],
            crit_weights=[max(0.0, float(c.weight)) for c in crits],
        ))
    return plans


def _norm_w(w: Sequence[float]) -> np.ndarray:
    a = np.maximum(0.0, np.asarray(w, dtype=np.float64))
    s = float(a.sum())
    return a / s if s > 1e-12 else np.full(a.size, 1.0 / max(1, a.size))


def aggregate_groups(
    G: np.ndarray, weights: Sequence[float], aggregation: str = "weighted_sum"
) -> np.ndarray:
    """Combine (M, K) group indices (0-100) into one 0-100 score.

    ``weighted_sum``: Σ w_k·G_k (compensatory).
    ``geometric``: 100·Π (G_k/100)^w_k — a zero in any group (e.g. no hazard)
    yields zero, which is the defining property of Risk = H × E × V.
    """
    G = np.clip(np.nan_to_num(np.asarray(G, dtype=np.float64)), 0.0, 100.0)
    if G.ndim == 1:
        G = G[:, None]
    w = _norm_w(weights)
    if G.shape[1] == 1:
        return G[:, 0].copy()
    if aggregation == "geometric":
        logs = np.log(np.maximum(G / 100.0, 1e-9))
        out = 100.0 * np.exp(logs @ w)
        return np.where(np.any(G <= 1e-9, axis=1), 0.0, np.clip(out, 0.0, 100.0))
    return np.clip(G @ w, 0.0, 100.0)


def compose_hierarchy(
    norm: Dict[str, np.ndarray],
    plans: List[GroupPlan],
    aggregation: str = "weighted_sum",
    mask: Optional[np.ndarray] = None,
    group_weights: Optional[Sequence[float]] = None,
    crit_weights: Optional[List[Sequence[float]]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return ``(composite, {group_id: sub_index})`` for the given plans.

    ``group_weights`` / ``crit_weights`` override the plan weights (used by
    Monte Carlo and sensitivity runs).
    """
    if not plans:
        raise ValueError("The model has no active criteria.")
    m = next(iter(norm.values())).size
    idx: Dict[str, np.ndarray] = {}
    for k, p in enumerate(plans):
        X = np.column_stack([norm[c] for c in p.crit_ids]) if p.crit_ids else np.zeros((m, 1))
        cw = crit_weights[k] if crit_weights is not None else p.crit_weights
        method = p.method
        idx[p.gid] = compose_scores(
            X, list(_norm_w(cw) * 100.0), method=method, constraint_mask=mask,
            owa_alpha=p.owa_alpha if method.startswith("owa") else None,
        )
    gw = group_weights if group_weights is not None else [p.weight for p in plans]
    comp = aggregate_groups(np.column_stack([idx[p.gid] for p in plans]), gw, aggregation)
    if mask is not None:
        comp = comp * (np.asarray(mask) > 0.5)
    return comp, idx


def hierarchical_drivers(
    norm: Dict[str, np.ndarray],
    plans: List[GroupPlan],
    group_idx: Dict[str, np.ndarray],
    aggregation: str,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Local explainability per hexagon.

    - ``top_group``: group with the largest weighted contribution w_g·I_g
    - ``limit_group``: weakest group (the factor capping a geometric index)
    - ``top_driver`` / ``driver_share``: criterion with the largest effective
      contribution (w_g · w_c · score) and its share of all contributions
    """
    gw = _norm_w([p.weight for p in plans])
    m = next(iter(norm.values())).size
    contribs: List[np.ndarray] = []
    names: List[str] = []
    for k, p in enumerate(plans):
        cw = _norm_w(p.crit_weights)
        for j, cid in enumerate(p.crit_ids):
            contribs.append(gw[k] * cw[j] * np.clip(norm[cid], 0.0, 100.0))
            names.append(p.crit_labels[j])
    C = np.column_stack(contribs) if contribs else np.zeros((m, 1))
    tot = C.sum(axis=1)
    top = np.argmax(C, axis=1)
    share = np.where(tot > 1e-9, C[np.arange(m), top] / np.maximum(tot, 1e-12) * 100.0, 0.0)
    G = np.column_stack([group_idx[p.gid] for p in plans])
    top_g = np.argmax(G * gw[None, :], axis=1)
    low_g = np.argmin(G, axis=1)
    excluded = (np.asarray(mask) <= 0.5) if mask is not None else np.zeros(m, dtype=bool)
    top_driver = [
        "Excluded" if excluded[i] else ("None" if tot[i] <= 1e-9 else names[int(top[i])])
        for i in range(m)
    ]
    labels = [p.label for p in plans]
    return {
        "top_driver": top_driver,
        "driver_share": np.where(excluded, 0.0, np.round(share, 1)),
        "top_group": ["Excluded" if excluded[i] else labels[int(top_g[i])] for i in range(m)],
        "limit_group": ["Excluded" if excluded[i] else labels[int(low_g[i])] for i in range(m)],
    }


def _perturbed(weights: Sequence[float], rng, p: float) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return w * rng.uniform(1.0 - p, 1.0 + p, size=w.size)


def hierarchical_montecarlo(
    norm: Dict[str, np.ndarray],
    plans: List[GroupPlan],
    aggregation: str,
    mask: Optional[np.ndarray] = None,
    runs: int = 50,
    perturbation: float = 0.15,
    risk_aversion: float = 0.5,
    seed: int = 42,
    cancelled: CancelFn = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perturb group *and* criterion weights; return ``(mean, std, robust)``."""
    rng = np.random.default_rng(int(seed))
    sims = []
    for _ in range(max(5, int(runs))):
        if cancelled is not None and cancelled():
            raise Cancelled()
        gw = _perturbed([p.weight for p in plans], rng, perturbation)
        cw = [_perturbed(p.crit_weights, rng, perturbation) for p in plans]
        comp, _ = compose_hierarchy(norm, plans, aggregation, mask, gw, cw)
        sims.append(comp)
    S = np.column_stack(sims)
    mean, std = S.mean(axis=1), S.std(axis=1)
    rob = np.clip(mean - risk_aversion * std, 0.0, 100.0)
    if mask is not None:
        keep = np.asarray(mask) > 0.5
        mean, std, rob = mean * keep, std * keep, rob * keep
    return np.round(mean, 2), np.round(std, 2), np.round(rob, 2)


def hierarchical_sensitivity(
    norm: Dict[str, np.ndarray],
    plans: List[GroupPlan],
    aggregation: str,
    mask: Optional[np.ndarray] = None,
    breaks: Optional[Sequence[float]] = None,
    deltas_pct: Sequence[float] = (-20.0, -10.0, 10.0, 20.0),
) -> List[Dict[str, object]]:
    """One-at-a-time sensitivity for every group weight and criterion weight."""
    b = list(breaks) if breaks is not None and len(breaks) == 6 else [0, 20, 40, 60, 80, 100]
    base, _ = compose_hierarchy(norm, plans, aggregation, mask)
    base_cls = _class_ids_with_breaks(base, b)
    feas = (np.asarray(mask) > 0.5) if mask is not None else np.ones(base.size, dtype=bool)
    if not np.any(feas):
        feas = np.ones(base.size, dtype=bool)

    rows: List[Dict[str, object]] = []
    targets: List[Tuple[str, str, int, int]] = []
    if len(plans) > 1:
        targets += [("group", p.label, k, -1) for k, p in enumerate(plans)]
    for k, p in enumerate(plans):
        if len(p.crit_ids) > 1:
            targets += [("criterion", p.crit_labels[j], k, j) for j in range(len(p.crit_ids))]
    g0 = [p.weight for p in plans]
    for level, label, k, j in targets:
        worst = {"mean": 0.0, "max": 0.0, "cls": 0.0}
        for d in deltas_pct:
            gw = list(g0)
            cw = [list(p.crit_weights) for p in plans]
            vec = gw if j < 0 else cw[k]
            pos = k if j < 0 else j
            share = _norm_w(vec)
            new = float(np.clip(share[pos] * (1.0 + d / 100.0), 0.0, 1.0))
            others = 1.0 - share[pos]
            scaled = [
                new if i == pos else (share[i] * (1.0 - new) / others if others > 1e-12 else (1.0 - new) / (len(vec) - 1))
                for i in range(len(vec))
            ]
            if j < 0:
                gw = scaled
            else:
                cw[k] = scaled
            comp, _ = compose_hierarchy(norm, plans, aggregation, mask, gw, cw)
            diff = np.abs(comp - base)[feas]
            changed = (_class_ids_with_breaks(comp, b) != base_cls)[feas]
            worst["mean"] = max(worst["mean"], float(diff.mean()))
            worst["max"] = max(worst["max"], float(diff.max()))
            worst["cls"] = max(worst["cls"], float(changed.mean() * 100.0))
        weight = (_norm_w(g0)[k] if j < 0 else _norm_w(plans[k].crit_weights)[j]) * 100.0
        rows.append({
            "level": level,
            "criterion": label if level == "criterion" else f"[{label}]",
            "group": plans[k].label,
            "weight": round(float(weight), 2),
            "mean_abs_change": round(worst["mean"], 3),
            "max_abs_change": round(worst["max"], 3),
            "class_change_pct": round(worst["cls"], 2),
        })
    rows.sort(key=lambda r: (-float(r["class_change_pct"]), -float(r["mean_abs_change"])))
    return rows


# ---------------------------------------------------------------------------
# 3. Run inputs / outputs
# ---------------------------------------------------------------------------

@dataclass
class RunInputs:
    model: ModelSpec
    snap: object                                       # HexSnapshot
    vectors: Dict[str, object] = field(default_factory=dict)     # key -> VectorSource
    rasters: Dict[str, object] = field(default_factory=dict)     # key -> RasterSource
    reuse_raw: Dict[str, np.ndarray] = field(default_factory=dict)  # cid -> cached raw
    signatures: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    columns: Dict[str, Tuple[str, object]] = field(default_factory=dict)  # name -> (kind, values)
    properties: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


def default_layer_resolver(spec):
    """Find the QGIS layer a SourceSpec points to (id → name → uri)."""
    from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

    proj = QgsProject.instance()
    if spec.layer_id:
        lyr = proj.mapLayer(spec.layer_id)
        if lyr is not None and lyr.isValid():
            return lyr
    if spec.layer_name:
        for lyr in proj.mapLayersByName(spec.layer_name):
            if lyr.isValid():
                return lyr
    if spec.layer_uri:
        if spec.kind == "raster":
            lyr = QgsRasterLayer(spec.layer_uri, spec.layer_name or "raster", spec.provider or "gdal")
        else:
            lyr = QgsVectorLayer(spec.layer_uri, spec.layer_name or "layer", spec.provider or "ogr")
        if lyr.isValid():
            return lyr
    return None


def _source_key(spec, field_name: str = "") -> str:
    return f"{spec.layer_id or spec.layer_uri or spec.layer_name}|{field_name}"


def prepare_run(h3_layer, model: ModelSpec, resolve=default_layer_resolver, force: bool = False) -> RunInputs:
    """Main-thread snapshot of everything the model needs."""
    problems = model.validate()
    if problems:
        raise ValueError("The model is not runnable:\n- " + "\n- ".join(problems))
    snap = snapshot_hex_layer(h3_layer)
    if snap.size == 0:
        raise ValueError("The H3 layer has no hexagons.")
    inputs = RunInputs(model=model.copy(), snap=snap)
    try:
        cache = json.loads(snap.properties.get("h3forge/raw_cache", "{}") or "{}")
    except ValueError:
        cache = {}

    for c in model.criteria:
        sig = c.source.signature()
        inputs.signatures[c.id] = sig
        if c.source.kind == "field":
            if c.source.field not in snap.numeric:
                raise ValueError(f"Criterion '{c.label}': field '{c.source.field}' is not on the H3 layer.")
            inputs.reuse_raw[c.id] = snap.numeric[c.source.field]
            continue
        if not force and cache.get(c.id) == sig and c.raw_field in snap.numeric:
            inputs.reuse_raw[c.id] = snap.numeric[c.raw_field]
            continue
        lyr = resolve(c.source)
        if lyr is None:
            raise ValueError(f"Criterion '{c.label}': source layer '{c.source.layer_name}' not found.")
        if c.source.kind == "raster":
            key = _source_key(c.source)
            if key not in inputs.rasters:
                inputs.rasters[key] = snapshot_raster_source(lyr)
        else:
            fld = c.source.field if c.source.kind == "vector" else ""
            key = _source_key(c.source, fld)
            if key not in inputs.vectors:
                inputs.vectors[key] = snapshot_vector_source(lyr, snap.crs, fld)

    for k in model.constraints:
        if not k.active or k.kind == "rule":
            continue
        lyr = resolve(k.source)
        if lyr is None:
            raise ValueError(f"Constraint '{k.label}': layer '{k.source.layer_name}' not found.")
        key = _source_key(k.source)
        if key not in inputs.vectors:
            inputs.vectors[key] = snapshot_vector_source(lyr, snap.crs)
    return inputs


def _rule(values: np.ndarray, op: str, ref: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        res = {
            "<": v < ref, "<=": v <= ref, ">": v > ref, ">=": v >= ref,
            "==": v == ref, "!=": v != ref,
        }[op]
    return np.where(np.isfinite(v), res, False)


def compute_run(inputs: RunInputs, progress: ProgressFn = None, cancelled: CancelFn = None) -> RunResult:
    """Thread-safe evaluation of the whole model."""
    model, snap = inputs.model, inputs.snap
    res = RunResult()
    n_raw = sum(1 for c in model.criteria if c.id not in inputs.reuse_raw) or 1
    done = [0]

    def _sub(lo: float, hi: float) -> ProgressFn:
        if progress is None:
            return None
        return lambda pct: progress(lo + (hi - lo) * pct / 100.0)

    def _check():
        if cancelled is not None and cancelled():
            raise Cancelled()

    # --- raw criteria (0-60 %) ---------------------------------------------
    raw: Dict[str, np.ndarray] = {}
    for c in model.criteria:
        _check()
        if c.id in inputs.reuse_raw:
            raw[c.id] = np.asarray(inputs.reuse_raw[c.id], dtype=np.float64)
            continue
        lo = 60.0 * done[0] / n_raw
        hi = 60.0 * (done[0] + 1) / n_raw
        if c.source.kind == "raster":
            src = inputs.rasters[_source_key(c.source)]
            arr = raster_zonal_array(snap.geoms, snap.crs, src, c.source.method or "mean",
                                     int(c.source.band or 1), cancelled=cancelled, log=res.log.append)
        elif c.source.kind == "distance":
            src = inputs.vectors[_source_key(c.source)]
            arr = distance_array(snap.geoms, src.geoms, _sub(lo, hi), cancelled)
        else:
            src = inputs.vectors[_source_key(c.source, c.source.field)]
            arr = vector_metric_array(snap.geoms, src, c.source.method or "count",
                                      progress=_sub(lo, hi), cancelled=cancelled)
        raw[c.id] = arr
        res.columns[c.raw_field] = ("double", np.round(np.where(np.isfinite(arr), arr, -1.0), 4))
        res.log.append(f"computed {c.raw_field} ({c.source.kind}: {c.source.layer_name or c.source.field})")
        done[0] += 1
        if progress is not None:
            progress(hi)

    # --- value functions --------------------------------------------------
    norm: Dict[str, np.ndarray] = {}
    for c in model.criteria:
        norm[c.id] = np.round(apply_value_function(raw[c.id], c.value), 2)
        res.columns[c.norm_field] = ("double", norm[c.id])
    if progress is not None:
        progress(62.0)

    # --- constraints --------------------------------------------------------
    m = snap.size
    mask = np.ones(m, dtype=np.int32)
    lookup = dict(snap.numeric)
    for c in model.criteria:
        lookup[c.raw_field] = raw[c.id]
        lookup[c.norm_field] = norm[c.id]
    excluded_by: Dict[str, int] = {}
    for k in model.constraints:
        _check()
        if not k.active:
            continue
        if k.kind == "rule":
            if k.field not in lookup:
                res.log.append(f"constraint '{k.label}' skipped: field '{k.field}' not found")
                continue
            hit = _rule(lookup[k.field], k.op, float(k.value))
        else:
            src = inputs.vectors[_source_key(k.source)]
            if k.kind == "near":
                hit = distance_array(snap.geoms, src.geoms, None, cancelled) < float(k.distance)
            else:
                inside = centroid_within_array(snap.geoms, src.geoms, cancelled)
                hit = inside if k.kind == "inside" else ~inside
        excluded_by[k.label] = int(np.sum(hit))
        mask[hit] = 0
    res.columns["constraint_mask"] = ("int", mask)
    active_mask = mask if model.apply_constraints else None
    if progress is not None:
        progress(70.0)

    # --- composition --------------------------------------------------------
    plans = plans_from_model(model)
    comp, gidx = compose_hierarchy(norm, plans, model.aggregation, active_mask)
    multi = len(plans) > 1
    if multi:
        for p in plans:
            res.columns[f"idx_{p.gid}"] = ("double", np.round(gidx[p.gid], 2))
    if model.smooth_alpha > 1e-6:
        comp = apply_h3_spatial_smoothing(comp, snap.h3_ids, alpha=model.smooth_alpha,
                                          constraint_mask=active_mask)
    drivers = hierarchical_drivers(norm, plans, gidx, model.aggregation, active_mask)
    cids, clabels, breaks = classify_scores_5_bands(comp, method=model.class_mode, theme=model.theme)
    res.columns.update({
        "comp_score": ("double", np.round(comp, 2)),
        "class_id": ("int", np.asarray(cids, dtype=np.int32)),
        "class_label": ("string", clabels),
        "top_driver": ("string", drivers["top_driver"]),
        "driver_share": ("double", drivers["driver_share"]),
    })
    if multi:
        res.columns["top_group"] = ("string", drivers["top_group"])
        if model.aggregation == "geometric":
            res.columns["limit_group"] = ("string", drivers["limit_group"])
    if progress is not None:
        progress(75.0)

    # --- Monte Carlo --------------------------------------------------------
    if model.run_montecarlo:
        _, mc_std, rob = hierarchical_montecarlo(
            norm, plans, model.aggregation, active_mask, model.mc_runs, model.mc_perturbation,
            cancelled=cancelled,
        )
    else:
        mc_std, rob = np.zeros(m), np.round(comp, 2)
    res.columns["mc_std"] = ("double", mc_std)
    res.columns["rob_score"] = ("double", rob)
    if progress is not None:
        progress(85.0)

    feas = (mask > 0) if model.apply_constraints else np.ones(m, dtype=bool)
    if not np.any(feas):
        feas = np.ones(m, dtype=bool)

    # --- hotspots -----------------------------------------------------------
    if model.run_hotspots and m >= 3:
        _check()
        valid = feas & np.isfinite(comp)
        nbrs = build_neighbor_lists(snap.h3_ids, valid=valid, k=model.hotspot_k)
        moran = global_morans_i(comp, nbrs, valid=valid)
        z, p, codes = getis_ord_gi_star(comp, nbrs, valid=valid)
        if model.hotspot_fdr:
            codes = classify_hotspots_fdr(z, p, valid=valid)
        res.columns.update({
            "gi_z": ("double", np.round(z, 4)),
            "gi_p": ("double", np.round(p, 5)),
            "hotspot": ("int", codes.astype(np.int32)),
            "hotspot_lbl": ("string", [
                HOTSPOT_LABELS[int(codes[i])] if valid[i] else EXCLUDED_LABEL for i in range(m)
            ]),
        })
        res.properties["h3forge/hotspot"] = json.dumps({
            "value_field": "comp_score", "k_ring": int(model.hotspot_k), "fdr": bool(model.hotspot_fdr),
            "moran": {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in moran.items()},
            "counts": summarize_hotspots(codes, valid=valid), "n_valid": int(valid.sum()),
        })
        res.stats["moran"] = moran
    if progress is not None:
        progress(92.0)

    # --- sensitivity ----------------------------------------------------------
    if model.run_sensitivity:
        _check()
        rows = hierarchical_sensitivity(norm, plans, model.aggregation, active_mask, breaks)
        res.properties["h3forge/sensitivity"] = json.dumps({
            "method": model.aggregation if multi else plans[0].method,
            "deltas_pct": [-20.0, -10.0, 10.0, 20.0], "rows": rows,
        })
        res.stats["sensitivity"] = rows

    # --- provenance -----------------------------------------------------------
    eff = model.effective_weights()
    res.properties["h3forge/model"] = model.to_json()
    res.properties["h3forge/last_run"] = json.dumps({
        "method": plans[0].method if not multi else f"{model.aggregation} of {len(plans)} groups",
        "owa_alpha": plans[0].owa_alpha if (not multi and plans[0].method.startswith("owa")) else None,
        "theme": model.theme, "class_method": model.class_mode,
        "breaks": [round(float(b), 3) for b in breaks], "alpha": float(model.smooth_alpha),
        "constraints": bool(model.apply_constraints), "montecarlo": bool(model.run_montecarlo),
        "mc_runs": int(model.mc_runs), "mc_perturbation": float(model.mc_perturbation),
        "weights": {model.criterion(cid).norm_field: round(w, 4) for cid, w in eff.items()},
        "aggregation": model.aggregation,
        "groups": [{"id": p.gid, "label": p.label, "weight": round(float(w) * 100, 2), "method": p.method}
                   for p, w in zip(plans, _norm_w([p.weight for p in plans]))],
        "excluded_by": excluded_by,
    })
    res.properties["h3forge/raw_cache"] = json.dumps(
        {c.id: inputs.signatures[c.id] for c in model.criteria if c.source.kind != "field"}
    )
    res.stats.update({
        "feature_count": m,
        "excluded_count": int(np.sum(mask == 0)),
        "mean_score": round(float(np.mean(comp[feas])), 2),
        "max_score": round(float(np.max(comp)), 2),
        "mean_mc_std": round(float(np.mean(mc_std[feas])), 2),
        "breaks": breaks,
        "computed_raw": [c.raw_field for c in model.criteria if c.id not in inputs.reuse_raw],
        "group_means": {p.label: round(float(np.mean(gidx[p.gid][feas])), 2) for p in plans},
    })
    if progress is not None:
        progress(100.0)
    return res


def apply_run(h3_layer, result: RunResult, snap=None, style: bool = True) -> None:
    """Write a :class:`RunResult` onto ``h3_layer`` (main thread)."""
    from .h3_grid_engine import ensure_layer_field
    from .styler_and_report import apply_h3_graduated_style

    fids = snap.fids if snap is not None else [int(f.id()) for f in h3_layer.getFeatures()]
    idx: Dict[str, int] = {}
    for name, (kind, _) in result.columns.items():
        idx[name] = ensure_layer_field(h3_layer, name, kind=kind, length=48)
    updates: Dict[int, Dict[int, object]] = {fid: {} for fid in fids}
    for name, (kind, values) in result.columns.items():
        col = idx[name]
        for i, fid in enumerate(fids):
            v = values[i]
            if kind == "string":
                updates[fid][col] = str(v)
            elif kind == "int":
                updates[fid][col] = int(v)
            else:
                fv = float(v)
                updates[fid][col] = fv if np.isfinite(fv) else None
    h3_layer.dataProvider().changeAttributeValues(updates)
    for key, val in result.properties.items():
        h3_layer.setCustomProperty(key, val)
    if style and "comp_score" in result.columns:
        theme = "risk"
        try:
            theme = json.loads(result.properties.get("h3forge/last_run", "{}")).get("theme", "risk")
        except ValueError:
            pass
        apply_h3_graduated_style(h3_layer, "comp_score", theme=theme, breaks=result.stats.get("breaks"))
    h3_layer.triggerRepaint()


def run_model_on_layer(h3_layer, model: ModelSpec, resolve=default_layer_resolver,
                       force: bool = False, progress: ProgressFn = None,
                       cancelled: CancelFn = None) -> RunResult:
    """Synchronous convenience runner (Processing, tests, scripts)."""
    inputs = prepare_run(h3_layer, model, resolve, force)
    result = compute_run(inputs, progress, cancelled)
    apply_run(h3_layer, result, inputs.snap)
    return result


__all__ = [
    "Cancelled", "GroupPlan", "RunInputs", "RunResult", "aggregate_groups", "apply_run",
    "apply_value_function", "compose_hierarchy", "compute_run", "default_layer_resolver",
    "hierarchical_drivers", "hierarchical_montecarlo", "hierarchical_sensitivity",
    "plans_from_model", "prepare_run", "run_model_on_layer", "RULE_OPERATORS",
]
