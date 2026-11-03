# -*- coding: utf-8 -*-
"""Graduated Multi-Ring Proximity Buffer & Critical Threshold Scorer for 02H3Forge.

Allows defining custom distance brackets (e.g. 0-100m, 100-250m, 250-500m)
around infrastructure or hazard layers (power transmission lines, active faults,
streams, highways) and scoring each H3 hexagon by its shortest planar distance.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .h3_grid_engine import ensure_layer_field, sanitize_field_slug

# Ring tuple structure: (min_dist_m, max_dist_m, score_0_100, is_hard_exclusion)
RingSpec = Tuple[float, float, float, bool]

MULTI_RING_PRESETS: Dict[str, List[RingSpec]] = {
    "Power Transmission Line (0-100, 100-250, 250-500 m)": [
        (0.0, 100.0, 100.0, False),
        (100.0, 250.0, 70.0, False),
        (250.0, 500.0, 35.0, False),
    ],
    "Power Line + Strict Safety Exclusion (<50m Excluded)": [
        (0.0, 50.0, 100.0, True),
        (50.0, 150.0, 80.0, False),
        (150.0, 300.0, 40.0, False),
        (300.0, 500.0, 15.0, False),
    ],
    "Active Fault Hazard Zones (0-100, 100-300, 300-1000 m)": [
        (0.0, 100.0, 100.0, False),
        (100.0, 300.0, 75.0, False),
        (300.0, 1000.0, 40.0, False),
    ],
    "Stream / Flood Riparian Buffer (0-50, 50-150, 150-400 m)": [
        (0.0, 50.0, 100.0, False),
        (50.0, 150.0, 65.0, False),
        (150.0, 400.0, 30.0, False),
    ],
    "Road / Transit Accessibility Bands (0-200, 200-500, 500-1000 m)": [
        (0.0, 200.0, 100.0, False),
        (200.0, 500.0, 75.0, False),
        (500.0, 1000.0, 40.0, False),
    ],
}


def validate_rings(rings: Sequence[RingSpec]) -> List[RingSpec]:
    """Validate and sort multi-ring buffer specifications by lower distance."""
    cleaned: List[RingSpec] = []
    for item in rings:
        if len(item) < 3:
            continue
        low = max(0.0, float(item[0]))
        high = float(item[1])
        if high <= low:
            raise ValueError(f"Ring upper bound ({high:g} m) must be greater than lower bound ({low:g} m).")
        score = float(np.clip(float(item[2]), 0.0, 100.0))
        excl = bool(item[3]) if len(item) >= 4 else False
        cleaned.append((low, high, score, excl))
    cleaned.sort(key=lambda r: (r[0], r[1]))
    return cleaned


def evaluate_graduated_rings(
    distances: Sequence[float] | np.ndarray,
    rings: Sequence[RingSpec],
    outside_score: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate graduated buffer ring scores and hard-exclusion flags for distances.

    Parameters
    ----------
    distances : array-like of float
        Shortest distance (m) from each H3 hexagon to the target feature set.
    rings : sequence of (min_m, max_m, score_0_100, is_hard_exclusion)
        Graduated distance brackets. Interval matching uses:
        ``min_m <= d <= max_m`` for the first ring, and ``min_m < d <= max_m``
        when ``min_m > 0`` if an earlier ring already claimed ``min_m``.
    outside_score : float
        Score assigned to hexagons that do not fall into any ring (default 0.0).

    Returns
    -------
    scores : np.ndarray of float64 (clipped 0..100)
    excluded : np.ndarray of bool (True where hexagon is excluded by a hard constraint ring)
    """
    d_arr = np.asarray(distances, dtype=np.float64)
    out_val = float(np.clip(outside_score, 0.0, 100.0))
    scores = np.full(d_arr.shape, out_val, dtype=np.float64)
    excluded = np.zeros(d_arr.shape, dtype=bool)
    assigned = np.zeros(d_arr.shape, dtype=bool)

    sorted_rings = validate_rings(rings)
    for low, high, sc, is_excl in sorted_rings:
        in_ring = (~assigned) & np.isfinite(d_arr) & (d_arr >= low) & (d_arr <= high)
        scores[in_ring] = sc
        if is_excl:
            excluded[in_ring] = True
        assigned |= in_ring

    return scores, excluded


def apply_multiring_buffer_to_h3(
    h3_layer,
    source_layer,
    rings: Sequence[RingSpec],
    criterion_name: str,
    outside_score: float = 0.0,
    measure_from: str = "boundary",
) -> Tuple[str, str, int]:
    """Compute shortest planar distance and graduated ring score on ``h3_layer``.

    Writes two columns onto ``h3_layer``:
      - ``dist_<slug>``: shortest Euclidean distance in meters
      - ``ring_<slug>``: graduated threshold score in [0, 100]
    Also sets ``constraint_mask = 0`` for hexagons falling inside any ring with
    ``is_hard_exclusion = True``.

    Returns
    -------
    (dist_field_name, ring_field_name, excluded_count)
    """
    from .compute import distance_array, snapshot_hex_layer, snapshot_vector_source

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if source_layer is None or not source_layer.isValid():
        raise ValueError("A valid target vector layer is required for multi-ring buffering.")

    validated_rings = validate_rings(rings)
    if not validated_rings:
        raise ValueError("At least one valid distance ring [min_m, max_m] is required.")

    dist_field = sanitize_field_slug(criterion_name, prefix="dist")
    ring_field = sanitize_field_slug(criterion_name, prefix="ring")

    snap = snapshot_hex_layer(h3_layer)
    src = snapshot_vector_source(source_layer, h3_layer.crs())
    if not src.geoms:
        raise ValueError(f"Source layer '{source_layer.name()}' contains no valid geometries.")
    if measure_from == "centroid":
        dists = distance_array([g.centroid() for g in snap.geoms], src.geoms)
    else:
        dists = distance_array(snap.geoms, src.geoms)

    scores_arr, excl_arr = evaluate_graduated_rings(dists, validated_rings, outside_score=outside_score)
    prev_mask = snap.numeric.get("constraint_mask", np.ones(snap.size))

    idx_dist = ensure_layer_field(h3_layer, dist_field, kind="double")
    idx_ring = ensure_layer_field(h3_layer, ring_field, kind="double")
    idx_mask = ensure_layer_field(h3_layer, "constraint_mask", kind="int")
    attr_updates: Dict[int, Dict[int, object]] = {}
    for i, fid in enumerate(snap.fids):
        attr_updates[fid] = {
            idx_dist: round(float(dists[i]), 2) if np.isfinite(dists[i]) else -1.0,
            idx_ring: round(float(scores_arr[i]), 2),
            idx_mask: 0 if (excl_arr[i] or prev_mask[i] == 0) else 1,
        }
    if attr_updates:
        h3_layer.dataProvider().changeAttributeValues(attr_updates)
    return dist_field, ring_field, int(np.sum(excl_arr))
