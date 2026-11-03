# -*- coding: utf-8 -*-
"""Graduated Multi-Ring Proximity Buffer & Critical Threshold Scorer for 02H3Forge.

Allows defining custom distance brackets (e.g. 0-100m, 100-250m, 250-500m)
around infrastructure or hazard layers (power transmission lines, active faults,
streams, highways) and scoring each H3 hexagon by its shortest planar distance.
"""
from __future__ import annotations

from contextlib import suppress
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
    from qgis.core import (
        QgsCoordinateTransform,
        QgsFeature,
        QgsGeometry,
        QgsProject,
        QgsSpatialIndex,
    )

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if source_layer is None or not source_layer.isValid():
        raise ValueError("A valid target vector layer is required for multi-ring buffering.")

    validated_rings = validate_rings(rings)
    if not validated_rings:
        raise ValueError("At least one valid distance ring [min_m, max_m] is required.")

    dist_field = sanitize_field_slug(criterion_name, prefix="dist")
    ring_field = sanitize_field_slug(criterion_name, prefix="ring")

    idx_dist = ensure_layer_field(h3_layer, dist_field, kind="double")
    idx_ring = ensure_layer_field(h3_layer, ring_field, kind="double")
    idx_mask = ensure_layer_field(h3_layer, "constraint_mask", kind="int")

    target_crs = h3_layer.crs()
    tr = None
    if source_layer.crs() != target_crs:
        tr = QgsCoordinateTransform(source_layer.crs(), target_crs, QgsProject.instance())

    spatial_index = QgsSpatialIndex()
    src_geoms: Dict[int, QgsGeometry] = {}
    for s_feat in source_layer.getFeatures():
        g = s_feat.geometry()
        if g is None or g.isEmpty():
            continue
        g_copy = QgsGeometry(g)
        if tr is not None:
            with suppress(Exception):
                g_copy.transform(tr)
        fid = int(s_feat.id())
        src_geoms[fid] = g_copy
        f_idx = QgsFeature(fid)
        f_idx.setGeometry(g_copy)
        spatial_index.addFeature(f_idx)

    if not src_geoms:
        raise ValueError(f"Source layer '{source_layer.name()}' contains no valid geometries.")

    fids: List[int] = []
    dists: List[float] = []
    curr_masks: List[int] = []

    for h_feat in h3_layer.getFeatures():
        h_geom = h_feat.geometry()
        if h_geom is None or h_geom.isEmpty():
            continue
        fids.append(int(h_feat.id()))
        m_val = h_feat.attribute(idx_mask)
        curr_masks.append(0 if (m_val is not None and int(m_val) == 0) else 1)

        ref_geom = h_geom.centroid() if measure_from == "centroid" else h_geom
        pt_center = h_geom.centroid().asPoint()
        candidate_ids = spatial_index.nearestNeighbor(pt_center, 8)
        if not candidate_ids:
            candidate_ids = list(src_geoms.keys())

        min_d = float("inf")
        for cid in candidate_ids:
            sg = src_geoms.get(cid)
            if sg is not None:
                d = float(ref_geom.distance(sg))
                if d < min_d:
                    min_d = d
        # If polygon/line crosses the hex but nearestNeighbor missed a long segment, verify bbox
        if min_d > 0.0:
            bbox_hits = spatial_index.intersects(h_geom.boundingBox())
            for cid in bbox_hits:
                sg = src_geoms.get(cid)
                if sg is not None:
                    d = float(ref_geom.distance(sg))
                    if d < min_d:
                        min_d = d
                        if min_d == 0.0:
                            break
        dists.append(min_d)

    scores_arr, excl_arr = evaluate_graduated_rings(
        dists,
        validated_rings,
        outside_score=outside_score,
    )

    attr_updates: Dict[int, Dict[int, object]] = {}
    excluded_count = 0
    for i, fid in enumerate(fids):
        d_val = round(float(dists[i]), 2) if np.isfinite(dists[i]) else -1.0
        s_val = round(float(scores_arr[i]), 2)
        new_mask = 0 if (excl_arr[i] or curr_masks[i] == 0) else 1
        if excl_arr[i]:
            excluded_count += 1
        attr_updates[fid] = {
            idx_dist: d_val,
            idx_ring: s_val,
            idx_mask: new_mask,
        }

    if attr_updates:
        h3_layer.dataProvider().changeAttributeValues(attr_updates)
    return dist_field, ring_field, excluded_count
