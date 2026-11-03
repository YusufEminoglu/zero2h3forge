# -*- coding: utf-8 -*-
"""Multi-Source Spatial Aggregator (Vector Join-by-Location & Raster Zonal Stats) for 02H3Forge.

Transfers attributes and geometric metrics from Point, Line, Polygon, and Raster
layers onto the active Uber H3 hexagonal grid layer.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .h3_grid_engine import ensure_layer_field, sanitize_field_slug


VECTOR_AGG_METHODS = [
    ("count", "Feature Count in Hexagon"),
    ("density_ha", "Density per Hectare (count/ha or m/ha)"),
    ("area_share_pct", "Polygon Coverage Share (0-100 pct of Hex)"),
    ("length_m", "Total Clipped Line Length (m)"),
    ("area_weighted_mean", "Area/Length-Weighted Mean of Field"),
    ("area_weighted_sum", "Area-Apportioned Sum of Field (Extensive)"),
    ("mean", "Mean of Field"),
    ("sum", "Sum of Field"),
    ("max", "Maximum of Field"),
    ("min", "Minimum of Field"),
    ("median", "Median of Field"),
    ("nearest_dist_m", "Shortest Distance to Nearest Feature (m)"),
]

RASTER_AGG_METHODS = [
    ("mean", "Zonal Mean"),
    ("median", "Zonal Median"),
    ("max", "Zonal Maximum"),
    ("min", "Zonal Minimum"),
    ("std", "Zonal Standard Deviation"),
    ("range", "Zonal Range (Max - Min)"),
    ("sum", "Zonal Sum"),
]


def compute_stat(values: Sequence[float], method: str, default: float = 0.0) -> float:
    """Compute a summary statistic over a sequence of finite numeric values."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    if method == "sum":
        return float(np.sum(arr))
    if method == "mean":
        return float(np.mean(arr))
    if method == "min":
        return float(np.min(arr))
    if method == "max":
        return float(np.max(arr))
    if method == "median":
        return float(np.median(arr))
    if method == "std":
        return float(np.std(arr))
    if method == "range":
        return float(np.max(arr) - np.min(arr))
    return float(np.mean(arr))


def _write_column(h3_layer, snap, col_name: str, values) -> None:
    idx_out = ensure_layer_field(h3_layer, col_name, kind="double")
    updates = {fid: {idx_out: round(float(v), 4)} for fid, v in zip(snap.fids, values)}
    if updates:
        h3_layer.dataProvider().changeAttributeValues(updates)


def aggregate_vector_to_h3(
    h3_layer,
    source_layer,
    method: str = "count",
    field_name: Optional[str] = None,
    output_name: Optional[str] = None,
    default_empty: float = 0.0,
) -> str:
    """Aggregate a Point, Line, or Polygon vector layer into ``h3_layer``.

    Returns the created/updated column name (``raw_<slug>``).
    """
    from .compute import snapshot_hex_layer, snapshot_vector_source, vector_metric_array

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if source_layer is None or not source_layer.isValid():
        raise ValueError("A valid source vector layer is required.")
    col_name = sanitize_field_slug(output_name or f"{source_layer.name()}_{method}", prefix="raw")
    snap = snapshot_hex_layer(h3_layer)
    src = snapshot_vector_source(source_layer, h3_layer.crs(), field_name or "")
    vals = vector_metric_array(snap.geoms, src, method, default=default_empty)
    _write_column(h3_layer, snap, col_name, vals)
    return col_name


def aggregate_raster_to_h3(
    h3_layer,
    raster_layer,
    method: str = "mean",
    band_number: int = 1,
    output_name: Optional[str] = None,
    default_empty: float = 0.0,
) -> str:
    """Exact pixel-based zonal statistic per H3 hexagon (13-point stencil fallback)."""
    from .compute import raster_zonal_array, snapshot_hex_layer, snapshot_raster_source

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if raster_layer is None or not raster_layer.isValid():
        raise ValueError("A valid raster layer is required.")
    col_name = sanitize_field_slug(output_name or f"{raster_layer.name()}_{method}", prefix="raw")
    snap = snapshot_hex_layer(h3_layer)
    vals = raster_zonal_array(
        snap.geoms, snap.crs, snapshot_raster_source(raster_layer), method, band_number, default_empty
    )
    _write_column(h3_layer, snap, col_name, vals)
    return col_name
