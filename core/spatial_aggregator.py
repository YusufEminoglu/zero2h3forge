# -*- coding: utf-8 -*-
"""Multi-Source Spatial Aggregator (Vector Join-by-Location & Raster Zonal Stats) for 02H3Forge.

Transfers attributes and geometric metrics from Point, Line, Polygon, and Raster
layers onto the active Uber H3 hexagonal grid layer.
"""
from __future__ import annotations

from contextlib import suppress
from typing import Dict, List, Optional, Sequence

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


def aggregate_vector_to_h3(
    h3_layer,
    source_layer,
    method: str = "count",
    field_name: Optional[str] = None,
    output_name: Optional[str] = None,
    default_empty: float = 0.0,
) -> str:
    """Aggregate a Point, Line, or Polygon vector layer into ``h3_layer``.

    Returns the created/updated column name (`raw_<slug>`).
    """
    from qgis.core import (
        QgsCoordinateTransform,
        QgsFeature,
        QgsGeometry,
        QgsProject,
        QgsSpatialIndex,
        QgsWkbTypes,
    )

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if source_layer is None or not source_layer.isValid():
        raise ValueError("A valid source vector layer is required.")

    raw_label = output_name or f"{source_layer.name()}_{method}"
    col_name = sanitize_field_slug(raw_label, prefix="raw")
    idx_out = ensure_layer_field(h3_layer, col_name, kind="double")

    target_crs = h3_layer.crs()
    tr = None
    if source_layer.crs() != target_crs:
        tr = QgsCoordinateTransform(source_layer.crs(), target_crs, QgsProject.instance())

    src_fld_idx = -1
    if field_name:
        src_fld_idx = source_layer.fields().indexFromName(field_name)

    spatial_index = QgsSpatialIndex()
    src_geoms: Dict[int, QgsGeometry] = {}
    src_vals: Dict[int, Optional[float]] = {}
    src_areas: Dict[int, float] = {}

    geom_type = source_layer.geometryType()
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
        if geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
            src_areas[fid] = max(1e-6, float(g_copy.area()))
        val: Optional[float] = None
        if src_fld_idx >= 0:
            raw_v = s_feat.attribute(src_fld_idx)
            if raw_v is not None:
                with suppress(ValueError, TypeError):
                    fv = float(raw_v)
                    if np.isfinite(fv):
                        val = fv
        src_vals[fid] = val

        f_idx = QgsFeature(fid)
        f_idx.setGeometry(g_copy)
        spatial_index.addFeature(f_idx)

    attr_updates: Dict[int, Dict[int, object]] = {}
    for h_feat in h3_layer.getFeatures():
        h_geom = h_feat.geometry()
        if h_geom is None or h_geom.isEmpty():
            continue
        hfid = int(h_feat.id())
        hex_area_m2 = max(1.0, float(h_geom.area()))
        hex_area_ha = hex_area_m2 / 10_000.0

        if method == "nearest_dist_m":
            if not src_geoms:
                attr_updates[hfid] = {idx_out: float(default_empty)}
                continue
            pt_c = h_geom.centroid().asPoint()
            near_ids = spatial_index.nearestNeighbor(pt_c, 8) or list(src_geoms.keys())
            min_d = float("inf")
            for cid in near_ids:
                sg = src_geoms.get(cid)
                if sg is not None:
                    d = float(h_geom.distance(sg))
                    if d < min_d:
                        min_d = d
            if min_d > 0.0:
                for cid in spatial_index.intersects(h_geom.boundingBox()):
                    sg = src_geoms.get(cid)
                    if sg is not None:
                        d = float(h_geom.distance(sg))
                        if d < min_d:
                            min_d = d
            attr_updates[hfid] = {
                idx_out: round(min_d, 2) if np.isfinite(min_d) else float(default_empty)
            }
            continue

        candidate_ids = spatial_index.intersects(h_geom.boundingBox())
        hit_ids: List[int] = []
        inter_weights: List[float] = []
        hit_vals: List[float] = []
        apportioned_vals: List[float] = []

        for cid in candidate_ids:
            sg = src_geoms.get(cid)
            if sg is None or not h_geom.intersects(sg):
                continue
            hit_ids.append(cid)
            v = src_vals.get(cid)

            if geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
                inter_g = h_geom.intersection(sg)
                inter_a = float(inter_g.area()) if (inter_g is not None and not inter_g.isEmpty()) else 0.0
                if inter_a <= 0.0:
                    continue
                inter_weights.append(inter_a)
                if v is not None:
                    hit_vals.append(v)
                    orig_a = src_areas.get(cid, hex_area_m2)
                    apportioned_vals.append(v * (inter_a / max(1e-6, orig_a)))
            elif geom_type == QgsWkbTypes.GeometryType.LineGeometry:
                inter_g = h_geom.intersection(sg)
                inter_len = float(inter_g.length()) if (inter_g is not None and not inter_g.isEmpty()) else 0.0
                if inter_len <= 0.0:
                    continue
                inter_weights.append(inter_len)
                if v is not None:
                    hit_vals.append(v)
                    apportioned_vals.append(v * inter_len)
            else:
                inter_weights.append(1.0)
                if v is not None:
                    hit_vals.append(v)
                    apportioned_vals.append(v)

        result_val = float(default_empty)
        if method == "count":
            result_val = float(len(inter_weights))
        elif method == "density_ha":
            if geom_type == QgsWkbTypes.GeometryType.LineGeometry:
                result_val = float(sum(inter_weights)) / max(1e-6, hex_area_ha)
            else:
                result_val = float(len(inter_weights)) / max(1e-6, hex_area_ha)
        elif method == "area_share_pct":
            total_inter = float(sum(inter_weights))
            result_val = float(np.clip((total_inter / hex_area_m2) * 100.0, 0.0, 100.0))
        elif method == "length_m":
            result_val = float(sum(inter_weights))
        elif method == "area_weighted_mean":
            if hit_vals and len(inter_weights) == len(hit_vals) and sum(inter_weights) > 0:
                w_arr = np.asarray(inter_weights, dtype=np.float64)
                v_arr = np.asarray(hit_vals, dtype=np.float64)
                result_val = float(np.sum(w_arr * v_arr) / np.sum(w_arr))
            elif hit_vals:
                result_val = compute_stat(hit_vals, "mean", default=default_empty)
        elif method == "area_weighted_sum":
            result_val = float(sum(apportioned_vals)) if apportioned_vals else float(default_empty)
        else:
            result_val = compute_stat(hit_vals, method, default=default_empty)

        attr_updates[hfid] = {idx_out: round(float(result_val), 4)}

    if attr_updates:
        h3_layer.dataProvider().changeAttributeValues(attr_updates)
    return col_name


def aggregate_raster_to_h3(
    h3_layer,
    raster_layer,
    method: str = "mean",
    band_number: int = 1,
    output_name: Optional[str] = None,
    default_empty: float = 0.0,
) -> str:
    """Sample a raster layer across each H3 hexagon and compute a zonal statistic.

    Uses ``QgsRasterDataProvider.sample()`` across a 13-point hexagonal stencil
    (centroid, 6 mid-radius points, and 6 near-vertex points) so it works on any
    raster provider (GeoTIFF, memory, COG) and any CRS without external files.
    """
    from qgis.core import QgsCoordinateTransform, QgsPointXY, QgsProject

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    if raster_layer is None or not raster_layer.isValid():
        raise ValueError("A valid raster layer is required.")

    raw_label = output_name or f"{raster_layer.name()}_{method}"
    col_name = sanitize_field_slug(raw_label, prefix="raw")
    idx_out = ensure_layer_field(h3_layer, col_name, kind="double")

    r_dp = raster_layer.dataProvider()
    tr = None
    if h3_layer.crs() != raster_layer.crs():
        tr = QgsCoordinateTransform(h3_layer.crs(), raster_layer.crs(), QgsProject.instance())

    attr_updates: Dict[int, Dict[int, object]] = {}
    for h_feat in h3_layer.getFeatures():
        h_geom = h_feat.geometry()
        if h_geom is None or h_geom.isEmpty():
            continue
        hfid = int(h_feat.id())
        c_pt = h_geom.centroid().asPoint()
        sample_pts: List[QgsPointXY] = [c_pt]

        verts = [QgsPointXY(v.x(), v.y()) for v in h_geom.vertices()]
        for v in verts[:-1]:
            mid_x = 0.5 * (c_pt.x() + v.x())
            mid_y = 0.5 * (c_pt.y() + v.y())
            in_x = 0.15 * c_pt.x() + 0.85 * v.x()
            in_y = 0.15 * c_pt.y() + 0.85 * v.y()
            sample_pts.append(QgsPointXY(mid_x, mid_y))
            sample_pts.append(QgsPointXY(in_x, in_y))

        valid_samples: List[float] = []
        for pt in sample_pts:
            rpt = pt
            if tr is not None:
                with suppress(Exception):
                    rpt = tr.transform(pt)
            val, ok = r_dp.sample(rpt, int(band_number))
            if ok and val is not None and np.isfinite(val):
                valid_samples.append(float(val))

        stat_val = compute_stat(valid_samples, method, default=default_empty)
        attr_updates[hfid] = {idx_out: round(float(stat_val), 4)}

    if attr_updates:
        h3_layer.dataProvider().changeAttributeValues(attr_updates)
    return col_name
