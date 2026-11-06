# -*- coding: utf-8 -*-
"""Thread-safe geometric compute kernels for 02H3Forge.

Design rule: *reading* QGIS layers happens on the main thread through the
``snapshot_*`` functions, which copy geometries (already transformed to the
H3 CRS) and attribute values into plain Python/NumPy containers. The
``*_array`` kernels then work only on those copies (plus raster provider
clones), so they can run inside a ``QgsTask`` without touching live layers.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ProgressFn = Optional[Callable[[float], None]]
CancelFn = Optional[Callable[[], bool]]


class Cancelled(Exception):
    """Raised inside kernels when the user cancels a background run."""


def _tick(i: int, n: int, progress: ProgressFn, cancelled: CancelFn, every: int = 256) -> None:
    if i % every:
        return
    if cancelled is not None and cancelled():
        raise Cancelled()
    if progress is not None and n:
        progress(100.0 * i / n)


# ---------------------------------------------------------------------------
# Snapshots (main thread)
# ---------------------------------------------------------------------------

@dataclass
class HexSnapshot:
    """Geometry + numeric attributes of an H3 layer at one moment."""

    fids: List[int]
    h3_ids: List[str]
    geoms: list                                   # QgsGeometry copies
    crs: object                                   # QgsCoordinateReferenceSystem
    area_ha: np.ndarray
    numeric: Dict[str, np.ndarray] = field(default_factory=dict)
    properties: Dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.fids)


def snapshot_hex_layer(h3_layer) -> HexSnapshot:
    from qgis.core import QgsGeometry

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    fields = h3_layer.fields()
    numeric_idx = {f.name(): i for i, f in enumerate(fields) if f.isNumeric()}
    idx_h3 = fields.indexFromName("h3_id")
    fids: List[int] = []
    h3_ids: List[str] = []
    geoms = []
    areas: List[float] = []
    cols: Dict[str, List[float]] = {name: [] for name in numeric_idx}
    for feat in h3_layer.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        fids.append(int(feat.id()))
        h3_ids.append(str(feat.attribute(idx_h3)) if idx_h3 >= 0 else str(feat.id()))
        gc = QgsGeometry(g)
        geoms.append(gc)
        areas.append(float(gc.area()) / 10_000.0)
        attrs = feat.attributes()
        for name, i in numeric_idx.items():
            v = attrs[i]
            try:
                fv = float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                fv = float("nan")
            cols[name].append(fv)
    props = {
        str(k): str(h3_layer.customProperty(k))
        for k in h3_layer.customPropertyKeys()
        if str(k).startswith("h3forge/")
    }
    return HexSnapshot(
        fids=fids, h3_ids=h3_ids, geoms=geoms, crs=h3_layer.crs(),
        area_ha=np.asarray(areas, dtype=np.float64),
        numeric={k: np.asarray(v, dtype=np.float64) for k, v in cols.items()},
        properties=props,
    )


@dataclass
class VectorSource:
    name: str
    geom_kind: str                                # point | line | polygon
    geoms: list                                   # QgsGeometry in the H3 CRS
    values: List[Optional[float]]                 # numeric field (None when absent)


def snapshot_vector_source(layer, target_crs, field_name: str = "") -> VectorSource:
    from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsGeometry, QgsProject, QgsWkbTypes

    if layer is None or not layer.isValid():
        raise ValueError("A valid source vector layer is required.")
    gt = layer.geometryType()
    kind = (
        "polygon" if gt == QgsWkbTypes.GeometryType.PolygonGeometry
        else "line" if gt == QgsWkbTypes.GeometryType.LineGeometry
        else "point"
    )
    src_crs = layer.crs()
    if not src_crs.isValid():
        ext = layer.extent()
        if -180.0 <= ext.xMinimum() and ext.xMaximum() <= 180.0 and -90.0 <= ext.yMinimum() and ext.yMaximum() <= 90.0:
            src_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        else:
            src_crs = target_crs
    tr = None
    if src_crs != target_crs:
        tr = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())
    fidx = layer.fields().indexFromName(field_name) if field_name else -1
    geoms = []
    vals: List[Optional[float]] = []
    for f in layer.getFeatures():
        g = f.geometry()
        if g is None or g.isEmpty():
            continue
        gc = QgsGeometry(g)
        if tr is not None:
            with suppress(Exception):
                gc.transform(tr)
        geoms.append(gc)
        val: Optional[float] = None
        if fidx >= 0:
            with suppress(TypeError, ValueError):
                fv = float(f.attribute(fidx))
                if np.isfinite(fv):
                    val = fv
        vals.append(val)
    return VectorSource(name=layer.name(), geom_kind=kind, geoms=geoms, values=vals)


@dataclass
class RasterSource:
    name: str
    provider: object                              # QgsRasterDataProvider clone
    crs: object
    extent: object
    pixel_x: float
    pixel_y: float


def snapshot_raster_source(layer) -> RasterSource:
    if layer is None or not layer.isValid():
        raise ValueError("A valid raster layer is required.")
    return RasterSource(
        name=layer.name(),
        provider=layer.dataProvider().clone(),
        crs=layer.crs(),
        extent=layer.extent(),
        pixel_x=float(layer.rasterUnitsPerPixelX()),
        pixel_y=float(layer.rasterUnitsPerPixelY()),
    )


# ---------------------------------------------------------------------------
# Kernels (thread-safe)
# ---------------------------------------------------------------------------

def _index(geoms):
    from qgis.core import QgsFeature, QgsSpatialIndex

    idx = QgsSpatialIndex()
    for i, g in enumerate(geoms):
        f = QgsFeature(i)
        f.setGeometry(g)
        idx.addFeature(f)
    return idx


def stat(values: Sequence[float], method: str, default: float = 0.0) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    fn = {
        "sum": np.sum, "mean": np.mean, "min": np.min, "max": np.max,
        "median": np.median, "std": np.std,
    }.get(method)
    if method == "range":
        return float(np.max(arr) - np.min(arr))
    return float(fn(arr)) if fn else float(np.mean(arr))


def distance_array(
    hex_geoms: list,
    src_geoms: list,
    progress: ProgressFn = None,
    cancelled: CancelFn = None,
) -> np.ndarray:
    """Shortest planar distance (CRS units, metres) from each hexagon boundary."""
    n = len(hex_geoms)
    out = np.full(n, np.inf, dtype=np.float64)
    if not src_geoms:
        return out
    idx = _index(src_geoms)
    for i, hg in enumerate(hex_geoms):
        _tick(i, n, progress, cancelled)
        c = hg.centroid().asPoint()
        cand = idx.nearestNeighbor(c, 8) or range(len(src_geoms))
        best = min((float(hg.distance(src_geoms[j])) for j in cand), default=np.inf)
        if best > 0.0:
            for j in idx.intersects(hg.boundingBox()):
                d = float(hg.distance(src_geoms[j]))
                if d < best:
                    best = d
                    if best == 0.0:
                        break
        out[i] = best
    return out


def centroid_within_array(hex_geoms: list, polygons: list, cancelled: CancelFn = None) -> np.ndarray:
    """True where the hexagon centroid lies inside any polygon."""
    n = len(hex_geoms)
    out = np.zeros(n, dtype=bool)
    if not polygons:
        return out
    idx = _index(polygons)
    for i, hg in enumerate(hex_geoms):
        _tick(i, n, None, cancelled)
        c = hg.centroid()
        for j in idx.intersects(c.boundingBox()):
            if polygons[j].contains(c):
                out[i] = True
                break
    return out


def vector_metric_array(
    hex_geoms: list,
    src: VectorSource,
    method: str = "count",
    default: float = 0.0,
    progress: ProgressFn = None,
    cancelled: CancelFn = None,
) -> np.ndarray:
    """Join-by-location metric per hexagon (see ``VECTOR_AGG_METHODS``)."""
    n = len(hex_geoms)
    out = np.full(n, float(default), dtype=np.float64)
    if method == "nearest_dist_m":
        d = distance_array(hex_geoms, src.geoms, progress, cancelled)
        return np.where(np.isfinite(d), np.round(d, 2), float(default))
    if not src.geoms:
        return out
    idx = _index(src.geoms)
    src_area = [max(1e-6, float(g.area())) for g in src.geoms] if src.geom_kind == "polygon" else None

    for i, hg in enumerate(hex_geoms):
        _tick(i, n, progress, cancelled)
        hex_m2 = max(1.0, float(hg.area()))
        hex_ha = hex_m2 / 10_000.0
        weights: List[float] = []
        vals: List[float] = []
        appor: List[float] = []
        for j in idx.intersects(hg.boundingBox()):
            sg = src.geoms[j]
            if not hg.intersects(sg):
                continue
            v = src.values[j]
            if src.geom_kind == "polygon":
                ig = hg.intersection(sg)
                a = float(ig.area()) if ig is not None and not ig.isEmpty() else 0.0
                if a <= 0.0:
                    continue
                weights.append(a)
                if v is not None:
                    vals.append(v)
                    appor.append(v * a / src_area[j])
            elif src.geom_kind == "line":
                ig = hg.intersection(sg)
                ln = float(ig.length()) if ig is not None and not ig.isEmpty() else 0.0
                if ln <= 0.0:
                    continue
                weights.append(ln)
                if v is not None:
                    vals.append(v)
                    appor.append(v * ln)
            else:
                weights.append(1.0)
                if v is not None:
                    vals.append(v)
                    appor.append(v)

        if method == "count":
            r = float(len(weights))
        elif method == "density_ha":
            r = (sum(weights) if src.geom_kind == "line" else len(weights)) / max(1e-6, hex_ha)
        elif method == "area_share_pct":
            r = float(np.clip(sum(weights) / hex_m2 * 100.0, 0.0, 100.0))
        elif method == "length_m":
            r = float(sum(weights))
        elif method == "area_weighted_mean":
            if vals and len(vals) == len(weights) and sum(weights) > 0:
                r = float(np.dot(weights, vals) / sum(weights))
            else:
                r = stat(vals, "mean", default)
        elif method == "area_weighted_sum":
            r = float(sum(appor)) if appor else float(default)
        else:
            r = stat(vals, method, default)
        out[i] = r
    return out


def raster_zonal_array(
    hex_geoms: list,
    hex_crs,
    src: RasterSource,
    method: str = "mean",
    band: int = 1,
    default: float = 0.0,
    cancelled: CancelFn = None,
    log: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Exact pixel-based zonal statistic per hexagon (QgsZonalStatistics).

    Runs on a private memory layer and a provider clone, so it is safe in a
    worker thread. Hexagons that contain no pixel centre (cells smaller than a
    pixel, or outside the raster) fall back to a 13-point sampling stencil;
    for ``sum`` the stencil mean is scaled by cell area / pixel area. The
    number of fallback cells is reported through ``log``.
    """
    from qgis.analysis import QgsZonalStatistics
    from qgis.core import (
        QgsCoordinateTransform, QgsFeature, QgsFields, QgsGeometry,
        QgsMemoryProviderUtils, QgsPointXY, QgsProject, QgsWkbTypes,
    )

    from .h3_grid_engine import _make_qgs_field

    n = len(hex_geoms)
    out = np.full(n, np.nan, dtype=np.float64)
    flags = {
        "mean": QgsZonalStatistics.Mean, "median": QgsZonalStatistics.Median,
        "max": QgsZonalStatistics.Max, "min": QgsZonalStatistics.Min,
        "std": QgsZonalStatistics.StDev, "range": QgsZonalStatistics.Range,
        "sum": QgsZonalStatistics.Sum,
    }
    try:
        # The provider CRS must be the hexagon CRS: a "Polygon?..." URI without
        # crs= leaves the memory provider in EPSG:4326 even after setCrs(), and
        # QgsZonalStatistics then finds no pixels at all (silently).
        fields = QgsFields()
        fields.append(_make_qgs_field("i", "int"))
        tmp = QgsMemoryProviderUtils.createMemoryLayer("h3forge_zs", fields, QgsWkbTypes.Polygon, hex_crs)
        feats = []
        for i, g in enumerate(hex_geoms):
            f = QgsFeature(tmp.fields())
            f.setGeometry(g)
            f.setAttribute(0, i)
            feats.append(f)
        tmp.dataProvider().addFeatures(feats)
        stats = QgsZonalStatistics.Statistics(flags.get(method, QgsZonalStatistics.Mean) | QgsZonalStatistics.Count)
        zs = QgsZonalStatistics(
            tmp, src.provider, src.crs, src.pixel_x, src.pixel_y, "zs_", int(band), stats
        )
        rc = zs.calculateStatistics(None)
        if rc == 0:
            names = [f.name() for f in tmp.fields()]
            i_cnt = names.index("zs_count") if "zs_count" in names else -1
            i_val = next((k for k, nm in enumerate(names) if nm.startswith("zs_") and k != i_cnt), -1)
            for f in tmp.getFeatures():
                a = f.attributes()
                with suppress(TypeError, ValueError):
                    if i_cnt >= 0 and float(a[i_cnt]) > 0 and a[i_val] is not None:
                        out[int(a[0])] = float(a[i_val])
        elif log is not None:
            log(f"zonal statistics on '{src.name}' returned code {rc}; using point sampling")
    except Exception as exc:  # noqa: BLE001 - reported, then point sampling takes over
        if log is not None:
            log(f"zonal statistics on '{src.name}' failed ({exc}); using point sampling")

    missing = np.flatnonzero(~np.isfinite(out))
    if missing.size:
        tr = None
        if hex_crs != src.crs:
            tr = QgsCoordinateTransform(hex_crs, src.crs, QgsProject.instance())
        ext = src.extent
        pixel_area = abs(float(src.pixel_x) * float(src.pixel_y))
        for count, i in enumerate(missing):
            _tick(count, missing.size, None, cancelled)
            hg = hex_geoms[i]
            c = hg.centroid().asPoint()
            pts = [c]
            verts = [QgsPointXY(v.x(), v.y()) for v in hg.vertices()]
            for v in verts[:-1]:
                pts.append(QgsPointXY(0.5 * (c.x() + v.x()), 0.5 * (c.y() + v.y())))
                pts.append(QgsPointXY(0.15 * c.x() + 0.85 * v.x(), 0.15 * c.y() + 0.85 * v.y()))
            samples = []
            for p in pts:
                rp = p
                if tr is not None:
                    with suppress(Exception):
                        rp = tr.transform(p)
                if not (ext.xMinimum() <= rp.x() < ext.xMaximum() and ext.yMinimum() < rp.y() <= ext.yMaximum()):
                    continue
                val, ok = src.provider.sample(rp, int(band))
                if ok and val is not None and np.isfinite(val):
                    samples.append(float(val))
            if method == "sum":
                # A stencil total counts 13 points, not pixels: scale the mean by
                # the share of the cell area each pixel represents.
                if samples and pixel_area > 0:
                    cell = QgsGeometry(hg)
                    if tr is not None:
                        with suppress(Exception):
                            cell.transform(tr)
                    out[i] = float(np.mean(samples)) * cell.area() / pixel_area
                else:
                    out[i] = float(default)
            else:
                out[i] = stat(samples, method, default)
        if log is not None:
            log(f"zonal statistics on '{src.name}': {missing.size:,} of {n:,} hexagons "
                f"({100.0 * missing.size / max(1, n):.1f}%) used point sampling")
    return np.where(np.isfinite(out), out, float(default))
