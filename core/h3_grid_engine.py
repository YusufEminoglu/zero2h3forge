# -*- coding: utf-8 -*-
"""Uber H3 Hexagonal Grid Generator and Topology Engine for 02H3Forge.

Supports both the official Uber ``h3`` library (v4 and v3 APIs) and a
zero-dependency mathematical H3-scale hexagonal tessellation fallback so the
plugin works across any QGIS 3.28+ / 4.x installation.
"""
from __future__ import annotations

import math
import re
from contextlib import suppress
from typing import Dict, List, Optional, Set, Tuple

try:
    import h3 as _h3_lib
    HAS_H3_LIB = True
except ImportError:
    _h3_lib = None
    HAS_H3_LIB = False


DEFAULT_H3_RESOLUTION: int = 10
# Above this many candidate hexagons the native H3 path is skipped and the
# metric fallback coarsens its spacing to stay responsive.
MAX_NATIVE_H3_CELLS: int = 65_000
MAX_METRIC_CELLS: int = 60_000

# Canonical Uber H3 resolution table (Resolutions 6 through 12).
H3_RESOLUTION_SPECS: Dict[int, Dict[str, object]] = {
    6: {
        "res": 6,
        "area_m2": 36_129_062.1,
        "area_ha": 3612.91,
        "edge_m": 3229.48,
        "diameter_m": 6458.96,
        "label": "Res 6 — ~3,613 ha | ~3,229 m edge (Regional Basin)",
    },
    7: {
        "res": 7,
        "area_m2": 5_161_293.4,
        "area_ha": 516.13,
        "edge_m": 1220.63,
        "diameter_m": 2441.26,
        "label": "Res 7 — ~516.1 ha | ~1,221 m edge (Sub-Regional 1/100k)",
    },
    8: {
        "res": 8,
        "area_m2": 737_327.6,
        "area_ha": 73.73,
        "edge_m": 461.35,
        "diameter_m": 922.71,
        "label": "Res 8 — ~73.7 ha | ~461 m edge (Metropolitan 1/50k)",
    },
    9: {
        "res": 9,
        "area_m2": 105_332.5,
        "area_ha": 10.53,
        "edge_m": 174.38,
        "diameter_m": 348.75,
        "label": "Res 9 — ~10.5 ha | ~174 m edge (District Master Plan 1/25k)",
    },
    10: {
        "res": 10,
        "area_m2": 15_047.5,
        "area_ha": 1.505,
        "edge_m": 65.91,
        "diameter_m": 131.82,
        "label": "Res 10 (Default ~2 ha) — ~1.50 ha (15,048 m²) | ~65.9 m edge (1/5k)",
    },
    11: {
        "res": 11,
        "area_m2": 2_149.6,
        "area_ha": 0.215,
        "edge_m": 24.91,
        "diameter_m": 49.82,
        "label": "Res 11 (Micro ~20m) — ~0.21 ha (2,150 m²) | ~24.9 m edge (1/1k)",
    },
    12: {
        "res": 12,
        "area_m2": 307.1,
        "area_ha": 0.031,
        "edge_m": 9.42,
        "diameter_m": 18.83,
        "label": "Res 12 (Ultra-Fine) — ~0.03 ha (307 m²) | ~9.4 m edge (Parcel Scale)",
    },
}


def get_resolution_spec(resolution: int) -> Dict[str, object]:
    """Return the H3 resolution dictionary, clamping to supported [6..12]."""
    res = max(6, min(12, int(resolution)))
    return H3_RESOLUTION_SPECS[res]


def circumradius_for_resolution(resolution: int) -> float:
    """Compute exact regular hexagon circumradius R (m) from target H3 area."""
    spec = get_resolution_spec(resolution)
    area_m2 = float(spec["area_m2"])
    return math.sqrt((2.0 * area_m2) / (3.0 * math.sqrt(3.0)))


def estimate_hex_count(width_m: float, height_m: float, resolution: int) -> int:
    """Approximate number of hexagons covering a ``width_m`` x ``height_m`` box.

    Adds one ring of edge cells around the box, which is what both the native
    H3 polyfill (buffered bbox) and the metric fallback produce in practice.
    """
    w = max(0.0, float(width_m))
    h = max(0.0, float(height_m))
    if w <= 0.0 or h <= 0.0:
        return 0
    spec = get_resolution_spec(resolution)
    radius = circumradius_for_resolution(resolution)
    pad = 2.0 * radius
    area = (w + pad) * (h + pad)
    return int(math.ceil(area / float(spec["area_m2"])))


def sanitize_field_slug(name: str, prefix: str = "raw", max_len: int = 28) -> str:
    """Build a clean ASCII attribute column name with the given prefix."""
    ascii_map = str.maketrans({
        "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g",
        "ü": "u", "Ü": "u", "ş": "s", "Ş": "s",
        "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    })
    cleaned = (name or "crit").translate(ascii_map).lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    # ``dist_`` is deliberately kept: normalizing ``dist_x`` must produce
    # ``norm_dist_x`` and never overwrite the ring-scored ``norm_x`` criterion.
    for pfx in ("raw_", "norm_", "ring_"):
        if cleaned.startswith(pfx):
            cleaned = cleaned[len(pfx):]
    if not cleaned:
        cleaned = "crit"
    full = f"{prefix}_{cleaned}" if prefix else cleaned
    return full[:max_len].rstrip("_")


def generate_metric_hex_cells(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    resolution: int = DEFAULT_H3_RESOLUTION,
    max_cells: int = MAX_METRIC_CELLS,
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Pure-Python regular hexagonal tessellation matching exact H3 metric area.

    Returns a list of ``(h3_id, [(x0, y0), ..., (x0, y0)])`` in planar meters.
    Pointy-top orientation:
      horizontal spacing dx = sqrt(3) * R
      vertical spacing   dy = 1.5 * R
    """
    if xmax <= xmin or ymax <= ymin:
        return []
    radius = circumradius_for_resolution(resolution)
    dx = math.sqrt(3.0) * radius
    dy = 1.5 * radius

    ncols = int(math.ceil((xmax - xmin) / dx)) + 2
    nrows = int(math.ceil((ymax - ymin) / dy)) + 2
    if ncols * nrows > max_cells:
        scale = math.sqrt((ncols * nrows) / float(max_cells))
        radius *= scale
        dx = math.sqrt(3.0) * radius
        dy = 1.5 * radius
        ncols = int(math.ceil((xmax - xmin) / dx)) + 2
        nrows = int(math.ceil((ymax - ymin) / dy)) + 2

    angles = [math.radians(60.0 * k + 30.0) for k in range(6)]
    cos_a = [math.cos(a) for a in angles]
    sin_a = [math.sin(a) for a in angles]

    cells: List[Tuple[str, List[Tuple[float, float]]]] = []
    for row in range(nrows):
        cy = ymin + row * dy
        if cy - radius > ymax:
            break
        x_offset = (0.5 * dx) if (row % 2 == 1) else 0.0
        for col in range(ncols):
            cx = xmin + col * dx + x_offset
            if cx - radius > xmax:
                break
            ring = [
                (cx + radius * cos_a[k], cy + radius * sin_a[k])
                for k in range(6)
            ]
            ring.append(ring[0])
            cell_id = f"h3r{int(resolution):02d}_{row:04x}_{col:04x}"
            cells.append((cell_id, ring))
    return cells


def _h3_cells_for_wgs84_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    resolution: int,
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Generate authentic Uber H3 cells (v4 or v3) covering a WGS84 bounding box.

    Returns ``[(h3_id, [(lon, lat), ...]), ...]``.
    """
    if not HAS_H3_LIB or _h3_lib is None:
        return []

    spec = get_resolution_spec(resolution)
    edge_deg = max(0.00015, float(spec["edge_m"]) / 111_000.0 * 1.75)
    b_min_lat = max(-89.9, min_lat - edge_deg)
    b_max_lat = min(89.9, max_lat + edge_deg)
    b_min_lon = max(-179.99, min_lon - edge_deg)
    b_max_lon = min(179.99, max_lon + edge_deg)

    outer_latlon = [
        (b_min_lat, b_min_lon),
        (b_min_lat, b_max_lon),
        (b_max_lat, b_max_lon),
        (b_max_lat, b_min_lon),
    ]

    cell_ids: Set[str] = set()
    # H3 v4 API
    if hasattr(_h3_lib, "LatLngPoly") and hasattr(_h3_lib, "polygon_to_cells"):
        poly = _h3_lib.LatLngPoly(outer_latlon)
        cell_ids.update(_h3_lib.polygon_to_cells(poly, int(resolution)))
        if not cell_ids and hasattr(_h3_lib, "latlng_to_cell"):
            c_lat = 0.5 * (min_lat + max_lat)
            c_lon = 0.5 * (min_lon + max_lon)
            seed = _h3_lib.latlng_to_cell(c_lat, c_lon, int(resolution))
            cell_ids.update(_h3_lib.grid_disk(seed, 2))
        out: List[Tuple[str, List[Tuple[float, float]]]] = []
        for cid in sorted(cell_ids):
            boundary_latlon = _h3_lib.cell_to_boundary(cid)
            ring_lonlat = [(float(lon), float(lat)) for lat, lon in boundary_latlon]
            if ring_lonlat and ring_lonlat[0] != ring_lonlat[-1]:
                ring_lonlat.append(ring_lonlat[0])
            out.append((str(cid), ring_lonlat))
        return out

    # H3 v3 fallback API
    if hasattr(_h3_lib, "polyfill"):
        geojson_poly = {
            "type": "Polygon",
            "coordinates": [[
                [b_min_lon, b_min_lat],
                [b_max_lon, b_min_lat],
                [b_max_lon, b_max_lat],
                [b_min_lon, b_max_lat],
                [b_min_lon, b_min_lat],
            ]],
        }
        cell_ids.update(_h3_lib.polyfill(geojson_poly, int(resolution), geo_json_conformant=True))
        out_v3: List[Tuple[str, List[Tuple[float, float]]]] = []
        for cid in sorted(cell_ids):
            boundary_latlon = _h3_lib.h3_to_geo_boundary(cid)
            ring_lonlat = [(float(lon), float(lat)) for lat, lon in boundary_latlon]
            if ring_lonlat and ring_lonlat[0] != ring_lonlat[-1]:
                ring_lonlat.append(ring_lonlat[0])
            out_v3.append((str(cid), ring_lonlat))
        return out_v3

    return []


def get_h3_neighbors(h3_ids: List[str]) -> Dict[str, List[str]]:
    """Build a 1-ring adjacency dictionary among the given H3 hexagon IDs."""
    id_set = set(h3_ids)
    adj: Dict[str, List[str]] = {cid: [] for cid in h3_ids}

    if HAS_H3_LIB and _h3_lib is not None:
        if hasattr(_h3_lib, "grid_disk"):
            for cid in h3_ids:
                if cid.startswith("h3r"):
                    continue
                with suppress(Exception):
                    ring = _h3_lib.grid_disk(cid, 1)
                    adj[cid] = [n for n in ring if n != cid and n in id_set]
            if any(adj.values()):
                return adj
        elif hasattr(_h3_lib, "k_ring"):
            for cid in h3_ids:
                if cid.startswith("h3r"):
                    continue
                with suppress(Exception):
                    ring = _h3_lib.k_ring(cid, 1)
                    adj[cid] = [n for n in ring if n != cid and n in id_set]
            if any(adj.values()):
                return adj

    # Fallback adjacency for synthetic metric IDs (h3rRR_RRRR_CCCC)
    rc_to_id: Dict[Tuple[int, int], str] = {}
    parsed: Dict[str, Tuple[int, int]] = {}
    for cid in h3_ids:
        parts = cid.split("_")
        if len(parts) == 3 and parts[0].startswith("h3r"):
            with suppress(ValueError):
                r = int(parts[1], 16)
                c = int(parts[2], 16)
                rc_to_id[(r, c)] = cid
                parsed[cid] = (r, c)
    for cid, (r, c) in parsed.items():
        if r % 2 == 0:
            offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        nbrs = []
        for dr, dc in offsets:
            nid = rc_to_id.get((r + dr, c + dc))
            if nid is not None:
                nbrs.append(nid)
        adj[cid] = nbrs
    return adj


# ---------------------------------------------------------------------------
# PyQGIS Layer & Field Helpers
# ---------------------------------------------------------------------------

def _make_qgs_field(name: str, kind: str = "double", length: int = 64):
    """Create a QgsField safely across QGIS 3.28–3.44 and QGIS 4.2 (TRAPS §3.2)."""
    from qgis.core import QgsField
    from qgis.PyQt.QtCore import QMetaType, QVariant

    try:
        if kind == "string":
            return QgsField(name, QMetaType.Type.QString, "string", length)
        if kind == "int":
            return QgsField(name, QMetaType.Type.Int, "integer")
        return QgsField(name, QMetaType.Type.Double, "double")
    except TypeError:
        if kind == "string":
            return QgsField(name, QVariant.String, "string", length)
        if kind == "int":
            return QgsField(name, QVariant.Int, "integer")
        return QgsField(name, QVariant.Double, "double")


def ensure_layer_field(layer, name: str, kind: str = "double", length: int = 64) -> int:
    """Ensure field ``name`` exists on ``layer`` and return its field index."""
    idx = layer.fields().indexFromName(name)
    if idx >= 0:
        return idx
    dp = layer.dataProvider()
    dp.addAttributes([_make_qgs_field(name, kind=kind, length=length)])
    layer.updateFields()
    return layer.fields().indexFromName(name)


def choose_metric_epsg(lon: float, lat: float) -> int:
    """Pick a conformal metric CRS that keeps scale error small at (lon, lat).

    - Inside Türkiye: the TUREF 3-degree Transverse Mercator zone whose central
      meridian is nearest (EPSG:5253 TM27 ... EPSG:5259 TM45).
    - Elsewhere: the WGS 84 / UTM zone of the location (EPSG:326xx north,
      EPSG:327xx south); polar caps fall back to UPS (EPSG:32661 / 32761).

    Web Mercator (EPSG:3857) is never used: its scale error is +30% at 40° and
    +100% at 60° latitude, which would corrupt distances, buffers and areas.
    """
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    lat = max(-90.0, min(90.0, float(lat)))
    if 25.5 <= lon <= 45.5 and 35.5 <= lat <= 42.5:
        cm = min(45, max(27, 27 + 3 * int(round((lon - 27.0) / 3.0))))
        return 5253 + (cm - 27) // 3
    if lat >= 84.0:
        return 32661
    if lat <= -80.0:
        return 32761
    zone = min(60, int((lon + 180.0) // 6.0) + 1)
    return (32600 if lat >= 0.0 else 32700) + zone


def _is_metric_projected(crs) -> bool:
    if crs is None or not crs.isValid() or crs.isGeographic():
        return False
    from qgis.core import Qgis, QgsUnitTypes

    unit_enum = getattr(Qgis, "DistanceUnit", None)  # QGIS 3.30+
    meters = unit_enum.Meters if unit_enum is not None else QgsUnitTypes.DistanceMeters
    if crs.mapUnits() != meters:
        return False
    # Web Mercator variants are metric in name only.
    return crs.authid() not in ("EPSG:3857", "EPSG:900913", "EPSG:3395", "EPSG:102100")


def resolve_metric_work_crs(extent_rect, target_crs):
    """Return ``(work_crs, extent_in_work_crs)`` in a true-scale metric CRS.

    Projected metre-based inputs are kept as they are. Geographic, non-metre
    or Web-Mercator inputs are moved to :func:`choose_metric_epsg` for the
    extent centre.
    """
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsPointXY,
        QgsProject,
    )

    if target_crs is None or not target_crs.isValid():
        target_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    if _is_metric_projected(target_crs):
        return target_crs, extent_rect
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    centre = QgsPointXY(extent_rect.center())
    if target_crs != wgs84:
        to_wgs = QgsCoordinateTransform(target_crs, wgs84, QgsProject.instance())
        centre = to_wgs.transform(centre)
    work_crs = QgsCoordinateReferenceSystem(f"EPSG:{choose_metric_epsg(centre.x(), centre.y())}")
    tr = QgsCoordinateTransform(target_crs, work_crs, QgsProject.instance())
    return work_crs, tr.transformBoundingBox(extent_rect)


def build_h3_grid_layer(
    extent_rect,
    target_crs,
    resolution: int = DEFAULT_H3_RESOLUTION,
    boundary_layer=None,
    coverage_mode: str = "intersect",
    prefer_native_h3: bool = True,
    layer_name: Optional[str] = None,
):
    """Create a memory QgsVectorLayer populated with H3 hexagonal polygons.

    Parameters
    ----------
    extent_rect : QgsRectangle
        Bounding box in ``target_crs``.
    target_crs : QgsCoordinateReferenceSystem
        CRS of ``extent_rect``. Metre-based projected CRSs are used as-is; any
        other CRS is replaced by a local true-scale CRS (see
        :func:`choose_metric_epsg`).
    resolution : int
        H3 resolution (6..12), default 10 (~1.50 ha).
    boundary_layer : QgsVectorLayer, optional
        Optional polygon layer used to filter hexagons by intersection or centroid.
    coverage_mode : str
        ``"intersect"`` (keep hexes intersecting boundary) or ``"centroid"``
        (keep hexes whose centroid falls inside boundary) or ``"bbox"`` (full bbox).
    prefer_native_h3 : bool
        Use authentic Uber ``h3`` WGS84 indexing when True and ``h3`` is installed.
    layer_name : str, optional
        Display name of the returned vector layer.
    """
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsFeature,
        QgsGeometry,
        QgsPointXY,
        QgsProject,
        QgsVectorLayer,
        QgsWkbTypes,
    )

    res = max(6, min(12, int(resolution)))
    spec = get_resolution_spec(res)

    # Move degree / non-metre / Web-Mercator inputs to a local true-scale CRS
    work_crs, extent_rect = resolve_metric_work_crs(extent_rect, target_crs)

    # Prepare optional boundary union geometry in work_crs
    boundary_geom = None
    if (
        boundary_layer is not None
        and boundary_layer.isValid()
        and boundary_layer.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry
        and coverage_mode in ("intersect", "centroid")
    ):
        tr_bnd = None
        if boundary_layer.crs() != work_crs:
            tr_bnd = QgsCoordinateTransform(boundary_layer.crs(), work_crs, QgsProject.instance())
        b_geoms = []
        for feat in boundary_layer.getFeatures():
            g = feat.geometry()
            if g is not None and not g.isEmpty():
                if tr_bnd is not None:
                    g = QgsGeometry(g)
                    with suppress(Exception):
                        g.transform(tr_bnd)
                b_geoms.append(g)
        if b_geoms:
            boundary_geom = QgsGeometry.unaryUnion(b_geoms)
            if boundary_geom is not None and not boundary_geom.isEmpty():
                extent_rect = boundary_geom.boundingBox()

    # Build candidate hexagons in work_crs
    hex_candidates: List[Tuple[str, object]] = []
    engine = "metric-fallback"
    if prefer_native_h3 and HAS_H3_LIB:
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        tr_to_wgs = QgsCoordinateTransform(work_crs, wgs84, QgsProject.instance())
        tr_from_wgs = QgsCoordinateTransform(wgs84, work_crs, QgsProject.instance())
        with suppress(Exception):
            wgs_box = tr_to_wgs.transformBoundingBox(extent_rect)
            raw_h3 = _h3_cells_for_wgs84_bbox(
                wgs_box.yMinimum(),
                wgs_box.xMinimum(),
                wgs_box.yMaximum(),
                wgs_box.xMaximum(),
                res,
            )
            if len(raw_h3) <= MAX_NATIVE_H3_CELLS:
                engine = "native-h3"
                for cid, ring_lonlat in raw_h3:
                    pts_work = [
                        tr_from_wgs.transform(QgsPointXY(lon, lat))
                        for lon, lat in ring_lonlat
                    ]
                    poly_geom = QgsGeometry.fromPolygonXY([pts_work])
                    if poly_geom is not None and not poly_geom.isEmpty():
                        hex_candidates.append((cid, poly_geom))

    if not hex_candidates:
        engine = "metric-fallback"
        metric_cells = generate_metric_hex_cells(
            extent_rect.xMinimum(),
            extent_rect.yMinimum(),
            extent_rect.xMaximum(),
            extent_rect.yMaximum(),
            resolution=res,
        )
        for cid, ring_xy in metric_cells:
            pts = [QgsPointXY(x, y) for x, y in ring_xy]
            poly_geom = QgsGeometry.fromPolygonXY([pts])
            if poly_geom is not None and not poly_geom.isEmpty():
                hex_candidates.append((cid, poly_geom))

    # Construct memory layer using URI field definitions (TRAPS.md §3.2)
    uri = (
        f"Polygon?crs={work_crs.authid()}"
        "&field=h3_id:string(32)"
        "&field=h3_res:integer"
        "&field=area_m2:double"
        "&field=area_ha:double"
        "&field=constraint_mask:integer"
    )
    title = layer_name or f"H3_Grid_Res{res}_{int(spec['edge_m'])}m"
    mem_layer = QgsVectorLayer(uri, title, "memory")
    dp = mem_layer.dataProvider()
    fields = mem_layer.fields()

    features_to_add = []
    for cid, poly_geom in hex_candidates:
        if boundary_geom is not None and not boundary_geom.isEmpty():
            if coverage_mode == "centroid":
                c_pt = poly_geom.centroid()
                if c_pt is None or not boundary_geom.intersects(c_pt):
                    continue
            else:
                if not boundary_geom.intersects(poly_geom):
                    continue
        area_m2 = float(poly_geom.area())
        if area_m2 <= 0.0:
            continue
        feat = QgsFeature(fields)
        feat.setGeometry(poly_geom)
        feat.setAttribute("h3_id", cid)
        feat.setAttribute("h3_res", res)
        feat.setAttribute("area_m2", round(area_m2, 2))
        feat.setAttribute("area_ha", round(area_m2 / 10_000.0, 4))
        feat.setAttribute("constraint_mask", 1)
        features_to_add.append(feat)

    if features_to_add:
        dp.addFeatures(features_to_add)
    mem_layer.updateExtents()

    # Provenance so the UI can tell the analyst exactly what was built.
    mean_area = (
        sum(float(f.attribute("area_m2")) for f in features_to_add) / len(features_to_add)
        if features_to_add else 0.0
    )
    mem_layer.setCustomProperty("h3forge/engine", engine)
    mem_layer.setCustomProperty("h3forge/resolution", res)
    mem_layer.setCustomProperty("h3forge/mean_area_m2", round(mean_area, 2))
    coarsened = bool(
        engine == "metric-fallback"
        and mean_area > 1.05 * float(spec["area_m2"])
    )
    mem_layer.setCustomProperty("h3forge/coarsened", coarsened)
    return mem_layer
