# -*- coding: utf-8 -*-
"""Cartographic Styling, Class Area Diagnostics, and HTML/CSV Exporter for 02H3Forge."""
from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .mcda_composer import CLASS_LABELS_RISK, CLASS_LABELS_SUITABILITY
from .spatial_stats import EXCLUDED_LABEL, HOTSPOT_LABELS


THEME_PALETTES: Dict[str, Tuple[str, ...]] = {
    "risk": ("#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"),
    "suitability": ("#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"),
    "susceptibility": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
    "magma": ("#fcfdbf", "#fe9f6d", "#de4968", "#8c2981", "#3b0f70"),
}

# Diverging blue-white-red ramp for Getis-Ord Gi* confidence bins (-3..3).
HOTSPOT_COLORS: Dict[int, str] = {
    3: "#b2182b",
    2: "#ef8a62",
    1: "#fddbc7",
    0: "#f7f7f7",
    -1: "#d1e5f0",
    -2: "#67a9cf",
    -3: "#2166ac",
}
HOTSPOT_ORDER: List[Tuple[int, str]] = [(c, HOTSPOT_LABELS[c]) for c in (3, 2, 1, 0, -1, -2, -3)]

THEME_OPTIONS: List[Tuple[str, str]] = [
    ("risk", "Vulnerability / Risk / Hazard (Yellow → Deep Red)"),
    ("suitability", "Land Suitability / Opportunity (PlanX Yellow → Dark Green)"),
    ("susceptibility", "Susceptibility / Sensitivity (Viridis Spectrum)"),
    ("magma", "Thermal / Hazard Intensity (Magma Spectrum)"),
]


def apply_h3_graduated_style(
    h3_layer,
    target_field: str = "comp_score",
    theme: str = "risk",
    breaks: Optional[Sequence[float]] = None,
    opacity: float = 0.82,
) -> bool:
    """Apply a 5-class graduated vector renderer to ``h3_layer``."""
    from qgis.core import (
        QgsFillSymbol,
        QgsGraduatedSymbolRenderer,
        QgsRendererRange,
    )

    if h3_layer is None or not h3_layer.isValid():
        return False
    if h3_layer.fields().indexFromName(target_field) < 0:
        return False

    colors = THEME_PALETTES.get(theme, THEME_PALETTES["risk"])
    labels = CLASS_LABELS_SUITABILITY if theme == "suitability" else CLASS_LABELS_RISK
    b_list = (
        list(breaks)
        if (breaks is not None and len(breaks) == 6)
        else [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    )

    ranges: List[QgsRendererRange] = []
    for i in range(5):
        lo = float(b_list[i])
        hi = float(b_list[i + 1])
        sym = QgsFillSymbol.createSimple({
            "color": colors[i],
            "outline_color": "#334155",
            "outline_width": "0.18",
            "outline_style": "solid",
        })
        sym.setOpacity(float(opacity))
        lbl = f"{labels[i]} ({lo:.1f} – {hi:.1f})"
        ranges.append(QgsRendererRange(lo, hi, sym, lbl))

    renderer = QgsGraduatedSymbolRenderer(target_field, ranges)
    h3_layer.setRenderer(renderer)
    h3_layer.triggerRepaint()
    return True


def apply_h3_hotspot_style(h3_layer, opacity: float = 0.85) -> bool:
    """Apply a categorized Gi* hotspot renderer keyed on ``hotspot_lbl``.

    Keying on the label (not the -3..3 code) keeps hard-excluded hexagons in
    their own neutral grey class instead of blending into "Not Significant".
    """
    from qgis.core import (
        QgsCategorizedSymbolRenderer,
        QgsFillSymbol,
        QgsRendererCategory,
    )

    if h3_layer is None or not h3_layer.isValid():
        return False
    if h3_layer.fields().indexFromName("hotspot_lbl") < 0:
        return False

    entries = [(HOTSPOT_COLORS[code], label) for code, label in HOTSPOT_ORDER]
    entries.append(("#9ca3af", EXCLUDED_LABEL))
    categories = []
    for color, label in entries:
        sym = QgsFillSymbol.createSimple({
            "color": color,
            "outline_color": "#475569",
            "outline_width": "0.15",
        })
        sym.setOpacity(float(opacity) * (0.5 if label == EXCLUDED_LABEL else 1.0))
        categories.append(QgsRendererCategory(label, sym, label))
    renderer = QgsCategorizedSymbolRenderer("hotspot_lbl", categories)
    h3_layer.setRenderer(renderer)
    h3_layer.triggerRepaint()
    return True


def save_h3_layer_to_geopackage(h3_layer, gpkg_path: str, layer_name: Optional[str] = None):
    """Write ``h3_layer`` to a GeoPackage and return the persistent layer.

    The returned OGR layer carries over the renderer and every ``h3forge/*``
    custom property (grid engine, run provenance, hotspot and sensitivity
    results), so the studio and the HTML report keep working on it.
    """
    import re as _re

    from qgis.core import QgsProject, QgsVectorFileWriter, QgsVectorLayer

    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required.")
    name = layer_name or _re.sub(r"[^A-Za-z0-9_]+", "_", h3_layer.name()).strip("_") or "h3_grid"
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = "GPKG"
    opts.layerName = name
    opts.fileEncoding = "UTF-8"
    try:
        import os as _os

        if _os.path.exists(gpkg_path):
            opts.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    except AttributeError:  # pragma: no cover
        pass
    ctx = QgsProject.instance().transformContext()
    if hasattr(QgsVectorFileWriter, "writeAsVectorFormatV3"):
        res = QgsVectorFileWriter.writeAsVectorFormatV3(h3_layer, gpkg_path, ctx, opts)
    else:  # pragma: no cover - very old QGIS
        res = QgsVectorFileWriter.writeAsVectorFormatV2(h3_layer, gpkg_path, ctx, opts)
    err = res[0]
    if err != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"Could not write GeoPackage: {res[1] if len(res) > 1 else err}")

    out = QgsVectorLayer(f"{gpkg_path}|layername={name}", h3_layer.name(), "ogr")
    if not out.isValid():
        raise RuntimeError(f"GeoPackage was written but layer '{name}' could not be opened.")
    for key in h3_layer.customPropertyKeys():
        if str(key).startswith("h3forge/"):
            out.setCustomProperty(key, h3_layer.customProperty(key))
    if h3_layer.renderer() is not None:
        out.setRenderer(h3_layer.renderer().clone())
    return out


def compute_class_summary(h3_layer, theme: str = "risk") -> List[Dict[str, object]]:
    """Compute per-class hexagon count, total area (ha), share (%), and top XAI driver."""
    labels = CLASS_LABELS_SUITABILITY if theme == "suitability" else CLASS_LABELS_RISK
    if h3_layer is None or not h3_layer.isValid():
        return []

    fields = h3_layer.fields()
    idx_cid = fields.indexFromName("class_id")
    idx_score = fields.indexFromName("comp_score")
    idx_ha = fields.indexFromName("area_ha")
    idx_drv = fields.indexFromName("top_driver")

    buckets: Dict[int, Dict[str, object]] = {
        cid: {
            "class_id": cid,
            "class_label": labels[cid - 1],
            "hex_count": 0,
            "area_ha": 0.0,
            "scores": [],
            "drivers": Counter(),
        }
        for cid in range(1, 6)
    }

    total_ha = 0.0
    for feat in h3_layer.getFeatures():
        cid = 1
        if idx_cid >= 0:
            v_c = feat.attribute(idx_cid)
            if v_c is not None:
                try:
                    cid = max(1, min(5, int(v_c)))
                except (ValueError, TypeError):
                    cid = 1
        ha = 0.0
        if idx_ha >= 0:
            v_ha = feat.attribute(idx_ha)
            if v_ha is not None:
                try:
                    ha = float(v_ha)
                except (ValueError, TypeError):
                    ha = 0.0
        if ha <= 0.0 and feat.hasGeometry():
            ha = float(feat.geometry().area()) / 10_000.0

        sc = 0.0
        if idx_score >= 0:
            v_s = feat.attribute(idx_score)
            if v_s is not None:
                try:
                    sc = float(v_s)
                except (ValueError, TypeError):
                    sc = 0.0

        drv = "None"
        if idx_drv >= 0:
            v_d = feat.attribute(idx_drv)
            if v_d:
                drv = str(v_d)

        b = buckets[cid]
        b["hex_count"] = int(b["hex_count"]) + 1
        b["area_ha"] = float(b["area_ha"]) + ha
        b["scores"].append(sc)
        b["drivers"][drv] += 1
        total_ha += ha

    rows: List[Dict[str, object]] = []
    for cid in range(1, 6):
        b = buckets[cid]
        cnt = int(b["hex_count"])
        area_ha = float(b["area_ha"])
        share_pct = (area_ha / total_ha * 100.0) if total_ha > 0 else 0.0
        sc_list = b["scores"]
        mean_sc = sum(sc_list) / len(sc_list) if sc_list else 0.0
        drv_counter: Counter = b["drivers"]
        top_drv = drv_counter.most_common(1)[0][0] if drv_counter else "-"
        rows.append({
            "class_id": cid,
            "class_label": b["class_label"],
            "hex_count": cnt,
            "area_ha": round(area_ha, 2),
            "share_pct": round(share_pct, 1),
            "mean_score": round(mean_sc, 2),
            "top_driver": top_drv,
        })
    return rows


def export_h3_table_csv(h3_layer, csv_path: str) -> int:
    """Export all attributes of ``h3_layer`` to a UTF-8 CSV file."""
    if h3_layer is None or not h3_layer.isValid():
        raise ValueError("A valid H3 hexagon layer is required for CSV export.")
    field_names = [f.name() for f in h3_layer.fields()]
    count = 0
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(field_names)
        for feat in h3_layer.getFeatures():
            writer.writerow([feat.attribute(fn) for fn in field_names])
            count += 1
    return count


def read_layer_json_property(h3_layer, key: str) -> Dict[str, object]:
    """Read a JSON-encoded custom property written by the 02H3Forge engines."""
    if h3_layer is None:
        return {}
    raw = h3_layer.customProperty(key, "")
    if not raw:
        return {}
    try:
        val = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}


def _read_numeric_column(h3_layer, field: str) -> List[float]:
    idx = h3_layer.fields().indexFromName(field)
    if idx < 0:
        return []
    out: List[float] = []
    for feat in h3_layer.getFeatures():
        v = feat.attribute(idx)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out.append(fv)
    return out


def _read_masked_flags(h3_layer) -> List[bool]:
    idx = h3_layer.fields().indexFromName("constraint_mask")
    flags: List[bool] = []
    for feat in h3_layer.getFeatures():
        if idx < 0:
            flags.append(False)
            continue
        v = feat.attribute(idx)
        try:
            flags.append(v is not None and int(v) == 0)
        except (TypeError, ValueError):
            flags.append(False)
    return flags


def svg_score_histogram(
    values: Sequence[float],
    breaks: Optional[Sequence[float]] = None,
    palette: Sequence[str] = THEME_PALETTES["risk"],
    n_bins: int = 20,
    width: int = 880,
    height: int = 220,
) -> str:
    """Inline SVG histogram of 0-100 scores, bars colored by their class band."""
    vals = [min(100.0, max(0.0, float(v))) for v in values if math.isfinite(float(v))]
    b_list = list(breaks) if breaks is not None and len(breaks) == 6 else [0, 20, 40, 60, 80, 100]
    if not vals:
        return "<p><i>No scores available.</i></p>"
    bins = [0] * n_bins
    for v in vals:
        k = min(n_bins - 1, int(v / (100.0 / n_bins)))
        bins[k] += 1
    peak = max(bins) or 1
    pad_l, pad_b, pad_t = 44, 28, 10
    plot_w = width - pad_l - 10
    plot_h = height - pad_b - pad_t
    bar_w = plot_w / n_bins
    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Composite score distribution" style="max-width:{width}px;">'
    ]
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + plot_h * (1.0 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - 10}" y2="{y:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="10" text-anchor="end" fill="#64748b">'
            f"{int(round(peak * frac))}</text>"
        )
    for k, cnt in enumerate(bins):
        mid = (k + 0.5) * (100.0 / n_bins)
        cls = 4
        for j in range(5):
            if mid <= float(b_list[j + 1]):
                cls = j
                break
        h = plot_h * cnt / peak
        x = pad_l + k * bar_w
        y = pad_t + plot_h - h
        parts.append(
            f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" '
            f'fill="{palette[cls]}" stroke="#334155" stroke-width="0.4">'
            f"<title>{k * 100 // n_bins}–{(k + 1) * 100 // n_bins}: {cnt} hexagons</title></rect>"
        )
    for tick in range(0, 101, 20):
        x = pad_l + plot_w * tick / 100.0
        parts.append(
            f'<text x="{x:.1f}" y="{height - 8}" font-size="10" text-anchor="middle" fill="#64748b">{tick}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_weight_bars(weights_map: Dict[str, float], width: int = 880) -> str:
    """Inline SVG horizontal bar chart of criterion weights (%)."""
    items = sorted(weights_map.items(), key=lambda kv: -float(kv[1]))
    if not items:
        return "<p><i>No active criteria.</i></p>"
    row_h = 24
    label_w = 230
    height = row_h * len(items) + 8
    peak = max(float(v) for _, v in items) or 1.0
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Criterion weights" style="max-width:{width}px;">'
    ]
    for i, (name, val) in enumerate(items):
        y = 4 + i * row_h
        bar = (width - label_w - 70) * float(val) / peak
        clean = html.escape(str(name).replace("norm_", ""))
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 15}" font-size="12" text-anchor="end" fill="#111827">{clean}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{bar:.1f}" height="{row_h - 8}" rx="3" fill="#13a0a0"/>'
            f'<text x="{label_w + bar + 6:.1f}" y="{y + 15}" font-size="12" fill="#0f6b6b" '
            f'font-weight="bold">{float(val):.1f}%</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _model_structure_html(model: Dict[str, object], run: Dict[str, object]) -> str:
    """Card describing groups, criteria (source → value function) and constraints."""
    if not model or not model.get("criteria"):
        return ""
    try:
        from .model import AGGREGATIONS, SOURCE_KINDS, ModelSpec

        spec = ModelSpec.from_dict(model)
    except Exception:
        return ""
    esc = html.escape
    eff = spec.effective_weights()
    groups = spec.active_groups()
    g_tot = sum(max(0.0, g.weight) for g in groups) or 1.0
    kinds = {k: v.split(" (")[0] for k, v in SOURCE_KINDS}
    rows = ""
    for g in groups:
        crits = spec.criteria_in(g.id, active_only=True)
        rows += (
            f"<tr class='grp'><td colspan='4'><b>{esc(g.label)}</b> — {g.weight / g_tot * 100:.1f}% · "
            f"{esc(g.method.upper())}{f' (α = {g.owa_alpha:g})' if g.method == 'owa' else ''}</td></tr>"
        )
        for c in crits:
            src = c.source.layer_name or c.source.field
            detail = f" · {c.source.method}" if c.source.kind in ("vector", "raster") else ""
            rows += (
                f"<tr><td>{esc(c.label)}</td><td>{esc(kinds.get(c.source.kind, c.source.kind))}: "
                f"{esc(src)}{esc(detail)}</td><td>{esc(c.value.describe())}</td>"
                f"<td class='num'><b>{eff.get(c.id, 0.0):.1f}%</b></td></tr>"
            )
    excluded_by = run.get("excluded_by", {}) if isinstance(run.get("excluded_by"), dict) else {}
    cons = "".join(
        f"<li>{esc(k.label)} — {esc(k.describe())}"
        + (f" <span class='muted'>({int(excluded_by[k.label]):,} hexagons)</span>" if k.label in excluded_by else "")
        + "</li>"
        for k in spec.constraints if k.active
    ) or "<li class='muted'>None</li>"
    agg = dict(AGGREGATIONS).get(spec.aggregation, spec.aggregation) if len(groups) > 1 else "single group"
    return f"""<div class="card">
    <h2>Model Structure — {esc(spec.name)}</h2>
    <p>Groups combined by <b>{esc(agg)}</b>. Effective weight = group weight × within-group weight.</p>
    <table><thead><tr><th>Criterion</th><th>Source</th><th>Value function</th>
    <th class="num">Effective weight</th></tr></thead><tbody>{rows}</tbody></table>
    <h3 style="font-size:13px;margin:14px 0 4px 0;">Constraints (hard exclusions)</h3>
    <ul style="margin:0;padding-left:18px;font-size:13px;">{cons}</ul>
  </div>"""


def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='kpi-sub'>{html.escape(sub)}</div>" if sub else ""
    return (
        f"<div class='kpi'><div class='kpi-label'>{html.escape(label)}</div>"
        f"<div class='kpi-value'>{html.escape(value)}</div>{sub_html}</div>"
    )


def export_h3_html_report(
    h3_layer,
    html_path: str,
    weights_map: Dict[str, float],
    compose_method: str = "wlc",
    theme: str = "risk",
) -> str:
    """Generate a self-contained HTML Executive Report for ``h3_layer``.

    Sections: KPIs, score distribution (inline SVG), class bands with XAI
    drivers, criterion weights, Monte Carlo robustness, spatial clustering
    (Moran's I + Gi* hotspots, when computed), and run parameters.
    """
    summary_rows = compute_class_summary(h3_layer, theme=theme)
    total_hex = sum(int(r["hex_count"]) for r in summary_rows)
    total_ha = sum(float(r["area_ha"]) for r in summary_rows)
    palette = THEME_PALETTES.get(theme, THEME_PALETTES["risk"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_info = read_layer_json_property(h3_layer, "h3forge/last_run")
    hot_info = read_layer_json_property(h3_layer, "h3forge/hotspot")
    sens_info = read_layer_json_property(h3_layer, "h3forge/sensitivity")
    model_info = read_layer_json_property(h3_layer, "h3forge/model")
    model_html = _model_structure_html(model_info, run_info)

    scores = _read_numeric_column(h3_layer, "comp_score")
    masked = _read_masked_flags(h3_layer)
    n_excluded = sum(1 for m in masked if m)
    feasible_scores = (
        [s for s, m in zip(scores, masked) if not m] if len(scores) == len(masked) else scores
    )
    mean_sc = sum(feasible_scores) / len(feasible_scores) if feasible_scores else 0.0
    mc_std = _read_numeric_column(h3_layer, "mc_std")
    mean_std = sum(mc_std) / len(mc_std) if mc_std else 0.0
    top_two = [r for r in summary_rows if int(r["class_id"]) >= 4]
    top_two_ha = sum(float(r["area_ha"]) for r in top_two)
    top_two_pct = (top_two_ha / total_ha * 100.0) if total_ha > 0 else 0.0
    upper_label = "High + Very High" if theme == "suitability" else "High + Critical"

    breaks = run_info.get("breaks") if isinstance(run_info.get("breaks"), list) else None
    hist_svg = svg_score_histogram(feasible_scores, breaks=breaks, palette=palette)
    weights_svg = svg_weight_bars(weights_map)

    weights_html = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td>"
        f"<td class='num'><b>{v:.2f}%</b></td></tr>"
        for k, v in sorted(weights_map.items(), key=lambda kv: -float(kv[1]))
    )

    classes_html = ""
    for i, r in enumerate(summary_rows):
        color = palette[min(i, len(palette) - 1)]
        classes_html += (
            f"<tr>"
            f"<td><span class='swatch' style='background:{color};'></span>"
            f"<b>{r['class_id']} — {html.escape(str(r['class_label']))}</b></td>"
            f"<td class='num'>{r['hex_count']:,}</td>"
            f"<td class='num'>{r['area_ha']:.2f} ha</td>"
            f"<td class='num'>{r['share_pct']:.1f}%"
            f"<div class='share'><span style='width:{min(100.0, float(r['share_pct'])):.1f}%;"
            f"background:{color};'></span></div></td>"
            f"<td class='num'>{r['mean_score']:.2f}</td>"
            f"<td><code>{html.escape(str(r['top_driver']))}</code></td>"
            f"</tr>"
        )

    # Spatial clustering section (optional)
    spatial_html = (
        "<p class='muted'>Run <b>Hotspot Analysis (Gi*)</b> in Stage 4 to add spatial "
        "clustering diagnostics to this report.</p>"
    )
    if hot_info:
        moran = hot_info.get("moran", {}) if isinstance(hot_info.get("moran"), dict) else {}
        counts = hot_info.get("counts", {}) if isinstance(hot_info.get("counts"), dict) else {}
        i_val = moran.get("I")
        z_val = moran.get("z")
        p_val = moran.get("p")
        try:
            sig = float(p_val) < 0.05
            pattern = (
                "significantly clustered" if sig and float(z_val) > 0
                else "significantly dispersed" if sig
                else "not significantly different from spatial randomness"
            )
            moran_txt = (
                f"Global Moran's I = <b>{float(i_val):.4f}</b> (z = {float(z_val):.2f}, "
                f"{'p &lt; 0.0001' if float(p_val) < 1e-4 else f'p = {float(p_val):.4f}'}) — the <code>{html.escape(str(hot_info.get('value_field', 'comp_score')))}"
                f"</code> surface is <b>{pattern}</b>."
            )
        except (TypeError, ValueError):
            moran_txt = "Global Moran's I could not be computed (too few or constant values)."
        hs_rows = "".join(
            f"<tr><td><span class='swatch' style='background:{HOTSPOT_COLORS.get(code, '#cccccc')};'></span>"
            f"{html.escape(lbl)}</td><td class='num'>{int(counts.get(lbl, 0)):,}</td></tr>"
            for code, lbl in HOTSPOT_ORDER
            if lbl in counts
        )
        if int(counts.get("Excluded", 0)):
            hs_rows += f"<tr><td>Excluded (hard constraint)</td><td class='num'>{int(counts['Excluded']):,}</td></tr>"
        spatial_html = (
            f"<p>{moran_txt}</p>"
            f"<table><thead><tr><th>Getis-Ord Gi* class (H3 1-ring)</th>"
            f"<th class='num'>Hexagons</th></tr></thead><tbody>{hs_rows}</tbody></table>"
        )

    sens_html = (
        "<p class='muted'>Run <b>Weight Sensitivity</b> in Stage 4 to test how strongly "
        "the map depends on each weight.</p>"
    )
    sens_rows = sens_info.get("rows") if isinstance(sens_info.get("rows"), list) else []
    if sens_rows:
        body = ""
        for r in sens_rows:
            try:
                pct = float(r.get("class_change_pct", 0.0))
                level = "high" if pct >= 10.0 else "moderate" if pct >= 3.0 else "low"
                body += (
                    f"<tr><td><code>{html.escape(str(r.get('criterion', '')))}</code></td>"
                    f"<td class='num'>{float(r.get('weight', 0.0)):.1f}%</td>"
                    f"<td class='num'>{float(r.get('mean_abs_change', 0.0)):.2f}</td>"
                    f"<td class='num'>{float(r.get('max_abs_change', 0.0)):.2f}</td>"
                    f"<td class='num'>{pct:.1f}%</td>"
                    f"<td><span class='pill {level}'>{level}</span></td></tr>"
                )
            except (TypeError, ValueError, AttributeError):
                continue
        deltas = ", ".join(f"{float(d):+.0f}%" for d in sens_info.get("deltas_pct", []))
        sens_html = (
            f"<p>Each weight was changed by {html.escape(deltas)} of its own value (others "
            f"rebalanced to 100%) using <b>{html.escape(str(sens_info.get('method', '')).upper())}</b>. "
            f"Worst case per criterion, most influential first.</p>"
            f"<table><thead><tr><th>Criterion</th><th class='num'>Weight</th>"
            f"<th class='num'>Mean |Δ score|</th><th class='num'>Max |Δ score|</th>"
            f"<th class='num'>Hexes changing class</th><th>Influence</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    params = [
        ("Overlay engine", str(run_info.get("method", compose_method)).upper() + (
            f" (WOWA α = {float(run_info['owa_alpha']):.2f}, orness {1.0 / (1.0 + float(run_info['owa_alpha'])):.2f})"
            if run_info.get("owa_alpha") is not None else ""
        )),
        ("Map theme", str(run_info.get("theme", theme)).title()),
        ("Class breaks", ", ".join(f"{float(b):.1f}" for b in breaks) if breaks else "Equal interval"),
        ("Classification", str(run_info.get("class_method", "equal")).title()),
        ("H3 1-ring contagion α", f"{float(run_info.get('alpha', 0.0)):.2f}"),
        ("Hard constraints", "Applied" if run_info.get("constraints", True) else "Ignored"),
        ("Monte Carlo", (
            f"{int(run_info.get('mc_runs', 50))} runs, ±{float(run_info.get('mc_perturbation', 0.15)) * 100:.0f}% weights"
            if run_info.get("montecarlo", True) else "Off"
        )),
        ("Hotspot neighbourhood", (
            f"{int(hot_info.get('k_ring', 1))}-ring, "
            + ("Benjamini-Hochberg FDR" if hot_info.get("fdr", False) else "no multiple-testing correction")
        ) if hot_info else "n/a"),
        ("Grid engine", str(h3_layer.customProperty("h3forge/engine", "n/a"))),
        ("CRS", h3_layer.crs().authid()),
    ]
    params_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in params
    )

    kpis = "".join([
        _kpi("H3 hexagons", f"{total_hex:,}", f"{n_excluded:,} excluded" if n_excluded else ""),
        _kpi("Evaluated area", f"{total_ha:,.1f} ha"),
        _kpi("Mean composite score", f"{mean_sc:.1f}", "feasible hexagons"),
        _kpi(upper_label, f"{top_two_pct:.1f}%", f"{top_two_ha:,.1f} ha"),
        _kpi("Mean MC std", f"{mean_std:.2f}", "lower = more robust"),
    ])

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>02H3Forge Executive Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f4f7fb; color: #111827; margin: 0; padding: 24px 16px; }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  .card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
          padding: 20px; margin: 0 0 18px 0; }}
  .banner {{ background: linear-gradient(90deg, #0f6b6b, #13a0a0); color: #ffffff;
            padding: 18px 22px; border-radius: 10px; margin: 0 0 18px 0; }}
  h1 {{ margin: 0 0 6px 0; font-size: 21px; }}
  h2 {{ font-size: 15px; color: #0f6b6b; border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px; margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 7px 10px; text-align: left; vertical-align: middle; }}
  thead th {{ background: #f1f5f9; font-weight: 600; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .meta {{ font-size: 12px; opacity: 0.92; }}
  .muted {{ color: #64748b; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; border: 1px solid #334155;
            margin-right: 6px; vertical-align: middle; border-radius: 2px; }}
  .share {{ height: 4px; background: #f1f5f9; border-radius: 2px; margin-top: 4px; }}
  .share span {{ display: block; height: 4px; border-radius: 2px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
  .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; }}
  .kpi-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .04em; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; color: #0f6b6b; font-variant-numeric: tabular-nums; }}
  .kpi-sub {{ font-size: 11px; color: #64748b; }}
  tr.grp td {{ background: #f1f5f9; }}
  .pill {{ display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px; font-weight: 600; }}
  .pill.high {{ background: #fde2e1; color: #b30000; }}
  .pill.moderate {{ background: #fef3c7; color: #92400e; }}
  .pill.low {{ background: #dcfce7; color: #166534; }}
  footer {{ text-align: center; font-size: 11px; color: #64748b; padding: 8px 0 20px 0; }}
  @media print {{ body {{ background: #ffffff; }} .card, .banner {{ break-inside: avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
  <div class="banner">
    <h1>02H3Forge — H3 Hexagonal Risk, Vulnerability &amp; Suitability Report</h1>
    <div class="meta">Layer: <b>{html.escape(h3_layer.name())}</b> | CRS: {html.escape(h3_layer.crs().authid())} |
    Method: <b>{html.escape(compose_method.upper())}</b> | Theme: <b>{html.escape(theme.title())}</b> | Generated: {ts}</div>
  </div>
  <div class="card">
    <h2>1. Key Figures</h2>
    <div class="kpis">{kpis}</div>
  </div>
  {model_html}
  <div class="card">
    <h2>2. Composite Score Distribution (0–100)</h2>
    {hist_svg}
  </div>
  <div class="card">
    <h2>3. Class Distribution &amp; Local Explainability (XAI Top Drivers)</h2>
    <table>
      <thead>
        <tr>
          <th>Class Band</th>
          <th class="num">H3 Hexagons</th>
          <th class="num">Area (ha)</th>
          <th class="num">Area Share</th>
          <th class="num">Mean Score</th>
          <th>Dominant Local Driver (XAI)</th>
        </tr>
      </thead>
      <tbody>{classes_html}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>4. Active Normalized Criteria &amp; Weights (Sum = 100%)</h2>
    {weights_svg}
    <table>
      <thead><tr><th>Criterion Column</th><th class="num">Weight (%)</th></tr></thead>
      <tbody>{weights_html}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>5. Spatial Clustering (Moran's I &amp; Getis-Ord Gi* Hotspots)</h2>
    {spatial_html}
  </div>
  <div class="card">
    <h2>6. Weight Sensitivity (One-at-a-Time)</h2>
    {sens_html}
  </div>
  <div class="card">
    <h2>7. Run Parameters (Reproducibility)</h2>
    <table><tbody>{params_html}</tbody></table>
  </div>
  <footer>Generated by 02H3Forge · Uber H3 hexagonal MCDA studio for QGIS</footer>
</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(doc)
    return html_path
