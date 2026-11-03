# -*- coding: utf-8 -*-
"""Cartographic Styling, Class Area Diagnostics, and HTML/CSV Exporter for 02H3Forge."""
from __future__ import annotations

import csv
import html
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .mcda_composer import CLASS_LABELS_RISK, CLASS_LABELS_SUITABILITY


THEME_PALETTES: Dict[str, Tuple[str, ...]] = {
    "risk": ("#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"),
    "suitability": ("#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"),
    "susceptibility": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
    "magma": ("#fcfdbf", "#fe9f6d", "#de4968", "#8c2981", "#3b0f70"),
}

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


def export_h3_html_report(
    h3_layer,
    html_path: str,
    weights_map: Dict[str, float],
    compose_method: str = "wlc",
    theme: str = "risk",
) -> str:
    """Generate a self-contained English HTML Executive Report for ``h3_layer``."""
    summary_rows = compute_class_summary(h3_layer, theme=theme)
    total_hex = sum(int(r["hex_count"]) for r in summary_rows)
    total_ha = sum(float(r["area_ha"]) for r in summary_rows)
    palette = THEME_PALETTES.get(theme, THEME_PALETTES["risk"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    weights_html = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td>"
        f"<td style='text-align:right;font-weight:bold;'>{v:.2f}%</td></tr>"
        for k, v in weights_map.items()
    )

    classes_html = ""
    for i, r in enumerate(summary_rows):
        color = palette[min(i, len(palette) - 1)]
        classes_html += (
            f"<tr>"
            f"<td><span style='display:inline-block;width:14px;height:14px;"
            f"background:{color};border:1px solid #334155;margin-right:6px;vertical-align:middle;'></span>"
            f"<b>{r['class_id']} — {html.escape(str(r['class_label']))}</b></td>"
            f"<td style='text-align:right;'>{r['hex_count']}</td>"
            f"<td style='text-align:right;'>{r['area_ha']:.2f} ha</td>"
            f"<td style='text-align:right;'>{r['share_pct']:.1f}%</td>"
            f"<td style='text-align:right;'>{r['mean_score']:.2f}</td>"
            f"<td><code>{html.escape(str(r['top_driver']))}</code></td>"
            f"</tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>02H3Forge Executive Synthesis Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f4f7fb; color: #111827; margin: 24px; }}
  .card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px;
          padding: 20px; max-width: 960px; margin: 0 auto 20px auto; }}
  .banner {{ background: linear-gradient(90deg, #0f6b6b, #13a0a0); color: #ffffff;
            padding: 16px 20px; border-radius: 8px; max-width: 960px; margin: 0 auto 20px auto; }}
  h1 {{ margin: 0 0 6px 0; font-size: 20px; }}
  h2 {{ font-size: 15px; color: #0f6b6b; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .meta {{ font-size: 12px; opacity: 0.9; }}
</style>
</head>
<body>
  <div class="banner">
    <h1>02H3Forge — H3 Hexagonal Risk, Vulnerability &amp; Suitability Report</h1>
    <div class="meta">Layer: <b>{html.escape(h3_layer.name())}</b> | CRS: {html.escape(h3_layer.crs().authid())} |
    Method: <b>{html.escape(compose_method.upper())}</b> | Theme: <b>{html.escape(theme.title())}</b> | Generated: {ts}</div>
  </div>
  <div class="card">
    <h2>1. Study Area &amp; H3 Grid Overview</h2>
    <p>Total H3 Hexagons: <b>{total_hex:,}</b> &nbsp;|&nbsp; Total Evaluated Area: <b>{total_ha:.2f} ha</b></p>
    <h2>2. Class Distribution &amp; Local Explainability (XAI Top Drivers)</h2>
    <table>
      <thead>
        <tr>
          <th>Class Band</th>
          <th style="text-align:right;">H3 Hexagons</th>
          <th style="text-align:right;">Area (ha)</th>
          <th style="text-align:right;">Area Share</th>
          <th style="text-align:right;">Mean Score</th>
          <th>Dominant Local Driver (XAI)</th>
        </tr>
      </thead>
      <tbody>{classes_html}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>3. Active Normalized Criteria &amp; Weights (Sum = 100%)</h2>
    <table>
      <thead><tr><th>Criterion Column</th><th style="text-align:right;">Weight (%)</th></tr></thead>
      <tbody>{weights_html}</tbody>
    </table>
  </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(doc)
    return html_path
