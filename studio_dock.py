# -*- coding: utf-8 -*-
"""02H3Forge Right-Docked Studio Widget (PlanX Studio visual language).

Provides a guided 5-stage workflow on Uber H3 hexagonal grids:
  Stage 0: H3 Hex Grid Generator (Default Res 10 ≈ 1.50 ha) & Graduated Multi-Ring Buffers
  Stage 1: Multi-Source Spatial Aggregation (Vector Join-by-Location & Raster Zonal Stats)
  Stage 2: Criterion Normalization (0-100) with Benefit (+) / Cost (-) Polarity & Fuzzy Functions
  Stage 3: Coupled Auto-Balancing Weight Sliders (Sum = 100%) + AHP/CRITIC/Entropy/PCA + Overlay
  Stage 4: Thematic Classification, Local XAI Attribution, & HTML/CSV Executive Report
"""
from __future__ import annotations

import os
import webbrowser
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsMapLayer,
    QgsProject,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QSignalBlocker, Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDockWidget,
)

from .ahp_dialog import AhpPairwiseDialog
from .core.h3_grid_engine import (
    DEFAULT_H3_RESOLUTION,
    H3_RESOLUTION_SPECS,
    build_h3_grid_layer,
    sanitize_field_slug,
)
from .core.mcda_composer import (
    COMPOSE_METHODS,
    critic_weights_pct,
    entropy_weights_pct,
    equal_weights_pct,
    extract_h3_criteria_matrix,
    pca_weights_pct,
    rebalance_coupled_weights,
    run_h3_overlay_on_layer,
)
from .core.normalizer import (
    NORMALIZATION_METHODS,
    POLARITY_OPTIONS,
    normalize_h3_field,
)
from .core.proximity_rings import (
    MULTI_RING_PRESETS,
    RingSpec,
    apply_multiring_buffer_to_h3,
)
from .core.spatial_aggregator import (
    RASTER_AGG_METHODS,
    VECTOR_AGG_METHODS,
    aggregate_raster_to_h3,
    aggregate_vector_to_h3,
)
from .core.styler_and_report import (
    THEME_OPTIONS,
    apply_h3_graduated_style,
    compute_class_summary,
    export_h3_html_report,
    export_h3_table_csv,
)


# PlanX Studio QSS — note: zero background/border rules on QCheckBox, QRadioButton,
# or QAbstractSpinBox so Qt6 native indicators never blank out (TRAPS.md §3.7).
_QSS = """
QWidget#h3ForgeDockBody { background: #f4f7fb; }
QLabel#h3ForgeHeader {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f6b6b,stop:1 #13a0a0);
    color: white; font-weight: bold; font-size: 13px;
    padding: 10px 12px; border-radius: 6px;
}
QGroupBox {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
    margin-top: 8px; padding-top: 10px; font-weight: bold; color: #0f6b6b;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QTableWidget, QTextEdit {
    background: #ffffff; color: #111827; border: 1px solid #cbd5e1; font-size: 12px;
}
QLineEdit, QComboBox {
    background: #ffffff; color: #111827; border: 1px solid #94a3b8;
    border-radius: 4px; padding: 4px 6px;
}
QPushButton {
    background: #e2e8f0; color: #111827; border: 1px solid #94a3b8;
    border-radius: 4px; padding: 5px 8px;
}
QPushButton:hover { background: #cbd5e1; }
QPushButton#primaryTealBtn {
    background: #13a0a0; color: white; border: none; font-weight: bold; padding: 7px 10px;
}
QPushButton#primaryTealBtn:hover { background: #0f6b6b; }
QPushButton#rateGoldBtn {
    background: #cf9836; color: white; border: none; font-weight: bold; padding: 5px 8px;
}
QPushButton#rateGoldBtn:hover { background: #b8862d; }
QCheckBox, QLabel { color: #111827; }
"""


class O2H3ForgeStudioDock(QDockWidget):
    """Right-docked 5-stage Uber H3 MCDA, Vulnerability, Risk & Suitability Studio."""

    def __init__(self, iface, parent=None):
        super().__init__("02H3Forge Studio", parent)
        self.iface = iface
        self.setObjectName("O2H3ForgeStudioDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )

        # Internal state for Stage 3 coupled weight rows:
        # Each entry: {"field": str, "chk": QCheckBox, "slider": QSlider,
        #              "spin": QDoubleSpinBox, "lock": QCheckBox}
        self._weight_rows: List[Dict[str, object]] = []
        self._last_breaks: Optional[List[float]] = None

        self._build_ui()
        self._connect_project_signals()
        self.refresh_layer_combos()

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        body = QWidget()
        body.setObjectName("h3ForgeDockBody")
        root = QVBoxLayout(body)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QLabel("02H3Forge — H3 Risk, Vulnerability & Suitability Studio")
        header.setObjectName("h3ForgeHeader")
        header.setWordWrap(True)
        root.addWidget(header)

        # Active H3 Grid bar
        grid_bar = QHBoxLayout()
        grid_bar.addWidget(QLabel("<b>Active H3 Layer:</b>"))
        self.h3_layer_combo = QComboBox()
        self.h3_layer_combo.currentIndexChanged.connect(self._on_active_h3_layer_changed)
        grid_bar.addWidget(self.h3_layer_combo, 1)
        refresh_btn = QPushButton("↻")
        refresh_btn.setToolTip("Refresh layers and H3 attribute columns")
        refresh_btn.setFixedWidth(30)
        refresh_btn.clicked.connect(self.refresh_layer_combos)
        grid_bar.addWidget(refresh_btn)
        root.addLayout(grid_bar)

        # 5-Stage Tab Widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab0_grid_and_buffers(), "0. Grid & Buffers")
        self.tabs.addTab(self._build_tab1_aggregate(), "1. Aggregate")
        self.tabs.addTab(self._build_tab2_normalize(), "2. Normalize")
        self.tabs.addTab(self._build_tab3_weights_overlay(), "3. Weights & Overlay")
        self.tabs.addTab(self._build_tab4_results(), "4. Results")
        root.addWidget(self.tabs, 1)

        # Log console + footer
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(78)
        self.log_box.setPlaceholderText("02H3Forge execution log...")
        root.addWidget(self.log_box)

        footer = QHBoxLayout()
        repo_btn = QPushButton("GitHub Repo")
        repo_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/YusufEminoglu/zero2h3forge"))
        )
        footer.addWidget(repo_btn)
        rate_btn = QPushButton("★ Rate on GeoPhilo")
        rate_btn.setObjectName("rateGoldBtn")
        rate_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://geophilo.com/feedback/?plugin=zero2h3forge&v=0.1.0")
            )
        )
        footer.addWidget(rate_btn)
        root.addLayout(footer)

        body.setStyleSheet(_QSS)
        self.setWidget(body)

    def _build_tab0_grid_and_buffers(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        # Group A: H3 Hexagonal Grid Generator
        grp_grid = QGroupBox("A. Uber H3 Hexagonal Grid Generator")
        form_g = QFormLayout(grp_grid)

        self.extent_source_combo = QComboBox()
        self.extent_source_combo.addItem("Reference Layer Extent / Polygon Boundary", "layer")
        self.extent_source_combo.addItem("Current Map Canvas Extent", "canvas")
        form_g.addRow("Extent Source:", self.extent_source_combo)

        self.boundary_layer_combo = QComboBox()
        form_g.addRow("Reference Layer:", self.boundary_layer_combo)

        self.res_combo = QComboBox()
        default_idx = 0
        for idx, res in enumerate(sorted(H3_RESOLUTION_SPECS.keys())):
            spec = H3_RESOLUTION_SPECS[res]
            self.res_combo.addItem(str(spec["label"]), int(res))
            if res == DEFAULT_H3_RESOLUTION:
                default_idx = idx
        self.res_combo.setCurrentIndex(default_idx)
        form_g.addRow("H3 Resolution:", self.res_combo)

        self.coverage_combo = QComboBox()
        self.coverage_combo.addItem("Intersects Polygon Boundary (Full Coverage)", "intersect")
        self.coverage_combo.addItem("Hexagon Centroid Inside Boundary", "centroid")
        self.coverage_combo.addItem("Full Bounding Box Extent", "bbox")
        form_g.addRow("Coverage Mode:", self.coverage_combo)

        self.gen_grid_btn = QPushButton("Generate H3 Hexagonal Grid")
        self.gen_grid_btn.setObjectName("primaryTealBtn")
        self.gen_grid_btn.clicked.connect(self.on_generate_h3_grid)
        form_g.addRow(self.gen_grid_btn)
        lay.addWidget(grp_grid)

        # Group B: Graduated Multi-Ring Buffer Scorer
        grp_buf = QGroupBox("B. Graduated Multi-Ring Proximity Buffer Scorer")
        vbox_b = QVBoxLayout(grp_buf)
        form_b = QFormLayout()

        self.buf_layer_combo = QComboBox()
        self.buf_layer_combo.currentIndexChanged.connect(self._on_buf_layer_changed)
        form_b.addRow("Hazard / Target Layer:", self.buf_layer_combo)

        self.buf_name_edit = QLineEdit("power_line")
        self.buf_name_edit.setPlaceholderText("e.g. power_line, fault, stream")
        form_b.addRow("Criterion Short Name:", self.buf_name_edit)

        self.buf_preset_combo = QComboBox()
        for preset_name in MULTI_RING_PRESETS:
            self.buf_preset_combo.addItem(preset_name)
        self.buf_preset_combo.currentIndexChanged.connect(self._load_selected_ring_preset)
        form_b.addRow("Distance Ring Preset:", self.buf_preset_combo)
        vbox_b.addLayout(form_b)

        self.ring_table = QTableWidget(0, 4)
        self.ring_table.setHorizontalHeaderLabels(
            ["Min (m)", "Max (m)", "Score (0-100)", "Exclude?"]
        )
        self.ring_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ring_table.setMaximumHeight(125)
        vbox_b.addWidget(self.ring_table)

        ring_btns = QHBoxLayout()
        add_ring_btn = QPushButton("+ Add Ring")
        add_ring_btn.clicked.connect(lambda: self._append_ring_row(500.0, 1000.0, 15.0, False))
        ring_btns.addWidget(add_ring_btn)
        rem_ring_btn = QPushButton("- Remove Ring")
        rem_ring_btn.clicked.connect(self._remove_selected_ring_row)
        ring_btns.addWidget(rem_ring_btn)
        ring_btns.addWidget(QLabel("Outside Score:"))
        self.outside_score_spin = QDoubleSpinBox()
        self.outside_score_spin.setRange(0.0, 100.0)
        self.outside_score_spin.setValue(0.0)
        ring_btns.addWidget(self.outside_score_spin)
        vbox_b.addLayout(ring_btns)

        self.auto_norm_ring_chk = QCheckBox(
            "Also register ring score directly as normalized criterion (norm_*)"
        )
        self.auto_norm_ring_chk.setChecked(True)
        vbox_b.addWidget(self.auto_norm_ring_chk)

        self.run_buf_btn = QPushButton("Compute Multi-Ring Buffer into H3")
        self.run_buf_btn.setObjectName("primaryTealBtn")
        self.run_buf_btn.clicked.connect(self.on_run_multiring_buffer)
        vbox_b.addWidget(self.run_buf_btn)

        lay.addWidget(grp_buf, 1)
        self._load_selected_ring_preset()
        return page

    def _build_tab1_aggregate(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        # Vector Join by Location
        grp_vec = QGroupBox("A. Vector Spatial Aggregation (Join by Location → H3)")
        form_v = QFormLayout(grp_vec)

        self.agg_vec_layer_combo = QComboBox()
        self.agg_vec_layer_combo.currentIndexChanged.connect(self._on_agg_vec_layer_changed)
        form_v.addRow("Source Vector Layer:", self.agg_vec_layer_combo)

        self.agg_vec_method_combo = QComboBox()
        for key, lbl in VECTOR_AGG_METHODS:
            self.agg_vec_method_combo.addItem(lbl, key)
        form_v.addRow("Aggregation Metric:", self.agg_vec_method_combo)

        self.agg_vec_field_combo = QComboBox()
        form_v.addRow("Numeric Attribute Field:", self.agg_vec_field_combo)

        self.agg_vec_out_edit = QLineEdit()
        self.agg_vec_out_edit.setPlaceholderText("Auto: <layer>_<metric>")
        form_v.addRow("Output Criterion Name:", self.agg_vec_out_edit)

        self.run_agg_vec_btn = QPushButton("Aggregate Vector Layer into H3 (raw_*)")
        self.run_agg_vec_btn.setObjectName("primaryTealBtn")
        self.run_agg_vec_btn.clicked.connect(self.on_aggregate_vector)
        form_v.addRow(self.run_agg_vec_btn)
        lay.addWidget(grp_vec)

        # Raster Zonal Sampling
        grp_ras = QGroupBox("B. Raster Zonal Aggregation (Raster → H3)")
        form_r = QFormLayout(grp_ras)

        self.agg_ras_layer_combo = QComboBox()
        self.agg_ras_layer_combo.currentIndexChanged.connect(self._on_agg_ras_layer_changed)
        form_r.addRow("Source Raster Layer:", self.agg_ras_layer_combo)

        self.agg_ras_band_spin = QSpinBox()
        self.agg_ras_band_spin.setRange(1, 64)
        self.agg_ras_band_spin.setValue(1)
        form_r.addRow("Raster Band:", self.agg_ras_band_spin)

        self.agg_ras_method_combo = QComboBox()
        for key, lbl in RASTER_AGG_METHODS:
            self.agg_ras_method_combo.addItem(lbl, key)
        form_r.addRow("Zonal Statistic:", self.agg_ras_method_combo)

        self.agg_ras_out_edit = QLineEdit()
        self.agg_ras_out_edit.setPlaceholderText("Auto: <raster>_<stat>")
        form_r.addRow("Output Criterion Name:", self.agg_ras_out_edit)

        self.run_agg_ras_btn = QPushButton("Sample Raster Layer into H3 (raw_*)")
        self.run_agg_ras_btn.setObjectName("primaryTealBtn")
        self.run_agg_ras_btn.clicked.connect(self.on_aggregate_raster)
        form_r.addRow(self.run_agg_ras_btn)
        lay.addWidget(grp_ras)

        lay.addStretch(1)
        return page

    def _build_tab2_normalize(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        grp_norm = QGroupBox("Standardize Criteria to 0–100 (Benefit vs. Cost Polarity)")
        vbox = QVBoxLayout(grp_norm)

        hint = QLabel(
            "<b>Benefit (+)</b>: Higher raw value → 100 (e.g. building density in risk, transit access in suitability).<br>"
            "<b>Cost (-)</b>: Lower raw value → 100 (e.g. distance to fault/power line in hazard, slope in suitability)."
        )
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        self.norm_table = QTableWidget(0, 6)
        self.norm_table.setHorizontalHeaderLabels(
            ["Use", "Source Field", "Polarity (+/-)", "Method", "Low / Mid", "High / Spread"]
        )
        self.norm_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vbox.addWidget(self.norm_table, 1)

        btn_bar = QHBoxLayout()
        refresh_norm_btn = QPushButton("Refresh H3 Fields")
        refresh_norm_btn.clicked.connect(self.populate_normalization_table)
        btn_bar.addWidget(refresh_norm_btn)

        self.run_norm_btn = QPushButton("Normalize Selected Fields to 0–100 (norm_*)")
        self.run_norm_btn.setObjectName("primaryTealBtn")
        self.run_norm_btn.clicked.connect(self.on_run_normalization)
        btn_bar.addWidget(self.run_norm_btn, 1)
        vbox.addLayout(btn_bar)

        lay.addWidget(grp_norm, 1)
        return page

    def _build_tab3_weights_overlay(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        grp_w = QGroupBox("A. Coupled Criteria Weights (Auto-Balanced Sum = 100%)")
        vbox_w = QVBoxLayout(grp_w)

        top_bar = QHBoxLayout()
        self.total_weight_lbl = QLabel("<b>Total Weight: 100.0%</b>")
        top_bar.addWidget(self.total_weight_lbl)
        top_bar.addStretch(1)
        eq_btn = QPushButton("Equal (1/N)")
        eq_btn.clicked.connect(self.on_weights_equal)
        top_bar.addWidget(eq_btn)
        critic_btn = QPushButton("CRITIC")
        critic_btn.setToolTip("Objective CRITIC weights (contrast × conflict)")
        critic_btn.clicked.connect(self.on_weights_critic)
        top_bar.addWidget(critic_btn)
        ent_btn = QPushButton("Entropy")
        ent_btn.setToolTip("Shannon Entropy objective weights")
        ent_btn.clicked.connect(self.on_weights_entropy)
        top_bar.addWidget(ent_btn)
        pca_btn = QPushButton("PCA")
        pca_btn.setToolTip("First Principal Component loadings")
        pca_btn.clicked.connect(self.on_weights_pca)
        top_bar.addWidget(pca_btn)
        ahp_btn = QPushButton("AHP Matrix...")
        ahp_btn.setToolTip("Interactive Saaty pairwise matrix with CR check")
        ahp_btn.clicked.connect(self.on_weights_ahp)
        top_bar.addWidget(ahp_btn)
        vbox_w.addLayout(top_bar)

        self.sliders_scroll = QScrollArea()
        self.sliders_scroll.setWidgetResizable(True)
        self.sliders_container = QWidget()
        self.sliders_layout = QVBoxLayout(self.sliders_container)
        self.sliders_layout.setContentsMargins(4, 4, 4, 4)
        self.sliders_layout.setSpacing(4)
        self.sliders_scroll.setWidget(self.sliders_container)
        vbox_w.addWidget(self.sliders_scroll, 1)
        lay.addWidget(grp_w, 1)

        grp_c = QGroupBox("B. Multi-Criteria Overlay & Spatial Contagion")
        form_c = QFormLayout(grp_c)

        self.compose_method_combo = QComboBox()
        for key, lbl in COMPOSE_METHODS:
            self.compose_method_combo.addItem(lbl, key)
        form_c.addRow("Overlay Engine:", self.compose_method_combo)

        self.theme_combo = QComboBox()
        for key, lbl in THEME_OPTIONS:
            self.theme_combo.addItem(lbl, key)
        form_c.addRow("Map Theme / Ramp:", self.theme_combo)

        self.class_mode_combo = QComboBox()
        self.class_mode_combo.addItem("Equal Interval (0–20–40–60–80–100)", "equal")
        self.class_mode_combo.addItem("Quantile (5 Equal-Count Bands)", "quantile")
        form_c.addRow("5-Class Breaks:", self.class_mode_combo)

        self.smooth_alpha_spin = QDoubleSpinBox()
        self.smooth_alpha_spin.setRange(0.0, 0.50)
        self.smooth_alpha_spin.setSingleStep(0.05)
        self.smooth_alpha_spin.setValue(0.0)
        self.smooth_alpha_spin.setToolTip(
            "0.0 = no smoothing; 0.20 = blend 20% of 1-ring H3 neighbor mean score"
        )
        form_c.addRow("H3 1-Ring Contagion (α):", self.smooth_alpha_spin)

        chk_row = QHBoxLayout()
        self.use_constraints_chk = QCheckBox("Apply Hard Constraints (constraint_mask)")
        self.use_constraints_chk.setChecked(True)
        chk_row.addWidget(self.use_constraints_chk)
        self.run_mc_chk = QCheckBox("Monte Carlo Stability (mc_std)")
        self.run_mc_chk.setChecked(True)
        chk_row.addWidget(self.run_mc_chk)
        form_c.addRow(chk_row)

        self.forge_btn = QPushButton("Forge Composite H3 Map (0–100)")
        self.forge_btn.setObjectName("primaryTealBtn")
        self.forge_btn.clicked.connect(self.on_forge_composite_map)
        form_c.addRow(self.forge_btn)

        lay.addWidget(grp_c)
        return page

    def _build_tab4_results(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        grp_res = QGroupBox("Composite Index Diagnostics & Local XAI Drivers")
        vbox = QVBoxLayout(grp_res)

        self.kpi_label = QLabel(
            "Run <b>Forge Composite H3 Map</b> in Stage 3 to populate diagnostics."
        )
        self.kpi_label.setWordWrap(True)
        vbox.addWidget(self.kpi_label)

        self.summary_table = QTableWidget(0, 6)
        self.summary_table.setHorizontalHeaderLabels(
            ["Class Band", "Hexagons", "Area (ha)", "Share (%)", "Mean Score", "Top XAI Driver"]
        )
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vbox.addWidget(self.summary_table, 1)

        exp_bar = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export H3 Table to CSV...")
        self.export_csv_btn.clicked.connect(self.on_export_csv)
        exp_bar.addWidget(self.export_csv_btn)

        self.export_html_btn = QPushButton("Generate Executive HTML Report...")
        self.export_html_btn.setObjectName("primaryTealBtn")
        self.export_html_btn.clicked.connect(self.on_export_html_report)
        exp_bar.addWidget(self.export_html_btn)
        vbox.addLayout(exp_bar)

        lay.addWidget(grp_res, 1)
        return page

    # ------------------------------------------------------------------ #
    # Project & Layer Synchronization
    # ------------------------------------------------------------------ #

    def _connect_project_signals(self) -> None:
        proj = QgsProject.instance()
        with suppress(Exception):
            proj.layersAdded.connect(lambda _lyrs: self.refresh_layer_combos())
            proj.layersRemoved.connect(lambda _ids: self.refresh_layer_combos())

    def dispose(self) -> None:
        proj = QgsProject.instance()
        with suppress(Exception):
            proj.layersAdded.disconnect()
            proj.layersRemoved.disconnect()

    def log(self, msg: str) -> None:
        self.log_box.append(msg)

    def _warn(self, msg: str) -> None:
        self.log(f"[WARN] {msg}")
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            raise RuntimeError(msg)
        QMessageBox.warning(self, "02H3Forge", msg)

    def _error(self, msg: str) -> None:
        self.log(f"[ERROR] {msg}")
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            raise RuntimeError(msg)
        QMessageBox.critical(self, "02H3Forge", msg)

    def active_h3_layer(self) -> Optional[QgsVectorLayer]:
        lid = self.h3_layer_combo.currentData()
        if not lid:
            return None
        lyr = QgsProject.instance().mapLayer(str(lid))
        if isinstance(lyr, QgsVectorLayer) and lyr.isValid():
            return lyr
        return None

    def refresh_layer_combos(self) -> None:
        prev_h3_id = self.h3_layer_combo.currentData()
        prev_bnd_id = self.boundary_layer_combo.currentData()
        prev_buf_id = self.buf_layer_combo.currentData()
        prev_vec_id = self.agg_vec_layer_combo.currentData()
        prev_ras_id = self.agg_ras_layer_combo.currentData()

        layers = list(QgsProject.instance().mapLayers().values())

        with QSignalBlocker(self.h3_layer_combo):
            self.h3_layer_combo.clear()
            first_h3_idx = -1
            for lyr in layers:
                if (
                    isinstance(lyr, QgsVectorLayer)
                    and lyr.isValid()
                    and lyr.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry
                ):
                    is_h3 = lyr.fields().indexFromName("h3_id") >= 0
                    prefix = "[H3] " if is_h3 else ""
                    self.h3_layer_combo.addItem(f"{prefix}{lyr.name()}", lyr.id())
                    if is_h3 and first_h3_idx < 0:
                        first_h3_idx = self.h3_layer_combo.count() - 1
            if prev_h3_id:
                idx = self.h3_layer_combo.findData(prev_h3_id)
                if idx >= 0:
                    self.h3_layer_combo.setCurrentIndex(idx)
            elif first_h3_idx >= 0:
                self.h3_layer_combo.setCurrentIndex(first_h3_idx)

        with QSignalBlocker(self.boundary_layer_combo):
            self.boundary_layer_combo.clear()
            for lyr in layers:
                if isinstance(lyr, QgsMapLayer) and lyr.isValid():
                    self.boundary_layer_combo.addItem(lyr.name(), lyr.id())
            if prev_bnd_id:
                idx = self.boundary_layer_combo.findData(prev_bnd_id)
                if idx >= 0:
                    self.boundary_layer_combo.setCurrentIndex(idx)

        with QSignalBlocker(self.buf_layer_combo):
            self.buf_layer_combo.clear()
            for lyr in layers:
                if isinstance(lyr, QgsVectorLayer) and lyr.isValid():
                    if lyr.fields().indexFromName("h3_id") >= 0:
                        continue
                    self.buf_layer_combo.addItem(lyr.name(), lyr.id())
            if prev_buf_id:
                idx = self.buf_layer_combo.findData(prev_buf_id)
                if idx >= 0:
                    self.buf_layer_combo.setCurrentIndex(idx)

        with QSignalBlocker(self.agg_vec_layer_combo):
            self.agg_vec_layer_combo.clear()
            for lyr in layers:
                if isinstance(lyr, QgsVectorLayer) and lyr.isValid():
                    if lyr.fields().indexFromName("h3_id") >= 0:
                        continue
                    self.agg_vec_layer_combo.addItem(lyr.name(), lyr.id())
            if prev_vec_id:
                idx = self.agg_vec_layer_combo.findData(prev_vec_id)
                if idx >= 0:
                    self.agg_vec_layer_combo.setCurrentIndex(idx)

        with QSignalBlocker(self.agg_ras_layer_combo):
            self.agg_ras_layer_combo.clear()
            for lyr in layers:
                if isinstance(lyr, QgsRasterLayer) and lyr.isValid():
                    self.agg_ras_layer_combo.addItem(lyr.name(), lyr.id())
            if prev_ras_id:
                idx = self.agg_ras_layer_combo.findData(prev_ras_id)
                if idx >= 0:
                    self.agg_ras_layer_combo.setCurrentIndex(idx)

        self._on_buf_layer_changed()
        self._on_agg_vec_layer_changed()
        self._on_agg_ras_layer_changed()
        self._on_active_h3_layer_changed()

    def _on_active_h3_layer_changed(self) -> None:
        self.populate_normalization_table()
        self.populate_weight_sliders()
        self.refresh_results_summary()

    def _on_buf_layer_changed(self) -> None:
        lid = self.buf_layer_combo.currentData()
        lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if lyr is not None:
            slug = sanitize_field_slug(lyr.name(), prefix="")
            self.buf_name_edit.setText(slug)

    def _on_agg_vec_layer_changed(self) -> None:
        self.agg_vec_field_combo.clear()
        lid = self.agg_vec_layer_combo.currentData()
        lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if not isinstance(lyr, QgsVectorLayer):
            return
        for fld in lyr.fields():
            if fld.isNumeric():
                self.agg_vec_field_combo.addItem(fld.name(), fld.name())

    def _on_agg_ras_layer_changed(self) -> None:
        lid = self.agg_ras_layer_combo.currentData()
        lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if isinstance(lyr, QgsRasterLayer):
            self.agg_ras_band_spin.setMaximum(max(1, lyr.bandCount()))

    # ------------------------------------------------------------------ #
    # Stage 0 Actions: Grid & Multi-Ring Buffers
    # ------------------------------------------------------------------ #

    def _load_selected_ring_preset(self) -> None:
        preset_key = self.buf_preset_combo.currentText()
        rings = MULTI_RING_PRESETS.get(preset_key, [])
        self.ring_table.setRowCount(0)
        for low, high, sc, excl in rings:
            self._append_ring_row(low, high, sc, excl)

    def _append_ring_row(self, low: float, high: float, score: float, excl: bool) -> None:
        r = self.ring_table.rowCount()
        self.ring_table.insertRow(r)
        self.ring_table.setItem(r, 0, QTableWidgetItem(f"{low:g}"))
        self.ring_table.setItem(r, 1, QTableWidgetItem(f"{high:g}"))
        self.ring_table.setItem(r, 2, QTableWidgetItem(f"{score:g}"))
        chk_item = QTableWidgetItem("Exclude")
        chk_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        chk_item.setCheckState(
            Qt.CheckState.Checked if excl else Qt.CheckState.Unchecked
        )
        self.ring_table.setItem(r, 3, chk_item)

    def _remove_selected_ring_row(self) -> None:
        r = self.ring_table.currentRow()
        if r < 0:
            r = self.ring_table.rowCount() - 1
        if r >= 0:
            self.ring_table.removeRow(r)

    def get_table_rings(self) -> List[RingSpec]:
        rings: List[RingSpec] = []
        for r in range(self.ring_table.rowCount()):
            it_lo = self.ring_table.item(r, 0)
            it_hi = self.ring_table.item(r, 1)
            it_sc = self.ring_table.item(r, 2)
            it_ex = self.ring_table.item(r, 3)
            if not it_lo or not it_hi or not it_sc:
                continue
            lo = float(it_lo.text())
            hi = float(it_hi.text())
            sc = float(it_sc.text())
            excl = (
                it_ex is not None
                and it_ex.checkState() == Qt.CheckState.Checked
            )
            rings.append((lo, hi, sc, excl))
        return rings

    def on_generate_h3_grid(self) -> None:
        res = int(self.res_combo.currentData() or DEFAULT_H3_RESOLUTION)
        cov_mode = str(self.coverage_combo.currentData() or "intersect")
        ext_mode = str(self.extent_source_combo.currentData() or "layer")

        bnd_layer = None
        extent_rect = None
        target_crs = QgsProject.instance().crs()
        if not target_crs.isValid():
            target_crs = QgsCoordinateReferenceSystem("EPSG:5253")

        if ext_mode == "layer":
            lid = self.boundary_layer_combo.currentData()
            bnd_layer = QgsProject.instance().mapLayer(str(lid)) if lid else None
            if bnd_layer is None and self.iface.activeLayer() is not None:
                bnd_layer = self.iface.activeLayer()
            if bnd_layer is not None and bnd_layer.isValid():
                extent_rect = bnd_layer.extent()
                target_crs = bnd_layer.crs()

        if extent_rect is None or extent_rect.isEmpty():
            canvas = self.iface.mapCanvas()
            if canvas is not None and not canvas.extent().isEmpty():
                extent_rect = canvas.extent()
                target_crs = canvas.mapSettings().destinationCrs()
            else:
                self._warn(
                    "Please load a reference layer or zoom the map canvas before generating an H3 grid."
                )
                return

        try:
            h3_lyr = build_h3_grid_layer(
                extent_rect=extent_rect,
                target_crs=target_crs,
                resolution=res,
                boundary_layer=bnd_layer if isinstance(bnd_layer, QgsVectorLayer) else None,
                coverage_mode=cov_mode,
                prefer_native_h3=True,
            )
        except Exception as exc:
            self._error(f"H3 grid generation failed:\n{exc}")
            return

        # Style initial grid with a clean hollow teal outline
        sym = QgsFillSymbol.createSimple({
            "color": "19,160,160,30",
            "outline_color": "#0f6b6b",
            "outline_width": "0.26",
        })
        h3_lyr.setRenderer(QgsSingleSymbolRenderer(sym))

        QgsProject.instance().addMapLayer(h3_lyr)
        self.refresh_layer_combos()
        idx = self.h3_layer_combo.findData(h3_lyr.id())
        if idx >= 0:
            self.h3_layer_combo.setCurrentIndex(idx)

        spec = H3_RESOLUTION_SPECS[res]
        self.log(
            f"[Stage 0] Generated H3 Grid '{h3_lyr.name()}': "
            f"{h3_lyr.featureCount()} hexagons (Res {res}, ~{spec['area_ha']} ha, ~{spec['edge_m']} m edge)."
        )

    def on_run_multiring_buffer(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            self._warn("Please generate or select an active H3 layer first.")
            return
        lid = self.buf_layer_combo.currentData()
        src_lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if not isinstance(src_lyr, QgsVectorLayer):
            self._warn("Please select a valid target vector layer.")
            return

        crit_name = self.buf_name_edit.text().strip() or src_lyr.name()
        try:
            rings = self.get_table_rings()
            dist_col, ring_col, excl_cnt = apply_multiring_buffer_to_h3(
                h3_layer=h3_lyr,
                source_layer=src_lyr,
                rings=rings,
                criterion_name=crit_name,
                outside_score=float(self.outside_score_spin.value()),
            )
            norm_msg = ""
            if self.auto_norm_ring_chk.isChecked():
                norm_col = normalize_h3_field(
                    h3_lyr,
                    source_field=ring_col,
                    polarity="benefit",
                    method="direct_clip",
                )
                norm_msg = f" -> normalized as '{norm_col}'"
            self.populate_normalization_table()
            self.populate_weight_sliders()
            self.log(
                f"[Stage 0] Multi-ring buffer '{crit_name}': wrote '{dist_col}' & '{ring_col}'"
                f"{norm_msg} (excluded hexes: {excl_cnt})."
            )
        except Exception as exc:
            self._error(f"Multi-ring buffer failed:\n{exc}")

    # ------------------------------------------------------------------ #
    # Stage 1 Actions: Spatial Aggregation
    # ------------------------------------------------------------------ #

    def on_aggregate_vector(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            self._warn("Please generate or select an active H3 layer first.")
            return
        lid = self.agg_vec_layer_combo.currentData()
        src_lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if not isinstance(src_lyr, QgsVectorLayer):
            self._warn("Please select a source vector layer.")
            return

        method = str(self.agg_vec_method_combo.currentData() or "count")
        fld = self.agg_vec_field_combo.currentData()
        out_name = self.agg_vec_out_edit.text().strip() or None
        try:
            col = aggregate_vector_to_h3(
                h3_layer=h3_lyr,
                source_layer=src_lyr,
                method=method,
                field_name=str(fld) if fld else None,
                output_name=out_name,
            )
            self.populate_normalization_table()
            self.log(f"[Stage 1] Aggregated vector '{src_lyr.name()}' ({method}) -> '{col}'.")
        except Exception as exc:
            self._error(f"Vector aggregation failed:\n{exc}")

    def on_aggregate_raster(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            self._warn("Please generate or select an active H3 layer first.")
            return
        lid = self.agg_ras_layer_combo.currentData()
        ras_lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if not isinstance(ras_lyr, QgsRasterLayer):
            self._warn("Please select a source raster layer.")
            return

        method = str(self.agg_ras_method_combo.currentData() or "mean")
        band = int(self.agg_ras_band_spin.value())
        out_name = self.agg_ras_out_edit.text().strip() or None
        try:
            col = aggregate_raster_to_h3(
                h3_layer=h3_lyr,
                raster_layer=ras_lyr,
                method=method,
                band_number=band,
                output_name=out_name,
            )
            self.populate_normalization_table()
            self.log(f"[Stage 1] Sampled raster '{ras_lyr.name()}' (band {band}, {method}) -> '{col}'.")
        except Exception as exc:
            self._error(f"Raster sampling failed:\n{exc}")

    # ------------------------------------------------------------------ #
    # Stage 2 Actions: Criterion Normalization (0-100)
    # ------------------------------------------------------------------ #

    def populate_normalization_table(self) -> None:
        self.norm_table.setRowCount(0)
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            return

        skip_fields = {
            "h3_id", "h3_res", "area_m2", "area_ha", "constraint_mask",
            "comp_score", "class_id", "class_label", "top_driver",
            "driver_share", "mc_std", "rob_score",
        }
        candidate_fields: List[str] = []
        for fld in h3_lyr.fields():
            name = fld.name()
            if name in skip_fields or name.startswith("norm_"):
                continue
            if fld.isNumeric():
                candidate_fields.append(name)

        for row_idx, fname in enumerate(candidate_fields):
            self.norm_table.insertRow(row_idx)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(Qt.CheckState.Checked)
            self.norm_table.setItem(row_idx, 0, chk)

            name_item = QTableWidgetItem(fname)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.norm_table.setItem(row_idx, 1, name_item)

            pol_cb = QComboBox()
            for key, lbl in POLARITY_OPTIONS:
                pol_cb.addItem(lbl, key)
            if fname.startswith("dist_"):
                pol_cb.setCurrentIndex(1)  # Default distance to Cost (-)
            self.norm_table.setCellWidget(row_idx, 2, pol_cb)

            meth_cb = QComboBox()
            for key, lbl in NORMALIZATION_METHODS:
                meth_cb.addItem(lbl, key)
            if fname.startswith("ring_"):
                meth_cb.setCurrentIndex(5)  # Direct 0-100 Score
            self.norm_table.setCellWidget(row_idx, 3, meth_cb)

            self.norm_table.setItem(row_idx, 4, QTableWidgetItem(""))
            self.norm_table.setItem(row_idx, 5, QTableWidgetItem(""))

    def on_run_normalization(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            self._warn("Please select an active H3 layer first.")
            return

        produced: List[str] = []
        for r in range(self.norm_table.rowCount()):
            chk = self.norm_table.item(r, 0)
            if chk is None or chk.checkState() != Qt.CheckState.Checked:
                continue
            src_item = self.norm_table.item(r, 1)
            if src_item is None:
                continue
            src_field = src_item.text().strip()
            pol_cb = self.norm_table.cellWidget(r, 2)
            meth_cb = self.norm_table.cellWidget(r, 3)
            polarity = str(pol_cb.currentData()) if isinstance(pol_cb, QComboBox) else "benefit"
            method = str(meth_cb.currentData()) if isinstance(meth_cb, QComboBox) else "minmax"

            p1_txt = self.norm_table.item(r, 4).text().strip() if self.norm_table.item(r, 4) else ""
            p2_txt = self.norm_table.item(r, 5).text().strip() if self.norm_table.item(r, 5) else ""
            p1_val = float(p1_txt) if p1_txt else None
            p2_val = float(p2_txt) if p2_txt else None

            try:
                norm_col = normalize_h3_field(
                    h3_layer=h3_lyr,
                    source_field=src_field,
                    polarity=polarity,
                    method=method,
                    low=p1_val,
                    high=p2_val,
                    mid=p1_val,
                    spread=p2_val,
                )
                produced.append(f"{norm_col} ({polarity[0].upper()})")
            except Exception as exc:
                self._warn(f"Could not normalize '{src_field}':\n{exc}")

        self.populate_weight_sliders()
        if produced:
            self.log(f"[Stage 2] Normalized {len(produced)} criteria (0–100): {', '.join(produced)}.")

    # ------------------------------------------------------------------ #
    # Stage 3 Actions: Coupled Weight Sliders & Multi-Criteria Overlay
    # ------------------------------------------------------------------ #

    def populate_weight_sliders(self) -> None:
        while self.sliders_layout.count():
            child = self.sliders_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._weight_rows.clear()

        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            return

        norm_fields = [
            f.name() for f in h3_lyr.fields()
            if f.name().startswith("norm_") and f.isNumeric()
        ]
        if not norm_fields:
            lbl = QLabel(
                "<i>No normalized criteria (<code>norm_*</code>) found on this H3 layer yet. "
                "Use Stage 0 (Multi-Ring) or Stage 2 (Normalize) to prepare criteria.</i>"
            )
            lbl.setWordWrap(True)
            self.sliders_layout.addWidget(lbl)
            self.sliders_layout.addStretch(1)
            self.total_weight_lbl.setText("<b>Total Weight: 0.0%</b>")
            return

        init_weights = equal_weights_pct(len(norm_fields))
        for idx, fname in enumerate(norm_fields):
            row_w = QWidget()
            h_lay = QHBoxLayout(row_w)
            h_lay.setContentsMargins(2, 2, 2, 2)

            chk = QCheckBox(fname.replace("norm_", ""))
            chk.setToolTip(f"Normalized column: {fname}")
            chk.setChecked(True)
            chk.setMinimumWidth(110)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(int(round(init_weights[idx] * 10.0)))

            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setSuffix("%")
            spin.setValue(init_weights[idx])

            lock = QCheckBox("🔒")
            lock.setToolTip("Lock this criterion's weight while other sliders rebalance")

            chk.toggled.connect(lambda _c, i=idx: self._on_criterion_active_toggled(i))
            slider.valueChanged.connect(lambda val, i=idx: self._on_slider_moved(i, val / 10.0))
            spin.valueChanged.connect(lambda val, i=idx: self._on_spin_changed(i, float(val)))

            h_lay.addWidget(chk)
            h_lay.addWidget(slider, 1)
            h_lay.addWidget(spin)
            h_lay.addWidget(lock)
            self.sliders_layout.addWidget(row_w)

            self._weight_rows.append({
                "field": fname,
                "chk": chk,
                "slider": slider,
                "spin": spin,
                "lock": lock,
            })

        self.sliders_layout.addStretch(1)
        self._update_total_weight_badge()

    def _current_weight_state(self) -> Tuple[List[float], List[bool], List[bool]]:
        weights = [float(r["spin"].value()) for r in self._weight_rows]
        locked = [bool(r["lock"].isChecked()) for r in self._weight_rows]
        active = [bool(r["chk"].isChecked()) for r in self._weight_rows]
        return weights, locked, active

    def _apply_weights_to_widgets(self, new_weights: List[float]) -> None:
        for i, r in enumerate(self._weight_rows):
            if i >= len(new_weights):
                break
            val = float(new_weights[i])
            slider: QSlider = r["slider"]
            spin: QDoubleSpinBox = r["spin"]
            with QSignalBlocker(slider), QSignalBlocker(spin):
                spin.setValue(val)
                slider.setValue(int(round(val * 10.0)))
        self._update_total_weight_badge()

    def _update_total_weight_badge(self) -> None:
        total = sum(
            float(r["spin"].value())
            for r in self._weight_rows
            if r["chk"].isChecked()
        )
        self.total_weight_lbl.setText(f"<b>Total Weight: {total:.1f}%</b>")

    def _on_slider_moved(self, index: int, new_pct: float) -> None:
        weights, locked, active = self._current_weight_state()
        if index < len(self._weight_rows) and locked[index]:
            # Locked slider snaps back
            self._apply_weights_to_widgets(weights)
            return
        balanced = rebalance_coupled_weights(
            weights, index, new_pct, locked_flags=locked, active_flags=active
        )
        self._apply_weights_to_widgets(balanced)

    def _on_spin_changed(self, index: int, new_pct: float) -> None:
        weights, locked, active = self._current_weight_state()
        if index < len(self._weight_rows) and locked[index]:
            self._apply_weights_to_widgets(weights)
            return
        balanced = rebalance_coupled_weights(
            weights, index, new_pct, locked_flags=locked, active_flags=active
        )
        self._apply_weights_to_widgets(balanced)

    def _on_criterion_active_toggled(self, _index: int) -> None:
        weights, locked, active = self._current_weight_state()
        balanced = rebalance_coupled_weights(
            weights, -1, 0.0, locked_flags=locked, active_flags=active
        )
        self._apply_weights_to_widgets(balanced)

    def get_active_criteria_and_weights(self) -> Tuple[List[str], List[float]]:
        fields: List[str] = []
        weights: List[float] = []
        for r in self._weight_rows:
            if r["chk"].isChecked():
                fields.append(str(r["field"]))
                weights.append(float(r["spin"].value()))
        return fields, weights

    def _set_active_weights_from_engine(self, active_weights: List[float], label: str) -> None:
        full_weights: List[float] = []
        ptr = 0
        for r in self._weight_rows:
            with QSignalBlocker(r["lock"]):
                r["lock"].setChecked(False)
            if r["chk"].isChecked() and ptr < len(active_weights):
                full_weights.append(float(active_weights[ptr]))
                ptr += 1
            else:
                full_weights.append(0.0)
        self._apply_weights_to_widgets(full_weights)
        self.log(f"[Stage 3] Applied {label} weights: {[round(w, 1) for w in active_weights]}.")

    def on_weights_equal(self) -> None:
        _, _, active = self._current_weight_state()
        for r in self._weight_rows:
            with QSignalBlocker(r["lock"]):
                r["lock"].setChecked(False)
        eq = equal_weights_pct(len(self._weight_rows), active_flags=active)
        self._apply_weights_to_widgets(eq)
        self.log("[Stage 3] Reset active criteria to Equal (1/N) weights.")

    def on_weights_critic(self) -> None:
        h3_lyr = self.active_h3_layer()
        fields, _ = self.get_active_criteria_and_weights()
        if h3_lyr is None or not fields:
            return
        _, _, mat, _ = extract_h3_criteria_matrix(h3_lyr, fields)
        w_pct, _, _ = critic_weights_pct(mat)
        self._set_active_weights_from_engine(w_pct, "CRITIC Objective")

    def on_weights_entropy(self) -> None:
        h3_lyr = self.active_h3_layer()
        fields, _ = self.get_active_criteria_and_weights()
        if h3_lyr is None or not fields:
            return
        _, _, mat, _ = extract_h3_criteria_matrix(h3_lyr, fields)
        w_pct = entropy_weights_pct(mat)
        self._set_active_weights_from_engine(w_pct, "Shannon Entropy")

    def on_weights_pca(self) -> None:
        h3_lyr = self.active_h3_layer()
        fields, _ = self.get_active_criteria_and_weights()
        if h3_lyr is None or not fields:
            return
        _, _, mat, _ = extract_h3_criteria_matrix(h3_lyr, fields)
        w_pct = pca_weights_pct(mat)
        self._set_active_weights_from_engine(w_pct, "PCA First-Component")

    def on_weights_ahp(self) -> None:
        fields, _ = self.get_active_criteria_and_weights()
        if len(fields) < 2:
            QMessageBox.information(
                self, "02H3Forge", "Select at least 2 normalized criteria to open the AHP Pairwise Matrix."
            )
            return
        dlg = AhpPairwiseDialog(fields, parent=self)
        if dlg.exec():
            self._set_active_weights_from_engine(
                dlg.computed_weights, f"AHP Pairwise (CR={dlg.cr_value:.3f})"
            )

    def on_forge_composite_map(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            self._warn("Please select an active H3 layer first.")
            return
        fields, weights = self.get_active_criteria_and_weights()
        if not fields:
            self._warn(
                "No active normalized criteria (norm_*) selected. Prepare criteria in Stage 0 or Stage 2 first."
            )
            return

        compose_method = str(self.compose_method_combo.currentData() or "wlc")
        theme = str(self.theme_combo.currentData() or "risk")
        class_mode = str(self.class_mode_combo.currentData() or "equal")
        alpha = float(self.smooth_alpha_spin.value())
        use_mask = bool(self.use_constraints_chk.isChecked())
        run_mc = bool(self.run_mc_chk.isChecked())

        try:
            stats = run_h3_overlay_on_layer(
                h3_layer=h3_lyr,
                criterion_fields=fields,
                weights_pct=weights,
                compose_method=compose_method,
                apply_constraints=use_mask,
                spatial_smoothing_alpha=alpha,
                run_montecarlo=run_mc,
                classification_method=class_mode,
                theme=theme,
            )
            self._last_breaks = stats.get("breaks")
            apply_h3_graduated_style(
                h3_lyr,
                target_field="comp_score",
                theme=theme,
                breaks=self._last_breaks,
            )
            if self.iface.mapCanvas() is not None:
                self.iface.mapCanvas().refresh()
            self.refresh_results_summary()
            self.tabs.setCurrentIndex(4)
            self.log(
                f"[Stage 3] Forged Composite H3 Map ({compose_method.upper()}, theme={theme}): "
                f"mean={stats['mean_score']}, max={stats['max_score']}, MC std={stats['mean_mc_std']}."
            )
        except Exception as exc:
            self._error(f"Composite overlay failed:\n{exc}")

    # ------------------------------------------------------------------ #
    # Stage 4 Actions: Diagnostics & Export
    # ------------------------------------------------------------------ #

    def refresh_results_summary(self) -> None:
        self.summary_table.setRowCount(0)
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None or h3_lyr.fields().indexFromName("comp_score") < 0:
            return

        theme = str(self.theme_combo.currentData() or "risk")
        rows = compute_class_summary(h3_lyr, theme=theme)
        tot_hex = sum(int(r["hex_count"]) for r in rows)
        tot_ha = sum(float(r["area_ha"]) for r in rows)
        self.kpi_label.setText(
            f"<b>Layer:</b> {h3_lyr.name()} &nbsp;|&nbsp; "
            f"<b>Hexagons:</b> {tot_hex:,} &nbsp;|&nbsp; "
            f"<b>Total Area:</b> {tot_ha:.2f} ha"
        )

        for idx, r in enumerate(rows):
            self.summary_table.insertRow(idx)
            vals = [
                f"{r['class_id']} — {r['class_label']}",
                str(r["hex_count"]),
                f"{r['area_ha']:.2f}",
                f"{r['share_pct']:.1f}%",
                f"{r['mean_score']:.2f}",
                str(r["top_driver"]),
            ]
            for col_i, txt in enumerate(vals):
                item = QTableWidgetItem(txt)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.summary_table.setItem(idx, col_i, item)

    def on_export_csv(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export H3 Table to CSV", f"{h3_lyr.name()}.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            cnt = export_h3_table_csv(h3_lyr, path)
            self.log(f"[Stage 4] Exported {cnt} H3 hexagons to CSV: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "02H3Forge", f"CSV export failed:\n{exc}")

    def on_export_html_report(self) -> None:
        h3_lyr = self.active_h3_layer()
        if h3_lyr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Executive HTML Report",
            f"{h3_lyr.name()}_report.html",
            "HTML Files (*.html)",
        )
        if not path:
            return
        fields, weights = self.get_active_criteria_and_weights()
        w_map = {fields[i]: weights[i] for i in range(len(fields))}
        compose_method = str(self.compose_method_combo.currentData() or "wlc")
        theme = str(self.theme_combo.currentData() or "risk")
        try:
            export_h3_html_report(
                h3_lyr,
                path,
                weights_map=w_map,
                compose_method=compose_method,
                theme=theme,
            )
            self.log(f"[Stage 4] Generated Executive HTML Report: {path}")
            if not os.environ.get("QT_QPA_PLATFORM") == "offscreen":
                webbrowser.open(Path(path).resolve().as_uri())
        except Exception as exc:
            QMessageBox.critical(self, "02H3Forge", f"HTML report export failed:\n{exc}")
