# -*- coding: utf-8 -*-
"""02H3Forge studio dock (v0.3, criterion-centric).

Workflow::

    1 Study Area   grid generation, persistence
    2 Criteria     criterion library (source → value function → group) + constraints
    3 Weights      group weights, within-group weights, methods, run options
    4 Results      diagnostics, hotspots, sensitivity, styling, exports

The whole analysis is a :class:`~zero2h3forge.core.model.ModelSpec`. *Run
model* snapshots the layers on the main thread, computes everything in a
background ``QgsTask`` (progress + cancel) and writes the results back in one
batch. Unchanged raw criteria are reused from the layer cache, so weight
experiments are near-instant.
"""
from __future__ import annotations

import os
import traceback
import webbrowser
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsMapLayer,
    QgsProject,
    QgsSingleSymbolRenderer,
    QgsTask,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QSignalBlocker, Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import PLUGIN_VERSION
from ..ahp_dialog import AhpPairwiseDialog
from ..core.compute import Cancelled
from ..core.h3_grid_engine import (
    DEFAULT_H3_RESOLUTION,
    H3_RESOLUTION_SPECS,
    MAX_METRIC_CELLS,
    build_h3_grid_layer,
    estimate_hex_count,
    resolve_metric_work_crs,
)
from ..core.mcda_composer import critic_weights_pct, entropy_weights_pct, owa_orness, pca_weights_pct
from ..core.model import (
    AGGREGATIONS,
    GROUP_METHODS,
    SOURCE_KINDS,
    TEMPLATES,
    CriterionSpec,
    GroupSpec,
    ModelSpec,
    SourceSpec,
    ValueFunction,
    apply_template,
    new_model,
)
from ..core.pipeline import apply_run, compute_run, prepare_run
from ..core.styler_and_report import (
    THEME_OPTIONS,
    apply_h3_graduated_style,
    apply_h3_hotspot_style,
    compute_class_summary,
    export_h3_html_report,
    export_h3_table_csv,
    read_layer_json_property,
    save_h3_layer_to_geopackage,
)
from .dialogs import ConstraintDialog, CriterionDialog, FieldImportDialog
from .widgets import CoupledWeightsWidget

RATE_URL = f"https://geophilo.com/feedback/?plugin=zero2h3forge&v={PLUGIN_VERSION}"
CAN_CANCEL = getattr(getattr(QgsTask, "Flag", QgsTask), "CanCancel")

_QSS = """
QWidget#h3ForgeDockBody { background: #f4f7fb; }
QLabel#h3ForgeHeader {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f6b6b,stop:1 #13a0a0);
    color: white; font-weight: bold; font-size: 13px; padding: 9px 12px; border-radius: 6px;
}
QGroupBox {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
    margin-top: 8px; padding-top: 10px; font-weight: bold; color: #0f6b6b;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QTableWidget, QTextEdit, QTreeWidget, QListWidget {
    background: #ffffff; color: #111827; border: 1px solid #cbd5e1; font-size: 12px;
}
QLineEdit, QComboBox {
    background: #ffffff; color: #111827; border: 1px solid #94a3b8; border-radius: 4px; padding: 4px 6px;
}
QPushButton {
    background: #e2e8f0; color: #111827; border: 1px solid #94a3b8; border-radius: 4px; padding: 5px 8px;
}
QPushButton:hover { background: #cbd5e1; }
QPushButton:disabled { color: #94a3b8; }
QPushButton#primaryTealBtn { background: #13a0a0; color: white; border: none; font-weight: bold; padding: 7px 10px; }
QPushButton#primaryTealBtn:hover { background: #0f6b6b; }
QPushButton#runBtn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #cf9836,stop:1 #e3b04b);
    color: white; border: none; font-weight: bold; font-size: 13px; padding: 8px 10px;
}
QPushButton#runBtn:hover { background: #b8862d; }
QPushButton#runBtn:disabled { background: #e5d3b0; }
QPushButton#rateGoldBtn { background: #cf9836; color: white; border: none; font-weight: bold; padding: 5px 8px; }
QLabel#muted { color: #475569; font-size: 11px; }
QCheckBox, QLabel { color: #111827; }
"""


class _ModelTask(QgsTask):
    """Runs :func:`compute_run` off the GUI thread."""

    def __init__(self, inputs, on_done):
        super().__init__("02H3Forge — running model", CAN_CANCEL)
        self.inputs = inputs
        self.on_done = on_done
        self.result = None
        self.error: Optional[str] = None
        self.cancelled_by_user = False

    def run(self) -> bool:  # worker thread
        try:
            self.result = compute_run(self.inputs, self.setProgress, self.isCanceled)
            return True
        except Cancelled:
            self.cancelled_by_user = True
            return False
        except Exception:  # pragma: no cover - surfaced to the user
            self.error = traceback.format_exc()
            return False

    def finished(self, ok: bool) -> None:  # main thread
        self.on_done(self, ok)


class O2H3ForgeStudioDock(QDockWidget):
    """Right-docked criterion-centric H3 MCDA studio."""

    def __init__(self, iface, parent=None):
        super().__init__("02H3Forge Studio", parent)
        self.iface = iface
        self.setObjectName("O2H3ForgeStudioDock")
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.model: ModelSpec = new_model("flat")
        self._task: Optional[_ModelTask] = None
        self._task_layer_id: Optional[str] = None
        self._task_snap = None
        self._project_slots: List[tuple] = []
        self._group_widgets: Dict[str, CoupledWeightsWidget] = {}
        self._build_ui()
        self._connect_project_signals()
        self.refresh_layer_combos()

    # ================================================================== UI
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

        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>H3 layer:</b>"))
        self.h3_layer_combo = QComboBox()
        self.h3_layer_combo.currentIndexChanged.connect(self._on_active_layer_changed)
        bar.addWidget(self.h3_layer_combo, 1)
        refresh = QPushButton("↻")
        refresh.setFixedWidth(30)
        refresh.setToolTip("Refresh layer lists")
        refresh.clicked.connect(self.refresh_layer_combos)
        bar.addWidget(refresh)
        root.addLayout(bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scrollable(self._build_tab_area()), "1. Study Area")
        self.tabs.addTab(self._scrollable(self._build_tab_criteria()), "2. Criteria")
        self.tabs.addTab(self._scrollable(self._build_tab_weights()), "3. Weights")
        self.tabs.addTab(self._scrollable(self._build_tab_results()), "4. Results")
        root.addWidget(self.tabs, 1)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("⚡ Run Model")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setToolTip("Compute criteria, constraints, composite index, uncertainty, hotspots "
                                "and sensitivity in the background.")
        self.run_btn.clicked.connect(lambda: self.run_model())
        run_row.addWidget(self.run_btn, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_run)
        run_row.addWidget(self.cancel_btn)
        root.addLayout(run_row)
        opt_row = QHBoxLayout()
        self.force_chk = QCheckBox("Recompute all raw criteria")
        self.force_chk.setToolTip("Ignore the cache (use after editing a source layer).")
        opt_row.addWidget(self.force_chk)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(16)
        opt_row.addWidget(self.progress, 1)
        root.addLayout(opt_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(80)
        self.log_box.setPlaceholderText("02H3Forge log…")
        root.addWidget(self.log_box)

        footer = QHBoxLayout()
        repo = QPushButton("GitHub Repo")
        repo.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/YusufEminoglu/zero2h3forge")))
        footer.addWidget(repo)
        rate = QPushButton("★ Rate on GeoPhilo")
        rate.setObjectName("rateGoldBtn")
        rate.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RATE_URL)))
        footer.addWidget(rate)
        root.addLayout(footer)

        body.setStyleSheet(_QSS)
        self.setWidget(body)

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(page)
        area.setMinimumWidth(page.minimumSizeHint().width() + area.verticalScrollBar().sizeHint().width())
        return area

    # ---- Tab 1: study area ------------------------------------------------
    def _build_tab_area(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        grp = QGroupBox("H3 Hexagonal Grid")
        form = QFormLayout(grp)
        self.extent_source_combo = QComboBox()
        self.extent_source_combo.addItem("Reference layer (polygon boundary / extent)", "layer")
        self.extent_source_combo.addItem("Current map canvas extent", "canvas")
        form.addRow("Extent:", self.extent_source_combo)
        self.boundary_layer_combo = QComboBox()
        form.addRow("Reference layer:", self.boundary_layer_combo)
        self.res_combo = QComboBox()
        for res in sorted(H3_RESOLUTION_SPECS):
            self.res_combo.addItem(str(H3_RESOLUTION_SPECS[res]["label"]), int(res))
        self.res_combo.setCurrentIndex(self.res_combo.findData(DEFAULT_H3_RESOLUTION))
        form.addRow("Resolution:", self.res_combo)
        self.coverage_combo = QComboBox()
        self.coverage_combo.addItem("Hexagons intersecting the boundary", "intersect")
        self.coverage_combo.addItem("Hexagon centroid inside the boundary", "centroid")
        self.coverage_combo.addItem("Full bounding box", "bbox")
        form.addRow("Coverage:", self.coverage_combo)
        self.grid_estimate_lbl = QLabel("")
        self.grid_estimate_lbl.setObjectName("muted")
        self.grid_estimate_lbl.setWordWrap(True)
        form.addRow(self.grid_estimate_lbl)
        for combo in (self.extent_source_combo, self.boundary_layer_combo, self.res_combo):
            combo.currentIndexChanged.connect(self._update_grid_estimate)
        self.gen_grid_btn = QPushButton("Generate H3 Grid")
        self.gen_grid_btn.setObjectName("primaryTealBtn")
        self.gen_grid_btn.clicked.connect(self.on_generate_h3_grid)
        form.addRow(self.gen_grid_btn)
        self.save_gpkg_btn = QPushButton("💾 Save Active H3 Layer to GeoPackage…")
        self.save_gpkg_btn.setToolTip("Generated grids are temporary. Save to keep the grid, results, "
                                      "styling and the model definition.")
        self.save_gpkg_btn.clicked.connect(self.on_save_to_geopackage)
        form.addRow(self.save_gpkg_btn)
        lay.addWidget(grp)

        self.grid_info_lbl = QLabel("")
        self.grid_info_lbl.setWordWrap(True)
        self.grid_info_lbl.setObjectName("muted")
        lay.addWidget(self.grid_info_lbl)
        lay.addStretch(1)
        return page

    # ---- Tab 2: criteria & constraints --------------------------------------
    def _build_tab_criteria(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        grp_model = QGroupBox("Model")
        f_model = QFormLayout(grp_model)
        self.model_name_edit = QLineEdit()
        self.model_name_edit.editingFinished.connect(self._on_model_name)
        f_model.addRow("Name:", self.model_name_edit)
        tpl_row = QHBoxLayout()
        self.template_combo = QComboBox()
        for key, lbl in TEMPLATES:
            self.template_combo.addItem(lbl, key)
        tpl_row.addWidget(self.template_combo, 1)
        apply_tpl = QPushButton("Apply")
        apply_tpl.setToolTip("Switch the group structure; existing criteria are kept.")
        apply_tpl.clicked.connect(self.on_apply_template)
        tpl_row.addWidget(apply_tpl)
        f_model.addRow("Structure:", tpl_row)
        lay.addWidget(grp_model)

        grp_c = QGroupBox("Criteria")
        v_c = QVBoxLayout(grp_c)
        self.crit_tree = QTreeWidget()
        self.crit_tree.setColumnCount(3)
        self.crit_tree.setHeaderLabels(["Criterion", "Source", "Value function"])
        self.crit_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.crit_tree.setMinimumHeight(220)
        self.crit_tree.itemDoubleClicked.connect(lambda *_: self.on_edit_criterion())
        self.crit_tree.itemChanged.connect(self._on_tree_item_changed)
        v_c.addWidget(self.crit_tree)
        row1 = QHBoxLayout()
        for text, slot in (("+ Add…", self.on_add_criterion), ("Edit…", self.on_edit_criterion),
                           ("Duplicate", self.on_duplicate_criterion), ("Remove", self.on_remove_criterion)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row1.addWidget(b)
        v_c.addLayout(row1)
        row2 = QHBoxLayout()
        imp = QPushButton("Use H3 fields…")
        imp.setToolTip("Register existing numeric H3 columns (e.g. norm_* from v0.2) as criteria")
        imp.clicked.connect(self.on_import_fields)
        row2.addWidget(imp)
        for text, slot in (("+ Group", self.on_add_group), ("Rename group", self.on_rename_group),
                           ("Remove group", self.on_remove_group)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row2.addWidget(b)
        v_c.addLayout(row2)
        lay.addWidget(grp_c, 1)

        grp_k = QGroupBox("Constraints (hard exclusions)")
        v_k = QVBoxLayout(grp_k)
        self.cons_list = QListWidget()
        self.cons_list.setMaximumHeight(110)
        self.cons_list.itemChanged.connect(self._on_constraint_toggled)
        self.cons_list.itemDoubleClicked.connect(lambda *_: self.on_edit_constraint())
        v_k.addWidget(self.cons_list)
        row3 = QHBoxLayout()
        for text, slot in (("+ Add…", self.on_add_constraint), ("Edit…", self.on_edit_constraint),
                           ("Remove", self.on_remove_constraint)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row3.addWidget(b)
        v_k.addLayout(row3)
        lay.addWidget(grp_k)
        return page

    # ---- Tab 3: weights & run options ----------------------------------------
    def _build_tab_weights(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)

        self.grp_groups = QGroupBox("Group Weights")
        v_g = QVBoxLayout(self.grp_groups)
        agg_row = QHBoxLayout()
        agg_row.addWidget(QLabel("Combine groups by:"))
        self.agg_combo = QComboBox()
        for key, lbl in AGGREGATIONS:
            self.agg_combo.addItem(lbl, key)
        self.agg_combo.currentIndexChanged.connect(self._on_aggregation)
        agg_row.addWidget(self.agg_combo, 1)
        v_g.addLayout(agg_row)
        self.group_weights = CoupledWeightsWidget()
        self.group_weights.weightsChanged.connect(self._on_group_weights)
        v_g.addWidget(self.group_weights)
        g_btns = QHBoxLayout()
        eq = QPushButton("Equal")
        eq.clicked.connect(lambda: self.group_weights.set_weights([100.0 / max(1, self.group_weights.count())]
                                                                  * self.group_weights.count()))
        g_btns.addWidget(eq)
        ahp = QPushButton("AHP…")
        ahp.clicked.connect(self.on_group_ahp)
        g_btns.addWidget(ahp)
        g_btns.addStretch(1)
        v_g.addLayout(g_btns)
        lay.addWidget(self.grp_groups)

        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.groups_container)

        self.eff_lbl = QLabel("")
        self.eff_lbl.setWordWrap(True)
        self.eff_lbl.setObjectName("muted")
        lay.addWidget(self.eff_lbl)

        grp_o = QGroupBox("Run Options")
        f_o = QFormLayout(grp_o)
        self.theme_combo = QComboBox()
        for key, lbl in THEME_OPTIONS:
            self.theme_combo.addItem(lbl, key)
        f_o.addRow("Map theme:", self.theme_combo)
        self.class_mode_combo = QComboBox()
        self.class_mode_combo.addItem("Equal interval (0–20–40–60–80–100)", "equal")
        self.class_mode_combo.addItem("Quantile (5 equal-count bands)", "quantile")
        self.class_mode_combo.addItem("Natural breaks (Fisher-Jenks)", "jenks")
        f_o.addRow("5-class breaks:", self.class_mode_combo)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 0.5)
        self.smooth_spin.setSingleStep(0.05)
        self.smooth_spin.setToolTip("Blend each hexagon with its 1-ring neighbour mean (0 = off).")
        f_o.addRow("Contagion smoothing α:", self.smooth_spin)
        self.constraints_chk = QCheckBox("Apply constraints")
        f_o.addRow(self.constraints_chk)
        mc_row = QHBoxLayout()
        self.mc_chk = QCheckBox("Monte Carlo")
        mc_row.addWidget(self.mc_chk)
        self.mc_runs = QSpinBox()
        self.mc_runs.setRange(10, 1000)
        self.mc_runs.setSuffix(" runs")
        mc_row.addWidget(self.mc_runs)
        self.mc_pert = QSpinBox()
        self.mc_pert.setRange(1, 50)
        self.mc_pert.setPrefix("± ")
        self.mc_pert.setSuffix(" %")
        mc_row.addWidget(self.mc_pert)
        f_o.addRow(mc_row)
        hs_row = QHBoxLayout()
        self.hot_chk = QCheckBox("Hotspots (Gi*)")
        hs_row.addWidget(self.hot_chk)
        self.hot_k = QSpinBox()
        self.hot_k.setRange(1, 6)
        self.hot_k.setPrefix("k = ")
        self.hot_k.setToolTip("Neighbourhood size in H3 grid steps (1 = 6 cells, 2 = 18, 3 = 36)")
        hs_row.addWidget(self.hot_k)
        self.hot_fdr = QCheckBox("FDR")
        self.hot_fdr.setToolTip("Benjamini-Hochberg false discovery rate correction")
        hs_row.addWidget(self.hot_fdr)
        f_o.addRow(hs_row)
        self.sens_chk = QCheckBox("Weight sensitivity (±10 %, ±20 %)")
        f_o.addRow(self.sens_chk)
        for w in (self.theme_combo, self.class_mode_combo):
            w.currentIndexChanged.connect(self._options_to_model)
        for w in (self.smooth_spin, self.mc_runs, self.mc_pert, self.hot_k):
            w.valueChanged.connect(self._options_to_model)
        for w in (self.constraints_chk, self.mc_chk, self.hot_chk, self.hot_fdr, self.sens_chk):
            w.toggled.connect(self._options_to_model)
        lay.addWidget(grp_o)

        scen = QHBoxLayout()
        save_b = QPushButton("💾 Save Model…")
        save_b.clicked.connect(self.on_save_model)
        scen.addWidget(save_b)
        load_b = QPushButton("📂 Load Model…")
        load_b.clicked.connect(self.on_load_model)
        scen.addWidget(load_b)
        lay.addLayout(scen)
        lay.addStretch(1)
        return page

    # ---- Tab 4: results ---------------------------------------------------------
    def _build_tab_results(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        grp = QGroupBox("Composite Index")
        v = QVBoxLayout(grp)
        self.kpi_label = QLabel("Run the model to see diagnostics.")
        self.kpi_label.setWordWrap(True)
        v.addWidget(self.kpi_label)
        self.summary_table = QTableWidget(0, 5)
        self.summary_table.setHorizontalHeaderLabels(["Class", "Hexagons", "Area (ha)", "Share", "Top driver"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setMinimumHeight(185)
        v.addWidget(self.summary_table)
        self.groups_lbl = QLabel("")
        self.groups_lbl.setWordWrap(True)
        v.addWidget(self.groups_lbl)
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Show on map:"))
        self.map_field_combo = QComboBox()
        self.map_field_combo.currentIndexChanged.connect(self.on_restyle)
        style_row.addWidget(self.map_field_combo, 1)
        v.addLayout(style_row)
        lay.addWidget(grp)

        grp_h = QGroupBox("Hotspots — Getis-Ord Gi* && Moran's I")
        vh = QVBoxLayout(grp_h)
        self.hotspot_lbl = QLabel("<i>Enable hotspots in Run Options.</i>")
        self.hotspot_lbl.setWordWrap(True)
        vh.addWidget(self.hotspot_lbl)
        lay.addWidget(grp_h)

        grp_s = QGroupBox("Weight Sensitivity")
        vs = QVBoxLayout(grp_s)
        self.sens_table = QTableWidget(0, 4)
        self.sens_table.setHorizontalHeaderLabels(["Weight", "Value %", "Mean |Δ|", "Class change %"])
        self.sens_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sens_table.setMinimumHeight(130)
        vs.addWidget(self.sens_table)
        lay.addWidget(grp_s)

        exp = QHBoxLayout()
        csv_b = QPushButton("Export CSV…")
        csv_b.clicked.connect(self.on_export_csv)
        exp.addWidget(csv_b)
        html_b = QPushButton("Executive HTML Report…")
        html_b.setObjectName("primaryTealBtn")
        html_b.clicked.connect(self.on_export_html_report)
        exp.addWidget(html_b)
        lay.addLayout(exp)
        lay.addStretch(1)
        return page

    # ============================================================ helpers
    def log(self, msg: str) -> None:
        self.log_box.append(msg)

    def _headless(self) -> bool:
        return os.environ.get("QT_QPA_PLATFORM") == "offscreen"

    def _warn(self, msg: str) -> None:
        self.log(f"[WARN] {msg}")
        if self._headless():
            raise RuntimeError(msg)
        QMessageBox.warning(self, "02H3Forge", msg)

    def _error(self, msg: str) -> None:
        self.log(f"[ERROR] {msg}")
        if self._headless():
            raise RuntimeError(msg)
        QMessageBox.critical(self, "02H3Forge", msg)

    @contextmanager
    def _busy(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()

    @staticmethod
    def _select(combo: QComboBox, data) -> None:
        i = combo.findData(data)
        if i >= 0:
            combo.setCurrentIndex(i)

    def active_h3_layer(self) -> Optional[QgsVectorLayer]:
        lid = self.h3_layer_combo.currentData()
        lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        return lyr if isinstance(lyr, QgsVectorLayer) and lyr.isValid() else None

    def _refresh_canvas(self) -> None:
        if self.iface.mapCanvas() is not None:
            self.iface.mapCanvas().refresh()

    # ======================================================= project sync
    def _connect_project_signals(self) -> None:
        proj = QgsProject.instance()
        on_added = lambda _l: self.refresh_layer_combos()  # noqa: E731
        on_removed = lambda _l: self.refresh_layer_combos()  # noqa: E731
        with suppress(Exception):
            proj.layersAdded.connect(on_added)
            self._project_slots.append((proj.layersAdded, on_added))
            proj.layersRemoved.connect(on_removed)
            self._project_slots.append((proj.layersRemoved, on_removed))

    def dispose(self) -> None:
        if self._task is not None:
            with suppress(Exception):
                self._task.cancel()
        for signal, slot in self._project_slots:
            with suppress(Exception):
                signal.disconnect(slot)
        self._project_slots = []

    def refresh_layer_combos(self) -> None:
        prev_h3 = self.h3_layer_combo.currentData()
        prev_bnd = self.boundary_layer_combo.currentData()
        layers = list(QgsProject.instance().mapLayers().values())
        with QSignalBlocker(self.h3_layer_combo):
            self.h3_layer_combo.clear()
            first_h3 = -1
            for lyr in layers:
                if (isinstance(lyr, QgsVectorLayer) and lyr.isValid()
                        and lyr.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry
                        and lyr.fields().indexFromName("h3_id") >= 0):
                    self.h3_layer_combo.addItem(lyr.name(), lyr.id())
                    if first_h3 < 0:
                        first_h3 = self.h3_layer_combo.count() - 1
            idx = self.h3_layer_combo.findData(prev_h3) if prev_h3 else -1
            self.h3_layer_combo.setCurrentIndex(idx if idx >= 0 else max(first_h3, 0))
        with QSignalBlocker(self.boundary_layer_combo):
            self.boundary_layer_combo.clear()
            for lyr in layers:
                if isinstance(lyr, QgsMapLayer) and lyr.isValid() and not (
                    isinstance(lyr, QgsVectorLayer) and lyr.fields().indexFromName("h3_id") >= 0
                ):
                    self.boundary_layer_combo.addItem(lyr.name(), lyr.id())
            idx = self.boundary_layer_combo.findData(prev_bnd) if prev_bnd else -1
            if idx >= 0:
                self.boundary_layer_combo.setCurrentIndex(idx)
        if self.h3_layer_combo.currentData() != prev_h3:
            self._on_active_layer_changed()
        self._update_grid_estimate()

    def _on_active_layer_changed(self) -> None:
        lyr = self.active_h3_layer()
        if lyr is not None:
            stored = lyr.customProperty("h3forge/model", "")
            if stored:
                with suppress(ValueError):
                    self.model = ModelSpec.from_json(str(stored))
                    self.log(f"Loaded model '{self.model.name}' stored on '{lyr.name()}'.")
            elif not self.model.criteria:
                self._auto_import_legacy(lyr)
            engine = lyr.customProperty("h3forge/engine", "")
            self.grid_info_lbl.setText(
                f"Active grid: <b>{lyr.name()}</b> · {lyr.featureCount():,} hexagons · {lyr.crs().authid()}"
                + (f" · {engine}" if engine else "")
            )
        else:
            self.grid_info_lbl.setText("Generate or load an H3 grid to begin.")
        self.refresh_model_views()
        self.refresh_results()

    def _auto_import_legacy(self, lyr: QgsVectorLayer) -> None:
        norm = [f.name() for f in lyr.fields() if f.name().startswith("norm_") and f.isNumeric()]
        if not norm:
            return
        for fld in norm:
            cid = self.model.unique_id(fld[5:])
            self.model.criteria.append(CriterionSpec(
                id=cid, label=cid, group=self.model.groups[0].id,
                source=SourceSpec(kind="field", field=fld), value=ValueFunction("direct_clip"),
            ))
        self.model.normalize_weights()
        self.log(f"Registered {len(norm)} existing norm_* field(s) as criteria (v0.2 layer).")

    # =================================================== grid generation
    def _resolve_grid_extent(self):
        ext_mode = str(self.extent_source_combo.currentData() or "layer")
        bnd = None
        extent = None
        crs = QgsProject.instance().crs()
        if not crs.isValid():
            crs = QgsCoordinateReferenceSystem("EPSG:4326")
        if ext_mode == "layer":
            lid = self.boundary_layer_combo.currentData()
            bnd = QgsProject.instance().mapLayer(str(lid)) if lid else None
            if bnd is not None and bnd.isValid():
                extent, crs = bnd.extent(), bnd.crs()
        if extent is None or extent.isEmpty():
            canvas = self.iface.mapCanvas()
            if canvas is None or canvas.extent().isEmpty():
                return None
            extent, crs = canvas.extent(), canvas.mapSettings().destinationCrs()
        return extent, crs, bnd

    def _update_grid_estimate(self) -> None:
        res = int(self.res_combo.currentData() or DEFAULT_H3_RESOLUTION)
        resolved = None
        with suppress(Exception):
            resolved = self._resolve_grid_extent()
        if resolved is None:
            self.grid_estimate_lbl.setText("Load a reference layer or zoom the canvas to preview the grid size.")
            return
        try:
            work_crs, ext = resolve_metric_work_crs(resolved[0], resolved[1])
            est = estimate_hex_count(ext.width(), ext.height(), res)
        except Exception:
            self.grid_estimate_lbl.setText("")
            return
        msg = f"≈ {est:,} hexagons in the bounding box at Res {res} · metric CRS {work_crs.authid()}"
        if est > MAX_METRIC_CELLS:
            msg = (f"<span style='color:#b30000;'><b>⚠ ≈ {est:,} hexagons</b> exceeds the "
                   f"{MAX_METRIC_CELLS:,} cell budget — choose a coarser resolution or smaller extent.</span>")
        self.grid_estimate_lbl.setText(msg)

    def on_generate_h3_grid(self) -> None:
        res = int(self.res_combo.currentData() or DEFAULT_H3_RESOLUTION)
        resolved = self._resolve_grid_extent()
        if resolved is None:
            self._warn("Load a reference layer or zoom the map canvas before generating a grid.")
            return
        extent, crs, bnd = resolved
        try:
            with self._busy():
                lyr = build_h3_grid_layer(
                    extent, crs, res,
                    boundary_layer=bnd if isinstance(bnd, QgsVectorLayer) else None,
                    coverage_mode=str(self.coverage_combo.currentData() or "intersect"),
                )
        except Exception as exc:
            self._error(f"H3 grid generation failed:\n{exc}")
            return
        lyr.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
            {"color": "19,160,160,30", "outline_color": "#0f6b6b", "outline_width": "0.26"})))
        QgsProject.instance().addMapLayer(lyr)
        self.refresh_layer_combos()
        self._select(self.h3_layer_combo, lyr.id())
        self._on_active_layer_changed()
        engine = lyr.customProperty("h3forge/engine", "")
        self.log(f"[Grid] '{lyr.name()}': {lyr.featureCount():,} hexagons, Res {res}, "
                 f"{lyr.crs().authid()}, {engine}.")
        if bool(lyr.customProperty("h3forge/coarsened", False)):
            self.log("[WARN] Cell budget exceeded: hexagons were coarsened.")

    def on_save_to_geopackage(self) -> None:
        lyr = self.active_h3_layer()
        if lyr is None:
            self._warn("Generate or select an H3 layer first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save H3 layer", f"{lyr.name()}.gpkg", "GeoPackage (*.gpkg)")
        if not path:
            return
        try:
            with self._busy():
                new = save_h3_layer_to_geopackage(lyr, path)
        except Exception as exc:
            self._error(f"GeoPackage export failed:\n{exc}")
            return
        QgsProject.instance().addMapLayer(new)
        self.refresh_layer_combos()
        self._select(self.h3_layer_combo, new.id())
        self.log(f"[Grid] Saved to {path}; the studio now uses the GeoPackage layer.")

    # =================================================== model ↔ views
    def refresh_model_views(self) -> None:
        m = self.model
        with QSignalBlocker(self.model_name_edit):
            self.model_name_edit.setText(m.name)
        with QSignalBlocker(self.template_combo):
            self._select(self.template_combo, m.template)
        self._fill_tree()
        self._fill_constraints()
        self._rebuild_weights()
        self._options_from_model()

    def _fill_tree(self) -> None:
        kinds = dict(SOURCE_KINDS)
        with QSignalBlocker(self.crit_tree):
            self.crit_tree.clear()
            for g in self.model.groups:
                gi = QTreeWidgetItem([f"{g.label}", "", f"{len(self.model.criteria_in(g.id))} criteria"])
                gi.setData(0, Qt.ItemDataRole.UserRole, ("group", g.id))
                f = gi.font(0)
                f.setBold(True)
                gi.setFont(0, f)
                self.crit_tree.addTopLevelItem(gi)
                for c in self.model.criteria_in(g.id):
                    src = c.source.layer_name or c.source.field
                    kind = kinds.get(c.source.kind, c.source.kind).split(" (")[0]
                    ci = QTreeWidgetItem([c.label, f"{kind}: {src}", c.value.describe()])
                    ci.setData(0, Qt.ItemDataRole.UserRole, ("criterion", c.id))
                    ci.setFlags(ci.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    ci.setCheckState(0, Qt.CheckState.Checked if c.active else Qt.CheckState.Unchecked)
                    ci.setToolTip(0, f"columns: {c.raw_field} → {c.norm_field}")
                    gi.addChild(ci)
                gi.setExpanded(True)

    def _fill_constraints(self) -> None:
        with QSignalBlocker(self.cons_list):
            self.cons_list.clear()
            for k in self.model.constraints:
                it = QListWidgetItem(f"{k.label} — {k.describe()}")
                it.setData(Qt.ItemDataRole.UserRole, k.id)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if k.active else Qt.CheckState.Unchecked)
                self.cons_list.addItem(it)

    def _rebuild_weights(self) -> None:
        m = self.model
        groups = m.active_groups()
        self.grp_groups.setVisible(len(groups) > 1)
        with QSignalBlocker(self.agg_combo):
            self._select(self.agg_combo, m.aggregation)
        g_tot = sum(max(0.0, g.weight) for g in groups) or 1.0
        self.group_weights.set_items([g.label for g in groups],
                                     [max(0.0, g.weight) / g_tot * 100.0 for g in groups],
                                     [g.locked for g in groups])
        while self.groups_layout.count():
            w = self.groups_layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)  # detach now; deleteLater alone leaves ghosts until idle
                w.deleteLater()
        self._group_widgets = {}
        for g in m.groups:
            crits = m.criteria_in(g.id, active_only=True)
            if not crits:
                continue
            box = QGroupBox(f"{g.label} — criteria weights")
            v = QVBoxLayout(box)
            mrow = QHBoxLayout()
            mrow.addWidget(QLabel("Method:"))
            mcombo = QComboBox()
            for key, lbl in GROUP_METHODS:
                mcombo.addItem(lbl, key)
            self._select(mcombo, g.method)
            mrow.addWidget(mcombo, 1)
            alpha = QDoubleSpinBox()
            alpha.setRange(0.1, 10.0)
            alpha.setSingleStep(0.25)
            alpha.setPrefix("α ")
            alpha.setValue(float(g.owa_alpha))
            alpha.setToolTip("WOWA risk attitude: α > 1 precautionary, α = 1 WLC, α < 1 opportunity")
            alpha.setVisible(g.method == "owa")
            orn = QLabel(f"orness {owa_orness(g.owa_alpha):.2f}" if g.method == "owa" else "")
            orn.setObjectName("muted")
            mrow.addWidget(alpha)
            mrow.addWidget(orn)
            v.addLayout(mrow)

            def _on_method(_i, gid=g.id, cb=mcombo, sp=alpha, lb=orn):
                grp = self.model.group(gid)
                grp.method = str(cb.currentData())
                sp.setVisible(grp.method == "owa")
                lb.setText(f"orness {owa_orness(grp.owa_alpha):.2f}" if grp.method == "owa" else "")

            def _on_alpha(val, gid=g.id, lb=orn):
                self.model.group(gid).owa_alpha = float(val)
                lb.setText(f"orness {owa_orness(float(val)):.2f}")

            mcombo.currentIndexChanged.connect(_on_method)
            alpha.valueChanged.connect(_on_alpha)

            cw = CoupledWeightsWidget()
            cw.set_items([c.label for c in crits], [c.weight for c in crits], [c.locked for c in crits],
                         [f"{c.norm_field} · {c.value.describe()}" for c in crits])
            cw.weightsChanged.connect(lambda w, gid=g.id: self._on_crit_weights(gid, w))
            v.addWidget(cw)
            brow = QHBoxLayout()
            for text, key in (("Equal", "equal"), ("CRITIC", "critic"), ("Entropy", "entropy"),
                              ("PCA", "pca"), ("AHP…", "ahp")):
                b = QPushButton(text)
                b.clicked.connect(lambda _c, gid=g.id, k=key: self.on_objective_weights(gid, k))
                brow.addWidget(b)
            v.addLayout(brow)
            self.groups_layout.addWidget(box)
            self._group_widgets[g.id] = cw
        self._update_effective_label()

    def _update_effective_label(self) -> None:
        eff = self.model.effective_weights()
        if not eff:
            self.eff_lbl.setText("<i>Add criteria in tab 2 to start weighting.</i>")
            return
        parts = [f"{self.model.criterion(cid).label} {w:.1f}%"
                 for cid, w in sorted(eff.items(), key=lambda kv: -kv[1])]
        self.eff_lbl.setText("<b>Effective weights:</b> " + " · ".join(parts))

    def _options_from_model(self) -> None:
        m = self.model
        widgets = (self.theme_combo, self.class_mode_combo, self.smooth_spin, self.constraints_chk,
                   self.mc_chk, self.mc_runs, self.mc_pert, self.hot_chk, self.hot_k, self.hot_fdr, self.sens_chk)
        blockers = [QSignalBlocker(w) for w in widgets]
        self._select(self.theme_combo, m.theme)
        self._select(self.class_mode_combo, m.class_mode)
        self.smooth_spin.setValue(m.smooth_alpha)
        self.constraints_chk.setChecked(m.apply_constraints)
        self.mc_chk.setChecked(m.run_montecarlo)
        self.mc_runs.setValue(m.mc_runs)
        self.mc_pert.setValue(int(round(m.mc_perturbation * 100)))
        self.hot_chk.setChecked(m.run_hotspots)
        self.hot_k.setValue(m.hotspot_k)
        self.hot_fdr.setChecked(m.hotspot_fdr)
        self.sens_chk.setChecked(m.run_sensitivity)
        del blockers

    def _options_to_model(self, *_args) -> None:
        m = self.model
        m.theme = str(self.theme_combo.currentData())
        m.class_mode = str(self.class_mode_combo.currentData())
        m.smooth_alpha = float(self.smooth_spin.value())
        m.apply_constraints = self.constraints_chk.isChecked()
        m.run_montecarlo = self.mc_chk.isChecked()
        m.mc_runs = int(self.mc_runs.value())
        m.mc_perturbation = self.mc_pert.value() / 100.0
        m.run_hotspots = self.hot_chk.isChecked()
        m.hotspot_k = int(self.hot_k.value())
        m.hotspot_fdr = self.hot_fdr.isChecked()
        m.run_sensitivity = self.sens_chk.isChecked()

    # weights callbacks
    def _on_group_weights(self, weights: List[float]) -> None:
        locks = self.group_weights.locked()
        for g, w, lk in zip(self.model.active_groups(), weights, locks):
            g.weight, g.locked = float(w), bool(lk)
        self._update_effective_label()

    def _on_crit_weights(self, gid: str, weights: List[float]) -> None:
        cw = self._group_widgets.get(gid)
        locks = cw.locked() if cw else [False] * len(weights)
        for c, w, lk in zip(self.model.criteria_in(gid, active_only=True), weights, locks):
            c.weight, c.locked = float(w), bool(lk)
        self._update_effective_label()

    def _on_aggregation(self) -> None:
        self.model.aggregation = str(self.agg_combo.currentData())

    def _on_model_name(self) -> None:
        self.model.name = self.model_name_edit.text().strip() or self.model.name

    def on_group_ahp(self) -> None:
        groups = self.model.active_groups()
        if len(groups) < 2:
            return
        dlg = AhpPairwiseDialog([g.label for g in groups], parent=self)
        if dlg.exec():
            self.group_weights.set_weights(dlg.computed_weights)
            self.log(f"[Weights] Group AHP applied (CR = {dlg.cr_value:.3f}).")

    def on_objective_weights(self, gid: str, key: str) -> None:
        crits = self.model.criteria_in(gid, active_only=True)
        cw = self._group_widgets.get(gid)
        if cw is None or not crits:
            return
        n = len(crits)
        if key == "equal":
            cw.set_weights([100.0 / n] * n)
            return
        if key == "ahp":
            if n < 2:
                return
            dlg = AhpPairwiseDialog([c.label for c in crits], parent=self)
            if dlg.exec():
                cw.set_weights(dlg.computed_weights)
                self.log(f"[Weights] AHP applied to '{self.model.group(gid).label}' (CR = {dlg.cr_value:.3f}).")
            return
        lyr = self.active_h3_layer()
        cols = [c.norm_field for c in crits]
        if lyr is None or any(lyr.fields().indexFromName(cname) < 0 for cname in cols):
            self._warn("Objective weights need criterion scores: run the model once first.")
            return
        idx = [lyr.fields().indexFromName(cname) for cname in cols]
        rows = []
        for f in lyr.getFeatures():
            a = f.attributes()
            rows.append([float(a[i]) if a[i] is not None else 0.0 for i in idx])
        mat = np.asarray(rows, dtype=np.float64)
        if key == "critic":
            w = critic_weights_pct(mat)[0]
        elif key == "entropy":
            w = entropy_weights_pct(mat)
        else:
            w = pca_weights_pct(mat)
        cw.set_weights(w)
        self.log(f"[Weights] {key.upper()} applied to '{self.model.group(gid).label}': "
                 f"{', '.join(f'{c.label} {x:.1f}%' for c, x in zip(crits, w))}.")

    # =================================================== criteria actions
    def _selected(self):
        it = self.crit_tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _on_tree_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "criterion":
            return
        c = self.model.criterion(data[1])
        c.active = item.checkState(0) == Qt.CheckState.Checked
        if c.active and c.weight <= 0.0:
            c.weight = 100.0 / max(1, len(self.model.criteria_in(c.group, active_only=True)))
        self.model.normalize_weights()
        self._rebuild_weights()

    def on_apply_template(self) -> None:
        tpl = str(self.template_combo.currentData())
        if tpl == self.model.template:
            return
        self.model = apply_template(self.model, tpl)
        self.refresh_model_views()
        self.log(f"[Model] Structure switched to '{dict(TEMPLATES)[tpl]}'. Move criteria between groups via Edit.")

    def add_criterion(self, spec: CriterionSpec) -> None:
        """Add a criterion (public for scripting/tests); gives it a fair weight share."""
        peers = self.model.criteria_in(spec.group, active_only=True)
        spec.weight = 100.0 / (len(peers) + 1) if spec.active else 0.0
        self.model.criteria.append(spec)
        self.model.normalize_weights()
        self.refresh_model_views()

    def on_add_criterion(self) -> None:
        sel = self._selected()
        dlg = CriterionDialog(self.model, self.active_h3_layer(), parent=self)
        if sel and sel[0] == "group":
            self._select(dlg.group_combo, sel[1])
        if dlg.exec() and dlg.result_spec is not None:
            self.add_criterion(dlg.result_spec)
            self.log(f"[Criteria] Added '{dlg.result_spec.label}'.")

    def on_edit_criterion(self) -> None:
        sel = self._selected()
        if not sel or sel[0] != "criterion":
            return
        old = self.model.criterion(sel[1])
        dlg = CriterionDialog(self.model, self.active_h3_layer(), old, parent=self)
        if dlg.exec() and dlg.result_spec is not None:
            i = self.model.criteria.index(old)
            self.model.criteria[i] = dlg.result_spec
            self.model.normalize_weights()
            self.refresh_model_views()

    def on_duplicate_criterion(self) -> None:
        sel = self._selected()
        if not sel or sel[0] != "criterion":
            return
        import copy as _copy

        c = _copy.deepcopy(self.model.criterion(sel[1]))
        c.label = f"{c.label} (copy)"
        c.id = self.model.unique_id(c.label)
        self.add_criterion(c)

    def on_remove_criterion(self) -> None:
        sel = self._selected()
        if not sel or sel[0] != "criterion":
            return
        self.model.criteria = [c for c in self.model.criteria if c.id != sel[1]]
        self.model.normalize_weights()
        self.refresh_model_views()

    def on_import_fields(self) -> None:
        lyr = self.active_h3_layer()
        if lyr is None:
            self._warn("Select an H3 layer first.")
            return
        used = [c.source.field for c in self.model.criteria if c.source.kind == "field"]
        used += [c.raw_field for c in self.model.criteria] + [c.norm_field for c in self.model.criteria]
        dlg = FieldImportDialog(lyr, used, self)
        if not dlg.exec():
            return
        for fld in dlg.selected():
            base = fld.split("_", 1)[1] if fld.startswith(("norm_", "raw_", "ring_")) else fld
            ready = fld.startswith(("norm_", "ring_"))
            self.add_criterion(CriterionSpec(
                id=self.model.unique_id(base), label=base, group=self.model.groups[0].id,
                source=SourceSpec(kind="field", field=fld),
                value=ValueFunction("direct_clip" if ready else "minmax"),
            ))

    def on_add_group(self) -> None:
        from qgis.PyQt.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "02H3Forge", "New group name:")
        if ok and name.strip():
            gid = self.model.unique_id(name)
            share = 100.0 / (len(self.model.groups) + 1)
            self.model.groups.append(GroupSpec(gid, name.strip(), share))
            self.model.normalize_weights()
            self.model.template = "custom"
            self.refresh_model_views()

    def _selected_group(self) -> Optional[GroupSpec]:
        sel = self._selected()
        if not sel:
            return None
        gid = sel[1] if sel[0] == "group" else self.model.criterion(sel[1]).group
        return self.model.group(gid)

    def on_rename_group(self) -> None:
        from qgis.PyQt.QtWidgets import QInputDialog

        g = self._selected_group()
        if g is None:
            return
        name, ok = QInputDialog.getText(self, "02H3Forge", "Group name:", text=g.label)
        if ok and name.strip():
            g.label = name.strip()
            self.refresh_model_views()

    def on_remove_group(self) -> None:
        g = self._selected_group()
        if g is None or len(self.model.groups) <= 1:
            return
        if self.model.criteria_in(g.id):
            self._warn("Move or remove the group's criteria first.")
            return
        self.model.groups = [x for x in self.model.groups if x.id != g.id]
        self.model.normalize_weights()
        self.refresh_model_views()

    # constraints
    def _on_constraint_toggled(self, item: QListWidgetItem) -> None:
        cid = item.data(Qt.ItemDataRole.UserRole)
        for k in self.model.constraints:
            if k.id == cid:
                k.active = item.checkState() == Qt.CheckState.Checked

    def on_add_constraint(self) -> None:
        dlg = ConstraintDialog(self.model, self.active_h3_layer(), parent=self)
        if dlg.exec() and dlg.result_spec is not None:
            self.model.constraints.append(dlg.result_spec)
            self._fill_constraints()

    def on_edit_constraint(self) -> None:
        it = self.cons_list.currentItem()
        if it is None:
            return
        cid = it.data(Qt.ItemDataRole.UserRole)
        old = next((k for k in self.model.constraints if k.id == cid), None)
        dlg = ConstraintDialog(self.model, self.active_h3_layer(), old, parent=self)
        if dlg.exec() and dlg.result_spec is not None:
            self.model.constraints[self.model.constraints.index(old)] = dlg.result_spec
            self._fill_constraints()

    def on_remove_constraint(self) -> None:
        it = self.cons_list.currentItem()
        if it is None:
            return
        cid = it.data(Qt.ItemDataRole.UserRole)
        self.model.constraints = [k for k in self.model.constraints if k.id != cid]
        self._fill_constraints()

    # model files
    def on_save_model(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save 02H3Forge model", f"{self.model.name}.h3model.json",
                                              "02H3Forge model (*.json)")
        if path:
            Path(path).write_text(self.model.to_json(), encoding="utf-8")
            self.log(f"[Model] Saved to {path}")

    def on_load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load 02H3Forge model", "", "02H3Forge model (*.json)")
        if not path:
            return
        try:
            self.model = ModelSpec.from_json(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._error(f"Could not load model:\n{exc}")
            return
        self.refresh_model_views()
        self.log(f"[Model] Loaded '{self.model.name}' from {path}")

    # ======================================================= running
    def run_model(self, blocking: bool = False) -> bool:
        """Validate, snapshot and compute the model (background unless ``blocking``)."""
        if self._task is not None:
            return False
        lyr = self.active_h3_layer()
        if lyr is None:
            self._warn("Generate or select an H3 grid first (tab 1).")
            return False
        self._options_to_model()
        self.model.normalize_weights()
        try:
            with self._busy():
                inputs = prepare_run(lyr, self.model, force=self.force_chk.isChecked())
        except Exception as exc:
            self._error(str(exc))
            return False
        n_new = sum(1 for c in self.model.criteria if c.id not in inputs.reuse_raw)
        self.log(f"[Run] '{self.model.name}': {len(self.model.criteria)} criteria "
                 f"({n_new} to compute, {len(self.model.criteria) - n_new} cached), "
                 f"{len([k for k in self.model.constraints if k.active])} constraints.")
        self.progress.setValue(0)
        if blocking:
            try:
                def _progress(pct: float) -> None:
                    self.progress.setValue(int(pct))
                    QApplication.processEvents()

                result = compute_run(inputs, _progress)
            except Exception as exc:
                self._error(f"Model run failed:\n{exc}")
                return False
            self._apply_result(lyr, inputs.snap, result)
            return True
        self._task = _ModelTask(inputs, self._on_task_done)
        self._task_layer_id = lyr.id()
        self._task_snap = inputs.snap
        self._task.progressChanged.connect(lambda p: self.progress.setValue(int(p)))
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        QgsApplication.taskManager().addTask(self._task)
        return True

    def cancel_run(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self.log("[Run] Cancelling…")

    def _on_task_done(self, task: _ModelTask, ok: bool) -> None:
        self._task = None
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        lyr = QgsProject.instance().mapLayer(self._task_layer_id or "")
        if not ok:
            self.progress.setValue(0)
            # A task cancelled before it started never enters run(): still a cancellation.
            if task.cancelled_by_user or task.isCanceled():
                self.log("[Run] Cancelled — the layer was not modified.")
            else:
                self._error(f"Model run failed:\n{task.error or 'unknown error'}")
            return
        if lyr is None:
            self.log("[Run] The H3 layer was removed during the run; results discarded.")
            return
        self._apply_result(lyr, self._task_snap, task.result)

    def _apply_result(self, lyr, snap, result) -> None:
        with self._busy():
            apply_run(lyr, result, snap)
        for line in result.log:
            self.log(f"  · {line}")
        st = result.stats
        self.log(f"[Run] Done: mean {st['mean_score']}, max {st['max_score']}, "
                 f"{st['excluded_count']:,} excluded, mean MC std {st['mean_mc_std']}.")
        self.progress.setValue(100)
        self._refresh_canvas()
        self.refresh_results()
        self.tabs.setCurrentIndex(3)

    # ======================================================= results
    def refresh_results(self) -> None:
        self.summary_table.setRowCount(0)
        self.sens_table.setRowCount(0)
        lyr = self.active_h3_layer()
        with QSignalBlocker(self.map_field_combo):
            self.map_field_combo.clear()
            if lyr is not None:
                names = [f.name() for f in lyr.fields()]
                for fld, lbl in (("comp_score", "Composite index"), ("rob_score", "Robust score (MC)"),
                                 ("mc_std", "Uncertainty (MC std)"), ("hotspot_lbl", "Hotspots (Gi*)")):
                    if fld in names:
                        self.map_field_combo.addItem(lbl, fld)
                for g in self.model.groups:
                    if f"idx_{g.id}" in names:
                        self.map_field_combo.addItem(f"Sub-index: {g.label}", f"idx_{g.id}")
        if lyr is None or lyr.fields().indexFromName("comp_score") < 0:
            self.kpi_label.setText("Run the model to see diagnostics.")
            self.groups_lbl.setText("")
            return
        run = read_layer_json_property(lyr, "h3forge/last_run")
        theme = str(run.get("theme", self.model.theme))
        rows = compute_class_summary(lyr, theme=theme)
        tot_hex = sum(int(r["hex_count"]) for r in rows)
        tot_ha = sum(float(r["area_ha"]) for r in rows)
        excl = sum(int(v) for v in (run.get("excluded_by") or {}).values()) if run else 0
        self.kpi_label.setText(f"<b>{tot_hex:,}</b> hexagons · <b>{tot_ha:,.1f}</b> ha"
                               + (f" · constraints hit {excl:,} times" if excl else ""))
        for i, r in enumerate(rows):
            self.summary_table.insertRow(i)
            for col, txt in enumerate((f"{r['class_id']} — {r['class_label']}", f"{r['hex_count']:,}",
                                       f"{r['area_ha']:.2f}", f"{r['share_pct']:.1f}%", str(r["top_driver"]))):
                it = QTableWidgetItem(txt)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.summary_table.setItem(i, col, it)
        groups = run.get("groups") or []
        if len(groups) > 1:
            self.groups_lbl.setText("<b>Groups:</b> " + " · ".join(
                f"{g['label']} {g['weight']:.0f}% ({g['method'].upper()})" for g in groups)
                + f" — combined by <b>{dict(AGGREGATIONS).get(run.get('aggregation'), '')}</b>")
        else:
            self.groups_lbl.setText("")
        hot = read_layer_json_property(lyr, "h3forge/hotspot")
        self._show_hotspot_summary(hot) if hot else self.hotspot_lbl.setText("<i>Hotspots not computed.</i>")
        sens = read_layer_json_property(lyr, "h3forge/sensitivity")
        for i, r in enumerate(sens.get("rows", []) if sens else []):
            self.sens_table.insertRow(i)
            for col, txt in enumerate((str(r["criterion"]), f"{float(r['weight']):.1f}",
                                       f"{float(r['mean_abs_change']):.2f}", f"{float(r['class_change_pct']):.1f}")):
                it = QTableWidgetItem(txt)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.sens_table.setItem(i, col, it)

    def _show_hotspot_summary(self, res: Dict[str, object]) -> None:
        counts = res.get("counts", {}) or {}
        moran = res.get("moran", {}) or {}
        hot = sum(v for k, v in counts.items() if k.startswith("Hot"))
        cold = sum(v for k, v in counts.items() if k.startswith("Cold"))
        try:
            i_val, z_val, p_val = float(moran["I"]), float(moran["z"]), float(moran["p"])
            verdict = ("clustered ✔" if z_val > 0 else "dispersed") if p_val < 0.05 else "random (not significant)"
            p_txt = "p &lt; 0.0001" if p_val < 1e-4 else f"p = {p_val:.4f}"
            moran_txt = f"Moran's I = <b>{i_val:.3f}</b> (z = {z_val:.2f}, {p_txt}) → <b>{verdict}</b>"
        except (KeyError, TypeError, ValueError):
            moran_txt = "Moran's I: n/a (too few or constant values)"
        k = res.get("k_ring", 1)
        fdr = "FDR-corrected" if res.get("fdr") else "uncorrected"
        self.hotspot_lbl.setText(
            f"{moran_txt}<br><span style='color:#b2182b;'><b>{hot:,}</b> hot</span> · "
            f"<span style='color:#2166ac;'><b>{cold:,}</b> cold</span> hexagons "
            f"({k}-ring, {fdr}, 90/95/99%)."
        )

    def on_restyle(self) -> None:
        lyr = self.active_h3_layer()
        fld = self.map_field_combo.currentData()
        if lyr is None or not fld:
            return
        if fld == "hotspot_lbl":
            apply_h3_hotspot_style(lyr)
        elif fld == "comp_score":
            run = read_layer_json_property(lyr, "h3forge/last_run")
            apply_h3_graduated_style(lyr, "comp_score", theme=str(run.get("theme", "risk")),
                                     breaks=run.get("breaks"))
        elif fld == "mc_std":
            vals = [f["mc_std"] for f in lyr.getFeatures() if f["mc_std"] is not None]
            qs = list(np.percentile(vals, [0, 20, 40, 60, 80, 100])) if vals else None
            apply_h3_graduated_style(lyr, "mc_std", theme="magma", breaks=qs)
        else:
            apply_h3_graduated_style(lyr, str(fld), theme=self.model.theme)
        self._refresh_canvas()

    def on_export_csv(self) -> None:
        lyr = self.active_h3_layer()
        if lyr is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{lyr.name()}.csv", "CSV (*.csv)")
        if path:
            n = export_h3_table_csv(lyr, path)
            self.log(f"[Export] {n:,} hexagons → {path}")

    def on_export_html_report(self) -> None:
        lyr = self.active_h3_layer()
        if lyr is None or lyr.fields().indexFromName("comp_score") < 0:
            self._warn("Run the model before exporting a report.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save HTML report", f"{lyr.name()}_report.html", "HTML (*.html)")
        if not path:
            return
        run = read_layer_json_property(lyr, "h3forge/last_run")
        try:
            export_h3_html_report(lyr, path, weights_map=dict(run.get("weights", {})),
                                  compose_method=str(run.get("method", "wlc")),
                                  theme=str(run.get("theme", "risk")))
        except Exception as exc:
            self._error(f"Report export failed:\n{exc}")
            return
        self.log(f"[Export] Report → {path}")
        if not self._headless():
            webbrowser.open(Path(path).resolve().as_uri())


__all__ = ["O2H3ForgeStudioDock"]
