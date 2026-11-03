# -*- coding: utf-8 -*-
"""Criterion / constraint editors for the 02H3Forge studio."""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
from qgis.core import QgsMapLayer, QgsProject, QgsRasterLayer, QgsVectorLayer, QgsWkbTypes
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.model import (
    CONSTRAINT_KINDS,
    RING_PRESETS,
    RULE_OPERATORS,
    SOURCE_KINDS,
    VALUE_METHODS,
    ConstraintSpec,
    CriterionSpec,
    ModelSpec,
    SourceSpec,
    ValueFunction,
)
from ..core.normalizer import POLARITY_OPTIONS
from ..core.spatial_aggregator import RASTER_AGG_METHODS, VECTOR_AGG_METHODS
from .widgets import ValueFunctionPreview


def _layers(kind: str, polygons_only: bool = False) -> List[QgsMapLayer]:
    out = []
    for lyr in QgsProject.instance().mapLayers().values():
        if not lyr.isValid():
            continue
        if kind == "raster" and isinstance(lyr, QgsRasterLayer):
            out.append(lyr)
        elif kind == "vector" and isinstance(lyr, QgsVectorLayer) and lyr.isSpatial():
            if lyr.fields().indexFromName("h3_id") >= 0:
                continue  # never aggregate an H3 grid into itself
            if polygons_only and lyr.geometryType() != QgsWkbTypes.GeometryType.PolygonGeometry:
                continue
            out.append(lyr)
    return sorted(out, key=lambda lyr_item: lyr_item.name().lower())


def _fill_layers(combo: QComboBox, layers: List[QgsMapLayer], current_id: str = "", current_name: str = ""):
    combo.clear()
    for lyr in layers:
        combo.addItem(lyr.name(), lyr.id())
    idx = combo.findData(current_id) if current_id else -1
    if idx < 0 and current_name:
        idx = combo.findText(current_name)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _source_from_combo(combo: QComboBox, kind: str) -> SourceSpec:
    lid = combo.currentData()
    lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
    if lyr is None:
        return SourceSpec(kind=kind)
    return SourceSpec(
        kind=kind, layer_id=lyr.id(), layer_name=lyr.name(),
        layer_uri=lyr.source(), provider=lyr.providerType(),
    )


def _opt_float(text: str) -> Optional[float]:
    t = (text or "").strip().replace(",", ".")
    if not t:
        return None
    return float(t)


def _combo(items) -> QComboBox:
    cb = QComboBox()
    for key, lbl in items:
        cb.addItem(lbl, key)
    return cb


def _select(combo: QComboBox, data) -> None:
    i = combo.findData(data)
    if i >= 0:
        combo.setCurrentIndex(i)


class CriterionDialog(QDialog):
    """Create or edit one criterion: source → raw value → value function → group."""

    def __init__(self, model: ModelSpec, h3_layer: Optional[QgsVectorLayer],
                 criterion: Optional[CriterionSpec] = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.h3_layer = h3_layer
        self.original = criterion
        self.result_spec: Optional[CriterionSpec] = None
        self.setWindowTitle("02H3Forge — Edit criterion" if criterion else "02H3Forge — New criterion")
        self.resize(560, 720)
        c = copy.deepcopy(criterion) if criterion else CriterionSpec(
            id="", label="", group=model.groups[0].id if model.groups else "index"
        )
        self._build(c)
        self._load(c)

    # ------------------------------------------------------------------ UI
    def _build(self, c: CriterionSpec) -> None:
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("e.g. Distance to power lines")
        form.addRow("Name:", self.label_edit)
        self.group_combo = QComboBox()
        for g in self.model.groups:
            self.group_combo.addItem(g.label, g.id)
        form.addRow("Group:", self.group_combo)
        lay.addLayout(form)

        # --- source -------------------------------------------------------
        box_src = QGroupBox("1 · Source of the raw value")
        v_src = QVBoxLayout(box_src)
        self.kind_combo = _combo(SOURCE_KINDS)
        v_src.addWidget(self.kind_combo)
        self.src_stack = QStackedWidget()

        w_vec = QWidget()
        f_vec = QFormLayout(w_vec)
        self.vec_layer = QComboBox()
        self.vec_method = _combo(VECTOR_AGG_METHODS)
        self.vec_field = QComboBox()
        f_vec.addRow("Layer:", self.vec_layer)
        f_vec.addRow("Metric:", self.vec_method)
        f_vec.addRow("Value field:", self.vec_field)
        self.vec_layer.currentIndexChanged.connect(self._fill_vec_fields)

        w_ras = QWidget()
        f_ras = QFormLayout(w_ras)
        self.ras_layer = QComboBox()
        self.ras_band = QSpinBox()
        self.ras_band.setRange(1, 64)
        self.ras_method = _combo(RASTER_AGG_METHODS)
        f_ras.addRow("Raster:", self.ras_layer)
        f_ras.addRow("Band:", self.ras_band)
        f_ras.addRow("Zonal statistic:", self.ras_method)

        w_dist = QWidget()
        f_dist = QFormLayout(w_dist)
        self.dist_layer = QComboBox()
        f_dist.addRow("Features:", self.dist_layer)
        f_dist.addRow(QLabel("<i>Shortest distance (m) from each hexagon edge to the nearest feature.</i>"))

        w_fld = QWidget()
        f_fld = QFormLayout(w_fld)
        self.h3_field = QComboBox()
        f_fld.addRow("H3 field:", self.h3_field)

        for w in (w_vec, w_ras, w_dist, w_fld):
            self.src_stack.addWidget(w)
        v_src.addWidget(self.src_stack)
        lay.addWidget(box_src)

        # --- value function ------------------------------------------------
        box_vf = QGroupBox("2 · Value function (raw → 0–100 score)")
        v_vf = QVBoxLayout(box_vf)
        f_vf = QFormLayout()
        self.vf_method = _combo(VALUE_METHODS)
        self.vf_polarity = _combo(POLARITY_OPTIONS)
        f_vf.addRow("Function:", self.vf_method)
        f_vf.addRow("Direction:", self.vf_polarity)
        self.p_low, self.p_high, self.p_mid, self.p_spread = (QLineEdit() for _ in range(4))
        for w, ph in ((self.p_low, "low (blank = data min)"), (self.p_high, "high (blank = data max)"),
                      (self.p_mid, "midpoint / optimum (blank = median)"),
                      (self.p_spread, "spread > 0 (blank = std dev)")):
            w.setPlaceholderText(ph)
            w.textChanged.connect(self._refresh_preview)
        f_vf.addRow("Low:", self.p_low)
        f_vf.addRow("High:", self.p_high)
        f_vf.addRow("Mid / optimum:", self.p_mid)
        f_vf.addRow("Spread:", self.p_spread)
        self._param_form = f_vf
        v_vf.addLayout(f_vf)

        self.rings_box = QWidget()
        v_r = QVBoxLayout(self.rings_box)
        v_r.setContentsMargins(0, 0, 0, 0)
        pr = QHBoxLayout()
        self.ring_preset = QComboBox()
        self.ring_preset.addItem("Load preset…", "")
        for name in RING_PRESETS:
            self.ring_preset.addItem(name, name)
        self.ring_preset.currentIndexChanged.connect(self._load_preset)
        pr.addWidget(self.ring_preset, 1)
        add_b = QPushButton("+")
        add_b.setFixedWidth(28)
        add_b.clicked.connect(lambda: self._add_ring(0, 100, 50))
        rem_b = QPushButton("−")
        rem_b.setFixedWidth(28)
        rem_b.clicked.connect(self._remove_ring)
        pr.addWidget(add_b)
        pr.addWidget(rem_b)
        v_r.addLayout(pr)
        self.ring_table = QTableWidget(0, 3)
        self.ring_table.setHorizontalHeaderLabels(["From (m)", "To (m)", "Score"])
        self.ring_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ring_table.setMaximumHeight(130)
        self.ring_table.itemChanged.connect(self._refresh_preview)
        v_r.addWidget(self.ring_table)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Score beyond the last ring:"))
        self.outside_spin = QDoubleSpinBox()
        self.outside_spin.setRange(0, 100)
        self.outside_spin.valueChanged.connect(self._refresh_preview)
        orow.addWidget(self.outside_spin)
        v_r.addLayout(orow)
        v_vf.addWidget(self.rings_box)

        self.preview = ValueFunctionPreview()
        v_vf.addWidget(self.preview)
        self.preview_note = QLabel("")
        self.preview_note.setWordWrap(True)
        self.preview_note.setStyleSheet("color:#64748b;font-size:11px;")
        v_vf.addWidget(self.preview_note)
        lay.addWidget(box_vf, 1)

        self.active_chk = QCheckBox("Active (included in the model)")
        lay.addWidget(self.active_chk)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.kind_combo.currentIndexChanged.connect(self._on_kind)
        self.vf_method.currentIndexChanged.connect(self._on_vf_method)
        self.vf_polarity.currentIndexChanged.connect(self._refresh_preview)

    # --------------------------------------------------------------- load
    def _load(self, c: CriterionSpec) -> None:
        self.label_edit.setText(c.label)
        _select(self.group_combo, c.group)
        _select(self.kind_combo, c.source.kind if self.original is not None else "vector")
        _fill_layers(self.vec_layer, _layers("vector"), c.source.layer_id, c.source.layer_name)
        _fill_layers(self.dist_layer, _layers("vector"), c.source.layer_id, c.source.layer_name)
        _fill_layers(self.ras_layer, _layers("raster"), c.source.layer_id, c.source.layer_name)
        _select(self.vec_method, c.source.method)
        _select(self.ras_method, c.source.method)
        self.ras_band.setValue(int(c.source.band or 1))
        self.h3_field.clear()
        if self.h3_layer is not None:
            for f in self.h3_layer.fields():
                if f.isNumeric() and f.name() not in ("h3_res", "constraint_mask"):
                    self.h3_field.addItem(f.name(), f.name())
        _select(self.h3_field, c.source.field)
        self._fill_vec_fields()
        _select(self.vec_field, c.source.field)

        v = c.value
        _select(self.vf_method, v.method)
        _select(self.vf_polarity, v.polarity)
        for w, val in ((self.p_low, v.low), (self.p_high, v.high), (self.p_mid, v.mid), (self.p_spread, v.spread)):
            w.setText("" if val is None else f"{val:g}")
        self.ring_table.setRowCount(0)
        for r in v.rings:
            self._add_ring(*r[:3])
        self.outside_spin.setValue(float(v.outside_score))
        self.active_chk.setChecked(bool(c.active))
        self._on_kind(initial=self.original is not None)
        self._on_vf_method()

    def _fill_vec_fields(self) -> None:
        self.vec_field.clear()
        self.vec_field.addItem("(none — geometry only)", "")
        lid = self.vec_layer.currentData()
        lyr = QgsProject.instance().mapLayer(str(lid)) if lid else None
        if isinstance(lyr, QgsVectorLayer):
            for f in lyr.fields():
                if f.isNumeric():
                    self.vec_field.addItem(f.name(), f.name())

    # ------------------------------------------------------------ behaviour
    def _kind(self) -> str:
        return str(self.kind_combo.currentData())

    def _on_kind(self, *_args, initial: bool = False) -> None:
        kind = self._kind()
        self.src_stack.setCurrentIndex(["vector", "raster", "distance", "field"].index(kind))
        if not initial:
            # sensible default value function per source kind
            if kind == "distance" and self.vf_method.currentData() != "rings":
                _select(self.vf_method, "rings")
                if self.ring_table.rowCount() == 0:
                    self.ring_preset.setCurrentIndex(1)
            elif kind == "field":
                fld = str(self.h3_field.currentData() or "")
                _select(self.vf_method, "direct_clip" if fld.startswith(("norm_", "ring_")) else "minmax")
            elif self.vf_method.currentData() == "rings":
                _select(self.vf_method, "minmax")
        self._refresh_preview()

    def _on_vf_method(self, *_args) -> None:
        m = str(self.vf_method.currentData())
        is_rings = m == "rings"
        self.rings_box.setVisible(is_rings)
        uses_lh = m in ("minmax", "fuzzy_linear")
        uses_ms = m in ("fuzzy_sigmoid", "fuzzy_gaussian")
        for w, on in ((self.p_low, uses_lh), (self.p_high, uses_lh), (self.p_mid, uses_ms), (self.p_spread, uses_ms)):
            w.setEnabled(on)
            label = self._param_form.labelForField(w)
            w.setVisible(on)
            if label is not None:
                label.setVisible(on)
        self.vf_polarity.setEnabled(not is_rings)
        self._refresh_preview()

    def _add_ring(self, lo, hi, sc) -> None:
        r = self.ring_table.rowCount()
        self.ring_table.insertRow(r)
        for col, val in enumerate((lo, hi, sc)):
            self.ring_table.setItem(r, col, QTableWidgetItem(f"{float(val):g}"))

    def _remove_ring(self) -> None:
        r = self.ring_table.currentRow()
        if r < 0:
            r = self.ring_table.rowCount() - 1
        if r >= 0:
            self.ring_table.removeRow(r)
        self._refresh_preview()

    def _load_preset(self) -> None:
        key = self.ring_preset.currentData()
        if not key:
            return
        rings, outside = RING_PRESETS[key]
        self.ring_table.blockSignals(True)
        self.ring_table.setRowCount(0)
        for r in rings:
            self._add_ring(*r)
        self.ring_table.blockSignals(False)
        self.outside_spin.setValue(outside)
        if not self.label_edit.text().strip():
            self.label_edit.setText(key.split(" (")[0])
        self._refresh_preview()

    def _rings(self) -> List[List[float]]:
        out = []
        for r in range(self.ring_table.rowCount()):
            vals = []
            for col in range(3):
                it = self.ring_table.item(r, col)
                vals.append(float(it.text().replace(",", ".")) if it and it.text().strip() else 0.0)
            out.append(vals)
        return out

    def _value_function(self) -> ValueFunction:
        return ValueFunction(
            method=str(self.vf_method.currentData()),
            polarity=str(self.vf_polarity.currentData()),
            low=_opt_float(self.p_low.text()), high=_opt_float(self.p_high.text()),
            mid=_opt_float(self.p_mid.text()), spread=_opt_float(self.p_spread.text()),
            rings=self._rings(), outside_score=float(self.outside_spin.value()),
        )

    def _existing_raw(self) -> Optional[np.ndarray]:
        """Raw values from a previous run (or the H3 field), for the histogram."""
        if self.h3_layer is None:
            return None
        if self._kind() == "field":
            name = str(self.h3_field.currentData() or "")
        elif self.original is not None:
            name = self.original.raw_field
        else:
            return None
        idx = self.h3_layer.fields().indexFromName(name)
        if idx < 0:
            return None
        vals = []
        for f in self.h3_layer.getFeatures():
            v = f.attribute(idx)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        arr = np.asarray(vals, dtype=np.float64)
        return arr[arr >= 0] if self._kind() == "distance" else arr

    def _refresh_preview(self, *_args) -> None:
        try:
            vf = self._value_function()
        except ValueError:
            self.preview_note.setText("<span style='color:#b30000'>Parameters must be numbers.</span>")
            return
        raw = self._existing_raw()
        unit = "m" if self._kind() == "distance" else ""
        self.preview.set_function(vf, raw, unit)
        if raw is None and vf.method in ("minmax", "percentile"):
            self.preview_note.setText("Range is taken from the data when the model runs. "
                                      "Run once to see your hexagons' distribution here.")
        elif raw is None:
            self.preview_note.setText("Run the model once to overlay the distribution of hexagon values.")
        else:
            self.preview_note.setText(f"Shaded: distribution of {raw.size:,} hexagon values from the last run.")

    # --------------------------------------------------------------- accept
    def _accept(self) -> None:
        kind = self._kind()
        if kind == "vector":
            src = _source_from_combo(self.vec_layer, "vector")
            src.method = str(self.vec_method.currentData())
            src.field = str(self.vec_field.currentData() or "")
            needs_field = src.method not in ("count", "density_ha", "area_share_pct", "length_m", "nearest_dist_m")
            if needs_field and not src.field:
                QMessageBox.warning(self, "02H3Forge", "This metric needs a numeric value field.")
                return
        elif kind == "raster":
            src = _source_from_combo(self.ras_layer, "raster")
            src.method = str(self.ras_method.currentData())
            src.band = int(self.ras_band.value())
        elif kind == "distance":
            src = _source_from_combo(self.dist_layer, "distance")
        else:
            src = SourceSpec(kind="field", field=str(self.h3_field.currentData() or ""))
        try:
            vf = self._value_function()
        except ValueError:
            QMessageBox.warning(self, "02H3Forge", "Value-function parameters must be numbers.")
            return
        label = self.label_edit.text().strip() or src.layer_name or src.field or "criterion"
        if self.original is not None:
            cid = self.original.id
            weight, locked = self.original.weight, self.original.locked
        else:
            cid = self.model.unique_id(label)
            weight, locked = 0.0, False
        spec = CriterionSpec(
            id=cid, label=label, group=str(self.group_combo.currentData()), source=src, value=vf,
            weight=weight, active=self.active_chk.isChecked(), locked=locked,
        )
        probe = self.model.copy()
        probe.criteria = [spec]
        problems = [p for p in probe.validate() if "at least one active" not in p]
        if problems:
            QMessageBox.warning(self, "02H3Forge", "\n".join(problems))
            return
        self.result_spec = spec
        self.accept()


class ConstraintDialog(QDialog):
    """Rule-based hard exclusion editor."""

    def __init__(self, model: ModelSpec, h3_layer: Optional[QgsVectorLayer],
                 constraint: Optional[ConstraintSpec] = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.original = constraint
        self.result_spec: Optional[ConstraintSpec] = None
        self.setWindowTitle("02H3Forge — Constraint")
        self.resize(460, 280)
        k = copy.deepcopy(constraint) if constraint else ConstraintSpec(id="", label="")

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.label_edit = QLineEdit(k.label)
        self.label_edit.setPlaceholderText("e.g. Power line safety corridor")
        form.addRow("Name:", self.label_edit)
        self.kind_combo = _combo(CONSTRAINT_KINDS)
        _select(self.kind_combo, k.kind)
        form.addRow("Rule:", self.kind_combo)
        self.layer_combo = QComboBox()
        form.addRow("Layer:", self.layer_combo)
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0.0, 1e6)
        self.dist_spin.setSuffix(" m")
        self.dist_spin.setValue(float(k.distance or 50.0))
        form.addRow("Distance:", self.dist_spin)
        rule_row = QHBoxLayout()
        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        names = set()
        if h3_layer is not None:
            names |= {f.name() for f in h3_layer.fields() if f.isNumeric()}
        for c in model.criteria:
            names |= {c.raw_field, c.norm_field}
        for n in sorted(names):
            self.field_combo.addItem(n)
        self.field_combo.setCurrentText(k.field)
        self.op_combo = QComboBox()
        for op in RULE_OPERATORS:
            self.op_combo.addItem(op)
        self.op_combo.setCurrentText(k.op)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-1e12, 1e12)
        self.value_spin.setDecimals(3)
        self.value_spin.setValue(float(k.value))
        rule_row.addWidget(self.field_combo, 2)
        rule_row.addWidget(self.op_combo)
        rule_row.addWidget(self.value_spin, 1)
        self.rule_widget = QWidget()
        self.rule_widget.setLayout(rule_row)
        form.addRow("Attribute rule:", self.rule_widget)
        lay.addLayout(form)
        hint = QLabel("<i>Excluded hexagons get <code>constraint_mask = 0</code> and a composite score of 0. "
                      "Attribute rules can use any criterion column, e.g. <code>raw_slope &gt; 30</code>.</i>")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._form = form
        self._init_source = k.source
        self.kind_combo.currentIndexChanged.connect(self._on_kind)
        self._on_kind()

    def _on_kind(self) -> None:
        kind = str(self.kind_combo.currentData())
        _fill_layers(self.layer_combo, _layers("vector", polygons_only=kind in ("inside", "outside")),
                     self._init_source.layer_id, self._init_source.layer_name)
        for w, on in ((self.layer_combo, kind != "rule"), (self.dist_spin, kind == "near"),
                      (self.rule_widget, kind == "rule")):
            w.setVisible(on)
            lbl = self._form.labelForField(w)
            if lbl is not None:
                lbl.setVisible(on)

    def _accept(self) -> None:
        kind = str(self.kind_combo.currentData())
        src = _source_from_combo(self.layer_combo, "vector") if kind != "rule" else SourceSpec()
        cid = self.original.id if self.original else self.model.unique_id(self.label_edit.text() or kind)
        spec = ConstraintSpec(
            id=cid, label=self.label_edit.text().strip() or kind, kind=kind, source=src,
            distance=float(self.dist_spin.value()), field=self.field_combo.currentText().strip(),
            op=self.op_combo.currentText(), value=float(self.value_spin.value()),
            active=self.original.active if self.original else True,
        )
        probe = self.model.copy()
        probe.constraints = [spec]
        problems = [p for p in probe.validate() if p.startswith("Constraint")]
        if problems:
            QMessageBox.warning(self, "02H3Forge", "\n".join(problems))
            return
        self.result_spec = spec
        self.accept()


class FieldImportDialog(QDialog):
    """Pick existing numeric H3 fields to register as criteria."""

    def __init__(self, h3_layer: QgsVectorLayer, exclude: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("02H3Forge — Use existing H3 fields as criteria")
        self.resize(380, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Tick the fields to add. <code>norm_*</code> / <code>ring_*</code> fields are "
                             "used as ready 0–100 scores; others get a linear value function."))
        self.list = QListWidget()
        skip = set(exclude) | {"h3_res", "area_m2", "area_ha", "constraint_mask", "comp_score", "class_id",
                               "driver_share", "mc_std", "rob_score", "gi_z", "gi_p", "hotspot"}
        for f in h3_layer.fields():
            if not f.isNumeric() or f.name() in skip or f.name().startswith("idx_"):
                continue
            it = QListWidgetItem(f.name())
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if f.name().startswith("norm_") else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        lay.addWidget(self.list, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selected(self) -> List[str]:
        return [
            self.list.item(i).text() for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        ]
