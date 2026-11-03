# -*- coding: utf-8 -*-
"""Reusable widgets for the 02H3Forge studio: coupled weights and value-function preview."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from qgis.PyQt.QtCore import QPointF, QRectF, QSignalBlocker, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QWidget,
)

from ..core.mcda_composer import rebalance_coupled_weights
from ..core.model import ValueFunction
from ..core.pipeline import apply_value_function


class CoupledWeightsWidget(QWidget):
    """Rows of slider + spinbox + lock whose values always sum to 100 %."""

    weightsChanged = pyqtSignal(list)  # noqa: N815 (Qt naming)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(2)
        self._rows: List[dict] = []

    # -- public API --------------------------------------------------------
    def set_items(self, labels: Sequence[str], weights: Sequence[float],
                  locked: Optional[Sequence[bool]] = None, tooltips: Optional[Sequence[str]] = None) -> None:
        while self._grid.count():
            w = self._grid.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._rows = []
        locked = list(locked) if locked is not None else [False] * len(labels)
        for i, lbl in enumerate(labels):
            name = QLabel(str(lbl))
            name.setMinimumWidth(90)
            if tooltips is not None:
                name.setToolTip(str(tooltips[i]))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSuffix(" %")
            spin.setSingleStep(1.0)
            lock = QCheckBox("🔒")
            lock.setToolTip("Keep this weight fixed while the others rebalance")
            lock.setChecked(bool(locked[i]))
            slider.valueChanged.connect(lambda v, k=i: self._on_edit(k, v / 10.0))
            spin.valueChanged.connect(lambda v, k=i: self._on_edit(k, float(v)))
            lock.toggled.connect(lambda _c: self.weightsChanged.emit(self.weights()))
            self._grid.addWidget(name, i, 0)
            self._grid.addWidget(slider, i, 1)
            self._grid.addWidget(spin, i, 2)
            self._grid.addWidget(lock, i, 3)
            self._rows.append({"slider": slider, "spin": spin, "lock": lock})
        self._grid.setColumnStretch(1, 1)
        self._show(list(weights))

    def weights(self) -> List[float]:
        return [float(r["spin"].value()) for r in self._rows]

    def locked(self) -> List[bool]:
        return [bool(r["lock"].isChecked()) for r in self._rows]

    def set_weights(self, weights: Sequence[float], emit: bool = True) -> None:
        for r in self._rows:
            with QSignalBlocker(r["lock"]):
                r["lock"].setChecked(False)
        self._show(list(weights))
        if emit:
            self.weightsChanged.emit(self.weights())

    def count(self) -> int:
        return len(self._rows)

    # -- internals -----------------------------------------------------------
    def _show(self, weights: List[float]) -> None:
        for r, w in zip(self._rows, weights):
            with QSignalBlocker(r["slider"]), QSignalBlocker(r["spin"]):
                r["spin"].setValue(float(w))
                r["slider"].setValue(int(round(float(w) * 10.0)))

    def _on_edit(self, index: int, value: float) -> None:
        current = self.weights()
        locks = self.locked()
        if locks[index]:
            self._show(current)
            return
        balanced = rebalance_coupled_weights(current, index, value, locked_flags=locks)
        self._show(balanced)
        self.weightsChanged.emit(self.weights())


class ValueFunctionPreview(QWidget):
    """Draws a value function (raw → 0-100) over a data-driven domain.

    When raw values are available (from a previous run) their histogram is
    shown behind the curve, so analysts see where their hexagons fall.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._vf: Optional[ValueFunction] = None
        self._raw: Optional[np.ndarray] = None
        self._unit = ""

    def set_function(self, vf: ValueFunction, raw: Optional[np.ndarray] = None, unit: str = "") -> None:
        self._vf = vf
        self._raw = None
        if raw is not None:
            r = np.asarray(raw, dtype=np.float64)
            r = r[np.isfinite(r)]
            self._raw = r if r.size else None
        self._unit = unit
        self.update()

    def domain(self) -> tuple:
        vf = self._vf
        if self._raw is not None and vf is not None and vf.method not in ("rings",):
            lo, hi = float(np.min(self._raw)), float(np.max(self._raw))
        elif vf is not None and vf.method == "rings" and vf.rings:
            lo, hi = 0.0, max(float(r[1]) for r in vf.rings) * 1.3
            if self._raw is not None:
                hi = max(hi, float(np.percentile(self._raw, 95)))
        elif vf is not None and vf.method == "fuzzy_linear" and vf.low is not None and vf.high is not None:
            span = float(vf.high) - float(vf.low)
            lo, hi = float(vf.low) - 0.25 * span, float(vf.high) + 0.25 * span
        elif vf is not None and vf.method in ("fuzzy_sigmoid", "fuzzy_gaussian") and vf.mid is not None:
            sp = float(vf.spread or 1.0)
            lo, hi = float(vf.mid) - 4 * sp, float(vf.mid) + 4 * sp
        else:
            lo, hi = 0.0, 100.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def paintEvent(self, _event):  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(34, 8, -8, -22)
        p.fillRect(self.rect(), QColor("#ffffff"))
        p.setPen(QPen(QColor("#cbd5e1"), 1))
        p.drawRect(rect)
        if self._vf is None:
            p.end()
            return
        lo, hi = self.domain()

        def X(v: float) -> float:
            return rect.left() + (v - lo) / (hi - lo) * rect.width()

        def Y(s: float) -> float:
            return rect.bottom() - s / 100.0 * rect.height()

        # histogram of observed raw values
        if self._raw is not None:
            bins = 30
            clipped = np.clip(self._raw, lo, hi)
            hist, edges = np.histogram(clipped, bins=bins, range=(lo, hi))
            peak = max(1, int(hist.max()))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(19, 160, 160, 45))
            for k, cnt in enumerate(hist):
                h = rect.height() * 0.9 * cnt / peak
                p.drawRect(QRectF(X(edges[k]), rect.bottom() - h, X(edges[k + 1]) - X(edges[k]), h))

        # grid + labels
        p.setPen(QPen(QColor("#e2e8f0"), 1, Qt.PenStyle.DashLine))
        for s in (25, 50, 75):
            p.drawLine(QPointF(rect.left(), Y(s)), QPointF(rect.right(), Y(s)))
        p.setPen(QColor("#64748b"))
        f = p.font()
        f.setPointSizeF(max(6.5, f.pointSizeF() - 1.5))
        p.setFont(f)
        p.drawText(QRectF(0, Y(100) - 7, 30, 14), Qt.AlignmentFlag.AlignRight, "100")
        p.drawText(QRectF(0, Y(0) - 7, 30, 14), Qt.AlignmentFlag.AlignRight, "0")
        for frac in (0.0, 0.5, 1.0):
            v = lo + frac * (hi - lo)
            txt = f"{v:,.4g}{(' ' + self._unit) if frac == 1.0 and self._unit else ''}"
            p.drawText(QRectF(X(v) - 45, rect.bottom() + 3, 90, 16), Qt.AlignmentFlag.AlignHCenter, txt)

        # curve
        xs = np.linspace(lo, hi, 240)
        if self._vf.method in ("minmax", "percentile") and self._raw is None:
            # data-driven: draw the generic linear ramp across the (unknown) observed range
            ys = np.linspace(0, 100, xs.size) if self._vf.polarity == "benefit" else np.linspace(100, 0, xs.size)
        else:
            sample = xs if self._raw is None or self._vf.method not in ("minmax", "percentile") else None
            if sample is None:
                # minmax/percentile depend on the observed data: evaluate on data + grid
                data = np.concatenate([self._raw, xs])
                ys = apply_value_function(data, self._vf)[self._raw.size:]
            else:
                ys = apply_value_function(sample, self._vf)
        path = QPainterPath()
        for k, (xv, yv) in enumerate(zip(xs, ys)):
            pt = QPointF(X(float(xv)), Y(float(np.clip(yv, 0, 100))))
            if k == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#0f6b6b"), 2.2))
        p.drawPath(path)
        p.end()


def format_pct(v: float) -> str:
    return f"{v:.1f} %" if math.isfinite(v) else "–"
