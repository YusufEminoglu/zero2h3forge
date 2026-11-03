# -*- coding: utf-8 -*-
"""Interactive Analytic Hierarchy Process (AHP) Pairwise Comparison Dialog for 02H3Forge."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from qgis.PyQt.QtCore import QSignalBlocker, Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .core.mcda_composer import ahp_weights_from_matrix, equal_weights_pct


SAATY_CHOICES = [
    ("9 — Extreme preference", 9.0),
    ("7 — Very strong preference", 7.0),
    ("5 — Strong preference", 5.0),
    ("3 — Moderate preference", 3.0),
    ("2 — Slight preference", 2.0),
    ("1 — Equal importance", 1.0),
    ("1/2 — Slight inverse (0.50)", 0.5),
    ("1/3 — Moderate inverse (0.33)", 1.0 / 3.0),
    ("1/5 — Strong inverse (0.20)", 0.2),
    ("1/7 — Very strong inverse (0.14)", 1.0 / 7.0),
    ("1/9 — Extreme inverse (0.11)", 1.0 / 9.0),
]


class AhpPairwiseDialog(QDialog):
    """Compact pairwise comparison matrix dialog with live CR verification."""

    def __init__(self, criterion_names: Sequence[str], parent=None):
        super().__init__(parent)
        self.criterion_names = list(criterion_names)
        self.n = len(self.criterion_names)
        self.matrix = np.eye(max(1, self.n), dtype=np.float64)
        self.computed_weights: List[float] = equal_weights_pct(self.n)
        self.cr_value: float = 0.0

        self.setWindowTitle("02H3Forge — AHP Pairwise Comparison Matrix")
        self.resize(680, 440)
        self._build_ui()
        self._recompute()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "<b>Analytic Hierarchy Process (Saaty Pairwise Matrix)</b><br>"
            "Adjust upper-triangle cells (row vs. column importance). "
            "Reciprocal lower-triangle values update automatically."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.table = QTableWidget(self.n, self.n)
        short_labels = [c.replace("norm_", "") for c in self.criterion_names]
        self.table.setHorizontalHeaderLabels(short_labels)
        self.table.setVerticalHeaderLabels(short_labels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    item = QTableWidgetItem("1.00")
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                    self.table.setItem(i, j, item)
                elif i < j:
                    cb = QComboBox()
                    for lbl, val in SAATY_CHOICES:
                        cb.addItem(lbl, val)
                    cb.setCurrentIndex(5)  # 1.0 Equal
                    cb.currentIndexChanged.connect(
                        lambda _idx, r=i, c=j: self._on_upper_changed(r, c)
                    )
                    self.table.setCellWidget(i, j, cb)
                else:
                    item = QTableWidgetItem("1.00")
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                    self.table.setItem(i, j, item)

        layout.addWidget(self.table, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset All to 1 (Equal)")
        reset_btn.clicked.connect(self._reset_equal)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.apply_btn = QPushButton("Apply AHP Weights to Sliders")
        self.apply_btn.setStyleSheet(
            "background:#13a0a0; color:white; font-weight:bold; padding:6px 12px; border-radius:4px;"
        )
        self.apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.apply_btn)
        layout.addLayout(btn_row)

    def _on_upper_changed(self, row: int, col: int) -> None:
        cb = self.table.cellWidget(row, col)
        if not isinstance(cb, QComboBox):
            return
        val = float(cb.currentData())
        self.matrix[row, col] = val
        self.matrix[col, row] = 1.0 / val
        lower_item = self.table.item(col, row)
        if lower_item is not None:
            lower_item.setText(f"{1.0 / val:.2f}")
        self._recompute()

    def _reset_equal(self) -> None:
        self.matrix = np.eye(max(1, self.n), dtype=np.float64)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cb = self.table.cellWidget(i, j)
                if isinstance(cb, QComboBox):
                    with QSignalBlocker(cb):
                        cb.setCurrentIndex(5)
                lower_item = self.table.item(j, i)
                if lower_item is not None:
                    lower_item.setText("1.00")
        self._recompute()

    def _recompute(self) -> None:
        if self.n <= 0:
            return
        w_pct, lam, ci, cr = ahp_weights_from_matrix(self.matrix)
        self.computed_weights = w_pct
        self.cr_value = cr
        ok = cr <= 0.10
        badge = "PASS (CR <= 0.10)" if ok else "WARNING: Inconsistent (CR > 0.10)"
        color = "#0f6b6b" if ok else "#b30000"
        preview = ", ".join(
            f"{self.criterion_names[k].replace('norm_', '')}: {w_pct[k]:.1f}%"
            for k in range(self.n)
        )
        self.status_label.setText(
            f"<span style='color:{color};font-weight:bold;'>"
            f"λ_max = {lam:.3f} | CI = {ci:.3f} | CR = {cr:.3f} — {badge}</span><br>"
            f"<b>Weights:</b> {preview}"
        )
