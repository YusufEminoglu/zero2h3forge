# -*- coding: utf-8 -*-
"""02H3Forge - H3 Hexagonal Risk, Vulnerability and Suitability Studio: plugin shell."""
from __future__ import annotations

import os
from contextlib import suppress

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QIcon

try:  # Qt5: QtWidgets / Qt6: QtGui
    from qgis.PyQt.QtWidgets import QAction
except ImportError:  # pragma: no cover - QGIS 4 / Qt6
    from qgis.PyQt.QtGui import QAction

PLUGIN_DIR = os.path.dirname(__file__)
MENU_NAME = "&02H3Forge"


class O2H3ForgePlugin:
    """QGIS plugin entry: toolbar button + Right-Docked H3 Studio panel."""

    def __init__(self, iface):
        self.iface = iface
        self.dock = None
        self.actions = []

    def initGui(self) -> None:  # noqa: N802 (QGIS API)
        icon = QIcon(os.path.join(PLUGIN_DIR, "icons", "icon.png"))
        action = QAction(icon, "02H3Forge Studio", self.iface.mainWindow())
        action.setToolTip(
            "Open the 02H3Forge H3 Hexagonal Risk, Vulnerability and Suitability Studio"
        )
        action.triggered.connect(self.open_dock)
        self.iface.addToolBarIcon(action)
        self.iface.addPluginToMenu(MENU_NAME, action)
        self.actions.append(action)

        rate_action = QAction("★ Rate 02H3Forge on GeoPhilo...", self.iface.mainWindow())
        rate_action.setToolTip("Submit verified rating & feedback on GeoPhilo")
        rate_action.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://geophilo.com/feedback/?plugin=zero2h3forge&v=0.1.1")
            )
        )
        self.iface.addPluginToMenu(MENU_NAME, rate_action)
        self.actions.append(rate_action)

    def unload(self) -> None:
        for action in self.actions:
            self.iface.removeToolBarIcon(action)
            self.iface.removePluginMenu(MENU_NAME, action)
            action.setParent(None)
            action.deleteLater()
        self.actions = []

        if self.dock is not None:
            with suppress(Exception):
                self.dock.dispose()
            self.iface.removeDockWidget(self.dock)
            self.dock.setParent(None)
            self.dock.deleteLater()
            self.dock = None

    def open_dock(self) -> None:
        if self.dock is None:
            try:
                from .studio_dock import O2H3ForgeStudioDock

                self.dock = O2H3ForgeStudioDock(self.iface, self.iface.mainWindow())
                self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
            except Exception as exc:
                self.iface.messageBar().pushWarning(
                    "02H3Forge", f"Studio panel unavailable: {exc}"
                )
                return
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            self.dock.show()
            self.dock.raise_()
            self.dock.activateWindow()
