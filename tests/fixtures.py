# -*- coding: utf-8 -*-
"""Shared synthetic QGIS fixtures for the 02H3Forge headless tests (İzmir, EPSG:5253)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
from osgeo import gdal, osr
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.gui import QgsMapCanvas
from qgis.PyQt.QtWidgets import QMainWindow, QToolBar

gdal.UseExceptions()


class DummyMessageBar:
    def pushWarning(self, title: str, msg: str) -> None:  # noqa: N802
        pass

    def pushInfo(self, title: str, msg: str) -> None:  # noqa: N802
        pass


class DummyIface:
    def __init__(self):
        self._win = QMainWindow()
        self._canvas = QgsMapCanvas(self._win)
        self._canvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:5253"))
        self._bar = DummyMessageBar()
        self._toolbar = QToolBar("DummyToolbar", self._win)
        self.added_docks = []
        self._active_layer = None

    def mainWindow(self):  # noqa: N802
        return self._win

    def mapCanvas(self):  # noqa: N802
        return self._canvas

    def messageBar(self):  # noqa: N802
        return self._bar

    def addToolBarIcon(self, action):  # noqa: N802
        self._toolbar.addAction(action)

    def removeToolBarIcon(self, action):  # noqa: N802
        self._toolbar.removeAction(action)

    def addPluginToMenu(self, menu_name, action):  # noqa: N802
        pass

    def removePluginMenu(self, menu_name, action):  # noqa: N802
        pass

    def addDockWidget(self, area, dock):  # noqa: N802
        self.added_docks.append(dock)

    def removeDockWidget(self, dock):  # noqa: N802
        if dock in self.added_docks:
            self.added_docks.remove(dock)

    def activeLayer(self):  # noqa: N802
        return self._active_layer


def create_synthetic_layers():
    """Build synthetic Izmir (EPSG:5253) study boundary, power line, buildings, and DEM."""
    crs_str = "EPSG:5253"

    # 1. Study Area Boundary (600m x 600m around Izmir Tinaztepe metric coords)
    x0, y0 = 515_000.0, 4_248_000.0
    bnd = QgsVectorLayer(f"Polygon?crs={crs_str}&field=name:string(32)", "Study_Area", "memory")
    f_b = QgsFeature(bnd.fields())
    f_b.setGeometry(
        QgsGeometry.fromPolygonXY([[
            QgsPointXY(x0, y0),
            QgsPointXY(x0 + 600.0, y0),
            QgsPointXY(x0 + 600.0, y0 + 600.0),
            QgsPointXY(x0, y0 + 600.0),
            QgsPointXY(x0, y0),
        ]])
    )
    f_b.setAttribute("name", "Tinaztepe_Test")
    bnd.dataProvider().addFeatures([f_b])
    bnd.updateExtents()

    # 2. High-Voltage Power Transmission Line crossing diagonally
    pline = QgsVectorLayer(f"LineString?crs={crs_str}&field=kv:integer", "Power_Line_154kV", "memory")
    f_l = QgsFeature(pline.fields())
    f_l.setGeometry(
        QgsGeometry.fromPolylineXY([
            QgsPointXY(x0, y0 + 50.0),
            QgsPointXY(x0 + 600.0, y0 + 550.0),
        ])
    )
    f_l.setAttribute("kv", 154)
    pline.dataProvider().addFeatures([f_l])
    pline.updateExtents()

    # 3. Building Polygons with population and floor count
    bld = QgsVectorLayer(
        f"Polygon?crs={crs_str}&field=pop:double&field=floors:integer",
        "Buildings",
        "memory",
    )
    b_feats = []
    for i in range(6):
        bx = x0 + 60.0 + i * 85.0
        by = y0 + 80.0 + (i % 3) * 140.0
        fb = QgsFeature(bld.fields())
        fb.setGeometry(
            QgsGeometry.fromPolygonXY([[
                QgsPointXY(bx, by),
                QgsPointXY(bx + 45.0, by),
                QgsPointXY(bx + 45.0, by + 45.0),
                QgsPointXY(bx, by + 45.0),
                QgsPointXY(bx, by),
            ]])
        )
        fb.setAttribute("pop", float(40 + i * 25))
        fb.setAttribute("floors", int(2 + i))
        b_feats.append(fb)
    bld.dataProvider().addFeatures(b_feats)
    bld.updateExtents()

    # 4. Synthetic Slope Raster (GeoTIFF)
    fd, tif_path = tempfile.mkstemp(prefix="h3forge_slope_", suffix=".tif")
    os.close(fd)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(tif_path, 30, 30, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((x0, 20.0, 0.0, y0 + 600.0, 0.0, -20.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(5253)
    ds.SetProjection(srs.ExportToWkt())
    grid_y, grid_x = np.mgrid[0:30, 0:30]
    slope_arr = (grid_x * 0.8 + grid_y * 0.5).astype(np.float32)
    ds.GetRasterBand(1).WriteArray(slope_arr)
    ds.FlushCache()
    ds = None
    ras = QgsRasterLayer(tif_path, "Slope_Pct")

    return bnd, pline, bld, ras, tif_path
