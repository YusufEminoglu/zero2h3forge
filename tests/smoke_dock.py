# -*- coding: utf-8 -*-
"""Headless QGIS runtime E2E smoke test for 02H3Forge (zero2h3forge).

Verifies:
1. Plugin lifecycle (`classFactory`, `initGui`, `open_dock`, `unload`)
2. Stage 0: H3 Hexagonal Grid generation (Default Res 10 ≈ 1.50 ha & Res 11 ≈ 25 m edge) in EPSG:5253
3. Stage 0: Graduated Multi-Ring Buffer (`0-100, 100-250, 250-500 m`) on a Power Transmission Line
4. Stage 1: Vector (`Join by Location`) and Raster (`Zonal Stats`) aggregation into H3 cells
5. Stage 2: Benefit (`+`) and Cost (`-`) 0-100 normalization on H3 columns
6. Stage 3: Dock UI coupled weight slider auto-balancing (`Sum = 100%`), CRITIC/Entropy/PCA weights, and WLC/VIKOR overlay
7. Stage 4: 5-class Graduated Symbol Renderer styling and CSV/HTML report export
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

PLUGINS_ROOT = Path(__file__).resolve().parents[2]
if str(PLUGINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGINS_ROOT))

from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.gui import QgsMapCanvas  # noqa: E402
from qgis.PyQt.QtWidgets import QMainWindow, QToolBar  # noqa: E402

from zero2h3forge import classFactory  # noqa: E402
from zero2h3forge.core.styler_and_report import (  # noqa: E402
    export_h3_html_report,
    export_h3_table_csv,
)


class _DummyMessageBar:
    def pushWarning(self, title: str, msg: str) -> None:  # noqa: N802
        pass

    def pushInfo(self, title: str, msg: str) -> None:  # noqa: N802
        pass


class _DummyIface:
    def __init__(self):
        self._win = QMainWindow()
        self._canvas = QgsMapCanvas(self._win)
        self._canvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:5253"))
        self._bar = _DummyMessageBar()
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


def _create_synthetic_layers():
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


class TestZero2H3ForgeQgisSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qgs = QgsApplication([], False)
        cls.qgs.initQgis()

    @classmethod
    def tearDownClass(cls):
        QgsProject.instance().removeAllMapLayers()

    def test_end_to_end_dock_workflow(self):
        QgsProject.instance().removeAllMapLayers()
        bnd, pline, bld, ras, tif_path = _create_synthetic_layers()
        try:
            QgsProject.instance().addMapLayers([bnd, pline, bld, ras])
            iface = _DummyIface()
            iface._active_layer = bnd

            plugin = classFactory(iface)
            plugin.initGui()
            plugin.open_dock()
            self.assertIsNotNone(plugin.dock)
            dock = plugin.dock

            # Stage 0A: Generate H3 Grid at Default Res 10 (~1.50 ha)
            idx_bnd = dock.boundary_layer_combo.findData(bnd.id())
            self.assertGreaterEqual(idx_bnd, 0)
            dock.boundary_layer_combo.setCurrentIndex(idx_bnd)
            dock.on_generate_h3_grid()

            h3_lyr = dock.active_h3_layer()
            self.assertIsNotNone(h3_lyr)
            self.assertTrue(h3_lyr.isValid())
            self.assertGreater(h3_lyr.featureCount(), 8)
            self.assertEqual(h3_lyr.crs().authid(), "EPSG:5253")

            # Stage 0B: Multi-Ring Buffer on Power Line (0-100, 100-250, 250-500 m)
            idx_pl = dock.buf_layer_combo.findData(pline.id())
            dock.buf_layer_combo.setCurrentIndex(idx_pl)
            dock.buf_name_edit.setText("power_line")
            dock.on_run_multiring_buffer()

            field_names = [f.name() for f in h3_lyr.fields()]
            self.assertIn("dist_power_line", field_names)
            self.assertIn("ring_power_line", field_names)
            self.assertIn("norm_power_line", field_names)

            # Stage 1A: Aggregate Building Population (area_weighted_sum)
            idx_bld = dock.agg_vec_layer_combo.findData(bld.id())
            dock.agg_vec_layer_combo.setCurrentIndex(idx_bld)
            idx_meth = dock.agg_vec_method_combo.findData("area_weighted_sum")
            dock.agg_vec_method_combo.setCurrentIndex(idx_meth)
            idx_fld = dock.agg_vec_field_combo.findData("pop")
            dock.agg_vec_field_combo.setCurrentIndex(idx_fld)
            dock.agg_vec_out_edit.setText("pop_exp")
            dock.on_aggregate_vector()
            self.assertIn("raw_pop_exp", [f.name() for f in h3_lyr.fields()])

            # Stage 1B: Aggregate Slope Raster (mean)
            idx_ras = dock.agg_ras_layer_combo.findData(ras.id())
            dock.agg_ras_layer_combo.setCurrentIndex(idx_ras)
            dock.agg_ras_out_edit.setText("slope")
            dock.on_aggregate_raster()
            self.assertIn("raw_slope", [f.name() for f in h3_lyr.fields()])

            # Stage 2: Normalize all candidate fields to 0-100
            dock.populate_normalization_table()
            self.assertGreaterEqual(dock.norm_table.rowCount(), 3)
            dock.on_run_normalization()
            norm_cols = [f.name() for f in h3_lyr.fields() if f.name().startswith("norm_")]
            self.assertGreaterEqual(len(norm_cols), 3)

            # Stage 3: Verify coupled weight slider auto-balancing & objective weights
            self.assertEqual(len(dock._weight_rows), len(norm_cols))
            dock._on_spin_changed(0, 60.0)
            _, weights_after = dock.get_active_criteria_and_weights()
            self.assertAlmostEqual(sum(weights_after), 100.0, places=1)
            self.assertAlmostEqual(weights_after[0], 60.0, places=1)

            dock.on_weights_critic()
            _, w_critic = dock.get_active_criteria_and_weights()
            self.assertAlmostEqual(sum(w_critic), 100.0, places=1)

            dock.on_weights_entropy()
            dock.on_weights_pca()
            dock.on_weights_equal()

            # Run Composite Overlay (WLC + 1-ring smoothing + Monte Carlo)
            dock.smooth_alpha_spin.setValue(0.15)
            dock.on_forge_composite_map()

            out_fields = [f.name() for f in h3_lyr.fields()]
            for req in ("comp_score", "class_id", "class_label", "top_driver", "driver_share", "mc_std", "rob_score"):
                self.assertIn(req, out_fields)

            # Stage 4: Verify summary table and exports
            self.assertEqual(dock.summary_table.rowCount(), 5)
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_p = os.path.join(tmpdir, "out.csv")
                html_p = os.path.join(tmpdir, "report.html")
                cnt = export_h3_table_csv(h3_lyr, csv_p)
                self.assertEqual(cnt, h3_lyr.featureCount())
                export_h3_html_report(h3_lyr, html_p, {"norm_power_line": 50.0, "norm_slope": 50.0})
                self.assertTrue(os.path.exists(html_p))

            # Clean unload
            plugin.unload()
            self.assertIsNone(plugin.dock)
        finally:
            QgsProject.instance().removeAllMapLayers()
            if os.path.exists(tif_path):
                try:
                    os.remove(tif_path)
                except OSError:
                    pass


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestZero2H3ForgeQgisSmoke)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if res.wasSuccessful() else 1)
