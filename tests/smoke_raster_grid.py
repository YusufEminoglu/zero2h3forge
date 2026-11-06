# -*- coding: utf-8 -*-
"""Headless QGIS value checks for raster zonal statistics and grid resolution.

Regression tests for two v0.3.0 defects: zonal statistics that silently found
no pixels (every cell fell back to point sampling, so ``sum`` returned 13 per
cell), and scattered study areas that left the native H3 path because their
bounding box exceeded the cell budget.

    QT_QPA_PLATFORM=offscreen python3 tests/smoke_raster_grid.py
"""
from __future__ import annotations

import glob
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for extra in [str(HERE.parents[1]), str(HERE)] + glob.glob("/usr/share/qgis/python/plugins"):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import numpy as np  # noqa: E402
from osgeo import gdal, osr  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingException,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
)

gdal.UseExceptions()

X0, Y0 = 480_000.0, 4_280_000.0          # EPSG:32635, İzmir
PIXEL = 100.0


def _ones_raster(path: str, size: int = 400) -> QgsRasterLayer:
    ds = gdal.GetDriverByName("GTiff").Create(path, size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((X0, PIXEL, 0, Y0, 0, -PIXEL))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32635)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.ones((size, size), np.float32))
    ds = None
    return QgsRasterLayer(path, "ones", "gdal")


def _polygons(rings, crs="EPSG:32635") -> QgsVectorLayer:
    lyr = QgsVectorLayer(f"Polygon?crs={crs}", "area", "memory")
    feats = []
    for g in rings:
        f = QgsFeature()
        f.setGeometry(g)
        feats.append(f)
    lyr.dataProvider().addFeatures(feats)
    lyr.updateExtents()
    return lyr


class TestRasterAndGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qgs = QgsApplication([], False)
        cls.qgs.initQgis()
        cls.crs = QgsCoordinateReferenceSystem("EPSG:32635")
        cls.tmpdir = tempfile.mkdtemp(prefix="h3forge_")
        cls.ras = _ones_raster(os.path.join(cls.tmpdir, "ones.tif"))

    def _cells_inside_raster(self, res: int, limit: int = 150, span: float = 36_000.0):
        from zero2h3forge.core.h3_grid_engine import build_h3_grid_layer

        ext = QgsRectangle(X0 + 2_000, Y0 - 2_000 - span, X0 + 2_000 + span, Y0 - 2_000)
        grid = build_h3_grid_layer(ext, self.crs, res, coverage_mode="bbox")
        self.assertFalse(grid.customProperty("h3forge/coarsened"))
        box = QgsGeometry.fromRect(self.ras.extent())
        return [g for g in (f.geometry() for f in grid.getFeatures()) if box.contains(g)][:limit]

    def test_zonal_sum_counts_pixels(self):
        from zero2h3forge.core.compute import raster_zonal_array, snapshot_raster_source

        src = snapshot_raster_source(self.ras)
        for res in (8, 9):
            geoms = self._cells_inside_raster(res)
            log = []
            sums = raster_zonal_array(geoms, self.crs, src, "sum", log=log.append)
            means = raster_zonal_array(geoms, self.crs, src, "mean")
            expected = np.array([g.area() for g in geoms]) / (PIXEL * PIXEL)
            # pixel-centre counting: within a few pixels of area / pixel area
            self.assertLess(np.abs(sums - expected).max(), 0.15 * expected.min() + 3, f"res {res}")
            self.assertGreater(len(set(np.round(sums))), 1, "sums must vary with cell coverage")
            np.testing.assert_allclose(means, 1.0)
            self.assertFalse(any("point sampling" in m for m in log), log)

    def test_sub_pixel_sum_is_area_share(self):
        from zero2h3forge.core.compute import raster_zonal_array, snapshot_raster_source

        geoms = self._cells_inside_raster(12, limit=60, span=1_000.0)   # ~350 m² cells, 10 000 m² pixels
        sums = raster_zonal_array(geoms, self.crs, snapshot_raster_source(self.ras), "sum")
        share = np.array([g.area() for g in geoms]) / (PIXEL * PIXEL)
        # a cell holds at most one pixel centre; otherwise the overlap share counts
        self.assertLessEqual(sums.max(), 1.0 + 1e-9)
        self.assertTrue(np.all((np.abs(sums - share) < 1e-3) | (np.abs(sums - 1.0) < 1e-9)))

    def test_scattered_area_keeps_native_resolution(self):
        from zero2h3forge.core.h3_grid_engine import HAS_H3_LIB, build_h3_grid_layer, get_resolution_spec

        if not HAS_H3_LIB:
            self.skipTest("h3 not installed")
        random.seed(1)
        blobs = [
            QgsGeometry.fromPointXY(QgsPointXY(450_000 + random.random() * 150_000,
                                               4_200_000 + random.random() * 150_000)).buffer(1_800, 16)
            for _ in range(60)
        ]
        bnd = _polygons(blobs)
        area = sum(g.area() for g in blobs)
        cell = float(get_resolution_spec(9)["area_m2"])
        for cov in ("centroid", "intersect"):
            grid = build_h3_grid_layer(bnd.extent(), bnd.crs(), 9, boundary_layer=bnd, coverage_mode=cov)
            feats = list(grid.getFeatures())
            self.assertEqual(grid.customProperty("h3forge/engine"), "native-h3", cov)
            self.assertFalse(grid.customProperty("h3forge/coarsened"))
            self.assertTrue(all(f["h3_res"] == 9 for f in feats))
            self.assertTrue(all(not str(f["h3_id"]).startswith("h3r") for f in feats))
            mean_area = np.mean([f["area_m2"] for f in feats])
            # H3 cell area varies with latitude (~12 ha at Res 9 in İzmir vs 10.5 ha global mean)
            self.assertLess(abs(mean_area / cell - 1.0), 0.25)
            if cov == "centroid":
                self.assertLess(abs(len(feats) * mean_area / area - 1.0), 0.05)
            else:
                self.assertGreater(len(feats) * mean_area, area)

    def test_coarsened_grid_is_flagged_and_refused(self):
        import qgis

        p_dir = Path(qgis.__file__).resolve().parent.parent / "plugins"
        if p_dir.exists() and str(p_dir) not in sys.path:
            sys.path.insert(0, str(p_dir))
        import processing
        from processing.core.Processing import Processing

        from zero2h3forge.core.h3_grid_engine import build_h3_grid_layer
        from zero2h3forge.h3forge_processing import RESOLUTIONS, H3ForgeProvider

        square = QgsGeometry.fromRect(QgsRectangle(X0, Y0 - 20_000, X0 + 20_000, Y0))
        bnd = _polygons([square])
        grid = build_h3_grid_layer(bnd.extent(), bnd.crs(), 11, boundary_layer=bnd, prefer_native_h3=False)
        self.assertTrue(grid.customProperty("h3forge/coarsened"))
        self.assertFalse(any(isinstance(f["h3_res"], int) for f in grid.getFeatures()))

        Processing.initialize()
        provider = H3ForgeProvider()
        QgsApplication.processingRegistry().addProvider(provider)
        params = {"INPUT": bnd, "RESOLUTION": RESOLUTIONS.index(11), "COVERAGE": 0,
                  "NATIVE_H3": False, "OUTPUT": "memory:"}
        with self.assertRaises(QgsProcessingException):
            processing.run("zero2h3forge:h3grid", params)
        out = processing.run("zero2h3forge:h3grid", dict(params, ALLOW_COARSE=True))["OUTPUT"]
        self.assertGreater(out.featureCount(), 0)
        QgsApplication.processingRegistry().removeProvider(provider)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRasterAndGrid)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush()
    os._exit(0 if result.wasSuccessful() else 1)
