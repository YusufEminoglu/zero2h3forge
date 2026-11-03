# -*- coding: utf-8 -*-
"""Headless QGIS Processing smoke test for the 02H3Forge provider.

Runs ``zero2h3forge:h3grid`` → ``zero2h3forge:compositeoverlay`` →
``zero2h3forge:hotspots`` through ``processing.run`` exactly as a model or a
PyQGIS script would, and checks that inputs are never modified.

Run with a Python interpreter that has the QGIS bindings::

    QT_QPA_PLATFORM=offscreen python3 tests/smoke_processing.py
"""
from __future__ import annotations

import glob
import os
import sys
import unittest
from pathlib import Path

PLUGINS_ROOT = Path(__file__).resolve().parents[2]
for extra in [str(PLUGINS_ROOT), str(Path(__file__).resolve().parent)] + glob.glob("/usr/share/qgis/python/plugins"):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingException,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant  # noqa: E402


def _study_area():
    x0, y0 = 515_000.0, 4_248_000.0
    lyr = QgsVectorLayer("Polygon?crs=EPSG:5253&field=name:string(16)", "Study_Area", "memory")
    f = QgsFeature(lyr.fields())
    f.setGeometry(QgsGeometry.fromPolygonXY([[
        QgsPointXY(x0, y0), QgsPointXY(x0 + 900, y0), QgsPointXY(x0 + 900, y0 + 900),
        QgsPointXY(x0, y0 + 900), QgsPointXY(x0, y0),
    ]]))
    lyr.dataProvider().addFeatures([f])
    lyr.updateExtents()
    return lyr


class TestH3ForgeProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qgs = QgsApplication([], False)
        cls.qgs.initQgis()
        import qgis
        p_dir = Path(qgis.__file__).resolve().parent.parent / "plugins"
        if p_dir.exists() and str(p_dir) not in sys.path:
            sys.path.insert(0, str(p_dir))
        from processing.core.Processing import Processing

        Processing.initialize()
        from zero2h3forge.h3forge_processing import H3ForgeProvider

        cls.provider = H3ForgeProvider()
        QgsApplication.processingRegistry().addProvider(cls.provider)

    def test_provider_registers_algorithms(self):
        ids = sorted(a.id() for a in self.provider.algorithms())
        self.assertEqual(ids, [
            "zero2h3forge:compositeoverlay", "zero2h3forge:h3grid", "zero2h3forge:hotspots",
            "zero2h3forge:runmodel",
        ])

    def test_parse_weights(self):
        from zero2h3forge.h3forge_processing import parse_weights

        self.assertEqual(parse_weights("", 4), [25.0] * 4)
        self.assertEqual(parse_weights("2; 1 1", 3), [50.0, 25.0, 25.0])
        with self.assertRaises(QgsProcessingException):
            parse_weights("1,2", 3)
        with self.assertRaises(QgsProcessingException):
            parse_weights("a,b", 2)

    def test_pipeline_grid_overlay_hotspots(self):
        import processing

        area = _study_area()
        grid = processing.run("zero2h3forge:h3grid", {
            "INPUT": area, "RESOLUTION": 4, "COVERAGE": 0, "NATIVE_H3": True,
            "OUTPUT": "TEMPORARY_OUTPUT",
        })["OUTPUT"]
        self.assertGreater(grid.featureCount(), 20)
        self.assertGreaterEqual(grid.fields().indexFromName("h3_id"), 0)

        # Two synthetic 0-100 criteria: an east-west gradient and its mirror
        grid.startEditing()
        for fname in ("norm_east", "norm_west"):
            grid.addAttribute(QgsField(fname, QVariant.Double))
        grid.updateFields()
        xs = [f.geometry().centroid().asPoint().x() for f in grid.getFeatures()]
        lo, hi = min(xs), max(xs)
        for f in grid.getFeatures():
            x = f.geometry().centroid().asPoint().x()
            east = (x - lo) / (hi - lo) * 100.0
            grid.changeAttributeValue(f.id(), grid.fields().indexFromName("norm_east"), east)
            grid.changeAttributeValue(f.id(), grid.fields().indexFromName("norm_west"), 100.0 - east)
        grid.commitChanges()
        n_fields_before = len(grid.fields())

        comp = processing.run("zero2h3forge:compositeoverlay", {
            "INPUT": grid, "CRITERIA": ["norm_east", "norm_west"], "WEIGHTS": "80, 20",
            "METHOD": 3, "THEME": 0, "CLASSIFY": 2, "ALPHA": 0.1,
            "CONSTRAINTS": True, "MONTECARLO": True, "OUTPUT": "TEMPORARY_OUTPUT",
        })["OUTPUT"]
        names = [f.name() for f in comp.fields()]
        for req in ("comp_score", "class_id", "top_driver", "mc_std", "rob_score"):
            self.assertIn(req, names)
        self.assertEqual(len(grid.fields()), n_fields_before, "input layer must not be modified")
        scores = [f["comp_score"] for f in comp.getFeatures()]
        self.assertTrue(all(0.0 <= s <= 100.0 for s in scores))
        drivers = {f["top_driver"] for f in comp.getFeatures()}
        self.assertIn("east", drivers)

        hot = processing.run("zero2h3forge:hotspots", {
            "INPUT": comp, "FIELD": "comp_score", "RESPECT_MASK": True, "K_RING": 2, "FDR": True,
            "OUTPUT": "TEMPORARY_OUTPUT",
        })
        hot_layer = hot["OUTPUT"]
        self.assertGreater(float(hot["MORAN_I"]), 0.5)  # smooth gradient => strong clustering
        codes = {f["hotspot"] for f in hot_layer.getFeatures()}
        self.assertTrue(any(c >= 2 for c in codes))
        self.assertTrue(any(c <= -2 for c in codes))

    def test_owa_alpha_changes_result(self):
        import processing

        grid = processing.run("zero2h3forge:h3grid", {
            "INPUT": _study_area(), "RESOLUTION": 4, "OUTPUT": "TEMPORARY_OUTPUT",
        })["OUTPUT"]
        grid.startEditing()
        for fname in ("norm_a", "norm_b"):
            grid.addAttribute(QgsField(fname, QVariant.Double))
        grid.updateFields()
        for i, f in enumerate(grid.getFeatures()):
            grid.changeAttributeValue(f.id(), grid.fields().indexFromName("norm_a"), float(i % 7) * 14.0)
            grid.changeAttributeValue(f.id(), grid.fields().indexFromName("norm_b"), 100.0 - float(i % 5) * 20.0)
        grid.commitChanges()
        means = {}
        for alpha in (0.3, 1.0, 3.0):
            out = processing.run("zero2h3forge:compositeoverlay", {
                "INPUT": grid, "CRITERIA": ["norm_a", "norm_b"], "METHOD": 1, "OWA_ALPHA": alpha,
                "MONTECARLO": False, "OUTPUT": "TEMPORARY_OUTPUT",
            })["OUTPUT"]
            vals = [f["comp_score"] for f in out.getFeatures()]
            means[alpha] = sum(vals) / len(vals)
        self.assertGreater(means[0.3], means[1.0])
        self.assertGreater(means[1.0], means[3.0])

    def test_runmodel_from_file(self):
        import tempfile

        import processing
        from qgis.core import QgsProject

        from fixtures import create_synthetic_layers
        from zero2h3forge.core.model import (
            ConstraintSpec, CriterionSpec, SourceSpec, ValueFunction, new_model,
        )

        bnd, pline, bld, ras, tif = create_synthetic_layers()
        QgsProject.instance().addMapLayers([bnd, pline, bld, ras])
        try:
            grid = processing.run("zero2h3forge:h3grid", {
                "INPUT": bnd, "RESOLUTION": 5, "OUTPUT": "TEMPORARY_OUTPUT",
            })["OUTPUT"]
            m = new_model("suitability", "Proc test")
            m.criteria = [
                CriterionSpec("pl", "Power line", "physical",
                              SourceSpec("distance", layer_name=pline.name()),
                              ValueFunction("fuzzy_linear", "benefit", low=0.0, high=400.0), 100),
                CriterionSpec("slope", "Slope", "physical",
                              SourceSpec("raster", layer_name=ras.name(), layer_uri=tif, provider="gdal",
                                         method="median"),
                              ValueFunction("minmax", "cost"), 0),
                CriterionSpec("pop", "Population", "socio",
                              SourceSpec("vector", layer_name=bld.name(), field="pop", method="sum"),
                              ValueFunction("minmax"), 100),
            ]
            m.constraints = [ConstraintSpec("near", "Line", "near", SourceSpec("vector", layer_name=pline.name()),
                                            distance=15.0)]
            m.normalize_weights()
            with tempfile.TemporaryDirectory() as tmp:
                path = f"{tmp}/model.json"
                with open(path, "w", encoding="utf-8") as fp:
                    fp.write(m.to_json())
                out = processing.run("zero2h3forge:runmodel", {
                    "INPUT": grid, "MODEL": path, "OUTPUT": "TEMPORARY_OUTPUT",
                })["OUTPUT"]
            names = [f.name() for f in out.fields()]
            for col in ("dist_pl", "raw_slope", "raw_pop", "idx_physical", "idx_socio", "comp_score",
                        "top_group", "constraint_mask", "hotspot"):
                self.assertIn(col, names)
            self.assertNotIn("comp_score", [f.name() for f in grid.fields()])
            self.assertTrue(any(f["constraint_mask"] == 0 for f in out.getFeatures()))
        finally:
            QgsProject.instance().removeAllMapLayers()

    def test_hotspots_requires_h3_id(self):
        import processing

        with self.assertRaises(QgsProcessingException):
            processing.run("zero2h3forge:hotspots", {
                "INPUT": _study_area(), "FIELD": "name", "OUTPUT": "TEMPORARY_OUTPUT",
            })


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestH3ForgeProcessing)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush()
    os._exit(0 if res.wasSuccessful() else 1)
