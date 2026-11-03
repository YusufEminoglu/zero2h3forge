# -*- coding: utf-8 -*-
"""Headless QGIS test of the v0.3 model pipeline (prepare → compute → apply).

    QT_QPA_PLATFORM=offscreen python3 tests/smoke_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for extra in (str(HERE.parents[1]), str(HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from qgis.core import QgsApplication, QgsProject  # noqa: E402


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qgs = QgsApplication([], False)
        cls.qgs.initQgis()

    def setUp(self):
        import fixtures as smoke_dock

        QgsProject.instance().removeAllMapLayers()
        self.bnd, self.pline, self.bld, self.ras, self.tif = smoke_dock.create_synthetic_layers()
        QgsProject.instance().addMapLayers([self.bnd, self.pline, self.bld, self.ras])
        from zero2h3forge.core.h3_grid_engine import build_h3_grid_layer

        self.grid = build_h3_grid_layer(self.bnd.extent(), self.bnd.crs(), 11, boundary_layer=self.bnd)
        QgsProject.instance().addMapLayer(self.grid)

    def tearDown(self):
        from contextlib import suppress
        self.ras = None
        QgsProject.instance().removeAllMapLayers()
        with suppress(Exception):
            if os.path.exists(self.tif):
                os.remove(self.tif)

    def _model(self):
        from zero2h3forge.core.model import (
            ConstraintSpec, CriterionSpec, SourceSpec, ValueFunction, new_model,
        )

        m = new_model("risk", "Test risk")
        m.criteria = [
            CriterionSpec("powerline", "Power line", "hazard",
                          SourceSpec("distance", layer_id=self.pline.id(), layer_name=self.pline.name()),
                          ValueFunction("rings", rings=[[0, 100, 100], [100, 250, 70], [250, 500, 35]]), 60),
            CriterionSpec("slope", "Slope", "hazard",
                          SourceSpec("raster", layer_id=self.ras.id(), layer_name=self.ras.name(), method="mean"),
                          ValueFunction("minmax"), 40),
            CriterionSpec("pop", "Population", "exposure",
                          SourceSpec("vector", layer_id=self.bld.id(), layer_name=self.bld.name(),
                                     field="pop", method="area_weighted_sum"),
                          ValueFunction("minmax"), 100),
            CriterionSpec("floors", "Floors", "vulnerability",
                          SourceSpec("vector", layer_id=self.bld.id(), layer_name=self.bld.name(),
                                     field="floors", method="max"),
                          ValueFunction("minmax"), 100),
        ]
        m.aggregation = "weighted_sum"   # synthetic data has many zero-exposure cells
        m.constraints = [
            ConstraintSpec("near_pl", "Power line corridor", "near",
                           SourceSpec("vector", layer_id=self.pline.id(), layer_name=self.pline.name()),
                           distance=20.0),
            ConstraintSpec("steep", "Steep slope", "rule", field="raw_slope", op=">", value=35.0),
        ]
        return m

    def test_full_run_and_cache(self):
        from zero2h3forge.core.pipeline import run_model_on_layer

        model = self._model()
        res = run_model_on_layer(self.grid, model)
        names = [f.name() for f in self.grid.fields()]
        for col in ("dist_powerline", "raw_slope", "raw_pop", "raw_floors", "norm_powerline",
                    "idx_hazard", "idx_exposure", "idx_vulnerability", "constraint_mask",
                    "comp_score", "class_id", "top_group", "top_driver", "mc_std",
                    "gi_z", "hotspot_lbl"):
            self.assertIn(col, names)
        self.assertGreater(res.stats["excluded_count"], 0)
        run = json.loads(self.grid.customProperty("h3forge/last_run"))
        self.assertEqual(len(run["groups"]), 3)
        self.assertIn("Power line corridor", run["excluded_by"])
        self.assertIn("Steep slope", run["excluded_by"])
        self.assertTrue(json.loads(self.grid.customProperty("h3forge/sensitivity"))["rows"])
        stored = self.grid.customProperty("h3forge/model")
        self.assertEqual(json.loads(stored)["name"], "Test risk")

        excluded = [f for f in self.grid.getFeatures() if f["constraint_mask"] == 0]
        self.assertTrue(all(f["comp_score"] == 0 for f in excluded))
        self.assertTrue(all(f["top_driver"] == "Excluded" for f in excluded))

        # Second run with new weights reuses every raw column (no recomputation)
        model.groups[0].weight = 60
        res2 = run_model_on_layer(self.grid, model)
        self.assertEqual(res2.stats["computed_raw"], [])
        # Changing a source invalidates only that criterion
        model.criteria[2].source.method = "sum"
        res3 = run_model_on_layer(self.grid, model)
        self.assertEqual(res3.stats["computed_raw"], ["raw_pop"])

    def test_thread_safe_compute_runs_in_qgstask(self):
        from qgis.core import QgsTask

        from zero2h3forge.core.pipeline import apply_run, compute_run, prepare_run

        inputs = prepare_run(self.grid, self._model())
        box = {}

        def _work(task):
            return compute_run(inputs, task.setProgress, task.isCanceled)

        task = QgsTask.fromFunction("h3forge test", _work, on_finished=lambda exc, r=None: box.update(exc=exc, res=r))
        QgsApplication.taskManager().addTask(task)
        import time

        t0 = time.time()
        while "exc" not in box and time.time() - t0 < 60:
            QgsApplication.processEvents()
            time.sleep(0.02)
        self.assertIsNone(box.get("exc"))
        apply_run(self.grid, box["res"], inputs.snap)
        self.assertIn("comp_score", [f.name() for f in self.grid.fields()])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestModelPipeline)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush()
    os._exit(0 if result.wasSuccessful() else 1)
