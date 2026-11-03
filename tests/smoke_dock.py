# -*- coding: utf-8 -*-
"""Headless end-to-end test of the v0.3 criterion-centric studio dock.

Covers plugin lifecycle, grid generation, criterion/constraint dialogs, the
risk template (Hazard × Exposure × Vulnerability), coupled weights, blocking
and background (QgsTask) runs incl. cancellation, caching, results views,
styling, reports, GeoPackage persistence, model reload from a layer and the
signal hygiene of ``dispose()``.

    QT_QPA_PLATFORM=offscreen python3 tests/smoke_dock.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for extra in (str(HERE.parents[1]), str(HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from qgis.core import QgsApplication, QgsProject, QgsVectorLayer  # noqa: E402

from fixtures import DummyIface, create_synthetic_layers  # noqa: E402
from zero2h3forge import classFactory  # noqa: E402


def _wait(predicate, timeout=60.0):
    t0 = time.time()
    while not predicate() and time.time() - t0 < timeout:
        QgsApplication.processEvents()
        time.sleep(0.02)
    QgsApplication.processEvents()
    return predicate()


class TestStudioDock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.qgs = QgsApplication([], False)
        cls.qgs.initQgis()

    def test_end_to_end(self):
        from zero2h3forge.core.model import (
            ConstraintSpec, CriterionSpec, ModelSpec, SourceSpec, ValueFunction,
        )
        from zero2h3forge.core.styler_and_report import export_h3_html_report
        from zero2h3forge.ui.dialogs import ConstraintDialog, CriterionDialog

        QgsProject.instance().removeAllMapLayers()
        bnd, pline, bld, ras, tif = create_synthetic_layers()
        try:
            QgsProject.instance().addMapLayers([bnd, pline, bld, ras])
            iface = DummyIface()
            plugin = classFactory(iface)
            plugin.initGui()
            plugin.open_dock()
            dock = plugin.dock

            # ---- 1. grid
            dock._select(dock.boundary_layer_combo, bnd.id())
            dock._select(dock.res_combo, 11)
            self.assertIn("hexagons", dock.grid_estimate_lbl.text())
            dock.on_generate_h3_grid()
            grid = dock.active_h3_layer()
            self.assertIsNotNone(grid)
            self.assertGreater(grid.featureCount(), 100)

            # ---- 2. risk structure + criteria (one through the real dialog)
            dock._select(dock.template_combo, "risk")
            dock.on_apply_template()
            self.assertEqual([g.id for g in dock.model.groups], ["hazard", "exposure", "vulnerability"])

            dlg = CriterionDialog(dock.model, grid, parent=dock)
            dlg.label_edit.setText("Power line")
            dock._select(dlg.group_combo, "hazard")
            dock._select(dlg.kind_combo, "distance")
            dock._select(dlg.dist_layer, pline.id())
            self.assertEqual(dlg.vf_method.currentData(), "rings")
            self.assertGreater(dlg.ring_table.rowCount(), 0)  # preset auto-loaded
            dlg._accept()
            self.assertIsNotNone(dlg.result_spec)
            dock.add_criterion(dlg.result_spec)

            dock.add_criterion(CriterionSpec(
                "slope", "Slope", "hazard",
                SourceSpec("raster", layer_id=ras.id(), layer_name=ras.name(), method="mean"),
                ValueFunction("minmax")))
            dock.add_criterion(CriterionSpec(
                "pop", "Population", "exposure",
                SourceSpec("vector", layer_id=bld.id(), layer_name=bld.name(), field="pop",
                           method="area_weighted_sum"),
                ValueFunction("minmax")))
            dock.add_criterion(CriterionSpec(
                "floors", "Floors", "vulnerability",
                SourceSpec("vector", layer_id=bld.id(), layer_name=bld.name(), field="floors", method="max"),
                ValueFunction("minmax")))
            self.assertEqual(dock.crit_tree.topLevelItemCount(), 3)

            kdlg = ConstraintDialog(dock.model, grid, parent=dock)
            kdlg.label_edit.setText("Corridor")
            dock._select(kdlg.kind_combo, "near")
            dock._select(kdlg.layer_combo, pline.id())
            kdlg.dist_spin.setValue(20.0)
            kdlg._accept()
            dock.model.constraints.append(kdlg.result_spec)
            dock.model.constraints.append(ConstraintSpec("steep", "Steep", "rule", field="raw_slope",
                                                         op=">", value=35.0))
            dock._fill_constraints()
            self.assertEqual(dock.cons_list.count(), 2)

            # ---- 3. weights: coupled widgets write through to the model
            dock._select(dock.agg_combo, "weighted_sum")
            hazard_w = dock._group_widgets["hazard"]
            hazard_w._on_edit(0, 70.0)
            self.assertAlmostEqual(sum(hazard_w.weights()), 100.0, places=1)
            self.assertAlmostEqual(dock.model.criterion(dock.model.criteria_in("hazard")[0].id).weight, 70.0, places=1)
            dock.group_weights._on_edit(0, 50.0)
            self.assertAlmostEqual(dock.model.groups[0].weight, 50.0, places=1)
            self.assertAlmostEqual(sum(dock.model.effective_weights().values()), 100.0, places=4)

            # ---- 4. blocking run
            self.assertTrue(dock.run_model(blocking=True))
            names = [f.name() for f in grid.fields()]
            for col in ("comp_score", "idx_hazard", "idx_exposure", "idx_vulnerability", "top_group",
                        "constraint_mask", "mc_std", "hotspot_lbl", "raw_slope"):
                self.assertIn(col, names)
            self.assertEqual(dock.summary_table.rowCount(), 5)
            self.assertGreater(dock.sens_table.rowCount(), 0)
            self.assertIn("Moran", dock.hotspot_lbl.text())
            for i in range(dock.map_field_combo.count()):
                dock.map_field_combo.setCurrentIndex(i)  # every map style applies cleanly
            self.assertGreaterEqual(dock.map_field_combo.count(), 6)

            # objective weights now possible (norm_* exist)
            dock.on_objective_weights("hazard", "critic")
            self.assertAlmostEqual(sum(dock._group_widgets["hazard"].weights()), 100.0, places=1)

            # ---- 5. background run (cached raw columns) + cancellation
            dock._select(dock.agg_combo, "geometric")
            self.assertTrue(dock.run_model())
            self.assertFalse(dock.run_btn.isEnabled())
            self.assertTrue(_wait(lambda: dock._task is None))
            self.assertTrue(dock.run_btn.isEnabled())
            run = json.loads(grid.customProperty("h3forge/last_run"))
            self.assertEqual(run["aggregation"], "geometric")
            self.assertIn("limit_group", [f.name() for f in grid.fields()])

            dock.force_chk.setChecked(True)
            self.assertTrue(dock.run_model())
            dock.cancel_run()
            self.assertTrue(_wait(lambda: dock._task is None))
            self.assertTrue(dock.run_btn.isEnabled())
            dock.force_chk.setChecked(False)

            # ---- 6. report, model JSON, persistence
            with tempfile.TemporaryDirectory() as tmp:
                html = os.path.join(tmp, "r.html")
                export_h3_html_report(grid, html, run["weights"], run["method"], run["theme"])
                doc = Path(html).read_text(encoding="utf-8")
                self.assertIn("Model Structure", doc)
                self.assertIn("Hazard", doc)
                self.assertIn("Corridor", doc)

                model_json = dock.model.to_json()
                self.assertEqual(ModelSpec.from_json(model_json).to_dict(), dock.model.to_dict())

                from zero2h3forge.core.styler_and_report import save_h3_layer_to_geopackage

                saved = save_h3_layer_to_geopackage(grid, os.path.join(tmp, "s.gpkg"))
                self.assertTrue(saved.customProperty("h3forge/model"))
                del saved

            # switching layers restores each layer's own model
            other = QgsVectorLayer("Polygon?crs=EPSG:5253&field=h3_id:string(20)", "empty_h3", "memory")
            QgsProject.instance().addMapLayer(other)
            dock._select(dock.h3_layer_combo, grid.id())
            dock._on_active_layer_changed()
            self.assertEqual(dock.model.template, "risk")
            self.assertEqual(len(dock.model.criteria), 4)

            # ---- 7. dispose only drops our own slots
            hits = []
            probe = lambda _l: hits.append(1)  # noqa: E731
            QgsProject.instance().layersAdded.connect(probe)
            plugin.unload()
            self.assertIsNone(plugin.dock)
            QgsProject.instance().addMapLayer(QgsVectorLayer("Point?crs=EPSG:5253", "probe", "memory"))
            self.assertEqual(hits, [1])
            QgsProject.instance().layersAdded.disconnect(probe)
        finally:
            from contextlib import suppress
            QgsProject.instance().removeAllMapLayers()
            with suppress(Exception):
                if os.path.exists(tif):
                    os.remove(tif)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestStudioDock)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.stdout.flush()
    os._exit(0 if res.wasSuccessful() else 1)
