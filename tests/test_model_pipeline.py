# -*- coding: utf-8 -*-
"""Pure-Python tests for the v0.3 model (core/model.py) and hierarchy maths (core/pipeline.py)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PLUGINS_DIR = Path(__file__).resolve().parents[2]
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from zero2h3forge.core.model import (  # noqa: E402
    ConstraintSpec,
    CriterionSpec,
    ModelSpec,
    SourceSpec,
    ValueFunction,
    apply_template,
    new_model,
    slugify,
)
from zero2h3forge.core.pipeline import (  # noqa: E402
    GroupPlan,
    aggregate_groups,
    apply_value_function,
    compose_hierarchy,
    hierarchical_drivers,
    hierarchical_montecarlo,
    hierarchical_sensitivity,
    plans_from_model,
)


def _risk_model() -> ModelSpec:
    m = new_model("risk", "Flood risk")
    m.criteria = [
        CriterionSpec("depth", "Flood depth", "hazard", SourceSpec("raster", layer_name="dem"),
                      ValueFunction("fuzzy_linear", "benefit", low=0.0, high=2.0), 60.0),
        CriterionSpec("river", "River distance", "hazard", SourceSpec("distance", layer_name="rivers"),
                      ValueFunction("rings", rings=[[0, 50, 100], [50, 200, 60]]), 40.0),
        CriterionSpec("pop", "Population", "exposure", SourceSpec("vector", layer_name="bld", field="pop",
                                                                  method="area_weighted_sum"),
                      ValueFunction("percentile"), 100.0),
        CriterionSpec("age", "Building age", "vulnerability", SourceSpec("field", field="raw_age"),
                      ValueFunction("minmax"), 100.0),
    ]
    m.constraints = [ConstraintSpec("pl", "Power line buffer", "near",
                                    SourceSpec("vector", layer_name="lines"), distance=50.0)]
    return m


class TestModelSpec(unittest.TestCase):
    def test_templates_and_validation(self):
        m = _risk_model()
        self.assertEqual([g.id for g in m.groups], ["hazard", "exposure", "vulnerability"])
        self.assertEqual(m.aggregation, "geometric")
        self.assertEqual(m.validate(), [])
        eff = m.effective_weights()
        self.assertAlmostEqual(sum(eff.values()), 100.0, places=6)
        self.assertAlmostEqual(eff["depth"], 100 / 3 * 0.6, places=6)

    def test_validation_reports_problems(self):
        m = _risk_model()
        m.criteria[1].value.rings = [[100, 50, 10]]
        m.criteria[3].source.field = ""
        m.constraints[0].distance = 0.0
        probs = m.validate()
        self.assertEqual(len(probs), 3, probs)

    def test_json_round_trip_and_forward_compat(self):
        m = _risk_model()
        d = json.loads(m.to_json())
        d["unknown_future_key"] = 1
        d["criteria"][0]["new_field"] = "x"
        m2 = ModelSpec.from_dict(d)
        self.assertEqual(m2.to_dict()["criteria"], m.to_dict()["criteria"])
        self.assertEqual(m2.constraints[0].distance, 50.0)
        with self.assertRaises(ValueError):
            ModelSpec.from_dict({"format": "something-else"})

    def test_normalize_weights_and_template_switch(self):
        m = _risk_model()
        m.groups[0].weight = 2.0
        m.groups[1].weight = 1.0
        m.groups[2].weight = 1.0
        m.normalize_weights()
        self.assertAlmostEqual(sum(g.weight for g in m.groups), 100.0, places=2)
        self.assertAlmostEqual(m.groups[0].weight, 50.0, places=2)
        flat = apply_template(m, "flat")
        self.assertEqual(len(flat.groups), 1)
        self.assertTrue(all(c.group == "index" for c in flat.criteria))
        self.assertAlmostEqual(sum(c.weight for c in flat.criteria), 100.0, places=2)
        self.assertEqual(len(flat.constraints), 1)

    def test_groups_keep_share_while_empty(self):
        m = new_model("risk")
        m.criteria.append(CriterionSpec("a", "A", "hazard", SourceSpec("field", field="x"), weight=100))
        m.normalize_weights()
        self.assertTrue(all(abs(g.weight - 100 / 3) < 0.02 for g in m.groups))
        self.assertAlmostEqual(m.effective_weights()["a"], 100.0)
        m.criteria.append(CriterionSpec("b", "B", "exposure", SourceSpec("field", field="y"), weight=100))
        m.normalize_weights()
        eff = m.effective_weights()
        self.assertAlmostEqual(eff["a"], 50.0, places=1)
        self.assertAlmostEqual(eff["b"], 50.0, places=1)

    def test_legacy_scenario_import(self):
        legacy = {
            "format": "02h3forge-scenario", "version": 1, "name": "old",
            "settings": {"compose_method": "owa_optimistic", "theme": "suitability"},
            "criteria": [
                {"field": "norm_power_line", "weight": 70, "active": True, "locked": True},
                {"field": "norm_slope", "weight": 30, "active": True, "locked": False},
            ],
        }
        m = ModelSpec.from_dict(legacy)
        self.assertEqual(m.groups[0].method, "owa")
        self.assertAlmostEqual(m.groups[0].owa_alpha, 0.5)
        self.assertEqual([c.source.field for c in m.criteria], ["norm_power_line", "norm_slope"])
        self.assertEqual(m.criteria[0].value.method, "direct_clip")
        self.assertEqual(m.validate(), [])

    def test_unique_ids_and_slugify(self):
        m = _risk_model()
        self.assertEqual(slugify("Eğim (%)"), "egim")
        self.assertEqual(m.unique_id("Population"), "population")
        self.assertEqual(m.unique_id("pop"), "pop_2")
        self.assertTrue(m.unique_id("hotspot score").startswith("c_"))


class TestHierarchyMaths(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.m = 300
        self.norm = {k: rng.uniform(0, 100, self.m) for k in ("a", "b", "c", "d")}
        self.plans = [
            GroupPlan("h", "Hazard", 50, "wlc", 2.0, ["a", "b"], ["A", "B"], [70, 30]),
            GroupPlan("e", "Exposure", 30, "wlc", 2.0, ["c"], ["C"], [100]),
            GroupPlan("v", "Vulnerability", 20, "owa", 2.0, ["d"], ["D"], [100]),
        ]

    def test_value_functions(self):
        rings = ValueFunction("rings", rings=[[0, 100, 100], [100, 250, 70]], outside_score=5)
        np.testing.assert_allclose(apply_value_function(np.array([10, 150, 900, np.nan]), rings), [100, 70, 5, 0])
        cost = ValueFunction("minmax", "cost")
        np.testing.assert_allclose(apply_value_function(np.array([0, 5, 10]), cost), [100, 50, 0])

    def test_weighted_sum_matches_flat_effective_weights(self):
        comp, idx = compose_hierarchy(self.norm, self.plans, "weighted_sum")
        flat = (0.5 * (0.7 * self.norm["a"] + 0.3 * self.norm["b"]) + 0.3 * self.norm["c"] + 0.2 * self.norm["d"])
        np.testing.assert_allclose(comp, flat, atol=1e-9)
        np.testing.assert_allclose(idx["e"], self.norm["c"], atol=1e-9)

    def test_geometric_is_non_compensatory(self):
        G = np.array([[80.0, 80.0, 80.0], [100.0, 100.0, 0.0], [50.0, 50.0, 50.0]])
        geo = aggregate_groups(G, [1, 1, 1], "geometric")
        lin = aggregate_groups(G, [1, 1, 1], "weighted_sum")
        np.testing.assert_allclose(geo[[0, 2]], [80.0, 50.0], atol=1e-6)
        self.assertEqual(float(geo[1]), 0.0)           # no vulnerability -> no risk
        self.assertAlmostEqual(float(lin[1]), 66.6667, places=3)
        self.assertTrue(np.all(geo <= lin + 1e-9))      # AM-GM

    def test_mask_drivers_mc_and_sensitivity(self):
        mask = np.ones(self.m, dtype=np.int32)
        mask[:10] = 0
        comp, idx = compose_hierarchy(self.norm, self.plans, "geometric", mask)
        self.assertTrue(np.all(comp[:10] == 0.0))
        drv = hierarchical_drivers(self.norm, self.plans, idx, "geometric", mask)
        self.assertEqual(drv["top_driver"][0], "Excluded")
        self.assertIn(drv["top_group"][50], ("Hazard", "Exposure", "Vulnerability"))
        self.assertTrue(np.all((drv["driver_share"] >= 0) & (drv["driver_share"] <= 100)))

        mean, std, rob = hierarchical_montecarlo(self.norm, self.plans, "geometric", mask, runs=30, perturbation=0.05)
        self.assertLess(float(np.max(np.abs(mean - comp))), 3.0)
        self.assertTrue(np.all(rob <= mean + 1e-9))
        self.assertTrue(np.all(std[:10] == 0.0))

        rows = hierarchical_sensitivity(self.norm, self.plans, "geometric", mask)
        levels = {r["level"] for r in rows}
        self.assertEqual(levels, {"group", "criterion"})
        self.assertEqual(len(rows), 3 + 2)  # 3 groups + 2 criteria of the multi-criterion group

    def test_plans_from_model_skip_empty_groups(self):
        m = _risk_model()
        m.criteria[3].active = False                      # vulnerability now empty
        plans = plans_from_model(m)
        self.assertEqual([p.gid for p in plans], ["hazard", "exposure"])


if __name__ == "__main__":
    unittest.main()
