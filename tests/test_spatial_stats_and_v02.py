# -*- coding: utf-8 -*-
"""Pure-Python tests for the 02H3Forge v0.2 engines (no QGIS required).

Covers Global Moran's I and Getis-Ord Gi* (checked against brute-force dense
formulas), constraint-aware TOPSIS / VIKOR / contagion smoothing / XAI,
method-consistent Monte Carlo, scenario files, grid-size estimates and the
inline SVG report charts.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = PLUGIN_ROOT.parent
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from zero2h3forge.core.h3_grid_engine import (  # noqa: E402
    estimate_hex_count,
    generate_metric_hex_cells,
    sanitize_field_slug,
)
from zero2h3forge.core.mcda_composer import (  # noqa: E402
    apply_h3_spatial_smoothing,
    compose_scores,
    compose_topsis,
    compose_vikor,
    compute_local_xai_drivers,
    _class_ids_with_breaks,
    classify_scores_5_bands,
    compute_montecarlo_stability,
    oat_weight_sensitivity,
    seed_weights_for_fields,
)
from zero2h3forge.core.scenario import (  # noqa: E402
    build_scenario,
    load_scenario,
    match_scenario_criteria,
    save_scenario,
    validate_scenario,
)
from zero2h3forge.core.spatial_stats import (  # noqa: E402
    build_neighbor_lists,
    classify_hotspots,
    getis_ord_gi_star,
    global_morans_i,
    summarize_hotspots,
)
from zero2h3forge.core.styler_and_report import (  # noqa: E402
    svg_score_histogram,
    svg_weight_bars,
)


def _grid(width=1200.0, height=1200.0, res=10):
    cells = generate_metric_hex_cells(0.0, 0.0, width, height, resolution=res)
    ids = [cid for cid, _ in cells]
    centers = np.array([
        [np.mean([p[0] for p in ring[:-1]]), np.mean([p[1] for p in ring[:-1]])]
        for _, ring in cells
    ])
    return ids, centers


def _dense_w(neighbors, n):
    W = np.zeros((n, n))
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            W[i, j] = 1.0
    return W


class TestSpatialStatistics(unittest.TestCase):
    def setUp(self):
        self.ids, self.centers = _grid()
        self.n = len(self.ids)
        self.nbrs = build_neighbor_lists(self.ids)

    def test_neighbors_are_symmetric_hexagonal(self):
        self.assertGreater(self.n, 50)
        degrees = [len(nb) for nb in self.nbrs]
        self.assertEqual(max(degrees), 6)
        for i, nb in enumerate(self.nbrs):
            for j in nb:
                self.assertIn(i, self.nbrs[j])

    def test_morans_i_matches_dense_formula(self):
        rng = np.random.default_rng(7)
        x = rng.normal(50.0, 10.0, self.n)
        res = global_morans_i(x, self.nbrs)

        W = _dense_w(self.nbrs, self.n)
        z = x - x.mean()
        s0 = W.sum()
        i_ref = (self.n / s0) * (z @ W @ z) / (z @ z)
        s1 = 0.5 * np.sum((W + W.T) ** 2)
        s2 = np.sum((W.sum(1) + W.sum(0)) ** 2)
        n = self.n
        e_i = -1.0 / (n - 1)
        var = (n * n * s1 - n * s2 + 3 * s0 * s0) / ((n * n - 1) * s0 * s0) - e_i ** 2
        z_ref = (i_ref - e_i) / math.sqrt(var)

        self.assertAlmostEqual(res["I"], i_ref, places=5)
        self.assertAlmostEqual(res["z"], z_ref, places=3)
        self.assertAlmostEqual(res["expected"], e_i, places=6)

    def test_morans_i_detects_clustering(self):
        # Smooth west-east gradient => strong positive autocorrelation
        x = self.centers[:, 0]
        res = global_morans_i(x, self.nbrs)
        self.assertGreater(res["I"], 0.7)
        self.assertGreater(res["z"], 5.0)
        self.assertLess(res["p"], 0.001)

    def test_gi_star_matches_dense_formula_and_finds_hotspot(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0.0, 20.0, self.n)
        # Plant a hot cluster in the grid center
        c = self.centers.mean(axis=0)
        hot = np.linalg.norm(self.centers - c, axis=1) < 200.0
        x[hot] += 80.0

        z, p, codes = getis_ord_gi_star(x, self.nbrs)

        W = _dense_w(self.nbrs, self.n) + np.eye(self.n)  # self-inclusive
        n = self.n
        xb = x.mean()
        s = math.sqrt(np.mean(x ** 2) - xb ** 2)
        wsum = W.sum(1)
        num = W @ x - xb * wsum
        den = s * np.sqrt((n * np.sum(W ** 2, axis=1) - wsum ** 2) / (n - 1))
        np.testing.assert_allclose(z, num / den, atol=1e-3)

        self.assertTrue(np.all(codes[hot] >= 2))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        counts = summarize_hotspots(codes)
        self.assertEqual(sum(counts.values()), n)

    def test_gi_star_ignores_excluded_cells(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(45.0, 55.0, self.n)  # spatially random field
        c = self.centers.mean(axis=0)
        blocked = np.linalg.norm(self.centers - c, axis=1) < 200.0
        x[blocked] = 0.0  # forced zero from a hard exclusion ring
        valid = ~blocked

        # Ring of feasible hexagons touching the excluded block
        touching = np.zeros(self.n, dtype=bool)
        for i in np.flatnonzero(blocked):
            touching[self.nbrs[i]] = True
        touching &= valid
        self.assertTrue(np.any(touching))

        # Naive analysis: the zeros fabricate a significant cold ring
        _, _, naive = getis_ord_gi_star(x, self.nbrs)
        self.assertTrue(np.any(naive[touching] <= -2))

        # Constraint-aware analysis: excluded cells are dropped entirely
        nbrs = build_neighbor_lists(self.ids, valid=valid)
        z, _, codes = getis_ord_gi_star(x, nbrs, valid=valid)
        self.assertTrue(np.all(z[blocked] == 0.0))
        self.assertTrue(np.all(codes[blocked] == 0))
        self.assertFalse(np.any(codes[touching] <= -2))
        for nb in nbrs:
            for j in nb:
                self.assertTrue(valid[j])

    def test_classify_hotspots_bins(self):
        codes = classify_hotspots([3.0, 2.0, 1.7, 0.5, -1.7, -2.0, -2.6])
        self.assertEqual(codes.tolist(), [3, 2, 1, 0, -1, -2, -3])

    def test_constant_values_are_safe(self):
        z, p, codes = getis_ord_gi_star(np.ones(self.n), self.nbrs)
        self.assertTrue(np.all(z == 0.0))
        self.assertTrue(np.all(codes == 0))
        res = global_morans_i(np.ones(self.n), self.nbrs)
        self.assertTrue(math.isnan(res["I"]))


class TestConstraintAwareMcda(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.X = rng.uniform(10.0, 90.0, size=(30, 3))
        self.w = [50.0, 30.0, 20.0]
        self.mask = np.ones(30, dtype=int)
        self.mask[0] = 0

    def test_topsis_and_vikor_ideals_ignore_excluded_rows(self):
        X_out = self.X.copy()
        X_out[0] = [100.0, 0.0, 100.0]  # extreme excluded row
        for fn in (compose_topsis, compose_vikor):
            with_out = fn(X_out, self.w, constraint_mask=self.mask)
            subset = fn(self.X[1:], self.w)
            np.testing.assert_allclose(with_out[1:], subset, atol=1e-9)
            self.assertEqual(float(with_out[0]), 0.0)

    def test_smoothing_does_not_leak_excluded_zeros(self):
        ids, _ = _grid(600.0, 600.0)
        scores = np.full(len(ids), 80.0)
        mask = np.ones(len(ids), dtype=int)
        mask[::3] = 0
        scores[mask == 0] = 0.0
        out = apply_h3_spatial_smoothing(scores, ids, alpha=0.4, constraint_mask=mask)
        np.testing.assert_allclose(out[mask == 1], 80.0)
        np.testing.assert_allclose(out[mask == 0], 0.0)

    def test_xai_marks_excluded(self):
        drivers, shares = compute_local_xai_drivers(
            self.X, self.w, ["norm_a", "norm_b", "norm_c"], constraint_mask=self.mask
        )
        self.assertEqual(drivers[0], "Excluded")
        self.assertEqual(float(shares[0]), 0.0)
        self.assertIn(drivers[1], ("a", "b", "c"))

    def test_montecarlo_uses_selected_engine(self):
        for method in ("wlc", "topsis", "vikor", "owa_pessimistic"):
            mc_mean, mc_std, rob = compute_montecarlo_stability(
                self.X, self.w, n_runs=40, perturbation=0.05, method=method
            )
            base = compose_scores(self.X, self.w, method=method)
            # Small perturbations => MC mean stays close to the deterministic map
            self.assertLess(float(np.max(np.abs(mc_mean - base))), 3.0, method)
            self.assertTrue(np.all(mc_std >= 0.0))
            self.assertTrue(np.all(rob <= mc_mean + 1e-9))


class TestWeightSensitivity(unittest.TestCase):
    def test_vectorized_classes_match_reference_classifier(self):
        scores = np.array([0.0, 19.99, 20.0, 20.01, 40.0, 59.9, 60.0, 80.0, 80.01, 100.0])
        ref, _, breaks = classify_scores_5_bands(scores)
        np.testing.assert_array_equal(_class_ids_with_breaks(scores, breaks), ref)

    def test_influential_criterion_ranks_first(self):
        rng = np.random.default_rng(21)
        m = 400
        X = np.column_stack([
            rng.uniform(0.0, 100.0, m),       # high-contrast criterion
            50.0 + rng.normal(0.0, 1.0, m),   # almost constant criterion
            rng.uniform(40.0, 60.0, m),
        ])
        rows = oat_weight_sensitivity(X, [40.0, 30.0, 30.0], ["norm_hazard", "norm_flat", "norm_mid"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["criterion"], "hazard")
        for r in rows:
            self.assertGreaterEqual(r["class_change_pct"], 0.0)
            self.assertLessEqual(r["class_change_pct"], 100.0)
            self.assertGreaterEqual(r["max_abs_change"], r["mean_abs_change"])
        pct = [r["class_change_pct"] for r in rows]
        self.assertEqual(pct, sorted(pct, reverse=True))

    def test_zero_weight_and_excluded_rows(self):
        X = np.array([[90.0, 10.0], [10.0, 90.0], [50.0, 50.0], [100.0, 100.0]])
        rows = oat_weight_sensitivity(
            X, [100.0, 0.0], ["norm_a", "norm_b"], constraint_mask=[1, 1, 1, 0], method="topsis"
        )
        zero = next(r for r in rows if r["criterion"] == "b")
        self.assertEqual(zero["mean_abs_change"], 0.0)
        self.assertEqual(zero["weight"], 0.0)


class TestWeightSeedingAndNaming(unittest.TestCase):
    def test_fresh_layer_gets_equal_weights(self):
        w, a, lk = seed_weights_for_fields(["norm_a", "norm_b", "norm_c", "norm_d"])
        self.assertAlmostEqual(sum(w), 100.0, places=2)
        np.testing.assert_allclose(w, 25.0, atol=0.01)
        self.assertTrue(all(a))
        self.assertFalse(any(lk))

    def test_newcomers_get_fair_share_and_known_keep_proportions(self):
        prev = {
            "norm_a": {"weight": 75.0, "active": True, "locked": False},
            "norm_b": {"weight": 25.0, "active": True, "locked": False},
        }
        w, _, _ = seed_weights_for_fields(["norm_a", "norm_b", "norm_c", "norm_d"], prev)
        self.assertAlmostEqual(sum(w), 100.0, places=2)
        self.assertAlmostEqual(w[2], 25.0, places=2)
        self.assertAlmostEqual(w[3], 25.0, places=2)
        self.assertAlmostEqual(w[0] / w[1], 3.0, places=3)

    def test_locks_and_inactive_are_preserved(self):
        prev = {
            "norm_a": {"weight": 40.0, "active": True, "locked": True},
            "norm_b": {"weight": 60.0, "active": False, "locked": False},
        }
        w, a, lk = seed_weights_for_fields(["norm_a", "norm_b", "norm_c"], prev)
        self.assertEqual(a, [True, False, True])
        self.assertEqual(lk, [True, False, False])
        self.assertAlmostEqual(w[0], 40.0, places=2)
        self.assertAlmostEqual(w[1], 0.0, places=2)
        self.assertAlmostEqual(w[2], 60.0, places=2)

    def test_distance_and_ring_normalize_to_distinct_columns(self):
        self.assertEqual(sanitize_field_slug("ring_power_line", "norm"), "norm_power_line")
        self.assertEqual(sanitize_field_slug("dist_power_line", "norm"), "norm_dist_power_line")
        self.assertEqual(sanitize_field_slug("raw_slope", "norm"), "norm_slope")


class TestScenarioFiles(unittest.TestCase):
    def test_round_trip_and_matching(self):
        sc = build_scenario(
            [
                {"field": "norm_power_line", "weight": 40.0, "active": True, "locked": True},
                {"field": "norm_slope", "weight": 60.0, "active": True, "locked": False},
                {"field": "", "weight": 1.0},
            ],
            {"compose_method": "topsis", "smooth_alpha": 0.9, "unknown_key": 1},
            name="Flood-safe housing",
            layer_name="H3_Grid",
            plugin_version="0.2.0",
        )
        self.assertEqual(len(sc["criteria"]), 2)
        self.assertNotIn("unknown_key", sc["settings"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.json")
            save_scenario(path, sc)
            loaded = load_scenario(path)
        self.assertEqual(loaded["settings"]["compose_method"], "topsis")
        self.assertEqual(loaded["settings"]["smooth_alpha"], 0.5)  # clamped
        self.assertTrue(loaded["criteria"][0]["locked"])

        m = match_scenario_criteria(loaded, ["norm_slope", "norm_roads"])
        self.assertEqual(list(m["matched"]), ["norm_slope"])
        self.assertEqual(m["missing"], ["norm_power_line"])

    def test_rejects_foreign_json(self):
        with self.assertRaises(ValueError):
            validate_scenario({"hello": "world"})
        with self.assertRaises(ValueError):
            validate_scenario({"format": "02h3forge-scenario", "version": 99})
        self.assertTrue(json.dumps(build_scenario([], {})))


class TestGridEstimateAndReportCharts(unittest.TestCase):
    def test_estimate_tracks_actual_metric_cell_count(self):
        for res, size in ((9, 3000.0), (10, 1500.0), (11, 600.0)):
            actual = len(generate_metric_hex_cells(0.0, 0.0, size, size, resolution=res))
            est = estimate_hex_count(size, size, res)
            self.assertLess(abs(est - actual) / actual, 0.35, (res, est, actual))
        self.assertEqual(estimate_hex_count(0.0, 100.0, 10), 0)

    def test_svg_charts(self):
        svg = svg_score_histogram([5.0, 15.0, 55.0, 95.0, 99.0], breaks=[0, 20, 40, 60, 80, 100])
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("#b30000", svg)  # top class color present
        self.assertIn("No scores", svg_score_histogram([]))
        bars = svg_weight_bars({"norm_slope": 60.0, "norm_<x>": 40.0})
        self.assertIn("slope", bars)
        self.assertIn("&lt;x&gt;", bars)


if __name__ == "__main__":
    unittest.main()


class TestV03MethodUpgrades(unittest.TestCase):
    def test_metric_crs_choice(self):
        from zero2h3forge.core.h3_grid_engine import choose_metric_epsg

        self.assertEqual(choose_metric_epsg(27.14, 38.42), 5253)   # Izmir -> TUREF TM27
        self.assertEqual(choose_metric_epsg(32.85, 39.93), 5255)   # Ankara -> TM33
        self.assertEqual(choose_metric_epsg(43.0, 39.9), 5258)     # Kars/Igdir -> TM42
        self.assertEqual(choose_metric_epsg(13.40, 52.52), 32633)  # Berlin -> UTM 33N
        self.assertEqual(choose_metric_epsg(-46.63, -23.55), 32723)  # Sao Paulo -> UTM 23S
        self.assertEqual(choose_metric_epsg(179.9, 10.0), 32660)
        self.assertEqual(choose_metric_epsg(0.0, 89.0), 32661)

    def test_k_ring_sizes(self):
        ids, _ = _grid(2000.0, 2000.0)
        for k, expected in ((1, 6), (2, 18), (3, 36)):
            nbrs = build_neighbor_lists(ids, k=k)
            self.assertEqual(max(len(nb) for nb in nbrs), expected)

    def test_bh_fdr(self):
        from zero2h3forge.core.spatial_stats import bh_fdr_threshold, classify_hotspots_fdr

        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
        # Classic BH example: only the two smallest survive at q = 0.05
        self.assertAlmostEqual(bh_fdr_threshold(p, 0.05), 0.008)
        self.assertEqual(bh_fdr_threshold([0.5, 0.9], 0.05), 0.0)

        rng = np.random.default_rng(0)
        z_null = rng.normal(0.0, 1.0, 5000)
        p_null = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z_null])
        naive = classify_hotspots(z_null)
        fdr = classify_hotspots_fdr(z_null, p_null)
        self.assertGreater(int(np.sum(naive != 0)), 400)  # ~10% false positives
        self.assertLess(int(np.sum(fdr != 0)), 20)         # FDR removes almost all

    def test_wowa_properties(self):
        from zero2h3forge.core.mcda_composer import compose_owa, compose_wlc, owa_orness

        rng = np.random.default_rng(9)
        X = rng.uniform(0.0, 100.0, size=(200, 4))
        w = [40.0, 30.0, 20.0, 10.0]
        np.testing.assert_allclose(compose_owa(X, w, alpha=1.0), compose_wlc(X, w), atol=1e-9)
        lo = compose_owa(X, w, alpha=3.0)
        mid = compose_wlc(X, w)
        hi = compose_owa(X, w, alpha=0.3)
        self.assertTrue(np.all(lo <= mid + 1e-9))
        self.assertTrue(np.all(mid <= hi + 1e-9))
        self.assertTrue(np.all(lo >= X.min(axis=1) - 1e-9))
        self.assertTrue(np.all(hi <= X.max(axis=1) + 1e-9))
        self.assertAlmostEqual(owa_orness(1.0), 0.5)
        # Equal weights, alpha=2 on [100, 0]: Q(.5)=.25 -> 0.25*100
        self.assertAlmostEqual(float(compose_owa(np.array([[100.0, 0.0]]), [50, 50], alpha=2.0)[0]), 25.0)

    def test_jenks_recovers_clusters(self):
        from zero2h3forge.core.mcda_composer import jenks_breaks

        rng = np.random.default_rng(4)
        centers = [5.0, 25.0, 50.0, 70.0, 95.0]
        vals = np.concatenate([rng.normal(c, 1.0, 300) for c in centers])
        br = jenks_breaks(vals, 5)
        self.assertEqual(len(br), 6)
        for lo_c, hi_c, b in zip(centers[:-1], centers[1:], br[1:5]):
            self.assertTrue(lo_c < b < hi_c, (lo_c, b, hi_c))
        cids, _, breaks = classify_scores_5_bands(vals, method="jenks")
        self.assertEqual(len(set(cids)), 5)
