# -*- coding: utf-8 -*-
"""Pure-Python unit tests for 02H3Forge (zero2h3forge) core engines and Qt6 guards."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = PLUGIN_ROOT.parent
if str(PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGINS_DIR))

from zero2h3forge.core.h3_grid_engine import (  # noqa: E402
    DEFAULT_H3_RESOLUTION,
    H3_RESOLUTION_SPECS,
    circumradius_for_resolution,
    generate_metric_hex_cells,
    get_h3_neighbors,
    sanitize_field_slug,
)
from zero2h3forge.core.mcda_composer import (  # noqa: E402
    ahp_weights_from_matrix,
    apply_h3_spatial_smoothing,
    classify_scores_5_bands,
    compose_owa,
    compose_topsis,
    compose_vikor,
    compose_wlc,
    compute_local_xai_drivers,
    compute_montecarlo_stability,
    critic_weights_pct,
    entropy_weights_pct,
    pca_weights_pct,
    rebalance_coupled_weights,
)
from zero2h3forge.core.normalizer import normalize_array  # noqa: E402
from zero2h3forge.core.proximity_rings import (  # noqa: E402
    MULTI_RING_PRESETS,
    evaluate_graduated_rings,
)


class TestPureH3ForgeCore(unittest.TestCase):
    def test_01_h3_resolution_specs_and_metric_hex_fallback(self):
        self.assertEqual(DEFAULT_H3_RESOLUTION, 10)
        self.assertAlmostEqual(float(H3_RESOLUTION_SPECS[10]["area_ha"]), 1.505, places=2)
        self.assertAlmostEqual(float(H3_RESOLUTION_SPECS[11]["edge_m"]), 24.91, places=1)

        r10 = circumradius_for_resolution(10)
        self.assertGreater(r10, 60.0)
        self.assertLess(r10, 85.0)

        cells = generate_metric_hex_cells(0.0, 0.0, 500.0, 500.0, resolution=10)
        self.assertGreater(len(cells), 10)
        ids = [cid for cid, _ in cells]
        adj = get_h3_neighbors(ids)
        self.assertTrue(any(len(nbrs) >= 3 for nbrs in adj.values()))
        self.assertEqual(sanitize_field_slug("Güç Nakil Hattı (m)", "raw"), "raw_guc_nakil_hatti_m")

    def test_02_graduated_multiring_buffers(self):
        rings = MULTI_RING_PRESETS["Power Transmission Line (0-100, 100-250, 250-500 m)"]
        dists = [15.0, 100.0, 180.0, 350.0, 750.0]
        scores, excl = evaluate_graduated_rings(dists, rings, outside_score=0.0)
        np.testing.assert_allclose(scores, [100.0, 100.0, 70.0, 35.0, 0.0])
        self.assertFalse(np.any(excl))

        excl_rings = MULTI_RING_PRESETS["Power Line + Strict Safety Exclusion (<50m Excluded)"]
        scores2, excl2 = evaluate_graduated_rings([25.0, 120.0, 600.0], excl_rings, outside_score=0.0)
        self.assertTrue(bool(excl2[0]))
        self.assertFalse(bool(excl2[1]))
        self.assertEqual(float(scores2[0]), 100.0)
        self.assertEqual(float(scores2[2]), 0.0)

    def test_03_normalization_benefit_vs_cost_and_fuzzy(self):
        raw = [0.0, 25.0, 50.0, 75.0, 100.0]
        ben = normalize_array(raw, polarity="benefit", method="minmax")
        cost = normalize_array(raw, polarity="cost", method="minmax")
        np.testing.assert_allclose(ben, [0.0, 25.0, 50.0, 75.0, 100.0])
        np.testing.assert_allclose(cost, [100.0, 75.0, 50.0, 25.0, 0.0])

        sig_b = normalize_array(raw, polarity="benefit", method="fuzzy_sigmoid", mid=50.0, spread=10.0)
        sig_c = normalize_array(raw, polarity="cost", method="fuzzy_sigmoid", mid=50.0, spread=10.0)
        self.assertAlmostEqual(float(sig_b[2]), 50.0, places=3)
        np.testing.assert_allclose(sig_b + sig_c, np.full(5, 100.0), atol=1e-6)

        gauss = normalize_array([30.0, 50.0, 70.0], polarity="benefit", method="fuzzy_gaussian", mid=50.0, spread=10.0)
        self.assertAlmostEqual(float(gauss[1]), 100.0, places=4)
        self.assertAlmostEqual(float(gauss[0]), float(gauss[2]), places=4)

    def test_04_coupled_weight_rebalancer_with_locks(self):
        w0 = [25.0, 25.0, 25.0, 25.0]
        # Increase criterion 0 from 25% to 55% -> remaining 45% split equally (15%, 15%, 15%)
        w1 = rebalance_coupled_weights(w0, 0, 55.0)
        self.assertAlmostEqual(sum(w1), 100.0, places=2)
        np.testing.assert_allclose(w1, [55.0, 15.0, 15.0, 15.0], atol=0.05)

        # Lock criterion 1 at 30%, adjust criterion 0 to 50% -> criteria 2 & 3 split remaining 20%
        w_locked = [20.0, 30.0, 25.0, 25.0]
        w2 = rebalance_coupled_weights(w_locked, 0, 50.0, locked_flags=[False, True, False, False])
        self.assertAlmostEqual(sum(w2), 100.0, places=2)
        self.assertAlmostEqual(w2[1], 30.0, places=2)
        self.assertAlmostEqual(w2[0], 50.0, places=2)
        self.assertAlmostEqual(w2[2] + w2[3], 20.0, places=2)

    def test_05_objective_and_ahp_weight_engines(self):
        rng = np.random.default_rng(123)
        mat = rng.uniform(0.0, 100.0, size=(40, 4))
        w_critic, sigma, conflict = critic_weights_pct(mat)
        self.assertEqual(len(w_critic), 4)
        self.assertAlmostEqual(sum(w_critic), 100.0, places=1)
        self.assertEqual(sigma.shape, (4,))
        self.assertEqual(conflict.shape, (4,))

        w_ent = entropy_weights_pct(mat)
        self.assertAlmostEqual(sum(w_ent), 100.0, places=1)

        w_pca = pca_weights_pct(mat)
        self.assertAlmostEqual(sum(w_pca), 100.0, places=1)

        ahp_m = [
            [1.0, 3.0, 5.0],
            [1.0 / 3.0, 1.0, 2.0],
            [1.0 / 5.0, 0.5, 1.0],
        ]
        w_ahp, lam, ci, cr = ahp_weights_from_matrix(ahp_m)
        self.assertAlmostEqual(sum(w_ahp), 100.0, places=1)
        self.assertGreater(w_ahp[0], w_ahp[1])
        self.assertGreater(w_ahp[1], w_ahp[2])
        self.assertLessEqual(cr, 0.10)
        self.assertGreaterEqual(lam, 3.0)
        self.assertGreaterEqual(ci, 0.0)

    def test_06_compose_wlc_owa_topsis_vikor_and_xai(self):
        mat = np.array([
            [90.0, 80.0, 70.0],
            [50.0, 50.0, 50.0],
            [10.0, 20.0, 30.0],
        ])
        weights = [50.0, 30.0, 20.0]
        mask = [1, 1, 0]

        wlc = compose_wlc(mat, weights, constraint_mask=mask)
        self.assertAlmostEqual(float(wlc[0]), 83.0, places=2)
        self.assertAlmostEqual(float(wlc[1]), 50.0, places=2)
        self.assertEqual(float(wlc[2]), 0.0)

        owa_p = compose_owa(mat, weights, attitude="pessimistic")
        owa_o = compose_owa(mat, weights, attitude="optimistic")
        self.assertLess(float(owa_p[0]), float(owa_o[0]))

        topsis = compose_topsis(mat, weights)
        self.assertGreater(float(topsis[0]), float(topsis[1]))
        self.assertGreater(float(topsis[1]), float(topsis[2]))

        vikor = compose_vikor(mat, weights)
        self.assertAlmostEqual(float(vikor[0]), 100.0, places=2)
        self.assertAlmostEqual(float(vikor[2]), 0.0, places=2)

        drivers, shares = compute_local_xai_drivers(
            mat, weights, ["norm_power_line", "norm_slope", "norm_pop"]
        )
        self.assertEqual(drivers[0], "power_line")
        self.assertGreater(float(shares[0]), 50.0)

        mc_mean, mc_std, rob = compute_montecarlo_stability(mat, weights, constraint_mask=mask)
        self.assertEqual(float(rob[2]), 0.0)
        self.assertLessEqual(float(rob[0]), float(mc_mean[0]))

        cids, clbls, _ = classify_scores_5_bands([10.0, 35.0, 55.0, 75.0, 95.0], theme="risk")
        self.assertEqual(cids, [1, 2, 3, 4, 5])
        self.assertEqual(clbls[-1], "Critical")

        cells = generate_metric_hex_cells(0.0, 0.0, 300.0, 300.0, resolution=10)
        h3_ids = [cid for cid, _ in cells[:3]]
        smoothed = apply_h3_spatial_smoothing(np.array([100.0, 50.0, 20.0]), h3_ids, alpha=0.2)
        self.assertEqual(smoothed.shape, (3,))

    def test_07_qt6_enum_and_metadata_safety_guard(self):
        # 1. Verify metadata.txt has no literal '%' (TRAPS.md §1.5)
        meta_text = (PLUGIN_ROOT / "metadata.txt").read_text(encoding="utf-8")
        self.assertNotIn("%", meta_text, "metadata.txt must not contain literal '%' characters")
        self.assertIn("02H3Forge", meta_text)
        self.assertIn("H3", meta_text)

        # 2. Verify no bare crashing Qt.* enums in plugin Python sources (TRAPS.md §3.1b)
        forbidden_bare_qt = re.compile(
            r"\bQt\.(LeftButton|RightButton|MiddleButton|SolidLine|DashLine|NoPen|"
            r"SolidPattern|NoBrush|Checked|Unchecked|Horizontal|Vertical|"
            r"RightDockWidgetArea|LeftDockWidgetArea|AlignCenter|ItemIsEnabled|ItemIsUserCheckable)\b"
        )
        for py_file in PLUGIN_ROOT.rglob("*.py"):
            if py_file.name == "test_pure_mcda_h3.py":
                continue
            src = py_file.read_text(encoding="utf-8")
            match = forbidden_bare_qt.search(src)
            self.assertIsNone(
                match,
                f"Unscoped Qt enum {match.group(0) if match else ''} found in {py_file.name}",
            )


if __name__ == "__main__":
    unittest.main()
