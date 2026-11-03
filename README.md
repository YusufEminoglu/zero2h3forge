<div align="center">

<img src="icons/icon.png" width="96" alt="02H3Forge icon"/>

# 02H3Forge — H3 Hexagonal Risk, Vulnerability and Suitability Studio

**Right-docked Uber H3 hexagonal grid generator, graduated multi-ring buffer scorer, multi-source spatial aggregator, and MCDA risk/suitability overlay studio for QGIS.**

[![QGIS](https://img.shields.io/badge/QGIS-3.28%2B%20%7C%204.x-93b023?logo=qgis&logoColor=white)](https://github.com/YusufEminoglu/zero2h3forge)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-orange)](LICENSE)

</div>

---

**02H3Forge** (`zero2h3forge`) is a right-docked vector Multi-Criteria Decision Analysis (MCDA) studio for QGIS that uses the **Uber H3 Discrete Global Grid System (DGGS)** hexagonal tessellation as a unified spatial data frame for **Vulnerability, Hazard, Risk, Susceptibility, Sensitivity, and Land Suitability** modeling.

## Key Capabilities

1. **Stage 0 — H3 Grid & Graduated Multi-Ring Buffer Prep**
   - Generate Uber H3 hexagonal grids across any study area polygon, layer extent, or canvas extent (`Res 7` to `Res 12`).
   - Defaults to **H3 Resolution 10 (~1.50 ha area, ~65.9 m edge)** for neighborhood/master-plan analysis, with **H3 Resolution 11 (~0.21 ha area, ~24.9 m edge)** for micro-scale ~20 m grids.
   - Build graduated multi-ring proximity buffers (e.g., `0–100 m`, `100–250 m`, `250–500 m`, `> 500 m` around high-voltage power transmission lines, active faults, flood channels, or roads) to assign critical threshold scores (`0–100`) and optional hard exclusion masks (`constraint_mask = 0`).
2. **Stage 1 — Multi-Source Spatial Aggregation (`Join by Location` & `Zonal Stats`)**
   - Aggregate **Points** (`count`, `density`, `sum`, `mean`, `min`, `max`, `median`, `nearest_distance`), **Lines** (`total_length`, `density_m_ha`, `weighted_mean`, `nearest_distance`), **Polygons** (`area_share_pct`, `area_weighted_mean`, `area_weighted_sum`, `count`), and **Rasters** (`mean`, `median`, `min`, `max`, `std`) directly into H3 hexagon attributes.
3. **Stage 2 — Criterion Normalization (`0–100`) with Benefit / Cost Polarity**
   - Transform raw criteria to a common `0–100` scale using **Benefit (`+`, higher raw = higher target score)** or **Cost (`-`, lower raw = higher target score)** polarity.
   - Supports **Min-Max Linear**, **Robust Percentile (`P2–P98`)**, **Fuzzy Linear (`[low, high]`)**, **Fuzzy Sigmoid (`mid`, `spread`)**, **Fuzzy Gaussian (`optimum`, `spread`)**, and **Direct Multi-Ring Scores**.
4. **Stage 3 — Coupled Auto-Balancing Weight Sliders (`Sum = 100%`) & Overlay**
   - Interactive sliders and spinboxes that automatically rebalance unlocked criteria weights so the total always equals `100.0%`.
   - One-click objective and expert weight calculators: **Equal (`1/N`)**, **CRITIC**, **Shannon Entropy**, **PCA**, and **Interactive AHP Pairwise Matrix** ($CR \le 0.10$).
   - Composite engines: **WLC (Weighted Linear Combination)**, **OWA (Ordered Weighted Averaging)**, **TOPSIS**, and **VIKOR**, plus optional **H3 1-Ring Spatial Contagion Smoothing**.
5. **Stage 4 — Explainability (XAI), Uncertainty & Thematic Styling**
   - Identifies each hexagon's primary local driver (`top_driver` and `driver_share %`), optional Monte Carlo weight sensitivity (`mc_std`, `rob_score`), 5-class graduated symbology, and HTML/CSV executive reports.

## Documentation & Links

- **Homepage:** [geophilo.com/zero2h3forge/](https://geophilo.com/zero2h3forge/)
- **Reference Manual:** [geophilo.com/zero2h3forge/manual.html](https://geophilo.com/zero2h3forge/manual.html)

## License & Author

- **Author:** **Geophilo** (<yusuf.eminoglu@deu.edu.tr>)
- **License:** GNU General Public License v3.0 or later ([GPL-3.0-or-later](LICENSE))
