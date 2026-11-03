<div align="center">

<img src="icons/icon.png" width="96" alt="02H3Forge icon"/>

# 02H3Forge — H3 Hexagonal Risk, Vulnerability and Suitability Studio

**Criterion-centric, hierarchical multi-criteria decision analysis (MCDA) on Uber H3 hexagons for QGIS.**

[![QGIS](https://img.shields.io/badge/QGIS-3.28%2B%20%7C%204.x-93b023?logo=qgis&logoColor=white)](https://github.com/YusufEminoglu/zero2h3forge)
[![tests](https://github.com/YusufEminoglu/zero2h3forge/actions/workflows/tests.yml/badge.svg)](https://github.com/YusufEminoglu/zero2h3forge/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-orange)](LICENSE)

</div>

---

**02H3Forge** (`zero2h3forge`) turns heterogeneous vector and raster data into explainable **risk, vulnerability, hazard, susceptibility and land-suitability** maps on the **Uber H3** discrete global grid. An analysis is a single, reproducible **model**: criteria with their data sources and value functions, groups, weights, constraints and run options. Press **⚡ Run Model** and everything is computed in the background.

## Workflow

| Tab | What you do |
|---|---|
| **1 · Study Area** | Tessellate a boundary, layer extent or the canvas with H3 cells (Res 6–12; native `h3` index when installed). A live estimate shows the cell count and the true-scale metric CRS (local UTM zone, or the nearest TUREF 3° zone in Türkiye). Save the grid to GeoPackage to keep it. |
| **2 · Criteria** | Build a **criterion library**. Each criterion = **source** (vector aggregation, exact raster zonal statistic, distance to features, or an existing field) → **value function** (linear, robust P2–P98, thresholds, sigmoid, Gaussian optimum, or graduated distance rings) → **group**. A live curve shows the function over the distribution of your hexagons. **Constraints** are separate rules: *closer than X m to*, *inside / outside polygons*, or *attribute rule* (e.g. `raw_slope > 30`). |
| **3 · Weights** | Choose a structure — **single index**, **Risk = Hazard × Exposure × Vulnerability** or **suitability** groups — and weight groups and criteria with coupled sliders (always 100 %), **AHP** (CR ≤ 0.10), **CRITIC**, **Entropy** or **PCA**. Each group has its own engine (**WLC, WOWA with risk attitude α, TOPSIS, VIKOR**); groups combine by **weighted sum** (compensatory) or **weighted geometric mean** (non-compensatory: no hazard → no risk). |
| **4 · Results** | Class areas with local explainability (`top_driver`, `top_group`, `limit_group`), group sub-indices (`idx_*`), Monte Carlo uncertainty over group *and* criterion weights (`mc_std`, `rob_score`), **Getis-Ord Gi\*** hotspots with k-ring neighbourhoods and **FDR** correction, **Global Moran's I**, one-at-a-time **weight sensitivity**, map styling for every output, CSV and an executive **HTML report**. |

### Why it is built this way

- **Reproducible:** the model is stored on the layer (and in GeoPackages) and can be saved as JSON, re-run on a new grid, or batch-processed.
- **Fast iteration:** raw criteria are cached per source signature — changing weights or value functions re-runs in seconds; only edited sources are recomputed.
- **Responsive:** computation runs in a `QgsTask` with progress and cancel; layers are only written when a run completes.
- **Statistically honest:** constraint-aware ideals, smoothing and hotspots; multiple-testing control; uncertainty and sensitivity reported next to every map.

### Processing Toolbox & Graphical Modeler

Four algorithms under **02H3Forge → H3 Hexagonal MCDA**:

| Algorithm | Purpose |
|---|---|
| `zero2h3forge:h3grid` | Generate an H3 grid for a study area |
| `zero2h3forge:runmodel` | Run a saved 02H3Forge model (JSON) on an H3 layer |
| `zero2h3forge:compositeoverlay` | Quick flat overlay of existing 0–100 fields |
| `zero2h3forge:hotspots` | Gi\* hotspots + Moran's I for any numeric field |

```python
import processing

grid = processing.run("zero2h3forge:h3grid", {
    "INPUT": study_area, "RESOLUTION": 4, "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
risk = processing.run("zero2h3forge:runmodel", {
    "INPUT": grid, "MODEL": "/path/flood_risk.h3model.json", "OUTPUT": "TEMPORARY_OUTPUT"})["OUTPUT"]
```

`RESOLUTION` is an index into resolutions 6–12 (4 = Res 10). Installing the `h3` Python package enables authentic Uber H3 cell IDs; without it an H3-scale metric grid is used.

## Testing

```bash
python3 -m unittest tests.test_pure_mcda_h3 tests.test_spatial_stats_and_v02 tests.test_model_pipeline
```

```bash
QT_QPA_PLATFORM=offscreen python3 tests/smoke_pipeline.py
```

```bash
QT_QPA_PLATFORM=offscreen python3 tests/smoke_dock.py
```

```bash
QT_QPA_PLATFORM=offscreen python3 tests/smoke_processing.py
```

The smoke tests need a Python interpreter with the QGIS bindings; CI runs them in the official QGIS 3.34 and latest images. Build the plugins.qgis.org archive with:

```bash
python3 scripts/build_plugin_zip.py
```

## Documentation & Links

- **Homepage:** [geophilo.com/zero2h3forge/](https://geophilo.com/zero2h3forge/)
- **Reference Manual:** [geophilo.com/zero2h3forge/manual.html](https://geophilo.com/zero2h3forge/manual.html)

## License & Author

- **Author:** **Geophilo** (<yusuf.eminoglu@deu.edu.tr>)
- **License:** GNU General Public License v3.0 or later ([GPL-3.0-or-later](LICENSE))
