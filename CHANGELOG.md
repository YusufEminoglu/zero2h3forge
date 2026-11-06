# Changelog

All notable changes to **02H3Forge - H3 Hexagonal Risk, Vulnerability and Suitability Studio** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-11-06

### Fixed
- **Raster zonal statistics found no pixels.** The private memory layer was created without `crs=` in its URI, so its provider stayed in EPSG:4326 and `QgsZonalStatistics` silently matched nothing; every hexagon fell back to the 13-point stencil (a `sum` of 13 per cell, e.g. 6.5× too low at Res 8 with a 100 m raster). The layer is now built with the hexagon CRS, failures and the share of stencil cells are written to the run log, and a stencil `sum` is scaled by cell area / pixel area.
- **Scattered study areas lost native H3 cells.** Native candidates were generated for the boundary's bounding box, which for a dispersed area exceeds the cell budget and dropped to an enlarged metric grid. Cells are now filled from the boundary polygons themselves, so the budget counts real cells.
- Enlarged (coarsened) cells no longer claim the requested resolution: `h3_res` is left empty. `zero2h3forge:h3grid` stops in that case unless the new `ALLOW_COARSE` parameter is set.

### Added
- `tests/smoke_raster_grid.py`: value checks on a constant raster (pixel-count sums, mean = 1, sub-pixel area shares), a scattered-area grid test and the coarsening guard.

## [0.3.0] - 2026-11-04

A criterion-centric, hierarchical redesign of the studio. Existing v0.2 layers keep working: their `norm_*` fields are registered as criteria automatically, and v0.2 scenario files load as models.

### Added
- **Model** (`core/model.py`): one JSON-serialisable definition of the whole analysis — criteria (source → raw value → value function → group), groups with weights and engines, rule-based constraints and run options. Stored on the layer (survives GeoPackage export), saved/loaded as `*.h3model.json`.
- **Criterion library** (tab 2): sources are vector aggregation, exact raster zonal statistics, distance to features or an existing field; value functions are linear, robust P2–P98, thresholds, sigmoid, Gaussian optimum, graduated distance rings (presets included) or already-0–100. A live curve preview overlays the distribution of hexagon values from the last run.
- **Constraints as rules**: *closer than X m to*, *inside polygons*, *outside polygons*, or *attribute rule* on any column (e.g. `raw_slope > 30`). The report lists how many hexagons each rule removed.
- **Hierarchical models**: single index, **Risk = Hazard × Exposure × Vulnerability** and suitability templates, custom groups. Each group has its own engine (WLC, WOWA with α, TOPSIS, VIKOR) and coupled weights with AHP/CRITIC/Entropy/PCA; groups combine by **weighted sum** or **weighted geometric mean** (non-compensatory). Outputs `idx_<group>`, `top_group`, `limit_group`.
- **Background execution**: layers are snapshotted on the main thread, the model runs in a `QgsTask` (progress bar, Cancel), results are written in one batch only when the run completes.
- **Raw-criterion cache**: unchanged sources are not recomputed — weight and value-function experiments re-run in seconds; *Recompute all* forces a refresh.
- **Hierarchical Monte Carlo and sensitivity** perturb group *and* criterion weights.
- **`zero2h3forge:runmodel`** Processing algorithm runs a model file on any H3 layer (Modeler / batch / PyQGIS).
- **HTML report**: *Model Structure* card with groups, sources, value functions, effective weights and constraint impact.
- Thread-safe compute kernels (`core/compute.py`) and a three-phase pipeline (`core/pipeline.py`: prepare → compute → apply) with 11 new pure tests and a QGIS pipeline smoke test (incl. a real `QgsTask` run).

### Changed
- The studio has four tabs (Study Area · Criteria · Weights · Results) and a single **⚡ Run Model** action replacing the five-stage flow and Auto-Forge. Multi-ring buffers are now a value function of a distance criterion; their former exclusion flag is a *closer than* constraint.
- Group weights are kept for empty groups so new groups and templates start with fair shares.

## [0.2.0] - 2026-11-03

### Added
- **⚡ Auto-Forge**: one-click pipeline that normalizes the checked Stage 2 fields, forges the composite map with the current Stage 3 weights and maps Gi* hotspots.
- **Spatial clustering engine** (`core/spatial_stats.py`): Getis-Ord Gi* hotspots (`gi_z`, `gi_p`, `hotspot`, `hotspot_lbl`; 90/95/99% bins) and Global Moran's I on the H3 1-ring topology. Hard-excluded hexagons are dropped so their forced zeros cannot fabricate cold spots. Includes a diverging hotspot renderer and a toggle back to composite classes.
- **Scenarios** (`core/scenario.py`): save/load weights, activity, locks and overlay settings as reproducible JSON.
- **Live grid-size preview** in Stage 0, with a warning before the cell budget forces coarsening; grids record their engine (`native-h3` / `metric-fallback`) and coarsening as layer properties.
- **QGIS Processing provider** (`h3forge_processing.py`): `zero2h3forge:h3grid`, `zero2h3forge:compositeoverlay` and `zero2h3forge:hotspots` for the Toolbox, Graphical Modeler, batch mode and PyQGIS. Inputs are never modified; outputs are auto-styled when loaded.
- **Weight sensitivity (one-at-a-time)**: each weight is nudged by ±10% / ±20% (others rebalanced) and the mean score shift and share of hexagons changing class are reported per criterion, in the dock and the HTML report. Auto-Forge runs it as its final step.
- **Hotspot rigour**: Gi* neighbourhood size is now selectable (H3 k-ring 1–6 grid steps, with the reach in metres shown) and Benjamini-Hochberg false-discovery-rate correction is on by default, removing the chance "hot spots" that thousands of simultaneous tests produce.
- **Fisher-Jenks natural breaks** as a third 5-class scheme (dock and Processing).
- **Save to GeoPackage**: persist the temporary grid with all criteria, results, styling and run provenance.
- **Continuous integration** (GitHub Actions): pure tests on Python 3.9/3.12 and end-to-end QGIS tests in the official QGIS 3.34 and latest images, with and without native `h3`; the plugin zip is built as a CI artifact.
- **Richer HTML report**: KPI cards, inline SVG score histogram and weight bars, Moran's I and hotspot table, and a run-parameters section for reproducibility.
- Wait cursor during long operations; 2-decimal weight spinboxes so totals display as exactly 100%.
- Stage pages scroll vertically, so the dock stays usable on short screens.
- `scripts/build_plugin_zip.py` builds and validates the plugins.qgis.org archive (honours `.zipignore`).
- New QGIS Processing smoke test `tests/smoke_processing.py`; native `h3` (v4) grid path verified end to end.
- New pure-Python test suite `tests/test_spatial_stats_and_v02.py` (Moran's I and Gi* checked against brute-force dense formulas) and v0.2 coverage in the QGIS smoke test.

### Changed
- **OWA is now Torra's weighted OWA (WOWA) with a RIM quantifier `Q(r) = r^α`** instead of an ad-hoc product of criterion and fixed order weights. The risk attitude α is adjustable (α > 1 precautionary, α = 1 identical to WLC, α < 1 opportunity; orness shown). Pessimistic/optimistic presets map to α = 2 / 0.5. OWA scores differ from v0.1.
- **Raster aggregation uses exact pixel-based zonal statistics** (`QgsZonalStatistics`, any CRS combination) instead of a 13-point sample per hexagon; the stencil remains only as a fallback for sub-pixel hexagons. Added a zonal *sum* statistic.
- Weights, activity and locks are preserved when criteria are added or refreshed; new criteria receive a fair 1/N share.
- Monte Carlo stability now perturbs weights through the selected overlay engine (OWA/TOPSIS/VIKOR), not always WLC.
- A `dist_*` field whose `ring_*` twin exists starts unchecked in Stage 2 to avoid double-counting.

### Fixed
- **Metric CRS selection**: outside Türkiye grids were built in Web Mercator (EPSG:3857), inflating distances by ~30% at 40° and ~100% at 60° latitude (areas ~70% / ~300%), which corrupted buffers, cell areas and densities. Grids now use the local UTM zone. Inside Türkiye every grid used TUREF TM27 (EPSG:5253) even ~17° from its central meridian; the nearest TUREF 3° zone (TM27…TM45) is now chosen. Projected metre-based inputs are kept as-is.
- Closing the plugin disconnected **every** slot on `QgsProject.layersAdded/layersRemoved`, including other plugins'; it now disconnects only its own.
- TOPSIS ideal/anti-ideal points and VIKOR best/worst values were computed including hard-excluded hexagons.
- H3 1-ring contagion smoothing pulled feasible hexagons toward the forced zero of excluded neighbors.
- Excluded hexagons reported a top XAI driver; they are now labelled `Excluded`.
- Normalizing `dist_x` and `ring_x` both wrote `norm_x`, silently overwriting one criterion; distances now map to `norm_dist_x`.
- Non-numeric ring or normalization parameters raised unhandled exceptions.
- Raster sampling probed outside the raster edge, spamming GDAL "access window out of range" errors.
- Ampersands in tab and group titles rendered as keyboard-mnemonic underscores.
## [0.1.1] - 2026-11-03

### Changed
- Aligned `metadata.txt` and documentation with Geophilo standards (`author=Geophilo`, structured workflow summary, and `geophilo.com` links).
- Updated `icons/icon.png` to a 512x512 3D isometric tactile studio icon with transparent background and zero margin (`bbox = (0, 0, 512, 512)`).
- Replaced unguarded polygon ring extraction in raster zonal sampling with `geometry.vertices()` for clean multi-part support.

## [0.1.0] - 2026-11-03

### Added
- Right-docked 5-stage Uber H3 spatial synthesis and MCDA studio (`O2H3ForgeStudioDock`) with PlanX Studio styling.
- Stage 0: Uber H3 Hexagonal Grid Generator supporting resolutions `7` to `12` (default `Res 10` ≈ `1.50 ha`, `Res 11` ≈ `24.9 m` edge length) with native `h3` (v4/v3) support and zero-dependency metric hexagonal grid fallback.
- Stage 0: Graduated Multi-Ring Proximity Buffer Scorer (`0–100m`, `100–250m`, `250–500m`, `>500m`, etc.) with shortest Euclidean distance (`dist_*`), threshold scoring (`ring_*`), and hard exclusion masks (`constraint_mask`).
- Stage 1: Multi-source spatial aggregator transferring Point, Line, Polygon, and Raster layers onto H3 hexagons via spatial index joins and NumPy zonal statistics.
- Stage 2: Criterion normalizer (`0–100`) with explicit Benefit (`+`) and Cost (`-`) polarity across Min-Max Linear, Robust Percentile (`P2–P98`), Fuzzy Linear, Fuzzy Sigmoid, Fuzzy Gaussian, and Direct Ring methods.
- Stage 3: Coupled auto-balancing weight sliders (`sum = 100%`) with per-criterion lock (`🔒`), one-click objective/expert weighting (Equal, CRITIC, Shannon Entropy, PCA, and interactive AHP with Saaty/Alonso-Lamata `CR <= 0.10` verification), and WLC, OWA, TOPSIS, and VIKOR composition engines with optional H3 1-ring spatial contagion smoothing.
- Stage 4: Local Explainable AI (`top_driver`, `driver_share`), Monte Carlo weight sensitivity (`mc_std`, `rob_score`), 5-class graduated/categorized cartographic styling (Risk/Vulnerability Red-Orange, Suitability Green, Susceptibility Viridis/Magma), and HTML/CSV executive reporting.
