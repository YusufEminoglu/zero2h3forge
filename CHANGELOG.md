# Changelog

All notable changes to **02H3Forge - H3 Hexagonal Risk, Vulnerability and Suitability Studio** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-03

### Added
- Right-docked 5-stage Uber H3 spatial synthesis and MCDA studio (`O2H3ForgeStudioDock`) with PlanX Studio styling.
- Stage 0: Uber H3 Hexagonal Grid Generator supporting resolutions `7` to `12` (default `Res 10` ≈ `1.50 ha`, `Res 11` ≈ `24.9 m` edge length) with native `h3` (v4/v3) support and zero-dependency metric hexagonal grid fallback.
- Stage 0: Graduated Multi-Ring Proximity Buffer Scorer (`0–100m`, `100–250m`, `250–500m`, `>500m`, etc.) with shortest Euclidean distance (`dist_*`), threshold scoring (`ring_*`), and hard exclusion masks (`constraint_mask`).
- Stage 1: Multi-source spatial aggregator transferring Point, Line, Polygon, and Raster layers onto H3 hexagons via spatial index joins and NumPy zonal statistics.
- Stage 2: Criterion normalizer (`0–100`) with explicit Benefit (`+`) and Cost (`-`) polarity across Min-Max Linear, Robust Percentile (`P2–P98`), Fuzzy Linear, Fuzzy Sigmoid, Fuzzy Gaussian, and Direct Ring methods.
- Stage 3: Coupled auto-balancing weight sliders (`sum = 100%`) with per-criterion lock (`🔒`), one-click objective/expert weighting (Equal, CRITIC, Shannon Entropy, PCA, and interactive AHP with Saaty/Alonso-Lamata `CR <= 0.10` verification), and WLC, OWA, TOPSIS, and VIKOR composition engines with optional H3 1-ring spatial contagion smoothing.
- Stage 4: Local Explainable AI (`top_driver`, `driver_share`), Monte Carlo weight sensitivity (`mc_std`, `rob_score`), 5-class graduated/categorized cartographic styling (Risk/Vulnerability Red-Orange, Suitability Green, Susceptibility Viridis/Magma), and HTML/CSV executive reporting.
