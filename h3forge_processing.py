# -*- coding: utf-8 -*-
"""QGIS Processing provider for 02H3Forge.

Exposes the core engines as Processing algorithms so they can be chained in the
Graphical Modeler, run in batch mode, or called from PyQGIS::

    processing.run("zero2h3forge:h3grid", {...})
    processing.run("zero2h3forge:compositeoverlay", {...})
    processing.run("zero2h3forge:hotspots", {...})

Every algorithm works on an in-memory copy of its input, so source layers are
never modified.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from qgis.core import (
    Qgis,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingProvider,
)
from qgis.PyQt.QtGui import QIcon

from . import PLUGIN_VERSION
from .core.h3_grid_engine import DEFAULT_H3_RESOLUTION, H3_RESOLUTION_SPECS, build_h3_grid_layer
from .core.mcda_composer import COMPOSE_METHODS, run_h3_overlay_on_layer
from .core.spatial_stats import run_hotspot_analysis_on_layer
from .core.styler_and_report import (
    THEME_OPTIONS,
    apply_h3_graduated_style,
    apply_h3_hotspot_style,
)

PROVIDER_ID = "zero2h3forge"
ICON_PATH = os.path.join(os.path.dirname(__file__), "icons", "icon.png")


def _first(*getters: Callable[[], object]):
    """Return the first enum that exists (QGIS 3.28 unscoped vs 3.36+/4.x scoped)."""
    for getter in getters:
        try:
            return getter()
        except AttributeError:
            continue
    raise AttributeError("No compatible QGIS enum found")


POLYGON_SOURCE = _first(
    lambda: Qgis.ProcessingSourceType.VectorPolygon,
    lambda: QgsProcessing.TypeVectorPolygon,
)
NUMERIC_FIELD = _first(
    lambda: Qgis.ProcessingFieldParameterDataType.Numeric,
    lambda: QgsProcessingParameterField.Numeric,
)
DOUBLE_NUMBER = _first(
    lambda: Qgis.ProcessingNumberParameterType.Double,
    lambda: QgsProcessingParameterNumber.Double,
)
FAST_INSERT = _first(
    lambda: QgsFeatureSink.Flag.FastInsert,
    lambda: QgsFeatureSink.FastInsert,
)

RESOLUTIONS: List[int] = sorted(H3_RESOLUTION_SPECS)
COVERAGE_MODES = [
    ("intersect", "Hexagons intersecting the boundary (full coverage)"),
    ("centroid", "Hexagon centroid inside the boundary"),
    ("bbox", "Full bounding box"),
]
CLASS_MODES = [
    ("equal", "Equal interval (0-20-40-60-80-100)"),
    ("quantile", "Quantile (5 bands)"),
    ("jenks", "Natural breaks (Fisher-Jenks)"),
]


def parse_weights(text: str, n: int) -> List[float]:
    """Parse ``"40, 30, 30"`` (comma/semicolon/space separated) into ``n`` weights.

    An empty string means equal weights. Weights are relative: they are
    rescaled to sum to 100.
    """
    raw = (text or "").replace(";", ",").replace(" ", ",")
    tokens = [t for t in raw.split(",") if t.strip()]
    if not tokens:
        return [100.0 / n] * n
    try:
        vals = [float(t) for t in tokens]
    except ValueError as exc:
        raise QgsProcessingException(f"Weights must be numbers, got '{text}'.") from exc
    if len(vals) != n:
        raise QgsProcessingException(
            f"{len(vals)} weights were given for {n} criteria; provide one weight per criterion."
        )
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise QgsProcessingException("Weights must be non-negative and not all zero.")
    total = sum(vals)
    return [v / total * 100.0 for v in vals]


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Applies 02H3Forge symbology and provenance to the layer loaded from an output."""

    _instances: List["_StylePostProcessor"] = []  # keep Python wrappers alive

    def __init__(self, kind: str, theme: str, breaks: Optional[List[float]], props: Dict[str, object]):
        super().__init__()
        self.kind = kind
        self.theme = theme
        self.breaks = breaks
        self.props = props

    @classmethod
    def create(cls, kind: str, theme: str = "risk", breaks=None, props=None) -> "_StylePostProcessor":
        inst = cls(kind, theme, breaks, dict(props or {}))
        cls._instances.append(inst)
        return inst

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802 (QGIS API)
        for key, val in self.props.items():
            layer.setCustomProperty(key, val)
        if self.kind == "hotspot":
            apply_h3_hotspot_style(layer)
        else:
            apply_h3_graduated_style(layer, "comp_score", theme=self.theme, breaks=self.breaks)


class _H3ForgeAlgorithm(QgsProcessingAlgorithm):
    def group(self):
        return "H3 Hexagonal MCDA"

    def groupId(self):  # noqa: N802
        return "h3mcda"

    def icon(self):
        return QIcon(ICON_PATH)

    def createInstance(self):  # noqa: N802
        return type(self)()

    @staticmethod
    def _materialize(source, feedback):
        """Copy a feature source into an editable memory layer."""
        return source.materialize(QgsFeatureRequest(), feedback)

    def _write_sink(self, parameters, context, layer, feedback):
        sink, dest_id = self.parameterAsSink(
            parameters, "OUTPUT", context, layer.fields(), layer.wkbType(), layer.crs()
        )
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, "OUTPUT"))
        total = max(1, layer.featureCount())
        for i, feat in enumerate(layer.getFeatures()):
            if feedback.isCanceled():
                break
            sink.addFeature(feat, FAST_INSERT)
            feedback.setProgress(80 + 20 * i / total)
        return dest_id

    @staticmethod
    def _attach_style(context, dest_id, layer, kind: str, theme: str = "risk", breaks=None) -> None:
        """Style the output when QGIS loads it, and carry the run provenance along."""
        if not context.willLoadLayerOnCompletion(dest_id):
            return
        props = {}
        for key in ("h3forge/last_run", "h3forge/hotspot", "h3forge/engine", "h3forge/model",
                    "h3forge/sensitivity"):
            val = layer.customProperty(key, None)
            if val is not None:
                props[key] = val
        details = context.layerToLoadOnCompletionDetails(dest_id)
        details.setPostProcessor(_StylePostProcessor.create(kind, theme, breaks, props))


class GenerateH3GridAlgorithm(_H3ForgeAlgorithm):
    def name(self):
        return "h3grid"

    def displayName(self):  # noqa: N802
        return "Generate H3 hexagonal grid"

    def shortHelpString(self):  # noqa: N802
        return (
            "Tessellates a study-area polygon layer with Uber H3 hexagons (native h3 index when "
            "the h3 Python package is installed, otherwise an H3-scale metric grid). Output fields: "
            "h3_id, h3_res, area_m2, area_ha, constraint_mask. Geographic inputs are projected to a "
            "true-scale metric CRS (nearest TUREF 3° zone in Türkiye, local UTM zone elsewhere). "
            "Native cells are filled from the study-area polygons themselves, so scattered areas "
            "stay at the requested resolution. If the cell budget is still exceeded the algorithm "
            "stops, unless 'Allow coarser cells' is ticked (h3_res is then left empty)."
        )

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            "INPUT", "Study area (polygon)", [POLYGON_SOURCE]
        ))
        self.addParameter(QgsProcessingParameterEnum(
            "RESOLUTION", "H3 resolution",
            options=[str(H3_RESOLUTION_SPECS[r]["label"]) for r in RESOLUTIONS],
            defaultValue=RESOLUTIONS.index(DEFAULT_H3_RESOLUTION),
        ))
        self.addParameter(QgsProcessingParameterEnum(
            "COVERAGE", "Coverage mode", options=[lbl for _, lbl in COVERAGE_MODES], defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            "NATIVE_H3", "Use native Uber H3 index when available", defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            "ALLOW_COARSE", "Allow coarser cells when the area exceeds the cell budget",
            defaultValue=False,
        ))
        self.addParameter(QgsProcessingParameterFeatureSink("OUTPUT", "H3 grid"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        source = self.parameterAsSource(parameters, "INPUT", context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, "INPUT"))
        res = RESOLUTIONS[self.parameterAsEnum(parameters, "RESOLUTION", context)]
        coverage = COVERAGE_MODES[self.parameterAsEnum(parameters, "COVERAGE", context)][0]
        native = self.parameterAsBool(parameters, "NATIVE_H3", context)
        allow_coarse = self.parameterAsBool(parameters, "ALLOW_COARSE", context)

        boundary = self._materialize(source, feedback)
        feedback.pushInfo(f"Building H3 grid at resolution {res} ({coverage})...")
        grid = build_h3_grid_layer(
            extent_rect=boundary.extent(),
            target_crs=boundary.crs(),
            resolution=res,
            boundary_layer=boundary,
            coverage_mode=coverage,
            prefer_native_h3=native,
        )
        feedback.setProgress(80)
        engine = grid.customProperty("h3forge/engine", "")
        feedback.pushInfo(f"{grid.featureCount():,} hexagons generated ({engine}).")
        if grid.customProperty("h3forge/coarsened", False):
            mean_ha = float(grid.customProperty("h3forge/mean_area_m2", 0.0)) / 10_000.0
            msg = (f"The study area exceeds the cell budget at resolution {res}: cells were "
                   f"enlarged to {mean_ha:.2f} ha (h3_res left empty). Use a coarser resolution "
                   "or a smaller study area for true resolution cells.")
            if not allow_coarse:
                raise QgsProcessingException(msg + " Enable 'Allow coarser cells' to accept them.")
            feedback.reportError(msg, fatalError=False)
        return {"OUTPUT": self._write_sink(parameters, context, grid, feedback)}


class CompositeOverlayAlgorithm(_H3ForgeAlgorithm):
    def name(self):
        return "compositeoverlay"

    def displayName(self):  # noqa: N802
        return "Composite MCDA overlay (WLC / OWA / TOPSIS / VIKOR)"

    def shortHelpString(self):  # noqa: N802
        return (
            "Combines 0-100 criterion fields (e.g. norm_* columns) of an H3 layer into a composite "
            "0-100 score with WLC, OWA, TOPSIS or VIKOR. Adds comp_score, class_id, class_label, "
            "top_driver, driver_share, mc_std and rob_score. Weights are relative values in the "
            "same order as the selected fields (e.g. '40, 35, 25'); leave empty for equal weights. "
            "Hexagons with constraint_mask = 0 are excluded when hard constraints are applied."
        )

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            "INPUT", "H3 layer", [POLYGON_SOURCE]
        ))
        self.addParameter(QgsProcessingParameterField(
            "CRITERIA", "Criterion fields (0-100)", parentLayerParameterName="INPUT",
            type=NUMERIC_FIELD, allowMultiple=True,
        ))
        self.addParameter(QgsProcessingParameterString(
            "WEIGHTS", "Weights (same order as fields, empty = equal)", defaultValue="", optional=True
        ))
        self.addParameter(QgsProcessingParameterEnum(
            "METHOD", "Overlay engine", options=[lbl for _, lbl in COMPOSE_METHODS], defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            "OWA_ALPHA", "OWA attitude alpha (Q(r)=r^alpha; >1 precautionary, <1 optimistic; OWA only)",
            type=DOUBLE_NUMBER, minValue=0.1, maxValue=10.0, defaultValue=2.0, optional=True,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            "THEME", "Map theme", options=[lbl for _, lbl in THEME_OPTIONS], defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            "CLASSIFY", "5-class breaks", options=[lbl for _, lbl in CLASS_MODES], defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            "ALPHA", "H3 1-ring contagion smoothing (alpha)", type=DOUBLE_NUMBER,
            minValue=0.0, maxValue=0.5, defaultValue=0.0,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            "CONSTRAINTS", "Apply hard constraints (constraint_mask)", defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            "MONTECARLO", "Monte Carlo weight stability (mc_std, rob_score)", defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterFeatureSink("OUTPUT", "Composite H3 map"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        source = self.parameterAsSource(parameters, "INPUT", context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, "INPUT"))
        # parameterAsStrings (QGIS 3.32+) replaces the deprecated parameterAsFields.
        if hasattr(self, "parameterAsStrings"):
            fields = self.parameterAsStrings(parameters, "CRITERIA", context)
        else:  # pragma: no cover - QGIS 3.28 / 3.30
            fields = self.parameterAsFields(parameters, "CRITERIA", context)
        if not fields:
            raise QgsProcessingException("Select at least one criterion field.")
        weights = parse_weights(self.parameterAsString(parameters, "WEIGHTS", context), len(fields))
        method = COMPOSE_METHODS[self.parameterAsEnum(parameters, "METHOD", context)][0]
        theme = THEME_OPTIONS[self.parameterAsEnum(parameters, "THEME", context)][0]
        class_mode = CLASS_MODES[self.parameterAsEnum(parameters, "CLASSIFY", context)][0]

        layer = self._materialize(source, feedback)
        feedback.pushInfo(
            "Weights: " + ", ".join(f"{f}={w:.1f}%" for f, w in zip(fields, weights))
        )
        stats = run_h3_overlay_on_layer(
            layer,
            criterion_fields=fields,
            weights_pct=weights,
            compose_method=method,
            apply_constraints=self.parameterAsBool(parameters, "CONSTRAINTS", context),
            spatial_smoothing_alpha=self.parameterAsDouble(parameters, "ALPHA", context),
            run_montecarlo=self.parameterAsBool(parameters, "MONTECARLO", context),
            classification_method=class_mode,
            theme=theme,
            owa_alpha=(
                self.parameterAsDouble(parameters, "OWA_ALPHA", context)
                if method.startswith("owa") and parameters.get("OWA_ALPHA") is not None
                else None
            ),
        )
        feedback.setProgress(80)
        feedback.pushInfo(
            f"{method.upper()}: mean={stats['mean_score']}, max={stats['max_score']}, "
            f"excluded={stats.get('excluded_count', 0)}, mean MC std={stats['mean_mc_std']}."
        )
        dest_id = self._write_sink(parameters, context, layer, feedback)
        self._attach_style(context, dest_id, layer, "composite", theme, stats.get("breaks"))
        return {"OUTPUT": dest_id}


class HotspotAlgorithm(_H3ForgeAlgorithm):
    def name(self):
        return "hotspots"

    def displayName(self):  # noqa: N802
        return "H3 hotspot analysis (Getis-Ord Gi* + Moran's I)"

    def shortHelpString(self):  # noqa: N802
        return (
            "Computes Getis-Ord Gi* z-scores and 90/95/99% hot/cold spot classes on the H3 1-ring "
            "neighbourhood, plus Global Moran's I (reported in the log). Requires an h3_id field. "
            "Adds gi_z, gi_p, hotspot (-3..3) and hotspot_lbl."
        )

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            "INPUT", "H3 layer", [POLYGON_SOURCE]
        ))
        self.addParameter(QgsProcessingParameterField(
            "FIELD", "Value field", parentLayerParameterName="INPUT", type=NUMERIC_FIELD,
            defaultValue="comp_score",
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            "RESPECT_MASK", "Drop hard-excluded hexagons (constraint_mask = 0)", defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            "K_RING", "Neighbourhood size (H3 k-ring, grid steps)",
            minValue=1, maxValue=6, defaultValue=1,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            "FDR", "Benjamini-Hochberg false discovery rate correction", defaultValue=True
        ))
        self.addParameter(QgsProcessingParameterFeatureSink("OUTPUT", "Hotspots"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        source = self.parameterAsSource(parameters, "INPUT", context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, "INPUT"))
        if source.fields().indexFromName("h3_id") < 0:
            raise QgsProcessingException(
                "The input has no 'h3_id' field. Generate the grid with 02H3Forge first."
            )
        field = self.parameterAsString(parameters, "FIELD", context) or "comp_score"
        layer = self._materialize(source, feedback)
        res = run_hotspot_analysis_on_layer(
            layer, value_field=field,
            respect_constraints=self.parameterAsBool(parameters, "RESPECT_MASK", context),
            k_ring=self.parameterAsInt(parameters, "K_RING", context),
            fdr=self.parameterAsBool(parameters, "FDR", context),
        )
        feedback.setProgress(80)
        moran = res["moran"]
        feedback.pushInfo(
            f"Global Moran's I = {moran.get('I')} (z = {moran.get('z')}, p = {moran.get('p')})."
        )
        for lbl, cnt in res["counts"].items():
            feedback.pushInfo(f"  {lbl}: {cnt}")
        dest_id = self._write_sink(parameters, context, layer, feedback)
        self._attach_style(context, dest_id, layer, "hotspot")
        return {"OUTPUT": dest_id, "MORAN_I": moran.get("I"), "MORAN_Z": moran.get("z")}


NO_THREADING = _first(
    lambda: Qgis.ProcessingAlgorithmFlag.NoThreading,
    lambda: QgsProcessingAlgorithm.FlagNoThreading,
)


class RunModelAlgorithm(_H3ForgeAlgorithm):
    """Run a complete 02H3Forge model file on an H3 layer."""

    def name(self):
        return "runmodel"

    def displayName(self):  # noqa: N802
        return "Run 02H3Forge model (criteria, constraints, groups)"

    def shortHelpString(self):  # noqa: N802
        return (
            "Executes a 02H3Forge model file (*.json saved from the studio: criteria with their "
            "sources and value functions, groups, weights, constraints and run options) on an H3 "
            "layer. Source layers are matched by layer id, then name in the current project, then "
            "by the file path stored in the model, so the same model can be re-run on new grids or "
            "batch-processed. Outputs every criterion column, group sub-indices, comp_score, "
            "classes, explainability, Monte Carlo uncertainty and hotspots."
        )

    def flags(self):
        # Source layers are resolved from the project, which must happen on the main thread.
        return super().flags() | NO_THREADING

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource("INPUT", "H3 layer", [POLYGON_SOURCE]))
        self.addParameter(QgsProcessingParameterFile("MODEL", "Model file", extension="json"))
        self.addParameter(QgsProcessingParameterFeatureSink("OUTPUT", "Model result"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        from pathlib import Path

        from .core.model import ModelSpec
        from .core.pipeline import run_model_on_layer

        source = self.parameterAsSource(parameters, "INPUT", context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, "INPUT"))
        path = self.parameterAsFile(parameters, "MODEL", context)
        try:
            model = ModelSpec.from_json(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise QgsProcessingException(f"Could not read model file: {exc}") from exc
        problems = model.validate()
        if problems:
            raise QgsProcessingException("Model is not runnable:\n" + "\n".join(problems))

        layer = self._materialize(source, feedback)
        feedback.pushInfo(f"Running model '{model.name}' ({len(model.criteria)} criteria, "
                          f"{len(model.constraints)} constraints) on {layer.featureCount():,} hexagons…")
        try:
            result = run_model_on_layer(
                layer, model, force=True,
                progress=lambda p: feedback.setProgress(0.8 * p),
                cancelled=feedback.isCanceled,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        for line in result.log:
            feedback.pushInfo(f"  {line}")
        st = result.stats
        feedback.pushInfo(f"mean = {st['mean_score']}, max = {st['max_score']}, excluded = {st['excluded_count']}")
        dest_id = self._write_sink(parameters, context, layer, feedback)
        self._attach_style(context, dest_id, layer, "composite", model.theme, st.get("breaks"))
        return {"OUTPUT": dest_id}


class H3ForgeProvider(QgsProcessingProvider):
    def id(self):
        return PROVIDER_ID

    def name(self):
        return "02H3Forge"

    def longName(self):  # noqa: N802
        return f"02H3Forge {PLUGIN_VERSION} — H3 Hexagonal MCDA"

    def icon(self):
        return QIcon(ICON_PATH)

    def versionInfo(self):  # noqa: N802
        return PLUGIN_VERSION

    def loadAlgorithms(self):  # noqa: N802
        for alg in (GenerateH3GridAlgorithm(), CompositeOverlayAlgorithm(), HotspotAlgorithm(),
                    RunModelAlgorithm()):
            self.addAlgorithm(alg)


__all__ = ["H3ForgeProvider", "parse_weights", "PROVIDER_ID"]
