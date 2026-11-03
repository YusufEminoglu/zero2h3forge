# -*- coding: utf-8 -*-
"""Declarative MCDA model for 02H3Forge (criterion-centric, hierarchical).

A :class:`ModelSpec` is the single source of truth for an analysis. It is pure
data (JSON-serializable, no QGIS imports) and describes *how* every column of
the H3 layer is produced, so a whole study can be re-run, shared and audited:

- **Criteria** (:class:`CriterionSpec`): where the raw value comes from
  (vector aggregation, raster zonal statistic, distance to features, or an
  existing field) and the **value function** that turns it into a 0-100 score
  (linear, robust, fuzzy, or graduated distance rings).
- **Groups** (:class:`GroupSpec`): criteria are combined *within* a group with
  WLC / OWA / TOPSIS / VIKOR into a 0-100 sub-index (e.g. Hazard).
- **Aggregation**: group sub-indices are combined into ``comp_score`` with a
  weighted sum (compensatory) or a weighted geometric mean (non-compensatory,
  the standard for Risk = Hazard × Exposure × Vulnerability).
- **Constraints** (:class:`ConstraintSpec`): rule-based hard exclusions,
  independent of the criteria (e.g. "closer than 50 m to a power line").
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

MODEL_FORMAT = "02h3forge-model"
MODEL_VERSION = 2

SOURCE_KINDS: List[Tuple[str, str]] = [
    ("vector", "Vector aggregation (points / lines / polygons → hexagon)"),
    ("raster", "Raster zonal statistic"),
    ("distance", "Distance to nearest feature (m)"),
    ("field", "Existing field on the H3 layer"),
]

VALUE_METHODS: List[Tuple[str, str]] = [
    ("minmax", "Linear (observed min → max)"),
    ("percentile", "Linear, robust (P2 → P98)"),
    ("fuzzy_linear", "Linear between thresholds [low, high]"),
    ("fuzzy_sigmoid", "Sigmoid (midpoint, spread)"),
    ("fuzzy_gaussian", "Gaussian optimum (centre, spread)"),
    ("rings", "Graduated distance rings (stepped)"),
    ("direct_clip", "Already 0–100 (use as is)"),
]

GROUP_METHODS: List[Tuple[str, str]] = [
    ("wlc", "WLC — weighted linear combination"),
    ("owa", "OWA — weighted OWA with risk attitude α"),
    ("topsis", "TOPSIS — closeness to ideal"),
    ("vikor", "VIKOR — compromise ranking"),
]

AGGREGATIONS: List[Tuple[str, str]] = [
    ("weighted_sum", "Weighted sum (compensatory)"),
    ("geometric", "Weighted geometric mean (non-compensatory, H × E × V)"),
]

CONSTRAINT_KINDS: List[Tuple[str, str]] = [
    ("near", "Exclude hexagons closer than a distance to features"),
    ("inside", "Exclude hexagons inside polygons"),
    ("outside", "Exclude hexagons outside polygons"),
    ("rule", "Exclude by attribute rule (e.g. raw_slope > 30)"),
]

RULE_OPERATORS = ("<", "<=", ">", ">=", "==", "!=")

TEMPLATES: List[Tuple[str, str]] = [
    ("flat", "Single index (all criteria in one group)"),
    ("risk", "Risk = Hazard × Exposure × Vulnerability"),
    ("suitability", "Land suitability (physical · access · socio-economic)"),
]

RESERVED_IDS = {
    "h3", "res", "area", "constraint", "comp", "class", "top", "driver", "mc", "rob",
    "gi", "hotspot",
}


def slugify(text: str, fallback: str = "c", max_len: int = 20) -> str:
    """ASCII identifier safe for column names (Turkish letters transliterated)."""
    table = str.maketrans({
        "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
        "ş": "s", "Ş": "s", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    })
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").translate(table).lower()).strip("_")
    if not s or s[0].isdigit():
        s = f"{fallback}_{s}" if s else fallback
    return s[:max_len].rstrip("_")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class ValueFunction:
    """Transforms raw criterion values into a 0-100 score."""

    method: str = "minmax"
    polarity: str = "benefit"            # benefit: high raw -> 100 ; cost: low raw -> 100
    low: Optional[float] = None
    high: Optional[float] = None
    mid: Optional[float] = None
    spread: Optional[float] = None
    rings: List[List[float]] = field(default_factory=list)   # [[min_m, max_m, score], ...]
    outside_score: float = 0.0

    def describe(self) -> str:
        pol = "+" if self.polarity == "benefit" else "−"
        if self.method == "rings":
            parts = [f"{r[0]:g}–{r[1]:g} m→{r[2]:g}" for r in self.rings]
            return f"rings: {', '.join(parts)}; else {self.outside_score:g}"
        if self.method == "fuzzy_linear":
            return f"linear [{self.low}, {self.high}] ({pol})"
        if self.method in ("fuzzy_sigmoid", "fuzzy_gaussian"):
            kind = "sigmoid" if self.method == "fuzzy_sigmoid" else "gaussian"
            return f"{kind} mid={self.mid}, spread={self.spread} ({pol})"
        return f"{dict(VALUE_METHODS).get(self.method, self.method)} ({pol})"


@dataclass
class SourceSpec:
    """Where a criterion's raw value comes from."""

    kind: str = "field"                  # vector | raster | distance | field
    layer_id: str = ""                   # QGIS layer id (preferred binding)
    layer_name: str = ""                 # display / fallback binding by name
    layer_uri: str = ""                  # provider source, for portable model files
    provider: str = ""
    field: str = ""                      # attribute (vector value field or H3 field)
    method: str = "count"                # vector or raster statistic
    band: int = 1

    def signature(self) -> str:
        """Stable key for raw-value caching (layer identity + extraction)."""
        return json.dumps(
            [self.kind, self.layer_id or self.layer_uri or self.layer_name, self.field,
             self.method, int(self.band)],
            sort_keys=True,
        )


@dataclass
class CriterionSpec:
    id: str
    label: str
    group: str
    source: SourceSpec = field(default_factory=SourceSpec)
    value: ValueFunction = field(default_factory=ValueFunction)
    weight: float = 0.0                  # % within its group
    active: bool = True
    locked: bool = False

    @property
    def raw_field(self) -> str:
        if self.source.kind == "field":
            return self.source.field
        prefix = "dist" if self.source.kind == "distance" else "raw"
        return f"{prefix}_{self.id}"

    @property
    def norm_field(self) -> str:
        return f"norm_{self.id}"


@dataclass
class ConstraintSpec:
    id: str
    label: str
    kind: str = "near"                   # near | inside | outside | rule
    source: SourceSpec = field(default_factory=SourceSpec)
    distance: float = 0.0                # metres, for "near"
    field: str = ""                      # for "rule"
    op: str = ">"
    value: float = 0.0
    active: bool = True

    def describe(self) -> str:
        lyr = self.source.layer_name or "?"
        if self.kind == "near":
            return f"closer than {self.distance:g} m to {lyr}"
        if self.kind == "inside":
            return f"inside {lyr}"
        if self.kind == "outside":
            return f"outside {lyr}"
        return f"{self.field} {self.op} {self.value:g}"


@dataclass
class GroupSpec:
    id: str
    label: str
    weight: float = 100.0                # % among groups
    method: str = "wlc"
    owa_alpha: float = 2.0
    locked: bool = False


@dataclass
class ModelSpec:
    name: str = "Untitled model"
    template: str = "flat"
    groups: List[GroupSpec] = field(default_factory=list)
    criteria: List[CriterionSpec] = field(default_factory=list)
    constraints: List[ConstraintSpec] = field(default_factory=list)
    aggregation: str = "weighted_sum"
    theme: str = "risk"
    class_mode: str = "equal"
    smooth_alpha: float = 0.0
    apply_constraints: bool = True
    run_montecarlo: bool = True
    mc_runs: int = 50
    mc_perturbation: float = 0.15
    hotspot_k: int = 1
    hotspot_fdr: bool = True
    run_hotspots: bool = True
    run_sensitivity: bool = True

    # ---- lookups -------------------------------------------------------- #

    def group(self, gid: str) -> Optional[GroupSpec]:
        return next((g for g in self.groups if g.id == gid), None)

    def criterion(self, cid: str) -> Optional[CriterionSpec]:
        return next((c for c in self.criteria if c.id == cid), None)

    def criteria_in(self, gid: str, active_only: bool = False) -> List[CriterionSpec]:
        return [c for c in self.criteria if c.group == gid and (c.active or not active_only)]

    def active_groups(self) -> List[GroupSpec]:
        """Groups that contain at least one active criterion."""
        return [g for g in self.groups if self.criteria_in(g.id, active_only=True)]

    def unique_id(self, label: str, existing: Optional[List[str]] = None) -> str:
        taken = set(existing if existing is not None else
                    [c.id for c in self.criteria] + [k.id for k in self.constraints]
                    + [g.id for g in self.groups])
        base = slugify(label)
        if base.split("_")[0] in RESERVED_IDS:
            base = f"c_{base}"[:20]
        cand, n = base, 2
        while cand in taken:
            suffix = f"_{n}"
            cand = base[: 20 - len(suffix)] + suffix
            n += 1
        return cand

    # ---- weights -------------------------------------------------------- #

    def normalize_weights(self) -> None:
        """Make group weights and active within-group weights each sum to 100.

        Group weights are kept for *every* group (an empty group keeps its
        share for when it receives criteria); runs normalise them over the
        groups that currently hold active criteria.
        """
        _rescale([g for g in self.groups], lambda g: g.weight,
                 lambda g, v: setattr(g, "weight", v), active=lambda g: True)
        for g in self.groups:
            _rescale(self.criteria_in(g.id), lambda c: c.weight,
                     lambda c, v: setattr(c, "weight", v), active=lambda c: c.active)

    def effective_weights(self) -> Dict[str, float]:
        """Global weight (%) of every active criterion = group % × within-group %.

        Exact for weighted-sum aggregation; for the geometric mean it is the
        exponent share and still the best single-number summary.
        """
        out: Dict[str, float] = {}
        groups = self.active_groups()
        g_tot = sum(max(0.0, g.weight) for g in groups) or 1.0
        for g in groups:
            crits = self.criteria_in(g.id, active_only=True)
            c_tot = sum(max(0.0, c.weight) for c in crits) or 1.0
            for c in crits:
                out[c.id] = max(0.0, g.weight) / g_tot * max(0.0, c.weight) / c_tot * 100.0
        return out

    # ---- validation ----------------------------------------------------- #

    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty = runnable)."""
        problems: List[str] = []
        gids = [g.id for g in self.groups]
        if len(set(gids)) != len(gids):
            problems.append("Group ids must be unique.")
        cids = [c.id for c in self.criteria]
        if len(set(cids)) != len(cids):
            problems.append("Criterion ids must be unique.")
        if not any(c.active for c in self.criteria):
            problems.append("Add at least one active criterion.")
        for c in self.criteria:
            if c.group not in gids:
                problems.append(f"Criterion '{c.label}' belongs to unknown group '{c.group}'.")
            if c.source.kind not in dict(SOURCE_KINDS):
                problems.append(f"Criterion '{c.label}': unknown source '{c.source.kind}'.")
            if c.source.kind == "field" and not c.source.field:
                problems.append(f"Criterion '{c.label}': choose a field.")
            if c.source.kind in ("vector", "raster", "distance") and not (
                c.source.layer_id or c.source.layer_uri or c.source.layer_name
            ):
                problems.append(f"Criterion '{c.label}': choose a source layer.")
            v = c.value
            if v.method == "rings":
                if not v.rings:
                    problems.append(f"Criterion '{c.label}': define at least one distance ring.")
                for r in v.rings:
                    if len(r) < 3 or float(r[1]) <= float(r[0]):
                        problems.append(f"Criterion '{c.label}': ring {r} needs max > min.")
            if v.method == "fuzzy_linear" and (v.low is None or v.high is None or v.high <= v.low):
                problems.append(f"Criterion '{c.label}': linear thresholds need high > low.")
            if v.method in ("fuzzy_sigmoid", "fuzzy_gaussian") and v.spread is not None and v.spread <= 0:
                problems.append(f"Criterion '{c.label}': spread must be > 0.")
        for k in self.constraints:
            if k.kind == "rule" and (not k.field or k.op not in RULE_OPERATORS):
                problems.append(f"Constraint '{k.label}': incomplete attribute rule.")
            if k.kind in ("near", "inside", "outside") and not (
                k.source.layer_id or k.source.layer_uri or k.source.layer_name
            ):
                problems.append(f"Constraint '{k.label}': choose a layer.")
            if k.kind == "near" and k.distance <= 0:
                problems.append(f"Constraint '{k.label}': distance must be > 0 m.")
        if self.aggregation not in dict(AGGREGATIONS):
            problems.append(f"Unknown aggregation '{self.aggregation}'.")
        return problems

    # ---- serialization -------------------------------------------------- #

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["format"] = MODEL_FORMAT
        d["version"] = MODEL_VERSION
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def copy(self) -> "ModelSpec":
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ModelSpec":
        if not isinstance(data, dict):
            raise ValueError("Model must be a JSON object.")
        fmt = data.get("format")
        if fmt == "02h3forge-scenario":
            return model_from_legacy_scenario(data)
        if fmt != MODEL_FORMAT:
            raise ValueError("Not a 02H3Forge model file.")
        if int(data.get("version", 0) or 0) > MODEL_VERSION:
            raise ValueError(f"Model version {data.get('version')} is newer than this plugin supports.")

        def _src(d) -> SourceSpec:
            return _known(SourceSpec, d or {})

        def _val(d) -> ValueFunction:
            v = _known(ValueFunction, d or {})
            v.rings = [[float(x) for x in r[:3]] for r in (v.rings or []) if len(r) >= 3]
            return v

        m = _known(cls, {k: v for k, v in data.items() if k not in ("groups", "criteria", "constraints")})
        m.groups = [_known(GroupSpec, g) for g in data.get("groups", []) or []]
        m.criteria = []
        for c in data.get("criteria", []) or []:
            crit = _known(CriterionSpec, {k: v for k, v in c.items() if k not in ("source", "value")})
            crit.source = _src(c.get("source"))
            crit.value = _val(c.get("value"))
            m.criteria.append(crit)
        m.constraints = []
        for k in data.get("constraints", []) or []:
            con = _known(ConstraintSpec, {kk: vv for kk, vv in k.items() if kk != "source"})
            con.source = _src(k.get("source"))
            m.constraints.append(con)
        m.smooth_alpha = min(0.5, max(0.0, float(m.smooth_alpha)))
        m.hotspot_k = min(6, max(1, int(m.hotspot_k)))
        return m

    @classmethod
    def from_json(cls, text: str) -> "ModelSpec":
        return cls.from_dict(json.loads(text))


def _known(klass, data: Dict[str, object]):
    """Instantiate a dataclass ignoring unknown keys (forward compatibility)."""
    names = set(klass.__dataclass_fields__)
    kwargs = {k: v for k, v in dict(data).items() if k in names}
    return klass(**kwargs)


def _rescale(items, get, put, active) -> None:
    act = [it for it in items if active(it)]
    for it in items:
        if not active(it):
            put(it, 0.0)
    if not act:
        return
    tot = sum(max(0.0, float(get(it))) for it in act)
    if tot <= 1e-9:
        for it in act:
            put(it, 100.0 / len(act))
        return
    vals = [round(max(0.0, float(get(it))) / tot * 100.0, 2) for it in act]
    vals[vals.index(max(vals))] += round(100.0 - sum(vals), 2)
    for it, v in zip(act, vals):
        put(it, round(v, 2))


# ---------------------------------------------------------------------------
# Templates & ring presets
# ---------------------------------------------------------------------------

RING_PRESETS: Dict[str, Tuple[List[List[float]], float]] = {
    "Power transmission line (0-100, 100-250, 250-500 m)": (
        [[0, 100, 100], [100, 250, 70], [250, 500, 35]], 0.0),
    "Active fault (0-100, 100-300, 300-1000 m)": (
        [[0, 100, 100], [100, 300, 75], [300, 1000, 40]], 0.0),
    "Stream / flood riparian (0-50, 50-150, 150-400 m)": (
        [[0, 50, 100], [50, 150, 65], [150, 400, 30]], 0.0),
    "Road / transit access (0-200, 200-500, 500-1000 m)": (
        [[0, 200, 100], [200, 500, 75], [500, 1000, 40]], 0.0),
}


def new_model(template: str = "flat", name: str = "") -> ModelSpec:
    """Create an empty model from a template."""
    if template == "risk":
        groups = [
            GroupSpec("hazard", "Hazard", 100 / 3, "wlc"),
            GroupSpec("exposure", "Exposure", 100 / 3, "wlc"),
            GroupSpec("vulnerability", "Vulnerability", 100 / 3, "wlc"),
        ]
        m = ModelSpec(name=name or "Risk model", template="risk", groups=groups,
                      aggregation="geometric", theme="risk")
    elif template == "suitability":
        groups = [
            GroupSpec("physical", "Physical", 40.0, "wlc"),
            GroupSpec("access", "Accessibility", 35.0, "wlc"),
            GroupSpec("socio", "Socio-economic", 25.0, "wlc"),
        ]
        m = ModelSpec(name=name or "Suitability model", template="suitability", groups=groups,
                      aggregation="weighted_sum", theme="suitability")
    else:
        m = ModelSpec(name=name or "Composite index", template="flat",
                      groups=[GroupSpec("index", "Index", 100.0, "wlc")],
                      aggregation="weighted_sum", theme="risk")
    return m


def apply_template(model: ModelSpec, template: str) -> ModelSpec:
    """Switch template keeping criteria/constraints; unknown groups go to the first group."""
    fresh = new_model(template, model.name)
    keep = {f: getattr(model, f) for f in (
        "class_mode", "smooth_alpha", "apply_constraints", "run_montecarlo", "mc_runs",
        "mc_perturbation", "hotspot_k", "hotspot_fdr", "run_hotspots", "run_sensitivity",
    )}
    for k, v in keep.items():
        setattr(fresh, k, v)
    fresh.constraints = copy.deepcopy(model.constraints)
    gids = {g.id for g in fresh.groups}
    for c in copy.deepcopy(model.criteria):
        if c.group not in gids:
            c.group = fresh.groups[0].id
        fresh.criteria.append(c)
    fresh.normalize_weights()
    return fresh


def model_from_legacy_scenario(data: Dict[str, object]) -> ModelSpec:
    """Convert a v0.2 scenario file (flat norm_* weights) into a flat model."""
    settings = data.get("settings", {}) if isinstance(data.get("settings"), dict) else {}
    m = new_model("flat", str(data.get("name", "") or "Imported scenario"))
    method = str(settings.get("compose_method", "wlc"))
    g = m.groups[0]
    g.method = "owa" if method.startswith("owa") else method
    if settings.get("owa_alpha") is not None:
        g.owa_alpha = float(settings["owa_alpha"])
    elif method == "owa_optimistic":
        g.owa_alpha = 0.5
    m.theme = str(settings.get("theme", m.theme))
    m.class_mode = str(settings.get("class_mode", m.class_mode))
    m.smooth_alpha = float(settings.get("smooth_alpha", 0.0))
    m.apply_constraints = bool(settings.get("use_constraints", True))
    m.run_montecarlo = bool(settings.get("run_montecarlo", True))
    for item in data.get("criteria", []) or []:
        fld = str(item.get("field", ""))
        if not fld:
            continue
        cid = m.unique_id(fld[5:] if fld.startswith("norm_") else fld)
        m.criteria.append(CriterionSpec(
            id=cid, label=cid, group=g.id,
            source=SourceSpec(kind="field", field=fld),
            value=ValueFunction(method="direct_clip"),
            weight=float(item.get("weight", 0.0)),
            active=bool(item.get("active", True)),
            locked=bool(item.get("locked", False)),
        ))
    m.normalize_weights()
    return m
