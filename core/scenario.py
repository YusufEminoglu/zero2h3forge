# -*- coding: utf-8 -*-
"""Reproducible MCDA scenario files (JSON) for 02H3Forge.

A scenario captures every analyst decision made after the H3 grid exists:
criterion weights / activity / locks, the overlay engine, map theme, class
breaks, contagion alpha, constraint and Monte Carlo switches. Saving it next to
a project makes a composite map fully reproducible and easy to compare.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Mapping

SCENARIO_FORMAT = "02h3forge-scenario"
SCENARIO_VERSION = 1

_SETTING_DEFAULTS: Dict[str, object] = {
    "compose_method": "wlc",
    "theme": "risk",
    "class_mode": "equal",
    "smooth_alpha": 0.0,
    "use_constraints": True,
    "run_montecarlo": True,
    "owa_alpha": 2.0,
}


def build_scenario(
    criteria: List[Mapping[str, object]],
    settings: Mapping[str, object],
    name: str = "",
    layer_name: str = "",
    plugin_version: str = "",
) -> Dict[str, object]:
    """Build a JSON-serializable scenario dictionary.

    ``criteria`` items: ``{"field": str, "weight": float, "active": bool, "locked": bool}``.
    """
    clean_criteria: List[Dict[str, object]] = []
    for c in criteria:
        field = str(c.get("field", "")).strip()
        if not field:
            continue
        clean_criteria.append({
            "field": field,
            "weight": round(float(c.get("weight", 0.0)), 4),
            "active": bool(c.get("active", True)),
            "locked": bool(c.get("locked", False)),
        })
    merged = dict(_SETTING_DEFAULTS)
    merged.update({k: settings[k] for k in settings if k in _SETTING_DEFAULTS})
    return {
        "format": SCENARIO_FORMAT,
        "version": SCENARIO_VERSION,
        "name": str(name or "Untitled scenario"),
        "layer": str(layer_name or ""),
        "plugin_version": str(plugin_version or ""),
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": merged,
        "criteria": clean_criteria,
    }


def validate_scenario(data: Mapping[str, object]) -> Dict[str, object]:
    """Validate and normalize a scenario dict; raise ``ValueError`` if unusable."""
    if not isinstance(data, Mapping):
        raise ValueError("Scenario must be a JSON object.")
    if data.get("format") != SCENARIO_FORMAT:
        raise ValueError("Not a 02H3Forge scenario file (missing 'format' marker).")
    version = int(data.get("version", 0) or 0)
    if version < 1 or version > SCENARIO_VERSION:
        raise ValueError(f"Unsupported scenario version {version}.")
    raw_criteria = data.get("criteria", [])
    if not isinstance(raw_criteria, list):
        raise ValueError("Scenario 'criteria' must be a list.")
    raw_settings = data.get("settings", {})
    if not isinstance(raw_settings, Mapping):
        raise ValueError("Scenario 'settings' must be an object.")
    out = build_scenario(
        [c for c in raw_criteria if isinstance(c, Mapping)],
        raw_settings,
        name=str(data.get("name", "")),
        layer_name=str(data.get("layer", "")),
        plugin_version=str(data.get("plugin_version", "")),
    )
    out["created"] = str(data.get("created", out["created"]))
    alpha = float(out["settings"]["smooth_alpha"])
    out["settings"]["smooth_alpha"] = min(0.5, max(0.0, alpha))
    return out


def save_scenario(path: str, scenario: Mapping[str, object]) -> str:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(scenario, fp, indent=2, ensure_ascii=False)
    return path


def load_scenario(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return validate_scenario(data)


def match_scenario_criteria(
    scenario: Mapping[str, object],
    available_fields: List[str],
) -> Dict[str, object]:
    """Split scenario criteria into those present on the layer and those missing."""
    avail = set(available_fields)
    matched: Dict[str, Dict[str, object]] = {}
    missing: List[str] = []
    for c in scenario.get("criteria", []):  # type: ignore[union-attr]
        fld = str(c["field"])
        if fld in avail:
            matched[fld] = dict(c)
        else:
            missing.append(fld)
    return {"matched": matched, "missing": missing}
