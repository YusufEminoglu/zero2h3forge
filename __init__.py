# -*- coding: utf-8 -*-
"""02H3Forge - H3 Hexagonal Risk, Vulnerability and Suitability Studio: entry point.

The package directory is ``zero2h3forge`` so the top-level folder is a valid
Python identifier as required by QGIS and PEP 8. All intra-package imports are
strictly relative (``from .module import X``).
"""
from __future__ import annotations

from typing import Any

PLUGIN_VERSION = "0.3.0"


def classFactory(iface: Any):
    """Load O2H3ForgePlugin class from main_plugin."""
    from .main_plugin import O2H3ForgePlugin

    return O2H3ForgePlugin(iface)
