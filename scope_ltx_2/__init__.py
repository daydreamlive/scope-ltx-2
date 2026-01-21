"""LTX-2 text-to-video pipeline plugin for Daydream Scope."""

import scope.core

from .pipeline import LTX2Pipeline
from .schema import LTX2Config


@scope.core.hookimpl
def register_pipelines(register):
    """Register the LTX2 pipeline with Scope."""
    register(LTX2Pipeline)


__all__ = ["LTX2Pipeline", "LTX2Config"]
