"""
Alias re-exporting rc_frame_model for convenience.
"""

from .rc_frame_model import (
    create_rc_frame,
    create_rc_frame_core,
    X_COORDS,
    Y_COORDS,
    Z_COORDS,
    STORY_ELEVATIONS,
    COL_WIDTH,
    COL_DEPTH,
    BEAM_WIDTH,
    BEAM_DEPTH,
    CONCRETE_E,
    CONCRETE_NU,
    CONCRETE_UNIT_WEIGHT,
    CONCRETE_RHO,
)

__all__ = [
    "create_rc_frame",
    "create_rc_frame_core",
    "X_COORDS",
    "Y_COORDS",
    "Z_COORDS",
    "STORY_ELEVATIONS",
    "COL_WIDTH",
    "COL_DEPTH",
    "BEAM_WIDTH",
    "BEAM_DEPTH",
    "CONCRETE_E",
    "CONCRETE_NU",
    "CONCRETE_UNIT_WEIGHT",
    "CONCRETE_RHO",
]
