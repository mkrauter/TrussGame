"""Screen pixels <-> the normalised coordinates the network works in.

The model only ever sees the CROP region, so that region is what gets mapped to
[-1, 1]. Targets in raw screen pixels span roughly 100..900, which is what gave
v1 an initial MSE around 1e5 and eight flat epochs before anything moved.

Positions outside the crop map outside [-1, 1], deliberately: about 1% of
settled targets land off the visible image, and the model has to be able to say
so. That is exactly why the answer is regressed rather than read off a heatmap,
which cannot place a peak outside its own grid.
"""
from __future__ import annotations

# Must match CROP in v2/src/config.js.
CROP_ORIGIN = 68.0
CROP_SIZE = 768.0

_HALF = CROP_SIZE / 2.0
_CENTRE = CROP_ORIGIN + _HALF


def position_to_model(xy):
    """Screen pixels -> [-1, 1] across the crop."""
    return (xy - _CENTRE) / _HALF


def position_from_model(v):
    return v * _HALF + _CENTRE


def delta_to_model(d):
    """Displacements carry no origin, only the scale factor."""
    return d / _HALF


def delta_from_model(d):
    return d * _HALF
