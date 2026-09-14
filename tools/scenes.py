"""Canonical market shapes and the chart renderings compared against them.

Prototypes live here rather than in stonkfly/display.py so that alternative
encodings can be measured before anything in the package changes and every
existing run directory is invalidated.

`polyline` is a frozen copy of the renderer the package shipped up to commit
a8e2186. It is kept byte-for-byte so the recorded baseline never moves when
the package changes. `filled` calls the package, so re-running `separation`
measures what actually ships rather than a prototype that has drifted from it.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from stonkfly.display import market_frame

BACKGROUND = (235, 240, 249)
HEADER = (19, 36, 71)
GRID = (200, 212, 233)
UP = (0, 101, 183)
DOWN = (197, 37, 78)
BASE = 60000.0
BID, ASK = BASE * 0.999, BASE * 1.001

SCENES = {
    "rally": lambda b: [b * (1 + 0.004 * i) for i in range(100)],
    "crash": lambda b: [b * (1 - 0.004 * i) for i in range(100)],
    "flat": lambda b: [b] * 100,
    "chop": lambda b: [b * (1 + 0.02 * math.sin(i * 0.9)) for i in range(100)],
    "spike_up": lambda b: [b] * 80 + [b * (1 + 0.0015 * i) for i in range(20)],
    "spike_down": lambda b: [b] * 80 + [b * (1 - 0.0015 * i) for i in range(20)],
    "peak": lambda b: [b * (1 + 0.004 * min(i, 99 - i)) for i in range(100)],
    "valley": lambda b: [b * (1 - 0.004 * min(i, 99 - i)) for i in range(100)],
}


def scene_history(name, base=BASE):
    return SCENES[name](base)


def render_polyline(history):
    """Frozen copy of the shipped renderer as of a8e2186: a 3px polyline.

    Do not refactor this to call the package. It is the fixed reference the
    measured 10/28 Kenyon-code collisions belong to.
    """
    im = Image.new("RGB", (320, 180), BACKGROUND)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 319, 27), fill=HEADER)
    d.text((9, 8), "BTC-USDC", fill=(219, 229, 249))
    for x in range(12, 310, 30):
        d.line((x, 34, x, 160), fill=GRID)
    for y in range(38, 162, 24):
        d.line((10, y, 308, y), fill=GRID)
    values = np.asarray(history[-100:], dtype=float)
    if len(values):
        span = max(float(np.ptp(values)), float(np.mean(values)) * 0.002)
        lo = float(values.min()) - span * 0.12
        span *= 1.24
        points = [
            (12 + i * 294 / max(1, len(values) - 1), 153 - (v - lo) / span * 109)
            for i, v in enumerate(values)
        ]
        if len(points) > 1:
            for a, b in zip(points, points[1:]):
                d.line((*a, *b), fill=UP if b[1] <= a[1] else DOWN, width=3)
        for x, y in points:
            d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=(27, 39, 81))
    d.text((9, 165), f"BID {BID}  ASK {ASK}"[:50], fill=(28, 46, 82))
    return np.asarray(im, dtype=np.uint8)


def render_filled(history):
    """What the package renders now. Measured, not prototyped."""
    return market_frame("BTC-USDC", history, BID, ASK)


def render_fixed_scale(history):
    """Filled, but on an axis that does not follow the data.

    Absolute price level carries information only if the axis stops moving.
    Measured cost: it separates the input further and reintroduces six code
    collisions, so it is kept as a rejected candidate, not as a proposal.
    """
    im = Image.new("RGB", (320, 180), BACKGROUND)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 319, 27), fill=HEADER)
    d.text((9, 8), "BTC-USDC", fill=(219, 229, 249))
    for x in range(12, 310, 30):
        d.line((x, 34, x, 160), fill=GRID)
    for y in range(38, 162, 24):
        d.line((10, y, 308, y), fill=GRID)
    values = np.asarray(history[-100:], dtype=float)
    if len(values) > 1:
        lo, span = BASE * 0.85, BASE * 0.30
        ys = np.clip(153 - (values - lo) / span * 109, 34.0, 153.0)
        xs = 12 + np.arange(len(values)) * 294 / (len(values) - 1)
        polygon = [(float(x), float(y)) for x, y in zip(xs, ys)]
        polygon += [(float(xs[-1]), 153.0), (float(xs[0]), 153.0)]
        d.polygon(polygon, fill=UP if values[-1] >= values[0] else DOWN)
    d.text((9, 165), f"BID {BID}  ASK {ASK}"[:50], fill=(28, 46, 82))
    return np.asarray(im, dtype=np.uint8)


# `polyline` is the frozen baseline; `filled` is what the package ships.
BASELINE = "polyline"
RENDERERS = {
    "polyline": render_polyline,
    "filled": render_filled,
    "filled_fixed_scale": render_fixed_scale,
}
