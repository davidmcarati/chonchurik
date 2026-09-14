"""Canonical market shapes and candidate chart renderings.

Prototypes live here rather than in stonkfly/display.py so that alternative
encodings can be measured before anything in the package changes and every
existing run directory is invalidated.
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


def render_current(history):
    """The renderer the package ships today: a 3px polyline."""
    return market_frame("BTC-USDC", history, BASE * 0.999, BASE * 1.001)


def render_filled(history, fixed_scale=False):
    """Fill the area under the curve.

    A 3px line puts the market on roughly 1.5% of the receptors. Filling the
    area makes direction a large-area property that many receptors see at once,
    which is what the measurement said was missing.
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
        if fixed_scale:
            # Absolute level carries information only if the axis stops moving.
            lo, span = BASE * 0.85, BASE * 0.30
        else:
            span = max(float(np.ptp(values)), float(np.mean(values)) * 0.002)
            lo = float(values.min()) - span * 0.12
            span *= 1.24
        ys = 153 - (values - lo) / span * 109
        ys = np.clip(ys, 34.0, 153.0)
        xs = 12 + np.arange(len(values)) * 294 / (len(values) - 1)
        rising = values[-1] >= values[0]
        polygon = [(float(x), float(y)) for x, y in zip(xs, ys)]
        polygon += [(float(xs[-1]), 153.0), (float(xs[0]), 153.0)]
        d.polygon(polygon, fill=UP if rising else DOWN)
    d.text((9, 165), "BID / ASK", fill=(28, 46, 82))
    return np.asarray(im, dtype=np.uint8)


RENDERERS = {
    "current": render_current,
    "filled": render_filled,
    "filled_fixed_scale": lambda h: render_filled(h, fixed_scale=True),
}
