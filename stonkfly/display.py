"""Render observations into an RGB chart; never reads future prices or P&L."""

import numpy as np
from PIL import Image, ImageDraw

BACKGROUND = (235, 240, 249)
HEADER = (19, 36, 71)
HEADER_TEXT = (219, 229, 249)
GRID = (200, 212, 233)
FOOTER_TEXT = (28, 46, 82)
UP = (0, 101, 183)
DOWN = (197, 37, 78)
LEFT, RIGHT, TOP, FLOOR = 12, 306, 34, 153


def market_frame(product, history, bid, ask):
    """Draw the price history as a filled area rather than a thin line.

    The 3-pixel polyline this replaces put the market on roughly 1.5% of the
    mapped receptors. Measured consequence (`tools/diagnose.py separation`): a
    rally and a crash reached the retina as 98.5% the same image, and ten of
    twenty-eight market-state pairs produced a byte-identical Kenyon-cell
    code -- pairs no reinforcement rule can ever tell apart. Filling the area
    makes direction a large-area property that many receptors see at once,
    which removed all ten collisions. No neural parameter changes with this;
    it is a display adapter, and sparseness is a separate problem.
    """
    im = Image.new("RGB", (320, 180), BACKGROUND)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 319, 27), fill=HEADER)
    d.text((9, 8), product, fill=HEADER_TEXT)
    for x in range(LEFT, 310, 30):
        d.line((x, TOP, x, 160), fill=GRID)
    for y in range(38, 162, 24):
        d.line((10, y, 308, y), fill=GRID)
    values = np.asarray(history[-100:], dtype=float)
    if len(values):
        span = max(float(np.ptp(values)), float(np.mean(values)) * 0.002)
        lo = float(values.min()) - span * 0.12
        span *= 1.24
        ys = np.clip(FLOOR - (values - lo) / span * (FLOOR - TOP), TOP, FLOOR)
        xs = LEFT + np.arange(len(values)) * (RIGHT - LEFT) / max(1, len(values) - 1)
        if len(values) == 1:
            xs = np.array([LEFT, RIGHT], dtype=float)
            ys = np.repeat(ys, 2)
        outline = [(float(x), float(y)) for x, y in zip(xs, ys)]
        d.polygon(
            outline + [(float(xs[-1]), float(FLOOR)), (float(xs[0]), float(FLOOR))],
            fill=UP if values[-1] >= values[0] else DOWN,
        )
    d.text((9, 165), f"BID {bid}  ASK {ask}"[:50], fill=FOOTER_TEXT)
    return np.asarray(im, dtype=np.uint8)
