"""Read the truss's connectivity off the screenshot.

Stage one recovers *where* the nodes are; this recovers *which pairs are joined*
by looking at the same image. A member is drawn as a straight line between two
nodes, so for a candidate pair the test is simply whether the pixels along the
segment between them are drawn rather than background.

That is deliberately not a learned component. There is nothing to generalise:
the question "is there a line here" is answered exactly by looking, and a
network would only add a way to be wrong. It reads the *full-resolution* frame
rather than the detector's 256px input -- both are the same screenshot, and
there is no reason to ask this question of a downsample that renders 2.5px
members 0.83px wide.
"""
from __future__ import annotations

import numpy as np

# grey25, matching RENDER.background in v2/src/config.js.
BACKGROUND = np.array([64, 64, 64], dtype=np.int16)

# Node markers are 20px tall and members converge at joints, so both ends of a
# candidate segment are uninformative. Sample only the clear middle.
TRIM = 0.22
SAMPLES = 24
# A drawn member is white (255) against 64, so anything close to background is
# not a member. Generous, because a segment crossing others must not be rescued
# by them -- the test is whether *most* of the span is drawn.
INK_THRESHOLD = 40
COVERAGE = 0.9
# How far to either side of the line to look, in screen pixels. Sized against
# the detector's node error, not the member width.
HALF_WIDTH = 3.0


def _bilinear(image, x, y):
    """Sample the frame at fractional coordinates."""
    h, w, _ = image.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def segment_coverage(image, a, b, samples=SAMPLES, trim=TRIM, half_width=HALF_WIDTH):
    """Fraction of the middle of segment a-b that is drawn rather than blank.

    Each sample looks a little way to either side of the line, because the node
    positions this is called with come from the detector and carry about 1.5px
    of error. Members are only 2.5px wide, so without the perpendicular search
    a 1.5px error at the endpoints walks the sampled line straight off the
    member -- which is exactly what it did: 139 of 200 frames got the member
    count wrong with detected nodes, against 12 of 400 with exact ones.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = np.hypot(*d)
    if length == 0:
        return 0.0
    normal = np.array([-d[1], d[0]]) / length

    t = np.linspace(trim, 1.0 - trim, samples)
    base = a[None, :] + d[None, :] * t[:, None]
    frame = image.astype(np.float32)

    best = np.zeros(samples)
    for offset in np.linspace(-half_width, half_width, 2 * int(half_width) + 1):
        p = base + normal[None, :] * offset
        values = _bilinear(frame, p[:, 0], p[:, 1])
        best = np.maximum(best, np.abs(values - BACKGROUND).max(axis=1))
    return float((best > INK_THRESHOLD).mean())


# A node lying this close to the line between two others, and between them, is a
# joint *on* that line. Tight on purpose: a genuinely collinear node sits almost
# exactly on the line, whereas a real member can pass a few pixels from an
# unrelated node. Swept over 400 frames -- 0px leaves 130 spurious members, 8px
# wrongly rejects 69 real ones, and 2px is the minimum of the two errors.
ON_SEGMENT = 2.0


def passes_through_node(a, b, points, tolerance=ON_SEGMENT):
    """Is some other node sitting on the segment a-b?

    Three collinear nodes A-B-C make the A-C segment look drawn, because A-B and
    B-C cover it. What is actually on screen is two members, and that is what a
    player reads, so the spanning pair has to be rejected. This was 334 spurious
    members over 1000 frames before the check, and every spurious one was this.
    """
    ab = b - a
    length = np.hypot(*ab)
    if length == 0:
        return False
    for p in points:
        if np.allclose(p, a) or np.allclose(p, b):
            continue
        t = float(np.dot(p - a, ab) / (length ** 2))
        if not (0.0 < t < 1.0):
            continue
        perpendicular = abs(float(np.cross(ab, p - a))) / length
        if perpendicular < tolerance:
            return True
    return False


def find_members(image, points, coverage=COVERAGE, **kwargs):
    """Every pair whose connecting segment is drawn. Returns sorted (i, j)."""
    points = np.asarray(points, dtype=np.float64)
    members = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if passes_through_node(points[i], points[j], points):
                continue
            if segment_coverage(image, points[i], points[j], **kwargs) >= coverage:
                members.append((i, j))
    return members
