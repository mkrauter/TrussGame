"""Stage one: recover the truss from the rendered frame, and nothing else.

This is the half that keeps the game fair -- the model's only input is the
colour screenshot a player looks at. It returns node positions and each node's
role; connectivity is read from the same image afterwards (see detect.py), and
the mechanics are left to the graph network, which is the right computational
class for a global implicit solve.

Detection is genuinely easy here in a way the physics never was: markers are
distinct shapes in known colours on a flat background. So the network is small
and shallow, downsamples only to stride 4, and spends its capacity on precision
rather than semantics. Precision is what matters -- the acuity curve says every
screen pixel of node error costs roughly a point of game score.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .detect_data import CLASSES


def block(cin, cout, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class TrussDetector(nn.Module):
    """Sized against the browser, not the GPU.

    This runs in plain JavaScript at inference, where measured throughput is
    ~220 MMAC/s. A width-32 version with a four-deep dilated context stack cost
    4.6 seconds per move -- accurate and unshippable. So the budget is roughly
    50 MMAC, and the architecture spends it where it buys something.

    What it does not need is a large receptive field. A node's role is written
    in the colour of the marker drawn on it -- green support, blue loaded, bare
    junction otherwise -- which is a local question, unlike the physics. The one
    convolution kept at stride 2 is there for precision, not semantics: the
    offset head needs fine detail to resolve sub-pixel position, and that detail
    only exists before the second downsample.
    """

    def __init__(self, width=8, context=3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            block(3, w, stride=2),          # 256 -> 128, fine detail lives here
            block(w, w),
            block(w, 2 * w, stride=2),      # 128 -> 64, the output stride
        )
        self.context = nn.Sequential(
            *[block(2 * w, 2 * w, dilation=1 if i == 0 else 2) for i in range(context)]
        )
        self.heatmap = nn.Conv2d(2 * w, CLASSES, 1)
        self.offset = nn.Conv2d(2 * w, 2, 1)

        # Start with the heatmap confidently empty. Almost every cell is
        # background, and without this the first steps are spent unlearning a
        # uniform 0.5 prior.
        nn.init.constant_(self.heatmap.bias, -4.0)

    def forward(self, image):
        features = self.context(self.stem(image))
        return self.heatmap(features), self.offset(features)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# How many of each role every truss has. These are properties of the game, not
# guesses: supports are the two extreme-x nodes and exactly one node is loaded,
# so the decoder takes the strongest 2 and 1 rather than thresholding and hoping.
COUNTS = (7, 2, 1)      # free, support, loaded


@torch.no_grad()
def decode(heat_logits, offset, counts=COUNTS, stride=4):
    """Heatmaps -> node positions in input-image pixels.

    Returns (points, scores) with points (B, sum(counts), 2) ordered by class:
    free nodes first, then supports, then the loaded node.
    """
    heat = heat_logits.sigmoid()
    # 3x3 max pooling as non-maximum suppression: keep only cells that are the
    # local peak, so one blob yields one node instead of nine.
    peak = torch.nn.functional.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (peak == heat)

    b, _, g, _ = heat.shape
    points, scores = [], []
    for c, k in enumerate(counts):
        flat = heat[:, c].reshape(b, -1)
        score, index = flat.topk(k, dim=1)
        cy = (index // g).float()
        cx = (index % g).float()
        # The offset head recovers what the stride discarded; without it the
        # best possible error is half a cell, which is 6 screen pixels.
        off = offset.reshape(b, 2, -1).gather(2, index.unsqueeze(1).expand(-1, 2, -1))
        points.append(torch.stack([(cx + off[:, 0]) * stride, (cy + off[:, 1]) * stride], dim=-1))
        scores.append(score)
    return torch.cat(points, dim=1), torch.cat(scores, dim=1)
