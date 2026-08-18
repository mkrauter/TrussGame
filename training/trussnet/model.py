"""The v3 architecture.

Every choice here answers a measured failure of the deployed v1 model:

- **Receptive field.** v1 reached 106px at 256x256 input while the support span
  is 190-233px, so no unit ever saw both supports and the trunk could not
  compute the thing that determines the answer. Five stages with dilation 2 and
  4 in the last two reach 237px. Dilation costs no parameters at all, which
  matters under a fixed budget.
- **Two heads.** v1 regressed the absolute settled position, but localisation is
  already solved (0.97 correlation) and soaked up a loss it had nothing left to
  learn from. Predicting start and delta separately puts the gradient on the
  mechanics, which is the part that scored a negative R-squared.
- **Soft-argmax instead of Flatten -> Dense.** That single dense layer was
  204,994 of v1's 307,618 parameters -- two thirds of the model doing all the
  global reasoning from a thin 10x10x32 map. A soft-argmax turns each feature
  map into a coordinate for zero parameters, which is the natural operation for
  coordinate regression, and unlike global average pooling it keeps position.
- **CoordConv.** Convolutions are translation-equivariant; the output is a
  translation-variant coordinate. Two extra input channels fix the mismatch.
- **Normalisation and residual-free depth.** v1 had no normalisation anywhere
  and fed raw 0-255 into ReLU with Glorot init, which is why its first epochs
  were flat at a constant output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Stage(nn.Module):
    """One same-padding conv (optionally dilated) then a stride-2 conv."""

    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            # Stride-2 convolution rather than max pooling: pooling discards
            # where the maximum was, which is the wrong invariance when the
            # output is a coordinate.
            nn.Conv2d(cout, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class SpatialSoftArgmax(nn.Module):
    """Expected (x, y) of each channel's softmax, in [-1, 1]. No parameters."""

    def __init__(self, temperature: float = 1.0, learn_temperature: bool = True):
        super().__init__()
        t = torch.tensor(float(temperature)).log()
        self.log_temperature = nn.Parameter(t) if learn_temperature else None
        self._t = temperature

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.log_temperature.exp() if self.log_temperature is not None else self._t
        weights = (x.flatten(2) / t).softmax(dim=-1).view(b, c, h, w)

        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        ex = (weights * xs).sum(dim=(2, 3))
        ey = (weights * ys).sum(dim=(2, 3))
        return torch.stack([ex, ey], dim=-1).flatten(1)      # (B, 2C)


class TrussNet(nn.Module):
    def __init__(
        self,
        widths=(24, 32, 48, 64, 80),
        dilations=(1, 1, 1, 2, 4),
        neck=32,
        mlp=(128, 128),
        in_channels=3,
    ):
        super().__init__()
        if len(widths) != len(dilations):
            raise ValueError('widths and dilations must be the same length')

        stages, cin = [], in_channels + 2      # +2 for the coordinate channels
        for width, dilation in zip(widths, dilations):
            stages.append(Stage(cin, width, dilation))
            cin = width
        self.stages = nn.Sequential(*stages)

        self.neck = nn.Sequential(
            nn.Conv2d(cin, neck, 1, bias=False),
            nn.BatchNorm2d(neck),
            nn.ReLU(inplace=True),
        )
        self.soft_argmax = SpatialSoftArgmax()

        layers, prev = [], neck * 3            # (x, y) per channel plus its mean
        for hidden in mlp:
            layers += [nn.Linear(prev, hidden), nn.ReLU(inplace=True)]
            prev = hidden
        self.mlp = nn.Sequential(*layers)

        self.start_head = nn.Linear(prev, 2)
        self.delta_head = nn.Linear(prev, 2)

    @staticmethod
    def _with_coords(x):
        b, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, image):
        """`image` is (B, 3, 256, 256) scaled to roughly [-1, 1].

        Returns start, delta and their sum, all in the normalised coordinates of
        trussnet.coords.
        """
        x = self.neck(self.stages(self._with_coords(image)))
        features = torch.cat([self.soft_argmax(x), x.mean(dim=(2, 3))], dim=1)
        hidden = self.mlp(features)
        start = self.start_head(hidden)
        delta = self.delta_head(hidden)
        return start, delta, start + delta


def receptive_field(dilations=(1, 1, 1, 2, 4)):
    """Pixels of the 256x256 input that one output unit can see."""
    rf, jump = 1, 1
    for d in dilations:
        rf += 2 * d * jump      # dilated 3x3
        rf += 2 * jump          # stride-2 3x3
        jump *= 2
    return rf


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
