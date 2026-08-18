"""Pairs rendered frames with the node annotations for the same trusses.

The pixel corpus (`corpus/`) and the structure corpus (`graph_corpus/`) are
generated from identical seed ranges and were cross-checked to agree to
0.000e+00 px, so no new generator is needed: the images come from one and the
labels from the other.

Labels are CenterNet-style. A heatmap at stride 4 carries a Gaussian bump per
node in one of three channels -- free, support, loaded -- and a shared two
channel offset head recovers the sub-pixel remainder the stride threw away.
That matters more than usual here: the game is scored on displacement, and
measurement says every pixel of node-position error costs about a point, so
landing on the right 4x4 cell is not good enough.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import data as pixel_data
from . import graph_data

# Must match CROP in v2/src/config.js.
CROP_ORIGIN = 68.0
CROP_SIZE = 768.0

STRIDE = 4
FREE, SUPPORT, LOADED = 0, 1, 2
CLASSES = 3


def screen_to_input(xy, input_size):
    """Screen pixels -> pixels of the model's input image."""
    return (np.asarray(xy) - CROP_ORIGIN) * (input_size / CROP_SIZE)


def input_to_screen(xy, input_size):
    return np.asarray(xy) * (CROP_SIZE / input_size) + CROP_ORIGIN


class TrussDetection(Dataset):
    def __init__(self, split, pixel_root=pixel_data.DEFAULT_ROOT,
                 graph_root=graph_data.DEFAULT_ROOT, blob_sigma=1.5, mirror=False, seed=0):
        self.split, self.pixel_root = split, pixel_root
        self._images = pixel_data.load_images(split, pixel_root)
        _, samples = graph_data.load_raw(split, graph_root)
        if len(self._images) != len(samples):
            raise ValueError(f'{split}: {len(self._images)} images vs {len(samples)} annotations')

        self.input_size = self._images.shape[1]
        self._count = len(self._images)
        self.grid = self.input_size // STRIDE
        self.blob_sigma = blob_sigma
        self.mirror = mirror
        self._rng = np.random.default_rng(seed)

        self.nodes = np.array([s['nodes'] for s in samples], dtype=np.float64)
        self.supports = np.array([s['supports'] for s in samples], dtype=np.int64)
        self.loaded = np.array([s['loadedNode'] for s in samples], dtype=np.int64)

    @property
    def images(self):
        """Reopened per process -- see the note in dataset.py.

        np.memmap pickles by value, and Windows spawns DataLoader workers, so
        leaving this in the pickled state hands every worker a private copy of
        the whole corpus.
        """
        if self._images is None:
            self._images = pixel_data.load_images(self.split, self.pixel_root, verbose=False)
        return self._images

    def __getstate__(self):
        return {**self.__dict__, '_images': None}

    def __len__(self):
        return self._count

    def _classes(self, i):
        kind = np.full(self.nodes.shape[1], FREE, dtype=np.int64)
        kind[self.supports[i]] = SUPPORT
        kind[self.loaded[i]] = LOADED
        return kind

    def __getitem__(self, i):
        frame = np.asarray(self.images[i])
        points = screen_to_input(self.nodes[i], self.input_size)
        kind = self._classes(i)

        if self.mirror and self._rng.random() < 0.5:
            frame = frame[:, ::-1]
            points = points.copy()
            points[:, 0] = (self.input_size - 1) - points[:, 0]

        heatmap = np.zeros((CLASSES, self.grid, self.grid), dtype=np.float32)
        offset = np.zeros((2, self.grid, self.grid), dtype=np.float32)
        present = np.zeros((self.grid, self.grid), dtype=np.float32)

        radius = int(3 * self.blob_sigma)
        for (x, y), c in zip(points / STRIDE, kind):
            cx, cy = int(x), int(y)
            if not (0 <= cx < self.grid and 0 <= cy < self.grid):
                continue        # a node can sit outside the crop; skip, do not clamp
            x0, x1 = max(0, cx - radius), min(self.grid, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(self.grid, cy + radius + 1)
            gy, gx = np.ogrid[y0:y1, x0:x1]
            # Centred on the integer cell, not the float position. Two reasons,
            # both of which cost a training run to learn: the focal loss keys
            # its positives off cells that equal exactly 1.0, which only happens
            # if a cell sits at the centre; and the cell NMS picks must be the
            # same cell the offset was written to, or the offset corrects the
            # wrong peak. The sub-cell remainder is what `offset` is for.
            blob = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * self.blob_sigma ** 2))
            # Maximum, not sum: two nearby nodes must stay two peaks.
            heatmap[c, y0:y1, x0:x1] = np.maximum(heatmap[c, y0:y1, x0:x1], blob)
            offset[0, cy, cx] = x - cx
            offset[1, cy, cx] = y - cy
            present[cy, cx] = 1.0

        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 127.5 - 1.0
        return {
            'image': image,
            'heatmap': torch.from_numpy(heatmap),
            'offset': torch.from_numpy(offset),
            'present': torch.from_numpy(present),
            'points': torch.from_numpy(points).float(),
            'kind': torch.from_numpy(kind),
        }
