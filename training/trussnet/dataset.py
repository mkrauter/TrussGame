"""Torch dataset over the generated corpus.

Images stay memory-mapped and are converted per sample; a 20k split is 3.9 GB
decoded at 256 and 8.8 GB at 384, so holding it outright would not fit
alongside everything else.

The memmap is deliberately not part of the pickled state. Windows spawns
DataLoader workers rather than forking, so the dataset is pickled once per
worker -- and np.memmap pickles *by value*, which handed each worker a private
copy of the whole corpus instead of a shared view of the file. At 384 that
overflowed the spawn pipe outright (OSError 22); at 256 it merely wasted
several GB per worker. Each worker reopens the file itself instead.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import coords, data


class TrussDataset(Dataset):
    def __init__(self, split, root=data.DEFAULT_ROOT, mirror=False, noise=0.0, seed=0):
        targets = data.load_targets(split, root)
        self.split, self.root = split, root
        self._images = data.load_images(split, root)
        if len(self._images) != targets['n']:
            raise ValueError(f'{split}: {len(self._images)} images vs {targets["n"]} targets')
        self._n = targets['n']

        self.start = coords.position_to_model(targets['start']).astype(np.float32)
        self.delta = coords.delta_to_model(targets['end'] - targets['start']).astype(np.float32)
        self.mirror = mirror
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    @property
    def images(self):
        """The memmap, reopened on first use in whichever process asks."""
        if self._images is None:
            self._images = data.load_images(self.split, self.root, verbose=False)
        return self._images

    def __getstate__(self):
        # Send the worker everything except the corpus itself.
        return {**self.__dict__, '_images': None}

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        frame = np.asarray(self.images[i])            # (H, W, 3) uint8
        start = self.start[i].copy()
        delta = self.delta[i].copy()

        # Mirroring in x is an exact symmetry of this problem: the supports are
        # argmin/argmax of x so they simply swap, dx negates and dy is
        # unchanged. Free 2x data, and a wrong sign here would show up as a
        # model that cannot predict horizontal motion at all.
        if self.mirror and self._rng.random() < 0.5:
            frame = frame[:, ::-1]
            start[0] = -start[0]
            delta[0] = -delta[0]

        image = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()
        image = image / 127.5 - 1.0                   # roughly [-1, 1]

        if self.noise > 0:
            # Cheap stand-in for rasteriser variation: browsers do not all
            # antialias identically, so the model should not depend on exact
            # pixel values.
            image = image + torch.from_numpy(
                self._rng.normal(0.0, self.noise, image.shape).astype(np.float32)
            )

        return image, torch.from_numpy(start), torch.from_numpy(delta)
