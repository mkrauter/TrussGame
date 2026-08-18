"""Loading the corpus produced by generate_corpus.mjs.

Targets are cheap to load. Images are not: 20k PNGs decode slowly and occupy
~3.9 GB as raw uint8, so they are decoded once into a .npy and memory-mapped
afterwards. The cache is derived data -- delete it and it rebuilds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# What web/src/config.js currently ships as MODEL.inputSize. The corpus's own
# resolution is read off its PNGs rather than trusted to match, so an
# experimental corpus generated with --size loads without editing this file.
DECLARED_INPUT_SIZE = 256

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / 'corpus'


def load_targets(split, root=DEFAULT_ROOT):
    payload = json.loads((Path(root) / split / 'targets.json').read_text(encoding='utf-8'))
    rows = payload['targets']

    def column(key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    return {
        'meta': payload['meta'],
        'split': split,
        'n': len(rows),
        'seed': np.array([r['seed'] for r in rows], dtype=np.int64),
        'start': column('start'),
        'end': column('end'),
        'support_a': column('supportA'),
        'support_b': column('supportB'),
    }


def image_paths(split, root=DEFAULT_ROOT):
    folder = Path(root) / split / 'images'
    return sorted(folder.glob('*.png'))


def detect_input_size(split, root=DEFAULT_ROOT):
    """The corpus's own resolution, read from its first PNG.

    Every image is checked against this while decoding, so a corpus mixing
    resolutions fails loudly instead of silently training on garbage.
    """
    paths = image_paths(split, root)
    if not paths:
        raise FileNotFoundError(f'no PNGs under {Path(root) / split / "images"} -- run generate_corpus.mjs first')
    with Image.open(paths[0]) as img:
        width, height = img.size
    if width != height:
        raise ValueError(f'{paths[0].name} is {width}x{height}; the model input is square')
    return width


def load_images(split, root=DEFAULT_ROOT, rebuild=False, verbose=True):
    """Decode the split's PNGs once, then memory-map the result.

    Returns a uint8 array of shape (N, size, size, 3) at the corpus's own
    resolution. Memory-mapped, so it does not consume RAM until touched -- a
    20k split at 256 is 3.9 GB decoded, and 8.8 GB at 384.
    """
    root = Path(root)
    size = detect_input_size(split, root)
    cache = root / split / f'images_{size}.npy'

    if cache.exists() and not rebuild:
        return np.load(cache, mmap_mode='r')

    paths = image_paths(split, root)

    array = np.lib.format.open_memmap(
        cache, mode='w+', dtype=np.uint8, shape=(len(paths), size, size, 3)
    )
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            frame = np.asarray(img.convert('RGB'), dtype=np.uint8)
        if frame.shape != (size, size, 3):
            raise ValueError(f'{path.name} is {frame.shape}, expected {(size, size, 3)}')
        array[i] = frame
        if verbose and (i + 1) % 2000 == 0:
            print(f'  decoded {i + 1}/{len(paths)}', flush=True)

    array.flush()
    return np.load(cache, mmap_mode='r')


def load_split(split, root=DEFAULT_ROOT, images=False, **kwargs):
    data = load_targets(split, root)
    if images:
        data['images'] = load_images(split, root, **kwargs)
        if len(data['images']) != data['n']:
            raise ValueError(
                f'{split}: {len(data["images"])} images but {data["n"]} targets -- '
                'the corpus is inconsistent, regenerate it'
            )
    return data
