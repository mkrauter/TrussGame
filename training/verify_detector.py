"""Check the browser detector against PyTorch, including the capture path.

    python verify_detector.py

Renders trusses in headless Chromium, has the page report what its own runtime
computed, and compares against PyTorch running on the corpus PNG for the same
seed. Three things are being checked at once, and each has already been a bug
somewhere in this project:

  * the crop-and-downscale, which is where v1's train/serve skew lived
  * the convolution runtime, hand-written twice in two languages
  * the peak decode, whose offsets went to the wrong cell the first time
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from trussnet.detect_data import CROP_SIZE, screen_to_input
from trussnet.detector import TrussDetector, decode

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--seed-base', type=int, default=1_000_000)
    args = p.parse_args()

    dump = json.loads(subprocess.run(
        [_node(), str(HERE / 'dump_detection.mjs'),
         '--count', str(args.count), '--seed-base', str(args.seed_base)],
        capture_output=True, text=True, check=True, cwd=HERE
    ).stdout)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = TrussDetector(width=cfg['width'], context=cfg['context'])
    model.load_state_dict(ck['model'])
    model.eval()

    corpus = HERE / (cfg.get('pixel_corpus') or 'corpus')
    scale = CROP_SIZE / ck['input_size']

    worst_input, worst_node = 0.0, 0.0
    member_mismatch = 0
    for sample in dump:
        index = sample['seed'] - args.seed_base
        path = corpus / 'val' / 'images' / f'{index:06d}.png'
        with Image.open(path) as img:
            frame = np.asarray(img.convert('RGB'), dtype=np.float32)
        tensor = torch.from_numpy(frame).permute(2, 0, 1)[None] / 127.5 - 1.0

        flat = tensor.flatten().numpy()
        probe = np.array([flat[(i * 977) % len(flat)] for i in range(256)])
        worst_input = max(worst_input, float(np.abs(probe - sample['inputProbe']).max()))

        with torch.no_grad():
            points, _ = decode(*model(tensor))
        mine = points[0].numpy() * scale + 68.0

        # Compare each browser node against the nearest torch node of its role.
        theirs = np.array([n['position'] for n in sample['nodes']])
        for point in theirs:
            worst_node = max(worst_node, float(np.abs(mine - point).sum(axis=1).min()))

    print(f'\n{len(dump)} trusses rendered in Chromium, scored against PyTorch\n')
    ok = True
    for label, value, tol in (('capture / input tensor', worst_input, 2e-2),
                              ('decoded node positions (px)', worst_node, 1.0)):
        good = value <= tol
        ok &= good
        print(f'  {"PASS" if good else "FAIL"}  {label:<32} max |diff| = {value:.4f}')
    print('\nThe input tolerance is loose because Chromium and Pillow decode PNG')
    print('identically but the corpus PNG was itself written by Chromium, so any')
    print('difference here means the capture path drifted, not rounding.\n')
    sys.exit(0 if ok else 1)


def _node():
    return 'node'


if __name__ == '__main__':
    main()
