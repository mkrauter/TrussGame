"""Export a trained TrussDetector for the browser.

    python export_detector.py runs_detector/20260818-221701/best.pt

Batch norm is folded into the convolution that precedes it. BN at inference is
an affine map per channel, and so is a convolution's weight-and-bias, so the two
compose exactly:

    w' = w * gamma / sqrt(var + eps)
    b' = beta - gamma * mean / sqrt(var + eps)

That halves the number of passes over each feature map and means convnet.js
only ever has to implement conv and SiLU. `verify_detector.mjs` checks the
folded model against PyTorch rather than trusting the algebra.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trussnet.detect_data import CLASSES, STRIDE
from trussnet.detector import COUNTS, TrussDetector


def b64(array):
    return base64.b64encode(
        np.ascontiguousarray(array.astype('<f4')).tobytes()
    ).decode('ascii')


def fold(conv, bn):
    """conv (bias-free) followed by batch norm -> one conv with bias."""
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    weight = conv.weight.detach().numpy() * scale[:, None, None, None]
    bias = beta - mean * scale
    return weight, bias


def conv_op(conv, weight, bias):
    return {
        'type': 'conv',
        'cin': conv.in_channels,
        'cout': conv.out_channels,
        'k': conv.kernel_size[0],
        'stride': conv.stride[0],
        'dilation': conv.dilation[0],
        'pad': conv.padding[0],
        'weight': b64(weight),
        'bias': b64(bias),
    }


def walk(sequential, ops):
    """Turn Sequential(Conv, BN, SiLU) blocks into folded conv + silu ops."""
    children = list(sequential)
    i = 0
    while i < len(children):
        node = children[i]
        if isinstance(node, nn.Sequential):
            walk(node, ops)
            i += 1
        elif isinstance(node, nn.Conv2d):
            if i + 1 < len(children) and isinstance(children[i + 1], nn.BatchNorm2d):
                ops.append(conv_op(node, *fold(node, children[i + 1])))
                i += 2
            else:
                ops.append(conv_op(node, node.weight.detach().numpy(),
                                   node.bias.detach().numpy()))
                i += 1
        elif isinstance(node, nn.SiLU):
            ops.append({'type': 'silu'})
            i += 1
        else:
            raise TypeError(f'cannot export {type(node).__name__}')
    return ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent.parent
                                        / 'web' / 'src' / 'model' / 'trussdetector.json'))
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = TrussDetector(width=cfg['width'], context=cfg['context'])
    model.load_state_dict(ck['model'])
    model.eval()

    trunk = walk(nn.Sequential(model.stem, model.context), [])
    payload = {
        'format': 'trussdetector/1',
        'inputSize': ck['input_size'],
        'stride': STRIDE,
        'classes': CLASSES,
        'counts': list(COUNTS),
        'medianPx': ck['median_px'],
        'epoch': ck['epoch'],
        'trunk': trunk,
        # Two heads off the same features, so they cannot be a flat op list.
        'heatmap': conv_op(model.heatmap, model.heatmap.weight.detach().numpy(),
                           model.heatmap.bias.detach().numpy()),
        'offset': conv_op(model.offset, model.offset.weight.detach().numpy(),
                          model.offset.bias.detach().numpy()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding='utf-8')
    n = sum(q.numel() for q in model.parameters())
    print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {n:,} parameters, '
          f'{ck["input_size"]}px input, median {ck["median_px"]:.2f}px)')


if __name__ == '__main__':
    main()
