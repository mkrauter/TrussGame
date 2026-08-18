"""Export a trained TrussGNN to a JSON file the browser game can load.

    python export_gnn.py runs_gnn/20260818-213323/best.pt

Weights go out as base64 little-endian float32 rather than JSON numbers: exact,
and about a third of the size. The layer list is emitted in execution order so
the JS side never has to know PyTorch's module naming -- it just walks it.

The point of this file is that v2/src/gnn.js and trussnet/gnn.py must compute
the same function. `verify_export.py` checks that they do, on real trusses,
rather than trusting that they do.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from trussnet.gnn import TrussGNN


def b64(tensor):
    return base64.b64encode(np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype('<f4')
    ).tobytes()).decode('ascii')


def dump_mlp(module):
    """A Sequential of Linear/SiLU -> a list of layers in execution order."""
    layers = []
    for child in module:
        if isinstance(child, torch.nn.Linear):
            layers.append({
                'type': 'linear',
                'in': child.in_features,
                'out': child.out_features,
                'weight': b64(child.weight),      # row-major, (out, in)
                'bias': b64(child.bias),
            })
        elif isinstance(child, torch.nn.SiLU):
            layers.append({'type': 'silu'})
        else:
            raise TypeError(f'unexpected layer in MLP: {type(child).__name__}')
    return layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent.parent
                                       / 'v2' / 'src' / 'model' / 'trussgnn.json'))
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = TrussGNN(hidden=cfg['hidden'], rounds=cfg['rounds'], shared=not cfg['unshared'])
    model.load_state_dict(ck['model'])
    model.eval()

    if cfg['unshared']:
        raise SystemExit('the browser runtime assumes shared rounds; retrain without --unshared')

    payload = {
        'format': 'trussgnn/1',
        'hidden': cfg['hidden'],
        'rounds': cfg['rounds'],
        # The perceptual handicap the model was trained under. The game applies
        # the same jitter, so the AI plays with the eyes it learned with.
        'sigma': cfg['sigma'],
        'valMean': ck['val_mean'],
        'epoch': ck['epoch'],
        'layerNormEps': model.norm.eps,
        'nodeEncoder': dump_mlp(model.node_encoder),
        'edgeEncoder': dump_mlp(model.edge_encoder),
        'message': dump_mlp(model.message[0]),
        'update': dump_mlp(model.update[0]),
        'normWeight': b64(model.norm.weight),
        'normBias': b64(model.norm.bias),
        'decoder': dump_mlp(model.decoder),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding='utf-8')
    n = sum(p.numel() for p in model.parameters())
    print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {n:,} parameters, '
          f'{cfg["rounds"]} rounds, sigma {cfg["sigma"]}px, val {ck["val_mean"]:.1f}%)')


if __name__ == '__main__':
    main()
