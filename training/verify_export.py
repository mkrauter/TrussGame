"""Dump what PyTorch computes, so the browser runtime can be checked against it.

    python verify_export.py            # writes expected.json
    node verify_export.mjs             # rebuilds it in JS and compares

Two implementations of one function is exactly the shape of v1's train/serve
skew bug. The difference is that this one is checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trussnet.graph_data import TrussGraphs
from trussnet.gnn import TrussGNN

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    # Newest run by default. A hardcoded path here once compared a freshly
    # exported model against a stale checkpoint and reported a failure that was
    # entirely the harness's own.
    p.add_argument('--checkpoint', default=str(sorted(
        (HERE / 'runs_gnn').glob('*/best.pt'), key=lambda q: q.stat().st_mtime)[-1]))
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--out', default=str(HERE / 'expected.json'))
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = TrussGNN(hidden=cfg['hidden'], rounds=cfg['rounds'], shared=not cfg['unshared'])
    model.load_state_dict(ck['model'])
    model.eval()

    # sigma 0: this checks that the two implementations compute the same
    # function, which is separate from whether their random jitter agrees.
    ds = TrussGraphs('val', sigma=0.0)
    samples = []
    for i in range(args.count):
        item = ds[i]
        batch = {k: v.unsqueeze(0) for k, v in item.items()}
        with torch.no_grad():
            out = model(batch)[0]
        samples.append({
            'seed': int(ds.nodes.shape[0] and 1_000_000 + i),
            'span': float(item['span']),
            'nodeFeat': item['node_feat'].flatten().tolist(),
            'edgeFeat': item['edge_feat'].flatten().tolist(),
            'edgeIndex': item['edge_index'].flatten().tolist(),
            'output': out.flatten().tolist(),
        })

    Path(args.out).write_text(json.dumps({
        'rounds': cfg['rounds'],
        'maxEdges': ds.max_edges,
        'samples': samples,
    }), encoding='utf-8')
    print(f'wrote {args.out}: {len(samples)} trusses from the val split')


if __name__ == '__main__':
    main()
