"""Train stage one: the truss detector, whose only input is the screenshot.

    python train_detector.py --epochs 20

Reported metric is what matters downstream: median and 95th-percentile node
localisation error **in screen pixels**, because that is the unit the acuity
curve is in. A perfect solver reading node positions with 1px of error scores
99%, with 3px 97%. So sub-pixel is the target, and anything above ~2px means
the detector, not the physics, is setting the ceiling.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from trussnet.detect_data import CROP_SIZE, STRIDE, TrussDetection
from trussnet.detector import TrussDetector, count_parameters, decode

HERE = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet's penalty-reduced focal loss.

    Cells near a node are 'less wrong' than cells far from one, which plain BCE
    cannot express -- and with ~1% of cells positive, plain BCE simply predicts
    background everywhere.
    """
    pred = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1.0).float()
    negative = 1.0 - positive
    pos_loss = -((1 - pred) ** alpha) * pred.log() * positive
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * (1 - pred).log() * negative
    n = positive.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


@torch.no_grad()
def evaluate(model, loader, device, input_size):
    """Localisation error per node, in screen pixels, after matching to truth."""
    model.eval()
    scale = CROP_SIZE / input_size
    errors, role_errors = [], []
    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        points, _ = decode(*model(image))
        points = points.cpu() * scale

        truth = batch['points'] * scale
        kind = batch['kind']
        for b in range(len(points)):
            # Decoded order is 7 free, 2 supports, 1 loaded. Compare each group
            # against the true nodes of that role, pairing greedily by distance.
            offset = 0
            for role, count in enumerate((7, 2, 1)):
                got = points[b, offset:offset + count]
                want = truth[b][kind[b] == role]
                offset += count
                if len(want) == 0:
                    continue
                # Greedy nearest-pair matching: repeatedly take the closest
                # detection/truth pair that is still unclaimed. With 10 nodes
                # 100px apart this agrees with the optimal assignment, and it
                # cannot silently reuse a detection the way the previous
                # version could.
                cost = torch.cdist(got, want).clone()
                for _ in range(min(len(got), len(want))):
                    idx = int(cost.argmin())
                    i, j = idx // cost.shape[1], idx % cost.shape[1]
                    d = float(cost[i, j])
                    errors.append(d)
                    if role == 2:
                        role_errors.append(d)
                    cost[i, :] = float('inf')
                    cost[:, j] = float('inf')
    errors = np.array(errors)
    return {
        'median': float(np.median(errors)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(errors.max()),
        'loaded_median': float(np.median(role_errors)) if role_errors else float('nan'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--offset-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--cpu', action='store_true')
    # Which rendered corpus to detect on. Resolution matters here in a way it
    # did not for the physics: the heatmap resolves position to a fraction of an
    # input pixel, and at 256 one input pixel is 3 screen pixels.
    p.add_argument('--pixel-corpus', default=None)
    p.add_argument('--out', default=str(HERE / 'runs_detector'))
    args = p.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    seed_everything(args.seed)

    kwargs = {} if args.pixel_corpus is None else {'pixel_root': Path(args.pixel_corpus)}
    if args.smoke:
        train_set = Subset(TrussDetection('val', **kwargs), range(8))
        val_set = train_set
        args.epochs = max(args.epochs, 200)
        args.batch = 8
    else:
        train_set = TrussDetection('train', mirror=True, seed=args.seed, **kwargs)
        val_set = TrussDetection('val', **kwargs)

    input_size = (train_set.dataset if args.smoke else train_set).input_size
    model = TrussDetector(width=args.width, context=args.context).to(device)
    print(f'TrussDetector: {count_parameters(model):,} parameters, {input_size}px input, '
          f'heatmap stride {STRIDE}')
    print(f'device: {device.type}\n')

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=not args.smoke,
                              num_workers=args.workers, drop_last=not args.smoke,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out) / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(
        {**vars(args), 'params': count_parameters(model), 'input_size': input_size}, indent=1))

    best, history = float('inf'), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            image = batch['image'].to(device, non_blocking=True)
            heat_t = batch['heatmap'].to(device)
            off_t = batch['offset'].to(device)
            present = batch['present'].to(device).unsqueeze(1)

            heat_p, off_p = model(image)
            loss = focal_loss(heat_p, heat_t)
            # Offsets only mean anything where a node actually is.
            loss = loss + args.offset_weight * (
                (F.l1_loss(off_p, off_t, reduction='none') * present).sum()
                / present.sum().clamp(min=1) / 2
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(image)
            n += len(image)
        sched.step()

        stats = evaluate(model, val_loader, device, input_size)
        history.append({'epoch': epoch, 'train_loss': running / n, **stats})
        flag = ''
        if stats['median'] < best:
            best = stats['median']
            torch.save({'model': model.state_dict(), 'config': vars(args),
                        'input_size': input_size, 'median_px': best, 'epoch': epoch},
                       out / 'best.pt')
            flag = '  <- best'
        (out / 'history.json').write_text(json.dumps(history, indent=1))
        print(f'epoch {epoch:>3}/{args.epochs}  loss {running/n:.4f}  '
              f'median {stats["median"]:.2f}px  p95 {stats["p95"]:.2f}px  '
              f'max {stats["max"]:.1f}px  {time.time()-t0:.0f}s{flag}')

    print(f'\nbest median localisation {best:.2f} screen px')
    print('acuity reference: 1px of node error costs ~1 point of game score, 3px ~3 points')
    print(f'artifacts in {out}')


if __name__ == '__main__':
    main()
