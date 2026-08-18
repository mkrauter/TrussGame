"""Train the graph model.

    python train_gnn.py --epochs 30
    python train_gnn.py --epochs 30 --sigma 0      # exact geometry, for the ceiling
    python train_gnn.py --smoke                    # overfit 8 trusses

Scoring is deliberately end-to-end and honest about perception: the model reads
jittered coordinates, so it clicks at the node where it *believes* the node is,
plus the displacement it predicts, and that click is scored against the truly
settled position. Perception error therefore counts against it twice, exactly as
it would for a player.
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

from trussnet import metrics
from trussnet.gnn import TrussGNN, count_parameters, physics_residual
from trussnet.graph_data import DEFAULT_ROOT, TrussGraphs

HERE = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, rounds=None):
    model.eval()
    scores, residuals = [], []
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch, rounds=rounds)
        residuals.append(physics_residual(pred, batch).abs().amax(dim=(1, 2)).cpu())

        loaded = batch['loaded'].view(-1, 1, 1).expand(-1, 1, 2)
        u = pred.gather(1, loaded).squeeze(1) * batch['span'].unsqueeze(-1)
        click = batch['seen_start'] + u          # where it thinks it should click
        scores.append(
            torch.from_numpy(
                metrics.accuracy(
                    batch['true_start'].cpu().numpy(),
                    batch['true_end'].cpu().numpy(),
                    click.cpu().numpy(),
                )
            )
        )
    scores = torch.cat(scores).numpy()
    return metrics.summarise(scores), float(torch.cat(residuals).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--unshared', action='store_true',
                   help='give each round its own weights; costs the iteration reading')
    # The fairness dial, in screen pixels of assumed perceptual error. ~2-3 is
    # roughly what a human eye extracts from a 900px canvas; 0 is exact
    # geometry, which measurement puts only ~2 points of score above human.
    p.add_argument('--sigma', type=float, default=2.5)
    # Chance that a training sample's member list is one edge wrong, matching
    # what the detector produces (~12% of frames) so the model trains on the
    # inputs it will actually be given.
    p.add_argument('--member-noise', type=float, default=0.0)
    p.add_argument('--physics-weight', type=float, default=0.1)
    p.add_argument('--beta', type=float, default=0.02, help='Huber transition point')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--corpus', default=str(DEFAULT_ROOT))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out', default=str(HERE / 'runs_gnn'))
    args = p.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    seed_everything(args.seed)

    model = TrussGNN(hidden=args.hidden, rounds=args.rounds, shared=not args.unshared).to(device)
    print(f'TrussGNN: {count_parameters(model):,} parameters, {args.rounds} message-passing '
          f'rounds ({"shared" if not args.unshared else "unshared"}), graph diameter is 3')
    print(f'perception: sigma {args.sigma}px   device: {device.type}\n')

    root = Path(args.corpus)
    if args.smoke:
        train_set = Subset(TrussGraphs('val', root=root, sigma=args.sigma, seed=args.seed), range(8))
        val_set = train_set
        args.epochs = max(args.epochs, 400)
    else:
        train_set = TrussGraphs('train', root=root, sigma=args.sigma, seed=args.seed,
                                member_noise=args.member_noise)
        # Validation keeps perfect members: it measures the physics, and the
        # end-to-end pixel harness measures what perception costs on top.
        val_set = TrussGraphs('val', root=root, sigma=args.sigma, seed=args.seed + 1)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=not args.smoke,
                              num_workers=args.workers, drop_last=not args.smoke)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out) / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'params': count_parameters(model)}, indent=1))

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            batch = to_device(batch, device)
            pred = model(batch)
            free = batch['free_mask'].unsqueeze(-1)
            # Supervised on the whole displacement field: 8 free nodes rather
            # than the 1 the pixel corpus kept, from the same trusses.
            supervised = (F.smooth_l1_loss(pred, batch['target'], beta=args.beta, reduction='none')
                          * free).sum() / free.sum().clamp(min=1) / 2
            equilibrium = physics_residual(pred, batch).pow(2).mean()
            loss = supervised + args.physics_weight * equilibrium

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(batch['target'])
            n += len(batch['target'])
        sched.step()
        train_loss = running / n

        stats, residual = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'residual': residual, **stats})
        flag = ''
        if stats['mean'] > best:
            best = stats['mean']
            torch.save({'model': model.state_dict(), 'config': vars(args),
                        'val_mean': best, 'epoch': epoch}, out / 'best.pt')
            flag = '  <- best'
        # Written every epoch: the interrupted 60-epoch pixel run lost its whole
        # history to a single kill because this only happened at the end.
        (out / 'history.json').write_text(json.dumps(history, indent=1))
        print(f'epoch {epoch:>3}/{args.epochs}  loss {train_loss:.5f}  '
              f'val {stats["mean"]:.1f}% (median {stats["median"]:.1f}%)  '
              f'|Ku-f| {residual:.3f}  {time.time()-t0:.0f}s{flag}')

    print(f'\nbest validation accuracy {best:.1f}%   (pixel CNN 77.0%, baseline 59.5%)')

    # Fewer rounds than trained with is a less-converged solver, and the
    # difficulty dial the design exists to provide.
    ck = torch.load(out / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    if not args.unshared:
        print('\nrounds at inference (trained at %d):' % args.rounds)
        for r in (1, 2, 3, 4, 6, 8, args.rounds, args.rounds + 5, args.rounds * 2):
            stats, residual = evaluate(model, val_loader, device, rounds=r)
            print(f'  {r:>3} rounds   {stats["mean"]:>5.1f}%   median {stats["median"]:>5.1f}%'
                  f'   |Ku-f| {residual:.3f}')
    print(f'\nartifacts in {out}')


if __name__ == '__main__':
    main()
