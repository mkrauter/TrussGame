"""Train the v3 model.

    python train.py --smoke              # overfit 8 samples, ~2 min, proves the plumbing
    python train.py --epochs 60          # a real run

Reproducibility rules this follows, all of them lessons from v1:
  - every RNG is seeded, and the config is written next to the result
  - the validation split is generated once from a fixed seed and never
    regenerated, so two runs sit the same exam
  - checkpoints go on best validation accuracy, not the final epoch; v1's
    val_mae swung +/-15% between consecutive epochs and saving the last one was
    a lottery
  - the GPU is asserted, not silently fallen back on
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

from trussnet import coords, data, metrics
from trussnet.dataset import TrussDataset
from trussnet.model import TrussNet, count_parameters, receptive_field

HERE = Path(__file__).resolve().parent


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(pred, start, delta, weights, beta):
    p_start, p_delta, p_final = pred
    l_start = F.smooth_l1_loss(p_start, start, beta=beta)
    l_delta = F.smooth_l1_loss(p_delta, delta, beta=beta)
    l_final = F.smooth_l1_loss(p_final, start + delta, beta=beta)
    total = weights[0] * l_start + weights[1] * l_delta + weights[2] * l_final
    return total, (l_start.item(), l_delta.item(), l_final.item())


@torch.no_grad()
def evaluate(model, loader, device, targets):
    """Score on the game's own metric, in screen pixels."""
    model.eval()
    finals, starts = [], []
    for image, _, _ in loader:
        p_start, _, p_final = model(image.to(device, non_blocking=True))
        finals.append(p_final.float().cpu())
        starts.append(p_start.float().cpu())

    pred_final = coords.position_from_model(torch.cat(finals).numpy())
    pred_start = coords.position_from_model(torch.cat(starts).numpy())
    scores = metrics.accuracy(targets['start'], targets['end'], pred_final)
    localisation = float(np.median(np.abs(pred_start[:, 0] - targets['start'][:, 0])))
    return metrics.summarise(scores), localisation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=0.1, help='Huber transition point')
    # delta is weighted above start deliberately: localisation is already easy,
    # and an unweighted loss lets it dominate a task it has nothing left to learn.
    p.add_argument('--weights', type=float, nargs=3, default=[1.0, 2.0, 1.0],
                   metavar=('START', 'DELTA', 'FINAL'))
    # Dilation is the one receptive-field lever that costs no parameters, which
    # is what makes it adjustable at all under a fixed budget. It has to move
    # with the input resolution: the crop is fixed, so a larger input scales the
    # support span up while the receptive field stays where it is.
    p.add_argument('--dilations', type=int, nargs='+', default=[1, 1, 1, 2, 4])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mirror', action='store_true', default=True)
    p.add_argument('--no-mirror', dest='mirror', action='store_false')
    p.add_argument('--noise', type=float, default=0.0)
    p.add_argument('--smoke', action='store_true', help='overfit 8 samples; loss must reach ~0')
    p.add_argument('--cpu', action='store_true')
    # Which corpus to train on. Points elsewhere for resolution experiments;
    # the model is resolution-agnostic (soft-argmax, no flatten) and targets are
    # normalised against the crop, so only the pixels change.
    p.add_argument('--corpus', default=str(data.DEFAULT_ROOT))
    p.add_argument('--out', default=str(HERE / 'runs'))
    args = p.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit('CUDA is not available. Pass --cpu to train on CPU deliberately.')
    device = torch.device('cpu' if args.cpu else 'cuda')

    seed_everything(args.seed)
    model = TrussNet(dilations=tuple(args.dilations)).to(device)
    n_params = count_parameters(model)
    # The span was measured as 190-233px at 256 input; it scales with the input
    # because the crop is fixed, while the receptive field does not. Printed
    # together because whether one covers the other is the point of the design.
    input_size = data.detect_input_size('val', Path(args.corpus))
    span = (190 * input_size / 256, 233 * input_size / 256)
    print(f'TrussNet: {n_params:,} parameters, receptive field '
          f'{receptive_field(tuple(args.dilations))}px '
          f'(support span is {span[0]:.0f}-{span[1]:.0f}px at {input_size}px input)')
    print(f'device: {torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}\n')

    corpus = Path(args.corpus)
    if args.smoke:
        train_set = Subset(TrussDataset('val', root=corpus, mirror=False), list(range(8)))
        val_set = train_set
        val_targets = {k: v[:8] for k, v in data.load_targets('val', corpus).items()
                       if isinstance(v, np.ndarray)}
        args.epochs = max(args.epochs, 300)
        args.batch = 8
    else:
        train_set = TrussDataset('train', root=corpus, mirror=args.mirror,
                                 noise=args.noise, seed=args.seed)
        val_set = TrussDataset('val', root=corpus, mirror=False)
        val_targets = data.load_targets('val', corpus)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, drop_last=not args.smoke,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out) / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({**vars(args), 'params': n_params}, indent=1))

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, parts, n = 0.0, np.zeros(3), 0
        t0 = time.time()
        for image, start, delta in train_loader:
            image = image.to(device, non_blocking=True)
            start, delta = start.to(device), delta.to(device)
            loss, split = loss_fn(model(image), start, delta, args.weights, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(image)
            parts += np.array(split) * len(image)
            n += len(image)
        sched.step()
        train_loss = running / n

        if args.smoke:
            print(f'  epoch {epoch:>4}  loss {train_loss:.6f}  '
                  f'(start {parts[0]/n:.6f}  delta {parts[1]/n:.6f})')
            if epoch % 50 == 0 and train_loss < 1e-4:
                print('\nSMOKE TEST PASSED: the model can fit 8 samples to ~zero.')
                return
            continue

        stats, localisation = evaluate(model, val_loader, device, val_targets)
        history.append({'epoch': epoch, 'train_loss': train_loss, **stats})
        flag = ''
        if stats['mean'] > best:
            best = stats['mean']
            torch.save({'model': model.state_dict(), 'config': vars(args),
                        'val_mean': best, 'epoch': epoch}, out / 'best.pt')
            flag = '  <- best'
        print(f'epoch {epoch:>3}/{args.epochs}  loss {train_loss:.5f}  '
              f'val {stats["mean"]:.1f}% (median {stats["median"]:.1f}%)  '
              f'loc {localisation:.0f}px  {time.time()-t0:.0f}s{flag}')

    (out / 'history.json').write_text(json.dumps(history, indent=1))
    print(f'\nbest validation accuracy {best:.1f}%   (baseline to beat 59.5%, target 70-80%)')
    print(f'artifacts in {out}')


if __name__ == '__main__':
    main()
