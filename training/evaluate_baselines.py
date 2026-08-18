"""What any model has to beat.

Constants are fitted on the training split and scored on validation, so nothing
here is graded on data it was fitted to.

    python evaluate_baselines.py
"""
from __future__ import annotations

import numpy as np

from trussnet import data, metrics


def main():
    train = data.load_targets('train')
    val = data.load_targets('val')
    print(f'fitted on train (n={train["n"]}), scored on val (n={val["n"]}), '
          f'seed bases {train["meta"]["seedBase"]} / {val["meta"]["seedBase"]}\n')

    fitted = metrics.fit_baselines(train)
    print('constants fitted on train:')
    print(f'  mean displacement       ({fitted["mean_delta"][0]:+.2f}, {fitted["mean_delta"][1]:+.2f}) px')
    print(f'  mean travel             {fitted["mean_travel"]:.2f} px')
    print(f'  mean travel / span      {fitted["mean_travel_over_span"]:.4f}\n')

    preds = metrics.baseline_predictions(val, fitted)
    rows = [(name, metrics.summarise(metrics.accuracy(val['start'], val['end'], guess)))
            for name, guess in preds.items()]
    rows.sort(key=lambda r: r[1]['mean'])

    head = f'{"predictor":<40}{"mean":>8}{"median":>9}{"p25":>8}{"p75":>8}{"zeros":>8}'
    print(head)
    print('-' * len(head))
    for name, s in rows:
        print(f'{name:<40}{s["mean"]:>7.1f}%{s["median"]:>8.1f}%'
              f'{s["p25"]:>7.1f}%{s["p75"]:>7.1f}%{s["zeros"]:>7.1f}%')

    fair = [r for r in rows if not r[0].startswith('[oracle]')]
    best_name, best = max(fair, key=lambda r: r[1]['mean'])
    print(f'\nbar to beat: {best["mean"]:.1f}% mean ({best_name})')
    print(f'target band: 70-80% -- roughly average human play')

    span = metrics.support_span(val['support_a'], val['support_b'])
    travel = np.linalg.norm(val['end'] - val['start'], axis=-1)
    print(f'\nwhy span-scaling might help: travel varies {travel.std() / travel.mean() * 100:.0f}% '
          f'about its mean, but travel/span only {(travel / span).std() / (travel / span).mean() * 100:.0f}%')


if __name__ == '__main__':
    main()
