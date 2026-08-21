"""How much does a 50-round average of this game wander?

    node eval_v2.mjs --count 800 --random 1 --dump spread.json
    python spread_of_short_runs.py

A player who has just finished fifty rounds naturally reads the average on
screen as "how good this opponent is". It is not, quite: the per-round score is
enormously spread -- anywhere from 0 to 100 -- so a fifty-round average carries
a standard error of several points and short runs land well away from the truth
in both directions.

This resamples real per-round scores to show how wide that window is, which is
the only honest way to answer "I saw 71%, your test says 61%".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    scores = np.array(json.loads((HERE / 'spread.json').read_text()))
    print(f'{len(scores)} rounds of real per-round scores')
    print(f'  long-run mean      {scores.mean():.2f}%')
    print(f'  per-round spread   sd {scores.std():.1f}, '
          f'quartiles {np.percentile(scores, 25):.0f}% / '
          f'{np.percentile(scores, 50):.0f}% / {np.percentile(scores, 75):.0f}%')
    print(f'  rounds scoring 0   {(scores == 0).mean() * 100:.1f}%')
    print(f'  rounds above 90    {(scores > 90).mean() * 100:.1f}%\n')

    rng = np.random.default_rng(0)
    for n in (25, 50, 100, 400):
        means = np.array([rng.choice(scores, n, replace=True).mean() for _ in range(20000)])
        low, high = np.percentile(means, [5, 95])
        print(f'  a {n:>3}-round average lands between {low:5.1f}% and {high:5.1f}% '
              f'nine times in ten   (sd {means.std():.1f})')

    print()
    for target in (68, 70, 71, 75):
        means = np.array([rng.choice(scores, 50, replace=True).mean() for _ in range(40000)])
        share = (means >= target).mean() * 100
        print(f'  fifty rounds averaging {target}% or better: {share:.1f}% of the time')


if __name__ == '__main__':
    main()
