"""Compare two v2 candidates on the scores the browser port actually produced.

    node eval_v2.mjs --count 600 --model trussv2-old --dump old.json
    node eval_v2.mjs --count 600 --model trussv2     --dump new.json
    python paired_v2.py

Both runs use the same seeds, so the comparison is paired. That matters here:
the standard error on either mean is around a point, which is the size of the
difference being claimed, and comparing the two means alone would not settle it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    old = np.array(json.loads((HERE / 'old.json').read_text()))
    new = np.array(json.loads((HERE / 'new.json').read_text()))
    if len(old) != len(new):
        raise SystemExit(f'{len(old)} vs {len(new)} rounds -- not the same trusses')

    print(f'{len(old)} trusses, as the v2 page renders them\n')
    for name, scores in (('the 2023 model in the repo', old), ('retrained_250', new)):
        print(f'  {name:<28} mean {scores.mean():5.2f}%   '
              f'median {np.median(scores):5.2f}%   zeros {(scores == 0).sum()}')

    difference = new - old
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(difference, difference.size, replace=True).mean()
                     for _ in range(10000)])
    low, high = np.percentile(boot, [2.5, 97.5])
    verdict = 'real' if low > 0 or high < 0 else 'not resolvable at this sample size'
    print(f'\n  retrained_250 advantage: {difference.mean():+.2f} points   '
          f'95% CI [{low:+.2f}, {high:+.2f}]   -> {verdict}')
    print(f'  better on {(difference > 0).mean() * 100:.0f}% of individual rounds')


if __name__ == '__main__':
    main()
