"""Settle whether retrained_250 is actually better on v2's own geometry.

    venv/Scripts/python.exe training/settle_v2.py

At 600 rounds the difference was +1.18 points with a confidence interval
straddling zero -- which says nothing either way. This runs both models over
2400 trusses rendered in v2's layout (offset 100,150, the one these models were
trained on) through LiteRT, which is ten times faster than the browser path and
so affords the sample size the question actually needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

HERE = Path(__file__).resolve().parent
CORPUS = HERE / 'corpus_v2geom' / 'val'
MODELS = {
    'the 2023 model in the repo': HERE.parent / 'truss_game_v2_model.tflite',
    'retrained_250 (recovered)': HERE / 'recovered' / 'retrained_250.tflite',
}


def accuracy(start, end, guess):
    travelled = np.linalg.norm(end - start, axis=-1)
    missed = np.linalg.norm(guess - end, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(travelled == 0, 0.0,
                        np.maximum((1.0 - missed / travelled) * 100.0, 0.0))


def main():
    payload = json.loads((CORPUS / 'targets.json').read_text(encoding='utf-8'))
    rows = payload['targets']
    start = np.array([r['start'] for r in rows], dtype=np.float64)
    end = np.array([r['end'] for r in rows], dtype=np.float64)
    images = sorted((CORPUS / 'images').glob('*.png'))
    print(f'{len(images)} trusses in v2 geometry (offset 100,150)\n')

    scores = {}
    for name, path in MODELS.items():
        interpreter = Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        i_in = interpreter.get_input_details()[0]['index']
        i_out = interpreter.get_output_details()[0]['index']
        guess = np.empty((len(images), 2))
        for k, q in enumerate(images):
            frame = np.asarray(Image.open(q).convert('RGB'), dtype=np.float32)
            interpreter.set_tensor(i_in, frame[np.newaxis, ...])
            interpreter.invoke()
            guess[k] = interpreter.get_tensor(i_out)[0][:2]
        scores[name] = accuracy(start, end, guess)
        moved = np.linalg.norm(guess - start, axis=-1)
        print(f'  {name:<28} mean {scores[name].mean():5.2f}%   '
              f'median {np.median(scores[name]):5.2f}%   '
              f'moves {moved.mean():5.1f}px')

    travelled = np.linalg.norm(end - start, axis=-1)
    print(f'  {"(the truth)":<28} {"":>23}   moves {travelled.mean():5.1f}px')

    a, b = list(scores.values())
    difference = b - a
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(difference, difference.size, replace=True).mean()
                     for _ in range(10000)])
    low, high = np.percentile(boot, [2.5, 97.5])
    verdict = 'real' if low > 0 or high < 0 else 'still not resolvable'
    print(f'\n  retrained_250 advantage: {difference.mean():+.2f} points   '
          f'95% CI [{low:+.2f}, {high:+.2f}]   -> {verdict}')
    print(f'  better on {(difference > 0).mean() * 100:.0f}% of rounds')


if __name__ == '__main__':
    main()
