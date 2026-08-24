"""Is retrained_250 actually better *in the game as it now renders*?

    venv/Scripts/python.exe training/compare_v2_models.py

Two questions that are easy to conflate:

  * On frames drawn the way they were in 2023, retrained_250 scores 64.0% and
    the other model 60.7%.
  * The game now draws its markers 50% larger, and the two models do not lose
    the same amount to that change.

So both are scored here on the *current* rendering, on the same trusses, and the
difference is tested pairwise rather than by eyeballing two means. At this
sample size the standard error on either mean is around a point, which is the
size of the effect being claimed -- comparing the means alone would not settle
it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

HERE = Path(__file__).resolve().parent
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


def run(path, frames):
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    i_in = interpreter.get_input_details()[0]['index']
    i_out = interpreter.get_output_details()[0]['index']
    out = []
    for frame in frames:
        interpreter.set_tensor(i_in, frame[np.newaxis, ...])
        interpreter.invoke()
        out.append(interpreter.get_tensor(i_out)[0][:2])
    return np.array(out, dtype=np.float64)


def main():
    for label, folder in (('as the game renders now', 'corpus768now'),
                          ('as it rendered in 2023', 'corpus768')):
        images = sorted((HERE / folder / 'val' / 'images').glob('*.png'))
        if not images:
            print(f'{folder}: no frames, skipping\n')
            continue

        samples = json.loads((HERE / 'graph_corpus' / 'val' / 'graphs.json')
                             .read_text(encoding='utf-8'))['samples'][:len(images)]
        start = np.array([s['nodes'][s['loadedNode']] for s in samples], dtype=np.float64)
        end = start + np.array([s['displacement'][s['loadedNode']] for s in samples],
                               dtype=np.float64)
        frames = np.stack([np.asarray(Image.open(q).convert('RGB'), dtype=np.float32)
                           for q in images])

        print(f'--- {label} ({len(images)} trusses) ---')
        scores = {}
        for name, path in MODELS.items():
            scores[name] = accuracy(start, end, run(path, frames))
            print(f'  {name:<28} {scores[name].mean():5.2f}%   '
                  f'median {np.median(scores[name]):5.2f}%')

        a, b = list(scores.values())
        difference = b - a                       # paired: same truss, same frame
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(difference, difference.size, replace=True).mean()
                         for _ in range(10000)])
        low, high = np.percentile(boot, [2.5, 97.5])
        verdict = 'real' if low > 0 or high < 0 else 'not resolvable'
        print(f'  retrained_250 advantage: {difference.mean():+.2f} points  '
              f'95% CI [{low:+.2f}, {high:+.2f}]  -> {verdict}\n')


if __name__ == '__main__':
    main()
