"""Score the recovered 2023 SavedModels against the one that actually shipped.

    training/tfvenv/Scripts/python.exe training/eval_savedmodels.py

Three Keras SavedModels were recovered from the Recycle Bin -- retrained_200,
retrained_250, and truss_game_AI_model. Only one was ever converted to TFLite
and put in the game, and the suspicion is that it was not the best of them.

This loads each through `tf.saved_model.load` rather than `keras.models.
load_model`: the signature is stable across TensorFlow versions, while Keras
changed how it loads SavedModel directories, and these files are from 2023.

Input frames come from corpus768/, which was rendered *before* the markers were
enlarged, so the models see the marker size they were trained on. That matters
-- v3's detector lost seventy points to that change alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).resolve().parent
FRAMES = HERE / 'corpus768' / 'val' / 'images'
TARGETS = HERE / 'graph_corpus' / 'val' / 'graphs.json'


def accuracy(start, end, guess):
    """The game's own metric, in the game's own terms."""
    travelled = np.linalg.norm(end - start, axis=-1)
    missed = np.linalg.norm(guess - end, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        score = (1.0 - missed / travelled) * 100.0
    return np.where(travelled == 0, 0.0, np.maximum(score, 0.0))


def load_targets(count):
    samples = json.loads(TARGETS.read_text(encoding='utf-8'))['samples'][:count]
    start, end = [], []
    for s in samples:
        n = s['loadedNode']
        p = np.array(s['nodes'][n], dtype=np.float64)
        start.append(p)
        end.append(p + np.array(s['displacement'][n], dtype=np.float64))
    return np.array(start), np.array(end)


def describe(model):
    """What the signature actually wants, rather than what we assume."""
    fn = model.signatures['serving_default']
    inputs = {k: (v.shape.as_list(), v.dtype.name) for k, v in fn.structured_input_signature[1].items()}
    outputs = {k: (v.shape.as_list(), v.dtype.name) for k, v in fn.structured_outputs.items()}
    return fn, inputs, outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=400)
    p.add_argument('--models', nargs='*', default=None)
    args = p.parse_args()

    paths = sorted(FRAMES.glob('*.png'))[:args.count]
    if not paths:
        raise SystemExit(f'no frames under {FRAMES}')
    start, end = load_targets(len(paths))

    frames = np.stack([
        np.asarray(Image.open(q).convert('RGB'), dtype=np.float32) for q in paths
    ])
    print(f'{len(paths)} frames of {frames.shape[1]}x{frames.shape[2]}, '
          f'raw 0-255 as the game fed them\n')

    names = args.models or ['retrained_200', 'retrained_250', 'truss_game_AI_model']
    results = {}
    for name in names:
        directory = HERE / 'recovered' / name
        if not directory.is_dir():
            print(f'{name}: not found, skipping')
            continue
        try:
            model = tf.saved_model.load(str(directory))
            fn, inputs, outputs = describe(model)
        except Exception as exc:                       # noqa: BLE001 - report, do not crash
            print(f'{name}: failed to load -- {type(exc).__name__}: {exc}\n')
            continue

        print(f'--- {name} ---')
        print(f'  inputs  {inputs}')
        print(f'  outputs {outputs}')

        in_key = list(inputs)[0]
        out_key = list(outputs)[0]
        guesses = []
        for i in range(0, len(frames), 16):
            batch = tf.constant(frames[i:i + 16])
            guesses.append(fn(**{in_key: batch})[out_key].numpy())
        guess = np.concatenate(guesses)[:, :2].astype(np.float64)

        scores = accuracy(start, end, guess)
        results[name] = scores
        moved = np.linalg.norm(guess - start, axis=-1)
        travelled = np.linalg.norm(end - start, axis=-1)
        print(f'  mean {scores.mean():.2f}%   median {np.median(scores):.2f}%   '
              f'zeros {(scores == 0).mean() * 100:.1f}%')
        print(f'  moves the node {moved.mean():.1f}px where the truth is '
              f'{travelled.mean():.1f}px\n')

    # The shipped model, on the very same frames. Its published 60.5% was
    # measured through the browser after the markers were enlarged, so quoting
    # it beside these would compare two different input distributions -- the
    # exact mistake that cost v3 seventy points earlier.
    shipped = HERE.parent / 'truss_game_AI_model.tflite'
    if shipped.exists():
        interpreter = tf.lite.Interpreter(model_path=str(shipped))
        interpreter.allocate_tensors()
        in_index = interpreter.get_input_details()[0]['index']
        out_index = interpreter.get_output_details()[0]['index']
        guesses = []
        for frame in frames:
            interpreter.set_tensor(in_index, frame[np.newaxis, ...])
            interpreter.invoke()
            guesses.append(interpreter.get_tensor(out_index)[0])
        guess = np.array(guesses)[:, :2].astype(np.float64)
        results['the .tflite that shipped'] = accuracy(start, end, guess)

    if results:
        print(f'summary, same {len(paths)} trusses, same frames:')
        print(f'  {"straight down by the average travel":<38} 59.5%   (baseline)')
        for name, scores in sorted(results.items(), key=lambda kv: -kv[1].mean()):
            print(f'  {name:<38} {scores.mean():.1f}%')


if __name__ == '__main__':
    main()
