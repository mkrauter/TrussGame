"""Convert a recovered SavedModel to TFLite, and check the conversion held.

    training/tfvenv/Scripts/python.exe training/convert_savedmodel.py retrained_250

The original project never produced a TFLite for this model, on the belief that
the tooling had stopped working on Windows. It has not: TensorFlow dropped
native *GPU* support on Windows after 2.10, but the CPU wheel still installs and
`TFLiteConverter` is part of it. This runs on tensorflow 2.15.1, Python 3.11.

Converting is only half the job -- a conversion that silently changes the answer
is worse than none, so the result is scored against the SavedModel it came from
on real frames before it is offered to anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).resolve().parent
FRAMES = HERE / 'corpus768' / 'val' / 'images'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', help='directory name under training/recovered/')
    p.add_argument('--out', default=None)
    p.add_argument('--check', type=int, default=40, help='frames to verify on')
    args = p.parse_args()

    source = HERE / 'recovered' / args.model
    out = Path(args.out) if args.out else HERE / 'recovered' / f'{args.model}.tflite'

    converter = tf.lite.TFLiteConverter.from_saved_model(str(source))
    blob = converter.convert()
    out.write_bytes(blob)
    print(f'wrote {out}  ({len(blob) / 1024 / 1024:.2f} MB)')

    # --- does it still compute the same thing? -----------------------------
    paths = sorted(FRAMES.glob('*.png'))[:args.check]
    frames = np.stack([
        np.asarray(Image.open(q).convert('RGB'), dtype=np.float32) for q in paths
    ])

    fn = tf.saved_model.load(str(source)).signatures['serving_default']
    in_key = list(fn.structured_input_signature[1])[0]
    out_key = list(fn.structured_outputs)[0]
    reference = np.concatenate([
        fn(**{in_key: tf.constant(frames[i:i + 8])})[out_key].numpy()
        for i in range(0, len(frames), 8)
    ])

    interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    in_index = interpreter.get_input_details()[0]['index']
    out_index = interpreter.get_output_details()[0]['index']
    converted = []
    for frame in frames:
        interpreter.set_tensor(in_index, frame[np.newaxis, ...])
        interpreter.invoke()
        converted.append(interpreter.get_tensor(out_index)[0])
    converted = np.array(converted)

    worst = float(np.abs(reference - converted).max())
    print(f'\n{len(paths)} frames, SavedModel vs TFLite: max |diff| = {worst:.4f} px')
    print('PASS' if worst < 0.5 else 'FAIL — the conversion changed the answer')


if __name__ == '__main__':
    main()
