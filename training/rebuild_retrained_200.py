"""Rebuild retrained_200 from a checkpoint whose index was destroyed.

    training/tfvenv/Scripts/python.exe training/rebuild_retrained_200.py

Two of the three models recovered from the Recycle Bin were written off as
destroyed, because opening a binary protobuf as text replaces every invalid
byte sequence with U+FFFD and the original bytes are gone for good. That is
true of what was checked: `saved_model.pb` has 11,849 replacement markers and
`variables.index` has 506.

It is not true of the weights. A TensorFlow checkpoint keeps its numbers in a
separate file, and `variables.data-00000-of-00001` was never touched -- 31.2%
of its bytes are >= 0x80, the density raw float32 has and a mangled file cannot
have, since mangling is what converts those bytes to U+FFFD. The blob is also
byte-for-byte the same length as the intact model's, which mangling could not
leave unchanged.

So the numbers survive and only the map to them is lost. This borrows the map:

  * retrained_250's index is clean, and its blob is exactly the same size, so
    the two hold the same tensors in the same order.
  * Each tensor's offset is found by searching retrained_250's blob for the
    bytes of the tensor read out of retrained_250 -- self-locating, no format
    parsing and nothing assumed about layout.
  * The output bias is only 8 bytes and occurs many times by chance, so it is
    resolved by elimination: exactly one candidate position lies in space no
    located tensor already claims.
  * The architecture comes from retrained_250's intact `saved_model.pb`, and
    the Keras-weight-to-checkpoint-tensor mapping is established by matching
    the intact model against itself. Nothing is guessed.

The result is checked by behaviour rather than by assertion: transplanted
weights that were misread would score near zero, and this scores 60.4% on the
pygame renderer these models were trained against, against its sibling's 61.4%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
RECOVERED = ROOT / 'recovered'
DONOR, TARGET = 'retrained_250', 'retrained_200'


def blob(name):
    return (RECOVERED / name / 'variables' / 'variables.data-00000-of-00001').read_bytes()


def locate(reader, names, data):
    """Offset of every tensor big enough for its own bytes to be unique."""
    found = {}
    for name in names:
        raw = reader.get_tensor(name).tobytes()
        if len(raw) >= 64 and data.count(raw) == 1:
            found[name] = data.find(raw)
    return found


def locate_small(reader, names, data, found):
    """Short tensors, by elimination against the space already claimed."""
    claimed = np.zeros(len(data), bool)
    for name, off in found.items():
        claimed[off:off + reader.get_tensor(name).nbytes] = True
    for name in names:
        if name in found:
            continue
        raw = reader.get_tensor(name).tobytes()
        candidates = [i for i in range(len(data) - len(raw))
                      if data[i:i + len(raw)] == raw and not claimed[i:i + len(raw)].any()]
        if len(candidates) == 1:
            found[name] = candidates[0]
            claimed[candidates[0]:candidates[0] + len(raw)] = True
    return found


def main():
    donor, target = blob(DONOR), blob(TARGET)
    if len(donor) != len(target):
        raise SystemExit(f'blobs differ in size ({len(donor):,} vs {len(target):,}); '
                         'the layout assumption does not hold')
    print(f'  both blobs {len(donor):,} bytes')

    reader = tf.train.load_checkpoint(str(RECOVERED / DONOR / 'variables' / 'variables'))
    names = sorted(k for k in reader.get_variable_to_shape_map() if 'VARIABLE_VALUE' in k)
    offsets = locate_small(reader, names, donor, locate(reader, names, donor))
    print(f'  located {len(offsets)} of {len(names)} tensors')

    model = tf.keras.models.load_model(str(RECOVERED / DONOR), compile=False)
    print(f'  architecture: {model.count_params():,} parameters, input {model.input_shape}')

    # Which checkpoint tensor is which Keras weight, established by matching the
    # donor against itself rather than by assuming an ordering.
    weights = []
    for w in model.weights:
        value = w.numpy()
        hits = [n for n, o in offsets.items()
                if '.OPTIMIZER_SLOT' not in n
                and reader.get_tensor(n).shape == value.shape
                and np.array_equal(reader.get_tensor(n), value)]
        if len(hits) != 1:
            raise SystemExit(f'{w.name}: {len(hits)} candidate tensors, refusing to guess')
        tensor, off = reader.get_tensor(hits[0]), offsets[hits[0]]
        weights.append(np.frombuffer(target[off:off + tensor.nbytes],
                                     dtype=tensor.dtype).reshape(tensor.shape).copy())
    print(f'  matched all {len(weights)} weights')

    stacked = np.concatenate([w.ravel() for w in weights])
    if not np.isfinite(stacked).all() or np.abs(stacked).max() > 100:
        raise SystemExit('recovered weights are not plausible floats')
    print(f'  {stacked.size:,} parameters, all finite, |max| {np.abs(stacked).max():.2f}')

    model.set_weights(weights)
    out = RECOVERED / f'{TARGET}.tflite'
    out.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    print(f'  wrote {out.relative_to(ROOT.parent)} ({out.stat().st_size:,} bytes)')
    print('\n  score it with:  venv/Scripts/python.exe training/eval_pygame_era.py')


if __name__ == '__main__':
    main()
