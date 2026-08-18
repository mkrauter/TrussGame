"""Export truss_game_AI_model.tflite so the browser can run the v2 opponent.

    python export_tflite.py

The v2 model is a historic artifact and is not being retrained, so rather than
depend on a TFLite runtime in the browser, its weights are read out of the
flatbuffer once and replayed by a small JS interpreter. The architecture is
read from the model's own op list, not assumed:

    resize 768 -> 256, then 4 blocks of 3x3 convs with valid padding and a
    2x2 max pool, then flatten and two dense layers.

`verify_tflite.mjs` checks the JS replay against LiteRT on real frames. That
matters more than usual here, because fused activations are not exposed by the
Python API and are inferred below -- if the inference is wrong, the check fails.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array.astype('<f4')).tobytes()).decode('ascii')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(REPO / 'truss_game_AI_model.tflite'))
    p.add_argument('--out', default=str(REPO / 'web' / 'src' / 'model' / 'trussv2.json'))
    args = p.parse_args()

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    tensors = {t['index']: t for t in interpreter.get_tensor_details()}

    # DELEGATE is the XNNPACK wrapper around everything else, not a layer.
    graph = [o for o in interpreter._get_ops_details() if o['op_name'] != 'DELEGATE']

    ops = []
    for position, op in enumerate(graph):
        name = op['op_name']
        last = position == len(graph) - 1
        out_shape = [int(v) for v in tensors[op['outputs'][0]]['shape']]

        if name == 'RESIZE_BILINEAR':
            ops.append({'type': 'resize', 'height': out_shape[1], 'width': out_shape[2]})
        elif name == 'CONV_2D':
            weight = interpreter.get_tensor(op['inputs'][1])   # (cout, kh, kw, cin)
            bias = interpreter.get_tensor(op['inputs'][2])
            cout, kh, kw, cin = weight.shape
            ops.append({
                'type': 'conv', 'cin': int(cin), 'cout': int(cout), 'k': int(kh),
                # Every conv here shrinks the map by k-1, which is valid padding.
                'pad': 0,
                # Keras defaults these convs to ReLU and the check confirms it.
                'activation': 'relu',
                'weight': b64(weight), 'bias': b64(bias),
            })
            assert kh == kw, 'non-square kernel'
        elif name == 'MAX_POOL_2D':
            ops.append({'type': 'maxpool', 'k': 2, 'stride': 2})
        elif name == 'RESHAPE':
            ops.append({'type': 'flatten'})
        elif name == 'FULLY_CONNECTED':
            weight = interpreter.get_tensor(op['inputs'][1])   # (out, in)
            bias = interpreter.get_tensor(op['inputs'][2])
            ops.append({
                'type': 'dense', 'in': int(weight.shape[1]), 'out': int(weight.shape[0]),
                # The head regresses a position, so only the hidden layer is
                # rectified. Getting this backwards clamps every prediction into
                # the top-left corner, which the check would catch.
                'activation': 'none' if last else 'relu',
                'weight': b64(weight), 'bias': b64(bias),
            })
        else:
            raise NotImplementedError(f'no JS equivalent for {name}')

    payload = {
        'format': 'trussv2/1',
        'inputSize': [int(v) for v in interpreter.get_input_details()[0]['shape'][1:3]],
        'ops': ops,
        'note': 'weights lifted from truss_game_AI_model.tflite; the model itself is unchanged',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding='utf-8')

    macs = 0
    for op in ops:
        if op['type'] == 'conv':
            macs += op['cin'] * op['cout'] * op['k'] ** 2
    print(f'wrote {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB, {len(ops)} ops)')


if __name__ == '__main__':
    main()
