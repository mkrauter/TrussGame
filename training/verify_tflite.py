"""Check the JS replay of the v2 model against LiteRT.

    python verify_tflite.py

Runs both on the same frames -- rendered by the browser, saved as PNG -- and
compares the predicted click. The architecture was read out of the flatbuffer,
but fused activations are not exposed by the Python API and were inferred, so
this is what confirms the inference was right. A wrong activation on the head,
for instance, clamps every prediction into the top-left corner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=str(REPO / 'truss_game_AI_model.tflite'))
    p.add_argument('--count', type=int, default=8)
    args = p.parse_args()

    dump = json.loads(subprocess.run(
        ['node', str(HERE / 'dump_v2.mjs'), '--count', str(args.count)],
        capture_output=True, text=True, check=True, cwd=HERE
    ).stdout)

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    in_index = interpreter.get_input_details()[0]['index']
    out_index = interpreter.get_output_details()[0]['index']

    worst = 0.0
    for sample in dump:
        # Both sides read the same file on disk; nothing is passed between them
        # except the prediction.
        path = HERE / 'corpus768' / 'val' / 'images' / f"{sample['image']}.png"
        with Image.open(path) as img:
            frame = np.asarray(img.convert('RGB'), dtype=np.float32)
        interpreter.set_tensor(in_index, frame[np.newaxis, ...])
        interpreter.invoke()
        theirs = interpreter.get_tensor(out_index)[0]
        worst = max(worst, float(np.abs(theirs - np.array(sample['prediction'])).max()))
        print(f"  seed {sample['seed']}: litert {theirs.round(2)}  js "
              f"{np.array(sample['prediction']).round(2)}")

    ok = worst <= 0.05
    print(f'\n  {"PASS" if ok else "FAIL"}  max |diff| = {worst:.4f} px')
    print('  (tolerance is 0.05px; the two run the same weights in different orders)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
