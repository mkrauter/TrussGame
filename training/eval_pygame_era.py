"""Score the 2023 models the way 2023 actually drew them.

    venv/Scripts/python.exe training/eval_pygame_era.py

Every figure so far has been measured on Canvas renders, which differ from what
these models were trained on in three ways at once: 2.5px members instead of
pygame's 1px aalines, v3's truss placement instead of v2's, and -- until
recently -- a different marker size.

This renders with pygame 2.1.2, which requirements.txt pins for precisely this
reason, using truss_game_v2.py's own Truss class and its own drawing code, and
crops exactly as its __predict does. It is the closest thing available to the
number these models really earned.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import numpy as np
import pygame
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from truss_game_v2 import Truss                                   # noqa: E402

FORCE = 100000
MODELS = {
    'the 2023 model in the repo': ROOT / 'truss_game_v2_model.tflite',
    'retrained_250 (recovered)': ROOT / 'training' / 'recovered' / 'retrained_250.tflite',
    'retrained_200 (rebuilt)': ROOT / 'training' / 'recovered' / 'retrained_200.tflite',
}


def draw_undeformed(screen, truss):
    """Exactly what truss_game_v2.py has on screen when it asks for a prediction.

    __draw_truss runs before __predict, and calculate() has not been called yet,
    so every node is at rest and every stress is zero -- which stress_color
    renders as white.
    """
    screen.fill(pygame.Color('grey25'))
    for p in truss.nodes[truss.supports]:
        pygame.draw.polygon(screen, (64, 128, 64),
                            [p, (p[0] - 10, p[1] + 20), (p[0] + 10, p[1] + 20)])
    for p in truss.nodes[truss.loaded_node]:
        pygame.draw.polygon(screen, (100, 100, 200),
                            [p, (p[0] - 10, p[1] - 20), (p[0] + 10, p[1] - 20)])
    for e in truss.elements:
        pygame.draw.aaline(screen, (255, 255, 255), truss.nodes[e[0]], truss.nodes[e[1]])


def accuracy(start, end, guess):
    travelled = np.linalg.norm(end - start, axis=-1)
    missed = np.linalg.norm(guess - end, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(travelled == 0, 0.0,
                        np.maximum((1.0 - missed / travelled) * 100.0, 0.0))


def main(count=1200):
    pygame.init()
    screen = pygame.display.set_mode((900, 900))

    np.random.seed(20230916)          # the Truss class uses numpy's global RNG
    frames, start, end = [], [], []
    for _ in range(count):
        truss = Truss()
        draw_undeformed(screen, truss)
        # surfarray is (width, height, 3); the model wants (height, width, 3).
        crop = pygame.surfarray.array3d(screen.subsurface(68, 68, 768, 768)).swapaxes(0, 1)
        frames.append(crop.astype(np.float32))

        node = truss.loaded_node[0]
        truss.calculate(FORCE)
        start.append(truss.nodes[node].copy())
        end.append(truss.nodes_moved[node].copy())
    pygame.quit()

    start, end = np.array(start), np.array(end)
    print(f'{count} trusses, rendered by pygame {pygame.version.ver} '
          f'as truss_game_v2.py draws them\n')

    scores = {}
    for name, path in MODELS.items():
        interpreter = Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        i_in = interpreter.get_input_details()[0]['index']
        i_out = interpreter.get_output_details()[0]['index']
        guess = np.empty((count, 2))
        for k, frame in enumerate(frames):
            interpreter.set_tensor(i_in, frame[np.newaxis, ...])
            interpreter.invoke()
            guess[k] = interpreter.get_tensor(i_out)[0][:2]
        scores[name] = accuracy(start, end, guess)
        print(f'  {name:<28} mean {scores[name].mean():5.2f}%   '
              f'median {np.median(scores[name]):5.2f}%   '
              f'moves {np.linalg.norm(guess - start, axis=-1).mean():5.1f}px')
    print(f'  {"(the truth)":<28} {"":>23}   '
          f'moves {np.linalg.norm(end - start, axis=-1).mean():5.1f}px')

    rng = np.random.default_rng(0)
    reference = list(scores)[0]
    for name in list(scores)[1:]:
        difference = scores[name] - scores[reference]
        boot = np.array([rng.choice(difference, difference.size, replace=True).mean()
                         for _ in range(10000)])
        low, high = np.percentile(boot, [2.5, 97.5])
        print(f'\n  {name} over {reference}: {difference.mean():+.2f} points   '
              f'95% CI [{low:+.2f}, {high:+.2f}]')
    print('\n  baseline for reference: 59.5%')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1200)
