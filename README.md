# Truss game

A random truss is loaded at one node. Guess where that node will end up once it
settles — an AI opponent guesses too, from a screenshot alone.

![Gameplay screenshot](images/screenshot.jpg "Gameplay screenshot")

The AI never sees node coordinates or the topology, only the rendered image.
That constraint is the point of the project: it is what makes the comparison
against a human fair.

Read more about the idea in
['Is a fruit fly a smarter engineer than you?'](https://marton-krauter.medium.com/is-a-fruit-fly-a-smarter-engineer-than-you-850db1031fe8)

## Three versions

| version | what it is | AI score |
|---|---|---|
| **original** | the 2019 game, human only | — |
| **v2** | the first AI opponent, a 2023 convolutional net | 59.8% |
| **v3** | the current AI: read the screen, recover the structure, solve it | **96.4%** |

All three are playable in the browser, no download:

**[Play all three](https://mkrauter.github.io/TrussGame/)** — or go straight to
[v3](https://mkrauter.github.io/TrussGame/web/v3/),
[v2](https://mkrauter.github.io/TrussGame/web/v2/), or
[the original](https://mkrauter.github.io/TrussGame/web/original/).

**[Download v2 for Windows](https://github.com/mkrauter/TrussGame/releases/latest)**
— the binary is unsigned, so SmartScreen will warn you; click
*More info → Run anyway*.

## What is here

The Python versions are the originals. The JavaScript under `web/` exists so the
games can be played without installing anything; it is a port, not the article.

| path | what it is |
|---|---|
| `truss_game_original.py` | **original** — the human-only game, frozen as a historic reference |
| `truss_game_AI.py` | **v2** — human against the 2023 model |
| `truss_game_AI_model.tflite` | the v2 model |
| `truss_game_AI_training.ipynb` | the v2 training notebook — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrauter/TrussGame/blob/master/truss_game_AI_training.ipynb) |
| `training/` | **v3** — corpus generation, models, training and the verification harnesses |
| `web/` | browser ports of all three, Canvas 2D, no dependencies |

## Running from source

    pip install -r requirements.txt
    python truss_game_original.py      # original
    python truss_game_AI.py            # v2

For the browser ports, serve the repository over HTTP — ES modules will not load
from `file://`:

    python -m http.server 8000

then open <http://127.0.0.1:8000/>.

## How the physics works

Ten nodes are sampled in a 700×500 box, and the connectivity comes from a
Delaunay triangulation. The two extreme-x nodes are fully pinned. One random
node is loaded vertically downward.

The solve is linear: the stiffness matrix is assembled from the undeformed
geometry and never updated, so `u = K⁻¹f` exactly. Displacements come out around
half the structure's length scale, which is why the deformation looks dramatic
for what is small-displacement theory. The animation ramps the load with a
decaying oscillation, but the target is always the settled state.

## How good are the AIs, really

Scoring: 100% means you clicked exactly where the node settled. 0% means you
missed by at least as far as the node travelled. The score is the fraction of
its movement you predicted, not a distance in pixels.

| predictor | mean | median |
|---|---|---|
| guess the starting point — never move | 0% | 0% |
| guess straight down by the average travel | 59.5% | 66.5% |
| **v2**, the 2023 model | 59.8% | 66.6% |
| **v3**, the current model | **96.4%** | **97.4%** |
| a perfect solver, reading the screen as well as v3 does | ~98% | — |

**v2 does not beat a one-line heuristic.** Decomposing it explains why: it locates
the loaded node almost perfectly from pixels alone — 0.97 correlation with the
true position — but its displacement prediction scores a negative R² against
simply always guessing the mean. It moves the node 137px where the truth
averages 152px. It learned to find the blue node and drop it by roughly the
average amount, which is most of the game's score and none of the mechanics.
The figure quoted in the v2 notebook was measured on training data, and the
notebook predates the deployed model and does not reproduce it. The browser port
draws members at v3's thickness rather than pygame's hairlines so the three
versions look alike; that costs v2 1.8 points, and it scores 61.6% if drawn
faithfully.

**v3 does the mechanics.** It takes the same screenshot, but instead of
regressing an answer straight from pixels it works in two stages: a small
keypoint network finds the nodes and their roles, connectivity is read by
checking which node pairs have a line drawn between them, and then a graph
network solves the recovered structure by passing messages along its members —
which is what an iterative solver does. Convolutions are the right tool for
finding marks in an image and the wrong tool for a global implicit solve; an
end-to-end network on the same screenshots stalls at 77%. Splitting the two is
the whole difference.

Its remaining error is almost entirely perception, not physics: a flawless
solver reading node positions as precisely as v3 does would score about 98%.

## Notes for anyone hacking on this

`pygame` and `scipy` are pinned deliberately, and not for packaging reasons.
There are no plans to retrain the v2 model, so whatever renders its input is
part of its contract — newer pygame ships a newer SDL that rasterises polygons
differently, which measurably moves the model's predictions. `requirements.txt`
explains each pin. Build tooling lives in `requirements-build.txt`.

The v3 renderer is frozen for the same reason: its detector was trained on the
Canvas output of `web/src/renderer.js`, so changing how members or markers are
drawn invalidates it.

Both browser models are checked against their Python originals rather than
assumed to match — `training/verify_*.py` compares each JavaScript runtime
against PyTorch or LiteRT on real frames, down to the capture path.

Windows binaries are built by GitHub Actions on a clean runner whenever a `v*`
tag is pushed, and attached to the release automatically.
