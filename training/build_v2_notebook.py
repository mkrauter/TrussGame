# -*- coding: utf-8 -*-
"""Merge the repo notebook's narrative cells onto TrussTraining_v2's real run.

Code cells and their outputs are copied verbatim -- never edited, never
reordered, never re-run. The single exception is the mosaic figure, which is
re-composited onto its intended background (see flatten_transparent_png).

The prose has been copy-edited for grammar and flow. The voice is deliberately
left alone: first person, conversational, and still carrying its own asides.
"""
import base64
import io
import json, copy
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'Colab Notebooks' / 'TrussTraining_v2.ipynb'
OUT = ROOT / 'truss_game_v2_training.ipynb'

# The game's background, and the colour matplotlib drew every panel interior
# with. Sampling the recorded figure, (64, 64, 64) is by far its commonest
# opaque pixel.
BACKDROP = (64, 64, 64)

nb = json.load(SRC.open(encoding='utf-8'))
cells = nb['cells']


def flatten_transparent_png(cell):
    """Composite a recorded RGBA figure onto BACKDROP, in place.

    The mosaic was saved with a fully transparent background (corner pixel
    (255, 255, 255, 0)) and its score captions drawn in white on top of that
    nothing. A dark-themed viewer -- PyCharm, or Colab in dark mode -- shows
    them; Colab's default light theme composites the same PNG onto white and
    the captions vanish.

    This re-composites the recorded pixels onto the grey the figure was drawn
    against. Every opaque pixel keeps its exact value; only alpha < 255 regions
    change, and they change to what a dark viewer already displays there. No
    cell is re-run and no pixel is invented.
    """
    for out in cell.get('outputs', []):
        data = out.get('data', {})
        if 'image/png' not in data:
            continue
        raw = base64.b64decode(data['image/png'])
        im = Image.open(io.BytesIO(raw))
        if im.mode != 'RGBA':
            continue
        flat = Image.new('RGB', im.size, BACKDROP)
        flat.paste(im, mask=im.split()[-1])
        buf = io.BytesIO()
        flat.save(buf, format='PNG', optimize=True)
        data['image/png'] = base64.b64encode(buf.getvalue()).decode('ascii')
    return cell


def code(i):
    c = copy.deepcopy(cells[i])
    assert c['cell_type'] == 'code', i
    # The TensorBoard cell's 23.8 MB `resources` blob is KEPT deliberately. It
    # is not cruft: it is a recorded HTTP replay of the live TensorBoard --
    # index.js, the page, fonts, and the scalar series themselves -- and Colab
    # replays it into a real, interactive training-progress chart.
    #
    # It is also this notebook's own run, not a leftover from another session.
    # The embedded validation series ends at epoch_loss 2954.2939 and
    # epoch_mae 42.4744, matching cell 5's final "val_loss: 2954.2939 -
    # val_mae: 42.4744" exactly, over 100 points and 927 s of wall clock.
    #
    # The cost is that the file is 24 MB, so GitHub will not render it in the
    # notebook viewer -- but the chart is only viewable in Colab anyway, which
    # is where the badge at the top sends the reader.
    return c


def md(text):
    lines = text.strip('\n').split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}


BADGE = ('<a href="https://colab.research.google.com/github/mkrauter/TrussGame/blob/master/'
         'truss_game_v2_training.ipynb" target="_parent">'
         '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')

BANNER = """
> **This notebook is historic. It is kept as a record of how the convolutional
> model was trained and what its structure was, not because its numbers or its
> method should be followed.**
>
> Four things to know before reading it:
>
> * The accuracy reported at the end was measured over **all** the samples,
>   including the 1600 the model trained on. Scored honestly on unseen trusses,
>   the model that shipped reaches **61.7%**, against **59.5%** for a single
>   line of arithmetic -- guess straight down by the average travel.
> * The grid of example predictions is drawn from **training** samples too
>   (indices 1500-1519; validation is 0-399), so it shows how well the model
>   recalls what it was fitted on, not how well it generalises.
> * The validation curve in the TensorBoard chart is the one honest measurement
>   in the notebook, and it is the more interesting one: it stops improving
>   around epoch 40.
> * The narrative cells come from this project's other Colab notebook and have
>   been adapted to describe the code here. **The code cells and their outputs
>   are one real, complete training run**, and nothing below was re-run. The
>   single alteration is to the example grid: it was saved with a transparent
>   background, which left its white captions invisible on a light page, so it
>   has been composited onto the grey it was drawn against. Every opaque pixel
>   is untouched.
>
> For the current model -- and for the reason the architecture here cannot do
> better -- see
> [`truss_game_v3_training.ipynb`](https://colab.research.google.com/github/mkrauter/TrussGame/blob/master/truss_game_v3_training.ipynb),
> which clones the repository and imports the real package instead of pasting a
> copy of it.
"""

INTRO = """
# Neural network training for the Truss game

This is the Google Colab notebook that trains the convolutional neural network behind the Truss game.

Read the story of the project in ['Can a machine learn engineering intuition?'](https://mkrauter.github.io/TrussGame/web/article/)


## Setting up the environment

Let's import the modules we need and check that the virtual machine is in good shape.
"""

GPU_NOTE = """
You should see `Num GPUs Available: 1`, which tells us the virtual machine is set up correctly and the GPU is there to be used.
"""

DATASET = """
# Preparing the training dataset

Before we can train a neural network we need a good number of samples -- input-output pairs -- for it to learn from.

There are two tasks to get through:

1.   Create the sample data and make it available to our virtual machine
2.   Convert that data into a format the training process can consume

## Data feed methods

Creating the images from the game itself, on your own computer, is the easy half. Pygame can capture any region of the window at any moment, so a simple loop does the job: build a random truss, save the screen as an image, solve the truss for its displacement, and write the resulting coordinates to a text file. That generator is not repeated in this notebook, and it did not survive into the repository either -- what we do here is consume the archive it produced at the time.

Getting those images to the virtual machine that does the training is where it turns awkward. Uploading 2000 separate files to a Google Drive folder takes a while, and reading them back one by one from the mounted Drive is slow all over again.

We can do considerably better by not saving images at all, and building a Numpy array instead. Numpy can write an entire data structure to a single file -- think of how Pickle works -- and it compresses it on the way out, so there is only ever one file to move. That is what `test2/images.npz` below is: one compressed archive holding all 2000 frames, with the matching coordinates in a plain text file beside it.

This comes at a price. A Numpy array has to fit entirely in memory -- here, the memory of the virtual machine -- which puts a hard ceiling on how large the training set can be. Go much past 2000 samples and you will run out.

There are two ways around that ceiling, neither of them taken here:

1.  Our game, and therefore our training data, is purely synthetic. We can generate any number of samples in a matter of seconds, so producing them _in situ_ on the Colab machine is the obvious move -- there would be nothing to transfer at all.
2.  Tensorflow, the framework running behind Colab, has its own optimised structure for training data, the `tfrecord`. It is not the easiest thing to get your head around at first, but it sidesteps the memory limit entirely, and it comes with tools you will be glad of later on.

We stay with the fixed Numpy set for simplicity. Keeping it fixed does buy one thing worth naming: every run reads exactly the same 2000 trusses, so two models can be compared without the dataset shifting underneath them.

Enough talking, let's roll up our sleeves...
"""

LOAD = """
Read the archive back from the mounted Drive and separate out the different kinds of data.

The frames are stored at their captured size of 768x768 and resized to 256x256 here, before they ever reach the network -- so the model below begins at its first convolution rather than with a Resizing layer in front of it. The coordinates are shifted by 68 pixels, which is the origin of the crop taken out of the 900x900 game window.
"""

SPLIT = """
We hold back 20% of the samples for validation. Validation is what reveals whether the model has begun to *overfit* -- to learn the training data itself rather than anything general about trusses, and so to do markedly worse on examples it has never seen.

Note the commented-out first line. Scaling the pixel values into a small range around zero is usually worth doing, and it is left here as a reminder, but this run trains on the raw 0-255 values as they come out of the game. Note also which end the validation samples come from: the split takes the **first** 400, so indices 0-399 are held back and 400-1999 are trained on. That boundary matters again further down.
"""

BUILD = """
# Build the convolutional neural network

Let's put the model together:
"""

TRAIN = """
# Train the model

Everything is in place, so all that is left is to run the training. We keep it going for 100 epochs -- think of an epoch as one repetition of the learning cycle. Running it longer than it needs will not improve the model; it will only deepen the overfit.

This run shows exactly that, which is why the log is worth reading rather than just its last line. The validation error bottoms out around **epoch 40 at roughly 36 px**, then climbs back to **42 px by epoch 100**, while the training error keeps falling to **21 px**. The model is still learning after epoch 40 -- it is simply learning these 1600 trusses rather than trusses in general. The `# Stable under 40` comment in the cell above is that same observation, noted at the time.

Training takes about 15 minutes on a GPU runtime.
"""

TB = """
## Evaluate the training process

Tensorflow comes with a tool called `Tensorboard`, which turns the saved log files into a picture of how the training went:

The chart below is live, and it is the clearest view of everything described above: choose `epoch_mae` and put the `train` and `validation` series side by side. They fall together for the first forty epochs or so, then separate and never meet again -- training error carrying on down while validation flattens and drifts back up. That gap is the overfit, and watching it open is far more convincing than any single number at the end of the run.
"""

PREDICT = """
Let's see whether the model can predict anything at all:
"""

GRID = """
There are signs of life, but a bare list of coordinates is hard to judge. Let's build something easier to read.

One thing to keep in mind while looking at the grid below. The split above put the **first 400** samples into validation and kept indices **400 to 1999** for training, and this call starts at 1500 -- so every truss in the mosaic is one the model was fitted on. The crosses sit close because the network is reproducing answers it was trained on. This is a picture of recall, not of generalisation. `show_image(0)` would have drawn twenty validation samples instead -- a single digit's difference, and worth knowing before reading too much into how tight the fit looks.

The honest counterpart to this picture is the validation curve in the chart above.
"""

ACC = """
Let's look at the overall accuracy of our model.

Note what this number is measured against: it predicts over **every** sample, the 1600 training ones included, so it flatters the model. The validation error in the training log above is the honest measure.
"""

SAVE = """
# Save the trained model

Finally, save the model and convert it to the Tensorflow Lite format. The conversion cell was left un-run in this session -- what reached Drive is the SavedModel written by the cell above it.
"""

CLOSING = """
To use the model in the game, all you need to do is download the tflite file and overwrite the existing one in your local copy of the repo.

Have fun experimenting!
"""

merged = [
    md(BADGE), md(BANNER), md(INTRO),
    code(1),                            # imports + GPU check
    md(GPU_NOTE), md(DATASET), md(LOAD),
    code(2),                            # load npz from Drive, resize to 256
    md(SPLIT),
    code(3),                            # 20% validation split
    md(BUILD),
    code(4),                            # model + summary
    md(TRAIN),
    code(5),                            # fit, 100 epochs
    md(TB),
    code(6),                            # tensorboard
    md(PREDICT),
    code(7),                            # predict
    md(GRID),
    flatten_transparent_png(code(8)),   # the example grid
    md(ACC),
    code(9),                            # overall accuracy 67%
    md(SAVE),
    code(11),                           # model.save
    code(12),                           # tflite conversion (no output)
    md(CLOSING),
]

nb['cells'] = merged
nb.setdefault('metadata', {}).setdefault('colab', {})['name'] = 'truss_game_v2_training.ipynb'
with OUT.open('w', encoding='utf-8') as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print(f'wrote {OUT.relative_to(ROOT)} with {len(merged)} cells')
