A truss is loaded at one node and deforms. Guess where that node ends up — the
AI opponent guesses too, from a screenshot alone.

## Windows

Download `truss_game_AI.exe` below and run it. Nothing to install; the model is
bundled inside.

**Windows will warn you.** The binary is unsigned, so SmartScreen shows
"Windows protected your PC". Click **More info → Run anyway**. Code signing
certificates cost money that a hobby project does not justify — the build is
public and reproducible from source instead, which is the honest alternative to
a signature you would have to trust anyway.

First launch takes a few seconds: a one-file PyInstaller build unpacks itself to
a temporary directory before the window appears.

## Play in a browser instead

The human-only version runs with no download at all — see `v2/` in the
repository.

## Running from source

    pip install -r requirements.txt
    python truss_game_AI.py

The pinned versions in `requirements.txt` are deliberate and documented there.
`pygame` and `scipy` in particular are frozen because the bundled model was
trained on the pixels those exact versions produce, and there is no plan to
retrain it.
