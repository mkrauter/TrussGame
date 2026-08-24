"""Retrain the 2022 model that scored 78%, from the notebook that recorded it.

    training/tfvenv/Scripts/python.exe training/retrain_v2.py

The best model this project ever produced was trained on a local machine on
2022-03-30 and saved to a relative path that no longer exists. Its notebook
survives, and records everything needed to reproduce it: the architecture, Adam
at 1e-3, 50 epochs, a 20% validation split taken from the *front* of the set,
and -- the part that appears to matter -- batches of 8.

Every Colab run used the fit() default of 32. At 1600 training images that is
50 gradient steps per epoch against 200, and the local run reached a validation
miss of 30.48 px where the Colab runs sat at 66.9 px. Same architecture, same
optimiser, same learning rate, same epoch count.

The original corpus (Truss_training_dataset.npz) is gone, but test2/images.npz
came back from Drive: 2000 frames of the same generator, same 768x768 crop,
same label format.

Two departures from the notebook, both exactly equivalent:

  * It trains on pre-resized 256x256 data and only wraps the Resizing layer
    around the result at export time. Resizing has no parameters and is the
    first layer, so this changes nothing about the model -- it just avoids
    pushing 3.5 GB of 768x768 frames through the input pipeline fifty times.
    The pre-resize uses antialias=False, which is what the Keras layer does.
  * The loss is written out rather than imported. mean_squared_euclidean_dist
    is mse doubled, and Adam is scale-invariant to first order, so this is
    cosmetic -- but it makes val_mean_euclidean_dist directly comparable to the
    30.48 px the notebook logged, which is the number worth chasing.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'TrussGame' / 'test2'
OUT = ROOT / 'training' / 'retrained_2026'
EPOCHS, BATCH, VAL_SPLIT = 50, 8, 0.2
TARGET_MISS = 30.48                      # what the 2022 notebook logged


def mean_squared_euclidean_dist(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))


def mean_euclidean_dist(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))


def load_resized(chunk=100):
    """768x768 frames -> 256x256, a chunk at a time so 3.5 GB never lands at once."""
    path = CORPUS / 'images.npz'
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as h:
        version = np.lib.format.read_magic(h)
        shape, _, dtype = np.lib.format._read_array_header(h, version)
        per = int(np.prod(shape[1:])) * dtype.itemsize
        out = np.empty((shape[0], 256, 256, 3), np.float32)
        for i in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - i)
            block = np.frombuffer(h.read(n * per), dtype=dtype).reshape((n,) + shape[1:])
            out[i:i + n] = tf.image.resize(block, [256, 256], antialias=False).numpy()
            print(f'\r  resizing {i + n}/{shape[0]}', end='')
    print()
    return out


def build():
    layers = []
    for _ in range(4):
        layers += [tf.keras.layers.Conv2D(32, 3, activation='relu') for _ in range(3)]
        layers.append(tf.keras.layers.MaxPooling2D())
    return tf.keras.Sequential([tf.keras.layers.Input((256, 256, 3))] + layers + [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')])


def main():
    images = load_resized()
    labels = np.genfromtxt(CORPUS / 'training_data.txt', delimiter=',')
    coords = labels[:, [3, 4]].astype(np.float32)          # where the joint ends up
    starts = labels[:, [1, 2]].astype(np.float32)

    # The notebook takes validation from the FRONT of the set, so the split is
    # reproduced rather than reinvented.
    cut = int(VAL_SPLIT * len(images))
    val_x, val_y = images[:cut], coords[:cut]
    train_x, train_y = images[cut:], coords[cut:]
    print(f'  {len(train_x)} train / {len(val_x)} validation')
    print(f'  mean travel {np.linalg.norm(coords - starts, axis=-1).mean():.1f} px\n')

    model = build()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=mean_squared_euclidean_dist, metrics=[mean_euclidean_dist])
    print(f'  {model.count_params():,} parameters (2022 notebook: 307,618)\n')

    OUT.mkdir(parents=True, exist_ok=True)
    train = (tf.data.Dataset.from_tensor_slices((train_x, train_y))
             .shuffle(len(train_x)).batch(BATCH))
    val = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH)
    history = model.fit(
        train, epochs=EPOCHS, validation_data=val,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
            str(OUT / 'best.keras'), monitor='val_mean_euclidean_dist',
            mode='min', save_best_only=True, verbose=0)])

    best = float(np.min(history.history['val_mean_euclidean_dist']))
    print(f'\n  best validation miss {best:.2f} px   (2022 logged {TARGET_MISS:.2f} px)')

    # Export with the Resizing layer in front, so the game can hand it 768x768
    # exactly as it does every other v2 model.
    model.load_weights(str(OUT / 'best.keras'))
    wrapped = tf.keras.Sequential([
        tf.keras.layers.Input((768, 768, 3)),
        tf.keras.layers.Resizing(256, 256),
        model])
    wrapped.save(str(OUT / 'saved_model'))
    blob = tf.lite.TFLiteConverter.from_keras_model(wrapped).convert()
    (OUT / 'retrained_2026.tflite').write_bytes(blob)
    (OUT / 'history.json').write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in history.history.items()}, indent=1))
    print(f'  wrote {OUT.relative_to(ROOT)}/retrained_2026.tflite ({len(blob):,} bytes)')
    print(f'\n  score it with:  venv/Scripts/python.exe training/eval_pygame_era.py')


if __name__ == '__main__':
    main()
