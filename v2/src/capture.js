// The single path from a rendered game canvas to the array the model sees.
//
// Used by the corpus generator when building training data, and by the game
// when asking the model for a prediction. One implementation, so the two cannot
// disagree about the crop, the downscale filter, or the channel order.

import { CROP, MODEL } from './config.js';

// Allocate once and reuse -- this runs every frame in the game.
export function createCaptureCanvas() {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(MODEL.inputSize, MODEL.inputSize);
  }
  const canvas = document.createElement('canvas');
  canvas.width = MODEL.inputSize;
  canvas.height = MODEL.inputSize;
  return canvas;
}

// Crop to the model's region and downscale it onto `target`.
export function captureModelInput(sourceCanvas, target) {
  const ctx = target.getContext('2d', { willReadFrequently: true });
  // Both set explicitly: the defaults differ between browsers, and this is a
  // pixel-affecting choice like any other.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = MODEL.smoothingQuality;
  // Scale to whatever `target` actually is rather than to MODEL.inputSize, so
  // the two cannot silently disagree and so the resolution can be varied for
  // experiments without editing this file.
  ctx.drawImage(
    sourceCanvas,
    CROP.x, CROP.y, CROP.width, CROP.height,
    0, 0, target.width, target.height
  );
  return ctx;
}

// ImageData is RGBA; the model wants RGB. Alpha is always 255 here because the
// scene paints an opaque background first, so it is dropped rather than
// composited.
export function toRGB(imageData) {
  const { data, width, height } = imageData;
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    out[j] = data[i];
    out[j + 1] = data[i + 1];
    out[j + 2] = data[i + 2];
  }
  return out;
}
