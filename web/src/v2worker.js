// Runs the v2 model off the main thread.
//
// truss_game_AI_model.tflite is 1.7 GMAC, measured at about 0.7s of JavaScript,
// three quarters of it in the first three convolutions at 254x254. On the main
// thread that is a visible stall on every new truss; here the board stays live
// and the answer arrives while you are still deciding where to click.

import { TrussV2Model } from './tflite.js';

let model = null;

self.onmessage = async (event) => {
  const { type, id, frame } = event.data;

  if (type === 'load') {
    model = await TrussV2Model.load(new URL('./model/trussv2.json', import.meta.url));
    self.postMessage({ type: 'ready' });
    return;
  }

  if (type === 'predict') {
    const started = performance.now();
    // `frame` arrives as a transferred RGB Float32Array, so nothing is copied.
    const prediction = model.predict({
      data: frame, height: 768, width: 768, channels: 3,
    });
    self.postMessage({ type: 'prediction', id, prediction, ms: performance.now() - started });
  }
};
