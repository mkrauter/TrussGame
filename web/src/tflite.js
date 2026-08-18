// Replays truss_game_AI_model.tflite in the browser.
//
// The v2 model is a historic artifact -- a 2023 Keras CNN, kept because it is
// what the project shipped, not because it is good. Rather than pull a TFLite
// runtime into the page for one frozen model, training/export_tflite.py lifts
// its weights out of the flatbuffer and this file replays the handful of ops it
// uses. training/verify_tflite.mjs checks the replay against LiteRT.
//
// It is heavy: 1.7 GMAC, three quarters of it in the first three convolutions
// at 254x254. Measured at ~0.7s per move here, so about 2.4 GMAC/s -- the loop
// order matters enormously, and iterating output channels innermost over
// contiguous weights is several times faster than the obvious arrangement.
// It still runs in a Web Worker: 0.7s of blocked main thread is a visible stall.
//
// Tensors are NHWC with C contiguous, matching TFLite, so the weight layout can
// be used exactly as exported.

function decode(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function resizeBilinear(t, height, width) {
  const { data, height: ih, width: iw, channels: c } = t;
  const out = new Float32Array(height * width * c);
  // Half-pixel centres: sample at (i + 0.5) * scale - 0.5, not i * scale. This
  // model uses them, checked against LiteRT's own resize output -- the plain
  // mapping is off by 191 intensity levels at worst and moved the final
  // prediction by up to 55px.
  const sy = ih / height;
  const sx = iw / width;
  for (let y = 0; y < height; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), ih - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, ih - 1);
    const wy = fy - y0;
    for (let x = 0; x < width; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), iw - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, iw - 1);
      const wx = fx - x0;
      const o = (y * width + x) * c;
      for (let k = 0; k < c; k++) {
        const a = data[(y0 * iw + x0) * c + k];
        const b = data[(y0 * iw + x1) * c + k];
        const d = data[(y1 * iw + x0) * c + k];
        const e = data[(y1 * iw + x1) * c + k];
        out[o + k] = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
      }
    }
  }
  return { data: out, height, width, channels: c };
}

// Valid padding throughout, which is why every conv shrinks the map by k-1.
function conv2d(op, t) {
  const { data, height: ih, width: iw, channels: cin } = t;
  const { k, cout, weight, bias, activation } = op;
  const oh = ih - k + 1;
  const ow = iw - k + 1;
  const out = new Float32Array(oh * ow * cout);
  const relu = activation === 'relu';

  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const o = (y * ow + x) * cout;
      for (let f = 0; f < cout; f++) out[o + f] = bias[f];

      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const inBase = ((y + ky) * iw + (x + kx)) * cin;
          const wBase = (ky * k + kx) * cin;
          for (let ci = 0; ci < cin; ci++) {
            const v = data[inBase + ci];
            if (v === 0) continue;
            // weight is (cout, kh, kw, cin), so stride over filters is k*k*cin.
            let wi = wBase + ci;
            for (let f = 0; f < cout; f++) {
              out[o + f] += weight[f * k * k * cin + wi] * v;
            }
          }
        }
      }
      if (relu) for (let f = 0; f < cout; f++) if (out[o + f] < 0) out[o + f] = 0;
    }
  }
  return { data: out, height: oh, width: ow, channels: cout };
}

function maxPool(op, t) {
  const { data, height: ih, width: iw, channels: c } = t;
  const { k, stride } = op;
  const oh = Math.floor((ih - k) / stride) + 1;
  const ow = Math.floor((iw - k) / stride) + 1;
  const out = new Float32Array(oh * ow * c);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const o = (y * ow + x) * c;
      for (let ch = 0; ch < c; ch++) out[o + ch] = -Infinity;
      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const i = ((y * stride + ky) * iw + (x * stride + kx)) * c;
          for (let ch = 0; ch < c; ch++) {
            if (data[i + ch] > out[o + ch]) out[o + ch] = data[i + ch];
          }
        }
      }
    }
  }
  return { data: out, height: oh, width: ow, channels: c };
}

function dense(op, t) {
  const { weight, bias, in: nin, out: nout, activation } = op;
  const x = t.data;
  const out = new Float32Array(nout);
  for (let o = 0; o < nout; o++) {
    let acc = bias[o];
    const base = o * nin;
    for (let i = 0; i < nin; i++) acc += weight[base + i] * x[i];
    out[o] = activation === 'relu' ? Math.max(0, acc) : acc;
  }
  return { data: out, height: 1, width: 1, channels: nout };
}

export class TrussV2Model {
  constructor(payload) {
    if (payload.format !== 'trussv2/1') {
      throw new Error(`unsupported v2 model format ${payload.format}`);
    }
    this.inputSize = payload.inputSize;
    this.ops = payload.ops.map((op) =>
      op.weight ? { ...op, weight: decode(op.weight), bias: decode(op.bias) } : op
    );
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load v2 model from ${url}: ${response.status}`);
    return new TrussV2Model(await response.json());
  }

  /**
   * @param frame {data, height, width, channels} of the 768x768 crop, in raw
   *   0-255 values -- the original fed pygame's surface straight in, with no
   *   normalisation, and the weights expect that.
   * @returns [x, y] in screen pixels.
   */
  predict(frame) {
    let t = frame;
    for (const op of this.ops) {
      if (op.type === 'resize') t = resizeBilinear(t, op.height, op.width);
      else if (op.type === 'conv') t = conv2d(op, t);
      else if (op.type === 'maxpool') t = maxPool(op, t);
      else if (op.type === 'flatten') t = { ...t, data: t.data };
      else if (op.type === 'dense') t = dense(op, t);
      else throw new Error(`unknown op ${op.type}`);
    }
    return [t.data[0], t.data[1]];
  }
}
