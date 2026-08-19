// Stage one: recover the truss from the rendered frame.
//
// This is what keeps the contest fair -- everything below reads the canvas and
// nothing else. Node positions and roles come from a small convolutional net;
// connectivity comes from looking at whether a line is drawn between each pair,
// at full canvas resolution.
//
// The physics then runs on what was seen, in gnn.js. Splitting it that way is
// not a convenience: convolutions are the right tool for finding marks in an
// image and the wrong tool for a global implicit solve, which is why widening
// the old end-to-end CNN's receptive field made it worse rather than better.

import { CROP, RENDER } from './config.js';
import { decodeOps, runOps } from './convnet.js';
import { captureModelInput } from './capture.js';

const FREE = 0;
const SUPPORT = 1;
const LOADED = 2;

// grey25, matching RENDER.background. Parsed rather than hard-coded so the two
// cannot drift apart.
const BACKGROUND = (() => {
  const m = RENDER.background.match(/^#(\w{2})(\w{2})(\w{2})$/);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [64, 64, 64];
})();

// Member sampling, tuned by training/sweep_members.mjs against the *real*
// detector's node error rather than against ground-truth positions. That
// distinction cost a run: parameters tuned on exact nodes recovered the member
// list on 97% of frames in Python and on 30% in the actual pipeline. Current
// settings recover it exactly on 138 of 150 frames.
const TRIM = 0.18;
// Markers are a fixed size in pixels, not a fraction of a member, so the ends
// have to be skipped in pixels too. A fractional trim alone sampled short
// members inside their own marker: when the markers grew to 30px it put a
// spurious edge in every single frame.
const TRIM_PX = 38;
const SAMPLES = 24;
const INK_THRESHOLD = 40;
// halfWidth 1 rather than 2 because the 384px detector localises to ~1px; the
// search only has to cover the error that is actually there. coverage 0.9 over
// 0.8 costs one frame in 150 and buys zero spurious members, and an invented
// member stiffens the structure in a way a missing one does not.
const COVERAGE = 0.9;
const HALF_WIDTH = 1;
const ON_SEGMENT = 4.0;

export class TrussDetector {
  constructor(payload) {
    if (payload.format !== 'trussdetector/1') {
      throw new Error(`unsupported detector format ${payload.format}`);
    }
    this.inputSize = payload.inputSize;
    this.stride = payload.stride;
    this.classes = payload.classes;
    this.counts = payload.counts;
    this.medianPx = payload.medianPx;
    this.trunk = decodeOps(payload.trunk);
    this.heatmapOp = decodeOps([payload.heatmap])[0];
    this.offsetOp = decodeOps([payload.offset])[0];
    this.canvas = null;
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load detector from ${url}: ${response.status}`);
    return new TrussDetector(await response.json());
  }

  // The crop and downscale go through capture.js, the same path the training
  // corpus was rendered through, so the pixels the net sees at play time are
  // the pixels it learned from.
  _input(sourceCanvas) {
    if (this.canvas === null) {
      this.canvas =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(this.inputSize, this.inputSize)
          : Object.assign(document.createElement('canvas'),
                          { width: this.inputSize, height: this.inputSize });
    }
    const ctx = captureModelInput(sourceCanvas, this.canvas);
    const { data } = ctx.getImageData(0, 0, this.inputSize, this.inputSize);

    const n = this.inputSize * this.inputSize;
    const chw = new Float32Array(3 * n);
    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      chw[p] = data[i] / 127.5 - 1;
      chw[n + p] = data[i + 1] / 127.5 - 1;
      chw[2 * n + p] = data[i + 2] / 127.5 - 1;
    }
    return chw;
  }

  /** Node positions (screen pixels) and roles, read from the frame. */
  findNodes(sourceCanvas) {
    const features = runOps(this.trunk, this._input(sourceCanvas), this.inputSize, this.inputSize);
    const heat = runOps([this.heatmapOp], features.data, features.width, features.height);
    const offset = runOps([this.offsetOp], features.data, features.width, features.height);

    const g = heat.width;
    const cells = g * g;
    const scale = CROP.width / this.inputSize;

    const nodes = [];
    for (let c = 0; c < this.classes; c++) {
      const plane = heat.data.subarray(c * cells, (c + 1) * cells);
      // 3x3 non-maximum suppression, then take the strongest `counts[c]`. The
      // counts are properties of the game -- two supports, one loaded node --
      // so this never has to threshold and hope.
      const peaks = [];
      for (let y = 0; y < g; y++) {
        for (let x = 0; x < g; x++) {
          const v = plane[y * g + x];
          let best = true;
          for (let dy = -1; dy <= 1 && best; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const ny = y + dy;
              const nx = x + dx;
              if (ny < 0 || ny >= g || nx < 0 || nx >= g) continue;
              if (plane[ny * g + nx] > v) { best = false; break; }
            }
          }
          if (best) peaks.push([v, x, y]);
        }
      }
      peaks.sort((a, b) => b[0] - a[0]);
      for (const [score, x, y] of peaks.slice(0, this.counts[c])) {
        // The offset head recovers what the stride threw away; without it the
        // best possible error is half a cell, which is 12 screen pixels.
        const ox = offset.data[y * g + x];
        const oy = offset.data[cells + y * g + x];
        nodes.push({
          position: [
            (x + ox) * this.stride * scale + CROP.x,
            (y + oy) * this.stride * scale + CROP.y,
          ],
          role: c,
          score: 1 / (1 + Math.exp(-score)),
        });
      }
    }
    return nodes;
  }
}

function bilinear(pixels, width, height, x, y) {
  const cx = Math.min(Math.max(x, 0), width - 1.001);
  const cy = Math.min(Math.max(y, 0), height - 1.001);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const fx = cx - x0;
  const fy = cy - y0;
  const at = (px, py, ch) => pixels[(py * width + px) * 4 + ch];
  const out = [0, 0, 0];
  for (let ch = 0; ch < 3; ch++) {
    const top = at(x0, y0, ch) * (1 - fx) + at(x0 + 1, y0, ch) * fx;
    const bottom = at(x0, y0 + 1, ch) * (1 - fx) + at(x0 + 1, y0 + 1, ch) * fx;
    out[ch] = top * (1 - fy) + bottom * fy;
  }
  return out;
}

/** Is another node sitting on the segment a-b? See training/members.py. */
function passesThroughNode(a, b, points, tolerance = ON_SEGMENT) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const length = Math.hypot(abx, aby);
  if (length === 0) return false;
  for (const p of points) {
    if (p === a || p === b) continue;
    const px = p[0] - a[0];
    const py = p[1] - a[1];
    const t = (px * abx + py * aby) / (length * length);
    if (t <= 0 || t >= 1) continue;
    if (Math.abs(abx * py - aby * px) / length < tolerance) return true;
  }
  return false;
}

/**
 * Which node pairs are joined by a drawn member.
 *
 * Reads the full-resolution canvas, not the detector's downscaled input: both
 * are the same screenshot, and there is no reason to ask "is a line here" of a
 * downsample that renders 2.5px members 0.83px wide.
 */
export function findMembers(sourceCanvas, points, options = {}) {
  const {
    coverage = COVERAGE,
    halfWidth = HALF_WIDTH,
    onSegment = ON_SEGMENT,
    pixels = null,
  } = options;

  let frame = pixels;
  if (frame === null) {
    const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true });
    frame = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  }
  const { data, width, height } = frame;

  // Look a little way to either side of the line. Node positions come from the
  // detector and carry about a pixel of error, while members are 2.5px wide --
  // without this, that error walks the sampled line clean off the member.
  const offsets = [];
  for (let o = -halfWidth; o <= halfWidth + 1e-9; o += 1) offsets.push(o);

  const members = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      if (passesThroughNode(points[i], points[j], points, onSegment)) continue;

      const dx = points[j][0] - points[i][0];
      const dy = points[j][1] - points[i][1];
      const length = Math.hypot(dx, dy);
      if (length === 0) continue;
      const nx = -dy / length;
      const ny = dx / length;

      // Clear the markers at both ends before sampling anything.
      const trim = Math.max(TRIM, TRIM_PX / length);
      if (trim >= 0.45) continue;        // too short to have a clear middle

      let drawn = 0;
      for (let s = 0; s < SAMPLES; s++) {
        const t = trim + ((1 - 2 * trim) * s) / (SAMPLES - 1);
        const px = points[i][0] + dx * t;
        const py = points[i][1] + dy * t;
        let best = 0;
        for (const o of offsets) {
          const [r, g, b] = bilinear(data, width, height, px + nx * o, py + ny * o);
          best = Math.max(best, Math.abs(r - BACKGROUND[0]), Math.abs(g - BACKGROUND[1]),
                          Math.abs(b - BACKGROUND[2]));
        }
        if (best > INK_THRESHOLD) drawn++;
      }
      if (drawn / SAMPLES >= coverage) members.push([i, j]);
    }
  }
  return members;
}

/**
 * The whole of stage one: a rendered frame in, a truss out.
 *
 * The returned object has the same shape the graph model expects from a real
 * Truss -- nodes, elements, supports, loadedNode -- so stage two cannot tell
 * whether it was handed a perceived truss or a real one.
 */
export function readTruss(detector, sourceCanvas) {
  const found = detector.findNodes(sourceCanvas);
  const nodes = found.map((n) => n.position);
  const supports = found
    .map((n, i) => (n.role === SUPPORT ? i : -1))
    .filter((i) => i >= 0);
  const loadedNode = found.findIndex((n) => n.role === LOADED);
  return {
    nodes,
    elements: findMembers(sourceCanvas, nodes),
    supports,
    loadedNode,
    scores: found.map((n) => n.score),
  };
}

export { FREE, SUPPORT, LOADED };
