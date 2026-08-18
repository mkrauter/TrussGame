// Runtime for the graph model, in plain JavaScript.
//
// The network is 42k parameters of small MLPs plus gather/scatter over ~42
// directed edges, so it needs no ONNX runtime, no WASM, and no network fetch
// beyond the weights themselves. Measured ~2.3ms per message-passing round,
// so ~23ms for a full-strength move -- invisible when the game asks for one
// prediction per round, but do not put this in a per-frame loop.
//
// This file must compute exactly what trussnet/gnn.py computes. That is not a
// hope -- training/verify_export.py checks the two against each other on real
// trusses and fails loudly if they drift.

function decode(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  // Little-endian float32, matching the '<f4' the exporter writes. Every
  // platform that runs a browser is little-endian, so the view is free.
  return new Float32Array(bytes.buffer);
}

function decodeMLP(layers) {
  return layers.map((layer) =>
    layer.type === 'linear'
      ? { type: 'linear', in: layer.in, out: layer.out, w: decode(layer.weight), b: decode(layer.bias) }
      : { type: layer.type }
  );
}

// y = x @ W^T + b, for `rows` stacked inputs. W is row-major (out, in), which
// is how torch stores nn.Linear.
function linear(layer, x, rows) {
  const { in: nin, out: nout, w, b } = layer;
  const y = new Float32Array(rows * nout);
  for (let r = 0; r < rows; r++) {
    const xo = r * nin;
    const yo = r * nout;
    for (let o = 0; o < nout; o++) {
      let acc = b[o];
      const wo = o * nin;
      for (let i = 0; i < nin; i++) acc += w[wo + i] * x[xo + i];
      y[yo + o] = acc;
    }
  }
  return y;
}

function silu(x) {
  for (let i = 0; i < x.length; i++) x[i] = x[i] / (1 + Math.exp(-x[i]));
  return x;
}

function runMLP(layers, x, rows) {
  let out = x;
  for (const layer of layers) {
    out = layer.type === 'linear' ? linear(layer, out, rows) : silu(out);
  }
  return out;
}

// Normalises each row over its features, matching torch.nn.LayerNorm -- which
// uses the biased variance, not the sample variance.
function layerNorm(x, rows, dim, weight, bias, eps) {
  for (let r = 0; r < rows; r++) {
    const o = r * dim;
    let mean = 0;
    for (let i = 0; i < dim; i++) mean += x[o + i];
    mean /= dim;
    let variance = 0;
    for (let i = 0; i < dim; i++) {
      const d = x[o + i] - mean;
      variance += d * d;
    }
    variance /= dim;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let i = 0; i < dim; i++) x[o + i] = (x[o + i] - mean) * inv * weight[i] + bias[i];
  }
  return x;
}

export class TrussGNN {
  constructor(payload) {
    if (payload.format !== 'trussgnn/1') {
      throw new Error(`unsupported model format ${payload.format}`);
    }
    this.hidden = payload.hidden;
    this.rounds = payload.rounds;
    this.sigma = payload.sigma;
    this.valMean = payload.valMean;
    this.eps = payload.layerNormEps;
    this.nodeEncoder = decodeMLP(payload.nodeEncoder);
    this.edgeEncoder = decodeMLP(payload.edgeEncoder);
    this.message = decodeMLP(payload.message);
    this.update = decodeMLP(payload.update);
    this.decoder = decodeMLP(payload.decoder);
    this.normWeight = decode(payload.normWeight);
    this.normBias = decode(payload.normBias);
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load model from ${url}: ${response.status}`);
    return new TrussGNN(await response.json());
  }

  /**
   * @param graph  {nodeFeat, edgeIndex, edgeFeat, freeMask, nodes, edges}
   * @param rounds how many message-passing rounds to run. Fewer than the model
   *   was trained with is a less-converged solver, which is the difficulty
   *   dial: 1 round plays at ~25%, 6 at ~68%, 10 at ~96%.
   * @returns Float32Array(nodes * 2) of displacements, in units of the support
   *   span the caller measured.
   */
  predict(graph, rounds = this.rounds) {
    const { nodeFeat, edgeIndex, edgeFeat, freeMask, nodes, edges } = graph;
    const H = this.hidden;

    let h = runMLP(this.nodeEncoder, Float32Array.from(nodeFeat), nodes);
    const e = runMLP(this.edgeEncoder, Float32Array.from(edgeFeat), edges);

    const catIn = new Float32Array(edges * 3 * H);
    const updIn = new Float32Array(nodes * 2 * H);

    for (let t = 0; t < rounds; t++) {
      for (let k = 0; k < edges; k++) {
        const src = edgeIndex[k * 2] * H;
        const dst = edgeIndex[k * 2 + 1] * H;
        const o = k * 3 * H;
        for (let i = 0; i < H; i++) {
          catIn[o + i] = h[src + i];
          catIn[o + H + i] = h[dst + i];
          catIn[o + 2 * H + i] = e[k * H + i];
        }
      }
      const m = runMLP(this.message, catIn, edges);

      // Sum aggregation at the destination node: assembling K sums every
      // member meeting at a node, so a busier node really is stiffer.
      const agg = new Float32Array(nodes * H);
      for (let k = 0; k < edges; k++) {
        const dst = edgeIndex[k * 2 + 1] * H;
        const o = k * H;
        for (let i = 0; i < H; i++) agg[dst + i] += m[o + i];
      }

      for (let n = 0; n < nodes; n++) {
        for (let i = 0; i < H; i++) {
          updIn[n * 2 * H + i] = h[n * H + i];
          updIn[n * 2 * H + H + i] = agg[n * H + i];
        }
      }
      const delta = runMLP(this.update, Float32Array.from(updIn), nodes);
      for (let i = 0; i < h.length; i++) h[i] += delta[i];
      h = layerNorm(h, nodes, H, this.normWeight, this.normBias, this.eps);
    }

    const out = runMLP(this.decoder, h, nodes);
    // Supports are pinned; the boundary condition is imposed, never predicted.
    for (let n = 0; n < nodes; n++) {
      out[n * 2] *= freeMask[n];
      out[n * 2 + 1] *= freeMask[n];
    }
    return out;
  }
}
