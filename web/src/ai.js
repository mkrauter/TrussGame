// Turns a Truss into the graph the model eats, and its output back into a click.
//
// This mirrors trussnet/graph_data.py feature for feature. If you change one,
// change the other and re-run training/verify_export.py, which compares the two
// implementations end to end on real trusses.

import { PHYSICS } from './config.js';

// Standard normal via Box-Muller. `random` is injectable so the verification
// harness can drive it with the same numbers Python used.
function gaussian(random) {
  let u = 0;
  let v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * How the AI perceives a truss.
 *
 * Connectivity, the supports and the loaded node are drawn unambiguously on
 * screen, so the AI is given those exactly -- a player reads them off the
 * canvas without effort. Only the coordinates are uncertain, and `sigma` is how
 * uncertain, in screen pixels. That is the fairness dial: measurement puts
 * exact geometry at only ~2 points of score above human-level perception, while
 * doing the mechanics at all is worth ~40.
 */
export function perceive(truss, { sigma = 0, random = Math.random } = {}) {
  return truss.nodes.map((p) =>
    sigma > 0
      ? [p[0] + sigma * gaussian(random), p[1] + sigma * gaussian(random)]
      : [p[0], p[1]]
  );
}

export function buildGraph(truss, seen) {
  const n = seen.length;
  const [a, b] = truss.supports;
  const span = Math.hypot(seen[b][0] - seen[a][0], seen[b][1] - seen[a][1]);
  const centre = [(seen[a][0] + seen[b][0]) / 2, (seen[a][1] + seen[b][1]) / 2];

  // Non-dimensionalising by the support span removes the scale degree of
  // freedom exactly: scaling a truss scales its displacements by the same
  // factor, so the model never has to learn that.
  const pos = seen.map((p) => [(p[0] - centre[0]) / span, (p[1] - centre[1]) / span]);

  const isSupport = new Float32Array(n);
  for (const i of truss.supports) isSupport[i] = 1;
  const freeMask = new Float32Array(n);
  for (let i = 0; i < n; i++) freeMask[i] = 1 - isSupport[i];

  const fHat = PHYSICS.force / (PHYSICS.E * PHYSICS.A);

  const nodeFeat = new Float32Array(n * 7);
  for (let i = 0; i < n; i++) {
    const o = i * 7;
    nodeFeat[o] = freeMask[i];
    nodeFeat[o + 1] = isSupport[i];
    nodeFeat[o + 2] = i === truss.loadedNode ? 1 : 0;
    nodeFeat[o + 3] = 0;                                        // fx
    nodeFeat[o + 4] = i === truss.loadedNode ? fHat : 0;        // fy, downward
    nodeFeat[o + 5] = pos[i][0];
    nodeFeat[o + 6] = pos[i][1];
  }

  // Members are undirected; the model passes messages along each one twice,
  // once toward each end.
  // Ordering matches graph_data.py exactly -- every forward direction, then
  // every reverse. Sum aggregation makes the order irrelevant to the result,
  // but keeping them identical means the two implementations can be compared
  // element by element rather than only at the output.
  const count = truss.elements.length;
  const edges = count * 2;
  const edgeIndex = new Int32Array(edges * 2);
  const edgeFeat = new Float32Array(edges * 4);
  for (let k = 0; k < edges; k++) {
    const [i, j] = truss.elements[k % count];
    const [src, dst] = k < count ? [i, j] : [j, i];
    edgeIndex[k * 2] = src;
    edgeIndex[k * 2 + 1] = dst;
    const dx = pos[dst][0] - pos[src][0];
    const dy = pos[dst][1] - pos[src][1];
    const length = Math.hypot(dx, dy);
    const c = dx / length;
    const s = dy / length;
    // Exactly the entries of the element stiffness matrix, so the network is
    // handed the ingredients of K and only has to learn to invert it.
    edgeFeat[k * 4] = c * c;
    edgeFeat[k * 4 + 1] = c * s;
    edgeFeat[k * 4 + 2] = s * s;
    edgeFeat[k * 4 + 3] = 1 / length;
  }

  return { nodeFeat, edgeIndex, edgeFeat, freeMask, nodes: n, edges, span, centre };
}

/**
 * Where the AI clicks.
 *
 * It clicks at the node where it *believes* the node is, plus the displacement
 * it predicts -- so its perceptual error costs it twice, exactly as it would a
 * player. Scoring against the true settled position is then honest.
 */
export function predictClick(model, truss, { sigma = model.sigma, rounds, random } = {}) {
  const seen = perceive(truss, { sigma, random });
  const graph = buildGraph(truss, seen);
  const u = model.predict(graph, rounds);
  const n = truss.loadedNode;
  return [
    seen[n][0] + u[n * 2] * graph.span,
    seen[n][1] + u[n * 2 + 1] * graph.span,
  ];
}
