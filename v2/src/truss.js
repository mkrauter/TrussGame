import { TRUSS, PHYSICS } from './config.js';
import { triangulate, edgesOf } from './delaunay.js';
import { solve, zeros } from './linalg.js';

export class Truss {
  // `rng` is any function returning [0, 1). Pass a seeded one for reproducible
  // trusses; defaults to Math.random for interactive play.
  constructor(rng = Math.random, options = {}) {
    const cfg = { ...TRUSS, ...options };

    this.nodes = sampleNodes(rng, cfg);
    this.elements = edgesOf(triangulate(this.nodes));

    // Supports are the extreme-x nodes, both fully pinned.
    const xs = this.nodes.map((p) => p[0]);
    this.supports = [argmin(xs), argmax(xs)];

    const moving = [];
    for (let i = 0; i < cfg.numNodes; i++) {
      if (!this.supports.includes(i)) moving.push(i);
    }
    this.loadedNode = moving[Math.floor(rng() * moving.length)];

    // Free degrees of freedom, in ascending order.
    this.movingRows = moving
      .flatMap((i) => [i * 2, i * 2 + 1])
      .sort((a, b) => a - b);

    this.sigmas = new Array(this.elements.length).fill(0);
    this.nodesMoved = this.nodes.map((p) => [...p]);
  }

  // Linear static solve. K is assembled from the undeformed geometry and never
  // updated, so this is exact rather than iterative -- deformations look large
  // but the theory is small-displacement.
  calculate(force, E = PHYSICS.E, A = PHYSICS.A) {
    const n = this.nodes.length;
    const size = n * 2;
    const K = zeros(size, size);
    const f = new Array(size).fill(0);

    // Load is vertical, downward in screen coordinates.
    f[this.loadedNode * 2 + 1] = force;

    for (const e of this.elements) {
      const { rows, length, c, s } = geometry(e, this.nodes);
      const k = (E * A) / length;
      const local = [
        [c * c, c * s, -c * c, -c * s],
        [c * s, s * s, -c * s, -s * s],
        [-c * c, -c * s, c * c, c * s],
        [-c * s, -s * s, c * s, s * s],
      ];
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) K[rows[i]][rows[j]] += k * local[i][j];
      }
    }

    const free = this.movingRows;
    const Kff = free.map((r) => free.map((c) => K[r][c]));
    const ff = free.map((r) => f[r]);
    const uf = solve(Kff, ff);

    const u = new Array(size).fill(0);
    free.forEach((row, i) => { u[row] = uf[i]; });

    this.nodesMoved = this.nodes.map((p, i) => [p[0] + u[i * 2], p[1] + u[i * 2 + 1]]);

    this.sigmas = this.elements.map((e) => {
      // Undeformed geometry, matching the K this solve used. Taking the
      // direction cosines from nodesMoved instead mixes linear theory with a
      // deformed configuration and breaks nodal equilibrium -- in v1 that put
      // some members fully red when they should have been fully blue.
      // (v1 also read dx and dy from different arrays here; both come from the
      // same one now.)
      const { rows, length, c, s } = geometry(e, this.nodes);
      const d0 = c * u[rows[0]] + s * u[rows[1]];
      const d1 = c * u[rows[2]] + s * u[rows[3]];
      return (E * (d1 - d0)) / length;
    });
  }

  get loadedStart() {
    return this.nodes[this.loadedNode];
  }

  get loadedEnd() {
    return this.nodesMoved[this.loadedNode];
  }
}

function sampleNodes(rng, cfg) {
  const points = [];
  let attempts = 0;
  while (points.length < cfg.numNodes) {
    if (++attempts > cfg.maxSampleAttempts) {
      throw new Error(
        `could not place ${cfg.numNodes} nodes with minDistance ${cfg.minDistance}`
      );
    }
    const p = [
      cfg.size[0] * rng() + cfg.offset[0],
      cfg.size[1] * rng() + cfg.offset[1],
    ];
    const clear = points.every(
      (q) => Math.hypot(p[0] - q[0], p[1] - q[1]) >= cfg.minDistance
    );
    if (clear) points.push(p);
  }
  return points;
}

function geometry(e, nodes) {
  const rows = [e[0] * 2, e[0] * 2 + 1, e[1] * 2, e[1] * 2 + 1];
  const dx = nodes[e[1]][0] - nodes[e[0]][0];
  const dy = nodes[e[1]][1] - nodes[e[0]][1];
  const length = Math.hypot(dx, dy);
  return { rows, length, c: dx / length, s: dy / length };
}

const argmin = (a) => a.reduce((best, v, i) => (v < a[best] ? i : best), 0);
const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);

// Game scoring: 100% means you clicked exactly on the settled position, 0%
// means you were at least as far off as the node actually travelled.
export function accuracy(start, end, guess) {
  const travelled = Math.hypot(start[0] - end[0], start[1] - end[1]);
  if (travelled === 0) return 0;
  const missed = Math.hypot(guess[0] - end[0], guess[1] - end[1]);
  return Math.max(0, (1 - missed / travelled) * 100);
}
