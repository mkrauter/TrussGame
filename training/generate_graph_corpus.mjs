// Build a training corpus of truss *structures* rather than screenshots.
//
//   node generate_graph_corpus.mjs --split train --count 20000 --seed-base 0
//   node generate_graph_corpus.mjs --split val   --count 2000  --seed-base 1000000
//
// Seeds match generate_corpus.mjs exactly, so a graph model and a pixel model
// can be scored on the same trusses.
//
// Why structures: the whole problem is 21 numbers -- 10 node coordinates plus
// which node is loaded. Supports are the extreme-x nodes, connectivity is the
// Delaunay triangulation of those coordinates, and the load is a constant. A
// rendered frame encodes those 21 numbers in 196,608 pixels, and measurement
// puts the value of recovering them exactly at ~2 points of game score against
// human-level perception, versus ~40 points for doing the mechanics. So the
// pixels were buying almost nothing and costing the whole compute budget.
//
// Perceptual fairness is handled at training time instead, by jittering the
// coordinates the model receives (see PERCEPTION in train_gnn.py). That makes
// the handicap an explicit parameter rather than a side effect of a downsample.
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Truss } from '../web/src/truss.js';
import { mulberry32 } from '../web/src/random.js';
import { PHYSICS } from '../web/src/config.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, [])
);

const SPLIT = args.split ?? 'train';
const COUNT = Number(args.count ?? 64);
const SEED_BASE = Number(args['seed-base'] ?? (SPLIT === 'val' ? 1_000_000 : 0));
const OUT = path.resolve(args.out ?? path.join(HERE, 'graph_corpus'), SPLIT);

const t0 = Date.now();
await fsp.mkdir(OUT, { recursive: true });

const samples = [];
for (let i = 0; i < COUNT; i++) {
  const seed = SEED_BASE + i;
  const truss = new Truss(mulberry32(seed));
  truss.calculate(PHYSICS.force);

  // The full displacement field, not just the loaded node. The solve produces
  // it anyway, and it is 8 free nodes' worth of supervision per sample instead
  // of 1 -- the pixel corpus threw all but one of them away.
  const displacement = truss.nodes.map((p, n) => [
    truss.nodesMoved[n][0] - p[0],
    truss.nodesMoved[n][1] - p[1],
  ]);

  samples.push({
    seed,
    nodes: truss.nodes,
    elements: truss.elements,
    supports: truss.supports,
    loadedNode: truss.loadedNode,
    displacement,
  });

  if ((i + 1) % 2000 === 0) {
    process.stdout.write(`\r  ${i + 1}/${COUNT}   `);
  }
}

const meta = {
  split: SPLIT,
  count: COUNT,
  seedBase: SEED_BASE,
  generatedAt: new Date().toISOString(),
  physics: { ...PHYSICS },
  note: 'exact linear-static solve; displacement is the full field, screen pixels',
};
const file = path.join(OUT, 'graphs.json');
await fsp.writeFile(file, JSON.stringify({ meta, samples }));

const { size } = await fsp.stat(file);
console.log(`\n\nwrote ${samples.length} trusses to ${file}`);
console.log(`  ${(size / 1e6).toFixed(1)} MB (${(size / samples.length).toFixed(0)} bytes each)`);
console.log(`  ${((Date.now() - t0) / 1000).toFixed(1)}s total`);
