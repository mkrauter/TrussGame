// Check that web/src/gnn.js computes what trussnet/gnn.py computes.
//
//   python verify_export.py && node verify_export.mjs
//
// Compares the input features and the network output, per element, on real
// trusses. Features first, because if those drift the output comparison tells
// you nothing about where the fault is.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Truss } from '../web/src/truss.js';
import { mulberry32 } from '../web/src/random.js';
import { TrussGNN } from '../web/src/gnn.js';
import { buildGraph, perceive } from '../web/src/ai.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const expected = JSON.parse(fs.readFileSync(path.join(HERE, 'expected.json'), 'utf-8'));
const payload = JSON.parse(
  fs.readFileSync(path.join(HERE, '..', 'web', 'src', 'model', 'trussgnn.json'), 'utf-8')
);
const model = new TrussGNN(payload);

let worstNode = 0;
let worstEdge = 0;
let worstOut = 0;
let worstSpan = 0;

for (const sample of expected.samples) {
  const truss = new Truss(mulberry32(sample.seed));
  const graph = buildGraph(truss, perceive(truss, { sigma: 0 }));
  const out = model.predict(graph, expected.rounds);

  worstSpan = Math.max(worstSpan, Math.abs(graph.span - sample.span));

  for (let i = 0; i < graph.nodeFeat.length; i++) {
    worstNode = Math.max(worstNode, Math.abs(graph.nodeFeat[i] - sample.nodeFeat[i]));
  }
  // Python pads its edge arrays to the corpus maximum; JS carries only the
  // real edges. Compare the real ones, and check the padding is inert.
  for (let i = 0; i < graph.edgeFeat.length; i++) {
    worstEdge = Math.max(worstEdge, Math.abs(graph.edgeFeat[i] - sample.edgeFeat[i]));
  }
  for (let i = graph.edges * 4; i < sample.edgeFeat.length; i++) {
    if (sample.edgeFeat[i] !== 0) throw new Error('padded edge feature is not zero');
  }
  for (let i = 0; i < out.length; i++) {
    worstOut = Math.max(worstOut, Math.abs(out[i] - sample.output[i]));
  }
}

const report = (label, value, tol) => {
  const ok = value <= tol;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label.padEnd(28)} max |diff| = ${value.toExponential(2)}`);
  return ok;
};

console.log(`\n${expected.samples.length} trusses, ${expected.rounds} rounds\n`);
const results = [
  // The span is ~640px and Python stores it as float32, so ~1e-4 of rounding
  // is expected here and means nothing; JS keeps it in float64.
  report('support span (px)', worstSpan, 1e-3),
  report('node features', worstNode, 1e-6),
  report('edge features', worstEdge, 1e-6),
  report('network output', worstOut, 1e-5),
];
console.log('\nfloat32 in JS vs float32 in torch, so exact equality is not expected;');
console.log('these tolerances are far below the ~1e-3 span units that would move a pixel.\n');

if (!results.every(Boolean)) process.exit(1);
