// Does a smaller truss make the game more interesting, or just easier?
//
//   node node_count_baselines.mjs
//
// Ten Delaunay-triangulated nodes are highly redundant: load spreads through
// many paths and the settled displacement clusters near its average. That is
// why "ignore the structure, move the node down by the mean distance" scores
// 59.5% -- roughly what an experienced human scores. The scoring rule divides
// the miss by the travel, so a game where travel barely varies is a game where
// knowing one constant is most of the skill.
//
// Fewer nodes means fewer load paths, so displacement should depend far more
// on geometry. If it does, the constant-guess baseline falls and there is room
// for a player -- or a model -- to earn points by actually reading the truss.
// If the baseline holds, six nodes is merely easier to look at.
//
// Constants are fitted on one seed range and scored on a disjoint one, so no
// baseline is graded on the data that tuned it.
import { Truss, accuracy } from '../web/src/truss.js';
import { mulberry32 } from '../web/src/random.js';
import { PHYSICS } from '../web/src/config.js';

const COUNTS = [4, 5, 6, 7, 8, 10];
const N = 4000;
const TRAIN_BASE = 1_000_000;
const VAL_BASE = 5_000_000;

// Thinning the truss risks producing a mechanism -- too few members to brace
// the free nodes, so K is singular and there is no settled state to predict.
// Those are skipped and counted: the rejection rate is itself a design number,
// because a game that has to discard boards is a worse game.
function sample(numNodes, base, n) {
  const start = [];
  const end = [];
  const span = [];
  let seed = base;
  let rejected = 0;
  while (start.length < n) {
    const truss = new Truss(mulberry32(seed++), { numNodes });
    try {
      truss.calculate(PHYSICS.force);
    } catch {
      rejected += 1;
      continue;
    }
    start.push(truss.loadedStart);
    end.push(truss.loadedEnd);
    const [a, b] = truss.supports;
    span.push(Math.hypot(truss.nodes[a][0] - truss.nodes[b][0],
                         truss.nodes[a][1] - truss.nodes[b][1]));
  }
  return { start, end, span, rejected, drawn: seed - base };
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const travelOf = (s) => s.start.map((p, i) =>
  Math.hypot(s.end[i][0] - p[0], s.end[i][1] - p[1]));

function score(split, guessFor) {
  const scores = split.start.map((p, i) =>
    accuracy(p, split.end[i], guessFor(p, i, split)));
  return mean(scores);
}

console.log(`  ${N} trusses fitted, ${N} disjoint trusses scored, per node count\n`);
console.log('  ' + 'nodes'.padStart(6) + 'travel'.padStart(9) + 'spread'.padStart(9)
  + 'cv'.padStart(7) + 'mean-delta'.padStart(11) + 'down-mean'.padStart(11)
  + 'down-span'.padStart(11) + 'best'.padStart(8) + 'mechanisms'.padStart(12));

for (const numNodes of COUNTS) {
  const train = sample(numNodes, TRAIN_BASE, N);
  const val = sample(numNodes, VAL_BASE, N);

  const dx = mean(train.start.map((p, i) => train.end[i][0] - p[0]));
  const dy = mean(train.start.map((p, i) => train.end[i][1] - p[1]));
  const t = travelOf(train);
  const meanTravel = mean(t);
  const meanOverSpan = mean(t.map((v, i) => v / train.span[i]));

  const vt = travelOf(val);
  const sd = Math.sqrt(mean(vt.map((v) => (v - mean(vt)) ** 2)));

  const a = score(val, (p) => [p[0] + dx, p[1] + dy]);
  const b = score(val, (p) => [p[0], p[1] + meanTravel]);
  const c = score(val, (p, i, s) => [p[0], p[1] + meanOverSpan * s.span[i]]);
  const best = Math.max(a, b, c);

  console.log('  ' + String(numNodes).padStart(6) + `${mean(vt).toFixed(1).padStart(8)}px`
    + `${sd.toFixed(1).padStart(8)}px${(sd / mean(vt)).toFixed(2).padStart(7)}`
    + `${a.toFixed(2).padStart(10)}%${b.toFixed(2).padStart(10)}%`
    + `${c.toFixed(2).padStart(10)}%${best.toFixed(2).padStart(7)}%`
    + `${(val.rejected / val.drawn * 100).toFixed(1).padStart(11)}%`);
}
console.log('\n  cv = spread/mean of travel. Higher cv means a constant guess'
  + '\n  explains less, so there is more for a player to actually win.');
