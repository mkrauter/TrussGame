// Tune member detection against the real detector's node error.
//
//   node sweep_members.mjs --count 200
//
// Tuning this on ground-truth node positions gave parameters that fell apart in
// the real pipeline -- 139 of 200 frames got the member count wrong. So the
// sweep runs the actual detector and tunes on what it actually produces.
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, [])
);
const COUNT = Number(args.count ?? 150);
const SEED_BASE = Number(args['seed-base'] ?? 1_000_000);

const configs = [];
for (const halfWidth of [1, 2, 3, 4]) {
  for (const onSegment of [2, 4, 6]) {
    for (const coverage of [0.8, 0.9, 1.0]) {
      configs.push({ halfWidth, onSegment, coverage });
    }
  }
}

const MIME = { '.html': 'text/html', '.js': 'application/javascript',
               '.mjs': 'application/javascript', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const file = path.join(REPO, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(REPO) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
page.on('pageerror', (e) => { throw e; });
await page.goto(`http://127.0.0.1:${server.address().port}/training/pixel_harness.html`);
await page.waitForFunction('window.harnessReady === true');

const totals = configs.map(() => ({ perfect: 0, missed: 0, spurious: 0 }));
const BATCH = 25;
for (let done = 0; done < COUNT; done += BATCH) {
  const n = Math.min(BATCH, COUNT - done);
  const part = await page.evaluate(
    ([start, count, seedBase, cfgs]) => window.sweepMembers(start, count, seedBase, cfgs),
    [done, n, SEED_BASE, configs]
  );
  part.forEach((t, i) => {
    totals[i].perfect += t.perfect;
    totals[i].missed += t.missed;
    totals[i].spurious += t.spurious;
  });
  process.stdout.write(`\r  ${done + n}/${COUNT}   `);
}
await browser.close();
server.close();

console.log(`\n\nmember detection on ${COUNT} frames, nodes from the real detector\n`);
console.log(`${'half'.padStart(6)}${'onSeg'.padStart(7)}${'cov'.padStart(6)}${'perfect'.padStart(11)}${'missed'.padStart(9)}${'spurious'.padStart(10)}`);
console.log('-'.repeat(49));
const ranked = configs
  .map((cfg, i) => ({ cfg, ...totals[i] }))
  .sort((a, b) => b.perfect - a.perfect || (a.missed + a.spurious) - (b.missed + b.spurious));
for (const r of ranked) {
  console.log(
    `${String(r.cfg.halfWidth).padStart(6)}${String(r.cfg.onSegment).padStart(7)}` +
    `${r.cfg.coverage.toFixed(2).padStart(6)}${String(r.perfect).padStart(8)}/${COUNT}` +
    `${String(r.missed).padStart(9)}${String(r.spurious).padStart(10)}`
  );
}
const b = ranked[0];
console.log(`\nbest: halfWidth ${b.cfg.halfWidth}, onSegment ${b.cfg.onSegment}, ` +
            `coverage ${b.cfg.coverage} -> ${b.perfect}/${COUNT} frames exactly right`);
