// Drive pixel_harness.html in headless Chromium and report the end-to-end
// score of the screenshot-only pipeline.
//
//   node eval_pixel_pipeline.mjs --count 500
//
// This is the number that matters: the AI's only input is the rendered canvas,
// and it is scored on the game's own metric against the true settled position.
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
const COUNT = Number(args.count ?? 200);
const SEED_BASE = Number(args['seed-base'] ?? 1_000_000);
const BATCH = Number(args.batch ?? 25);
const ROUNDS = args.rounds === undefined ? undefined : Number(args.rounds);

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
const port = server.address().port;

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
page.on('pageerror', (e) => { throw e; });
await page.goto(`http://127.0.0.1:${port}/training/pixel_harness.html`);
await page.waitForFunction('window.harnessReady === true');

const rows = [];
const t0 = Date.now();
for (let done = 0; done < COUNT; done += BATCH) {
  const n = Math.min(BATCH, COUNT - done);
  rows.push(...await page.evaluate(
    ([start, count, seedBase, rounds]) => window.evaluateRange(start, count, seedBase, rounds),
    [done, n, SEED_BASE, ROUNDS]
  ));
  process.stdout.write(`\r  ${done + n}/${COUNT}   `);
}
await browser.close();
server.close();

const scores = rows.map((r) => r.score).sort((a, b) => a - b);
const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
const pick = (q) => scores[Math.min(scores.length - 1, Math.floor(q * scores.length))];
const avg = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
const wrongMembers = rows.filter((r) => r.members !== r.trueMembers).length;

console.log(`\n\nscreenshot-only pipeline, ${rows.length} trusses (seeds from ${SEED_BASE})\n`);
console.log(`  mean score          ${mean.toFixed(2)}%`);
console.log(`  median              ${pick(0.5).toFixed(2)}%`);
console.log(`  25th percentile     ${pick(0.25).toFixed(2)}%`);
console.log(`  5th percentile      ${pick(0.05).toFixed(2)}%`);
console.log(`  failed to read      ${rows.filter((r) => !r.ok).length}`);
console.log(`\n  mean node error     ${avg((r) => r.nodeError).toFixed(2)} px`);
console.log(`  worst node error    ${Math.max(...rows.map((r) => r.worstNode)).toFixed(1)} px`);
console.log(`  wrong member count  ${wrongMembers}/${rows.length} frames`);
console.log(`  detect time         ${avg((r) => r.detectMs).toFixed(0)} ms per move`);
console.log(`\n  ${((Date.now() - t0) / 1000).toFixed(0)}s total`);
