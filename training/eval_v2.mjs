// Score the v2 model through the browser port.
//
//   node eval_v2.mjs --count 200
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((a, x, i, arr) => {
  if (x.startsWith('--')) a.push([x.slice(2), arr[i + 1]]);
  return a;
}, []));
const COUNT = Number(args.count ?? 100);
const SEED_BASE = Number(args['seed-base'] ?? 1_000_000);

const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript', '.json': 'application/json', '.png': 'image/png' };
const server = http.createServer((req, res) => {
  const file = path.join(REPO, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(REPO) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return res.writeHead(404).end('no');
  res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 1 });
page.on('pageerror', (e) => { throw e; });
await page.goto(`http://127.0.0.1:${server.address().port}/training/v2_harness.html`);
await page.waitForFunction('window.harnessReady === true');

const rows = [];
for (let done = 0; done < COUNT; done += 10) {
  const n = Math.min(10, COUNT - done);
  rows.push(...await page.evaluate(([s, c, b]) => window.evaluateRange(s, c, b), [done, n, SEED_BASE]));
  process.stdout.write(`\r  ${done + n}/${COUNT}   `);
}
await browser.close();
server.close();

const scores = rows.map((r) => r.score).sort((a, b) => a - b);
const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
const avg = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
console.log(`\n\nv2 tflite model through the browser port, ${rows.length} trusses\n`);
console.log(`  mean score        ${mean.toFixed(2)}%`);
console.log(`  median            ${scores[Math.floor(scores.length / 2)].toFixed(2)}%`);
console.log(`  zero rounds       ${scores.filter((s) => s === 0).length}`);
console.log(`  mean predicted move ${avg((r) => r.fromStart).toFixed(1)} px vs true ${avg((r) => r.travelled).toFixed(1)} px`);
console.log(`  ${avg((r) => r.ms).toFixed(0)} ms per move`);
