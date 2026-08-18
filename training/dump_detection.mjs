// Report what the browser's own detector computes, as JSON on stdout, so
// verify_detector.py can hold it against PyTorch. Nothing is asserted here --
// this half only observes.
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
const COUNT = Number(args.count ?? 16);
const SEED_BASE = Number(args['seed-base'] ?? 1_000_000);

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

const out = [];
for (let i = 0; i < COUNT; i++) {
  out.push(await page.evaluate((seed) => window.dumpDetection(seed), SEED_BASE + i));
}
await browser.close();
server.close();
process.stdout.write(JSON.stringify(out));
