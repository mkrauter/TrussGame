// Build a training corpus by driving the real v2 renderer inside headless
// Chromium, so the pixels the model learns from are the pixels players will
// generate.
//
//   node generate_corpus.mjs --split train --count 20000 --seed-base 0
//   node generate_corpus.mjs --split val   --count 2000  --seed-base 1000000
//
// Seed ranges must not overlap. The validation set is generated once from a
// fixed seed and never regenerated -- with a heavy-tailed target, redrawing it
// moves the metric independently of the model.

import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
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

const SPLIT = args.split ?? 'train';
const COUNT = Number(args.count ?? 64);
const SEED_BASE = Number(args['seed-base'] ?? (SPLIT === 'val' ? 1_000_000 : 0));
const BATCH = Number(args.batch ?? 200);
const OUT = path.resolve(args.out ?? path.join(HERE, 'corpus'), SPLIT);

const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript' };

function serve(root) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      const file = path.join(root, rel);
      if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404).end('not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
      fs.createReadStream(file).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

const t0 = Date.now();
await fsp.mkdir(path.join(OUT, 'images'), { recursive: true });

const server = await serve(REPO);
const port = server.address().port;
// --headed exists to prove headless rasterises identically to a real browser
// window; it is not a normal generation mode.
const browser = await chromium.launch({ headless: args.headed === undefined });
// Pin the device pixel ratio: on a HiDPI machine a default of 2 would silently
// change what the canvas rasterises into.
const page = await browser.newPage({ deviceScaleFactor: 1 });

page.on('pageerror', (e) => { throw e; });
const query = args.size ? `?size=${Number(args.size)}` : '';
await page.goto(`http://127.0.0.1:${port}/training/harness.html${query}`);
await page.waitForFunction('window.harnessReady === true');

const targets = [];
for (let done = 0; done < COUNT; done += BATCH) {
  const n = Math.min(BATCH, COUNT - done);
  const { targets: t, images } = await page.evaluate(
    ([start, count, seedBase]) => window.generateBatch(start, count, seedBase),
    [done, n, SEED_BASE]
  );

  await Promise.all(images.map((dataUrl, i) => {
    const png = Buffer.from(dataUrl.slice('data:image/png;base64,'.length), 'base64');
    return fsp.writeFile(path.join(OUT, 'images', `${String(done + i).padStart(6, '0')}.png`), png);
  }));
  targets.push(...t);

  const pct = ((done + n) / COUNT * 100).toFixed(0);
  const rate = (done + n) / ((Date.now() - t0) / 1000);
  process.stdout.write(`\r  ${done + n}/${COUNT} (${pct}%)  ${rate.toFixed(0)}/s   `);
}

await browser.close();
server.close();

const meta = {
  split: SPLIT,
  count: COUNT,
  seedBase: SEED_BASE,
  generatedAt: new Date().toISOString(),
  renderer: 'chromium/skia via playwright',
  note: 'images are the undeformed frame -- what the game shows when it asks for a prediction',
};
await fsp.writeFile(path.join(OUT, 'targets.json'), JSON.stringify({ meta, targets }, null, 1));

const bytes = (await Promise.all(
  targets.map((_, i) => fsp.stat(path.join(OUT, 'images', `${String(i).padStart(6, '0')}.png`)).then((s) => s.size))
)).reduce((a, b) => a + b, 0);

console.log(`\n\nwrote ${targets.length} samples to ${OUT}`);
console.log(`  ${(bytes / 1e6).toFixed(1)} MB of PNG (${(bytes / targets.length / 1024).toFixed(1)} KB each)`);
console.log(`  ${((Date.now() - t0) / 1000).toFixed(1)}s total`);
