// Run the JS replay of the v2 model on real frames and print its predictions,
// so verify_tflite.py can hold them against LiteRT.
//
// The frames are the 768px PNGs in corpus768/, which are exactly the crop the
// model was built to take. Using files on disk rather than shipping pixel
// arrays out of the page keeps both sides reading identical bytes -- and a
// 768x768x3 frame is 1.7M numbers, which is not something to send as JSON.
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
const COUNT = Number(args.count ?? 8);

const MIME = { '.html': 'text/html', '.js': 'application/javascript',
               '.mjs': 'application/javascript', '.json': 'application/json',
               '.png': 'image/png' };
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
await page.goto(`http://127.0.0.1:${port}/training/v2_probe.html`);
await page.waitForFunction('window.probeReady === true');

const out = [];
for (let i = 0; i < COUNT; i++) {
  const name = String(i).padStart(6, '0');
  const prediction = await page.evaluate(
    (url) => window.predictFromPng(url),
    `/training/corpus768/val/images/${name}.png`
  );
  out.push({ seed: 1_000_000 + i, image: name, prediction });
}
await browser.close();
server.close();
process.stdout.write(JSON.stringify(out));
