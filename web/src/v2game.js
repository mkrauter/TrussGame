// v2 in the browser: you against the 2023 tflite model.
//
// A port of truss_game_AI.py. The model is the one that shipped, unchanged --
// its weights are lifted out of the flatbuffer by training/export_tflite.py and
// replayed by src/tflite.js, checked against LiteRT to 0.0002px.
//
// It is not good, and that is the point of keeping it. Measured through this
// port over 600 trusses it scores 60.5%, level with the 59.5% you get from
// "drop it straight down by the average distance" -- it moves the node 135px
// where the truth is 155px, and its predicted displacement has a negative
// R-squared against simply guessing the mean. It finds the node and drops it.
// v3 scores 96.3% on the same task.
//
// It was trained on pygame's hairlines and this page draws v3's thicker
// members. Measured either way the difference is a point or so, which 150
// trusses cannot resolve -- hence the 600 above.

import { WINDOW, PHYSICS, CROP, HUD, RENDER } from './config.js';
import { Truss, accuracy } from './truss.js';
import { drawScene, drawCross } from './renderer.js';

// truss_game_AI.py sampled with a 100px minimum spacing and offset (100, 150).
const TRUSS_OPTIONS = { offset: [100, 150], minDistance: 100 };
// Members are drawn at v3's thickness. truss_game_AI.py used pygame's 1px
// aaline, but hairlines look wrong beside v3 and the difference costs this
// model almost nothing -- measured below.
const STYLE = {};
const SETTLE = 500;
const ramp = (t) => 1 - Math.exp(-0.015 * t) * Math.cos(0.1 * t);

const canvas = document.getElementById('game');
canvas.width = WINDOW.width;
canvas.height = WINDOW.height;
const ctx = canvas.getContext('2d');

// The model reads a clean frame with only the truss on it, exactly as the
// original did -- it predicted after drawing the truss and before the HUD.
const view = document.createElement('canvas');
view.width = WINDOW.width;
view.height = WINDOW.height;
const viewCtx = view.getContext('2d', { willReadFrequently: true });

const worker = new Worker(new URL('./v2worker.js', import.meta.url), { type: 'module' });
let workerReady = false;
let pending = 0;

let truss = new Truss(Math.random, TRUSS_OPTIONS);
let prediction = null;
let thinking = false;
let thinkMs = 0;
let guess = null;
let time = 0;
let rounds = 0;
let scoreUser = 0;
let scoreAI = 0;
let draws = 0;
let averageUser = 0;
let averageAI = 0;
let shown = null;

worker.onmessage = (event) => {
  if (event.data.type === 'ready') {
    workerReady = true;
    think();
    return;
  }
  if (event.data.type === 'prediction') {
    // Ignore an answer to a truss that is no longer on the board.
    if (event.data.id !== pending) return;
    prediction = event.data.prediction;
    thinkMs = event.data.ms;
    thinking = false;
  }
};
worker.postMessage({ type: 'load' });

function think() {
  if (!workerReady) return;
  thinking = true;
  pending += 1;

  drawScene(viewCtx, truss, STYLE);
  const { data } = viewCtx.getImageData(CROP.x, CROP.y, CROP.width, CROP.height);
  // RGBA to RGB, raw 0-255: the original passed pygame's surface straight in
  // with no normalisation, so the weights expect that range.
  const rgb = new Float32Array(CROP.width * CROP.height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i];
    rgb[j + 1] = data[i + 1];
    rgb[j + 2] = data[i + 2];
  }
  worker.postMessage({ type: 'predict', id: pending, frame: rgb }, [rgb.buffer]);
}

function nextTruss() {
  truss = new Truss(Math.random, TRUSS_OPTIONS);
  guess = null;
  prediction = null;
  shown = null;
  think();
}

canvas.addEventListener('click', (event) => {
  const rect = canvas.getBoundingClientRect();
  const point = [
    ((event.clientX - rect.left) / rect.width) * WINDOW.width,
    ((event.clientY - rect.top) / rect.height) * WINDOW.height,
  ];

  if (guess === null) {
    if (thinking) return;        // the AI has not answered yet; let it finish
    guess = point;
    time = 0;
    return;
  }

  // Score against the settled state at exactly `force`, never the animating
  // frame -- clicking early would otherwise be scored against an overshoot.
  truss.calculate(PHYSICS.force);
  const user = accuracy(truss.loadedStart, truss.loadedEnd, guess);
  const ai = prediction ? accuracy(truss.loadedStart, truss.loadedEnd, prediction) : 0;

  if (user > ai) scoreUser += 1;
  else if (ai > user) scoreAI += 1;
  // Accuracy clamps at zero, so both missing by more than the travel distance
  // is a real 0-0 draw rather than a win for the AI.
  else draws += 1;

  rounds += 1;
  averageUser += (user - averageUser) / rounds;
  averageAI += (ai - averageAI) / rounds;
  nextTruss();
});

// Two rows above y = 80, where the truss begins. A third row put the running
// averages down among the members -- see the same fix in versus.js.
const ROW_LABEL = 30;
const ROW_VALUE = 68;
const BAND = 80;

function drawHud() {
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'center';

  // Repainted before the text so the truss passes underneath it, not through it.
  ctx.fillStyle = RENDER.background;
  ctx.fillRect(0, 0, WINDOW.width, BAND);

  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  const left = rounds > 0 ? `You  ·  ${averageUser.toFixed(1)}% avg` : 'You';
  const right = rounds > 0 ? `${averageAI.toFixed(1)}% avg  ·  AI` : 'AI';
  ctx.fillText(left, 150, ROW_LABEL);
  ctx.fillText(right, WINDOW.width - 150, ROW_LABEL);

  const big = (l, r) => {
    ctx.font = HUD.scoreFont;
    ctx.fillStyle = HUD.color;
    ctx.fillText(l, 150, ROW_VALUE);
    ctx.fillText(r, WINDOW.width - 150, ROW_VALUE);
  };

  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  if (guess !== null && shown) {
    ctx.fillText('Accuracy', WINDOW.width / 2, ROW_LABEL);
    big(`${shown.user.toFixed(0)}%`, `${shown.ai.toFixed(0)}%`);
    const centre = WINDOW.width / 2;
    const top = 52;
    const length = Math.min(Math.abs(shown.user - shown.ai) * 2.2, 200);
    ctx.fillStyle = shown.user >= shown.ai ? '#5cc46a' : '#d86a6a';
    ctx.fillRect(shown.user >= shown.ai ? centre - length : centre, top, length, 7);
    ctx.fillStyle = HUD.dimColor;
    ctx.fillRect(centre - 1, top - 7, 2, 21);
  } else {
    ctx.fillText('Score', WINDOW.width / 2, ROW_LABEL);
    big(`${scoreUser}`, `${scoreAI}`);
    if (draws > 0) {
      ctx.font = HUD.labelFont;
      ctx.fillStyle = HUD.dimColor;
      ctx.fillText(`${draws} draw${draws === 1 ? '' : 's'}`, WINDOW.width / 2, ROW_VALUE);
    }
  }

  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  ctx.textAlign = 'center';
  ctx.fillText(
    thinking
      ? 'AI is thinking...'
      : guess === null
        ? 'Click where you think the blue node will move'
        : 'Click anywhere to score and continue',
    WINDOW.width / 2,
    HUD.bottomBaseline
  );

  ctx.font = HUD.hintFont;
  ctx.fillStyle = HUD.hintColor;
  ctx.textAlign = 'left';
  ctx.fillText(
    thinkMs > 0 ? `2023 tflite model · ${(thinkMs / 1000).toFixed(1)}s per move` : '2023 tflite model',
    HUD.margin,
    HUD.bottomBaseline
  );
}

function frame() {
  if (guess !== null) {
    if (time < SETTLE) {
      truss.calculate(PHYSICS.force * ramp(time));
      time += 1;
    }
    shown = {
      user: accuracy(truss.loadedStart, truss.loadedEnd, guess),
      ai: prediction ? accuracy(truss.loadedStart, truss.loadedEnd, prediction) : 0,
    };
  }

  drawScene(ctx, truss, STYLE);
  if (guess !== null) {
    drawCross(ctx, guess, 'white');
    if (prediction) drawCross(ctx, prediction, 'yellow');
  }
  drawHud();
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
