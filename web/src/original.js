// The original game, in the browser.
//
// A port of truss_game_original.py, which is frozen as a historic reference.
// The point of this page is that it plays the way that file plays, so its
// quirks are reproduced rather than corrected:
//
//   * nodes are sampled uniformly with no minimum spacing, so they sometimes
//     sit almost on top of each other and the truss comes out degenerate
//   * members are hairlines, and their stress colours are computed from the
//     *deformed* geometry, which is not what linear theory says but is what
//     this version drew
//   * the load never settles -- it oscillates about `force` for as long as you
//     leave it, and the accuracy readout chases it
//   * there is no running score and no opponent; that came later
//
// The corrected physics and the AI opponent live in v2 and v3.

import { WINDOW, PHYSICS } from './config.js';
import { Truss } from './truss.js';
import { drawScene } from './renderer.js';

const TRUSS_OPTIONS = {
  // The original sampled `[700,500] * rand + 100` with no rejection sampling.
  offset: [100, 100],
  minDistance: 0,
  stressFrom: 'deformed',
};
// Thickness matches v2 and v3 so the three look like one game; the original's
// own 1px aalines are the one quirk not reproduced. The stress scale is its
// own, which is what actually makes its colours look different.
const STYLE = { stressScale: 3 };

// f = force - force * exp(-0.015 t) * cos(0.1 t), and nothing ever stops it.
const ramp = (t) => 1 - Math.exp(-0.015 * t) * Math.cos(0.1 * t);

const canvas = document.getElementById('game');
canvas.width = WINDOW.width;
canvas.height = WINDOW.height;
const ctx = canvas.getContext('2d');

let truss = new Truss(Math.random, TRUSS_OPTIONS);
let mouse = null;
let time = 0;

canvas.addEventListener('click', (event) => {
  const rect = canvas.getBoundingClientRect();
  const point = [
    ((event.clientX - rect.left) / rect.width) * WINDOW.width,
    ((event.clientY - rect.top) / rect.height) * WINDOW.height,
  ];
  if (mouse === null) {
    mouse = point;
    time = 0;
  } else {
    truss = new Truss(Math.random, TRUSS_OPTIONS);
    mouse = null;
  }
});

function drawCross(p) {
  ctx.strokeStyle = 'rgb(128, 128, 128)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(p[0] - 20, p[1]);
  ctx.lineTo(p[0] + 20, p[1]);
  ctx.moveTo(p[0], p[1] - 20);
  ctx.lineTo(p[0], p[1] + 20);
  ctx.stroke();
}

function frame() {
  if (mouse !== null) {
    truss.calculate(PHYSICS.force * ramp(time));
    time += 1;
  }

  drawScene(ctx, truss, mouse === null ? { ...STYLE, plainMembers: true } : STYLE);

  ctx.fillStyle = 'rgb(255, 255, 255)';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';

  if (mouse === null) {
    ctx.font = '24px "Segoe UI", system-ui, sans-serif';
    ctx.fillText('Click where you think the blue node will move!', 200, 800);
  } else {
    drawCross(mouse);

    // The original divides by max(travel, 1) and clamps at zero, so a node that
    // has barely moved yet reads as a near miss rather than a divide by zero.
    const start = truss.loadedStart;
    const end = truss.loadedEnd;
    const travelled = Math.max(Math.hypot(end[0] - start[0], end[1] - start[1]), 1);
    const missed = Math.hypot(mouse[0] - end[0], mouse[1] - end[1]);
    const score = Math.max(100 - (100 * missed) / travelled, 0);

    ctx.font = '24px "Segoe UI", system-ui, sans-serif';
    ctx.fillText('accuracy:', 750, 30);
    ctx.font = '72px "Segoe UI", system-ui, sans-serif';
    ctx.fillText(`${Math.floor(score)}%`, 710, 50);
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
