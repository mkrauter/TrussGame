import { WINDOW, PHYSICS, ANIMATION } from './config.js';
import { Truss, accuracy } from './truss.js';
import { drawScene, drawCross, drawHud } from './renderer.js';

// Human-only game loop for now. The AI slot -- capture the CROP region, run
// the model, draw a second cross -- goes in where noted below.

const canvas = document.getElementById('game');
canvas.width = WINDOW.width;
canvas.height = WINDOW.height;
const ctx = canvas.getContext('2d');

let truss = new Truss();
let guess = null;
let time = 0;
let rounds = 0;
let averageAccuracy = 0;
// Live while the load ramps, then frozen at the settled score once the round is
// scored, so the last result stays on screen during the next setup.
let shownAccuracy = null;

canvas.addEventListener('click', (event) => {
  const rect = canvas.getBoundingClientRect();
  const point = [
    ((event.clientX - rect.left) / rect.width) * WINDOW.width,
    ((event.clientY - rect.top) / rect.height) * WINDOW.height,
  ];

  if (guess === null) {
    guess = point;
    time = 0;
  } else {
    // Score against the settled state, never whatever frame happened to be on
    // screen -- v1 read accuracy mid-overshoot if you clicked early.
    truss.calculate(PHYSICS.force);
    const score = accuracy(truss.loadedStart, truss.loadedEnd, guess);
    rounds += 1;
    averageAccuracy += (score - averageAccuracy) / rounds;
    shownAccuracy = score;
    truss = new Truss();
    guess = null;
  }
});

function frame() {
  if (guess !== null) {
    if (time < ANIMATION.settleTime) {
      truss.calculate(PHYSICS.force * ANIMATION.ramp(time));
      time += 1;
    }
    // Recomputed every frame against the current deformed position, so the
    // number tracks the oscillation as it decays rather than only appearing
    // once the round ends.
    shownAccuracy = accuracy(truss.loadedStart, truss.loadedEnd, guess);
  }

  drawScene(ctx, truss);
  if (guess !== null) drawCross(ctx, guess, 'white');

  drawHud(ctx, {
    accuracy: shownAccuracy,
    rounds,
    average: averageAccuracy,
    prompt: guess === null
      ? 'Click where you think the blue node will move'
      : 'Click anywhere to score and continue',
  });

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
