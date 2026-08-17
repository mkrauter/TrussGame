import { WINDOW, PHYSICS, ANIMATION } from './config.js';
import { Truss, accuracy } from './truss.js';
import { drawScene, drawCross } from './renderer.js';

// Human-only game loop for now. The AI slot -- capture the CROP region, run
// the model, draw a second cross -- goes in where noted below.

const canvas = document.getElementById('game');
canvas.width = WINDOW.width;
canvas.height = WINDOW.height;
const ctx = canvas.getContext('2d');

const hud = {
  userScore: document.getElementById('user-score'),
  userAvg: document.getElementById('user-average'),
  prompt: document.getElementById('prompt'),
};

let truss = new Truss();
let guess = null;
let time = 0;
let rounds = 0;
let averageAccuracy = 0;

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
    hud.userScore.textContent = `${score.toFixed(0)}%`;
    hud.userAvg.textContent = `${rounds} round${rounds === 1 ? '' : 's'}, average ${averageAccuracy.toFixed(1)}%`;
    truss = new Truss();
    guess = null;
  }
});

function frame() {
  if (guess !== null && time < ANIMATION.settleTime) {
    truss.calculate(PHYSICS.force * ANIMATION.ramp(time));
    time += 1;
  }

  drawScene(ctx, truss);

  if (guess !== null) {
    drawCross(ctx, guess, 'white');
    hud.prompt.textContent = 'Click anywhere to score and continue';
  } else {
    hud.prompt.textContent = 'Click where you think the blue node will move';
  }

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
