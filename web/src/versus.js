// You against the AI, the browser version of truss_game_AI.py.
//
// The AI's only input is a colour screenshot of the scene -- the same thing you
// are looking at. It never receives node coordinates or the topology. A small
// convolutional net finds the nodes and their roles in the image, connectivity
// is read by checking which pairs have a line drawn between them, and a graph
// network does the mechanics on whatever structure came out of that.
//
// Splitting perception from physics is not a convenience. Convolutions are the
// right tool for finding marks in an image and the wrong tool for a global
// implicit solve: an end-to-end CNN on the same screenshots scored 77%, and
// widening its receptive field to cover the support span made it worse, not
// better. This pipeline scores ~96%.

import { WINDOW, PHYSICS, ANIMATION, HUD, SCORING_HINT, RENDER } from './config.js';
import { Truss, accuracy } from './truss.js';
import { drawScene, drawCross } from './renderer.js';
import { TrussDetector, readTruss } from './detect.js';
import { TrussGNN } from './gnn.js';
import { predictClick } from './ai.js';

const USER_COLOR = 'white';
const AI_COLOR = 'gold';

const canvas = document.getElementById('game');
canvas.width = WINDOW.width;
canvas.height = WINDOW.height;
const ctx = canvas.getContext('2d');

// The AI's view: a clean render of the scene, with no HUD and no guess markers.
// Detecting from the visible canvas would let whatever happened to be painted
// this frame into the model's input.
const view = document.createElement('canvas');
view.width = WINDOW.width;
view.height = WINDOW.height;
const viewCtx = view.getContext('2d', { willReadFrequently: true });

const [detector, gnn] = await Promise.all([
  TrussDetector.load(new URL('./model/trussdetector.json', import.meta.url)),
  TrussGNN.load(new URL('./model/trussgnn.json', import.meta.url)),
]);

// Fewer message-passing rounds is a less-converged solver, so difficulty is a
// physically meaningful dial rather than injected noise.
const LEVELS = [1, 2, 3, 4, 6, 8, 10];
let levelIndex = LEVELS.length - 1;

let truss = new Truss();
let perceived = null;
let prediction = null;
let thinking = false;
let guess = null;
let time = 0;
let rounds = 0;
let scoreUser = 0;
let scoreAI = 0;
let draws = 0;
let averageUser = 0;
let averageAI = 0;
let shown = null;
let showVision = false;

function think() {
  thinking = true;
  // Deferred a frame so the browser can paint "looking..." first: reading the
  // frame and solving costs about half a second and blocks the main thread.
  setTimeout(() => {
    drawScene(viewCtx, truss);
    const seen = readTruss(detector, view);
    const ok = seen.loadedNode >= 0 && seen.supports.length === 2;
    perceived = seen;
    prediction = ok ? predictClick(gnn, seen, { sigma: 0, rounds: LEVELS[levelIndex] }) : null;
    if (prediction && !(Number.isFinite(prediction[0]) && Number.isFinite(prediction[1]))) {
      prediction = null;
    }
    thinking = false;
  }, 0);
}

function nextTruss() {
  truss = new Truss();
  guess = null;
  prediction = null;
  perceived = null;
  shown = null;
  think();
}

think();

canvas.addEventListener('click', (event) => {
  if (thinking) return;             // do not let a click land mid-think
  const rect = canvas.getBoundingClientRect();
  const point = [
    ((event.clientX - rect.left) / rect.width) * WINDOW.width,
    ((event.clientY - rect.top) / rect.height) * WINDOW.height,
  ];

  if (guess === null) {
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

window.addEventListener('keydown', (event) => {
  if (event.key === 'v' || event.key === 'V') {
    showVision = !showVision;
    return;
  }
  const step = event.key === ']' ? 1 : event.key === '[' ? -1 : 0;
  if (step === 0) return;
  levelIndex = Math.min(LEVELS.length - 1, Math.max(0, levelIndex + step));
  // Only re-think while the round is still open; changing the AI's strength
  // after it has committed would be rewriting its answer.
  if (guess === null && !thinking) think();
});

// What the AI actually saw: the nodes it found and the members it read. Worth
// being able to look at -- it is the whole basis of the fairness claim.
function drawVision() {
  if (!perceived) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 170, 60, 0.55)';
  ctx.lineWidth = 1;
  for (const [i, j] of perceived.elements) {
    ctx.beginPath();
    ctx.moveTo(perceived.nodes[i][0], perceived.nodes[i][1]);
    ctx.lineTo(perceived.nodes[j][0], perceived.nodes[j][1]);
    ctx.stroke();
  }
  for (const p of perceived.nodes) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawLabels() {
  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  ctx.textAlign = 'center';
  ctx.fillText('You', 150, HUD.topBaseline);
  ctx.fillText('AI', WINDOW.width - 150, HUD.topBaseline);
}

function drawBigPair(left, right) {
  ctx.font = HUD.scoreFont;
  ctx.fillStyle = HUD.color;
  ctx.textAlign = 'center';
  ctx.fillText(left, 150, HUD.topBaseline + 52);
  ctx.fillText(right, WINDOW.width - 150, HUD.topBaseline + 52);
}

// A bar leaning toward whoever is ahead, length proportional to the gap.
function drawLeadBar(user, ai) {
  const centre = WINDOW.width / 2;
  const top = HUD.topBaseline + 40;
  const length = Math.min(Math.abs(user - ai) * 2.2, 200);
  ctx.fillStyle = user >= ai ? '#5cc46a' : '#d86a6a';
  ctx.fillRect(user >= ai ? centre - length : centre, top, length, 7);
  ctx.fillStyle = HUD.dimColor;
  ctx.fillRect(centre - 1, top - 7, 2, 21);
}

function drawHud() {
  ctx.textBaseline = 'alphabetic';
  drawLabels();

  if (guess !== null && shown) {
    ctx.font = HUD.labelFont;
    ctx.fillStyle = HUD.dimColor;
    ctx.textAlign = 'center';
    ctx.fillText('Accuracy', WINDOW.width / 2, HUD.topBaseline);
    drawBigPair(`${shown.user.toFixed(0)}%`, `${shown.ai.toFixed(0)}%`);
    drawLeadBar(shown.user, shown.ai);
  } else {
    ctx.font = HUD.labelFont;
    ctx.fillStyle = HUD.dimColor;
    ctx.textAlign = 'center';
    ctx.fillText('Score', WINDOW.width / 2, HUD.topBaseline);
    drawBigPair(`${scoreUser}`, `${scoreAI}`);
    if (rounds > 0) {
      ctx.font = HUD.labelFont;
      ctx.fillStyle = HUD.dimColor;
      ctx.fillText(`${averageUser.toFixed(1)}% avg`, 150, HUD.topBaseline + 80);
      ctx.fillText(`${averageAI.toFixed(1)}% avg`, WINDOW.width - 150, HUD.topBaseline + 80);
    }
    if (draws > 0) {
      ctx.font = HUD.labelFont;
      ctx.fillStyle = HUD.dimColor;
      ctx.fillText(`${draws} draw${draws === 1 ? '' : 's'}`, WINDOW.width / 2, HUD.topBaseline + 52);
    }
  }

  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  ctx.textAlign = 'center';
  ctx.fillText(
    thinking
      ? 'AI is looking at the screen...'
      : guess === null
        ? 'Click where you think the blue node will move'
        : 'Click anywhere to score and continue',
    WINDOW.width / 2,
    HUD.bottomBaseline
  );

  ctx.font = HUD.hintFont;
  ctx.fillStyle = HUD.hintColor;
  ctx.fillText(SCORING_HINT, WINDOW.width / 2, HUD.hintBaseline);

  // On the prompt's line, not the hint's: the hint is centred and full width,
  // so a left-aligned line beneath it collides with its first characters.
  ctx.textAlign = 'left';
  ctx.fillText(
    `AI: ${LEVELS[levelIndex]} round${LEVELS[levelIndex] === 1 ? '' : 's'}` +
    `   [/] strength   V vision`,
    HUD.margin,
    HUD.bottomBaseline
  );
}

function frame() {
  if (guess !== null) {
    if (time < ANIMATION.settleTime) {
      truss.calculate(PHYSICS.force * ANIMATION.ramp(time));
      time += 1;
    }
    shown = {
      user: accuracy(truss.loadedStart, truss.loadedEnd, guess),
      ai: prediction ? accuracy(truss.loadedStart, truss.loadedEnd, prediction) : 0,
    };
  }

  drawScene(ctx, truss);
  if (showVision) drawVision();
  if (guess !== null) {
    drawCross(ctx, guess, USER_COLOR);
    if (prediction) drawCross(ctx, prediction, AI_COLOR);
  }
  drawHud();
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
