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
// physically meaningful dial rather than injected noise: the opponent is not
// handicapped, it has simply thought about the structure for less time.
//
// The scores are measured, not guessed -- `node training/eval_pixel_pipeline.mjs
// --rounds N` over the validation seeds. Human players sit around 70-80%, which
// is why Medium is the default: it is the level that makes a real contest.
const LEVELS = [
  { name: 'Easy', rounds: 4, score: 46 },
  { name: 'Medium', rounds: 6, score: 68 },
  { name: 'Hard', rounds: 8, score: 86 },
  { name: 'Expert', rounds: 10, score: 96 },
];
let levelIndex = 1;

// Filled in while drawing the HUD so a click can be tested against the labels;
// the canvas takes every click, so the board has to know what is not a guess.
let levelHitboxes = [];

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
    prediction = ok
      ? predictClick(gnn, seen, { sigma: 0, rounds: LEVELS[levelIndex].rounds })
      : null;
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
  const rect = canvas.getBoundingClientRect();
  const point = [
    ((event.clientX - rect.left) / rect.width) * WINDOW.width,
    ((event.clientY - rect.top) / rect.height) * WINDOW.height,
  ];

  // The difficulty labels live on the same canvas as the board, so they get
  // first refusal on a click. Without this, choosing a level would also be
  // read as a guess at the bottom of the screen.
  const picked = levelHitboxes.findIndex(
    (b) => point[0] >= b.x && point[0] <= b.x + b.width &&
           point[1] >= b.y && point[1] <= b.y + b.height
  );
  if (picked >= 0) {
    setLevel(picked);
    return;
  }

  if (thinking) return;             // do not let a click land mid-think

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

function setLevel(index) {
  if (index === levelIndex) return;
  levelIndex = index;
  // Only re-think while the round is still open; changing the AI's strength
  // after it has committed would be rewriting an answer it already gave. Mid
  // round the choice simply applies from the next truss.
  if (guess === null && !thinking) think();
}

window.addEventListener('keydown', (event) => {
  if (event.key === 'v' || event.key === 'V') {
    showVision = !showVision;
    return;
  }
  // 1-4 pick a level directly; the brackets still step through them.
  const digit = Number(event.key);
  if (digit >= 1 && digit <= LEVELS.length) {
    setLevel(digit - 1);
    return;
  }
  const step = event.key === ']' ? 1 : event.key === '[' ? -1 : 0;
  if (step === 0) return;
  setLevel(Math.min(LEVELS.length - 1, Math.max(0, levelIndex + step)));
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

// The truss occupies y >= 100 and the loaded node's marker points 20px up from
// its node, so anything drawn below y = 80 can end up tangled in the structure.
// Two rows fit above that line and three do not, which is what put the running
// averages on top of the truss; they now share the label row.
const ROW_LABEL = 30;
const ROW_VALUE = 68;
const BAND = 80;

// Repainted before the text so the truss passes cleanly underneath rather than
// through the glyphs. It hides nothing -- no member reaches this high.
function drawBand() {
  ctx.fillStyle = RENDER.background;
  ctx.fillRect(0, 0, WINDOW.width, BAND);
}

function drawLabels(leftNote, rightNote) {
  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  ctx.textAlign = 'center';
  ctx.fillText(leftNote ? `You  ·  ${leftNote}` : 'You', 150, ROW_LABEL);
  ctx.fillText(rightNote ? `${rightNote}  ·  AI` : 'AI', WINDOW.width - 150, ROW_LABEL);
}

function drawBigPair(left, right) {
  ctx.font = HUD.scoreFont;
  ctx.fillStyle = HUD.color;
  ctx.textAlign = 'center';
  ctx.fillText(left, 150, ROW_VALUE);
  ctx.fillText(right, WINDOW.width - 150, ROW_VALUE);
}

// A bar leaning toward whoever is ahead, length proportional to the gap.
function drawLeadBar(user, ai) {
  const centre = WINDOW.width / 2;
  const top = 52;
  const length = Math.min(Math.abs(user - ai) * 2.2, 200);
  ctx.fillStyle = user >= ai ? '#5cc46a' : '#d86a6a';
  ctx.fillRect(user >= ai ? centre - length : centre, top, length, 7);
  ctx.fillStyle = HUD.dimColor;
  ctx.fillRect(centre - 1, top - 7, 2, 21);
}

function drawHud() {
  ctx.textBaseline = 'alphabetic';
  drawBand();

  const notes = rounds > 0
    ? [`${averageUser.toFixed(1)}% avg`, `${averageAI.toFixed(1)}% avg`]
    : [null, null];
  drawLabels(notes[0], notes[1]);

  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  ctx.textAlign = 'center';

  if (guess !== null && shown) {
    ctx.fillText('Accuracy', WINDOW.width / 2, ROW_LABEL);
    drawBigPair(`${shown.user.toFixed(0)}%`, `${shown.ai.toFixed(0)}%`);
    drawLeadBar(shown.user, shown.ai);
  } else {
    ctx.fillText('Score', WINDOW.width / 2, ROW_LABEL);
    drawBigPair(`${scoreUser}`, `${scoreAI}`);
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

  drawLevels();

  // Right-aligned so it balances the level picker and clears the centred
  // prompt between them.
  ctx.font = HUD.hintFont;
  ctx.fillStyle = HUD.hintColor;
  ctx.textAlign = 'right';
  ctx.fillText('V  see what the AI saw', WINDOW.width - HUD.margin, HUD.bottomBaseline);
}

// The difficulty picker: four names on the prompt's line, the active one lit.
// Drawing it here rather than as HTML keeps the whole game one element, which
// is what lets the page scale to any window without a layout.
function drawLevels() {
  const y = HUD.bottomBaseline;
  levelHitboxes = [];

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';

  // Names only on this line. The centred prompt starts around x=277, and
  // anything more here runs straight into it.
  let x = HUD.margin;
  LEVELS.forEach((level, i) => {
    const active = i === levelIndex;
    ctx.font = active ? `600 ${HUD.hintFont}` : HUD.hintFont;
    const width = ctx.measureText(level.name).width;

    if (active) {
      // A quiet underline rather than a button: this sits in the margin and
      // should not compete with the board for attention.
      ctx.fillStyle = HUD.color;
      ctx.fillText(level.name, x, y);
      ctx.fillRect(x, y + 4, width, 1.5);
    } else {
      ctx.fillStyle = HUD.hintColor;
      ctx.fillText(level.name, x, y);
    }

    // Padded well past the glyphs: these are click targets, and the canvas is
    // scaled down by CSS on smaller screens, so a box that feels generous here
    // is only two thirds the size on a 600px window.
    levelHitboxes.push({ x: x - 9, y: y - 19, width: width + 18, height: 31 });
    x += width + 18;
  });

  // What the chosen level actually scores, on the line below where there is
  // room. Measured, so it doubles as a statement about the opponent rather
  // than a difficulty label invented to feel fair.
  ctx.font = HUD.hintFont;
  ctx.fillStyle = HUD.hintColor;
  ctx.fillText(`${LEVELS[levelIndex].name} scores ${LEVELS[levelIndex].score}%`,
               HUD.margin, HUD.hintBaseline);
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
