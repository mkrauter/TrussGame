// Single source of truth for every constant that must stay identical between
// the playable game and the headless training-corpus generator. Anything that
// changes the pixels the model sees belongs in here, not inline at the call
// site -- train/serve skew in v1 came from exactly that kind of drift.

export const WINDOW = { width: 900, height: 900 };

// The region handed to the model. Spans 68..836 on both axes.
export const CROP = { x: 68, y: 68, width: 768, height: 768 };

export const TRUSS = {
  numNodes: 10,
  size: [700, 500],
  // v1 disagreed with itself: offset=100 in the committed game, (100,150) in
  // the working tree. The shift down existed to stop the truss colliding with
  // pygame's on-canvas HUD text at y=130, which sits inside the model crop.
  // v2 draws the HUD in HTML outside the canvas, so we take the original value:
  // measured over 5000 rounds it drops settled targets landing outside the
  // 768px crop from 2.9% to 1.0%.
  offset: [100, 100],
  minDistance: 100,
  // Rejection sampling can in principle never terminate. Fail loudly instead.
  maxSampleAttempts: 10000,
};

export const PHYSICS = {
  E: 200,
  A: 1000,
  force: 1e5,
};

// The load ramp is cosmetic: it animates from 0 up to `force`, settling by
// SETTLE_TIME. The prediction target is always the settled state, so the
// corpus generator ignores this entirely and does one static solve.
export const ANIMATION = {
  settleTime: 500,
  ramp: (t) => 1 - Math.exp(-0.015 * t) * Math.cos(0.1 * t),
};

export const RENDER = {
  background: '#404040', // pygame's grey25
  supportColor: 'rgb(64, 128, 64)',
  loadedColor: 'rgb(100, 100, 200)',
  markerSize: [10, 20],
  // v1 used aaline, which is 1px only. CLAUDE.md wants 2-3px so the lines
  // survive the 768->256 bilinear downsample against grey25.
  lineWidth: 2.5,
  // v1 had three different values for this (20 in the game, 5 in the working
  // tree, 3 in the notebook). One place now.
  stressScale: 5,
};

// The HUD is drawn onto the canvas so the whole game is one square element that
// fits a 1080px screen. It is deliberately confined to the bands above and
// below the CROP region (which spans 68..836), so overlay text can never end up
// in a training image. drawScene() -- the only thing the corpus generator calls
// -- does not draw any of this.
export const HUD = {
  topBaseline: 44,
  // Two stacked lines in the bottom band: the prompt, then the scoring hint
  // beneath it. The band is only 64px tall (836..900), so these are tuned to
  // sit inside it -- re-check the headroom if the fonts change.
  bottomBaseline: 862,
  hintBaseline: 886,
  margin: 26,
  scoreFont: '600 34px "Segoe UI", system-ui, sans-serif',
  labelFont: '17px "Segoe UI", system-ui, sans-serif',
  hintFont: '13px "Segoe UI", system-ui, sans-serif',
  color: '#f0f0f0',
  dimColor: 'rgba(240, 240, 240, 0.65)',
  hintColor: 'rgba(240, 240, 240, 0.42)',
};

// What the percentage actually means. The metric is relative -- the error is
// divided by how far the node travelled -- which is the part nobody guesses,
// so it is worth saying on screen rather than only in the README.
export const SCORING_HINT =
  '100% = exactly where it settles  ·  0% = missed by as far as it moved';
