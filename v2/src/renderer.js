import { WINDOW, RENDER, HUD } from './config.js';

// Everything here takes a plain CanvasRenderingContext2D, so the same code
// drives the on-screen game and a headless canvas in the corpus generator.
// That is the whole point: one rasteriser, so train/serve skew is impossible.

export function drawScene(ctx, truss) {
  ctx.fillStyle = RENDER.background;
  ctx.fillRect(0, 0, WINDOW.width, WINDOW.height);
  drawTruss(ctx, truss);
}

export function drawTruss(ctx, truss) {
  for (const i of truss.supports) {
    drawMarker(ctx, truss.nodes[i], RENDER.supportColor, true);
  }
  drawMarker(ctx, truss.loadedEnd, RENDER.loadedColor, false);

  ctx.lineWidth = RENDER.lineWidth;
  ctx.lineCap = 'round';
  truss.elements.forEach((e, i) => {
    const a = truss.nodesMoved[e[0]];
    const b = truss.nodesMoved[e[1]];
    ctx.strokeStyle = stressColor(truss.sigmas[i]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  });
}

// Support markers point up, the loaded node points down.
function drawMarker(ctx, p, color, up) {
  const [halfWidth, height] = RENDER.markerSize;
  const dir = up ? 1 : -1;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(p[0] - halfWidth, p[1] + height * dir);
  ctx.lineTo(p[0] + halfWidth, p[1] + height * dir);
  ctx.closePath();
  ctx.fill();
}

export function drawCross(ctx, p, color, size = 40) {
  const offset = size / 2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(p[0] - offset, p[1]);
  ctx.lineTo(p[0] + offset, p[1]);
  ctx.moveTo(p[0], p[1] - offset);
  ctx.lineTo(p[0], p[1] + offset);
  ctx.stroke();
}

// Score, round tally and prompt, drawn in the margins outside the model crop.
// Kept out of drawScene on purpose: the corpus generator must never render it.
export function drawHud(ctx, { accuracy, rounds, average, prompt }) {
  ctx.textBaseline = 'alphabetic';

  // Nothing at all until there is a real score -- before the first guess of the
  // session there is nothing to measure, and a placeholder glyph sitting alone
  // in the margin reads as an artefact rather than as "not yet".
  if (accuracy !== null) {
    ctx.font = HUD.scoreFont;
    ctx.fillStyle = HUD.color;
    ctx.textAlign = 'left';
    ctx.fillText(`${accuracy.toFixed(0)}%`, HUD.margin, HUD.topBaseline);
  }

  if (rounds > 0) {
    ctx.font = HUD.labelFont;
    ctx.fillStyle = HUD.dimColor;
    ctx.textAlign = 'right';
    const plural = rounds === 1 ? '' : 's';
    ctx.fillText(
      `${rounds} round${plural} · average ${average.toFixed(1)}%`,
      WINDOW.width - HUD.margin,
      HUD.topBaseline
    );
  }

  ctx.font = HUD.labelFont;
  ctx.fillStyle = HUD.dimColor;
  ctx.textAlign = 'center';
  ctx.fillText(prompt, WINDOW.width / 2, HUD.bottomBaseline);
}

// White at zero stress, red in compression, blue in tension.
export function stressColor(sigma, scale = RENDER.stressScale) {
  const stress = sigma * scale;
  const clamp = (v) => Math.round(Math.max(0, Math.min(255, v)));
  return `rgb(${clamp(255 - stress)}, ${clamp(255 - Math.abs(stress))}, ${clamp(255 + stress)})`;
}
