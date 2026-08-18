// Bowyer-Watson Delaunay triangulation, replacing scipy.spatial.Delaunay.
// Only ever runs on 10 points once per round, so the O(n^2) formulation is
// more than fast enough and avoids a dependency entirely.

// Positive when a, b, c are counter-clockwise in standard math axes.
function orient(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

// Stores a triangle with its vertices normalised to counter-clockwise, which
// is what the in-circumcircle determinant below assumes.
function makeTriangle(points, i, j, k) {
  return orient(points[i], points[j], points[k]) < 0 ? [i, k, j] : [i, j, k];
}

function inCircumcircle(p, a, b, c) {
  const ax = a[0] - p[0], ay = a[1] - p[1];
  const bx = b[0] - p[0], by = b[1] - p[1];
  const cx = c[0] - p[0], cy = c[1] - p[1];
  const det =
    (ax * ax + ay * ay) * (bx * cy - by * cx) -
    (bx * bx + by * by) * (ax * cy - ay * cx) +
    (cx * cx + cy * cy) * (ax * by - ay * bx);
  return det > 0;
}

// Returns triangles as index triples into `points`.
export function triangulate(points) {
  const n = points.length;
  if (n < 3) return [];

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  // Super-triangle, generously oversized so no real point lands near its edges.
  const pts = points.concat([
    [midX - 20 * span, midY - span],
    [midX, midY + 20 * span],
    [midX + 20 * span, midY - span],
  ]);

  let triangles = [makeTriangle(pts, n, n + 1, n + 2)];

  for (let i = 0; i < n; i++) {
    const surviving = [];
    const edgeCount = new Map();

    for (const t of triangles) {
      if (inCircumcircle(pts[i], pts[t[0]], pts[t[1]], pts[t[2]])) {
        // Cavity triangle: tally its edges so we can find the boundary.
        for (let e = 0; e < 3; e++) {
          const a = t[e], b = t[(e + 1) % 3];
          const key = a < b ? `${a},${b}` : `${b},${a}`;
          edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
        }
      } else {
        surviving.push(t);
      }
    }

    triangles = surviving;
    // Edges seen once are the cavity boundary; fan them to the new point.
    for (const [key, count] of edgeCount) {
      if (count !== 1) continue;
      const [a, b] = key.split(',').map(Number);
      triangles.push(makeTriangle(pts, a, b, i));
    }
  }

  return triangles.filter((t) => t.every((v) => v < n));
}

// Unique undirected edges, each as a sorted [lo, hi] index pair.
export function edgesOf(triangles) {
  const seen = new Map();
  for (const t of triangles) {
    for (let e = 0; e < 3; e++) {
      const a = t[e], b = t[(e + 1) % 3];
      const lo = Math.min(a, b), hi = Math.max(a, b);
      seen.set(`${lo},${hi}`, [lo, hi]);
    }
  }
  return [...seen.values()].sort((p, q) => p[0] - q[0] || p[1] - q[1]);
}
