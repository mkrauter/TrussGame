"""Generate the article's figures.

    python training/build_figures.py

Every figure is drawn from the project's own data -- real trusses out of the
corpus, a real stiffness matrix, a real published curve -- so none of them can
quietly disagree with the text they illustrate. Same reasoning as the notebook
importing `trussnet` rather than pasting it.

Output goes to images/, twice per figure: `name.png` and `name-dark.png`. A
single transparent figure with a compromise palette was tried first and is
muddy at one end or the other -- dark line art vanishes on a dark page. Since
each variant knows its own background it can also halo label text properly.
build_article.py pairs them into a <picture> automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'training'))
IMAGES = ROOT / 'images'

# One palette per theme. A single transparent figure tuned for both ends up
# muddy at one of them -- dark line art disappears on a dark page -- so each
# figure is rendered twice and the page picks with prefers-color-scheme.
THEMES = {
    'light': dict(INK='#6a6560', STRONG='#2f2d2b', BLUE='#4a4ac4',
                  GREEN='#3d883d', RED='#c04a30', FAINT='#c9c4be', CMAP='Blues',
                  BG='#fbfaf8'),
    'dark':  dict(INK='#a39e98', STRONG='#e2dfdb', BLUE='#9d9dea',
                  GREEN='#6fc06f', RED='#e0836b', FAINT='#4a4c50', CMAP='Blues_r',
                  BG='#191a1c'),
}

# Rebound by render_all() before each pass.
INK = STRONG = BLUE = GREEN = RED = FAINT = CMAP = BG = None
SUFFIX = ''


def save(fig, name):
    path = IMAGES / f'{name}{SUFFIX}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)')


# --------------------------------------------------------------------------
# 1. Von Neumann's elephant, actually drawn.
# --------------------------------------------------------------------------
def elephant():
    """The curve from Mayer, Khairy & Howard (2010), 'Drawing an elephant with
    four complex parameters'. Four complex numbers produce the animal; the
    fifth moves the trunk, which is the joke in the quote made literal."""
    p = [50 - 30j, 18 + 8j, 12 - 10j, -14 - 60j, 40 + 20j]

    def fourier(t, coeffs):
        out = np.zeros_like(t)
        for k, c in enumerate(coeffs):
            out += c.real * np.cos(k * t) + c.imag * np.sin(k * t)
        return out

    def body(t, wiggle=0.0):
        cx = np.zeros(6, dtype=complex)
        cy = np.zeros(6, dtype=complex)
        cx[1] = p[0].real * 1j
        cx[2] = p[1].real * 1j
        cx[3] = p[2].real
        cx[5] = p[3].real
        cy[1] = p[3].imag + p[0].imag * 1j
        cy[2] = p[1].imag * 1j
        cy[3] = p[2].imag * 1j
        x, y = fourier(t, cx), fourier(t, cy)
        # Plotted as (y, -x), which is the orientation the paper's figure uses.
        px, py = y, -x
        if wiggle:
            # The trunk is the run of points furthest to the right; swing its
            # tip about the point where it leaves the body.
            tip = px > np.percentile(px, 88)
            px, py = px.copy(), py.copy()
            span = float(px[tip].max() - px[tip].min())
            reach = (px[tip] - px[tip].min()) / max(span, 1e-9)
            py[tip] += wiggle * 26 * reach ** 1.6
        return px, py

    # One full turn. Going further retraces the curve over itself and the
    # animal stops being recognisable, which is how the first version looked.
    t = np.linspace(0, 2 * np.pi, 1500)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for w, alpha in ((-1.0, 0.25), (1.0, 0.25)):
        px, py = body(t, w)
        ax.plot(px, py, color=BLUE, lw=1.5, alpha=alpha)

    px, py = body(t)
    ax.plot(px, py, color=STRONG, lw=2.4)
    ax.plot([p[4].imag], [p[4].imag], marker='o', ms=5, color=STRONG)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Four complex numbers. The fifth one makes him wiggle.',
                 color=INK, fontsize=11, pad=14)
    save(fig, 'elephant')


# --------------------------------------------------------------------------
# 2. K is sparse. Its inverse is not. The whole argument in one picture.
# --------------------------------------------------------------------------
def stiffness():
    from trussnet import graph_data

    _, samples = graph_data.load_raw('val')
    s = samples[7]
    nodes = np.array(s['nodes'])
    n = len(nodes)

    K = np.zeros((2 * n, 2 * n))
    for a, b in s['elements']:
        d = nodes[b] - nodes[a]
        L = float(np.hypot(*d))
        c, sn = d / L
        rows = [a * 2, a * 2 + 1, b * 2, b * 2 + 1]
        block = np.array([[c * c, c * sn, -c * c, -c * sn],
                          [c * sn, sn * sn, -c * sn, -sn * sn],
                          [-c * c, -c * sn, c * c, c * sn],
                          [-c * sn, -sn * sn, c * sn, sn * sn]])
        K[np.ix_(rows, rows)] += block / L

    free = np.array(sorted(set(range(2 * n)) - {s['supports'][0] * 2, s['supports'][0] * 2 + 1,
                                                s['supports'][1] * 2, s['supports'][1] * 2 + 1}))
    Kff = K[np.ix_(free, free)]
    inverse = np.linalg.inv(Kff)

    # State what the picture actually shows. A ten-joint truss is small, so K is
    # only moderately sparse; the striking half is the inverse, which has no
    # zeros at all. Claiming "mostly empty" would overstate the left panel.
    empty_k = (np.abs(Kff) < 1e-12).mean() * 100
    empty_inv = (np.abs(inverse) < 1e-12).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, matrix, title, note in (
        (axes[0], Kff, 'K', f'one block per member.\n{empty_k:.0f}% of it is zero'),
        (axes[1], inverse, 'K$^{-1}$', f'{empty_inv:.0f}% zero. every joint\nmoves every other joint'),
    ):
        magnitude = np.abs(matrix) / np.abs(matrix).max()
        ax.imshow(magnitude ** 0.35, cmap=CMAP, vmin=0, vmax=1)
        ax.set_title(title, color=STRONG, fontsize=15, pad=8)
        ax.text(0.5, -0.13, note, transform=ax.transAxes, ha='center',
                va='top', color=INK, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(FAINT)

    fig.suptitle('The structure is local. The answer to it is not.',
                 color=INK, fontsize=12, y=1.02)
    save(fig, 'stiffness')


# --------------------------------------------------------------------------
# 3. The loaded joint does not go straight down.
# --------------------------------------------------------------------------
def not_straight_down():
    from trussnet import graph_data

    _, samples = graph_data.load_raw('val')

    # Pick a truss whose movement is emphatically not vertical.
    best, best_ratio = None, 0.0
    for s in samples[:400]:
        d = np.array(s['displacement'][s['loadedNode']])
        ratio = abs(d[0]) / (abs(d[1]) + 1e-9)
        if ratio > best_ratio and np.hypot(*d) > 120:
            best, best_ratio = s, ratio
    s = best

    pts = np.array(s['nodes'])
    moved = pts + np.array(s['displacement'])
    n = s['loadedNode']

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for a, b in s['elements']:
        ax.plot(*zip(pts[a], pts[b]), color=FAINT, lw=1.3, zorder=1)
        ax.plot(*zip(moved[a], moved[b]), color=STRONG, lw=1.7, zorder=3)

    ax.scatter(*pts[s['supports']].T, marker='^', s=150, color=GREEN, zorder=5)

    travel = np.hypot(*(moved[n] - pts[n]))
    ax.plot([pts[n][0], pts[n][0]], [pts[n][1], pts[n][1] + travel],
            color=RED, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.scatter([pts[n][0]], [pts[n][1] + travel], marker='x', s=70, color=RED, zorder=6)

    # Both labels sit over members, so they carry a halo in the page's own
    # background colour -- which is knowable now that each figure is rendered
    # per theme rather than once for both.
    halo = [path_effects.withStroke(linewidth=3.5, foreground=BG)]

    ax.annotate('where everyone\nexpects it to go', (pts[n][0], pts[n][1] + travel),
                textcoords='offset points', xytext=(-14, -6), ha='right',
                color=RED, fontsize=10, path_effects=halo, zorder=9)

    ax.annotate('', xy=moved[n], xytext=pts[n],
                arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=2.2,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.scatter([pts[n][0]], [pts[n][1]], s=70, color=BLUE, zorder=8)
    ax.annotate('where it actually goes', moved[n], textcoords='offset points',
                xytext=(-8, 26), ha='right', color=BLUE, fontsize=10,
                fontweight='bold', path_effects=halo, zorder=9)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.margins(0.16)
    save(fig, 'not-straight-down')


def render_all():
    global INK, STRONG, BLUE, GREEN, RED, FAINT, CMAP, BG, SUFFIX
    for theme, palette in THEMES.items():
        INK, STRONG = palette['INK'], palette['STRONG']
        BLUE, GREEN, RED = palette['BLUE'], palette['GREEN'], palette['RED']
        FAINT, CMAP, BG = palette['FAINT'], palette['CMAP'], palette['BG']
        # The dark variant is what the -dark suffix means to build_article.py.
        SUFFIX = '' if theme == 'light' else '-dark'
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'text.color': INK,
            'axes.labelcolor': INK,
            'savefig.transparent': True,
        })
        print(f'  [{theme}]')
        elephant()
        stiffness()
        not_straight_down()


if __name__ == '__main__':
    IMAGES.mkdir(exist_ok=True)
    print('rendering figures:')
    render_all()
