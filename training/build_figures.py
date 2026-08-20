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

    joints = [i for i in range(n) if i not in s['supports']]
    free = np.array([d for i in joints for d in (i * 2, i * 2 + 1)])
    Kff = K[np.ix_(free, free)]
    inverse = np.linalg.inv(Kff)

    # Collapse each 2x2 block to one number per pair of joints. The raw matrices
    # carry two rows per joint, one per direction, which is a detail the article
    # never mentions and a reader would have to be told to ignore. One cell per
    # pair of joints is a claim they can read straight off: does this joint
    # affect that one.
    def per_joint(matrix):
        size = len(joints)
        out = np.zeros((size, size))
        for a in range(size):
            for b in range(size):
                out[a, b] = np.linalg.norm(matrix[2 * a:2 * a + 2, 2 * b:2 * b + 2])
        return out

    connected = per_joint(Kff)
    influence = per_joint(inverse)

    # Somewhere the two panels visibly disagree: a pair with no member between
    # them that nonetheless moves each other. Pointing at one specific square is
    # what turns this from decoration into an argument.
    gap = None
    for a in range(len(joints)):
        for b in range(len(joints)):
            if a != b and connected[a, b] < 1e-12:
                gap = (a, b)
                break
        if gap:
            break

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    panels = (
        (axes[0], connected, 'Joined by a member',
         'a few neighbours each —\nthe blank squares are pairs\nwith nothing between them'),
        (axes[1], influence, 'Actually moves it',
         'no blank squares at all —\nload one joint and every\nother joint shifts'),
    )
    for ax, matrix, title, note in panels:
        magnitude = matrix / matrix.max()
        ax.imshow(magnitude ** 0.35, cmap=CMAP, vmin=0, vmax=1)
        ax.set_title(title, color=STRONG, fontsize=13, pad=10)
        ax.text(0.5, -0.14, note, transform=ax.transAxes, ha='center',
                va='top', color=INK, fontsize=9.5)
        ax.set_xlabel('each column is a joint', color=INK, fontsize=9, labelpad=2)
        ax.set_ylabel('each row is a joint', color=INK, fontsize=9, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(FAINT)

        if gap:
            ax.add_patch(plt.Rectangle((gap[1] - 0.5, gap[0] - 0.5), 1, 1,
                                       fill=False, edgecolor=RED, lw=2.2, zorder=3))

    if gap:
        # Above the grid, not on it: red text over blue cells is unreadable, and
        # a halo alone does not rescue it.
        halo = [path_effects.withStroke(linewidth=3.5, foreground=BG)]
        for ax, text in ((axes[0], 'these two are not connected'),
                         (axes[1], 'and they still move each other')):
            ax.annotate(text, (gap[1], gap[0] - 0.5),
                        textcoords='offset points', xytext=(0, 24),
                        ha='center', va='bottom', color=RED, fontsize=9.5,
                        path_effects=halo, annotation_clip=False,
                        arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.3,
                                        shrinkA=2, shrinkB=2))

    fig.suptitle('The structure is local. What it does is not.',
                 color=INK, fontsize=12.5, y=1.03)
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
        receptive_field()
        message_passing()
        difficulty()




# --------------------------------------------------------------------------
# 4. What one part of a convolutional network can see at once.
# --------------------------------------------------------------------------
def receptive_field():
    from trussnet import graph_data

    _, samples = graph_data.load_raw('val')
    s = samples[3]
    pts = np.array(s['nodes'])
    a, b = s['supports']
    span = float(np.hypot(*(pts[b] - pts[a])))

    # The network saw a 256px crop of a 768px region, so a receptive field of
    # 106px there covers 106 * 3 = 318px of the picture the player sees.
    field = 106 * 768 / 256
    centre = pts[s['loadedNode']]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, j in s['elements']:
        ax.plot(*zip(pts[i], pts[j]), color=FAINT, lw=1.4, zorder=1)
    ax.scatter(*pts[[a, b]].T, marker='^', s=150, color=GREEN, zorder=4)
    ax.scatter(*centre, s=70, color=BLUE, zorder=4)

    ax.add_patch(plt.Rectangle((centre[0] - field / 2, centre[1] - field / 2),
                               field, field, facecolor=BLUE, alpha=0.13,
                               edgecolor=BLUE, lw=1.6, zorder=2))

    # The span it needed to see, drawn between the two supports.
    y = max(pts[:, 1]) + 70
    ax.annotate('', xy=(pts[a][0], y), xytext=(pts[b][0], y),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.4))
    ax.text((pts[a][0] + pts[b][0]) / 2, y + 26,
            f'the supports are {span:.0f}px apart', ha='center', va='top',
            color=RED, fontsize=10)
    ax.text(centre[0], centre[1] - field / 2 - 14,
            f'one unit sees {field:.0f}px', ha='center', va='bottom',
            color=BLUE, fontsize=10, fontweight='bold',
            path_effects=[path_effects.withStroke(linewidth=3.5, foreground=BG)])

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.margins(0.15)
    ax.set_title('No part of the network ever saw both supports at once.',
                 color=INK, fontsize=11, pad=12)
    save(fig, 'receptive-field')


# --------------------------------------------------------------------------
# 5. Message passing: one round is one step outward.
# --------------------------------------------------------------------------
def message_passing():
    from trussnet import graph_data

    _, samples = graph_data.load_raw('val')

    def hops_from_load(sample):
        start = sample['loadedNode']
        neighbours = {i: [] for i in range(len(sample['nodes']))}
        for i, j in sample['elements']:
            neighbours[i].append(j)
            neighbours[j].append(i)
        hop = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for u in frontier:
                for v in neighbours[u]:
                    if v not in hop:
                        hop[v] = hop[u] + 1
                        nxt.append(v)
            frontier = nxt
        return hop

    # Pick a truss where the news genuinely takes three rounds to arrive --
    # on a tightly connected one the second and third panels are identical and
    # the figure argues against itself.
    s = max(samples[:400], key=lambda x: max(hops_from_load(x).values()))
    hop = hops_from_load(s)
    pts = np.array(s['nodes'])
    start = s['loadedNode']

    rounds = [1, 2, 3]
    fig, axes = plt.subplots(1, len(rounds), figsize=(11.4, 3.9))
    for ax, r in zip(axes, rounds):
        for i, j in s['elements']:
            lit = hop[i] < r and hop[j] <= r or hop[j] < r and hop[i] <= r
            ax.plot(*zip(pts[i], pts[j]), color=BLUE if lit else FAINT,
                    lw=2.0 if lit else 1.2, zorder=2 if lit else 1)
        reached = [i for i in range(len(pts)) if hop[i] <= r]
        ax.scatter(*pts[reached].T, s=42, color=BLUE, zorder=3)
        ax.scatter(*pts[start], s=90, color=STRONG, zorder=4)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.margins(0.12)
        covered = len(reached)
        ax.set_title(f'round {r} — {covered} of {len(pts)} joints',
                     color=INK, fontsize=10.5, pad=8)

    fig.suptitle('News of the load travels one member per round.',
                 color=INK, fontsize=12, y=1.04)
    save(fig, 'message-passing')


# --------------------------------------------------------------------------
# 6. Difficulty is how long it thinks.
# --------------------------------------------------------------------------
def difficulty():
    # Measured on the deployed model through the browser, over the validation
    # seeds: `node training/eval_pixel_pipeline.mjs --rounds N`. Hardcoded
    # rather than recomputed here because that needs a browser and the corpus;
    # re-measure if the model is ever retrained.
    rounds = np.array([1, 2, 4, 6, 8, 10])
    score = np.array([24.6, 30.7, 46.4, 68.3, 85.6, 95.7])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhspan(70, 80, color=GREEN, alpha=0.14, zorder=0)
    ax.text(1.15, 75, 'where human players sit', color=GREEN, fontsize=10,
            va='center')
    ax.axhline(59.5, color=RED, lw=1.3, ls=(0, (5, 4)), zorder=1)
    # Right-aligned under the line: on the left it sat on top of the 4-round
    # point, which is exactly where the curve crosses this threshold.
    ax.text(10.4, 57, 'guessing straight down by the average', color=RED,
            fontsize=10, va='top', ha='right')

    ax.plot(rounds, score, color=BLUE, lw=2.4, marker='o', ms=7, zorder=3)
    for r, sc in zip(rounds, score):
        ax.annotate(f'{sc:.0f}%', (r, sc), textcoords='offset points',
                    xytext=(0, 11), ha='center', color=BLUE, fontsize=9.5,
                    fontweight='bold')

    ax.set_xlabel('rounds of thinking before it answers', color=INK)
    ax.set_ylabel('score', color=INK)
    ax.set_ylim(0, 108)
    ax.set_xticks(rounds)
    ax.tick_params(colors=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(FAINT)
    ax.grid(axis='y', color=FAINT, alpha=0.35, lw=0.7)
    ax.set_axisbelow(True)
    save(fig, 'difficulty')


if __name__ == '__main__':
    IMAGES.mkdir(exist_ok=True)
    print('rendering figures:')
    render_all()
