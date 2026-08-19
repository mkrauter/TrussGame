"""Render article.md into a typeset page under web/article/.

    python training/build_article.py

The article is written and edited as markdown; this only handles presentation.
Everything about the page below is typography: a measure of about 65 characters,
a serif face for the body at a size meant for reading rather than scanning, and
generous space around headings. It follows the reader's light/dark preference,
because 2,500 words is long enough that forcing either one is rude.
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'article.md'
OUT = ROOT / 'web' / 'article' / 'index.html'

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">

<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' fill='%23404040'/><polygon points='16,26 6,8 26,8' fill='%236c6cd2'/></svg>">

<meta property="og:type" content="article">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="https://mkrauter.github.io/TrussGame/images/screenshot.jpg">
<meta property="og:url" content="https://mkrauter.github.io/TrussGame/web/article/">
<meta name="twitter:card" content="summary_large_image">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

<style>
  :root {{
    color-scheme: light dark;
    --bg: #fbfaf8;
    --text: #1e1c1a;
    --muted: #6a6560;
    --rule: #e2ddd6;
    --accent: #3b4fd8;
    --panel: #f2efea;
    --serif: "Source Serif 4", Charter, Georgia, "Times New Roman", serif;
    --sans: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
  }}

  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #191a1c;
      --text: #e8e6e3;
      --muted: #a39e98;
      --rule: #35373a;
      --accent: #9d9dea;
      --panel: #232427;
    }}
  }}

  * {{ box-sizing: border-box; }}

  html {{ -webkit-text-size-adjust: 100%; }}

  body {{
    margin: 0;
    padding: 0 24px 96px;
    background: var(--bg);
    color: var(--text);
    font-family: var(--serif);
    /* 20px with generous leading: this is meant to be read, not skimmed. */
    font-size: 20px;
    line-height: 1.7;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
  }}

  /* ~65 characters per line, the measure typographers keep arriving at. */
  article {{ max-width: 34em; margin: 0 auto; }}

  /* Figures break out past the text column: wide enough to carry detail, still
     centred on the same axis as the prose. Transparent PNGs, so they sit on
     whichever background the reader's theme provides. */
  .fig {{
    width: min(44em, 94vw);
    margin: 2.8em calc(50% - min(22em, 47vw));
  }}
  .fig img {{ display: block; width: 100%; height: auto; }}

  /* Full bleed. The article is a centred column, so the hero escapes it simply
     by living outside <article> -- no negative-margin tricks needed. The image
     is cropped to a letterbox rather than shown whole, so its height stays
     predictable whatever you drop in, and it is capped in vh so it cannot eat
     a laptop screen. */
  .hero {{ margin: 0; }}
  .hero img {{
    display: block;
    width: 100%;
    height: min(46vh, 420px);
    object-fit: cover;
    /* Slightly above centre: the interesting part of most images is not the
       geometric middle. */
    object-position: center 42%;
  }}
  .hero figcaption {{
    max-width: 34em;
    margin: 10px auto 0;
    font-family: var(--sans);
    font-size: 0.8rem;
    color: var(--muted);
  }}
  .hero + .masthead {{ padding-top: 22px; }}

  .masthead {{
    max-width: 34em;
    margin: 0 auto;
    padding: 28px 0 0;
    font-family: var(--sans);
    font-size: 0.72rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--muted);
  }}
  .masthead a {{ color: inherit; text-decoration: none; }}
  .masthead a:hover {{ color: var(--accent); }}

  h1 {{
    font-family: var(--sans);
    font-size: clamp(2.1rem, 6vw, 3.1rem);
    line-height: 1.12;
    letter-spacing: -0.025em;
    font-weight: 700;
    margin: 40px 0 0;
  }}

  .standfirst {{
    font-size: 1.22rem;
    line-height: 1.55;
    color: var(--muted);
    font-style: italic;
    margin: 22px 0 0;
  }}

  .byline {{
    font-family: var(--sans);
    font-size: 0.86rem;
    color: var(--muted);
    font-style: normal;
    margin: 30px 0 0;
    padding-bottom: 30px;
    border-bottom: 1px solid var(--rule);
  }}
  .byline b {{ color: var(--text); font-weight: 600; }}

  h2 {{
    font-family: var(--sans);
    font-size: 1.42rem;
    line-height: 1.25;
    letter-spacing: -0.012em;
    font-weight: 600;
    margin: 2.6em 0 0.7em;
  }}

  p {{ margin: 0 0 1.35em; }}

  /* The opening line gets the newspaper treatment -- the first paragraph of the
     body, not the standfirst, which is why this hangs off .byline. */
  .byline + p::first-letter {{
    float: left;
    font-size: 3.5em;
    line-height: 0.84;
    padding: 0.05em 0.09em 0 0;
    font-weight: 600;
  }}

  a {{ color: var(--accent); text-underline-offset: 0.18em; }}

  strong {{ font-weight: 600; }}

  blockquote {{
    margin: 2em 0;
    padding: 0 0 0 1.2em;
    border-left: 3px solid var(--accent);
    font-size: 1.16rem;
    color: var(--muted);
  }}
  blockquote p {{ margin: 0; }}

  ul {{ padding-left: 1.2em; }}
  li {{ margin-bottom: 0.6em; }}

  /* Tables read as small inset panels rather than spreadsheets. */
  .table-wrap {{ overflow-x: auto; margin: 2em 0; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--sans);
    font-size: 0.92rem;
    background: var(--panel);
    border-radius: 8px;
    overflow: hidden;
  }}
  th, td {{ padding: 11px 16px; text-align: left; }}
  thead th {{
    font-size: 0.74rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
  }}
  tbody tr + tr td {{ border-top: 1px solid var(--rule); }}
  td:last-child, th:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}

  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.86em;
    background: var(--panel);
    padding: 0.1em 0.35em;
    border-radius: 4px;
  }}

  hr {{ border: 0; border-top: 1px solid var(--rule); margin: 3em 0; }}

  .play {{
    margin: 3em 0;
    padding: 26px 28px;
    background: var(--panel);
    border-radius: 12px;
    font-family: var(--sans);
    font-size: 0.98rem;
    line-height: 1.6;
  }}
  .play strong {{ display: block; font-size: 1.1rem; margin-bottom: 6px; }}

  footer {{
    max-width: 34em;
    margin: 4em auto 0;
    padding-top: 26px;
    border-top: 1px solid var(--rule);
    font-family: var(--sans);
    font-size: 0.88rem;
    color: var(--muted);
  }}
</style>
</head>
<body>
{hero}
<div class="masthead"><a href="../../">Truss game</a></div>

<article>
<h1>{title}</h1>
<p class="standfirst">{standfirst}</p>
<p class="byline">By <b>Márton Krauter</b> · {reading} min read</p>

{body}
</article>

<footer>
  <p>All three versions of the game, the source, and the notebook are at
  <a href="https://github.com/mkrauter/TrussGame">github.com/mkrauter/TrussGame</a>.</p>
</footer>

</body>
</html>
"""


def read_front_matter(text):
    """Strip an optional `key: value` block fenced by --- at the very top.

    Deliberately tiny -- no YAML dependency, no nesting. It exists so article.md
    can declare presentation bits, like a hero image, without anyone ever
    hand-editing the generated HTML.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != '---':
        return {}, text

    meta = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            return meta, ''.join(lines[i + 1:]).lstrip()
        if ':' in line:
            key, _, value = line.partition(':')
            meta[key.strip()] = value.strip().strip('"\'')
    return {}, text          # unterminated block: treat it as ordinary text


def hero_markup(meta):
    """Full-bleed image above the title, if article.md asked for one."""
    src = meta.get("hero")
    if not src:
        return ""
    alt = meta.get('heroAlt', '')
    caption = meta.get('heroCaption', '')
    # Paths in the markdown are repo-relative; the page sits two levels down.
    if not src.startswith(('http', '/')):
        src = '../../' + src
    caption_html = f'  <figcaption>{caption}</figcaption>' if caption else ''
    return f'''
<figure class="hero">
  <img src="{src}" alt="{alt}">
{caption_html}
</figure>
'''


def main():
    raw = SOURCE.read_text(encoding='utf-8')
    meta, text = read_front_matter(raw)

    title = re.search(r'^#\s+(.+)$', text, re.M).group(1).strip()

    # The italic block under the title becomes the standfirst rather than body copy.
    stand = re.search(r'^\*(.+?)\*\s*$', text, re.M | re.S)
    standfirst = ' '.join(stand.group(1).split()) if stand else ''

    body_md = text.split('---', 1)[1] if '---' in text else text
    body = markdown.markdown(body_md.strip(), extensions=['tables', 'smarty'])
    body = body.replace('<table>', '<div class="table-wrap"><table>').replace(
        '</table>', '</table></div>')

    # Images are written repo-relative in the markdown so the source stays
    # readable; the page lives two levels down, so fix them on the way out.
    body = body.replace('src="images/', 'src="../../images/')

    # A paragraph holding nothing but an image is a figure, not a paragraph.
    # Where a -dark variant exists beside the file, offer it to dark readers:
    # a single figure tuned for both themes is muddy in at least one of them.
    def as_figure(match):
        tag = match.group(1)
        src = re.search(r'src="([^"]+)"', tag).group(1)
        dark = src.replace('.png', '-dark.png')
        picture = tag
        if (ROOT / dark.replace('../../', '')).exists():
            picture = (f'<picture><source srcset="{dark}" '
                       f'media="(prefers-color-scheme: dark)">{tag}</picture>')
        return '<figure class="fig">' + picture + '</figure>'

    body = re.sub(r'<p>(<img [^>]*/?>)</p>', as_figure, body)

    words = len(re.findall(r'\w+', body_md))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.format(
        hero=hero_markup(meta),
        title=title,
        description=standfirst[:180],
        standfirst=standfirst,
        reading=max(1, round(words / 230)),
        body=body,
    ), encoding='utf-8')
    print(f'wrote {OUT}  ({words} words, ~{max(1, round(words / 230))} min read)')


if __name__ == '__main__':
    main()
