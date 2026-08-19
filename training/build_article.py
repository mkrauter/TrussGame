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


def main():
    text = SOURCE.read_text(encoding='utf-8')

    title = re.search(r'^#\s+(.+)$', text, re.M).group(1).strip()

    # The italic block under the title becomes the standfirst rather than body copy.
    stand = re.search(r'^\*(.+?)\*\s*$', text, re.M | re.S)
    standfirst = ' '.join(stand.group(1).split()) if stand else ''

    body_md = text.split('---', 1)[1] if '---' in text else text
    body = markdown.markdown(body_md.strip(), extensions=['tables', 'smarty'])
    body = body.replace('<table>', '<div class="table-wrap"><table>').replace(
        '</table>', '</table></div>')

    words = len(re.findall(r'\w+', body_md))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.format(
        title=title,
        description=standfirst[:180],
        standfirst=standfirst,
        reading=max(1, round(words / 230)),
        body=body,
    ), encoding='utf-8')
    print(f'wrote {OUT}  ({words} words, ~{max(1, round(words / 230))} min read)')


if __name__ == '__main__':
    main()
