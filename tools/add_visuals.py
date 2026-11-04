#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Схемы и графики для шпаргалок.

Описания лежат в tools/visuals/*.py: каждый модуль задаёт словарь
    VISUALS = {"<тема>": [рисунок, ...], ...}
где рисунок — словарь:
    {"title": "Заголовок блока", "svg": "<svg viewBox=...>...</svg>", "caption": "подпись"}   — SVG-схема
    {"title": "Заголовок блока", "chart": функция(fig), "caption": "подпись"}               — график matplotlib

Скрипт вставляет рисунки в cheatsheets/<тема>_cheatsheet.html сразу после первого блока («Суть»),
между метками <!-- mlcs-visual:start --> и <!-- mlcs-visual:end -->. Повторный запуск заменяет вставку,
поэтому рисунок правят в tools/visuals, а не в HTML.

Оформление SVG — классы из assets/theme.css (раздел «Схемы»): цвета подстраиваются под тёмную тему.
    c1…c6  заливка+обводка фигуры (синий, зелёный, оранжевый, красный, фиолетовый, серый)
    s1…s6  только обводка линии тем же цветом;  t1…t6 — цветной текст или точка (fill)
    ln     линия цветом текста;  mut — приглушённый текст;  ah — наконечник стрелки (fill цветом текста)
Текст берёт цвет текста страницы (fill: currentColor) — явные цвета в SVG не нужны.

Запуск из корня репозитория:
    python tools/add_visuals.py                  # все модули
    python tools/add_visuals.py --only classic   # только tools/visuals/classic.py
    python tools/add_visuals.py --check          # проверить описания, ничего не записывая
    python tools/add_visuals.py --preview out.html   # все рисунки на одной странице для просмотра
"""

import argparse
import base64
import importlib
import io
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHEATS = ROOT / "cheatsheets"
sys.path.insert(0, str(Path(__file__).resolve().parent))

START, END = "<!-- mlcs-visual:start -->", "<!-- mlcs-visual:end -->"
REGION = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
# цвета палитры — те же, что у классов c1…c6 в theme.css (светлая тема): для графиков matplotlib
PALETTE = ["#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"]


def modules(only=None):
    names = sorted(p.stem for p in (ROOT / "tools" / "visuals").glob("*.py") if not p.stem.startswith("_"))
    if only:
        names = [n for n in names if n in only]
    for name in names:
        yield name, importlib.import_module(f"visuals.{name}").VISUALS


def check_svg(svg, where):
    errs = []
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return [f"{where}: SVG не разбирается — {e}"], []
    if not root.tag.endswith("svg"):
        errs.append(f"{where}: корневой элемент не <svg>")
    if "viewBox" not in root.attrib:
        errs.append(f"{where}: у <svg> нет viewBox")
    if re.search(r"<script|<[^>]*\son\w+\s*=", svg, re.I):   # обработчик — только атрибут внутри тега
        errs.append(f"{where}: скрипты и обработчики событий в SVG запрещены")
    ids = re.findall(r'\bid="([^"]+)"', svg)
    return errs, ids


def render_chart(func):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    with plt.rc_context({"font.size": 10, "axes.prop_cycle": cycler(color=PALETTE), "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 110, "savefig.dpi": 110}):
        fig = plt.figure(figsize=(4.2, 3.0))
        func(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def figure_html(stem, item, n):
    if "svg" in item:
        body = item["svg"].strip()
    else:
        alt = esc(item.get("alt") or item["title"]).replace('"', "&quot;")
        body = f'<img src="data:image/png;base64,{render_chart(item["chart"])}" alt="{alt}">'
    cap = f'<figcaption>{item["caption"]}</figcaption>' if item.get("caption") else ""
    return (f'<div class="block mlcs-visual" id="vis-{stem}-{n}"><h2>🖼 {esc(item["title"])}</h2>'
            f'<figure>{body}{cap}</figure></div>')


def first_block_end(html):
    """Позиция сразу после закрывающего </div> первого <div class="block">."""
    start = html.index('<div class="block"')
    depth = 0
    for m in re.compile(r"<div\b|</div>").finditer(html, start):
        depth += 1 if m.group() != "</div>" else -1
        if depth == 0:
            return m.end()
    raise ValueError("не нашёл конец первого блока")


def validate(stem, items):
    errs, all_ids = [], []
    path = CHEATS / f"{stem}_cheatsheet.html"
    if not path.exists():
        errs.append(f"{stem}: нет шпаргалки {path.name}")
    if not items:
        errs.append(f"{stem}: пустой список рисунков")
    for n, item in enumerate(items, 1):
        where = f"{stem} #{n}"
        if not item.get("title"):
            errs.append(f"{where}: нет title")
        if ("svg" in item) == ("chart" in item):
            errs.append(f"{where}: нужен ровно один из ключей svg / chart")
        elif "svg" in item:
            e, ids = check_svg(item["svg"], where)
            errs += e
            all_ids += ids
        elif not callable(item["chart"]):
            errs.append(f"{where}: chart должен быть функцией fig → None")
    dup = {i for i in all_ids if all_ids.count(i) > 1}
    if dup:
        errs.append(f"{stem}: повторяются id внутри страницы: {', '.join(sorted(dup))}")
    return errs


def apply(stem, items):
    path = CHEATS / f"{stem}_cheatsheet.html"
    html = path.read_text(encoding="utf-8")
    html = REGION.sub("", html)
    region = START + "".join(figure_html(stem, it, n) for n, it in enumerate(items, 1)) + END
    pos = first_block_end(html)
    new = html[:pos] + region + html[pos:]
    if new != path.read_text(encoding="utf-8"):
        path.write_text(new, encoding="utf-8")
        return True
    return False


def preview(all_visuals, out):
    parts = []
    for module, visuals in all_visuals:
        parts.append(f"<h1>{module}</h1>")
        for stem, items in visuals.items():
            parts.append(f"<h3>{stem}</h3><div class='container'>"
                         + "".join(figure_html(stem, it, n) for n, it in enumerate(items, 1)) + "</div>")
    rel = (ROOT / "assets").as_uri()
    Path(out).write_text(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<link rel="stylesheet" href="{rel}/theme.css"><style>body{{font-family:Segoe UI,Arial,sans-serif;padding:12px}}
.container{{column-count:3;column-gap:16px}}.block{{break-inside:avoid;margin-bottom:12px;padding:10px;background:#fff;
border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}h2{{font-size:1em;color:#1a5fb4}}</style></head>
<body>{''.join(parts)}</body></html>""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="имена модулей tools/visuals/<имя>.py")
    ap.add_argument("--check", action="store_true", help="только проверить описания")
    ap.add_argument("--preview", help="записать страницу со всеми рисунками")
    args = ap.parse_args()

    all_visuals = list(modules(args.only))
    errs, seen = [], {}
    for module, visuals in all_visuals:
        for stem, items in visuals.items():
            if stem in seen:
                errs.append(f"{stem}: описан и в {seen[stem]}, и в {module}")
            seen[stem] = module
            errs += validate(stem, items)
    if errs:
        print("\n".join("  ✖ " + e for e in errs))
        sys.exit(1)
    if args.preview:
        preview(all_visuals, args.preview)
        print(f"Превью: {args.preview}")
        return
    if args.check:
        print(f"✔ описания в порядке: тем {len(seen)}, рисунков {sum(len(v) for _, vs in all_visuals for v in vs.values())}")
        return
    changed = sum(apply(stem, items) for _, visuals in all_visuals for stem, items in visuals.items())
    print(f"Рисунки: тем {len(seen)}, изменено шпаргалок {changed}")


if __name__ == "__main__":
    main()
