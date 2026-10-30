#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Подключает общую тему оформления (assets/theme.css + assets/theme.js) к шпаргалкам и тестам
в cheatsheets/ — все они свёрстаны по одному шаблону. После этого страницы поддерживают тёмную тему,
выбранную в портале (index.html) или в настройках системы.

Запуск из корня репозитория:
    python tools/apply_theme.py           # подключить там, где ещё нет
    python tools/apply_theme.py --check   # только проверить (код возврата 1, если есть страницы без темы)

Скрипт идемпотентен: повторный запуск ничего не меняет. Запускайте его после добавления новых шпаргалок.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
TARGETS = ["cheatsheets/*.html"]
MARKER = b"<!-- mlcs-theme -->"


def snippet(page, newline):
    """Строки подключения с путём к assets/ относительно страницы."""
    rel = Path(os.path.relpath(ASSETS, page.parent)).as_posix()
    lines = [
        MARKER.decode(),
        f'<link rel="stylesheet" href="{rel}/theme.css">',
        f'<script src="{rel}/theme.js"></script>',
    ]
    return (newline.join(lines) + newline).encode("utf-8")


def pages():
    for pattern in TARGETS:
        yield from sorted(ROOT.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Подключение темы оформления к страницам")
    parser.add_argument("--check", action="store_true", help="только проверить, ничего не менять")
    args = parser.parse_args()

    added, missing, no_head = [], [], []
    for page in pages():
        raw = page.read_bytes()
        if MARKER in raw:
            continue
        pos = raw.lower().find(b"</head>")
        if pos < 0:
            no_head.append(page)
            continue
        if args.check:
            missing.append(page)
            continue
        newline = "\r\n" if b"\r\n" in raw[:pos] else "\n"
        page.write_bytes(raw[:pos] + snippet(page, newline) + raw[pos:])
        added.append(page)

    rel = lambda p: p.relative_to(ROOT).as_posix()
    if args.check:
        for p in missing:
            print(f"  без темы: {rel(p)}")
        print(f"Страниц без темы: {len(missing)}")
    else:
        print(f"Тема подключена к {len(added)} страницам")
    for p in no_head:
        print(f"  ⚠ нет </head>, пропущено: {rel(p)}")
    return 1 if args.check and missing else 0


if __name__ == "__main__":
    sys.exit(main())
