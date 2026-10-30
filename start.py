#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Запуск учебного портала одной командой:

    python start.py            # http://localhost:8000
    python start.py --port 8080

Собирает index.html (если его нет или README/ROADMAP новее),
поднимает локальный сервер и открывает портал в браузере.
Остановить — Ctrl+C.
"""

import argparse
import functools
import http.server
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def portal_is_stale():
    index = ROOT / "index.html"
    if not index.exists():
        return True
    sources = [ROOT / "README.md", ROOT / "ROADMAP.md", ROOT / "tools" / "portal_template.html"]
    sources += list((ROOT / "exercises").rglob("task.md")) + list((ROOT / "questions").glob("*.md"))
    return any(p.stat().st_mtime > index.stat().st_mtime for p in sources if p.exists())


def main():
    parser = argparse.ArgumentParser(description="Учебный портал ML Cheatsheets")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = parser.parse_args()

    if portal_is_stale():
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_portal.py")], check=True)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *a: None  # без лога каждого запроса
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/index.html"
    print(f"Портал открыт: {url}\nОстановить: Ctrl+C")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")


if __name__ == "__main__":
    main()
