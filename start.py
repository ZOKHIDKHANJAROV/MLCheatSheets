#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Запуск учебного портала одной командой:

    python start.py            # http://localhost:8000
    python start.py --port 8080
    docker build -t mlcs . && docker run --rm -p 8000:8000 mlcs   # в Docker

Собирает index.html (если его нет или README/ROADMAP новее),
поднимает локальный сервер и открывает портал в браузере.
Остановить — Ctrl+C.

Профили учеников (имя + PIN) хранятся в my_profiles.json: там же прогресс и уровень, поэтому
их видно с любого устройства, которое открывает этот сервер. Другой файл — `--profiles ПУТЬ`
или переменная MLCS_PROFILES (в Docker: -v mlcs-data:/data -e MLCS_PROFILES=/data/profiles.json).
"""

import argparse
import base64
import functools
import hashlib
import hmac
import http.server
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES_FILE = "my_profiles.json"


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


class ProfileStore:
    """Профили: имя, PIN (PBKDF2 с солью), прогресс и уровень. Вход даёт подписанный токен на 30 дней."""

    NAME = re.compile(r"^[\w\-]{2,24}$")
    TOKEN_DAYS = 30
    MAX_FAILS, LOCK_SECONDS = 5, 60
    MAX_STATE_BYTES = 2_000_000

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.fails = {}      # имя → [число неверных попыток, время, до которого вход закрыт]
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("users", {})
        if not self.data.get("secret"):
            self.data["secret"] = secrets.token_hex(32)
            self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _hash(pin, salt):
        return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, 120_000).hex()

    def _token(self, key):
        payload = base64.urlsafe_b64encode(json.dumps({"u": key, "e": int(time.time()) + self.TOKEN_DAYS * 86400}).encode()).decode()
        sig = hmac.new(self.data["secret"].encode(), payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def user_of(self, token):
        """Ключ пользователя по токену или ApiError 401."""
        try:
            payload, sig = token.rsplit(".", 1)
            good = hmac.new(self.data["secret"].encode(), payload.encode(), hashlib.sha256).hexdigest()
            info = json.loads(base64.urlsafe_b64decode(payload.encode()))
            if hmac.compare_digest(sig, good) and info["e"] > time.time() and info["u"] in self.data["users"]:
                return info["u"]
        except (ValueError, KeyError, TypeError):
            pass
        raise ApiError(401, "Сессия истекла — войдите снова")

    @staticmethod
    def _check_progress(state, xp, level):
        if not isinstance(state, dict):
            raise ApiError(400, "Нет данных прогресса")
        if len(json.dumps(state, ensure_ascii=False)) > ProfileStore.MAX_STATE_BYTES:
            raise ApiError(413, "Слишком большой прогресс")
        clamp = lambda v, lo, hi: max(lo, min(hi, int(v))) if isinstance(v, (int, float)) else lo
        return clamp(xp, 0, 10**6), clamp(level, 1, 99)

    def _session(self, key):
        u = self.data["users"][key]
        return {"name": u["name"], "token": self._token(key), "xp": u["xp"], "level": u["level"], "updated": u["updated"]}

    def register(self, name, pin, state, xp, level):
        name, pin = str(name or "").strip(), str(pin or "")
        if not self.NAME.match(name):
            raise ApiError(400, "Имя — 2–24 символа: буквы, цифры, «_» и «-»")
        if not 4 <= len(pin) <= 32:
            raise ApiError(400, "PIN — от 4 до 32 символов")
        xp, level = self._check_progress(state if state is not None else {}, xp, level)
        with self.lock:
            key = name.lower()
            if key in self.data["users"]:
                raise ApiError(409, "Такое имя уже занято")
            salt = secrets.token_bytes(16)
            self.data["users"][key] = {"name": name, "salt": salt.hex(), "hash": self._hash(pin, salt), "created": int(time.time() * 1000),
                                       "updated": int(time.time() * 1000), "xp": xp, "level": level, "state": state or {}}
            self._write()
            return self._session(key)

    def login(self, name, pin):
        key = str(name or "").strip().lower()
        with self.lock:
            fails = self.fails.setdefault(key, [0, 0.0])
            if fails[1] > time.time():
                raise ApiError(429, f"Слишком много попыток — подождите {int(fails[1] - time.time()) + 1} с")
            user = self.data["users"].get(key)
            ok = user and hmac.compare_digest(self._hash(str(pin or ""), bytes.fromhex(user["salt"])), user["hash"])
            if not ok:
                fails[0] += 1
                if fails[0] >= self.MAX_FAILS:
                    fails[:] = [0, time.time() + self.LOCK_SECONDS]
                raise ApiError(401, "Неверное имя или PIN")
            self.fails.pop(key, None)
            return self._session(key)

    def get_progress(self, key):
        with self.lock:
            u = self.data["users"][key]
            return {"state": u["state"], "xp": u["xp"], "level": u["level"], "updated": u["updated"]}

    def put_progress(self, key, state, xp, level):
        xp, level = self._check_progress(state, xp, level)
        with self.lock:
            u = self.data["users"][key]
            u.update(state=state, xp=xp, level=level, updated=int(time.time() * 1000))
            self._write()
            return {"updated": u["updated"]}


class PortalHandler(http.server.SimpleHTTPRequestHandler):
    store = None       # ProfileStore — задаётся в main()
    MAX_BODY = 3_000_000

    def end_headers(self):
        # Браузер перепроверяет файлы при каждом открытии — после обновления шпаргалок не нужен Ctrl+F5
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    # ---- API профилей: /api/ping, /api/register, /api/login, /api/progress
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > self.MAX_BODY:
            raise ApiError(413, "Слишком большой запрос")
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise ApiError(400, "Нужен JSON")
        if not isinstance(data, dict):
            raise ApiError(400, "Нужен JSON-объект")
        return data

    def _api(self, method):
        route = self.path.split("?")[0]
        try:
            if method == "GET" and route == "/api/ping":
                return self._json(200, {"ok": True})
            if method == "POST" and route == "/api/register":
                b = self._body()
                return self._json(200, self.store.register(b.get("name"), b.get("pin"), b.get("state"), b.get("xp"), b.get("level")))
            if method == "POST" and route == "/api/login":
                b = self._body()
                return self._json(200, self.store.login(b.get("name"), b.get("pin")))
            if route == "/api/progress" and method in ("GET", "PUT"):
                auth = self.headers.get("Authorization", "")
                key = self.store.user_of(auth[7:] if auth.startswith("Bearer ") else "")
                if method == "GET":
                    return self._json(200, self.store.get_progress(key))
                b = self._body()
                return self._json(200, self.store.put_progress(key, b.get("state"), b.get("xp"), b.get("level")))
            raise ApiError(404, "Нет такого метода")
        except ApiError as e:
            self._json(e.code, {"error": e.message})

    def do_POST(self):
        self._api("POST")

    def do_PUT(self):
        self._api("PUT")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/"):
            return self._api("GET")
        # файл профилей (хеши PIN, ключ подписи) наружу не отдаём
        if Path(path).name in {PROFILES_FILE, Path(self.store.path).name if self.store else PROFILES_FILE}:
            return self.send_error(404)
        # Прогресс проверок (tools/check_exercise.py) — пустой, пока ни одно упражнение не решено
        if path == "/my_progress.json" and not (ROOT / "my_progress.json").exists():
            body = b'{"exercises": {}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):
        pass  # без лога каждого запроса


def portal_is_stale():
    index = ROOT / "index.html"
    if not index.exists():
        return True
    sources = [ROOT / "README.md", ROOT / "ROADMAP.md", ROOT / "glossary.md", ROOT / "prerequisites.md", ROOT / "tools" / "portal_template.html"]
    sources += list((ROOT / "exercises").rglob("task.md")) + list((ROOT / "questions").glob("*.md"))
    sources += list((ROOT / "interactive").glob("*.html")) + list((ROOT / "projects").glob("*/README.md"))
    sources += list((ROOT / "understanding").glob("*.md")) + list((ROOT / "lectures").glob("*.md")) + list((ROOT / "tests").glob("*.md"))
    return any(p.stat().st_mtime > index.stat().st_mtime for p in sources if p.exists())


def main():
    parser = argparse.ArgumentParser(description="Учебный портал ML Cheatsheets")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="адрес сервера (0.0.0.0 — для Docker и доступа из сети)")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    parser.add_argument("--profiles", default=os.environ.get("MLCS_PROFILES") or str(ROOT / PROFILES_FILE),
                        help="файл профилей учеников (имя, PIN, прогресс, уровень)")
    args = parser.parse_args()

    if portal_is_stale():
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_portal.py")], check=True)

    PortalHandler.store = ProfileStore(args.profiles)
    handler = functools.partial(PortalHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    # ?v=… — чтобы браузер не открыл старую копию портала из кэша после обновления проекта
    url = f"http://localhost:{args.port}/index.html?v={int((ROOT / 'index.html').stat().st_mtime)}"
    print(f"Портал открыт: {url}\nПрофили учеников: {args.profiles}\nОстановить: Ctrl+C")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")


if __name__ == "__main__":
    main()
