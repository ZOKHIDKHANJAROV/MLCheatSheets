#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Проверка целостности проекта одной командой.

Ищет то, что легко сломать при добавлении материалов:
  • битые ссылки на шпаргалки в README.md и ROADMAP.md;
  • битые относительные ссылки в markdown-файлах;
  • шпаргалки без подключённой темы (assets/theme.css + theme.js);
  • битые ссылки внутри шпаргалок и интерактивов, неэкранированный HTML в примерах кода,
    вложенные друг в друга блоки (незакрытый <div>);
  • ссылки портала: темы, разделы, пути, упражнения, тесты и интерактивы из index.html;
  • вопросы (questions/*.md) и интерактивы (interactive/*.html) с несуществующими темами;
  • упражнения без нужных файлов и с решением, не совпадающим с эталоном;
  • устаревший index.html (портал не пересобран после изменений).

Запуск из корня репозитория:
    python tools/check_project.py           # всё, включая запуск решений упражнений
    python tools/check_project.py --fast    # без запуска solution.py

Код возврата 1, если найдены ошибки (предупреждения на него не влияют).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

CHEATSHEETS = ROOT / "cheatsheets"
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
CHEATSHEET_LINK = re.compile(r"cheatsheets/([\w\-]+\.html)")

errors, warnings = [], []
rel = lambda p: p.relative_to(ROOT).as_posix()


def section(title):
    print(f"\n▸ {title}")


def report(found_errors, found_warnings, ok_message):
    errors.extend(found_errors)
    warnings.extend(found_warnings)
    for e in found_errors:
        print(f"  ❌ {e}")
    for w in found_warnings:
        print(f"  ⚠ {w}")
    if not found_errors and not found_warnings:
        print(f"  ✅ {ok_message}")


def check_cheatsheet_links():
    section("Ссылки на шпаргалки в README и ROADMAP")
    errs, total = [], 0
    for doc in ["README.md", "ROADMAP.md", "exercises.md", "COURSE_GUIDE.md"]:
        path = ROOT / doc
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name in CHEATSHEET_LINK.findall(line):
                total += 1
                if not (CHEATSHEETS / name).exists():
                    errs.append(f"{doc}:{n}: нет файла cheatsheets/{name}")
    report(errs, [], f"ссылок проверено: {total}, все ведут на существующие файлы")


def check_markdown_links():
    section("Относительные ссылки в markdown")
    errs, total = [], 0
    files = [p for p in ROOT.rglob("*.md") if not any(part in {"venv", ".git", "node_modules", ".ipynb_checkpoints"} for part in p.parts)]
    for md in files:
        for n, line in enumerate(md.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for target in MD_LINK.findall(line):
                if re.match(r"^(https?:|mailto:|#)", target):
                    continue
                path = target.split("#")[0].split("?")[0]
                if not path:
                    continue
                total += 1
                if not (md.parent / path).exists():
                    errs.append(f"{rel(md)}:{n}: ссылка «{target}» никуда не ведёт")
    report(errs, [], f"относительных ссылок проверено: {total} (файлов: {len(files)}), все работают")


def check_theme():
    section("Тема оформления в шпаргалках")
    from apply_theme import with_current_version
    pages = sorted(CHEATSHEETS.glob("*.html"))
    missing = [rel(p) for p in pages if b"<!-- mlcs-theme -->" not in p.read_bytes()]
    outdated = [rel(p) for p in pages if b"<!-- mlcs-theme -->" in p.read_bytes() and with_current_version(p.read_bytes()) is not None]
    errs = [f"{p}: тема не подключена — запустите python tools/apply_theme.py" for p in missing]
    errs += [f"{p}: старая версия подключения темы — запустите python tools/apply_theme.py" for p in outdated]
    report(errs, [], f"тема подключена ко всем страницам ({len(list(CHEATSHEETS.glob('*.html')))})")


HTML_REF = re.compile(r'(?:href|src)\s*=\s*"([^"]+)"', re.I)
DANGEROUS_IN_CODE = {"script", "style", "iframe", "link", "meta"}
PLAIN_IN_CODE = {"span", "b", "strong", "em", "i", "br", "code", "u", "sub", "sup", "mark", "a"}


class BlockNesting(HTMLParser):
    """Считает блоки <div class="block">, оказавшиеся внутри другого блока (признак незакрытого <div>)."""
    VOID = {"br", "img", "hr", "meta", "link", "input", "source", "col", "area", "base", "wbr", "embed", "param", "track"}

    def __init__(self):
        super().__init__()
        self.stack, self.nested = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            return
        is_block = tag == "div" and "block" in (dict(attrs).get("class") or "").split()
        if is_block and any(b for _, b in self.stack):
            self.nested += 1
        self.stack.append((tag, is_block))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def check_pages():
    section("Ссылки и разметка внутри шпаргалок и интерактивов")
    errs, warns, total = [], [], 0
    pages = sorted(CHEATSHEETS.glob("*.html")) + sorted((ROOT / "interactive").glob("*.html"))
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="ignore")
        markup = re.sub(r"<pre[^>]*>.*?</pre>", "", text, flags=re.S | re.I)   # в примерах кода — просто текст
        for ref in HTML_REF.findall(markup):
            if re.match(r"^(data:|#|https?:|//|mailto:|javascript:)", ref):
                continue
            total += 1
            if not (page.parent / ref.split("#")[0].split("?")[0]).exists():
                errs.append(f"{rel(page)}: ссылка «{ref}» никуда не ведёт")
        # HTML внутри примеров кода должен быть экранирован: иначе браузер выполнит <script> или сломает вёрстку
        for block in re.finditer(r"<pre[^>]*>(.*?)</pre>", text, re.S | re.I):
            tags = {t.lower() for t in re.findall(r"</?([a-zA-Z][a-zA-Z0-9-]*)\b", block.group(1))} - PLAIN_IN_CODE
            if tags & DANGEROUS_IN_CODE:
                errs.append(f"{rel(page)}: в примере кода неэкранированный <{sorted(tags & DANGEROUS_IN_CODE)[0]}> — браузер его выполнит (замените < на &lt;)")
            elif tags:
                warns.append(f"{rel(page)}: внутри <pre> разметка <{sorted(tags)[0]}> — блок кода, скорее всего, не закрыт")
                break
        if page.parent == CHEATSHEETS:
            nest = BlockNesting()
            nest.feed(text)
            if nest.nested:
                warns.append(f"{rel(page)}: {nest.nested} блок(ов) .block вложены друг в друга — не закрыт <div>")
            # внутри незакрытой таблицы браузер игнорирует </div> — следующие блоки «проваливаются» внутрь
            plain = re.sub(r"<pre\b.*?</pre>|<script\b.*?</script>|<!--.*?-->", "", text, flags=re.S | re.I)
            if len(re.findall(r"<table\b", plain, re.I)) != len(re.findall(r"</table\s*>", plain, re.I)):
                warns.append(f"{rel(page)}: незакрытый <table> — ломает вёрстку следующих блоков")
    report(errs, warns, f"ссылок проверено: {total} (страниц: {len(pages)}), разметка кода в порядке")


def check_portal_data():
    section("Данные портала (index.html)")
    index = ROOT / "index.html"
    if not index.exists():
        return
    raw = index.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (\{.*?\});\nconst STORE_KEY", raw, re.S)
    if not match:
        report(["не удалось найти данные в index.html — пересоберите портал"], [], "")
        return
    data = json.loads(match.group(1).replace("<\\/", "</"))
    errs = []
    exists = lambda path: (ROOT / path).exists()
    for f in data["titles"]:
        if not exists(f"cheatsheets/{f}"):
            errs.append(f"тема {f}: нет файла")
    topics = set(data["titles"])
    for part in data["parts"]:
        for s in part["sections"]:
            errs += [f"раздел «{s['title']}»: неизвестная тема {f}" for f in s["items"] if f not in topics]
    for p in data["paths"]:
        for st in p["stages"]:
            errs += [f"путь «{p['title']}»: неизвестная тема {f}" for f in st["items"] if f not in topics]
            errs += [f"путь «{p['title']}»: нет упражнения {e}" for e in st.get("exercises", []) if not exists(f"exercises/{e}")]
    for e in data["exercises"]:
        errs += [f"упражнение {e['id']}: нет файла {n}" for n in e["files"] if not exists(f"{e['id']}/{n}")]
    for cs, quiz in data.get("quizzes", {}).items():
        if not exists(f"cheatsheets/{quiz}"):
            errs.append(f"тест {quiz}: нет файла")
    for d in data.get("demos", []):
        if not exists(f"interactive/{d['file']}"):
            errs.append(f"интерактив {d['file']}: нет файла")
        errs += [f"интерактив {d['file']}: неизвестная тема {t}" for t in d["topics"] if t not in topics]
    errs += [f"вопросы к {f}: неизвестная тема" for f in data.get("questions", {}) if f not in topics]
    report(errs, [], f"все ссылки портала ведут на существующие темы и файлы (тем: {len(topics)})")


def check_questions_and_demos():
    section("Вопросы и интерактивы")
    errs, warns = [], []
    q_files = [p for p in (ROOT / "questions").glob("*.md") if p.name.lower() != "readme.md"]
    n_questions = 0
    for md in q_files:
        if not (CHEATSHEETS / f"{md.stem}_cheatsheet.html").exists():
            errs.append(f"{rel(md)}: нет шпаргалки {md.stem}_cheatsheet.html")
        count = sum(1 for line in md.read_text(encoding="utf-8").splitlines() if line.startswith("### "))
        n_questions += count
        if count == 0:
            errs.append(f"{rel(md)}: нет ни одного вопроса (заголовков «### »)")
        else:
            for block in md.read_text(encoding="utf-8").split("\n### ")[1:]:
                if not block.split("\n", 1)[-1].strip() or "\n" not in block:
                    errs.append(f"{rel(md)}: у вопроса «{block.splitlines()[0][:50]}» нет ответа")
    demos = sorted((ROOT / "interactive").glob("*.html"))
    for page in demos:
        raw = page.read_text(encoding="utf-8")
        meta = re.search(r'<meta name="mlcs-topics" content="([^"]*)"', raw)
        if not meta:
            errs.append(f"{rel(page)}: нет <meta name=\"mlcs-topics\">")
            continue
        for stem in meta.group(1).split():
            if not (CHEATSHEETS / f"{stem}_cheatsheet.html").exists():
                errs.append(f"{rel(page)}: тема «{stem}» — нет такой шпаргалки")
        for asset in ["../assets/theme.js", "lab.js", "lab.css"]:
            if asset not in raw:
                warns.append(f"{rel(page)}: не подключён {asset}")
    report(errs, warns, f"вопросов: {n_questions} (файлов: {len(q_files)}), интерактивов: {len(demos)} — все темы на месте")


def check_exercises(run_solutions):
    section("Упражнения" + ("" if run_solutions else " (без запуска решений)"))
    from check_exercise import compare
    import pandas as pd

    errs, warns, checked = [], [], 0
    tasks = sorted((ROOT / "exercises").rglob("task.md"))
    for task in tasks:
        folder = task.parent
        for need in ["input.csv", "answer.csv"]:
            if not (folder / need).exists():
                errs.append(f"{rel(folder)}: нет {need}")
        solution = folder / "solution.py"
        if not run_solutions or not solution.exists() or not (folder / "answer.csv").exists():
            continue
        # решение запускаем во временной копии папки, чтобы не трогать файлы упражнения
        with tempfile.TemporaryDirectory() as tmp:
            for f in folder.iterdir():
                if f.is_file() and f.name not in {"my_answer.csv", "solution_output.csv"}:
                    shutil.copy(f, tmp)
            run = subprocess.run([sys.executable, "solution.py"], cwd=tmp, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
            out = Path(tmp) / "solution_output.csv"
            if run.returncode != 0:
                errs.append(f"{rel(solution)}: падает — {run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'код ' + str(run.returncode)}")
            elif not out.exists():
                warns.append(f"{rel(solution)}: не создаёт solution_output.csv — сравнить с эталоном нельзя")
            else:
                problems = compare(pd.read_csv(folder / "answer.csv"), pd.read_csv(out))
                if problems:
                    errs.append(f"{rel(solution)}: результат не совпадает с answer.csv — {problems[0]}")
                else:
                    checked += 1
    msg = f"упражнений: {len(tasks)}, файлы на месте" + (f"; решений совпадает с эталоном: {checked}" if run_solutions else "")
    report(errs, warns, msg)


def check_portal_fresh():
    section("Портал index.html")
    index = ROOT / "index.html"
    if not index.exists():
        report(["index.html не собран — запустите python tools/build_portal.py"], [], "")
        return
    sources = [ROOT / "README.md", ROOT / "ROADMAP.md", ROOT / "tools" / "portal_template.html"]
    sources += list((ROOT / "exercises").rglob("task.md")) + list((ROOT / "questions").glob("*.md"))
    sources += list((ROOT / "interactive").glob("*.html"))
    newer = [rel(p) for p in sources if p.stat().st_mtime > index.stat().st_mtime]
    warns = [f"index.html старше {len(newer)} исходных файлов (например, {newer[0]}) — пересоберите: python tools/build_portal.py"] if newer else []
    report([], warns, "собран после последних изменений")


def main():
    parser = argparse.ArgumentParser(description="Проверка целостности проекта")
    parser.add_argument("--fast", action="store_true", help="не запускать solution.py упражнений")
    args = parser.parse_args()

    check_cheatsheet_links()
    check_markdown_links()
    check_theme()
    check_pages()
    check_questions_and_demos()
    check_exercises(run_solutions=not args.fast)
    check_portal_fresh()
    check_portal_data()

    print()
    if errors:
        print(f"Найдено ошибок: {len(errors)}, предупреждений: {len(warnings)}")
        return 1
    print(f"✅ Всё в порядке{f' (предупреждений: {len(warnings)})' if warnings else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
