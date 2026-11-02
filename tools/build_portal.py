#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Сборка учебного портала index.html.

Читает README.md (разделы и шпаргалки), ROADMAP.md (учебные пути)
и папку exercises/ (упражнения), а затем подставляет эти данные
в шаблон tools/portal_template.html.

Запуск из корня репозитория:
    python tools/build_portal.py

Перезапускайте скрипт после добавления новых шпаргалок или упражнений.
"""

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHEATSHEETS = ROOT / "cheatsheets"
EXERCISES = ROOT / "exercises"
QUESTIONS = ROOT / "questions"
INTERACTIVE = ROOT / "interactive"
TEMPLATE = Path(__file__).resolve().parent / "portal_template.html"
OUTPUT = ROOT / "index.html"

LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*?cheatsheets/([\w\-]+\.html)\)")
LOCAL_FILES = {"my_answer.csv", "solution_output.csv", "answer.key"}  # личные файлы ученика, в git не попадают
LEVEL_ICONS = {"🟢": "beginner", "🟡": "intermediate", "🔴": "advanced"}


def clean_heading(text):
    """Убирает markdown-разметку и номер из заголовка раздела."""
    text = text.replace("*", "").strip()
    text = re.sub(r"^\d+\.\s*", "", text)
    return text.replace("🎨", "").strip()


def cheatsheet_title(file_name):
    """Берёт заголовок из <title> шпаргалки (для тем, которых нет в README)."""
    raw = (CHEATSHEETS / file_name).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"<title>(.*?)</title>", raw, re.S | re.I)
    if not match:
        return file_name.replace("_cheatsheet.html", "").replace("_", " ")
    title = html.unescape(match.group(1)).strip()
    return re.sub(r"\s*(Cheatsheet|cheatsheet)?\s*(—|-)\s*\d\s*колонк\w*$", "", title).strip()


def parse_readme():
    """Части (I, II, III) -> разделы -> шпаргалки."""
    parts, part, section = [], None, None
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and re.match(r"## (I|II|III) ", line):
            part = {"title": line[3:].strip(), "sections": []}
            parts.append(part)
            section = None
        elif line.startswith("## "):
            part = section = None
        elif part is not None and line.startswith("### "):
            section = {"title": clean_heading(line[4:]), "items": []}
            part["sections"].append(section)
        elif section is not None:
            for title, file_name in LINK_RE.findall(line):
                if (CHEATSHEETS / file_name).exists():
                    section["items"].append(file_name)
    return parts


def parse_roadmap(titles):
    """Учебные пути из ROADMAP.md: путь -> этапы -> шпаргалки."""
    paths, path, stage = [], None, None
    for line in (ROOT / "ROADMAP.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("### ") and "Путь" in line:
            path = {"title": line[4:].strip(), "stages": []}
            paths.append(path)
            stage = None
        elif line.startswith("## ") or (line.startswith("### ") and path):
            path = stage = None
        elif path is not None and line.startswith("#### "):
            heading = line[5:].strip()
            level = next((v for k, v in LEVEL_ICONS.items() if k in heading), "")
            for icon in LEVEL_ICONS:
                heading = heading.replace(icon, "")
            stage = {"title": heading.strip(" -"), "level": level, "items": [], "notes": [], "exercises": []}
            path["stages"].append(stage)
        elif stage is not None:
            links = LINK_RE.findall(line)
            for title, file_name in links:
                if (CHEATSHEETS / file_name).exists():
                    stage["items"].append(file_name)
                    titles.setdefault(file_name, title)
            text = re.sub(r"^\s*\d+\.\s*", "", line).strip()
            if not links and text and not text.startswith("-") and not text.endswith(":"):
                text = text.replace("**", "")
                # «Упражнения: linreg, logreg» -> ссылки на папки exercises/<имя>
                named = [w.strip() for w in text.split(":", 1)[-1].split(",")]
                if text.startswith("Упражнения:") and all((EXERCISES / w).is_dir() for w in named):
                    stage["exercises"] = named
                else:
                    stage["notes"].append(text)
    return paths


def first_heading(md_text, fallback):
    match = re.search(r"^#\s+(.+)$", md_text, re.M)
    return match.group(1).strip() if match else fallback


def parse_projects():
    """projects/<имя>/: project.json (метрика, пороги, номер пути) + README.md (задание)."""
    projects = []
    for cfg_path in sorted((ROOT / "projects").glob("*/project.json")):
        folder = cfg_path.parent
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        hidden = {"answer.csv"} | LOCAL_FILES | {"my_predictions.csv", "solution_predictions.csv"}
        projects.append({
            "id": folder.relative_to(ROOT).as_posix(), "name": folder.name, "title": cfg["title"], "path": cfg.get("path"),
            "metric": cfg["metric"], "higher": cfg["higher_is_better"], "thresholds": cfg["thresholds"],
            "files": sorted(p.name for p in folder.iterdir() if p.is_file() and p.name not in hidden and p.name != "project.json"),
            "readme": (folder / "README.md").read_text(encoding="utf-8"),
        })
    return projects


def split_task(md):
    """task.md → (условие без подсказок и разбора, [подсказки по одной], разбор)."""
    md = re.sub(r"\A\s*# .*\n", "", md)                # заголовок упражнения портал показывает сам
    sections = re.split(r"^(?=## )", md, flags=re.M)
    body, hints, review = [], [], ""
    for sec in sections:
        head, _, rest = sec.partition("\n")
        if head.strip() == "## Подсказки":
            hints = [re.sub(r"\s*\n\s*", " ", h).strip() for h in re.split(r"^- ", rest, flags=re.M) if h.strip()]
        elif head.strip() == "## Задание":
            body.append(rest)                         # заголовок «📝 Задание» уже есть в портале
        elif head.strip() == "## Проверка":
            continue                                  # в портале свой блок самопроверки
        elif head.strip() == "## Разбор":
            review = "\n".join(l for l in rest.splitlines() if not l.startswith("> Читайте после")).strip()
        else:
            body.append(sec)
    return "".join(body).rstrip() + "\n", hints, review


def parse_exercises():
    """Каждая папка с task.md — одно упражнение. В task.md можно указать <!-- шпаргалка: имя -->."""
    exercises = []
    for task in sorted(EXERCISES.rglob("task.md")):
        folder = task.parent
        rel = folder.relative_to(ROOT).as_posix()
        topic = folder.relative_to(EXERCISES).parts[0]
        md = task.read_text(encoding="utf-8")
        meta = re.search(r"<!--\s*шпаргалка:\s*([\w-]+)\s*-->", md)
        cheatsheet = f"{meta.group(1) if meta else topic}_cheatsheet.html"
        body, hints, review = split_task(re.sub(r"<!--.*?-->\n?", "", md))
        exercises.append({
            "id": rel,
            "title": first_heading(md, folder.name).replace("Упражнение:", "").strip(),
            "topic": topic,
            "kind": "bug" if topic.startswith("bug_") else "calc",
            "cheatsheet": cheatsheet if (CHEATSHEETS / cheatsheet).exists() else None,
            "files": sorted(p.name for p in folder.iterdir() if p.is_file() and p.name not in LOCAL_FILES),
            "task": body, "hints": hints, "review": review,
        })
    return exercises


def parse_questions():
    """questions/<тема>.md: каждый заголовок «### » — вопрос, текст под ним — ответ."""
    questions = {}
    for md in sorted(QUESTIONS.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        file_name = f"{md.stem}_cheatsheet.html"
        if not (CHEATSHEETS / file_name).exists():
            print(f"  ⚠ {md.relative_to(ROOT).as_posix()}: нет шпаргалки {file_name}, пропущено")
            continue
        items, current = [], None
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):
                current = {"q": line[4:].strip(), "a": []}
                items.append(current)
            elif current is not None:
                current["a"].append(line)
        for item in items:
            item["a"] = "\n".join(item["a"]).strip()
        if items:
            questions[file_name] = items
    return questions


def parse_glossary():
    """glossary.md: «### Термин (перевод)», абзац определения, строка «→ стем1, стем2» → ([термины], [ошибки])."""
    terms, errors = [], []
    path = ROOT / "glossary.md"
    if not path.exists():
        return terms, errors
    current = None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## ") or line.startswith("# "):
            current = None
        elif line.startswith("### "):
            current = {"t": line[4:].strip(), "d": [], "topics": []}
            terms.append(current)
        elif current is not None and line.startswith("→"):
            for stem in [x.strip() for x in line[1:].split(",") if x.strip()]:
                if (CHEATSHEETS / f"{stem}_cheatsheet.html").exists():
                    current["topics"].append(f"{stem}_cheatsheet.html")
                else:
                    errors.append(f"glossary.md:{n}: нет шпаргалки «{stem}»")
        elif current is not None:
            current["d"].append(line)
    for t in terms:
        t["d"] = "\n".join(t["d"]).strip()
        if not t["d"]:
            errors.append(f"glossary.md: у термина «{t['t']}» нет определения")
    seen = set()
    for t in terms:
        if t["t"].lower() in seen:
            errors.append(f"glossary.md: термин «{t['t']}» повторяется")
        seen.add(t["t"].lower())
    return sorted(terms, key=lambda t: t["t"].lower()), errors


UNDERSTANDING = {"predict": "predict.md", "case": "cases.md"}


def parse_understanding():
    """understanding/*.md: «## Вопрос», <!-- демо: файл | темы: стемы -->, условие, варианты «- …» (верный с ✅), «**Почему.** …»."""
    tasks, errors = [], []
    for kind, name in UNDERSTANDING.items():
        path = ROOT / "understanding" / name
        if not path.exists():
            continue
        for n, chunk in enumerate(re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)[1:], 1):
            head, _, rest = chunk.partition("\n")
            where = f"understanding/{name}: «{head.strip()}»"
            meta = re.search(r"<!--(.*?)-->", rest)
            fields = dict(re.findall(r"(демо|темы):\s*([^|]+)", meta.group(1))) if meta else {}
            demo = fields.get("демо", "").strip() or None
            if demo and not (INTERACTIVE / demo).exists():
                errors.append(f"{where}: нет интерактива {demo}")
                demo = None
            topics = []
            for stem in [s.strip() for s in fields.get("темы", "").split(",") if s.strip()]:
                if (CHEATSHEETS / f"{stem}_cheatsheet.html").exists():
                    topics.append(f"{stem}_cheatsheet.html")
                else:
                    errors.append(f"{where}: нет шпаргалки «{stem}»")
            body = re.sub(r"<!--.*?-->\n?", "", rest)
            question, _, tail = body.partition("\n- ")
            options_text, _, why = ("- " + tail).partition("**Почему.**")
            options = [o.strip() for o in re.split(r"^- ", options_text, flags=re.M) if o.strip()]
            answer = [i for i, o in enumerate(options) if o.endswith("✅")]
            if len(options) < 2 or len(answer) != 1 or not why.strip():
                errors.append(f"{where}: нужно ≥ 2 вариантов, ровно один с ✅ и разбор «**Почему.**»")
                continue
            tasks.append({
                "id": f"{kind}:{n}", "kind": kind, "q": head.strip(), "text": question.strip(),
                "options": [o.rstrip("✅ ").strip() for o in options], "answer": answer[0],
                "why": why.strip(), "demo": demo, "topics": topics,
            })
    return tasks, errors


def parse_prerequisites():
    """prerequisites.md: строки «- тема: пред1, пред2» → ({файл темы: [файлы-предпосылки]}, [ошибки])."""
    deps, errors = {}, []
    path = ROOT / "prerequisites.md"
    if not path.exists():
        return deps, errors
    exists = lambda stem: (CHEATSHEETS / f"{stem}_cheatsheet.html").exists()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r"^- ([\w-]+):\s*(.+)$", line.strip())
        if not m:
            continue
        topic, before = m.group(1), [x.strip() for x in m.group(2).split(",") if x.strip()]
        for stem in [topic] + before:
            if not exists(stem):
                errors.append(f"prerequisites.md:{n}: нет шпаргалки «{stem}»")
        key = f"{topic}_cheatsheet.html"
        deps.setdefault(key, [])
        deps[key] += [f"{b}_cheatsheet.html" for b in before if exists(b) and f"{b}_cheatsheet.html" not in deps[key]]
    # циклы: поиск в глубину с цветами
    state = {}
    def visit(node, trail):
        state[node] = "в пути"
        for nxt in deps.get(node, []):
            if state.get(nxt) == "в пути":
                cycle = trail[trail.index(nxt):] + [nxt] if nxt in trail else [node, nxt]
                errors.append("цикл в prerequisites.md: " + " → ".join(c.replace("_cheatsheet.html", "") for c in cycle))
            elif nxt not in state:
                visit(nxt, trail + [nxt])
        state[node] = "готово"
    for node in sorted(deps):
        if node not in state:
            visit(node, [node])
    return deps, errors


def parse_demos():
    """interactive/*.html: темы берутся из <meta name="mlcs-topics" content="linreg knn ...">."""
    demos = []
    for page in sorted(INTERACTIVE.glob("*.html")):
        raw = page.read_text(encoding="utf-8")
        meta = re.search(r'<meta name="mlcs-topics" content="([^"]*)"', raw)
        title = re.search(r"<title>(.*?)</title>", raw, re.S)
        if not meta:
            print(f"  ⚠ {page.relative_to(ROOT).as_posix()}: нет <meta name=\"mlcs-topics\">, пропущено")
            continue
        topics = []
        for stem in meta.group(1).split():
            file_name = f"{stem}_cheatsheet.html"
            if (CHEATSHEETS / file_name).exists():
                topics.append(file_name)
            else:
                print(f"  ⚠ {page.relative_to(ROOT).as_posix()}: нет шпаргалки {file_name}")
        demos.append({
            "file": page.name,
            "title": html.unescape(title.group(1)).split(" — ")[0].strip() if title else page.stem,
            "topics": topics,
        })
    return demos


def build_version():
    """Метка сборки для адресов файлов в портале (…/x.html?v=метка) — сбрасывает старый кэш браузера.
    Зависит от времени изменения материалов, поэтому повторная сборка без изменений даёт ту же метку."""
    sources = [*CHEATSHEETS.glob("*.html"), *INTERACTIVE.glob("*"), *(ROOT / "assets").glob("*"), *EXERCISES.rglob("*.*")]
    return format(int(max(p.stat().st_mtime for p in sources)), "x")


def collect_titles(parts):
    titles = {}
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        for title, file_name in LINK_RE.findall(line):
            titles.setdefault(file_name, title.strip())
    return titles


def main():
    parts = parse_readme()
    titles = collect_titles(parts)
    paths = parse_roadmap(titles)
    exercises = parse_exercises()
    projects = parse_projects()

    listed = {f for p in parts for s in p["sections"] for f in s["items"]}
    orphans = sorted(
        f.name for f in CHEATSHEETS.glob("*.html")
        if f.name not in listed and f.name.endswith("_cheatsheet.html")
    )
    if orphans:
        parts.append({
            "title": "Прочие шпаргалки (нет в README)",
            "sections": [{"title": "Без раздела", "items": orphans}],
        })
    for file_name in sorted({f for p in parts for s in p["sections"] for f in s["items"]}):   # sorted — сборка детерминирована
        if file_name not in titles:
            titles[file_name] = cheatsheet_title(file_name)

    # Тесты вида <тема>_test.html привязываем к шпаргалке <тема>_cheatsheet.html
    quizzes = {
        q.name.replace("_test.html", "_cheatsheet.html"): q.name
        for q in CHEATSHEETS.glob("*_test.html")
        if (CHEATSHEETS / q.name.replace("_test.html", "_cheatsheet.html")).exists()
    }

    questions = parse_questions()
    demos = parse_demos()
    prereqs, prereq_errors = parse_prerequisites()
    for e in prereq_errors:
        print(f"  ⚠ {e}")
    glossary, glossary_errors = parse_glossary()
    for e in glossary_errors:
        print(f"  ⚠ {e}")
    understanding, understanding_errors = parse_understanding()
    for e in understanding_errors:
        print(f"  ⚠ {e}")

    data = {"parts": parts, "paths": paths, "exercises": exercises, "titles": titles,
            "quizzes": quizzes, "questions": questions, "demos": demos, "prereqs": prereqs, "projects": projects,
            "glossary": glossary, "understanding": understanding, "build": build_version()}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    OUTPUT.write_text(page, encoding="utf-8")

    print(f"Готово: {OUTPUT.relative_to(ROOT)}")
    print(f"  тем: {len(titles)}, разделов: {sum(len(p['sections']) for p in parts)}, "
          f"путей: {len(paths)}, упражнений: {len(exercises)}, проектов: {len(projects)}, тестов: {len(quizzes)}, "
          f"вопросов: {sum(map(len, questions.values()))} к {len(questions)} темам, "
          f"интерактивов: {len(demos)}, тем со связями: {len(prereqs)}, терминов: {len(glossary)}, задач на понимание: {len(understanding)}, вне README: {len(orphans)}")


if __name__ == "__main__":
    main()
