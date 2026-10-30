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
TEMPLATE = Path(__file__).resolve().parent / "portal_template.html"
OUTPUT = ROOT / "index.html"

LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*?cheatsheets/([\w\-]+\.html)\)")
LOCAL_FILES = {"my_answer.csv", "solution_output.csv"}  # личные файлы ученика, в git не попадают
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


def parse_exercises():
    """Каждая папка с task.md — одно упражнение."""
    exercises = []
    for task in sorted(EXERCISES.rglob("task.md")):
        folder = task.parent
        rel = folder.relative_to(ROOT).as_posix()
        topic = folder.relative_to(EXERCISES).parts[0]
        md = task.read_text(encoding="utf-8")
        cheatsheet = f"{topic}_cheatsheet.html"
        exercises.append({
            "id": rel,
            "title": first_heading(md, folder.name).replace("Упражнение:", "").strip(),
            "topic": topic,
            "cheatsheet": cheatsheet if (CHEATSHEETS / cheatsheet).exists() else None,
            "files": sorted(p.name for p in folder.iterdir() if p.is_file() and p.name not in LOCAL_FILES),
            "task": md,
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
    for file_name in {f for p in parts for s in p["sections"] for f in s["items"]}:
        if file_name not in titles:
            titles[file_name] = cheatsheet_title(file_name)

    # Тесты вида <тема>_test.html привязываем к шпаргалке <тема>_cheatsheet.html
    quizzes = {
        q.name.replace("_test.html", "_cheatsheet.html"): q.name
        for q in CHEATSHEETS.glob("*_test.html")
        if (CHEATSHEETS / q.name.replace("_test.html", "_cheatsheet.html")).exists()
    }

    questions = parse_questions()

    data = {"parts": parts, "paths": paths, "exercises": exercises, "titles": titles,
            "quizzes": quizzes, "questions": questions}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    OUTPUT.write_text(page, encoding="utf-8")

    print(f"Готово: {OUTPUT.relative_to(ROOT)}")
    print(f"  тем: {len(titles)}, разделов: {sum(len(p['sections']) for p in parts)}, "
          f"путей: {len(paths)}, упражнений: {len(exercises)}, тестов: {len(quizzes)}, "
          f"вопросов: {sum(map(len, questions.values()))} к {len(questions)} темам, вне README: {len(orphans)}")


if __name__ == "__main__":
    main()
