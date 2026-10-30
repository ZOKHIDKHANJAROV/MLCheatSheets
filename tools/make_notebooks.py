#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Создаёт тетрадку Jupyter exercise.ipynb в каждой папке упражнения (exercises/**/task.md).

В тетрадке: условие задачи, загрузка input.csv, ячейка для решения и ячейка проверки,
которая сравнивает ваш результат с answer.csv (та же логика, что в tools/check_exercise.py).

Запуск из корня репозитория:
    python tools/make_notebooks.py           # создать тетрадки там, где их ещё нет
    python tools/make_notebooks.py --force   # пересоздать все (ваши решения в тетрадках пропадут!)
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "exercise.ipynb"


def lines(text):
    """Текст ячейки в формате ipynb: список строк с переводами строк."""
    parts = text.strip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(text)}


SETUP = """
import pandas as pd
import numpy as np

df = pd.read_csv('input.csv')
print(df.shape)
df.head()
"""

SOLUTION = """
# Ваше решение. Итог запишите в DataFrame `result` — в формате, описанном в задании.

result = None
"""

CHECK = """
# Проверка: сравнение result с эталоном answer.csv (запускайте после своего решения)
import sys
from pathlib import Path

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'tools' / 'check_exercise.py').exists())
sys.path.insert(0, str(root / 'tools'))
from check_exercise import compare

if result is None:
    print('Сначала запишите решение в переменную result (ячейка выше).')
else:
    result.to_csv('my_answer.csv', index=False)   # тот же файл проверяет и python tools/check_exercise.py
    problems = compare(pd.read_csv('answer.csv'), pd.read_csv('my_answer.csv'))
    print('✅ Всё верно!' if not problems else '❌ Есть расхождения:\\n' + '\\n'.join('  • ' + p for p in problems))
"""


def notebook(folder):
    task = (folder / "task.md").read_text(encoding="utf-8")
    cells = [
        md(task),
        md("## Данные"),
        code(SETUP),
        md("## Решение"),
        code(SOLUTION),
        md("## Проверка"),
        code(CHECK),
    ]
    if (folder / "solution.py").exists():
        cells.append(md("---\nЭталонное решение с пояснениями — в файле `solution.py`. Загляните в него после своей попытки."))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description="Тетрадки Jupyter для упражнений")
    parser.add_argument("--force", action="store_true", help="перезаписать существующие тетрадки")
    args = parser.parse_args()

    made = skipped = 0
    for task in sorted((ROOT / "exercises").rglob("task.md")):
        target = task.parent / NAME
        if target.exists() and not args.force:
            skipped += 1
            continue
        nb = notebook(task.parent)
        for i, cell in enumerate(nb["cells"]):
            cell["id"] = f"cell-{i}"
        target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        made += 1
    print(f"Тетрадок создано: {made}, уже было: {skipped}")


if __name__ == "__main__":
    main()
