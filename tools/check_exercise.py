#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Самопроверка упражнений.

Сравнивает ваш результат с эталонным answer.csv и показывает,
какие столбцы и строки отличаются.

Использование (из корня репозитория):
    python tools/check_exercise.py exercises/scaling_normalization
    python tools/check_exercise.py exercises/scaling_normalization путь/к/моему.csv
    python tools/check_exercise.py --list

По умолчанию ваш файл ищется как my_answer.csv в папке упражнения.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "exercises"
TOLERANCE = 1e-3  # допустимая погрешность для чисел (ответы округлены до 3-4 знаков)


def list_exercises():
    print("Доступные упражнения:")
    for task in sorted(EXERCISES.rglob("task.md")):
        folder = task.parent
        mark = "✔" if (folder / "my_answer.csv").exists() else " "
        print(f"  [{mark}] {folder.relative_to(ROOT).as_posix()}")
    print("\n[✔] — в папке уже есть ваш my_answer.csv")


def compare(expected, actual):
    """Возвращает список найденных проблем (пустой — значит всё верно)."""
    problems = []

    missing = [c for c in expected.columns if c not in actual.columns]
    extra = [c for c in actual.columns if c not in expected.columns]
    if missing:
        problems.append(f"Не хватает столбцов: {missing}")
    if extra:
        problems.append(f"Лишние столбцы: {extra}")
    if len(expected) != len(actual):
        problems.append(f"Число строк: ожидалось {len(expected)}, у вас {len(actual)}")

    rows = min(len(expected), len(actual))
    for col in [c for c in expected.columns if c in actual.columns]:
        exp, act = expected[col].iloc[:rows], actual[col].iloc[:rows].reset_index(drop=True)
        exp = exp.reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(exp) and pd.api.types.is_numeric_dtype(act):
            bad = ~np.isclose(exp.astype(float), act.astype(float), atol=TOLERANCE, equal_nan=True)
        else:
            bad = (exp.astype(str).str.strip() != act.astype(str).str.strip()).to_numpy()
        if bad.any():
            idx = list(np.flatnonzero(bad))
            examples = ", ".join(f"строка {i}: ожидалось {exp[i]}, у вас {act[i]}" for i in idx[:3])
            problems.append(f"Столбец '{col}': {len(idx)} несовпадений ({examples}{', …' if len(idx) > 3 else ''})")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Проверка решения упражнения")
    parser.add_argument("exercise", nargs="?", help="папка упражнения, например exercises/preprocessing")
    parser.add_argument("answer", nargs="?", help="ваш CSV (по умолчанию my_answer.csv в папке упражнения)")
    parser.add_argument("--list", action="store_true", help="показать список упражнений")
    args = parser.parse_args()

    if args.list or not args.exercise:
        list_exercises()
        return 0

    folder = Path(args.exercise)
    if not folder.is_absolute():
        folder = (Path.cwd() / folder) if (Path.cwd() / folder).exists() else ROOT / folder
    reference = folder / "answer.csv"
    mine = Path(args.answer) if args.answer else folder / "my_answer.csv"

    if not reference.exists():
        print(f"❌ Не найден эталон {reference}. Проверьте путь к упражнению (--list покажет все).")
        return 2
    if not mine.exists():
        print(f"❌ Не найден ваш файл {mine}.")
        print("   Сохраните результат так:  df.to_csv('my_answer.csv', index=False)")
        print("   (не перезаписывайте answer.csv — это эталон для проверки)")
        return 2

    problems = compare(pd.read_csv(reference), pd.read_csv(mine))
    if problems:
        print(f"❌ Есть расхождения с эталоном ({len(problems)}):")
        for p in problems:
            print("   •", p)
        print("\nПодсказка: перечитайте task.md — порядок столбцов, округление и порядок строк важны.")
        return 1
    print("✅ Всё верно! Отметьте упражнение как решённое на портале (index.html).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
