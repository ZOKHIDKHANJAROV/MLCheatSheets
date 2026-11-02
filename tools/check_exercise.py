#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Самопроверка упражнений.

Сравнивает ваш результат с эталоном, показывает, какие столбцы и строки отличаются,
и подсказывает типичную причину ошибки (порядок строк, округление, ddof, проценты, сдвиг…).
Верное решение записывается в my_progress.json — портал (python start.py) сам отметит упражнение.

Использование (из корня репозитория):
    python tools/check_exercise.py exercises/scaling_normalization
    python tools/check_exercise.py exercises/scaling_normalization путь/к/моему.csv
    python tools/check_exercise.py --list

По умолчанию ваш файл ищется как my_answer.csv в папке упражнения.
Эталон лежит в answer.key в закодированном виде — чтобы не подсмотреть его случайно.
"""

import argparse
import base64
import io
import json
import math
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "exercises"
PROGRESS = ROOT / "my_progress.json"
TOLERANCE = 1e-3  # допустимая погрешность для чисел (ответы округлены до 3-4 знаков)
KEY_NAME = "answer.key"
KEY_HEADER = ("# Эталонный ответ упражнения в закодированном виде (zlib + base64), чтобы не подсмотреть случайно.\n"
              "# Проверка решения: python tools/check_exercise.py <папка упражнения>\n")


# ---------------------------------------------------------------- эталон

def encode_reference(csv_text):
    return KEY_HEADER + base64.b64encode(zlib.compress(csv_text.encode("utf-8"), 9)).decode("ascii") + "\n"


def save_reference(folder, source):
    """Записывает эталон в folder/answer.key; source — DataFrame или путь к CSV (файл удаляется)."""
    folder = Path(folder)
    if isinstance(source, pd.DataFrame):
        text = source.to_csv(index=False)
    else:
        text = Path(source).read_text(encoding="utf-8")
        Path(source).unlink()
    (folder / KEY_NAME).write_text(encode_reference(text), encoding="utf-8")
    old = folder / "answer.csv"
    if old.exists():
        old.unlink()


def reference_path(folder):
    folder = Path(folder)
    return folder / KEY_NAME if (folder / KEY_NAME).exists() else folder / "answer.csv"


def load_reference(folder):
    path = reference_path(folder)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    payload = "".join(l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#"))
    return pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(payload)).decode("utf-8")))


# ---------------------------------------------------------------- сравнение

def _bad_mask(exp, act):
    if pd.api.types.is_numeric_dtype(exp) and pd.api.types.is_numeric_dtype(act):
        return ~np.isclose(exp.astype(float), act.astype(float), atol=TOLERANCE, equal_nan=True)
    return (exp.astype(str).str.strip() != act.astype(str).str.strip()).to_numpy()


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
        exp = expected[col].iloc[:rows].reset_index(drop=True)
        act = actual[col].iloc[:rows].reset_index(drop=True)
        bad = _bad_mask(exp, act)
        if bad.any():
            idx = list(np.flatnonzero(bad))
            examples = ", ".join(f"строка {i}: ожидалось {exp[i]}, у вас {act[i]}" for i in idx[:3])
            problems.append(f"Столбец '{col}': {len(idx)} несовпадений ({examples}{', …' if len(idx) > 3 else ''})")
    return problems


def _const(values, rel=1e-3):
    """Если все значения почти одинаковы — вернуть это значение, иначе None."""
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return None
    m = float(np.median(v))
    return m if np.allclose(v, m, rtol=rel, atol=TOLERANCE) else None


def _numeric_hints(col, exp, act, n_rows):
    """Типичные причины расхождения в числовом столбце."""
    e, a = exp.astype(float).to_numpy(), act.astype(float).to_numpy()
    if np.isnan(a).sum() > np.isnan(e).sum():
        return [f"'{col}': у вас NaN в {int(np.isnan(a).sum() - np.isnan(e).sum())} строках — пропуски не обработаны или деление на ноль."]
    ok = np.isfinite(e) & np.isfinite(a)
    e, a = e[ok], a[ok]
    if len(e) == 0:
        return []
    if np.allclose(a, -e, atol=TOLERANCE):
        return [f"'{col}': значения совпадают с точностью до знака — проверьте направление разности или знак градиента."]
    if set(np.unique(e)) <= {0, 1} and set(np.unique(a)) <= {0, 1} and np.array_equal(a, 1 - e):
        return [f"'{col}': классы перепутаны (0↔1) — проверьте, какой класс положительный и сторону порога."]
    shift = _const(a - e)
    if shift is not None and abs(shift) > TOLERANCE:
        tip = f"'{col}': все ваши значения отличаются от эталона на {shift:+.4g}"
        if abs(abs(shift) - 1) < 1e-9:
            tip += " — нумерация с 0 вместо 1 (или наоборот)?"
        else:
            tip += " — забыли вычесть/прибавить константу (среднее, intercept, смещение)?"
        return [tip]
    if len(e) > 2 and (np.allclose(a[1:], e[:-1], atol=TOLERANCE) or np.allclose(a[:-1], e[1:], atol=TOLERANCE)):
        return [f"'{col}': значения сдвинуты на одну строку — проверьте shift/lag и с какого шага начинается расчёт."]
    nz = np.abs(e) > 1e-12
    single = nz.sum() == 1       # одно число (метрика) — постоянство отношения не проверить, ищем только известные множители
    ratio = float(a[nz][0] / e[nz][0]) if single else _const(a[nz] / e[nz]) if nz.sum() >= 2 else None
    if ratio is not None and abs(ratio - 1) > 1e-3:
        known = [
            (100, "в процентах, а нужно в долях"), (0.01, "в долях, а нужно в процентах"),
            (math.log(10), "натуральный логарифм ln вместо log10"), (1 / math.log(10), "log10 вместо натурального логарифма ln"),
            (math.log(2), "ln вместо log2"), (1 / math.log(2), "log2 вместо натурального логарифма ln"),
        ]
        for n in sorted({n_rows, n_rows - 1, n_rows + 1} - {0, 1}):
            known += [(math.sqrt(n / (n - 1)), "std/дисперсия с ddof=1, а нужно ddof=0 (у NumPy по умолчанию 0, у pandas — 1)"),
                      (math.sqrt((n - 1) / n), "std с ddof=0, а нужно ddof=1"),
                      (n / (n - 1), "дисперсия с ddof=1, а нужно ddof=0"), ((n - 1) / n, "дисперсия с ddof=0, а нужно ddof=1")]
        for value, why in known:
            if abs(ratio / value - 1) < 2e-3:
                return [f"'{col}': ваши значения в {ratio:.4g} раза больше эталона — похоже, {why}."]
        if not single:
            return [f"'{col}': все ваши значения отличаются в {ratio:.4g} раза — проверьте нормировку (деление на n, масштаб, единицы)."]
    if np.allclose(np.round(a, 1), np.round(e, 1), atol=0.051) or np.allclose(a, e, rtol=0.01):
        return [f"'{col}': значения близки, но не совпадают (макс. разница {np.max(np.abs(a - e)):.3g}) — "
                "проверьте округление (в конце, а не в промежуточных шагах), параметры модели и random_state."]
    return []


def diagnose(expected, actual):
    """Подсказки о вероятной причине ошибки — по типичным промахам."""
    tips = []
    if any(str(c).startswith("Unnamed") for c in actual.columns):
        tips.append("В файле есть безымянный столбец индекса — сохраняйте так: df.to_csv('my_answer.csv', index=False).")
    norm = {str(c).strip().lower(): c for c in expected.columns}
    for c in actual.columns:
        if c not in expected.columns and str(c).strip().lower() in norm:
            tips.append(f"Столбец '{c}' назван иначе, чем в задании ('{norm[str(c).strip().lower()]}') — регистр или пробелы.")
    common = [c for c in expected.columns if c in actual.columns]

    if len(actual) != len(expected):
        if len(actual) == len(expected) - 1 and list(actual.columns) == [str(x) for x in expected.iloc[0].tolist()]:
            tips.append("Первая строка данных стала заголовком — файл сохранён с header=False?")
        elif len(actual) > len(expected):
            tips.append("Строк больше, чем нужно: возможно, в ответ попали обучающие строки (нужны только test) или дубликаты после merge.")
        else:
            tips.append("Строк меньше, чем нужно: возможно, удалены строки с пропусками или выбросами, хотя по заданию их надо обработать иначе.")
        return tips

    if not common:
        return tips
    exp, act = expected[common].reset_index(drop=True), actual[common].reset_index(drop=True)
    bad_cols = [c for c in common if _bad_mask(exp[c], act[c]).any()]
    if not bad_cols:
        return tips

    # те же строки, но в другом порядке
    try:
        key = common
        if compare(exp.sort_values(key).reset_index(drop=True), act.sort_values(key).reset_index(drop=True)) == []:
            tips.append("Все строки верные, но порядок другой — сохраните исходный порядок строк (или как сказано в задании).")
            return tips
    except TypeError:
        pass

    # столбцы перепутаны местами
    for c in bad_cols:
        for other in common:
            if other != c and pd.api.types.is_numeric_dtype(act[c]) == pd.api.types.is_numeric_dtype(exp[other]) \
                    and not _bad_mask(exp[other], act[c]).any():
                tips.append(f"В столбце '{c}' у вас значения, которые должны быть в '{other}' — столбцы перепутаны.")
                break

    for c in bad_cols:
        if pd.api.types.is_numeric_dtype(exp[c]) and pd.api.types.is_numeric_dtype(act[c]):
            tips += _numeric_hints(c, exp[c], act[c], len(exp))
        elif pd.api.types.is_numeric_dtype(exp[c]) and not pd.api.types.is_numeric_dtype(act[c]):
            tips.append(f"'{c}': ожидаются числа, а у вас текст — проверьте кодирование категорий и разделитель дробной части.")
        else:
            e, a = exp[c].astype(str).str.strip(), act[c].astype(str).str.strip()
            if (e.str.lower() == a.str.lower()).all():
                tips.append(f"'{c}': отличается только регистр букв.")
    share = 1 - sum(_bad_mask(exp[c], act[c]).sum() for c in common) / (len(exp) * len(common))
    tips.append(f"Совпадает {share:.0%} значений.")
    return tips


# ---------------------------------------------------------------- прогресс для портала

def record_success(folder):
    """Отмечает упражнение решённым в my_progress.json — портал прочитает его при запуске через start.py."""
    try:
        exercise_id = Path(folder).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return
    data = {}
    if PROGRESS.exists():
        try:
            data = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    solved = data.setdefault("exercises", {})
    entry = solved.setdefault(exercise_id, {"first": datetime.now().isoformat(timespec="seconds"), "count": 0})
    entry["last"] = datetime.now().isoformat(timespec="seconds")
    entry["count"] += 1
    PROGRESS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def report(folder, actual):
    """Проверяет DataFrame actual и печатает результат. Возвращает True, если всё верно."""
    expected = load_reference(folder)
    problems = compare(expected, actual)
    if problems:
        print(f"❌ Есть расхождения с эталоном ({len(problems)}):")
        for p in problems:
            print("   •", p)
        tips = diagnose(expected, actual)
        if tips:
            print("\n🔎 Возможная причина:")
            for t in tips:
                print("   →", t)
        print("\nЕщё: подсказки в task.md (раздел «Подсказки») или по одной в портале.")
        return False
    record_success(folder)
    print("✅ Всё верно! Портал (python start.py) отметит упражнение решённым и откроет разбор.")
    return True


def check_result(result, folder="."):
    """Для тетрадок: сохраняет result в my_answer.csv и проверяет."""
    if result is None:
        print("Сначала запишите решение в переменную result (ячейка выше).")
        return False
    result.to_csv(Path(folder) / "my_answer.csv", index=False)
    return report(folder, pd.read_csv(Path(folder) / "my_answer.csv"))


# ---------------------------------------------------------------- командная строка

def list_exercises():
    solved = {}
    if PROGRESS.exists():
        try:
            solved = json.loads(PROGRESS.read_text(encoding="utf-8")).get("exercises", {})
        except ValueError:
            pass
    print("Доступные упражнения:")
    for task in sorted(EXERCISES.rglob("task.md")):
        folder = task.parent
        rel = folder.relative_to(ROOT).as_posix()
        mark = "✔" if rel in solved else ("…" if (folder / "my_answer.csv").exists() else " ")
        print(f"  [{mark}] {rel}")
    print("\n[✔] — решено, […] — есть my_answer.csv, но верного решения ещё не было")


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
    mine = Path(args.answer) if args.answer else folder / "my_answer.csv"

    if not reference_path(folder).exists():
        print(f"❌ Не найден эталон в {folder}. Проверьте путь к упражнению (--list покажет все).")
        return 2
    if not mine.exists():
        print(f"❌ Не найден ваш файл {mine}.")
        print("   Сохраните результат так:  df.to_csv('my_answer.csv', index=False)")
        return 2
    return 0 if report(folder, pd.read_csv(mine)) else 1


if __name__ == "__main__":
    sys.exit(main())
