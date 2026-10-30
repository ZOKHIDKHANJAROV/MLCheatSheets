#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Проверка мини-проекта: считает метрику ваших прогнозов на скрытых метках и показывает уровень.

Использование (из корня репозитория):
    python tools/check_mini_project.py projects/churn_prediction
    python tools/check_mini_project.py projects/churn_prediction путь/к/прогнозам.csv
    python tools/check_mini_project.py --list

По умолчанию прогнозы берутся из my_predictions.csv в папке проекта.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
METRIC_NAMES = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "mae": "MAE", "cost": "Средняя стоимость ошибок"}


def score(cfg, y_true, y_pred):
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
    kind = cfg["metric"]
    if kind == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    if kind == "accuracy":
        return accuracy_score(y_true, y_pred)
    if kind == "mae":
        return mean_absolute_error(y_true, y_pred)
    if kind == "cost":
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return (cfg["cost"]["fn"] * fn + cfg["cost"]["fp"] * fp) / len(y_true)
    raise ValueError(kind)


def level(cfg, value):
    t, better = cfg["thresholds"], (lambda a, b: a >= b) if cfg["higher_is_better"] else (lambda a, b: a <= b)
    if better(value, t["excellent"]):
        return "🏆 отлично"
    if better(value, t["good"]):
        return "✅ хорошо"
    if better(value, t["baseline"]):
        return "🙂 на уровне бейзлайна — модель работает, но её можно заметно улучшить"
    return "⚠ хуже простого бейзлайна — проверьте предобработку и формат прогнозов"


def evaluate(folder, predictions):
    """Возвращает (значение метрики, список проблем) — используется и в check_project.py."""
    cfg = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    answer = pd.read_csv(folder / "answer.csv")
    pred = pd.read_csv(predictions)
    problems = []
    for col in (cfg["id"], cfg["prediction"]):
        if col not in pred.columns:
            problems.append(f"нет колонки «{col}» (есть: {list(pred.columns)})")
    if problems:
        return None, problems
    merged = answer.merge(pred[[cfg["id"], cfg["prediction"]]], on=cfg["id"], how="left", suffixes=("", "_pred"))
    pred_col = cfg["prediction"] + "_pred" if cfg["prediction"] == cfg["target"] else cfg["prediction"]
    missing = merged[pred_col].isna().sum()
    if missing:
        problems.append(f"нет прогнозов для {missing} объектов из {len(answer)}")
    extra = len(set(pred[cfg["id"]]) - set(answer[cfg["id"]]))
    if extra:
        problems.append(f"{extra} лишних id, которых нет в test.csv")
    if problems:
        return None, problems
    return float(score(cfg, merged[cfg["target"]].to_numpy(), merged[pred_col].to_numpy())), []


def main():
    parser = argparse.ArgumentParser(description="Проверка мини-проекта")
    parser.add_argument("project", nargs="?", help="папка проекта, например projects/churn_prediction")
    parser.add_argument("predictions", nargs="?", help="CSV с прогнозами (по умолчанию my_predictions.csv)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or not args.project:
        print("Мини-проекты:")
        for cfg_path in sorted(PROJECTS.glob("*/project.json")):
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            mark = "✔" if (cfg_path.parent / "my_predictions.csv").exists() else " "
            print(f"  [{mark}] {cfg_path.parent.relative_to(ROOT).as_posix():32s} {cfg['title']}")
        return 0

    folder = Path(args.project)
    folder = folder if folder.is_absolute() or folder.exists() else ROOT / folder
    if not (folder / "project.json").exists():
        print(f"❌ {folder} — не похоже на папку проекта (нет project.json). Список: --list")
        return 2
    predictions = Path(args.predictions) if args.predictions else folder / "my_predictions.csv"
    if not predictions.exists():
        print(f"❌ Не найден файл прогнозов {predictions}.\n   Сохраните прогнозы: result.to_csv('my_predictions.csv', index=False)")
        return 2

    cfg = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    value, problems = evaluate(folder, predictions)
    if problems:
        print("❌ Прогнозы не прошли проверку формата:")
        for p in problems:
            print("   •", p)
        return 1
    t = cfg["thresholds"]
    arrow = "↑ больше — лучше" if cfg["higher_is_better"] else "↓ меньше — лучше"
    print(f"{cfg['title']}\n{METRIC_NAMES[cfg['metric']]}: {value:.4f}   ({arrow})")
    print(f"Пороги: базовый {t['baseline']} · хорошо {t['good']} · отлично {t['excellent']}")
    print("Уровень:", level(cfg, value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
