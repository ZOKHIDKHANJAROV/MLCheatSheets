#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям по ИИ-агентам (группа A): основы агентов, вызов инструментов, ReAct —
только NumPy, pandas и стандартная библиотека (без сети, без LLM и без реальных вызовов API).

Темы: симулятор цикла агента с лимитом шагов (шаги, стоимость, успешность), валидатор вызовов инструментов
по JSON-схеме, разбор лога ReAct (повторы действий, зацикливание, ошибки формата, «выдуманные» наблюдения).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_agents_a.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build


def to_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- данные

def data_agent_loop(rng):
    """Журнал шагов агента: по строке на шаг; задачи перемешаны, у части агент не уложился в лимит."""
    actions = ["search", "calc", "read_file", "query_db"]
    rows = []
    for task_id in range(1, 31):
        n = int(rng.integers(2, 10))                  # сколько шагов агент реально сделал
        finishes = rng.random() < 0.8
        correct = int(rng.random() < 0.75)
        for step in range(1, n + 1):
            is_finish = finishes and step == n
            rows.append({
                "task_id": task_id,
                "step": step,
                "action": "finish" if is_finish else str(rng.choice(actions)),
                "tokens": int(rng.integers(200, 1500)),
                "correct": correct if is_finish else 0,
            })
    df = pd.DataFrame(rows)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


SCHEMAS = {
    "get_weather": {
        "required": ["city"],
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["c", "f"]},
        },
    },
    "add_task": {
        "required": ["title"],
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            "done": {"type": "boolean"},
        },
    },
    "convert_currency": {
        "required": ["amount", "from", "to"],
        "properties": {
            "amount": {"type": "number", "minimum": 0},
            "from": {"type": "string", "enum": ["USD", "EUR", "RUB"]},
            "to": {"type": "string", "enum": ["USD", "EUR", "RUB"]},
        },
    },
}

CITIES = ["Москва", "Казань", "Омск", "Пермь", "Сочи"]
TITLES = ["купить молоко", "написать отчёт", "позвонить в банк", "проверить почту"]


def _valid_args(rng, tool):
    if tool == "get_weather":
        args = {"city": str(rng.choice(CITIES))}
        if rng.random() < 0.5:
            args["units"] = str(rng.choice(["c", "f"]))
    elif tool == "add_task":
        args = {"title": str(rng.choice(TITLES))}
        if rng.random() < 0.6:
            args["priority"] = int(rng.integers(1, 6))
        if rng.random() < 0.4:
            args["done"] = bool(rng.random() < 0.5)
    else:
        cur = rng.permutation(["USD", "EUR", "RUB"])
        amount = int(rng.integers(1, 500)) if rng.random() < 0.5 else round(float(rng.uniform(1, 500)), 2)
        args = {"amount": amount, "from": str(cur[0]), "to": str(cur[1])}
    return args


def _corrupt(rng, tool, args, kind):
    """Портит корректные аргументы ровно одним способом; возвращает строку для столбца arguments."""
    args = dict(args)
    if kind == "missing_field":
        key = str(rng.choice(SCHEMAS[tool]["required"]))
        del args[key]
    elif kind == "extra_field":
        args[str(rng.choice(["lang", "timeout", "verbose"]))] = "ru" if rng.random() < 0.5 else 5
    elif kind == "bad_type":
        if tool == "get_weather":
            args["city"] = int(rng.integers(1, 100)) if rng.random() < 0.5 else [args["city"]]
        elif tool == "add_task":
            args["priority"] = [True, 2.5, "3"][int(rng.integers(0, 3))]    # bool и float — не integer
        else:
            args["amount"] = str(args["amount"])
    elif kind == "bad_enum":
        if tool == "get_weather":
            args["units"] = "kelvin"
        elif tool == "add_task":
            args["priority"] = 3                                             # у add_task нет enum — запасной вариант
        else:
            args["to"] = "GBP"
    elif kind == "out_of_range":
        if tool == "add_task":
            args["priority"] = int(rng.choice([0, 6, 9]))
        elif tool == "convert_currency":
            args["amount"] = -float(rng.integers(1, 50))
    return json.dumps(args, ensure_ascii=False)


def data_tool_calls(rng):
    plan = (["ok"] * 22 + ["unknown_tool"] * 4 + ["bad_json"] * 6 + ["missing_field"] * 6
            + ["extra_field"] * 5 + ["bad_type"] * 6 + ["bad_enum"] * 4 + ["out_of_range"] * 4)
    rows = []
    for kind in plan:
        tools = list(SCHEMAS)
        if kind == "bad_enum":
            tools = ["get_weather", "convert_currency"]
        if kind == "out_of_range":
            tools = ["add_task", "convert_currency"]
        tool = str(rng.choice(tools))
        args = _valid_args(rng, tool)
        if kind == "unknown_tool":
            tool = str(rng.choice(["send_email", "delete_file", "get_wether"]))
            text = json.dumps(args, ensure_ascii=False)
        elif kind == "bad_json":
            good = json.dumps(args, ensure_ascii=False)
            text = [good[:-1], good.replace('"', "'"), json.dumps(list(args.values()), ensure_ascii=False)][
                int(rng.integers(0, 3))]
        elif kind == "ok":
            text = json.dumps(args, ensure_ascii=False)
        else:
            text = _corrupt(rng, tool, args, kind)
        rows.append((tool, text))
    # ловушки порядка проверок: сразу несколько нарушений — побеждает самое раннее правило
    rows.append(("get_weather", json.dumps({"units": "kelvin"}, ensure_ascii=False)))                 # missing_field
    rows.append(("add_task", json.dumps({"title": "тест", "priority": 9, "lang": "ru"})))            # extra_field
    rows.append(("convert_currency", json.dumps({"amount": -5, "from": "USD", "to": "GBP"})))       # bad_enum
    order = rng.permutation(len(rows))
    df = pd.DataFrame([rows[i] for i in order], columns=["tool", "arguments"])
    df.insert(0, "call_id", [f"c{i:03d}" for i in range(1, len(df) + 1)])
    return {"input.csv": df, "schemas.json": to_json(SCHEMAS)}


QUERIES = ["столица Франции", "население Парижа", "курс евро", "высота Эвереста", "автор Войны и мира",
           "площадь озера Байкал", "год основания Москвы", "формула воды", "скорость света", "ВВП Японии"]
TOOLS = ["search", "lookup", "calc"]


def _step(lines, rng, tool, arg, obs=None):
    lines.append(f"Thought: нужно проверить: {arg}")
    lines.append(f"Action: {tool}[{arg}]")
    lines.append(f"Observation: {obs if obs is not None else 'найдено: ' + arg[:3]}")


def data_react_log(rng):
    scenarios = (["clean"] * 7 + ["repeat2"] * 3 + ["loop"] * 5 + ["format"] * 4 + ["hallu"] * 3 + ["limit"] * 2)
    scenarios = [scenarios[i] for i in rng.permutation(len(scenarios))]
    rows = []
    for run_no, kind in enumerate(scenarios, start=1):
        run_id = f"run{run_no:02d}"
        qs = [str(q) for q in rng.permutation(QUERIES)]
        lines = []
        if kind == "clean":
            for q in qs[: int(rng.integers(2, 5))]:
                _step(lines, rng, str(rng.choice(TOOLS)), q)
            lines += ["Thought: данных достаточно", "Final Answer: готово"]
        elif kind == "repeat2":
            _step(lines, rng, "search", qs[0])
            _step(lines, rng, "lookup", qs[1])
            _step(lines, rng, "search", qs[0])
            lines += ["Thought: ясно", "Final Answer: готово"]
        elif kind == "loop":
            if rng.random() < 0.5:
                _step(lines, rng, "lookup", qs[1])
            variants = [qs[0], "  " + qs[0].upper(), qs[0] + " ", qs[0].capitalize(), qs[0]]
            for v in variants[: int(rng.integers(3, 6))]:
                _step(lines, rng, "search", v.strip() if rng.random() < 0.5 else v, obs="ничего не найдено")
        elif kind == "format":
            _step(lines, rng, "search", qs[0])
            lines += ["Thought: попробую ещё", f"Action: search {qs[1]}", "Observation: ошибка разбора"]
            if rng.random() < 0.5:
                lines += ["Thought: ещё раз", f"Action: lookup({qs[2]})", "Observation: ошибка разбора"]
            _step(lines, rng, "lookup", qs[3])
            lines += ["Thought: всё", "Final Answer: готово"]
        elif kind == "hallu":
            _step(lines, rng, "search", qs[0])
            lines += ["Thought: ответ, видимо, такой", "Observation: найдено сразу всё, что нужно"]
            lines += ["Final Answer: готово"]
        else:  # limit: шесть разных действий и ни одного ответа
            for q in qs[:6]:
                _step(lines, rng, str(rng.choice(TOOLS)), q)
        for no, text in enumerate(lines, start=1):
            if rng.random() < 0.15:
                text = "  " + text + " "                      # лишние пробелы по краям строки
            rows.append((run_id, no, text))
    df = pd.DataFrame(rows, columns=["run_id", "line_no", "text"])
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "ai_agents_basics",
        "title": "Симулятор цикла агента: шаги, стоимость и успешность при лимите шагов",
        "data": data_agent_loop,
        "task": """
            В файле `input.csv` — журнал работы агента на 30 задачах: по одной строке на шаг
            (`task_id, step, action, tokens, correct`). Строки перемешаны. Действие `finish` означает, что агент выдал
            финальный ответ; в этой строке `correct` равно 1, если ответ верный, и 0, если неверный (в остальных строках
            `correct` = 0). Если у задачи нет строки `finish`, агент так и не ответил (завершился сбоем).

            Заказчик вводит жёсткий **лимит шагов** `MAX_STEPS = 6`: шаги с номером `step` больше 6 агент
            выполнить не успел, поэтому их нужно полностью игнорировать (и при подсчёте шагов, и при подсчёте токенов,
            и при поиске `finish`). Цена — 0.5 условной единицы за 1000 токенов.

            Для каждой задачи посчитайте:
            - `steps_used` — число шагов с `step` ≤ 6;
            - `tokens` — сумма `tokens` по этим шагам;
            - `cost` — `tokens / 1000 * 0.5`, округлить до 4 знаков;
            - `status` — `success`, если среди учтённых шагов есть `finish` с `correct = 1`; `wrong`, если есть `finish` с
              `correct = 0`; `limit`, если учтённых `finish` нет (агент не ответил в лимит или не ответил вовсе).

            ## Подсказки

            - Сначала отфильтруйте строки `step <= 6`, и только потом группируйте по `task_id` — тогда `finish`,
              случившийся на шаге 7 и позже, не повлияет на статус.
            - `finish` на шаге ровно 6 ещё успевает: граница включительная.
            - Для статуса достаточно взять строки `finish` в группе: пустая группа → `limit`, иначе смотрите на `correct`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `task_id, steps_used, tokens, cost, status` — 30 строк, по одной на задачу, отсортированных
            по `task_id` по возрастанию; `cost` округлить до 4 знаков.
        """,
        "solution": """
            MAX_STEPS = 6
            PRICE_PER_1K = 0.5


            def task_status(group):
                finish = group[group['action'] == 'finish']
                if finish.empty:
                    return 'limit'                                # ответа в пределах лимита нет
                return 'success' if finish['correct'].iloc[0] == 1 else 'wrong'


            def main():
                log = pd.read_csv('input.csv')
                log = log[log['step'] <= MAX_STEPS]               # шаги после лимита не существуют
                rows = []
                for task_id, group in log.groupby('task_id'):
                    tokens = int(group['tokens'].sum())
                    rows.append({'task_id': task_id, 'steps_used': len(group), 'tokens': tokens,
                                 'cost': round(tokens / 1000 * PRICE_PER_1K, 4), 'status': task_status(group)})
                result = pd.DataFrame(rows).sort_values('task_id').reset_index(drop=True)
                print(result['status'].value_counts().to_string())
                return result
        """,
        "review": """
            **Идея.** Цикл агента нельзя пускать на самотёк: у него есть лимит шагов, и результат задачи зависит от того,
            успел ли агент ответить в этот лимит. Журнал шагов — основной материал для оценки агента: по нему считают число
            шагов, расход токенов (то есть деньги) и долю успехов. Лимит шагов «отрезает» хвост траектории, и всё, что после
            него, в метрики попадать не должно.

            **Типичные ошибки:**
            - Считали токены по всем шагам, а статус — по учтённым (или наоборот): стоимость и статус рассогласованы.
            - Нашли `finish` на шаге 7–9 и засчитали успех, хотя в лимит агент не уложился.
            - Граница `step < 6` вместо `step <= 6`: теряются задачи, завершившиеся ровно на шестом шаге.
            - Статус `wrong` перепутали с `limit`: «ответил неверно» и «не ответил» — разные сбои.
            - Не отсортировали результат по `task_id` (порядок строк в журнале перемешан).

            **Что вынести:** у агента всегда есть ограничители (шаги, токены, время), а метрики — доля успешных задач,
            число шагов и стоимость — считают при тех же ограничениях, с которыми агент работает в реальности.
            → шпаргалка «Основы ИИ-агентов».
        """,
    },
    {
        "name": "tool_use_function_calling",
        "title": "Валидатор вызовов инструментов по JSON-схеме",
        "data": data_tool_calls,
        "task": """
            Модель сформировала вызовы инструментов — они в `input.csv` (`call_id, tool, arguments`). Столбец `arguments` —
            строка, которая должна быть JSON-объектом. Описания инструментов — в `schemas.json`: для каждого инструмента
            список `required` и словарь `properties` (у поля есть `type`, а иногда `enum`, `minimum`, `maximum`).

            Проверьте каждый вызов и присвойте ему **ровно один статус** — по первому нарушенному правилу в таком порядке:

            1. `unknown_tool` — инструмента `tool` нет в `schemas.json`;
            2. `bad_json` — `arguments` не разбирается как JSON (`json.loads` падает) **или** результат — не объект (не `dict`);
            3. `missing_field` — нет хотя бы одного поля из `required`;
            4. `extra_field` — есть поле, которого нет в `properties`;
            5. `bad_type` — тип значения не совпадает с `type`. Соответствие: `string` → `str`; `integer` → `int`;
               `number` → `int` или `float`; `boolean` → `bool`. Значение `bool` **не** считается ни `integer`, ни `number`;
               `float` (например, `2.5`) не считается `integer`;
            6. `bad_enum` — значение не входит в `enum` (если он задан);
            7. `out_of_range` — число меньше `minimum` или больше `maximum` (если заданы);
            8. `ok` — нарушений нет.

            Каждое правило применяется ко **всем** полям вызова, прежде чем переходить к следующему
            (например, вызов без обязательного поля и с неверным `enum` получает `missing_field`).

            Подсчитайте, сколько вызовов получило каждый статус.

            ## Подсказки

            - Реализуйте функцию `validate(tool, arguments_text) -> status` и проверяйте правила последовательно:
              как только нашли нарушение — сразу возвращайте статус.
            - Проверка типа: `isinstance(True, int)` истинно в Python, поэтому `bool` отсеивайте отдельным условием до проверки
              `int`/`float`. Для `number` допустимы `int` и `float` (но не `bool`).
            - Читайте `input.csv` с `dtype=str, keep_default_na=False`; в `status` включите все 8 статусов, даже если у
              некоторых счётчик нулевой.

            ## Ожидаемый формат выходного файла

            CSV с колонками `status, count, share` — 8 строк в порядке `ok, unknown_tool, bad_json, missing_field,
            extra_field, bad_type, bad_enum, out_of_range`; `share` — доля вызовов с этим статусом от общего числа,
            округлить до 4 знаков.
        """,
        "solution": """
            import json

            STATUSES = ['ok', 'unknown_tool', 'bad_json', 'missing_field', 'extra_field', 'bad_type', 'bad_enum',
                        'out_of_range']


            def type_ok(value, expected):
                if expected == 'string':
                    return isinstance(value, str)
                if expected == 'boolean':
                    return isinstance(value, bool)
                if isinstance(value, bool):                      # bool — подкласс int, но числом не считается
                    return False
                if expected == 'integer':
                    return isinstance(value, int)
                if expected == 'number':
                    return isinstance(value, (int, float))
                return False


            def validate(tool, arguments_text, schemas):
                if tool not in schemas:
                    return 'unknown_tool'
                try:
                    args = json.loads(arguments_text)
                except ValueError:
                    return 'bad_json'
                if not isinstance(args, dict):
                    return 'bad_json'
                schema = schemas[tool]
                props = schema['properties']
                if any(name not in args for name in schema['required']):
                    return 'missing_field'
                if any(name not in props for name in args):
                    return 'extra_field'
                if any(not type_ok(value, props[name]['type']) for name, value in args.items()):
                    return 'bad_type'
                if any('enum' in props[name] and value not in props[name]['enum'] for name, value in args.items()):
                    return 'bad_enum'
                for name, value in args.items():
                    spec = props[name]
                    if 'minimum' in spec and value < spec['minimum']:
                        return 'out_of_range'
                    if 'maximum' in spec and value > spec['maximum']:
                        return 'out_of_range'
                return 'ok'


            def main():
                calls = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
                with open('schemas.json', encoding='utf-8') as f:
                    schemas = json.load(f)
                calls['status'] = [validate(t, a, schemas) for t, a in zip(calls['tool'], calls['arguments'])]
                counts = calls['status'].value_counts().reindex(STATUSES, fill_value=0)
                result = pd.DataFrame({'status': STATUSES, 'count': counts.to_numpy()})
                result['share'] = (result['count'] / len(calls)).round(4)
                print(f"корректных вызовов: {int(counts['ok'])} из {len(calls)}")
                return result
        """,
        "review": """
            **Идея.** Аргументы, которые вернула модель, — недоверенный ввод: она может выдумать инструмент, пропустить
            обязательное поле, перепутать тип или выйти за допустимый диапазон. Поэтому приложение проверяет вызов по схеме
            **до** выполнения функции, а причину отказа возвращает модели в `tool_result` — так она сможет исправить вызов.
            Фиксированный порядок правил делает диагностику однозначной: сообщается самое раннее нарушение.

            **Типичные ошибки:**
            - `isinstance(True, int)` истинно: `priority = true` проходит как целое число — это `bad_type`.
            - Значение `2.5` для `integer` принято как «число» — тип целого строгий.
            - Проверки перемешаны: вызов без обязательного поля, но с неверным `enum`, получил `bad_enum` вместо `missing_field`.
            - JSON-список (`["Москва"]`) разобрался без ошибки и пошёл дальше — но это не объект, то есть `bad_json`.
            - Забыли `dtype=str`: pandas интерпретирует часть строк иначе, а пропуски превращаются в `NaN`.
            - В итоге нет строк с нулевым счётчиком статусов.

            **Что вынести:** валидация — первая линия защиты цикла tool use; схему используют и для подсказки модели,
            и для проверки её ответа, а ошибки сообщают модели понятным текстом.
            → шпаргалка «Вызов инструментов (function calling)».
        """,
    },
    {
        "name": "react_agents",
        "title": "Разбор лога ReAct: повторы действий, зацикливание и эффективность",
        "data": data_react_log,
        "task": """
            В `input.csv` — логи 24 запусков ReAct-агента (`run_id, line_no, text`): каждая строка лога лежит в отдельной
            строке таблицы, **порядок строк в файле перемешан** — восстановите его по `line_no` внутри каждого `run_id`.
            По краям `text` могут быть лишние пробелы — уберите их (`strip`).

            Строки лога бывают такими:
            - `Thought: ...` — рассуждение;
            - `Action: имя[аргумент]` — действие; **корректный** формат — регулярное выражение `^Action: (\\w+)\\[(.*)\\]$`
              (например, `Action: search[столица Франции]`); строки `Action: search Париж` и `Action: lookup(Париж)`
              — это **ошибки формата**;
            - `Observation: ...` — наблюдение;
            - `Final Answer: ...` — финальный ответ.

            Нормализуйте действие: имя и аргумент — в нижний регистр, пробелы по краям аргумента убрать, внутренние
            пробелы схлопнуть (`' '.join(text.split())`). Два действия одинаковы, если одинаковы нормализованные пары
            `(имя, аргумент)`.

            Для каждого запуска посчитайте:
            - `steps` — число строк, начинающихся с `Action:` (включая ошибочные);
            - `repeats` — сколько корректных действий **уже встречались раньше** в этом запуске (первое появление не
              считается: три одинаковых действия дают `repeats = 2`);
            - `looped` — 1, если какое-то корректное действие встретилось **3 и более раз**, иначе 0;
            - `format_errors` — число строк `Action:` с неверным форматом;
            - `orphan_obs` — число строк `Observation:`, которым **не предшествует** строка `Action:`
              (предыдущая строка запуска — не `Action:`): такое наблюдение модель выдумала сама;
            - `finished` — 1, если в запуске есть строка `Final Answer:`, иначе 0;
            - `efficiency` — число **различных** корректных действий, делённое на `steps` (0, если `steps = 0`),
              округлить до 4 знаков.

            ## Подсказки

            - Сначала `sort_values(['run_id', 'line_no'])`, и только потом разбирайте строки подряд: понятие «предыдущая
              строка» имеет смысл лишь в восстановленном порядке.
            - Для повторов ведите множество уже виденных пар `(имя, аргумент)` и счётчик `Counter`: `repeats` — сумма
              `(count - 1)`, `looped` — наличие `count >= 3`.
            - `Action: search Париж` — это строка `Action:`, значит она учитывается в `steps` и `format_errors`, но не в
              повторах и не в различных действиях.

            ## Ожидаемый формат выходного файла

            CSV с колонками `run_id, steps, repeats, looped, format_errors, orphan_obs, finished, efficiency` — 24 строки,
            по одной на запуск, отсортированные по `run_id`; `efficiency` округлить до 4 знаков.
        """,
        "solution": """
            import re
            from collections import Counter

            ACTION = re.compile(r'^Action: (\\w+)\\[(.*)\\]$')


            def norm(text):
                return ' '.join(text.split()).lower()


            def analyse(lines):
                steps = format_errors = orphan_obs = 0
                counts = Counter()
                finished = 0
                prev = ''
                for line in lines:
                    if line.startswith('Action:'):
                        steps += 1
                        match = ACTION.match(line)
                        if match is None:
                            format_errors += 1
                        else:
                            counts[(norm(match.group(1)), norm(match.group(2)))] += 1
                    elif line.startswith('Observation:') and not prev.startswith('Action:'):
                        orphan_obs += 1                              # наблюдение без действия — выдумано моделью
                    elif line.startswith('Final Answer:'):
                        finished = 1
                    prev = line
                repeats = sum(c - 1 for c in counts.values())
                return {'steps': steps, 'repeats': repeats, 'looped': int(any(c >= 3 for c in counts.values())),
                        'format_errors': format_errors, 'orphan_obs': orphan_obs, 'finished': finished,
                        'efficiency': round(len(counts) / steps, 4) if steps else 0.0}


            def main():
                log = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
                log['line_no'] = log['line_no'].astype(int)
                log['text'] = log['text'].str.strip()
                log = log.sort_values(['run_id', 'line_no'])
                rows = []
                for run_id, group in log.groupby('run_id'):
                    rows.append({'run_id': run_id, **analyse(group['text'].tolist())})
                result = pd.DataFrame(rows).sort_values('run_id').reset_index(drop=True)
                print(f"зациклились: {int(result['looped'].sum())} из {len(result)}")
                return result
        """,
        "review": """
            **Идея.** Лог ReAct — это трасса, по которой видно, что делал агент: что думал, какие действия вызывал и какие
            получил наблюдения. Из трассы считают простые метрики здоровья цикла: сколько шагов, сколько повторов, не
            зациклился ли агент, не ломается ли формат и не выдумывает ли модель наблюдения сама. Эти же проверки
            встраивают в сам цикл: остановка при повторе действия, ошибка формата возвращается модели.

            **Типичные ошибки:**
            - Разбирали строки в порядке файла, а не по `line_no`: «предыдущая строка» и повторы определяются неверно.
            - Не нормализовали действие: `search[Столица Франции]` и `search[столица франции ]` посчитаны разными.
            - Считали `repeats` как число действий, а не как `count - 1` для каждого действия.
            - Ошибочные строки `Action:` попали в повторы или в «различные действия».
            - `orphan_obs` определяли как «наблюдений больше, чем действий» — это неточно: нужен именно порядок строк.
            - Забыли `strip()` у строк с лишними пробелами по краям.

            **Что вынести:** зацикливание ловится простой проверкой повторов нормализованных действий, а выдуманные
            наблюдения — стоп-последовательностью и сверкой порядка строк. Анализ трассы — главный инструмент отладки
            агентов. → шпаргалка «ReAct-агенты».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям по ИИ-агентам (группа A):")
    for seed, exercise in enumerate(EXERCISE_LIST, start=901):
        build(exercise, seed)


if __name__ == "__main__":
    main()
