#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям по ИИ-агентам (группа D) — только NumPy, pandas и стандартная библиотека.

Темы: оценка агентов (несмещённые pass@k и pass^k по таблице попыток) и безопасность агентов
(политика вызовов инструментов allow / ask / deny и проверка «смертельной триады» по сессиям).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_agents_d.py
"""

import numpy as np
import pandas as pd

from generate_basic_exercises import build


# ---------------------------------------------------------------- данные

def data_agent_evaluation(rng):
    """Таблица попыток: на каждой задаче n прогонов агента, из них c успешных."""
    n_tasks = 14
    n = rng.integers(6, 13, n_tasks)
    p = rng.beta(2.0, 1.2, n_tasks)
    c = rng.binomial(n, p)
    c[0], c[1], c[2] = 0, n[1], n[2] - 1          # крайние случаи: ни одного успеха, все успехи, один сбой
    c[3] = 2                                       # c < 3: у pass^3 и pass^5 здесь ноль
    c[4] = n[4] - 2                                # n − c = 2: у pass@3 и pass@5 здесь единица
    df = pd.DataFrame({"task_id": [f"t{i:02d}" for i in range(1, n_tasks + 1)], "n": n, "c": c})
    return df


def data_agent_safety(rng):
    """Журнал вызовов инструментов агента по восьми сессиям."""
    small = lambda: int(rng.integers(20, 101))     # сумма перевода, подпадающая под «спросить»
    large = lambda: int(rng.integers(150, 900))    # сумма перевода, подпадающая под «запретить»
    calls = [
        (1, "read_file", "/work/report.csv", ""),
        (1, "http_get", "docs.python.org", ""),
        (1, "send_email", "ceo@company.com", ""),
        (2, "read_file", "/etc/passwd", ""),
        (2, "http_get", "evil.example", ""),
        (2, "send_email", "x@gmail.com", ""),
        (3, "query_db", "orders", ""),
        (3, "read_email", "inbox", ""),
        (3, "http_post", "api.company.com", ""),
        (4, "read_file", "/work/../etc/shadow", ""),
        (4, "query_db", "users_pii", ""),
        (4, "http_get", "blog.example", ""),
        (4, "http_post", "paste.example", ""),
        (5, "write_file", "/work/tmp/a.txt", ""),
        (5, "run_shell", "", ""),
        (5, "delete_db", "orders", ""),
        (5, "transfer_money", "acc-17", small()),
        (5, "transfer_money", "acc-18", large()),
        (6, "read_file", "/home/user/notes.txt", ""),
        (6, "read_email", "inbox", ""),
        (6, "send_email", "ceo@company.com", ""),
        (7, "query_db", "orders", ""),
        (7, "write_file", "/root/x", ""),
        (7, "http_get", "api.company.com", ""),
        (8, "browser_click", "button", ""),
        (8, "read_email", "inbox", ""),
        (8, "http_post", "api.company.com", ""),
        (8, "query_db", "payments", ""),
    ]
    df = pd.DataFrame(calls, columns=["session_id", "tool", "target", "amount"])
    df.insert(0, "call_id", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "agent_evaluation",
        "title": "Оценка агента: несмещённые pass@k и pass^k по таблице попыток",
        "data": data_agent_evaluation,
        "task": """
            Агент службы поддержки прогнали на 14 задачах по нескольку раз. В `input.csv` — `task_id, n, c`:
            `n` — число независимых попыток на задаче, `c` — сколько из них оказались успешными.

            Для каждой задачи и каждого k ∈ {1, 3, 5} посчитайте **несмещённые** оценки:

            - pass@k = 1 − C(n − c, k) / C(n, k) — вероятность, что среди k попыток хотя бы одна успешна
              (если n − c < k, то результат равен 1);
            - pass^k = C(c, k) / C(n, k) — вероятность, что все k попыток успешны (если c < k, то результат равен 0).

            Затем усредните каждую метрику **по задачам** (простое среднее по 14 задачам, а не по попыткам).

            ## Подсказки

            - Все задачи имеют n ≥ 6, так что C(n, k) для k ≤ 5 никогда не равно нулю. Функция `math.comb(n, k)`
              возвращает 0, если k > n, — это удобно для pass^k.
            - Сначала посчитайте метрики на каждой задаче (столбцы или список), и только потом усредняйте. Среднее по задачам
              не равно метрике, построенной по суммарной доле успеха всех попыток.
            - `from math import comb`; `pass_at = 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)`;
              `pass_hat = comb(c, k) / comb(n, k)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `k, pass_at_k, pass_hat_k` — 3 строки (k = 1, 3, 5 по порядку). Значения округлить до 4 знаков.
        """,
        "solution": """
            from math import comb

            KS = (1, 3, 5)


            def pass_at_k(n, c, k):
                # хотя бы одна успешная среди k попыток: 1 − C(n−c, k)/C(n, k)
                if n - c < k:
                    return 1.0
                return 1.0 - comb(n - c, k) / comb(n, k)


            def pass_hat_k(n, c, k):
                # все k попыток успешны: C(c, k)/C(n, k); при c < k comb возвращает 0
                return comb(c, k) / comb(n, k)


            def main():
                df = pd.read_csv('input.csv')
                p_hat = df['c'].sum() / df['n'].sum()           # общая доля успеха — для сравнения

                rows = []
                for k in KS:
                    at = np.mean([pass_at_k(n, c, k) for n, c in zip(df['n'], df['c'])])
                    hat = np.mean([pass_hat_k(n, c, k) for n, c in zip(df['n'], df['c'])])
                    naive = p_hat ** k                          # наивная оценка надёжности «по общей доле»
                    print(f"k={k}: pass@k={at:.4f}  pass^k={hat:.4f}  (наивно p̂^k = {naive:.4f})")
                    rows.append({'k': k, 'pass_at_k': at, 'pass_hat_k': hat})

                result = pd.DataFrame(rows)
                assert result['pass_at_k'].is_monotonic_increasing
                assert result['pass_hat_k'].is_monotonic_decreasing
                return result.round(4)
        """,
        "review": """
            **Идея.** pass@k отвечает на вопрос «способен ли агент решить задачу, если дать k попыток», а pass^k — «можно ли
            на него положиться: решит ли он задачу все k раз подряд». С ростом k первая метрика растёт, вторая падает.
            Поэтому для сервиса, где каждый сбой заметен клиенту, смотрят на pass^k (так сделано в бенчмарке τ-bench),
            а для генерации кода с автотестами, где можно выбрать лучший из k вариантов, — на pass@k.

            ```python
            from math import comb
            pass_at_k = 1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
            ```

            **Типичные ошибки:**
            - Усреднили по попыткам, а не по задачам: задачи с большим n получили больший вес.
            - Подставили общую долю успеха в 1 − (1 − p)^k или p^k — это смещённая оценка, не учитывающая разброс между задачами.
            - Забыли случай n − c < k: `comb(n - c, k)` равен 0, формула даёт 1, но при ручной реализации легко получить деление
              на ноль или отрицательное число.
            - Перепутали pass@k и pass^k (первое — «хотя бы одна», второе — «все»).

            **Что вынести:** метрики считаются по каждой задаче из таблицы (n, c) и затем усредняются; pass@k — способность,
            pass^k — надёжность. → шпаргалка «Оценка агентов».
        """,
    },
    {
        "name": "agent_safety",
        "title": "Безопасность агента: политика вызовов инструментов и «смертельная триада»",
        "data": data_agent_safety,
        "task": """
            В `input.csv` — журнал вызовов инструментов агента по 8 сессиям: `call_id, session_id, tool, target, amount`
            (`target` — путь, домен, адрес или таблица; `amount` заполнен только у `transfer_money`).
            Читайте файл с `keep_default_na=False`, чтобы пустые поля остались пустыми строками.

            **Шаг 1. Политика.** Для каждого вызова определите вердикт `allow` (разрешить), `ask` (спросить человека)
            или `deny` (запретить) по правилам; применяется **первое** подходящее правило:

            1. Неизвестный инструмент (не из списка ниже) — `deny`.
            2. Если `target` содержит `..` (выход из каталога) — `deny`.
            3. `run_shell`, `delete_db` — `deny`.
            4. `read_file`: путь начинается с `/etc/`, содержит `.ssh` или заканчивается на `.env` — `deny`;
               начинается с `/work/` — `allow`; иначе — `ask`.
            5. `write_file`: начинается с `/work/tmp/` — `allow`; с `/work/` — `ask`; иначе — `deny`.
            6. `query_db`: таблицы `users_pii` и `payments` — `ask`, остальные — `allow`.
            7. `read_email` — `allow`.
            8. `http_get`: домены `docs.python.org` и `api.company.com` — `allow`, остальные — `ask`.
            9. `http_post`: домен `api.company.com` — `ask`, остальные — `deny`.
            10. `send_email`: адрес оканчивается на `@company.com` — `ask`, иначе — `deny`.
            11. `transfer_money`: сумма ≤ 100 — `ask`, иначе — `deny`.

            Список известных инструментов: `read_file, write_file, query_db, read_email, http_get, http_post, send_email,
            run_shell, delete_db, transfer_money`.

            **Шаг 2. Триада.** Вызов с вердиктом `allow` или `ask` считается доступной возможностью (запрещённый — нет).
            Возможности сессии по инструментам:

            - `private_data` — `read_file`, `query_db`;
            - `untrusted_content` — `http_get`, `read_email`;
            - `external_channel` — `http_post`, `send_email`.

            Сессия образует «смертельную триаду» (`trifecta = 1`), если у неё есть все три возможности.

            ## Подсказки

            - Правила — это цепочка `if … return` в порядке номеров: сначала проверки 1–3 (они общие), затем разбор по инструменту.
              Если `target` пустой, `..` в нём не встречается.
            - Для домена и адреса достаточно сравнения строк: `target in {…}` и `target.endswith('@company.com')`.
              Сумму переводите в число только для `transfer_money`.
            - Постройте таблицу вердиктов, затем `groupby('session_id')`: число вердиктов каждого вида и множество возможностей
              из строк с вердиктом не `deny`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `session_id, allow, ask, deny, private_data, untrusted_content, external_channel, trifecta` —
            8 строк (сессии 1…8 по порядку). `allow, ask, deny` — числа вызовов с таким вердиктом, остальные колонки — 0 или 1.
        """,
        "solution": """
            KNOWN = {'read_file', 'write_file', 'query_db', 'read_email', 'http_get', 'http_post', 'send_email',
                     'run_shell', 'delete_db', 'transfer_money'}
            CAPABILITY = {'read_file': 'private_data', 'query_db': 'private_data',
                          'http_get': 'untrusted_content', 'read_email': 'untrusted_content',
                          'http_post': 'external_channel', 'send_email': 'external_channel'}
            CAPS = ['private_data', 'untrusted_content', 'external_channel']


            def decide(tool, target, amount):
                if tool not in KNOWN:
                    return 'deny'                                  # deny by default
                if '..' in target:
                    return 'deny'                                  # выход из каталога
                if tool in ('run_shell', 'delete_db'):
                    return 'deny'
                if tool == 'read_file':
                    if target.startswith('/etc/') or '.ssh' in target or target.endswith('.env'):
                        return 'deny'
                    return 'allow' if target.startswith('/work/') else 'ask'
                if tool == 'write_file':
                    if target.startswith('/work/tmp/'):
                        return 'allow'
                    return 'ask' if target.startswith('/work/') else 'deny'
                if tool == 'query_db':
                    return 'ask' if target in ('users_pii', 'payments') else 'allow'
                if tool == 'read_email':
                    return 'allow'
                if tool == 'http_get':
                    return 'allow' if target in ('docs.python.org', 'api.company.com') else 'ask'
                if tool == 'http_post':
                    return 'ask' if target == 'api.company.com' else 'deny'
                if tool == 'send_email':
                    return 'ask' if target.endswith('@company.com') else 'deny'
                return 'ask' if float(amount) <= 100 else 'deny'   # transfer_money


            def main():
                df = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
                df['verdict'] = [decide(t, g, a) for t, g, a in zip(df['tool'], df['target'], df['amount'])]
                df['session_id'] = df['session_id'].astype(int)

                rows = []
                for sid, g in df.groupby('session_id', sort=True):
                    alive = g[g['verdict'] != 'deny']              # запрещённые вызовы возможностей не дают
                    have = set(alive['tool'].map(CAPABILITY).dropna())
                    row = {'session_id': sid}
                    row.update({v: int((g['verdict'] == v).sum()) for v in ('allow', 'ask', 'deny')})
                    row.update({cap: int(cap in have) for cap in CAPS})
                    row['trifecta'] = int(all(cap in have for cap in CAPS))
                    rows.append(row)
                    if row['trifecta']:
                        print(f"сессия {sid}: триада! уберите одну из трёх возможностей")

                return pd.DataFrame(rows)
        """,
        "review": """
            **Идея.** Безопасность агента обеспечивают не уговоры в промпте, а код вокруг модели: политика решает, что можно
            выполнять сразу, что только после подтверждения человека, а что нельзя никогда. Неизвестный инструмент по умолчанию
            запрещён (deny by default), выход за пределы каталога (`..`) режется до любых других проверок. Вторая часть —
            архитектурный тест «смертельной триады»: приватные данные + недоверенный контент + внешний канал в одной сессии
            дают возможность утечки; достаточно убрать любую одну ногу.

            ```python
            lethal = {'private_data', 'untrusted_content', 'external_channel'} <= capabilities
            ```

            **Типичные ошибки:**
            - Проверили префикс `/work/` раньше, чем `..`: путь `/work/../etc/shadow` ошибочно получил бы `allow`.
            - Учли запрещённые вызовы как возможности — сессия 2 (чтение `/etc/passwd` отклонено, письмо наружу запрещено) ложно
              считалась бы опасной.
            - Забыли, что `ask` — тоже доступная возможность (с одобрения человека она выполнится).
            - Считали `amount` числом для всех строк, не отделив `transfer_money` (пустые поля дают ошибку).
            - Применяли не первое подходящее правило, а все подряд.

            **Что вынести:** least privilege и deny by default задаются в коде; запрещённое действие не считается возможностью;
            для триады достаточно убрать одну из трёх составляющих. → шпаргалка «Безопасность агентов».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям по ИИ-агентам (группа D):")
    for seed, exercise in enumerate(EXERCISE_LIST, start=991):
        build(exercise, seed)


if __name__ == "__main__":
    main()
