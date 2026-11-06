#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений по теме «ИИ-агенты» (группа C) — только NumPy, pandas и стандартная библиотека.

Темы: планирование агента (топологический порядок DAG подзадач, критический путь, бюджет токенов),
стоимость координации мультиагентных схем (supervisor, все-со-всеми, конвейер), разбор лога JSON-RPC-обмена
в стиле MCP (сопоставление request/response по id, ошибки и таймауты).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_agents_c.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build


def to_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- данные

# Подзадачи агента-исследователя: (название, номера зависимостей в этом списке)
PLAN_TASKS = [
    ("найти источники", []),
    ("прочитать отчёт A", [0]),
    ("прочитать отчёт B", [0]),
    ("извлечь цифры", [1, 2]),
    ("сверить данные", [3]),
    ("построить таблицу", [4]),
    ("написать введение", [0]),
    ("написать выводы", [5]),
    ("собрать черновик", [5, 6, 7]),
    ("проверить факты", [8]),
    ("отредактировать стиль", [8]),
    ("опубликовать отчёт", [9, 10]),
]


def _plan_statuses(order, deps, tokens, budget):
    """Жадное выполнение в топологическом порядке: нужен тот же результат, что и в решении."""
    done, spent, out = set(), 0, {}
    for t in order:
        if any(p not in done for p in deps[t]):
            out[t] = "blocked"
        elif spent + tokens[t] > budget:
            out[t] = "skipped_budget"
        else:
            done.add(t)
            spent += tokens[t]
            out[t] = "done"
    return out


def data_planning(rng):
    n = len(PLAN_TASKS)
    ids = rng.permutation(n) + 1                       # номера не совпадают с порядком выполнения
    rows = []
    for i, (title, d) in enumerate(PLAN_TASKS):
        rows.append({"task_id": int(ids[i]), "title": title,
                     "deps": ";".join(str(int(x)) for x in sorted(ids[j] for j in d)),
                     "tokens": int(rng.integers(4, 31)) * 50, "duration": int(rng.integers(1, 6))})
    df = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)

    # порядок Кана с минимальным номером — как в решении
    deps = {int(r.task_id): [int(x) for x in r.deps.split(";") if x] for r in df.itertuples()}
    tokens = dict(zip(df.task_id, df.tokens))
    order, left = [], dict(deps)
    while left:
        t = min(k for k, d in left.items() if all(p in order for p in d))
        order.append(t)
        del left[t]
    total = int(df.tokens.sum())
    for frac in np.arange(0.55, 0.95, 0.01):           # бюджет подбираем так, чтобы были оба вида пропусков
        budget = int(round(total * frac / 50)) * 50
        st = set(_plan_statuses(order, deps, tokens, budget).values())
        if {"skipped_budget", "blocked", "done"} <= st:
            break
    else:
        raise RuntimeError("не нашёлся бюджет с пропусками обоих видов")
    return {"input.csv": df, "budget.json": to_json({"token_budget": budget})}


def data_multi_agent(rng):
    n = 7
    roles = ["worker"] * n
    roles[int(rng.integers(1, n - 1))] = "orchestrator"   # оркестратор — не первая и не последняя строка
    names = ["planner", "coder", "tester", "reviewer", "researcher", "writer", "critic"]
    df = pd.DataFrame({
        "agent_id": np.arange(1, n + 1),
        "name": rng.permutation(names),
        "role": roles,
        "msg_tokens": rng.integers(2, 19, n) * 50,
        "price_per_1k": rng.choice([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], n),
    })
    return df


def data_mcp(rng):
    events = []   # (ts, порядок вставки, направление, сообщение)

    def add(ts, direction, msg):
        events.append((int(ts), len(events), direction, msg))

    methods = ["tools/list", "tools/call", "resources/list", "resources/read", "prompts/list", "prompts/get",
               "tools/call", "tools/call"]

    def respond(req_ts, rid, outcome, direction):
        """Ответ на запрос: outcome — ok / error / late / none."""
        if outcome == "none":
            return
        if outcome == "ok":
            lat, body = int(rng.integers(20, 1800)), {"result": {"content": [{"type": "text", "text": "..."}]}}
        elif outcome == "late":
            lat, body = int(rng.integers(5200, 9000)), {"result": {"content": []}}
        else:
            code = int(rng.choice([-32601, -32602, -32603]))
            lat, body = int(rng.integers(20, 600)), {"error": {"code": code, "message": "ошибка обработки"}}
        add(req_ts + lat, direction, {"jsonrpc": "2.0", "id": rid, **body})

    # рукопожатие
    add(0, "C->S", {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"clientInfo": {"name": "demo-client"}}})
    add(85, "S->C", {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "demo-server"}}})
    add(120, "C->S", {"jsonrpc": "2.0", "method": "notifications/initialized"})

    # запросы клиента: id 2…17
    outcomes = list(rng.permutation(["ok"] * 9 + ["error"] * 3 + ["none"] * 2 + ["late"] * 2))
    ts = 300
    for rid, outcome in zip(range(2, 18), outcomes):
        ts += int(rng.integers(150, 900))
        method = methods[int(rng.integers(0, len(methods)))]
        params = {"name": "search", "arguments": {"q": "agents"}} if method == "tools/call" else {}
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params:
            msg["params"] = params
        add(ts, "C->S", msg)
        respond(ts, rid, outcome, "S->C")

    # запросы сервера клиенту: id снова начинаются с 1 — коллизия номеров с запросами клиента
    s_out = list(rng.permutation(["ok", "error", "none"]))
    for rid, (method, outcome) in enumerate(zip(["ping", "roots/list", "sampling/createMessage"], s_out), start=1):
        sts = int(rng.integers(700, ts))
        add(sts, "S->C", {"jsonrpc": "2.0", "id": rid, "method": method})
        respond(sts, rid, outcome, "C->S")

    # уведомления сервера и «сирота» — ответ на запрос, которого в логе нет
    for _ in range(3):
        add(int(rng.integers(500, ts)), "S->C", {"jsonrpc": "2.0", "method": "notifications/progress",
                                                  "params": {"progress": int(rng.integers(1, 100))}})
    add(int(ts * 0.6), "S->C", {"jsonrpc": "2.0", "id": 77, "result": {}})

    events.sort(key=lambda e: (e[0], e[1]))
    rows = [(i + 1, e[0], e[2], json.dumps(e[3], ensure_ascii=False)) for i, e in enumerate(events)]
    return pd.DataFrame(rows, columns=["line_no", "ts_ms", "direction", "raw"])


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "agent_planning",
        "title": "План агента: порядок подзадач, критический путь и бюджет токенов",
        "data": data_planning,
        "task": """
            Агент-исследователь разбил задачу «подготовить отчёт» на 12 подзадач, которые образуют DAG
            (ориентированный граф без циклов). Файл `input.csv`: `task_id`, `title`, `deps` (номера подзадач,
            от которых зависит эта, через `;`; пусто — зависимостей нет), `tokens` (сколько токенов
            стоит выполнение), `duration` (длительность в условных шагах). Файл `budget.json` —
            `{"token_budget": N}`, общий бюджет токенов на весь план. Подзадачи с независимыми ветками
            в принципе можно выполнять параллельно, но токены тратятся в любом случае.

            Постройте план и разбор его выполнения:

            1. **Порядок.** Топологический порядок (алгоритм Кана): на каждом шаге из всех готовых подзадач
               (все зависимости уже в порядке) берите подзадачу с **наименьшим `task_id`**. Номер в порядке — `order` (с 1).
            2. **Расписание без ограничений по ресурсам.** `start` = максимум `finish` среди зависимостей (0, если их нет),
               `finish = start + duration`. Общая длительность плана T = максимальный `finish`.
            3. **Запас.** Для каждой подзадачи найдите самое позднее время окончания `LF`, при котором план всё
               ещё укладывается в T: для подзадач без «детей» `LF = T`, иначе `LF` = минимум по детям
               (`LF(ребёнка) − duration(ребёнка)`). `slack = LF − finish`. Подзадачи с `slack = 0` лежат
               на критическом пути.
            4. **Бюджет.** Проходите подзадачи в порядке `order` и копите потраченные токены. Статус подзадачи:
               - `blocked` — хотя бы одна зависимость не получила статус `done`;
               - иначе `skipped_budget`, если `spent + tokens > token_budget`;
               - иначе `done` (тогда `spent += tokens`).
               Дешёвая подзадача после пропущенной дорогой может поместиться в бюджет — алгоритм не останавливается.

            ## Подсказки

            - Храните зависимости словарём `{task_id: [id, ...]}`, а для алгоритма Кана — ещё и словарь «детей» и счётчик
              входящих рёбер. Готовые подзадачи кладите в кучу `heapq`, тогда `heappop` сам даст наименьший номер.
            - Прямой проход (start/finish) идёт по `order`, обратный проход (LF) — по `reversed(order)`:
              `lf[t] = min((lf[c] - duration[c] for c in children[t]), default=T)`.
            - Статус считайте отдельным циклом по `order`: `blocked` проверяйте раньше бюджета, а `spent` увеличивайте
              только для `done`. В колонку `spent` пишите накопленную сумму **после** обработки подзадачи.

            ## Ожидаемый формат выходного файла

            CSV с колонками `order, task_id, start, finish, slack, status, spent` — 12 строк в порядке `order`
            (1…12). `status` — `done`, `skipped_budget` или `blocked`. Все значения — целые числа или строки.
        """,
        "solution": """
            import heapq
            import json


            def main():
                df = pd.read_csv('input.csv', dtype={'deps': str}, keep_default_na=False)
                budget = json.load(open('budget.json', encoding='utf-8'))['token_budget']

                deps = {int(r.task_id): [int(x) for x in r.deps.split(';') if x] for r in df.itertuples()}
                dur = dict(zip(df['task_id'], df['duration']))
                tok = dict(zip(df['task_id'], df['tokens']))
                children = {t: [] for t in deps}
                for t, ps in deps.items():
                    for p in ps:
                        children[p].append(t)

                # 1. алгоритм Кана: из готовых берём наименьший task_id
                indeg = {t: len(ps) for t, ps in deps.items()}
                ready = [t for t, k in indeg.items() if k == 0]
                heapq.heapify(ready)
                order = []
                while ready:
                    t = heapq.heappop(ready)
                    order.append(t)
                    for c in children[t]:
                        indeg[c] -= 1
                        if indeg[c] == 0:
                            heapq.heappush(ready, c)
                assert len(order) == len(deps), 'в графе есть цикл'

                # 2. прямой проход: самое раннее начало и окончание
                start, finish = {}, {}
                for t in order:
                    start[t] = max((finish[p] for p in deps[t]), default=0)
                    finish[t] = start[t] + dur[t]
                total = max(finish.values())

                # 3. обратный проход: самое позднее окончание и запас
                lf = {}
                for t in reversed(order):
                    lf[t] = min((lf[c] - dur[c] for c in children[t]), default=total)

                # 4. бюджет токенов
                done, spent, rows = set(), 0, []
                for i, t in enumerate(order, start=1):
                    if any(p not in done for p in deps[t]):
                        status = 'blocked'
                    elif spent + tok[t] > budget:
                        status = 'skipped_budget'
                    else:
                        status = 'done'
                        done.add(t)
                        spent += int(tok[t])
                    rows.append({'order': i, 'task_id': t, 'start': int(start[t]), 'finish': int(finish[t]),
                                 'slack': int(lf[t] - finish[t]), 'status': status, 'spent': spent})
                print('длительность плана:', total, '| потрачено токенов:', spent, 'из', budget)
                return pd.DataFrame(rows)
        """,
        "review": """
            **Идея.** План агента — это DAG: у подзадачи есть предпосылки, и порядок выполнения обязан их уважать.
            Топологическая сортировка даёт допустимый порядок, прямой и обратный проходы — самые ранние и самые
            поздние сроки. Подзадачи с нулевым запасом образуют критический путь: ускорять нужно именно их, а
            остальные можно выполнять параллельно или чуть позже. Бюджет токенов — второе ограничение: пропуск
            одной подзадачи «блокирует» всех её потомков, поэтому дорогой узел в начале цепочки опаснее дорогого в конце.

            **Типичные ошибки:**
            - Сортировка по `task_id` вместо топологической — зависимая подзадача оказывается раньше своей предпосылки.
            - Остановка цикла на первой пропущенной подзадаче: более дешёвые независимые подзадачи ещё могут поместиться.
            - Проверка бюджета раньше проверки `blocked`: подзадача без выполненных предпосылок не должна тратить бюджет.
            - Запас считается только вперёд (по `finish`) — для `slack` нужен ещё и обратный проход от T.

            **Что вынести.** Хороший планировщик агента явно хранит зависимости, знает критический путь и держит
            бюджет шагов и токенов — иначе длинный план легко обрывается на полпути с половиной результата.
        """,
    },
    {
        "name": "multi_agent_systems",
        "title": "Стоимость координации: supervisor, все-со-всеми и конвейер",
        "data": data_multi_agent,
        "task": """
            Команда из 7 LLM-агентов решает задачу. Файл `input.csv`: `agent_id`, `name`, `role`
            (`orchestrator` — ровно один, остальные `worker`), `msg_tokens` (длина сообщения, которое
            агент отправляет за один раз) и `price_per_1k` (цена в долларах за 1000 входных токенов
            модели этого агента: платит тот, кто **читает** сообщение).

            Сравните три схемы координации. Один раунд устроен так:

            - `supervisor` — оркестратор шлёт задачу каждому исполнителю, и каждый исполнитель отвечает оркестратору;
            - `all_to_all` — каждый агент отправляет своё сообщение каждому другому агенту;
            - `pipeline` — агенты по возрастанию `agent_id` образуют цепочку: каждый отправляет сообщение следующему
              (последний — никому).

            Сообщение длиной `msg_tokens` отправителя считается за `msg_tokens` токенов; его стоимость —
            `msg_tokens · price_per_1k(получателя) / 1000`. Всего **3 раунда** (схема повторяется без изменений).
            Для каждой схемы посчитайте:

            - `messages` — число сообщений за 3 раунда;
            - `tokens` — суммарное число токенов во всех сообщениях;
            - `cost_usd` — суммарная стоимость, округление до 4 знаков;
            - `busiest_agent` — `agent_id` агента, получившего больше всех токенов (при равенстве — меньший `agent_id`);
            - `busiest_tokens` — сколько токенов он получил за 3 раунда.

            ## Подсказки

            - Представьте каждую схему списком рёбер `(отправитель, получатель)` за один раунд: у `supervisor` их `2·k`,
              у `all_to_all` — `n·(n−1)`, у `pipeline` — `n−1`. Число сообщений за 3 раунда — длина списка · 3.
            - Один цикл по рёбрам даёт всё сразу: `tokens += msg[s]·3`, `cost += msg[s]·3·price[r]/1000`,
              `inbox[r] += msg[s]·3`. Оркестратора найдите по `role`, а не по номеру строки.
            - Цену берите у получателя, а не у отправителя. Для `busiest_agent` используйте
              `min(ids, key=lambda a: (-inbox[a], a))`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `scheme, messages, tokens, cost_usd, busiest_agent, busiest_tokens` — 3 строки
            в порядке `supervisor`, `all_to_all`, `pipeline`.
        """,
        "solution": """
            ROUNDS = 3


            def main():
                ag = pd.read_csv('input.csv').sort_values('agent_id')
                ids = [int(a) for a in ag['agent_id']]
                msg = dict(zip(ids, ag['msg_tokens']))
                price = dict(zip(ids, ag['price_per_1k']))
                boss = int(ag.loc[ag['role'] == 'orchestrator', 'agent_id'].iloc[0])
                workers = [a for a in ids if a != boss]

                schemes = {
                    'supervisor': [(boss, w) for w in workers] + [(w, boss) for w in workers],
                    'all_to_all': [(a, b) for a in ids for b in ids if a != b],
                    'pipeline': list(zip(ids, ids[1:])),
                }
                rows = []
                for name, edges in schemes.items():
                    inbox = {a: 0 for a in ids}
                    tokens, cost = 0, 0.0
                    for s, r in edges:
                        t = int(msg[s]) * ROUNDS
                        tokens += t
                        inbox[r] += t
                        cost += t * price[r] / 1000
                    top = min(ids, key=lambda a: (-inbox[a], a))
                    rows.append({'scheme': name, 'messages': len(edges) * ROUNDS, 'tokens': tokens,
                                 'cost_usd': round(cost, 4), 'busiest_agent': top, 'busiest_tokens': inbox[top]})
                return pd.DataFrame(rows)
        """,
        "review": """
            **Идея.** Число сообщений у схем растёт по-разному: у `supervisor` линейно (2·k), у `all_to_all` квадратично
            (n·(n−1)), у конвейера линейно (n−1). Каждое сообщение — это токены, которые кто-то читает и оплачивает,
            поэтому «все-со-всеми» дорожает быстрее всего. Зато у `supervisor` появляется узкое место: оркестратор
            получает ответы всех исполнителей и его контекст растёт быстрее остальных — именно он оказывается
            «busiest» и первым упирается в лимит контекста.

            **Типичные ошибки:**
            - Цена отправителя вместо цены получателя.
            - Оркестратор определён как первая строка таблицы, а не по колонке `role`.
            - В `supervisor` забыты ответы исполнителей (получилось `k`, а не `2·k` сообщений за раунд).
            - Раунды не учтены ни в сообщениях, ни в токенах, ни в стоимости.

            **Что вынести.** Перед тем как добавлять агентов, посчитайте сообщения и токены схемы: стоимость координации
            нередко превышает выигрыш от разделения работы. Если задача решается одним агентом — многоагентность лишняя.
        """,
    },
    {
        "name": "mcp_protocol",
        "title": "Разбор лога JSON-RPC: запросы, ответы, ошибки и таймауты",
        "data": data_mcp,
        "task": """
            MCP-подобный обмен между клиентом и сервером записан в `input.csv`: `line_no`, `ts_ms` (время сообщения
            в миллисекундах), `direction` (`C->S` — от клиента серверу, `S->C` — от сервера клиенту) и `raw` —
            само сообщение в формате JSON-RPC 2.0. Типы сообщений:

            - **запрос** — есть `method` и `id`;
            - **уведомление** — есть `method`, но нет `id` (ответа не ждёт);
            - **ответ** — нет `method`, есть `id` и либо `result`, либо `error` (с полями `code` и `message`).

            Важно: номера `id` назначает каждая сторона независимо, поэтому запрос клиента с `id = 1` и запрос
            сервера с `id = 1` — разные запросы. Ответ идёт в **противоположном** направлении относительно запроса
            и несёт тот же `id`.

            Для каждого запроса (от любой стороны) найдите ответ и определите итог. Таймаут — 5000 мс:

            - `timeout` — ответа нет совсем **или** он пришёл позже чем через 5000 мс после запроса (строго больше);
            - иначе `error`, если ответ содержит поле `error`;
            - иначе `ok`.

            Ответы, для которых в логе нет запроса (с тем же `id` от противоположной стороны), игнорируйте.

            ## Подсказки

            - Разберите каждую строку через `json.loads(raw)`. Различайте типы по наличию ключей `method` и `id`
              (`'id' in msg`), а не по значению — `id` бывает равен 0 или 1.
            - Заведите словарь ожидающих запросов с ключом `(направление запроса, id)`. Ответ с направлением `S->C` ищите
              по ключу `('C->S', id)`, а ответ `C->S` — по ключу `('S->C', id)`.
            - Задержка — `ts_ms ответа − ts_ms запроса`. Статус вычисляйте после обхода всего лога; запросы без ответа
              получают `timeout` и пустые `latency_ms` и `error_code`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `line_no, id, method, requester, status, latency_ms, error_code` — по одной строке на запрос
            в порядке `line_no` запроса. `line_no` — номер строки **запроса**, `requester` — `client` или `server`,
            `status` — `ok`, `error` или `timeout`, `latency_ms` — задержка ответа (пусто, если ответа нет),
            `error_code` — `code` из ответа с ошибкой (иначе пусто). Числа — целые.
        """,
        "solution": """
            import json

            TIMEOUT_MS = 5000


            def main():
                log = pd.read_csv('input.csv')
                pending, rows = {}, []
                for r in log.itertuples():
                    msg = json.loads(r.raw)
                    if 'method' in msg and 'id' in msg:                       # запрос
                        row = {'line_no': int(r.line_no), 'id': int(msg['id']), 'method': msg['method'],
                               'requester': 'client' if r.direction == 'C->S' else 'server',
                               'ts': int(r.ts_ms), 'latency_ms': None, 'error_code': None}
                        pending[(r.direction, msg['id'])] = row
                        rows.append(row)
                    elif 'method' not in msg and 'id' in msg:                 # ответ идёт в обратную сторону
                        req_dir = 'S->C' if r.direction == 'C->S' else 'C->S'
                        row = pending.get((req_dir, msg['id']))
                        if row is None or row['latency_ms'] is not None:      # «сирота» или повторный ответ
                            continue
                        row['latency_ms'] = int(r.ts_ms) - row['ts']
                        if 'error' in msg:
                            row['error_code'] = int(msg['error']['code'])
                    # уведомления (method без id) ответа не ждут

                for row in rows:
                    if row['latency_ms'] is None or row['latency_ms'] > TIMEOUT_MS:
                        row['status'] = 'timeout'
                    else:
                        row['status'] = 'error' if row['error_code'] is not None else 'ok'
                out = pd.DataFrame(rows).drop(columns='ts')[['line_no', 'id', 'method', 'requester', 'status',
                                                             'latency_ms', 'error_code']]
                out['latency_ms'] = out['latency_ms'].astype('Int64')
                out['error_code'] = out['error_code'].astype('Int64')
                print(out['status'].value_counts().to_dict())
                return out
        """,
        "review": """
            **Идея.** JSON-RPC 2.0 различает три вида сообщений: запрос (`method` + `id`), уведомление (`method` без `id`)
            и ответ (`id` + `result` или `error`). Ответ привязывается к запросу по `id`, а поскольку `id` выдаёт каждая
            сторона сама, ключ сопоставления — пара «кто спрашивал + id». Протокол MCP построен именно на таком обмене:
            клиент вызывает `tools/call`, `resources/read` и т. д., а сервер в ответ присылает результат или ошибку.
            Таймауты в протокол не вшиты — их обычно задаёт клиент, поэтому в логе «зависший» вызов выглядит
            как запрос без ответа либо с очень поздним ответом.

            **Типичные ошибки:**
            - Сопоставление только по `id`: запрос сервера с `id = 1` «забирает» ответ на запрос клиента с `id = 1`.
            - Уведомления посчитаны запросами (у них нет `id`, ответа не будет — это не таймаут).
            - Поздний ответ (больше 5000 мс) записан как `ok`; граница строгая: ровно 5000 мс ещё не таймаут.
            - Код ошибки перепутан со статусом: `error_code` заполняется только для ответов с полем `error`.

            **Что вынести.** При отладке агента читайте журнал обмена как таблицу: запрос, ответ, задержка, код ошибки.
            Большинство «странных» зависаний видно сразу — это запросы без ответа и ответы, пришедшие слишком поздно.
        """,
    },
]


def main():
    print("Генерация упражнений по ИИ-агентам (группа C):")
    for seed, exercise in enumerate(EXERCISE_LIST, start=961):
        build(exercise, seed)


if __name__ == "__main__":
    main()
