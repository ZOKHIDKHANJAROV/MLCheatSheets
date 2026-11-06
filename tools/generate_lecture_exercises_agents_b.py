#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям группы «ИИ-агенты, часть b»: RAG, векторный поиск, память агента —
только NumPy, pandas и стандартная библиотека (без сети и LLM, результат детерминирован).

Темы: чанкинг с перекрытием + recall@k и MRR; точный поиск по косинусу против IVF-подобного
приближения (recall@k); симуляция буфера памяти с вытеснением (FIFO, LRU, по важности, смешанное).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_agents_b.py
"""

import numpy as np
import pandas as pd

from generate_basic_exercises import build


# ---------------------------------------------------------------- данные

FILLER = ["система", "данные", "запрос", "сервер", "модель", "ошибка", "версия", "поток", "сеть", "файл",
          "доступ", "журнал", "задача", "проверка", "сервис", "память", "кэш", "очередь", "отчёт", "узел",
          "канал", "пакет", "режим", "сигнал"]


def data_rag(rng):
    docs, queries = [], []
    for i in range(1, 9):
        words = [str(w) for w in rng.choice(FILLER, 60)]
        topic, answer = f"тема{i}", f"код{i}"
        pos = int(rng.integers(5, 52))
        words[pos:pos] = [topic, "имеет", "кодовое", "значение", answer]   # «факт» из пяти слов
        docs.append({"doc_id": i, "text": " ".join(words)})
        for _ in range(2):
            extra = [str(w) for w in rng.choice(FILLER, 2, replace=False)]
            queries.append({"doc_id": i, "answer": answer, "query": " ".join([topic] + extra)})
    queries = [queries[j] for j in rng.permutation(len(queries))]
    for n, q in enumerate(queries, start=1):
        q["query_id"] = n
    qdf = pd.DataFrame(queries)[["query_id", "query", "doc_id", "answer"]]
    return {"input.csv": pd.DataFrame(docs), "queries.csv": qdf}


def data_vector_search(rng):
    d, n_cells, per_cell = 8, 6, 50
    centers = rng.normal(0, 1.1, (n_cells, d))
    labels = np.repeat(np.arange(n_cells), per_cell)
    X = centers[labels] + rng.normal(0, 0.9, (len(labels), d))
    X *= rng.uniform(0.6, 2.0, (len(labels), 1))               # разные длины векторов
    order = rng.permutation(len(X))
    X = X[order].round(3)
    base = pd.DataFrame(X, columns=[f"x{j}" for j in range(d)])
    base.insert(0, "id", np.arange(1, len(X) + 1))

    q_labels = rng.integers(0, n_cells, 20)
    Q = (centers[q_labels] + rng.normal(0, 0.9, (20, d))) * rng.uniform(0.6, 2.0, (20, 1))
    queries = pd.DataFrame(Q.round(3), columns=[f"x{j}" for j in range(d)])
    queries.insert(0, "qid", np.arange(1, 21))

    cents = (centers / np.linalg.norm(centers, axis=1, keepdims=True)).round(3)
    cdf = pd.DataFrame(cents, columns=[f"x{j}" for j in range(d)])
    cdf.insert(0, "cell", np.arange(n_cells))
    return {"input.csv": base, "queries.csv": queries, "centroids.csv": cdf}


MEMORY_KEYS = {  # ключ: (базовая важность, вес популярности)
    "имя": (0.9, 3), "город": (0.5, 1), "язык": (0.8, 3), "часовой_пояс": (0.6, 1), "стиль_ответов": (0.9, 3),
    "цель": (0.7, 2), "дедлайн": (0.8, 2), "команда": (0.3, 1), "инструмент": (0.4, 2), "бюджет": (0.6, 1),
    "риск": (0.3, 1), "заметка": (0.1, 4),
}


def data_agent_memory(rng):
    keys = list(MEMORY_KEYS)
    weights = np.array([MEMORY_KEYS[k][1] for k in keys], float)
    weights /= weights.sum()
    rows = []
    for t in range(1, 121):
        key = str(rng.choice(keys, p=weights))
        if rng.random() < 0.4:
            imp = float(np.clip(MEMORY_KEYS[key][0] + rng.normal(0, 0.05), 0.0, 1.0))
            rows.append({"t": t, "kind": "write", "key": key, "importance": round(imp, 2)})
        else:
            rows.append({"t": t, "kind": "read", "key": key, "importance": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "rag",
        "title": "RAG: чанкинг с перекрытием, recall@k и MRR",
        "data": data_rag,
        "task": """
            Мини-RAG без нейросетей: нарезаем документы на чанки, ищем чанки по запросу и оцениваем поиск.

            `input.csv` — 8 документов (`doc_id, text`, слова разделены пробелом, по 65 слов). В каждом документе спрятан
            «факт» из пяти слов вида `темаN имеет кодовое значение кодN`; вокруг — случайные слова-заполнители.
            `queries.csv` — 16 запросов (`query_id, query, doc_id, answer`): в запросе слово `темаN` и два случайных слова,
            `doc_id` — документ с ответом, `answer` — слово-ответ (`кодN`, в запросе его нет).

            **Чанкинг.** Для документа из слов w₀…w₍ₘ₋₁₎ чанки — это срезы `words[i : i + size]` для
            `i = 0, step, 2·step, …` пока `i < max(m − overlap, 1)`, где `step = size − overlap`. Последний чанк может быть
            короче. Чанки нумеруются сквозным индексом: сначала все чанки документа 1, затем документа 2 и так далее.

            **Релевантные чанки запроса** — чанки документа `doc_id`, в которых встречается слово `answer`
            (при перекрытии таких чанков бывает два).

            **Поиск.** Пусть N — общее число чанков, nₜ — число чанков, содержащих слово t.
            Вес слова: IDF(t) = ln(1 + (N − nₜ + 0.5) / (nₜ + 0.5)). Оценка чанка для запроса — сумма IDF по **различным**
            словам запроса, которые встречаются в чанке (повторы слов внутри чанка не учитываются). Чанки сортируются
            по убыванию оценки; при равенстве — по возрастанию индекса чанка.

            **Метрики** (усредняются по 16 запросам), K = 3:
            - `recall_at_3` = (число релевантных чанков среди 3 лучших) / (число всех релевантных чанков запроса);
            - `mrr` = среднее от 1 / rank, где rank — место первого релевантного чанка в полном ранжировании (с единицы).

            Посчитайте метрики для четырёх конфигураций чанкинга: (size, overlap) = (10, 0), (10, 3), (20, 0), (20, 5).

            ## Подсказки

            - Сначала только чанкинг: при size = 10 и overlap = 0 из документа в 65 слов получается 7 чанков, при overlap = 3
              шаг равен 7 — проверьте число чанков вручную.
            - IDF считайте отдельно для каждой конфигурации (N и nₜ зависят от разбиения). Сумму по словам запроса ведите
              в одном и том же порядке (`sorted(set(query.split()))`) и округляйте оценку до 9 знаков — иначе шум
              плавающей точки сломает ничьи.
            - Для каждого запроса: `order = sorted(range(N), key=lambda i: (-score[i], i))`, `relevant = {i : owner[i] == doc_id
              and answer in chunk[i]}`, `rank = первый номер в order, где i in relevant`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `size, overlap, n_chunks, recall_at_3, mrr` — 4 строки в порядке конфигураций из условия.
            `n_chunks` — общее число чанков N, метрики округлить до 4 знаков.
        """,
        "solution": """
            CONFIGS = [(10, 0), (10, 3), (20, 0), (20, 5)]
            K = 3


            def chunk(words, size, overlap):
                step = size - overlap
                return [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]


            def evaluate(docs, queries, size, overlap):
                chunks, owner = [], []                          # сквозная нумерация: документ 1, 2, ...
                for doc_id, text in zip(docs['doc_id'], docs['text']):
                    for c in chunk(text.split(), size, overlap):
                        chunks.append(set(c))
                        owner.append(doc_id)
                n = len(chunks)
                df = {}
                for c in chunks:
                    for w in c:
                        df[w] = df.get(w, 0) + 1
                idf = lambda w: np.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))

                recalls, rr = [], []
                for q in queries.itertuples():
                    terms = sorted(set(q.query.split()))
                    scores = [round(sum(idf(w) for w in terms if w in c), 9) for c in chunks]
                    order = sorted(range(n), key=lambda i: (-scores[i], i))
                    relevant = {i for i in range(n) if owner[i] == q.doc_id and q.answer in chunks[i]}
                    recalls.append(len(set(order[:K]) & relevant) / len(relevant))
                    rank = next(r for r, i in enumerate(order, 1) if i in relevant)
                    rr.append(1 / rank)
                return n, np.mean(recalls), np.mean(rr)


            def main():
                docs = pd.read_csv('input.csv')
                queries = pd.read_csv('queries.csv')
                rows = []
                for size, overlap in CONFIGS:
                    n, recall, mrr = evaluate(docs, queries, size, overlap)
                    rows.append({'size': size, 'overlap': overlap, 'n_chunks': n,
                                 'recall_at_3': recall, 'mrr': mrr})
                    print(f"size={size:2d} overlap={overlap}: чанков {n}, recall@3 = {recall:.3f}, MRR = {mrr:.3f}")
                result = pd.DataFrame(rows)
                result[['recall_at_3', 'mrr']] = result[['recall_at_3', 'mrr']].round(4)
                return result
        """,
        "review": """
            **Идея.** Качество RAG начинается с поиска: если нужный фрагмент не попал в top-k, модель его не увидит. Чанкинг
            определяет, что вообще можно найти. «Факт» из пяти слов может оказаться на стыке двух чанков: слово-тема в одном,
            слово-ответ в другом — тогда чанк с ответом не похож на запрос и стоит низко. Перекрытие повторяет границу,
            и такой разрыв случается реже; зато чанков становится больше и в индексе появляются дубли. Сравните строки
            таблицы: при size = 10 перекрытие помогает, а при size = 20 — нет, поэтому параметры подбирают по метрикам
            на своих данных, а не по общему правилу.

            **Типичные ошибки:**
            - Шаг окна равен `size`, а не `size − overlap` — перекрытия нет, число чанков другое.
            - Один IDF на все конфигурации: N и nₜ меняются вместе с разбиением.
            - Релевантными считают весь документ, а не чанки со словом-ответом, или только один чанк при перекрытии.
            - Ранг считают среди top-3, а не по полному ранжированию (MRR получается заниженным или нулевым).
            - Забывают про различные слова запроса и считают повторы слов в чанке несколько раз.

            **Что вынести:** recall@k и MRR нужны, чтобы подбирать size и overlap по данным, а не «на глаз»; крупные чанки
            дают меньше разрывов, но размывают оценку; перекрытие защищает границы ценой дублей. В реальных системах вместо
            IDF-суммы используют BM25 и эмбеддинги, а потом реранкинг, но метрики считаются так же. → шпаргалка «RAG».
        """,
    },
    {
        "name": "vector_search",
        "title": "Векторный поиск: косинус, нормализация и IVF-подобное приближение",
        "data": data_vector_search,
        "task": """
            `input.csv` — 300 векторов размерности 8 (`id, x0…x7`), длины у них разные (отличаются до 3 раз). `queries.csv` —
            20 запросов (`qid, x0…x7`). `centroids.csv` — 6 готовых центроидов ячеек IVF (`cell, x0…x7`).

            1. **Нормализация.** Приведите к единичной длине (L2) векторы базы, запросы и центроиды.
            2. **Точный поиск.** Для каждого запроса найдите 5 ближайших векторов базы по косинусному сходству
               (после нормализации это скалярное произведение). При равенстве — меньший `id` первым.
            3. **Нормализация против сырого скалярного произведения.** Посчитайте, у скольких запросов вектор с наибольшим
               **сырым** скалярным произведением (без нормализации) отличается от лучшего вектора по косинусу.
            4. **IVF-подобное приближение.** Каждый вектор базы относится к ячейке с наибольшим косинусом к центроиду
               (центроиды нормализованы). Для запроса берутся `nprobe` ячеек с наибольшим косинусом к запросу; кандидаты —
               векторы этих ячеек; из них — 5 лучших по косинусу. Для `nprobe` = 1, 2, 3 и 6 посчитайте:
               - `recall_at_5` = (доля точных 5 соседей, найденных приближением), усреднённая по запросам;
               - `scanned_share` = (число кандидатов) / 300, усреднённое по запросам.

            ## Подсказки

            - Нормализация строк: `M / np.linalg.norm(M, axis=1, keepdims=True)`. Матрица косинусов запросов к базе —
              `Qn @ Xn.T` (20 × 300); лучшие соседи — `np.argsort(-cos, axis=1, kind='stable')[:, :5]`.
            - Ячейка вектора — `np.argmax(Xn @ Cn.T, axis=1)`. Для запроса: `probes = np.argsort(-(Cn @ qn))[:nprobe]`,
              кандидаты — `np.flatnonzero(np.isin(cell, probes))`.
            - При `nprobe` = 6 просматриваются все ячейки, поэтому recall должен быть ровно 1.0, а доля просмотренного — 1.0:
              это хорошая проверка. Строка базы с `id = k` лежит в позиции `k − 1`, поэтому сравнивайте позиции, а не `id`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `name, value` — 9 строк в таком порядке: `top1_differs_dot_vs_cosine` (целое число запросов),
            затем для nprobe = 1, 2, 3, 6 пары `recall_at_5_nprobe_N` и `scanned_share_nprobe_N`. Значения округлить до 4 знаков.
        """,
        "solution": """
            K, NPROBES = 5, [1, 2, 3, 6]


            def normalize(M):
                return M / np.linalg.norm(M, axis=1, keepdims=True)


            def main():
                base = pd.read_csv('input.csv')
                queries = pd.read_csv('queries.csv')
                cents = pd.read_csv('centroids.csv')
                cols = [f'x{j}' for j in range(8)]
                X, Q, C = base[cols].to_numpy(), queries[cols].to_numpy(), cents[cols].to_numpy()
                Xn, Qn, Cn = normalize(X), normalize(Q), normalize(C)

                cos = Qn @ Xn.T                                          # косинусы: 20 × 300
                exact = np.argsort(-cos, axis=1, kind='stable')[:, :K]   # точный top-5 (позиции в базе)

                raw_best = np.argmax(Q @ X.T, axis=1)                    # лучший по сырому скалярному произведению
                differs = int((raw_best != exact[:, 0]).sum())

                cell = np.argmax(Xn @ Cn.T, axis=1)                      # ячейка каждого вектора базы
                rows = [{'name': 'top1_differs_dot_vs_cosine', 'value': differs}]
                for nprobe in NPROBES:
                    recalls, scanned = [], []
                    for i in range(len(Q)):
                        probes = np.argsort(-(Cn @ Qn[i]), kind='stable')[:nprobe]
                        cand = np.flatnonzero(np.isin(cell, probes))
                        top = cand[np.argsort(-cos[i, cand], kind='stable')[:K]]
                        recalls.append(len(set(top) & set(exact[i])) / K)
                        scanned.append(len(cand) / len(X))
                    rows.append({'name': f'recall_at_5_nprobe_{nprobe}', 'value': np.mean(recalls)})
                    rows.append({'name': f'scanned_share_nprobe_{nprobe}', 'value': np.mean(scanned)})
                    print(f"nprobe={nprobe}: recall@5 = {np.mean(recalls):.3f}, просмотрено {100 * np.mean(scanned):.0f}% базы")

                print(f"лучший по сырому скалярному произведению и по косинусу различаются у {differs} запросов из {len(Q)}")
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Точный перебор сравнивает запрос со всеми векторами и служит эталоном. IVF делит базу на ячейки вокруг
            центроидов и смотрит только `nprobe` ближайших: сравнений меньше, но сосед из непросмотренной ячейки теряется.
            Параметр `nprobe` — рычаг обмена скорости на recall; при `nprobe` = числу ячеек получается точный поиск.

            **Типичные ошибки:**
            - Не нормализовали векторы: скалярное произведение отдаёт предпочтение длинным векторам, и «ближайшим» становится
              не самый похожий по направлению (именно это считает `top1_differs_dot_vs_cosine`).
            - Нормализовали базу, но не запросы или центроиды — косинус сравнивается с ненормированными величинами.
            - Сравнивали `id` с позициями в массиве (`id` начинается с 1) — recall получается заниженным.
            - Выбирали ячейки запроса по L2-расстоянию до ненормированных центроидов — другой набор ячеек.
            - Считали долю просмотренных векторов от числа кандидатов, а не от размера базы.

            **Что вынести:** нормализация делает косинус, скалярное произведение и L2 согласованными; ANN всегда измеряют
            через recall@k относительно точного поиска; кривая «recall — доля просмотренного» помогает выбрать nprobe.
            В настоящем индексе центроиды получают k-means, а HNSW и PQ дают другие способы обменять точность на скорость.
            → шпаргалка «Векторный поиск».
        """,
    },
    {
        "name": "agent_memory",
        "title": "Память агента: буфер с вытеснением и подсчёт попаданий",
        "data": data_agent_memory,
        "task": """
            Агент хранит факты в буфере ограниченной ёмкости (`CAPACITY = 5` записей). `input.csv` — журнал из 120 событий:
            `t, kind, key, importance`. Событие `write` записывает факт `key` с важностью `importance` (от 0 до 1);
            событие `read` обращается к факту `key` (у него `importance` пуста).

            Правила симуляции (события обрабатываются по возрастанию `t`):
            - `write`: если ключ уже в буфере — обновить его важность и время последнего обращения (`last = t`), ничего
              не вытесняя; иначе, если буфер полон, сначала вытеснить одну запись по выбранной политике, затем добавить новую
              (`born = last = t`);
            - `read`: если ключ в буфере — **попадание** (`last = t`), иначе — **промах** (в буфер при промахе ничего не
              добавляется).

            Четыре политики вытеснения (на шаге вытеснения `t` — время текущего события `write`):
            - `fifo` — вытеснить запись с наименьшим `born` (самая давно добавленная; повторная запись `born` не меняет);
            - `lru` — вытеснить запись с наименьшим `last`;
            - `importance` — вытеснить запись с наименьшей важностью; при равенстве — с наименьшим `last`;
            - `mixed` — ценность `value = 0.6·importance + 0.4·0.5^((t − last) / 10)`, вытеснить наименьшую ценность
              (округлите `value` до 9 знаков); при равенстве — с наименьшим `last`.

            Для каждой политики запустите симуляцию с пустого буфера и посчитайте `hits` (число попаданий), `misses`,
            `hit_rate = hits / (hits + misses)` и `evictions` (число вытеснений).

            ## Подсказки

            - Буфер удобно хранить словарём `key → {'born', 'last', 'imp'}`; политику вытеснения вынесите в функцию,
              возвращающую ключ-жертву через `min(buf, key=...)`.
            - Не забудьте, что `read` тоже обновляет `last`, а повторный `write` существующего ключа вытеснения не вызывает.
            - Проверка: `hits + misses` равно числу событий `read` и одинаково у всех политик; `evictions` у политик
              может отличаться, потому что вытесненный ключ может потом снова оказаться «новым» при следующем `write`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `policy, hits, misses, hit_rate, evictions` — 4 строки в порядке `fifo, lru, importance, mixed`.
            `hit_rate` округлить до 4 знаков.
        """,
        "solution": """
            CAPACITY, HALF_LIFE = 5, 10.0


            def victim(policy, buf, t):
                if policy == 'fifo':
                    return min(buf, key=lambda k: buf[k]['born'])
                if policy == 'lru':
                    return min(buf, key=lambda k: buf[k]['last'])
                if policy == 'importance':
                    return min(buf, key=lambda k: (buf[k]['imp'], buf[k]['last']))
                value = lambda k: round(0.6 * buf[k]['imp'] + 0.4 * 0.5 ** ((t - buf[k]['last']) / HALF_LIFE), 9)
                return min(buf, key=lambda k: (value(k), buf[k]['last']))


            def simulate(events, policy):
                buf, hits, misses, evictions = {}, 0, 0, 0
                for e in events.itertuples():
                    if e.kind == 'write':
                        if e.key in buf:
                            buf[e.key].update(imp=e.importance, last=e.t)
                        else:
                            if len(buf) == CAPACITY:
                                del buf[victim(policy, buf, e.t)]
                                evictions += 1
                            buf[e.key] = {'born': e.t, 'last': e.t, 'imp': e.importance}
                    elif e.key in buf:
                        hits += 1
                        buf[e.key]['last'] = e.t
                    else:
                        misses += 1
                return hits, misses, evictions


            def main():
                events = pd.read_csv('input.csv').sort_values('t')
                rows = []
                for policy in ['fifo', 'lru', 'importance', 'mixed']:
                    hits, misses, evictions = simulate(events, policy)
                    rows.append({'policy': policy, 'hits': hits, 'misses': misses,
                                 'hit_rate': hits / (hits + misses), 'evictions': evictions})
                    print(f"{policy:>10}: попаданий {hits}, промахов {misses}, вытеснений {evictions}")
                result = pd.DataFrame(rows)
                result['hit_rate'] = result['hit_rate'].round(4)
                return result
        """,
        "review": """
            **Идея.** Буфер памяти конечен, поэтому при переполнении нужно решить, что забыть. Политика вытеснения определяет,
            какие факты окажутся под рукой в момент обращения. FIFO не смотрит на пользу записи, LRU бережёт недавно
            использованное, «по важности» бережёт ценное, но может держать старьё, а смешанная ценность
            комбинирует важность и свежесть (свежесть затухает экспоненциально с периодом полураспада 10).

            **Типичные ошибки:**
            - `read` не обновляет `last` — LRU превращается в FIFO.
            - При повторной записи существующего ключа вытесняют другую запись или сбрасывают `born`.
            - Добавляют в буфер при промахе чтения — в условии этого нет.
            - Считают `value` с `t` не текущего события, а последнего события буфера.
            - Не разрешают ничьи по `last` — результат зависит от порядка ключей в словаре.

            **Что вынести:** разные политики дают разный hit rate на одном и том же журнале; выбор зависит от того, насколько
            важность и повторные обращения предсказывают будущую пользу. Поэтому в памяти агента хранят и метаданные записи
            (время, важность, число обращений), а политику забывания проверяют на журнале реальных обращений.
            → шпаргалка «Память агента».
        """,
    },
]


def main():
    print("Генерация упражнений группы «ИИ-агенты, часть b»:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=931):
        build(exercise, seed)


if __name__ == "__main__":
    main()
