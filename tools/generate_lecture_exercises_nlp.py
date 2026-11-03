#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям по NLP, трансформерам и фреймворкам обучения — только NumPy, pandas,
scikit-learn и SciPy (без PyTorch и TensorFlow).

Темы: аддитивное внимание (Bahdanau), multi-head attention, MLM-маскирование BERT, дообучение головы BERT
с warmup + linear decay, стратегии генерации GPT, позиционное кодирование и Post-LN/Pre-LN блок трансформера,
метрики NER по сущностям, оценка вариантов промпта и self-consistency, BPE, NMF мультипликативными
обновлениями, семантика Trainer в PyTorch Lightning и колбэки Keras.

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_nlp.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build


def to_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- данные

def data_bahdanau(rng):
    words = ["я", "люблю", "читать", "книги", "по", "вечерам"]
    H = rng.normal(0, 1, (6, 3)).round(2)
    enc = pd.DataFrame({"pos": np.arange(1, 7), "word": words, "h0": H[:, 0], "h1": H[:, 1], "h2": H[:, 2]})
    S = rng.normal(0, 1, (3, 2)).round(2)
    dec = pd.DataFrame({"step": [1, 2, 3], "s0": S[:, 0], "s1": S[:, 1]})
    weights = {
        "W_q": rng.normal(0, 0.9, (4, 2)).round(2).tolist(),   # для состояния декодера (размерность 2)
        "W_k": rng.normal(0, 0.9, (4, 3)).round(2).tolist(),   # для состояний энкодера (размерность 3)
        "v": rng.normal(0, 1.0, 4).round(2).tolist(),
    }
    return {"input.csv": enc, "decoder.csv": dec, "weights.json": to_json(weights)}


def data_mha(rng):
    words = ["мама", "мыла", "раму", "утром", "рано"]
    X = rng.normal(0, 1, (5, 4)).round(2)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(4)])
    df.insert(0, "token", words)
    weights = {name: rng.normal(0, 0.7, (4, 4)).round(2).tolist() for name in ["W_Q", "W_K", "W_V", "W_O"]}
    return {"input.csv": df, "weights.json": to_json(weights)}


def data_mlm(rng):
    seqs = [
        "[CLS] кошка спит на тёплом диване [SEP] собака лает во дворе [SEP]",
        "[CLS] утром мы пили горячий чай с лимоном [SEP] [PAD] [PAD] [PAD]",
        "[CLS] студенты сдали трудный экзамен по математике [SEP] преподаватель был доволен [SEP]",
    ]
    rows = [(s, p, tok) for s, text in enumerate(seqs, start=1) for p, tok in enumerate(text.split())]
    df = pd.DataFrame(rows, columns=["seq", "pos", "token"])
    n = len(df)
    special = df["token"].isin(["[CLS]", "[SEP]", "[PAD]"]).to_numpy()
    content_words = sorted(set(df.loc[~special, "token"]))

    u_select = rng.uniform(0.16, 1.0, n)
    chosen = rng.choice(np.flatnonzero(~special), 7, replace=False)
    u_select[chosen] = rng.uniform(0.0, 0.14, 7)
    u_select[rng.choice(np.flatnonzero(special), 2, replace=False)] = rng.uniform(0.0, 0.14, 2)  # ловушка
    u_action = rng.uniform(0, 1, n)
    actions = rng.permutation(["mask"] * 4 + ["random", "random", "keep"])
    for idx, act in zip(chosen, actions):
        u_action[idx] = {"mask": rng.uniform(0.0, 0.79), "random": rng.uniform(0.81, 0.89),
                         "keep": rng.uniform(0.91, 1.0)}[act]
    selected = np.zeros(n, bool)
    selected[chosen] = True
    p_true = np.where(selected, rng.uniform(0.05, 0.6, n), rng.uniform(0.7, 0.99, n))

    df["u_select"] = u_select.round(3)
    df["u_action"] = u_action.round(3)
    df["random_token"] = rng.choice(content_words, n)
    df["p_true"] = p_true.round(3)
    return df


def data_cls_head(rng):
    n, d = 60, 8
    y = rng.permutation(np.repeat([0, 1, 2], 20))
    means = rng.normal(0, 1.0, (3, d))
    X = (means[y] + rng.normal(0, 1.0, (n, d))).round(3)
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(d)])
    df.insert(0, "id", np.arange(1, n + 1))
    df["label"] = y
    return df


def data_gpt_sampling(rng):
    vocab = ["<s>", "кот", "пёс", "спит", "ест", "рыбу", "кость", "."]
    logits = rng.normal(0, 1.5, (8, 8)).round(2)
    logits[:, 0] = -9.0                        # начало последовательности не генерируется
    df = pd.DataFrame(logits, columns=[f"l{j}" for j in range(8)])
    df.insert(0, "token", vocab)
    df.insert(0, "id", np.arange(8))
    u = pd.DataFrame({"step": np.arange(1, 7), "u": rng.uniform(0.02, 0.98, 6).round(3)})
    return {"input.csv": df, "uniforms.csv": u}


def data_transformer_block(rng):
    tokens = ["кошка", "ловит", "мышь", "."]
    E = rng.normal(0, 1, (4, 6)).round(2)
    df = pd.DataFrame(E, columns=[f"e{j}" for j in range(6)])
    df.insert(0, "token", tokens)
    weights = {
        "W1": rng.normal(0, 0.4, (6, 12)).round(2).tolist(),
        "b1": rng.normal(0, 0.1, 12).round(2).tolist(),
        "W2": rng.normal(0, 0.3, (12, 6)).round(2).tolist(),
        "b2": rng.normal(0, 0.1, 6).round(2).tolist(),
        "gamma": rng.uniform(0.8, 1.2, 6).round(2).tolist(),
        "beta": rng.normal(0, 0.1, 6).round(2).tolist(),
    }
    return {"input.csv": df, "weights.json": to_json(weights)}


def data_ner(rng):
    sents = [
        ("Анна Петрова работает в Сбербанке в Москве .",
         "B-PER I-PER O O B-ORG O B-LOC O",
         "B-PER O O O B-ORG O B-LOC O"),
        ("Илон Маск основал SpaceX в Калифорнии .",
         "B-PER I-PER O B-ORG O B-LOC O",
         "B-PER I-PER O B-LOC O B-LOC O"),
        ("Вчера Мария и Иван Иванов прилетели в Нижний Новгород .",
         "O B-PER O B-PER I-PER O O B-LOC I-LOC O",
         "O B-PER O I-PER I-PER O O B-LOC I-LOC O"),
        ("Компания Яндекс открыла офис в Казани .",
         "O B-ORG O O O B-LOC O",
         "B-ORG I-ORG O O O O O"),
        ("Президент Франции встретился с Олафом Шольцем в Берлине .",
         "O B-LOC O O B-PER I-PER O B-LOC O",
         "O B-LOC O O B-PER B-PER O B-LOC O"),
        ("Об этом сообщили ТАСС и Газпром .",
         "O O O B-ORG O B-ORG O",
         "B-PER O B-ORG I-ORG O B-ORG O"),
        ("Мы гуляли по Красной площади .",
         "O O O B-LOC I-LOC O",
         "O O O B-LOC I-LOC O"),
    ]
    rows = []
    for sid, (text, gold, pred) in enumerate(sents, start=1):
        toks, g, p = text.split(), gold.split(), pred.split()
        assert len(toks) == len(g) == len(p)
        rows += [(sid, i, t, a, b) for i, (t, a, b) in enumerate(zip(toks, g, p))]
    return pd.DataFrame(rows, columns=["sent_id", "token_id", "token", "gold", "pred"])


def data_prompts(rng):
    qs, outs = [], []
    variants = {
        "zero_shot": (0.5, ["{v}", "{v}.", " {v} ", "Ответ: {v}", "Думаю, получится {v}", "{v} конфет"],
                      [0.35, 0.15, 0.1, 0.15, 0.15, 0.1]),
        "few_shot": (0.62, ["{v}", "{v}.", "Ответ: {v}"], [0.7, 0.15, 0.15]),
        "cot": (0.72, ["Было {a}, купила {b}·{c} = {bc}. Итого {a} + {bc}. Ответ: {v}.",
                       "Сначала {b}·{c} = {bc}, потом прибавим {a}. Ответ: {v}",
                       "Посчитаем: {a} + {b}·{c}, итого {v}"], [0.6, 0.32, 0.08]),
    }
    for qid in range(1, 21):
        a, b, c = int(rng.integers(5, 30)), int(rng.integers(2, 6)), int(rng.integers(2, 9))
        ans = a + b * c
        qs.append({"question_id": qid, "question": f"У Маши было {a} конфет. Она купила ещё {b} пачки по {c} конфет. "
                   f"Сколько конфет у неё стало?", "answer": ans})
        wrong_pool = [ans - 1, ans + 1, ans + c, a + b + c, a * b]
        for variant, (p, templates, probs) in variants.items():
            for sample in range(1, 6):
                v = ans if rng.random() < p else wrong_pool[int(rng.integers(0, len(wrong_pool)))]
                t = templates[rng.choice(len(templates), p=probs)]
                outs.append({"question_id": qid, "variant": variant, "sample": sample,
                             "output": t.format(v=v, a=a, b=b, c=c, bc=b * c)})
    return {"input.csv": pd.DataFrame(outs), "questions.csv": pd.DataFrame(qs)}


def data_bpe(rng):
    corpus = {"кот": 10, "коты": 6, "кота": 5, "крот": 4, "кроты": 3, "рот": 7, "роты": 2,
              "вот": 5, "вода": 4, "воды": 3}
    for w in corpus:
        assert all(a != b for a, b in zip(w, w[1:]))   # без повторов подряд — счёт пар однозначен
    return pd.DataFrame({"word": list(corpus), "count": list(corpus.values())})


def data_nmf(rng):
    words = ["матч", "гол", "команда", "тренер", "рецепт", "мука", "сахар", "пирог",
             "ракета", "орбита", "спутник", "запуск"]
    topic_word = np.zeros((3, 12))
    for k in range(3):
        topic_word[k, 4 * k:4 * k + 4] = rng.uniform(2.0, 5.0, 4)
    topic_word += 0.1
    mix = np.full((12, 3), 0.05)
    for doc in range(12):
        mix[doc, doc % 3] = 1.0
        mix[doc, (doc + 1) % 3] += rng.uniform(0, 0.3)
    X = rng.poisson(mix @ topic_word)
    df = pd.DataFrame(X, columns=words)
    df.insert(0, "doc_id", np.arange(1, 13))
    # Случайная инициализация может застрять в «смешанных» темах — перебираем (детерминированно),
    # пока 200 итераций не дадут три чистые темы: топ-4 слова каждой — из одного сюжета, сюжеты разные.
    for _ in range(100):
        W0, H0 = rng.uniform(0.1, 1.0, (12, 3)).round(3), rng.uniform(0.1, 1.0, (3, 12)).round(3)
        W, H = W0.copy(), H0.copy()
        for _ in range(200):
            W *= (X @ H.T) / (W @ H @ H.T)
            H *= (W.T @ X) / (W.T @ W @ H)
        blocks = [set(np.argsort(-H[k])[:4] // 4) for k in range(3)]
        if all(len(b) == 1 for b in blocks) and len(set.union(*blocks)) == 3:
            break
    else:
        raise RuntimeError("не нашлась инициализация с чистыми темами")
    init = {"W0": W0.tolist(), "H0": H0.tolist()}
    return {"input.csv": df, "init.json": to_json(init)}


def data_lightning(rng):
    val_mean = [0.92, 0.71, 0.60, 0.545, 0.512, 0.505, 0.47, 0.462, 0.468, 0.455, 0.466, 0.472, 0.49, 0.50, 0.51]
    acc_mean = [0.55, 0.64, 0.70, 0.73, 0.75, 0.755, 0.77, 0.775, 0.772, 0.78, 0.776, 0.771, 0.765, 0.76, 0.755]
    rows = [("sanity_check", 0, 0, 32, 1.102, 0.3125), ("sanity_check", 0, 1, 16, 1.087, 0.375)]
    for epoch in range(15):
        target = 1.0 * np.exp(-0.18 * epoch) + 0.25
        for b, size in enumerate([32, 32, 32, 8]):
            rows.append(("train", epoch, b, size, round(target + rng.normal(0, 0.04), 3), np.nan))
        d = round(rng.uniform(0.005, 0.03), 3)
        da = round(rng.uniform(0.005, 0.02), 4)
        # веса 32 и 16: (32·(m + d) + 16·(m − 2d)) / 48 = m
        rows.append(("val", epoch, 0, 32, round(val_mean[epoch] + d, 3), round(acc_mean[epoch] - da, 4)))
        rows.append(("val", epoch, 1, 16, round(val_mean[epoch] - 2 * d, 3), round(acc_mean[epoch] + 2 * da, 4)))
    return pd.DataFrame(rows, columns=["stage", "epoch", "batch_idx", "batch_size", "loss", "acc"])


def data_keras(rng):
    val = [0.900, 0.700, 0.610, 0.570, 0.568, 0.575, 0.540, 0.536, 0.541, 0.530, 0.532, 0.529, 0.531,
           0.533, 0.530, 0.534, 0.535]
    tail = 0.535 + np.cumsum(rng.uniform(-0.002, 0.006, 40 - len(val)))
    val_loss = np.round(np.concatenate([val, tail]), 3)
    loss = np.round(0.95 * np.exp(-0.09 * np.arange(40)) + 0.2 + rng.normal(0, 0.005, 40), 3)
    return pd.DataFrame({"epoch": np.arange(1, 41), "loss": loss, "val_loss": val_loss})


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "attention_mechanism",
        "title": "Аддитивное внимание Bahdanau",
        "data": data_bahdanau,
        "task": """
            Модель перевода «энкодер — декодер» переводит фразу «я люблю читать книги по вечерам».
            Энкодер уже выдал скрытые состояния слов — файл `input.csv` (`pos, word, h0, h1, h2`, размерность 3).
            Декодер на трёх шагах генерации имел состояния `decoder.csv` (`step, s0, s1`, размерность 2).
            Веса аддитивного внимания — `weights.json`: `W_q` (4 × 2), `W_k` (4 × 3), `v` (4).

            Для каждого шага декодера t посчитайте внимание Bahdanau к шести словам источника:

            1. Оценка соответствия: eⱼ = vᵀ · tanh(W_q · sₜ + W_k · hⱼ), j = 1…6
            2. Веса внимания: α = softmax(e) по шести словам
            3. Вектор контекста: cₜ = Σⱼ αⱼ · hⱼ (размерность 3)

            Заметьте: размерности запроса (2) и ключей (3) разные — скалярное произведение q · k здесь просто не
            определено, а аддитивное внимание работает, потому что обе стороны сначала проецируются в общее
            пространство размерности 4.

            ## Подсказки

            - Посчитайте W_k · hⱼ для всех слов один раз (матрица 6 × 4) — эта часть не зависит от шага декодера.
            - На каждом шаге прибавьте вектор W_q · sₜ (4) ко всем шести строкам (broadcasting), примените tanh и
              умножьте на v — получится 6 оценок. Softmax делайте устойчиво (вычитайте максимум).
            - `keys = H @ W_k.T`; `e = np.tanh(W_q @ s + keys) @ v`; `alpha = softmax(e)`; `c = alpha @ H`;
              `top_word` — слово с наибольшим αⱼ.

            ## Ожидаемый формат выходного файла

            CSV с колонками `step, a1, a2, a3, a4, a5, a6, c0, c1, c2, top_word` — 3 строки (шаги 1, 2, 3).
            `a1…a6` — веса внимания к словам в порядке `pos`, `c0…c2` — вектор контекста, `top_word` — слово
            с наибольшим весом. Числа округлить до 4 знаков.
        """,
        "solution": """
            import json


            def softmax(z):
                e = np.exp(z - z.max())
                return e / e.sum()


            def main():
                enc = pd.read_csv('input.csv')
                dec = pd.read_csv('decoder.csv')
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                H = enc[['h0', 'h1', 'h2']].to_numpy()             # 6 × 3 — «ключи» и «значения»

                keys = H @ w['W_k'].T                              # W_k·hⱼ для всех слов: 6 × 4, считаем один раз
                rows = []
                for step, s in zip(dec['step'], dec[['s0', 's1']].to_numpy()):
                    scores = np.tanh(w['W_q'] @ s + keys) @ w['v']  # eⱼ = vᵀ·tanh(W_q·s + W_k·hⱼ)
                    alpha = softmax(scores)                         # распределение по словам источника
                    context = alpha @ H                             # cₜ = Σ αⱼ·hⱼ
                    assert np.isclose(alpha.sum(), 1)
                    row = {'step': step}
                    row.update({f'a{j + 1}': alpha[j] for j in range(len(alpha))})
                    row.update({f'c{k}': context[k] for k in range(H.shape[1])})
                    row['top_word'] = enc['word'][int(alpha.argmax())]
                    rows.append(row)
                    print(f"шаг {step}: декодер смотрит в основном на «{row['top_word']}» (α = {alpha.max():.3f})")

                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Bahdanau (2014) дал декодеру возможность на каждом шаге заново «оглядываться» на все состояния энкодера.
            Оценку соответствия считает маленькая сеть: запрос и ключ проецируются в общее пространство, складываются,
            проходят через tanh и сворачиваются вектором v в одно число. Softmax превращает оценки в веса, а контекст —
            взвешенная сумма состояний энкодера.

            **Типичные ошибки:**
            - Softmax по шагам декодера, а не по словам источника — веса строки перестают суммироваться в 1.
            - Перепутали W_q и W_k (размерности 4 × 2 и 4 × 3 сразу подскажут ошибкой формы).
            - Контекст как сумма проекций W_k·hⱼ, а не исходных состояний hⱼ — получится вектор размерности 4.
            - tanh применили к каждому слагаемому отдельно: tanh(W_q·s) + tanh(W_k·h) — это уже другая функция.

            **Что вынести:** аддитивное внимание = score(q, k) = vᵀ·tanh(W₁q + W₂k); оно работает при разных размерностях
            q и k, но требует отдельной маленькой сети на каждую пару. Скалярное q·k/√d победило, потому что сводится к одному
            матричному умножению. → шпаргалка «Механизм внимания».
        """,
    },
    {
        "name": "bert_finetuning",
        "title": "Голова классификатора на CLS-векторах: warmup + linear decay",
        "data": data_cls_head,
        "task": """
            Энкодер BERT заморожен, и для 60 отзывов уже посчитаны векторы токена `[CLS]` (сокращены до 8 чисел):
            `input.csv` — `id, c0…c7, label` (0 — негатив, 1 — нейтрально, 2 — позитив).
            Обучите поверх них линейную голову классификатора — то, что `AutoModelForSequenceClassification`
            добавляет к BERT, — с расписанием learning rate «линейный разогрев + линейное затухание»
            (как `get_linear_schedule_with_warmup` в Hugging Face).

            - Модель: логиты z = X·W + b, W (8 × 3) и b (3) в начале **нулевые**; p = softmax(z).
            - Потери: средняя кросс-энтропия по всем 60 объектам, L = −mean(log p_y).
            - Обучение: полный батч, обычный градиентный спуск, всего T = 40 шагов, из них W = 4 шага разогрева,
              пиковый learning rate η = 0.5.
            - Расписание для шага t = 1…T:
              - lrₜ = η · t / W при t ≤ W (разогрев от нуля);
              - lrₜ = η · (T − t) / (T − W) при t > W (линейно до нуля к последнему шагу).
            - На шаге t: посчитать L при текущих весах (до обновления), затем градиент и обновление
              W ← W − lrₜ·∂L/∂W, b ← b − lrₜ·∂L/∂b.

            ## Подсказки

            - Сначала выпишите таблицу lr: на шаге 4 он достигает пика 0.5, на шаге 40 равен нулю.
            - При нулевых весах все вероятности равны 1/3, поэтому loss первого шага = ln 3 ≈ 1.0986 — проверьте себя.
            - Градиент softmax + кросс-энтропии: G = (P − Y_onehot) / n, ∂L/∂W = Xᵀ·G, ∂L/∂b = сумма G по строкам.

            ## Ожидаемый формат выходного файла

            CSV с колонками `step, lr, loss` — 40 строк (t = 1…40); `loss` — потери до обновления на этом шаге.
            `lr` округлить до 6 знаков, `loss` — до 4.
        """,
        "solution": """
            T, WARMUP, PEAK_LR = 40, 4, 0.5


            def lr_at(t):
                # get_linear_schedule_with_warmup: разогрев от 0 до пика, затем линейно к нулю
                if t <= WARMUP:
                    return PEAK_LR * t / WARMUP
                return PEAK_LR * (T - t) / (T - WARMUP)


            def softmax(z):
                e = np.exp(z - z.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)


            def main():
                df = pd.read_csv('input.csv')
                X = df[[f'c{j}' for j in range(8)]].to_numpy()
                y = df['label'].to_numpy()
                n, k = len(y), 3
                Y = np.eye(k)[y]                                   # one-hot метки

                W, b = np.zeros((X.shape[1], k)), np.zeros(k)       # голова классификатора
                rows = []
                for t in range(1, T + 1):
                    P = softmax(X @ W + b)
                    loss = -np.mean(np.log(P[np.arange(n), y]))
                    G = (P - Y) / n                                 # ∂L/∂z
                    lr = lr_at(t)
                    W -= lr * (X.T @ G)
                    b -= lr * G.sum(axis=0)
                    rows.append({'step': t, 'lr': lr, 'loss': loss})

                assert np.isclose(rows[0]['loss'], np.log(3))
                acc = (softmax(X @ W + b).argmax(axis=1) == y).mean()
                print(f"loss: {rows[0]['loss']:.4f} → {rows[-1]['loss']:.4f}, accuracy на обучении = {acc:.3f}")
                result = pd.DataFrame(rows)
                result['lr'] = result['lr'].round(6)
                result['loss'] = result['loss'].round(4)
                return result
        """,
        "review": """
            **Идея.** При дообучении BERT к энкодеру добавляют одну линейную голову над вектором `[CLS]`. Пока голова случайная,
            её градиенты «шумные», и большой шаг испортил бы предобученные веса энкодера — поэтому learning rate
            сначала разгоняют от нуля (warmup, ~6–10% шагов), а затем линейно гасят до нуля к концу обучения.

            ```python
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4, num_training_steps=40)
            ```

            **Типичные ошибки:**
            - Расписание «с нуля»: в Hugging Face lr считается от счётчика шагов, начинающегося с 0, поэтому самый первый
              шаг там делается с lr = 0. Здесь шаги нумеруются с 1 — сдвиг на шаг даст всю таблицу lr «не той».
            - Записывали loss после обновления, а не до — все значения сдвинуты на шаг.
            - Сумма вместо среднего в кросс-энтропии — градиент в 60 раз больше, обучение расходится.
            - Забыли смещение b или обновили его градиентом без деления на n.

            **Что вынести:** при дообучении маленький пиковый lr (для настоящего BERT 2e-5…5e-5 с AdamW), warmup и затухание;
            часто ещё и послойное затухание lr (layer-wise LR decay: нижним слоям — меньший шаг). Замороженный энкодер +
            обучаемая голова — самый дешёвый вариант, его полезно иметь как бейзлайн. → шпаргалка «Дообучение BERT».
        """,
    },
    {
        "name": "bert_language_models",
        "title": "BERT MLM: маскирование 80/10/10 и потери только на масках",
        "data": data_mlm,
        "task": """
            Подготовьте батч для предобучения BERT задачей Masked Language Modeling и посчитайте потери.
            В `input.csv` — три последовательности по токенам: `seq, pos, token`. Чтобы результат был воспроизводим,
            случайные числа уже «выброшены» и лежат рядом:

            - `u_select` — случайное число для выбора позиции в MLM;
            - `u_action` — случайное число для выбора действия;
            - `random_token` — токен, который подставляется, если выпало действие «случайный токен»;
            - `p_true` — вероятность, которую модель после прямого прохода дала **исходному** токену этой позиции.

            Правила (как в BERT):

            1. Служебные токены `[CLS]`, `[SEP]`, `[PAD]` **никогда** не выбираются, какие бы числа ни выпали.
            2. Обычный токен выбирается, если `u_select < 0.15` (≈ 15% токенов).
            3. Для выбранного токена: `u_action < 0.8` → заменить на `[MASK]` (действие `mask`);
               `0.8 ≤ u_action < 0.9` → заменить на `random_token` (`random`); иначе оставить как есть (`keep`).
            4. Потери MLM — средняя −ln(p_true) **только по выбранным позициям** (все три действия входят в потери,
               невыбранные позиции — нет).

            ## Подсказки

            - Сначала отфильтруйте служебные токены — среди них есть «ловушки» с маленьким `u_select`.
            - Действие определяется по `u_action` только для выбранных позиций; для `keep` вход модели совпадает
              с исходным токеном, но позиция всё равно участвует в потерях.
            - `mask = ~special & (u_select < 0.15)`; `np.select([u < 0.8, u < 0.9], ['mask', 'random'], 'keep')`;
              `nll = -np.log(p_true)`; итоговые потери — `nll[mask].mean()`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `seq, pos, token, input_token, action, nll`: по строке на каждую выбранную позицию
            в порядке `seq`, затем `pos`; `token` — исходный токен, `input_token` — что подаётся модели
            (`[MASK]`, случайный токен или исходный). Последней строкой — итог: `seq = 0, pos = 0`,
            `token = [ALL]`, `input_token = [ALL]`, `action = mean`, `nll` — средние потери MLM.
            `nll` округлить до 4 знаков.
        """,
        "solution": """
            SPECIAL = {'[CLS]', '[SEP]', '[PAD]'}


            def main():
                df = pd.read_csv('input.csv')
                special = df['token'].isin(SPECIAL)
                selected = ~special & (df['u_select'] < 0.15)      # служебные токены не маскируем никогда

                sel = df[selected].copy()
                u = sel['u_action']
                sel['action'] = np.select([u < 0.8, u < 0.9], ['mask', 'random'], 'keep')   # 80 / 10 / 10
                sel['input_token'] = np.select([sel['action'] == 'mask', sel['action'] == 'random'],
                                               ['[MASK]', sel['random_token']], sel['token'])
                sel['nll'] = -np.log(sel['p_true'])                # −log p(исходный токен)

                loss = sel['nll'].mean()
                wrong = -np.log(df.loc[~special, 'p_true']).mean()
                print(f"выбрано {len(sel)} из {(~special).sum()} обычных токенов; действия: {sel['action'].value_counts().to_dict()}")
                print(f"MLM loss = {loss:.4f}  (если ошибочно усреднить по всем токенам: {wrong:.4f})")

                result = sel[['seq', 'pos', 'token', 'input_token', 'action', 'nll']].sort_values(['seq', 'pos'])
                total = pd.DataFrame([{'seq': 0, 'pos': 0, 'token': '[ALL]', 'input_token': '[ALL]',
                                       'action': 'mean', 'nll': loss}])
                return pd.concat([result, total], ignore_index=True).round(4)
        """,
        "review": """
            **Идея.** MLM учит BERT восстанавливать скрытые токены по контексту с обеих сторон. Потери считаются только
            там, где модель «должна угадывать», — на выбранных 15% позиций. Трюк 80/10/10 нужен потому, что при дообучении
            `[MASK]` в текстах не встречается: модель обязана строить хорошие представления любого токена, а не только маски.

            ```python
            # В Hugging Face это делает DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15);
            # у невыбранных позиций метка -100 — кросс-энтропия PyTorch их игнорирует.
            ```

            **Типичные ошибки:**
            - Маскировали `[CLS]`/`[SEP]`/`[PAD]` — в данных есть такие позиции с маленьким `u_select`.
            - Не включили в потери позиции `random` и `keep` — модель «должна угадать» и их.
            - Усреднили −log p по всем токенам — потери получаются намного меньше (модель легко «угадывает» видимые токены).
            - Границы 0.8 и 0.9 сравнили с `u_action` невыбранных позиций — лишние строки в ответе.

            **Что вынести:** метки MLM есть только у выбранных позиций (у остальных −100), 80% `[MASK]`, 10% случайный токен,
            10% без изменений. Поэтому BERT — энкодер для понимания текста, а не генератор. → шпаргалка «BERT».
        """,
    },
    {
        "name": "gpt_architectures",
        "title": "Генерация GPT: greedy, temperature, top-k и top-p",
        "data": data_gpt_sampling,
        "task": """
            Крошечная «языковая модель» задана таблицей логитов следующего токена: `input.csv` — `id, token, l0…l7`.
            Строка — предыдущий токен, `lⱼ` — логит того, что следующим будет токен с `id = j` (словарь из 8 токенов,
            контекст — только последний токен, как у биграммной модели). Генерация начинается с `<s>`; ровно
            6 шагов, даже если встретилась точка. Случайные числа для сэмплирования — `uniforms.csv` (`step, u`);
            на шаге t все стратегии используют одно и то же uₜ.

            Стратегии (на каждом шаге — из строки логитов последнего сгенерированного токена):

            1. `greedy` — токен с максимальным логитом; prob — его вероятность в softmax(логиты).
            2. `temperature` — p = softmax(логиты / T), T = 0.7, сэмплирование из всех 8 токенов.
            3. `top_k` — k = 3 токена с наибольшими логитами, остальным −∞, p = softmax (T = 1).
            4. `top_p` — p₀ = softmax(логиты); оставить **минимальный** набор самых вероятных токенов с суммарной
               вероятностью ≥ 0.8, обнулить остальные и перенормировать.

            Сэмплирование по u (метод обратной функции распределения): перебирайте токены **в порядке id** и
            накапливайте вероятности; выберите первый id, у которого накопленная сумма **больше** u.

            ## Подсказки

            - Каждая стратегия — своя последовательность: после первого же расхождения они «смотрят» в разные строки таблицы.
            - top-k и top-p меняют только набор допустимых токенов и нормировку; сэмплирование для всех трёх
              одинаковое: `np.argmax(np.cumsum(p) > u)`.
            - top-p: `order = np.argsort(-p0)`, `cum = np.cumsum(p0[order])`, оставить `order[:np.argmax(cum >= 0.8) + 1]`;
              prob в ответе — вероятность выбранного токена **после** перенормировки.

            ## Ожидаемый формат выходного файла

            CSV с колонками `strategy, step, token, prob` — 24 строки: стратегии в порядке `greedy, temperature, top_k,
            top_p`, внутри — шаги 1…6. `token` — текст токена, `prob` — вероятность выбранного токена в распределении,
            из которого его выбирали; округлить до 4 знаков.
        """,
        "solution": """
            TEMPERATURE, TOP_K, TOP_P = 0.7, 3, 0.8


            def softmax(z):
                e = np.exp(z - z.max())
                return e / e.sum()


            def distribution(logits, strategy):
                if strategy in ('greedy', 'top_p'):
                    p = softmax(logits)
                    if strategy == 'top_p':
                        order = np.argsort(-p)
                        cum = np.cumsum(p[order])
                        keep = order[:np.argmax(cum >= TOP_P) + 1]     # минимальное «ядро» с массой ≥ p
                        q = np.zeros_like(p)
                        q[keep] = p[keep]
                        p = q / q.sum()
                    return p
                if strategy == 'temperature':
                    return softmax(logits / TEMPERATURE)
                masked = np.full_like(logits, -np.inf)                  # top-k
                top = np.argsort(-logits)[:TOP_K]
                masked[top] = logits[top]
                return softmax(masked)


            def main():
                table = pd.read_csv('input.csv')
                logits = table[[f'l{j}' for j in range(8)]].to_numpy()
                vocab = table['token'].tolist()
                us = pd.read_csv('uniforms.csv')['u'].to_numpy()

                rows = []
                for strategy in ['greedy', 'temperature', 'top_k', 'top_p']:
                    current = vocab.index('<s>')
                    for step, u in enumerate(us, start=1):
                        p = distribution(logits[current], strategy)
                        if strategy == 'greedy':
                            nxt = int(np.argmax(p))
                        else:
                            nxt = int(np.argmax(np.cumsum(p) > u))       # обратная функция распределения
                        rows.append({'strategy': strategy, 'step': step, 'token': vocab[nxt], 'prob': p[nxt]})
                        current = nxt
                    print(f"{strategy:>11}:", ' '.join(r['token'] for r in rows[-6:]))

                # Для справки: параметры GPT-2 small ≈ 12 слоёв × 12·d² + эмбеддинги V·d
                d, layers, V = 768, 12, 50257
                print(f"GPT-2 small ≈ {(12 * layers * d * d + V * d) / 1e6:.0f} млн параметров")
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Модель выдаёт только логиты; «характер» текста задаёт стратегия выбора. Greedy детерминирован и склонен
            к повторам; температура меняет остроту распределения, не меняя порядок токенов; top-k отсекает фиксированное
            число кандидатов; top-p — столько, сколько нужно, чтобы набрать заданную массу вероятности.

            **Типичные ошибки:**
            - Делили на температуру вероятности, а не логиты — получается совсем другое распределение.
            - Не перенормировали после top-k/top-p — накопленная сумма не доходит до 1, и при большом u выбор «проваливается».
            - top-p с условием «> 0.8» вместо «≥ 0.8» или без самого токена, на котором сумма пересекла порог.
            - Сэмплировали в порядке убывания вероятности, а не по id — другой токен при том же u.
            - Генерировали все стратегии по траектории greedy, а не по собственной.

            **Что вынести:** `P = softmax(logits / T)`; top-k и top-p — фильтры перед сэмплированием; генерация
            авторегрессионна: выбранный токен становится контекстом следующего шага (в настоящем GPT — через каузальную
            маску и KV-кэш). → шпаргалка «Архитектуры GPT».
        """,
    },
    {
        "name": "multi_head_attention",
        "title": "Multi-head attention: разрезание на головы, склейка и W_O",
        "data": data_mha,
        "task": """
            В `input.csv` — эмбеддинги 5 токенов (`token, x0…x3`, d_model = 4). В `weights.json` — матрицы
            `W_Q, W_K, W_V, W_O` размера 4 × 4 в раскладке «вход × выход»: Q = X·W_Q (как в лекции по трансформерам).

            Вычислите multi-head self-attention с h = 2 головами (d_k = d_model / h = 2):

            1. Q = X·W_Q, K = X·W_K, V = X·W_V (5 × 4).
            2. Разрежьте каждую на головы по столбцам: голова 1 — столбцы 0–1, голова 2 — столбцы 2–3
               (форма (n, d_model) → (h, n, d_k)).
            3. В каждой голове: Aᵢ = softmax(Qᵢ·Kᵢᵀ / √d_k) по строкам, headᵢ = Aᵢ·Vᵢ.
            4. Склейте головы обратно: Concat(head₁, head₂) — 5 × 4 (сначала столбцы головы 1, потом головы 2).
            5. Выход: O = Concat·W_O.

            ## Подсказки

            - Делите на √d_k = √2, а не на √d_model: каждая голова работает в своём подпространстве размерности 2.
            - Векторизованно: `M.reshape(n, h, d_k).transpose(1, 0, 2)` даёт (h, n, d_k); обратная склейка —
              `heads.transpose(1, 0, 2).reshape(n, d_model)`. Проверьте себя циклом по головам со срезами столбцов.
            - `top_head1` — токен, на который сильнее всего смотрит данный токен в голове 1: `vocab[A[0, i].argmax()]`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `token, top_head1, top_head2, o0, o1, o2, o3` — 5 строк в порядке файла.
            `top_head1/2` — токен с наибольшим весом внимания в строке этого токена в голове 1/2,
            `o0…o3` — выход multi-head attention; числа округлить до 4 знаков.
        """,
        "solution": """
            import json

            H_HEADS = 2


            def softmax(z):
                e = np.exp(z - z.max(axis=-1, keepdims=True))
                return e / e.sum(axis=-1, keepdims=True)


            def split_heads(M, h):
                n, d = M.shape
                return M.reshape(n, h, d // h).transpose(1, 0, 2)      # (n, d_model) → (h, n, d_k)


            def main():
                df = pd.read_csv('input.csv')
                X = df[[f'x{j}' for j in range(4)]].to_numpy()
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                n, d_model = X.shape
                d_k = d_model // H_HEADS

                Q, K, V = (split_heads(X @ w[name], H_HEADS) for name in ['W_Q', 'W_K', 'W_V'])
                A = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))   # (h, n, n) — все головы одним умножением
                heads = A @ V                                          # (h, n, d_k)
                concat = heads.transpose(1, 0, 2).reshape(n, d_model)  # склейка голов
                O = concat @ w['W_O']

                # Проверка циклом по головам со срезами столбцов
                for i in range(H_HEADS):
                    cols = slice(i * d_k, (i + 1) * d_k)
                    q, k, v = (X @ w[name][:, cols] for name in ['W_Q', 'W_K', 'W_V'])
                    assert np.allclose(softmax(q @ k.T / np.sqrt(d_k)) @ v, concat[:, cols])

                print(f"параметров (без смещений): 4·d² = {4 * d_model ** 2} — не зависит от числа голов")
                vocab = df['token'].tolist()
                result = pd.DataFrame({'token': vocab,
                                       'top_head1': [vocab[j] for j in A[0].argmax(axis=1)],
                                       'top_head2': [vocab[j] for j in A[1].argmax(axis=1)]})
                for j in range(d_model):
                    result[f'o{j}'] = O[:, j]
                return result.round(4)
        """,
        "review": """
            **Идея.** Multi-head attention не создаёт новых весов — оно делит те же d_model измерений на h независимых
            подпространств. Каждая голова делает свой «поиск по словарю» (у голов здесь разные любимые токены), результаты
            склеиваются и смешиваются матрицей W_O.

            **Типичные ошибки:**
            - Масштаб 1/√d_model вместо 1/√d_k — веса внимания «тупее», выход другой.
            - Неверная склейка: `heads.reshape(n, d_model)` без transpose перемешивает токены и головы.
            - Забыли W_O — в ответе склейка голов вместо выхода слоя.
            - Резали на головы по строкам (по токенам), а не по столбцам признаков.

            **Что вынести:** `MultiHead = Concat(head₁…head_h)·W_O`, d_k = d_model/h, параметров 4·d_model² при любом h;
            реализация — один большой matmul, reshape в (h, n, d_k) и батчевое внимание. → шпаргалка «Multi-head attention».
        """,
    },
    {
        "name": "nlp_tasks_ner_classification_summarization",
        "title": "NER: от BIO-тегов к сущностям и F1 по сущностям",
        "data": data_ner,
        "task": """
            В `input.csv` — 7 предложений, разбитых на токены (`sent_id, token_id, token`), с эталонной разметкой `gold`
            и предсказанием модели `pred` в схеме BIO (типы `PER`, `LOC`, `ORG`).

            1. Превратите теги в сущности-спаны (sent_id, начало, конец, тип) — по правилам conlleval / seqeval:
               - сущность начинается на `B-X`, а также на `I-X`, если предыдущий тег не `B-X` и не `I-X` (то есть «висячий»
                 `I-X` после `O` или после сущности другого типа тоже начинает сущность);
               - сущность продолжается, пока идут теги `I-X` того же типа; `B-X` сразу после `B-X` — уже новая сущность.
            2. Сравните множества спанов: предсказанная сущность верна (TP), только если **точно** совпали границы и тип.
               FP — предсказанные спаны без пары, FN — эталонные спаны без пары.
            3. Для каждого типа и для всех вместе (micro) посчитайте precision = TP/(TP+FP), recall = TP/(TP+FN),
               F1 = 2PR/(P+R); при нулевом знаменателе — 0.

            ## Подсказки

            - Разберите первое предложение вручную: «Анна Петрова» в эталоне — одна сущность, а модель нашла только «Анна» —
              это одновременно FP и FN.
            - Извлечение спанов — один проход по тегам с запоминанием начала и типа текущей сущности; сущность закрывается
              при `O`, при `B-…` и при `I-Y` другого типа.
            - Храните спаны как кортежи `(sent_id, start, end, type)` в множествах: `tp = len(gold & pred)` по типу.
              micro — суммы TP, FP, FN по всем типам.

            ## Ожидаемый формат выходного файла

            CSV с колонками `entity, tp, fp, fn, precision, recall, f1` — 4 строки: `LOC`, `ORG`, `PER` (по алфавиту),
            затем `micro`. precision, recall, f1 округлить до 4 знаков.
        """,
        "solution": """
            def extract_spans(tags):
                # BIO → [(start, end, type)] по правилам conlleval: «висячий» I-X тоже начинает сущность
                spans, start, kind = [], None, None
                for i, tag in enumerate(list(tags) + ['O']):           # 'O' в конце закрывает последнюю сущность
                    prefix, _, t = tag.partition('-')
                    continues = prefix == 'I' and kind == t
                    if start is not None and not continues:
                        spans.append((start, i - 1, kind))
                        start, kind = None, None
                    if prefix == 'B' or (prefix == 'I' and not continues):
                        start, kind = i, t
                return spans


            def prf(tp, fp, fn):
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                f = 2 * p * r / (p + r) if p + r else 0.0
                return p, r, f


            def main():
                df = pd.read_csv('input.csv')
                gold, pred = set(), set()
                for sid, g in df.groupby('sent_id', sort=True):
                    gold |= {(sid, *s) for s in extract_spans(g['gold'])}
                    pred |= {(sid, *s) for s in extract_spans(g['pred'])}

                rows, totals = [], np.zeros(3, dtype=int)
                for kind in ['LOC', 'ORG', 'PER']:
                    g = {s for s in gold if s[3] == kind}
                    p = {s for s in pred if s[3] == kind}
                    counts = np.array([len(g & p), len(p - g), len(g - p)])
                    totals += counts
                    rows.append({'entity': kind, 'tp': counts[0], 'fp': counts[1], 'fn': counts[2],
                                 **dict(zip(['precision', 'recall', 'f1'], prf(*counts)))})
                rows.append({'entity': 'micro', 'tp': totals[0], 'fp': totals[1], 'fn': totals[2],
                             **dict(zip(['precision', 'recall', 'f1'], prf(*totals)))})

                token_acc = (df['gold'] == df['pred']).mean()
                print(f"точность по токенам = {token_acc:.3f} — выглядит хорошо, но F1 по сущностям = {rows[-1]['f1']:.3f}")
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Пользователю нужны сущности, а не теги, поэтому NER оценивают по спанам: «Анна» вместо «Анна Петрова» —
            это и ложная находка, и пропуск. Большинство токенов имеют тег `O`, и точность по токенам почти всегда
            завышена — здесь она заметно выше F1 по сущностям.

            ```python
            from seqeval.metrics import classification_report   # считает ровно то же по спискам тегов
            ```

            **Типичные ошибки:**
            - Считали метрики по токенам — получаются красивые, но бесполезные числа.
            - Игнорировали «висячий» `I-PER` («Иван Иванов» в предложении 3) — по правилам conlleval это верная сущность.
            - Склеили `B-PER B-PER` в одну сущность — на самом деле это два отдельных спана (и оба ошибочные).
            - Засчитали спан с верными границами, но неверным типом (SpaceX как LOC) — это FP для LOC и FN для ORG.

            **Что вынести:** сущность верна только при точном совпадении границ и типа; precision/recall/F1 по типам и micro
            по суммам TP/FP/FN. Для классификации текстов аналогично смотрят F1 по классам и macro-F1, для суммаризации —
            ROUGE. → шпаргалка «Задачи NLP: NER, классификация, суммаризация».
        """,
    },
    {
        "name": "prompt_engineering",
        "title": "Оценка вариантов промпта: exact match и self-consistency",
        "data": data_prompts,
        "task": """
            Три варианта промпта (`zero_shot`, `few_shot`, `cot` — с рассуждением «шаг за шагом») прогнали на 20 текстовых
            задачах, по 5 ответов модели на задачу с temperature = 0.7. Ответы — `input.csv`
            (`question_id, variant, sample, output`), эталоны — `questions.csv` (`question_id, question, answer`).
            Модель вызывать не нужно — только честно посчитать метрики.

            **Нормализация ответа** (применяйте ко всем ответам модели и к эталону):
            1. если в тексте есть `Ответ:`, взять часть после **последнего** вхождения `Ответ:`;
            2. схлопнуть пробелы (`' '.join(text.split())`), привести к нижнему регистру, убрать точки в конце,
               обрезать пробелы по краям.

            **Метрики для каждого варианта:**
            - `em_first` — доля задач, где нормализованный ответ из `sample = 1` в точности равен эталону;
            - `em_majority` — то же для ответа большинства (self-consistency) среди 5 нормализованных ответов;
              при равенстве голосов побеждает тот из лидеров, что встретился раньше (меньший `sample`);
            - `agreement` — средняя по задачам доля ответов, совпавших с ответом большинства (5/5 = 1.0, 2/5 = 0.4);
            - `format_ok` — доля всех 100 ответов варианта, которые после нормализации являются целым числом
              (регулярное выражение `-?\\d+` на всю строку).

            ## Подсказки

            - «Думаю, получится 23» после нормализации остаётся текстом и не равен «23» — это и есть ошибка формата,
              которую few-shot и явная инструкция обычно лечат.
            - Голосование: `Counter(answers)`, максимум голосов, затем первый по порядку ответ с этим числом голосов.
            - Читайте файлы с `dtype=str, keep_default_na=False`, чтобы `" 23 "` и числа не превратились в другие типы;
              эталон сравнивайте как строку.

            ## Ожидаемый формат выходного файла

            CSV с колонками `variant, em_first, em_majority, agreement, format_ok` — 3 строки в порядке
            `zero_shot, few_shot, cot`; значения округлить до 4 знаков.
        """,
        "solution": """
            import re
            from collections import Counter


            def normalize(text):
                text = str(text)
                if 'Ответ:' in text:
                    text = text.rsplit('Ответ:', 1)[1]                 # часть после последнего «Ответ:»
                return ' '.join(text.split()).lower().rstrip('.').strip()


            def majority(answers):
                counts = Counter(answers)
                best = max(counts.values())
                return next(a for a in answers if counts[a] == best)  # тай-брейк: кто встретился раньше


            def main():
                outs = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
                gold = pd.read_csv('questions.csv', dtype=str, keep_default_na=False)
                gold = dict(zip(gold['question_id'], gold['answer'].map(normalize)))
                outs['norm'] = outs['output'].map(normalize)
                outs['sample'] = outs['sample'].astype(int)

                rows = []
                for variant in ['zero_shot', 'few_shot', 'cot']:
                    part = outs[outs['variant'] == variant]
                    first, vote, agree = [], [], []
                    for qid, g in part.groupby('question_id'):
                        answers = g.sort_values('sample')['norm'].tolist()
                        maj = majority(answers)
                        first.append(answers[0] == gold[qid])
                        vote.append(maj == gold[qid])
                        agree.append(answers.count(maj) / len(answers))
                    fmt = part['norm'].map(lambda s: re.fullmatch(r'-?\\d+', s) is not None).mean()
                    rows.append({'variant': variant, 'em_first': np.mean(first), 'em_majority': np.mean(vote),
                                 'agreement': np.mean(agree), 'format_ok': fmt})

                result = pd.DataFrame(rows).round(4)
                print(result.to_string(index=False))
                return result
        """,
        "review": """
            **Идея.** Prompt engineering — это инженерия: варианты промпта сравнивают на фиксированном наборе задач по метрике,
            а не по паре удачных ответов «на глаз». Главная часть такой оценки — честный разбор ответа (извлечь финальный
            ответ, нормализовать) и понятная метрика. Self-consistency (голосование по нескольким сэмплам при temperature > 0)
            сглаживает случайные ошибки рассуждения ценой нескольких запросов.

            **Типичные ошибки:**
            - Сравнивали сырые строки: `" 23 "`, `"23."` и `"Ответ: 23"` считались неверными — метрика занижена.
            - Брали часть после **первого** «Ответ:» или искали первое число в тексте рассуждения — в CoT там промежуточные числа.
            - Тай-брейк голосования «как получится» (`Counter.most_common` при равенстве зависит от порядка вставки —
              здесь это порядок сэмплов, но лучше задавать правило явно).
            - Читали эталон как число и сравнивали `23 == "23"`.

            **Что вынести:** тестовый набор + нормализация + метрика (exact match, доля корректного формата) — минимальная
            оценка промптов; голосование почти всегда поднимает точность над первым сэмплом, а agreement показывает,
            насколько модель уверена. → шпаргалка «Prompt engineering».
        """,
    },
    {
        "name": "tokenization",
        "title": "BPE вручную: список слияний на маленьком корпусе",
        "data": data_bpe,
        "task": """
            Обучите токенизатор BPE на корпусе `input.csv` (`word, count` — слово и сколько раз оно встретилось).

            1. Разбейте каждое слово на символы и добавьте маркер конца слова `</w>` как отдельный символ:
               `кот → к о т </w>`.
            2. Посчитайте частоты всех пар **соседних** символов с учётом `count` слова.
            3. Выберите самую частую пару; при равенстве частот — пару (левый, правый), наименьшую при обычном сравнении
               строк Python (сначала по левому символу, затем по правому).
            4. Слейте эту пару во всех словах в новый символ (просмотр слева направо), запишите слияние.
            5. Повторите шаги 2–4 — всего **10 слияний**.

            Размер словаря: в начале — число различных символов корпуса (включая `</w>`); после каждого слияния —
            число различных токенов среди начальных символов и всех созданных к этому моменту.

            ## Подсказки

            - Первое слияние почти угадывается глазами: «о т» встречается в большинстве слов корпуса.
            - Слово удобно хранить как кортеж символов: `{('к', 'о', 'т', '</w>'): 10, …}`; пары — `Counter` по `zip(s, s[1:])`.
            - Выбор пары вместе с тай-брейком — одна строка: `min(counts, key=lambda p: (-counts[p], p))`.
              Новый токен — просто склейка строк, например `т</w>`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `step, left, right, new_token, freq, vocab_size` — 10 строк (слияния 1…10 по порядку):
            склеиваемая пара, новый токен, частота пары на этом шаге и размер словаря после слияния.
        """,
        "solution": """
            from collections import Counter

            N_MERGES = 10


            def pair_counts(vocab):
                counts = Counter()
                for symbols, freq in vocab.items():
                    for pair in zip(symbols, symbols[1:]):
                        counts[pair] += freq
                return counts


            def merge_word(symbols, pair):
                out, i = [], 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                        out.append(symbols[i] + symbols[i + 1])
                        i += 2
                    else:
                        out.append(symbols[i])
                        i += 1
                return tuple(out)


            def tokenize(word, merges):
                symbols = tuple(word) + ('</w>',)
                for pair in merges:                     # слияния применяются в порядке обучения
                    symbols = merge_word(symbols, pair)
                return symbols


            def main():
                df = pd.read_csv('input.csv')
                vocab = {tuple(w) + ('</w>',): int(c) for w, c in zip(df['word'], df['count'])}
                tokens = {s for symbols in vocab for s in symbols}

                rows, merges = [], []
                for step in range(1, N_MERGES + 1):
                    counts = pair_counts(vocab)
                    best = min(counts, key=lambda p: (-counts[p], p))   # самая частая; тай-брейк по строкам
                    vocab = {merge_word(s, best): f for s, f in vocab.items()}
                    merges.append(best)
                    tokens.add(best[0] + best[1])
                    rows.append({'step': step, 'left': best[0], 'right': best[1], 'new_token': best[0] + best[1],
                                 'freq': counts[best], 'vocab_size': len(tokens)})

                for w in ['кроты', 'ворот', 'котлета']:
                    print(w, '→', ' '.join(tokenize(w, merges)))
                return pd.DataFrame(rows)
        """,
        "review": """
            **Идея.** BPE строит словарь снизу вверх: начинает с символов и жадно сливает самую частую соседнюю пару.
            Результат обучения — упорядоченный список слияний; новый текст токенизируют, применяя слияния в том же порядке.
            Частые куски («от</w>», «кот») становятся одним токеном, редкие слова раскладываются на знакомые части.

            **Типичные ошибки:**
            - Забыли маркер `</w>` — «т» в конце и в середине слова становятся одним символом, частоты пар другие.
            - Считали каждое слово один раз, без `count`.
            - Не пересчитывали частоты после слияния — после склейки «о т» пары «к о» больше нет, зато есть «к от».
            - Тай-брейк в другую сторону или по порядку появления — при равных частотах выбирается другая пара.

            **Что вынести:** BPE = частоты пар → слияние → повтор; словарь растёт ровно на один токен за слияние.
            Byte-level BPE (GPT-2) делает то же на байтах UTF-8 — поэтому русский текст выходит «дороже» английского
            по числу токенов. → шпаргалка «Токенизация».
        """,
    },
    {
        "name": "topic_modeling_lda_nmf",
        "title": "NMF для тем: мультипликативные обновления Ли — Сына",
        "data": data_nmf,
        "task": """
            В `input.csv` — матрица «документ × слово» X: 12 коротких текстов (`doc_id`) и счётчики 12 слов (колонки).
            Найдите 3 темы неотрицательным матричным разложением X ≈ W·H (W — 12 × 3, «документ × тема»; H — 3 × 12,
            «тема × слово»). Начальные матрицы — `init.json` (`W0`, `H0`).

            Сделайте **200 итераций** мультипликативных обновлений Ли — Сына для нормы Фробениуса ‖X − WH‖²;
            на каждой итерации сначала W, затем H (уже с новым W):

            - W ← W ⊙ (X·Hᵀ) / (W·H·Hᵀ)
            - H ← H ⊙ (Wᵀ·X) / (Wᵀ·W·H)

            (⊙ и / — поэлементно, без добавления ε: знаменатели положительны.) Затем для каждой темы k выпишите
            4 слова с наибольшими весами в строке H[k].

            ## Подсказки

            - Обновления не выводят из неотрицательной области: всё, что умножается и делится, неотрицательно.
              Ошибка ‖X − WH‖ должна монотонно убывать — выведите её каждые 50 итераций.
            - Порядок важен: H обновляется с уже новым W. Если поменять порядок — результат другой.
            - Проверка: `sklearn.decomposition.NMF(n_components=3, init='custom', solver='mu', beta_loss='frobenius',
              max_iter=200, tol=0)` и `fit_transform(X, W=W0, H=H0)` — `components_` совпадёт с вашим H.

            ## Ожидаемый формат выходного файла

            CSV с колонками `topic, rank, word, weight` — 12 строк: темы 0, 1, 2 (номер строки H), внутри — ранги 1…4
            по убыванию веса; `weight` — значение H[topic, слово], округлить до 4 знаков.
        """,
        "solution": """
            import json
            import warnings

            N_ITER = 200


            def main():
                df = pd.read_csv('input.csv')
                words = df.columns[1:].tolist()
                X = df[words].to_numpy(dtype=float)
                init = json.load(open('init.json', encoding='utf-8'))
                W0, H0 = np.array(init['W0']), np.array(init['H0'])

                W, H = W0.copy(), H0.copy()
                for it in range(1, N_ITER + 1):
                    W *= (X @ H.T) / (W @ H @ H.T)                  # сначала W
                    H *= (W.T @ X) / (W.T @ W @ H)                  # потом H — уже с новым W
                    if it % 50 == 0:
                        print(f"итерация {it}: ‖X − WH‖ = {np.linalg.norm(X - W @ H):.4f}")

                # Проверка: то же самое делает sklearn
                from sklearn.decomposition import NMF
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = NMF(n_components=3, init='custom', solver='mu', beta_loss='frobenius',
                                max_iter=N_ITER, tol=0)
                    model.fit_transform(X, W=W0.copy(), H=H0.copy())
                assert np.allclose(model.components_, H, atol=1e-6)

                rows = []
                for k in range(H.shape[0]):
                    top = np.argsort(-H[k])[:4]
                    rows += [{'topic': k, 'rank': r, 'word': words[j], 'weight': H[k, j]}
                             for r, j in enumerate(top, start=1)]
                    print(f"тема {k}:", ', '.join(words[j] for j in top))
                print("главная тема документов:", W.argmax(axis=1))
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** NMF ищет X ≈ WH с неотрицательными множителями: документ складывается из тем, тема — из слов, ничего
            не вычитается, поэтому темы интерпретируемы. Мультипликативные обновления — это градиентный спуск с особым
            поэлементным шагом, при котором вычитание превращается в умножение на отношение «положительная часть градиента /
            отрицательная часть»: матрицы остаются неотрицательными, а ошибка не растёт.

            **Типичные ошибки:**
            - Обновили H со старым W (обе формулы «одновременно») — другая траектория и другие веса.
            - Перепутали Hᵀ и H в числителе — ошибка размерностей или, хуже, бессмысленные темы.
            - Отсортировали слова по возрастанию веса (`argsort` без минуса) — в ответе «анти-темы».
            - Нормировали строки H перед выводом — в ответе ожидаются сырые веса.

            **Что вынести:** W — «документ × тема», H — «тема × слово», топ-слова темы = наибольшие элементы строки H;
            результат зависит от инициализации (`random_state`/`init`), а LDA делает похожее разложение вероятностно —
            на счётчиках, с распределениями Дирихле. → шпаргалка «Тематическое моделирование: LDA и NMF».
        """,
    },
    {
        "name": "transformers",
        "title": "Позиционное кодирование и блок трансформера: Post-LN и Pre-LN",
        "data": data_transformer_block,
        "task": """
            В `input.csv` — эмбеддинги 4 токенов (`token, e0…e5`, d_model = 6). В `weights.json` — веса подслоя FFN
            `W1` (6 × 12), `b1` (12), `W2` (12 × 6), `b2` (6) и параметры LayerNorm `gamma`, `beta` (по 6).

            1. **Позиционное кодирование** (позиции pos = 0, 1, 2, 3; i = 0, 1, 2):
               PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)), d = 6.
               Вход блока: x = E + PE.
            2. **FFN** (применяется к каждому токену отдельно): F(z) = ReLU(z·W1 + b1)·W2 + b2.
            3. **LayerNorm** по признакам каждого токена: LN(z) = γ ⊙ (z − mean(z)) / √(var(z) + 10⁻⁵) + β,
               дисперсия — смещённая (делим на d, как `np.var`).
            4. Два варианта остаточного блока с одним и тем же F и одной и той же LN:
               - **Post-LN** (исходная статья): post = LN(x + F(x));
               - **Pre-LN** (современные модели): pre = x + F(LN(x)).

            ## Подсказки

            - Проверьте PE: в позиции 0 чётные измерения равны 0, нечётные — 1; в столбцах 0 и 1 частота самая высокая.
            - Показатель степени 2i/d — по номеру **пары** измерений: для столбцов 0 и 1 он 0, для 2 и 3 — 2/6, для 4 и 5 — 4/6.
            - `pe[:, 0::2] = np.sin(pos / 10000 ** (np.arange(0, d, 2) / d))`, аналогично cos для `1::2`;
              LayerNorm — `mean`/`var` по `axis=-1, keepdims=True`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `token, pos, dim, pe, post, pre` — 24 строки (токены в порядке файла, внутри dim = 0…5):
            значение позиционного кодирования, выхода Post-LN и выхода Pre-LN блока. Числа округлить до 4 знаков.
        """,
        "solution": """
            import json

            EPS = 1e-5


            def positional_encoding(n, d):
                pos, i = np.arange(n)[:, None], np.arange(0, d, 2)[None, :]
                pe = np.zeros((n, d))
                pe[:, 0::2] = np.sin(pos / 10000 ** (i / d))
                pe[:, 1::2] = np.cos(pos / 10000 ** (i / d))
                return pe


            def layer_norm(z, gamma, beta):
                mu = z.mean(axis=-1, keepdims=True)
                var = z.var(axis=-1, keepdims=True)                   # смещённая дисперсия, как в nn.LayerNorm
                return gamma * (z - mu) / np.sqrt(var + EPS) + beta


            def main():
                df = pd.read_csv('input.csv')
                E = df[[f'e{j}' for j in range(6)]].to_numpy()
                p = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                n, d = E.shape

                ffn = lambda z: np.maximum(0, z @ p['W1'] + p['b1']) @ p['W2'] + p['b2']
                ln = lambda z: layer_norm(z, p['gamma'], p['beta'])

                pe = positional_encoding(n, d)
                x = E + pe
                post = ln(x + ffn(x))          # Post-LN: нормализация после сложения
                pre = x + ffn(ln(x))           # Pre-LN: «дорога» остаточной связи остаётся чистой

                # Post-LN нормирует выход: при γ = 1, β = 0 у каждой строки среднее 0 и дисперсия 1
                assert np.allclose(layer_norm(x, 1, 0).mean(axis=1), 0)
                print("норма строк post:", np.linalg.norm(post, axis=1).round(3), " pre:", np.linalg.norm(pre, axis=1).round(3))

                rows = [{'token': df['token'][t], 'pos': t, 'dim': j, 'pe': pe[t, j], 'post': post[t, j], 'pre': pre[t, j]}
                        for t in range(n) for j in range(d)]
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Внимание не знает порядка токенов, поэтому к эмбеддингам прибавляют позиционное кодирование: пары
            sin/cos с геометрически убывающими частотами. Каждый подслой блока обёрнут остаточной связью и LayerNorm;
            вопрос лишь в том, где стоит нормализация. Post-LN нормирует сумму (выход всегда «стандартизован»),
            Pre-LN — только вход подслоя, а основная дорога x + … остаётся без изменений: градиент течёт напрямую,
            глубокие модели обучаются стабильнее.

            **Типичные ошибки:**
            - Показатель 10000^(i/d) по номеру столбца вместо номера пары (2i/d) — частоты нечётных столбцов съезжают.
            - Позиции с 1, а не с 0 — весь PE сдвинут.
            - LayerNorm по токенам (axis=0), а не по признакам, или с несмещённой дисперсией (ddof = 1).
            - В Pre-LN к остатку прибавили LN(x) вместо x.

            **Что вынести:** x = E + PE; Post-LN: LN(x + F(x)), Pre-LN: x + F(LN(x)); форма (n, d_model) не меняется от
            блока к блоку, поэтому блоки ставят стопкой. → шпаргалка «Трансформеры».
        """,
    },
    {
        "name": "pytorch_lightning",
        "title": "Trainer Lightning на NumPy: логирование, EarlyStopping и ModelCheckpoint",
        "data": data_lightning,
        "task": """
            Повторите логику `lightning.Trainer` без PyTorch. В `input.csv` — значения, которые
            `training_step` и `validation_step` вернули бы на каждом батче, если бы обучение шло все 15 эпох:
            `stage` (`sanity_check` / `train` / `val`), `epoch` (с 0, как в Lightning), `batch_idx`, `batch_size`,
            `loss`, `acc` (только для валидации).

            ```python
            # в LightningModule: self.log('train_loss', loss, on_epoch=True, batch_size=len(x))
            #                    self.log('val_loss', loss, batch_size=len(x)); self.log('val_acc', acc, batch_size=len(x))
            trainer = L.Trainer(max_epochs=15, callbacks=[
                EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=0.01),
                ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=2)])
            ```

            Семантика, которую нужно воспроизвести:

            1. **Sanity check** — два батча валидации перед обучением; метрики и колбэки их **не видят**.
            2. **Метрика эпохи** = среднее по батчам, **взвешенное по `batch_size`** (так `self.log` агрегирует при
               `on_epoch=True`; последний батч обучения меньше остальных).
            3. **ModelCheckpoint** после валидации каждой эпохи: если сохранено меньше 2 чекпоинтов или val_loss строго
               меньше худшего из сохранённых — сохранить эпоху (худший при переполнении удаляется). Лучший чекпоинт — с
               минимальным val_loss.
            4. **EarlyStopping**: улучшение, если val_loss < best − min_delta; тогда best = val_loss и wait = 0, иначе
               wait += 1 (best при этом не меняется). Если wait ≥ patience — обучение останавливается после этой эпохи.

            ## Подсказки

            - Невзвешенное среднее двух батчей валидации отличается от взвешенного — проверьте на эпохе 0.
            - Улучшение меньше min_delta для EarlyStopping не считается, а для ModelCheckpoint — считается: «лучший
              чекпоинт» и «best» колбэка остановки могут разойтись.
            - `np.average(g['loss'], weights=g['batch_size'])` по группам `epoch`; колбэки удобно оформить классами
              с методом `on_validation_end(epoch, metrics)` — как в Lightning.

            ## Ожидаемый формат выходного файла

            CSV с колонками `epoch, train_loss, val_loss, val_acc, es_wait, in_top_k, is_best` — по строке на каждую
            **пройденную** эпоху (с 0 до эпохи остановки включительно). `es_wait` — счётчик EarlyStopping после этой эпохи;
            `in_top_k` = 1, если чекпоинт этой эпохи остался среди сохранённых в конце обучения, иначе 0; `is_best` = 1
            только у эпохи лучшего чекпоинта (`best_model_path`). Числа округлить до 4 знаков.
        """,
        "solution": """
            MAX_EPOCHS = 15


            class EarlyStopping:
                def __init__(self, patience, min_delta):
                    self.patience, self.min_delta = patience, min_delta
                    self.best, self.wait = np.inf, 0

                def on_validation_end(self, epoch, metrics):
                    current = metrics['val_loss']
                    if current < self.best - self.min_delta:            # улучшение больше min_delta
                        self.best, self.wait = current, 0
                    else:
                        self.wait += 1
                    return self.wait >= self.patience                   # trainer.should_stop


            class ModelCheckpoint:
                def __init__(self, save_top_k):
                    self.k, self.saved = save_top_k, {}                 # эпоха → val_loss

                def on_validation_end(self, epoch, metrics):
                    score = metrics['val_loss']
                    if len(self.saved) < self.k or score < max(self.saved.values()):
                        if len(self.saved) == self.k:
                            del self.saved[max(self.saved, key=self.saved.get)]   # удаляем худший
                        self.saved[epoch] = score
                    return False

                @property
                def best_epoch(self):
                    return min(self.saved, key=self.saved.get)


            def weighted(g, col):
                return np.average(g[col], weights=g['batch_size'])       # как self.log(on_epoch=True, batch_size=…)


            def main():
                logs = pd.read_csv('input.csv')
                logs = logs[logs['stage'] != 'sanity_check']            # sanity check не логируется
                ckpt = ModelCheckpoint(save_top_k=2)
                es = EarlyStopping(patience=3, min_delta=0.01)

                rows = []
                for epoch in range(MAX_EPOCHS):
                    train = logs[(logs['stage'] == 'train') & (logs['epoch'] == epoch)]
                    val = logs[(logs['stage'] == 'val') & (logs['epoch'] == epoch)]
                    metrics = {'train_loss': weighted(train, 'loss'), 'val_loss': weighted(val, 'loss'),
                               'val_acc': weighted(val, 'acc')}
                    ckpt.on_validation_end(epoch, metrics)
                    should_stop = es.on_validation_end(epoch, metrics)
                    rows.append({'epoch': epoch, **metrics, 'es_wait': es.wait})
                    if should_stop:
                        print(f"EarlyStopping: остановка после эпохи {epoch}")
                        break

                result = pd.DataFrame(rows)
                result['in_top_k'] = result['epoch'].isin(list(ckpt.saved)).astype(int)
                result['is_best'] = (result['epoch'] == ckpt.best_epoch).astype(int)
                print(f"сохранены эпохи {sorted(ckpt.saved)}, best_model_path → epoch={ckpt.best_epoch}")
                return result.round(4)
        """,
        "review": """
            **Идея.** Lightning убирает из вашего кода цикл обучения, но не отменяет его: Trainer сам вызывает
            `training_step`/`validation_step`, агрегирует `self.log` по эпохе (среднее, взвешенное по размеру батча),
            а колбэки после валидации решают, сохранять ли чекпоинт и не пора ли остановиться. Понимание этой семантики
            объясняет «странные» эффекты: почему лучший чекпоинт не совпадает с эпохой, где сработал EarlyStopping, и
            почему метрика эпохи не равна среднему по батчам.

            **Типичные ошибки:**
            - Учли sanity check как эпоху 0 — меняется val_loss эпохи 0.
            - Невзвешенное среднее по батчам — последний маленький батч весит столько же, сколько полный.
            - Обновляли best у EarlyStopping при улучшении меньше min_delta — wait сбрасывается, остановка уезжает.
            - Обработали все 15 эпох из файла, хотя обучение остановилось раньше.
            - Нумерация эпох с 1 — в Lightning (и в именах чекпоинтов `epoch=…`) счёт с 0.

            **Что вынести:** `EarlyStopping(patience, min_delta)` смотрит на «значимые» улучшения, `ModelCheckpoint(save_top_k)`
            — на любые; после `fit` используйте `trainer.test(ckpt_path='best')`, а не последние веса.
            → шпаргалка «PyTorch Lightning».
        """,
    },
    {
        "name": "tensorflow_keras_full_guide",
        "title": "Колбэки Keras: EarlyStopping и ReduceLROnPlateau по истории обучения",
        "data": data_keras,
        "task": """
            Модель Keras учится с колбэками

            ```python
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_delta=0.005,
                                                  min_lr=1e-4),
            ]
            model.fit(..., epochs=40, callbacks=callbacks)    # начальный learning rate 0.001
            ```

            В `input.csv` — значения `loss` и `val_loss`, которые получились бы на эпохах 1…40 (для упражнения считаем,
            что они не зависят от lr). Воспроизведите, что сделают колбэки. В конце каждой эпохи (сначала EarlyStopping,
            затем ReduceLROnPlateau):

            **EarlyStopping** (best = +∞, wait = 0 в начале): wait += 1; если val_loss < best − min_delta, то best = val_loss,
            best_epoch = эпоха, wait = 0; иначе, если wait ≥ patience, — остановить обучение после этой эпохи.

            **ReduceLROnPlateau** (свой best = +∞, свой wait = 0): если val_loss < best − min_delta, то best = val_loss,
            wait = 0; иначе wait += 1, и если wait ≥ patience: если текущий lr > min_lr, то
            lr = max(lr · factor, min_lr) и wait = 0 (если lr уже равен min_lr — ничего не происходит, wait не сбрасывается).
            Новый lr действует со следующей эпохи.

            ## Подсказки

            - У колбэков разные min_delta: падение val_loss на 0.002 — улучшение для EarlyStopping, но не для ReduceLROnPlateau.
            - Колонка `lr` — learning rate, с которым **шла** эпоха (именно его Keras пишет в `history`), уменьшение
              видно только в следующей строке.
            - Оформите колбэки классами с `on_epoch_end(epoch, val_loss)`; когда EarlyStopping вернёт «стоп», выйдите из цикла.
              `best_epoch` — эпоха, веса которой вернёт `restore_best_weights=True`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `epoch, val_loss, lr, es_wait, rlr_wait, best_epoch` — по строке на каждую пройденную эпоху
            (с 1 до эпохи остановки включительно); счётчики и `best_epoch` — после обработки эпохи. `lr` округлить
            до 6 знаков.
        """,
        "solution": """
            INITIAL_LR, MAX_EPOCHS = 1e-3, 40


            class EarlyStopping:
                def __init__(self, patience, min_delta=0.0):
                    self.patience, self.min_delta = patience, min_delta
                    self.best, self.wait, self.best_epoch = np.inf, 0, None

                def on_epoch_end(self, epoch, val_loss):
                    self.wait += 1
                    if val_loss < self.best - self.min_delta:
                        self.best, self.best_epoch, self.wait = val_loss, epoch, 0
                        return False
                    return self.wait >= self.patience                 # model.stop_training = True


            class ReduceLROnPlateau:
                def __init__(self, factor, patience, min_delta, min_lr):
                    self.factor, self.patience, self.min_delta, self.min_lr = factor, patience, min_delta, min_lr
                    self.best, self.wait = np.inf, 0

                def on_epoch_end(self, epoch, val_loss, lr):
                    if val_loss < self.best - self.min_delta:
                        self.best, self.wait = val_loss, 0
                    else:
                        self.wait += 1
                        if self.wait >= self.patience and lr > self.min_lr:
                            lr = max(lr * self.factor, self.min_lr)
                            self.wait = 0
                    return lr


            def main():
                history = pd.read_csv('input.csv')
                es = EarlyStopping(patience=5, min_delta=0.0)
                rlr = ReduceLROnPlateau(factor=0.5, patience=2, min_delta=0.005, min_lr=1e-4)

                lr, rows = INITIAL_LR, []
                for epoch, val_loss in zip(history['epoch'], history['val_loss']):
                    stop = es.on_epoch_end(epoch, val_loss)
                    new_lr = rlr.on_epoch_end(epoch, val_loss, lr)
                    rows.append({'epoch': epoch, 'val_loss': val_loss, 'lr': lr, 'es_wait': es.wait,
                                 'rlr_wait': rlr.wait, 'best_epoch': es.best_epoch})
                    if new_lr != lr:
                        print(f"эпоха {epoch}: ReduceLROnPlateau → lr = {new_lr:g}")
                    lr = new_lr
                    if stop:
                        print(f"эпоха {epoch}: EarlyStopping, восстановлены веса эпохи {es.best_epoch}")
                        break

                result = pd.DataFrame(rows)
                result['lr'] = result['lr'].round(6)
                return result
        """,
        "review": """
            **Идея.** Колбэки — это маленькие конечные автоматы, которые Keras вызывает в конце каждой эпохи. EarlyStopping
            считает эпохи без улучшения и останавливает обучение, `restore_best_weights=True` возвращает веса лучшей эпохи.
            ReduceLROnPlateau на плато уменьшает шаг: часто после этого val_loss снова начинает падать. Их счётчики
            независимы, а разные `min_delta` дают разные представления о том, что такое «улучшение».

            **Типичные ошибки:**
            - Записывали в `lr` уже уменьшенное значение — история сдвигается на эпоху.
            - Один общий счётчик wait для двух колбэков или общий best.
            - Продолжали уменьшать lr ниже `min_lr` или сбрасывали wait, когда lr уже на минимуме.
            - Не остановились на эпохе EarlyStopping и обработали все 40 строк.
            - Забыли `restore_best_weights=True` в реальном коде: модель после `fit` — с весами последней, а не лучшей эпохи.

            **Что вынести:** patience EarlyStopping должен быть больше patience ReduceLROnPlateau — иначе шаг не успеет
            уменьшиться до остановки. `history.history` хранит и `learning_rate` — по нему видно, когда сработал колбэк.
            → шпаргалка «TensorFlow / Keras: полное руководство».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям по NLP и фреймворкам:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=761):
        build(exercise, seed)


if __name__ == "__main__":
    main()
