#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор упражнений к лекциям по Deep Learning и Computer Vision (только NumPy — без PyTorch).

Путь 2, этап 3 (CNN): пулинг, архитектуры CNN (формы и параметры), ResNet-блок, transfer learning,
data augmentation, декодирование выхода YOLO.
Путь 2, этап 4 (RNN): RNN и BPTT, ячейка GRU.
Путь 2, этапы 5–6: ViT (патчи и токены), автоэнкодер для аномалий, диффузия (прямой процесс и шаг DDPM).
Путь 4 / CV: 3D CV (стерео и камера-обскура), оценка позы (OKS, PCK), neural style transfer (матрица Грама).

Структура упражнения та же, что в tools/generate_basic_exercises.py: input.csv (+ доп. файлы),
task.md (условие, подсказки, проверка, разбор), solution.py и answer.key.

Запуск из корня репозитория:
    python tools/generate_lecture_exercises_dl_cv.py
"""

import json

import numpy as np
import pandas as pd

from generate_basic_exercises import build


def dump(obj):
    return json.dumps(obj, indent=2) + "\n"


# ---------------------------------------------------------------- данные


def data_3d_cv(rng):
    cam = {"f": 700.0, "cx": 320.0, "cy": 240.0, "baseline_m": 0.12}
    n = 8
    Z = np.r_[rng.uniform(1.5, 12, n - 1), 38.0]                   # последняя точка — очень далеко
    X = rng.uniform(-0.35, 0.35, n) * Z
    Y = rng.uniform(-0.25, 0.25, n) * Z
    uL = cam["f"] * X / Z + cam["cx"]
    v = cam["f"] * Y / Z + cam["cy"]
    uR = uL - cam["f"] * cam["baseline_m"] / Z
    df = pd.DataFrame({"point_id": np.arange(1, n + 1), "u_left": uL.round(1),
                       "u_right": uR.round(1), "v": v.round(1)})
    return {"input.csv": df, "camera.json": dump(cam)}


def data_autoencoders(rng):
    A = np.array([[1.0, 0.6, -0.4, 0.8], [0.2, -0.9, 0.7, 0.5]])
    scale = np.array([12.0, 1.5, 40.0, 3.0])
    offset = np.array([50.0, 5.0, 200.0, 10.0])

    def normal(n):
        return rng.normal(0, 1, (n, 2)) @ A + rng.normal(0, 0.12, (n, 4))

    train = normal(80)
    test = normal(12)
    # аномалии: сильный сдвиг поперёк плоскости, где лежат нормальные данные
    test[3] += np.array([1.2, -0.6, -1.3, 0.4]) * 1.6
    test[7] += np.array([-0.9, 1.1, 0.8, -1.0]) * 1.4
    test[10] += np.array([0.5, 0.9, 1.4, -1.6]) * 1.0
    x = np.vstack([train, test]) * scale + offset
    df = pd.DataFrame(x.round(2), columns=["f1", "f2", "f3", "f4"])
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["split"] = ["train"] * 80 + ["test"] * 12
    return df


def data_cnn_architectures(rng):
    rows = [
        (1, "conv", 16, 3, 1, 1, 1), (2, "bn", None, None, None, None, None), (3, "relu", None, None, None, None, None),
        (4, "maxpool", None, 2, 2, 0, None),
        (5, "conv", 32, 3, 1, 1, 0), (6, "bn", None, None, None, None, None), (7, "relu", None, None, None, None, None),
        (8, "maxpool", None, 3, 2, 1, None),
        (9, "conv", 64, 3, 2, 0, 1), (10, "relu", None, None, None, None, None),
        (11, "conv", 32, 1, 1, 0, 1), (12, "flatten", None, None, None, None, None),
        (13, "dense", 10, None, None, None, 1),
    ]
    df = pd.DataFrame(rows, columns=["layer_id", "type", "out", "kernel", "stride", "padding", "bias"])
    return df.astype({c: "Int64" for c in ["out", "kernel", "stride", "padding", "bias"]})


def data_augmentation(rng):
    H, W = 6, 8
    img = rng.integers(0, 100, (H, W))
    pixels = pd.DataFrame([{"row": r, "col": c, "value": int(img[r, c])} for r in range(H) for c in range(W)])
    box = pd.DataFrame([{"x_min": 1, "y_min": 2, "x_max": 5, "y_max": 5}])
    ops = pd.DataFrame([
        {"step": 1, "op": "hflip", "top": None, "left": None, "height": None, "width": None},
        {"step": 2, "op": "rot90", "top": None, "left": None, "height": None, "width": None},
        {"step": 3, "op": "vflip", "top": None, "left": None, "height": None, "width": None},
        {"step": 4, "op": "crop", "top": 4, "left": 1, "height": 4, "width": 4},
    ]).astype({c: "Int64" for c in ["top", "left", "height", "width"]})
    return {"input.csv": pixels, "box.csv": box, "ops.csv": ops}


def data_diffusion(rng):
    steps = [1, 10, 100, 250, 500, 1000]
    x0 = rng.uniform(-1, 1, 4).round(2)
    eps = rng.normal(0, 1, 4).round(2)
    rows = []
    for t in steps:
        err = 0.03 if t < 100 else 0.08                             # «модель» ошибается сильнее на шумных шагах
        eps_pred = (eps + rng.normal(0, err, 4)).round(3)
        for k in range(4):
            rows.append({"t": t, "k": k + 1, "x0": x0[k], "eps": eps[k], "eps_pred": eps_pred[k]})
    return pd.DataFrame(rows)


def data_style_transfer(rng):
    C, H, W = 3, 4, 4
    content = rng.uniform(0, 3, (C, H, W)).round(2)
    base = rng.uniform(0, 3, (H, W))
    style = np.stack([base, 0.8 * base + rng.uniform(0, 0.6, (H, W)), 3 - base]).round(2)   # каналы скоррелированы
    perm = rng.permutation(H * W)
    maps = {
        "content": content,
        "style": style,
        "copy": content.copy(),
        "style_shuffled": style.reshape(C, -1)[:, perm].reshape(C, H, W),
        "blend": (0.7 * content + 0.3 * style).round(2),
        "noise": rng.uniform(0, 3, (C, H, W)).round(2),
    }
    rows = []
    for name, F in maps.items():
        for ch in range(C):
            for r in range(H):
                for c in range(W):
                    rows.append({"map": name, "channel": ch, "row": r, "col": c, "value": F[ch, r, c]})
    return pd.DataFrame(rows)


def data_pooling(rng):
    H, W = 6, 7
    x = rng.integers(-9, 10, (H, W))
    x[:2, :2] = -rng.integers(1, 10, (2, 2))                        # левый верхний угол — только отрицательные
    pixels = pd.DataFrame([{"row": r, "col": c, "value": int(x[r, c])} for r in range(H) for c in range(W)])
    configs = pd.DataFrame([
        {"config_id": 1, "mode": "max", "kernel": 2, "stride": 2, "padding": 0},
        {"config_id": 2, "mode": "max", "kernel": 3, "stride": 2, "padding": 1},
        {"config_id": 3, "mode": "avg", "kernel": 3, "stride": 2, "padding": 1},
        {"config_id": 4, "mode": "global_avg", "kernel": 0, "stride": 0, "padding": 0},
        {"config_id": 5, "mode": "global_max", "kernel": 0, "stride": 0, "padding": 0},
    ])
    return {"input.csv": pixels, "configs.csv": configs}


COCO_KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
           "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO_SIGMAS = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
               0.107, 0.107, 0.087, 0.087, 0.089, 0.089]


def data_pose(rng):
    template = np.array([(0.5, 0.1), (0.53, 0.08), (0.47, 0.08), (0.56, 0.09), (0.44, 0.09), (0.65, 0.25),
                         (0.35, 0.25), (0.72, 0.42), (0.28, 0.42), (0.75, 0.57), (0.25, 0.57), (0.6, 0.58),
                         (0.4, 0.58), (0.62, 0.78), (0.38, 0.78), (0.63, 0.97), (0.37, 0.97)])
    people = [(50, 40, 120, 300, 3.0), (300, 60, 80, 200, 6.0), (500, 100, 60, 150, 2.0), (150, 300, 40, 100, 5.0)]
    persons, kps = [], []
    for pid, (x0, y0, w, h, noise) in enumerate(people, start=1):
        persons.append({"person_id": pid, "bbox_w": w, "bbox_h": h, "area": round(0.55 * w * h, 1)})
        gt = template * [w, h] + [x0, y0]
        pred = gt + rng.normal(0, noise, gt.shape)
        v = np.full(17, 2)
        v[rng.choice(17, 2, replace=False)] = 1                     # размечены, но закрыты
        hidden = rng.choice(17, 2 + pid % 2, replace=False)
        v[hidden] = 0                                               # не размечены: в метриках не участвуют
        pred[hidden] += rng.uniform(40, 80, (len(hidden), 2))       # предсказание там «улетело» — не должно влиять
        pred[rng.integers(17)] += [25, -15]                         # один грубый промах
        for i in range(17):
            kps.append({"person_id": pid, "kp_id": i, "name": COCO_KP[i],
                        "x_gt": round(gt[i, 0], 1) if v[i] else 0.0, "y_gt": round(gt[i, 1], 1) if v[i] else 0.0,
                        "v": int(v[i]), "x_pred": round(pred[i, 0], 1), "y_pred": round(pred[i, 1], 1)})
    sig = pd.DataFrame({"kp_id": np.arange(17), "name": COCO_KP, "sigma": COCO_SIGMAS})
    return {"input.csv": pd.DataFrame(kps), "persons.csv": pd.DataFrame(persons), "sigmas.csv": sig}


def data_resnet(rng):
    x = rng.normal(0, 1, (2, 5, 5)).round(1)
    rows = [{"channel": c, "row": r, "col": k, "value": x[c, r, k]} for c in range(2) for r in range(5) for k in range(5)]

    def bn(n):
        return {"gamma": rng.uniform(0.6, 1.4, n).round(2).tolist(), "beta": rng.normal(0, 0.2, n).round(2).tolist(),
                "running_mean": rng.normal(0, 0.3, n).round(2).tolist(),
                "running_var": rng.uniform(0.5, 2.0, n).round(2).tolist()}

    weights = {
        "conv1": rng.normal(0, 0.3, (4, 2, 3, 3)).round(2).tolist(),    # (out, in, kH, kW), как nn.Conv2d.weight
        "bn1": bn(4),
        "conv2": rng.normal(0, 0.2, (4, 4, 3, 3)).round(2).tolist(),
        "bn2": bn(4),
        "proj": rng.normal(0, 0.5, (4, 2, 1, 1)).round(2).tolist(),
        "bn_proj": bn(4),
        "eps": 1e-5,
    }
    return {"input.csv": pd.DataFrame(rows), "weights.json": dump(weights)}


def data_transfer(rng):
    classes = ["cat", "dog", "fox"]
    means = np.array([[2.0, 0.3, 1.2, 0.1, 0.8, 0.5, 0.0, 1.0],
                      [0.4, 1.8, 1.0, 0.9, 0.2, 0.6, 0.3, 0.4],
                      [1.2, 0.6, 0.2, 1.6, 0.9, 0.1, 1.1, 0.7]])
    rows = []
    split = ["train"] * 45 + ["test"] * 15
    labels = np.r_[np.repeat([0, 1, 2], 15), np.repeat([0, 1, 2], 5)]
    order = np.r_[rng.permutation(45), 45 + rng.permutation(15)]
    for idx, j in enumerate(order):
        y = labels[j]
        emb = np.maximum(0, means[y] + rng.normal(0, 0.55, 8))     # выход GAP после ReLU — неотрицательный
        rows.append({"id": idx + 1, **{f"e{k + 1}": round(emb[k], 3) for k in range(8)},
                     "label": classes[y], "split": split[idx]})
    return pd.DataFrame(rows)


def data_vit(rng):
    H, W, C = 6, 6, 2
    img = rng.uniform(0, 1, (H, W, C)).round(1)
    rows = [{"row": r, "col": c, "channel": ch, "value": img[r, c, ch]}
            for r in range(H) for c in range(W) for ch in range(C)]
    weights = {
        "E": rng.normal(0, 0.3, (18, 4)).round(2).tolist(),        # патч (3·3·2 = 18) → эмбеддинг D = 4
        "b": rng.normal(0, 0.1, 4).round(2).tolist(),
        "cls": rng.normal(0, 0.5, 4).round(2).tolist(),
        "pos": rng.normal(0, 0.2, (5, 4)).round(2).tolist(),       # 1 CLS + 4 патча
    }
    return {"input.csv": pd.DataFrame(rows), "weights.json": dump(weights)}


def data_yolo(rng):
    S, stride = 4, 32
    anchors = [[30, 60], [70, 40]]
    logit = lambda p: float(np.log(p / (1 - p)))
    pred = {}
    for gy in range(S):
        for gx in range(S):
            for a in range(2):
                pred[(gy, gx, a)] = [rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 0.3), rng.normal(0, 0.3),
                                     rng.normal(-6, 1), rng.normal(0, 1), rng.normal(0, 1)]

    def put(gy, gx, a, cx, cy, w, h, score_obj, cls, cls_p):
        tx = logit(cx / stride - gx)
        ty = logit(cy / stride - gy)
        tw = float(np.log(w / anchors[a][0]))
        th = float(np.log(h / anchors[a][1]))
        lp, lc = (logit(cls_p), -2.0) if cls == 0 else (-2.0, logit(cls_p))
        pred[(gy, gx, a)] = [tx, ty, tw, th, logit(score_obj), lp, lc]

    put(2, 1, 0, 45, 70, 28, 62, 0.93, 0, 0.95)                     # человек
    put(2, 1, 1, 47, 72, 32, 58, 0.71, 0, 0.90)                     # дубль человека другим якорем
    put(1, 1, 0, 44, 63, 30, 60, 0.62, 0, 0.88)                     # дубль из соседней клетки
    put(0, 3, 1, 104, 28, 62, 38, 0.90, 1, 0.97)                    # машина у правого края — рамку обрежем
    put(0, 3, 0, 102, 30, 30, 50, 0.55, 0, 0.70)                    # «человек» на машине — другой класс, NMS не трогает
    put(1, 2, 1, 74, 40, 56, 30, 0.80, 1, 0.92)                     # вторая машина: умеренное перекрытие с первой
    put(3, 0, 0, 20, 110, 14, 30, 0.52, 0, 0.80)                    # маленький человек внизу
    put(3, 2, 1, 80, 112, 40, 20, 0.30, 1, 0.60)                    # слабая рамка — отсечётся порогом
    rows = [{"gy": gy, "gx": gx, "anchor": a, **dict(zip(["tx", "ty", "tw", "th", "obj", "cls_person", "cls_car"],
                                                         np.round(v, 3)))}
            for (gy, gx, a), v in sorted(pred.items())]
    cfg = {"image_size": 128, "grid": S, "stride": stride, "anchors": anchors, "classes": ["person", "car"],
           "score_threshold": 0.25, "iou_threshold": 0.5}
    return {"input.csv": pd.DataFrame(rows), "config.json": dump(cfg)}


def data_gru(rng):
    T, D, H = 5, 2, 3
    x = rng.uniform(-1, 1, (T, D)).round(2)
    inputs = pd.DataFrame({"t": np.arange(1, T + 1), "x1": x[:, 0], "x2": x[:, 1]})
    weights = {
        "W_ih": rng.normal(0, 0.6, (3 * H, D)).round(2).tolist(),   # как nn.GRU.weight_ih_l0: (3H × D), порядок r, z, n
        "W_hh": rng.normal(0, 0.6, (3 * H, H)).round(2).tolist(),   # weight_hh_l0: (3H × H)
        "b_ih": rng.normal(0, 0.3, 3 * H).round(2).tolist(),
        "b_hh": rng.normal(0, 0.3, 3 * H).round(2).tolist(),
        "h0": rng.uniform(-0.5, 0.5, H).round(2).tolist(),
    }
    return {"input.csv": inputs, "weights.json": dump(weights)}


def data_rnn(rng):
    T, D, H = 4, 2, 3
    x = rng.uniform(-1, 1, (T, D)).round(2)
    target = rng.uniform(-1, 1, T).round(2)
    inputs = pd.DataFrame({"t": np.arange(1, T + 1), "x1": x[:, 0], "x2": x[:, 1], "target": target})
    weights = {
        "W_xh": rng.normal(0, 0.6, (H, D)).round(2).tolist(),
        "W_hh": rng.normal(0, 0.5, (H, H)).round(2).tolist(),
        "b_h": rng.normal(0, 0.1, H).round(2).tolist(),
        "W_hy": rng.normal(0, 0.8, (1, H)).round(2).tolist(),
        "b_y": [0.1],
    }
    return {"input.csv": inputs, "weights.json": dump(weights)}


# ---------------------------------------------------------------- упражнения

EXERCISE_LIST = [
    {
        "name": "3d_computer_vision",
        "title": "Стереозрение: глубина по диспаритету и 3D-точки",
        "data": data_3d_cv,
        "task": """
            Две откалиброванные камеры смотрят параллельно (кадры ректифицированы: одна и та же точка лежит на одной
            строке `v` в обоих кадрах). Параметры в `camera.json`: фокусное расстояние `f` (в пикселях), главная точка
            `cx, cy` и база `baseline_m` — расстояние между камерами в метрах.

            Файл `input.csv` — 8 найденных соответствий: `point_id, u_left, u_right, v` (пиксели).

            Для каждой точки вычислите в системе координат **левой** камеры:

            1. Диспаритет d = u_left − u_right (пиксели).
            2. Глубину Z = f · B / d (метры).
            3. Координаты X, Y обращением модели камеры-обскуры u = f·X/Z + cx, v = f·Y/Z + cy:
               X = (u_left − cx) · Z / f, Y = (v − cy) · Z / f.
            4. Чувствительность глубины к ошибке сопоставления на 1 пиксель: dZ = Z² / (f · B)
               (это |∂Z/∂d| — насколько метров «уедет» глубина, если d ошибся на 1 пиксель).

            ## Подсказки

            - Большой диспаритет — объект близко; маленький — далеко. Последняя точка в файле — самая далёкая.
            - Проверьте себя обратной проекцией: f·X/Z + cx должно дать ровно u_left, а f·(X − B)/Z + cx — u_right.
            - Векторно: `d = uL - uR`, `Z = f * B / d`, `X = (uL - cx) * Z / f`, `Y = (v - cy) * Z / f`,
              `dZ = Z**2 / (f * B)`. Производная: Z = fB/d ⇒ |dZ/dd| = fB/d² = Z²/(fB).

            ## Ожидаемый формат выходного файла

            CSV с колонками `point_id, disparity, Z, X, Y, dZ_1px` — 8 строк в порядке `point_id`; значения округлить
            до 4 знаков.
        """,
        "solution": """
            import json


            def main():
                df = pd.read_csv('input.csv')
                cam = json.load(open('camera.json', encoding='utf-8'))
                f, cx, cy, B = cam['f'], cam['cx'], cam['cy'], cam['baseline_m']

                d = df['u_left'] - df['u_right']          # диспаритет: насколько точка «сдвинулась» между кадрами
                Z = f * B / d                              # глубина из подобия треугольников
                X = (df['u_left'] - cx) * Z / f            # обращение u = f·X/Z + cx
                Y = (df['v'] - cy) * Z / f
                dZ = Z ** 2 / (f * B)                      # ошибка глубины при ошибке d на 1 пиксель

                # Обратная проекция в обе камеры (правая сдвинута на B по оси X)
                assert np.allclose(f * X / Z + cx, df['u_left'])
                assert np.allclose(f * (X - B) / Z + cx, df['u_right'])

                result = pd.DataFrame({'point_id': df['point_id'], 'disparity': d, 'Z': Z, 'X': X, 'Y': Y,
                                       'dZ_1px': dZ})
                far = result.loc[result['Z'].idxmax()]
                print(f"самая далёкая точка: Z = {far['Z']:.1f} м, ошибка на 1 px ≈ {far['dZ_1px']:.1f} м")
                return result.round(4)
        """,
        "review": """
            **Идея.** У параллельной стереопары точка на глубине Z смещается между кадрами на d = f·B/Z пикселей —
            это подобие треугольников. Зная d, получаем глубину, а из модели камеры-обскуры — и X, Y: пиксель задаёт луч,
            глубина — точку на нём. Без калибровки (f, cx, cy, B) пиксели в метры не перевести.

            **Типичные ошибки:**
            - Диспаритет со знаком «правый минус левый» — глубина отрицательная.
            - X = (u − cx) / f без умножения на Z — получили направление луча, а не точку.
            - Не вычли главную точку (cx, cy) — все точки «съехали» вбок.
            - dZ = Z / (fB) или fB/d — неверная чувствительность; правильная растёт как Z².

            **Что вынести:** Z = f·B/d; точность стерео быстро падает с расстоянием (ошибка ∝ Z²), поэтому для дальних
            объектов увеличивают базу или фокусное расстояние. → шпаргалка «3D Computer Vision».
        """,
    },
    {
        "name": "autoencoders",
        "title": "Линейный автоэнкодер (≈ PCA) и поиск аномалий по ошибке восстановления",
        "data": data_autoencoders,
        "task": """
            Файл `input.csv` — показания 4 датчиков `f1…f4` (`id, f1, f2, f3, f4, split`). Строки `train` — только
            нормальная работа оборудования, строки `test` — новые наблюдения, среди которых есть аномалии.

            Известно, что **линейный** автоэнкодер с MSE и горлышком k находит то же подпространство, что PCA.
            Поэтому «обучите» автоэнкодер 4 → 2 → 4 аналитически:

            1. Стандартизуйте признаки по **train** (среднее и стандартное отклонение с ddof = 0, как `StandardScaler`);
               test преобразуйте теми же параметрами.
            2. Энкодер: W — матрица 4 × 2 из двух первых главных компонент стандартизованного train
               (правые сингулярные векторы SVD), код z = x·W. Декодер: x̂ = z·Wᵀ.
            3. Ошибка восстановления объекта — **средний** квадрат по 4 признакам: err = mean((x − x̂)²)
               (в стандартизованных единицах).
            4. Порог — **95-й перцентиль** ошибок на train (`np.percentile` с интерполяцией по умолчанию).
               Объект test — аномалия, если err > порога.

            ## Подсказки

            - Стандартизация обязательна: у `f3` масштаб в десятки раз больше, без неё «главной компонентой» станет просто `f3`.
            - Знак главных компонент не определён (SVD может выдать −v), но x̂ = x·W·Wᵀ от знака не зависит — поэтому
              в ответе только ошибки.
            - `U, S, Vt = np.linalg.svd(Z_train, full_matrices=False)`, `W = Vt[:2].T`, `X_hat = Z @ W @ W.T`;
              то же даёт `PCA(n_components=2)` и `pca.inverse_transform(pca.transform(Z))`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, recon_error, is_anomaly` — только строки test (12 строк) в порядке `id`; `recon_error`
            округлить до 4 знаков, `is_anomaly` — 1 или 0.
        """,
        "solution": """
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            FEATURES = ['f1', 'f2', 'f3', 'f4']


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                scaler = StandardScaler().fit(train[FEATURES])       # параметры — только по train
                Z_train = scaler.transform(train[FEATURES])
                Z_test = scaler.transform(test[FEATURES])

                # «Обучение» линейного AE: оптимальное подпространство = 2 первые главные компоненты
                _, _, Vt = np.linalg.svd(Z_train, full_matrices=False)
                W = Vt[:2].T                                          # энкодер 4 → 2; декодер — Wᵀ

                def recon_error(Z):
                    Z_hat = (Z @ W) @ W.T                             # x → z → x̂
                    return ((Z - Z_hat) ** 2).mean(axis=1)

                # Сверка с PCA из sklearn
                pca = PCA(n_components=2).fit(Z_train)
                assert np.allclose(pca.inverse_transform(pca.transform(Z_test)), Z_test @ W @ W.T)

                err_train, err_test = recon_error(Z_train), recon_error(Z_test)
                threshold = np.percentile(err_train, 95)
                assert np.min(np.abs(err_test - threshold)) > 1e-3   # нет объектов «на пороге»
                print(f"порог (95-й перцентиль train): {threshold:.4f}")

                return pd.DataFrame({'id': test['id'].to_numpy(), 'recon_error': err_test.round(4),
                                     'is_anomaly': (err_test > threshold).astype(int)})
        """,
        "review": """
            **Идея.** Автоэнкодер, обученный на нормальных данных, выучивает их «форму» — здесь это плоскость в
            4-мерном пространстве. Нормальные объекты лежат близко к ней и восстанавливаются хорошо, аномальные — далеко
            от неё, и ошибка восстановления большая. Порог берут по распределению ошибок на train, меток аномалий не нужно.

            **Типичные ошибки:**
            - Нет стандартизации или scaler обучен на всех данных — подпространство «смотрит» не туда, ошибки другие.
            - Сумма квадратов вместо среднего — ошибки в 4 раза больше (порог тоже, флаги совпадут, а числа — нет).
            - Взяли строки `U` вместо столбцов `Vt.T` или последние компоненты вместо первых.
            - Порог по test — утечка: аномалии сами поднимают порог и «прячутся».

            **Что вынести:** линейный AE = PCA; ошибка восстановления — универсальный скор аномальности, а нелинейный
            (глубокий) AE просто умеет «разгибать» изогнутые многообразия. → шпаргалка «Автоэнкодеры».
        """,
    },
    {
        "name": "cnn_architectures",
        "title": "Архитектура CNN: формы выходов и число параметров по слоям",
        "data": data_cnn_architectures,
        "task": """
            Файл `input.csv` описывает небольшую VGG-подобную сеть для картинок **3 × 32 × 32** (каналы × высота × ширина):
            `layer_id, type, out, kernel, stride, padding, bias`. Типы слоёв:

            - `conv` — свёртка `out` фильтров k × k с шагом `stride` и паддингом `padding`; `bias` = 1/0 — есть ли смещение;
            - `bn` — BatchNorm2d; `relu` — активация;
            - `maxpool` — max pooling окном k × k с шагом и паддингом;
            - `flatten` — вытягивание в вектор; `dense` — полносвязный слой на `out` выходов (со смещением).

            Для каждого слоя вычислите форму выхода и число **обучаемых** параметров:

            - размер выхода свёртки и пулинга: ⌊(вход + 2p − k) / s⌋ + 1;
            - conv: k·k·C_in·out + (out, если bias = 1); dense: n_in·out + out;
            - bn: 2·C (γ и β; running mean/var — буферы, не параметры); relu, maxpool, flatten: 0.

            ## Подсказки

            - Пулинг и BN не меняют число каналов; flatten превращает C × H × W в вектор длины C·H·W.
            - Свёртка без смещения (`bias = 0`) — обычная пара с BN: β из BN всё равно играет роль сдвига.
            - Идите по слоям, храня текущую форму `(C, H, W)`: `H = (H + 2*p - k) // s + 1`; после flatten
              запишите `out_c = C*H*W, out_h = out_w = 1`; для dense — `out_c = out, out_h = out_w = 1`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `layer_id, type, out_c, out_h, out_w, params, cum_params` (cum_params — накопленная сумма
            параметров) — 13 строк в порядке `layer_id`. Все значения — целые числа.
        """,
        "solution": """
            def main():
                layers = pd.read_csv('input.csv')
                C, H, W = 3, 32, 32
                rows, total = [], 0
                for L in layers.itertuples():
                    if L.type == 'conv':
                        k, s, p = int(L.kernel), int(L.stride), int(L.padding)
                        params = k * k * C * int(L.out) + (int(L.out) if L.bias == 1 else 0)
                        C, H, W = int(L.out), (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
                    elif L.type == 'maxpool':
                        k, s, p = int(L.kernel), int(L.stride), int(L.padding)
                        params = 0
                        H, W = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
                    elif L.type == 'bn':
                        params = 2 * C                          # γ и β на каждый канал
                    elif L.type == 'flatten':
                        params = 0
                        C, H, W = C * H * W, 1, 1
                    elif L.type == 'dense':
                        params = C * int(L.out) + int(L.out)
                        C, H, W = int(L.out), 1, 1
                    else:                                       # relu
                        params = 0
                    total += params
                    rows.append({'layer_id': L.layer_id, 'type': L.type, 'out_c': C, 'out_h': H, 'out_w': W,
                                 'params': params, 'cum_params': total})

                result = pd.DataFrame(rows)
                conv_share = result.loc[result['type'] == 'conv', 'params'].sum() / total
                print(f"всего параметров: {total}, из них в свёртках: {conv_share:.0%}")
                return result
        """,
        "review": """
            **Идея.** Архитектура CNN — это цепочка преобразований формы тензора. Одна формула ⌊(n + 2p − k)/s⌋ + 1
            описывает и свёртки, и пулинг, а число параметров свёртки зависит только от k, C_in и C_out — но не от
            размера картинки. Полносвязный слой, наоборот, растёт с площадью карты признаков: так в VGG16 из 138 млн
            параметров 124 млн — в трёх dense-слоях.

            **Типичные ошибки:**
            - Забыли C_in в k·k·C_in·out — посчитали параметры одного «плоского» фильтра.
            - Округлили размер вверх вместо ⌊…⌋ (свёртка 3×3 со stride 2 по карте 8×8 даёт 3, а не 4).
            - Посчитали running mean/var BatchNorm как параметры (4C вместо 2C).
            - Прибавили смещение свёртке с `bias = 0`.

            **Что вынести:** форма → формула размера; параметры → k²·C_in·C_out (+C_out). Свёртки 1×1 дёшево меняют
            число каналов, а stride/пулинг дёшево уменьшают карту. → шпаргалка «Архитектуры CNN».
        """,
    },
    {
        "name": "data_augmentation",
        "title": "Геометрические аугментации: картинка и рамка синхронно",
        "data": data_augmentation,
        "task": """
            Дано изображение 6 × 8 (высота × ширина) в длинном формате `input.csv` (`row, col, value`) и рамка объекта
            `box.csv` (`x_min, y_min, x_max, y_max`). Координаты рамки — по **границам** пикселей: рамка покрывает
            столбцы x_min … x_max − 1 и строки y_min … y_max − 1, то есть пиксели `img[y_min:y_max, x_min:x_max]`.

            Примените к картинке **последовательно** преобразования из `ops.csv` и после каждого шага пересчитайте рамку:

            - `hflip` — отражение по горизонтали (`img[:, ::-1]`): x → W − x;
            - `vflip` — отражение по вертикали (`img[::-1]`): y → H − y;
            - `rot90` — поворот на 90° **против часовой стрелки**, как `np.rot90(img)`: новая форма W × H,
              точка (x, y) → (y, W − x), где W — ширина **до** поворота;
            - `crop` — вырезать `img[top:top+height, left:left+width]`; рамку сдвинуть на (−left, −top) и обрезать
              по границам нового кадра (x в [0, width], y в [0, height]).

            После отражений min и max меняются местами — снова упорядочьте координаты.

            ## Подсказки

            - Сумма пикселей внутри рамки — отличная самопроверка: отражения и повороты её не меняют, меняет только crop.
            - Проверьте rot90 на одном пикселе: `np.rot90(img)[W − 1 − c, r] == img[r, c]`.
            - Храните картинку как 2D-массив `img` и рамку как 4 числа; на каждом шаге обновляйте оба и записывайте
              строку результата (включая шаг 0 — исходное состояние).

            ## Ожидаемый формат выходного файла

            CSV с колонками `step, op, height, width, x_min, y_min, x_max, y_max, box_area, box_sum, corner` —
            5 строк: шаг 0 (`op = original`) и шаги 1–4 из `ops.csv`. `height, width` — размер картинки после шага,
            `box_area` — площадь рамки в пикселях, `box_sum` — сумма значений `img[y_min:y_max, x_min:x_max]`,
            `corner` — значение левого верхнего пикселя `img[0, 0]`. Все значения — целые.
        """,
        "solution": """
            def apply(op, img, box, r):
                H, W = img.shape
                x0, y0, x1, y1 = box
                if op == 'hflip':
                    return img[:, ::-1], (W - x1, y0, W - x0, y1)          # min/max меняются местами
                if op == 'vflip':
                    return img[::-1], (x0, H - y1, x1, H - y0)
                if op == 'rot90':
                    # (x, y) → (y, W − x): x-диапазон берётся из y, y-диапазон — из отражённого x
                    return np.rot90(img), (y0, W - x1, y1, W - x0)
                if op == 'crop':
                    t, l, h, w = int(r.top), int(r.left), int(r.height), int(r.width)
                    clip = lambda v, hi: min(max(v, 0), hi)
                    return img[t:t + h, l:l + w], (clip(x0 - l, w), clip(y0 - t, h), clip(x1 - l, w), clip(y1 - t, h))
                raise ValueError(op)


            def main():
                px = pd.read_csv('input.csv')
                img = px.pivot(index='row', columns='col', values='value').to_numpy()
                box = tuple(pd.read_csv('box.csv').iloc[0])
                ops = pd.read_csv('ops.csv')

                def record(step, op):
                    x0, y0, x1, y1 = box
                    return {'step': step, 'op': op, 'height': img.shape[0], 'width': img.shape[1],
                            'x_min': x0, 'y_min': y0, 'x_max': x1, 'y_max': y1,
                            'box_area': (x1 - x0) * (y1 - y0), 'box_sum': int(img[y0:y1, x0:x1].sum()),
                            'corner': int(img[0, 0])}

                rows = [record(0, 'original')]
                for r in ops.itertuples():
                    img, box = apply(r.op, img, box, r)
                    rows.append(record(r.step, r.op))
                    if r.op != 'crop':                              # геометрия без потерь: содержимое рамки то же
                        assert rows[-1]['box_sum'] == rows[0]['box_sum']

                result = pd.DataFrame(rows)
                print(result.to_string(index=False))
                return result
        """,
        "review": """
            **Идея.** В детекции и сегментации разметка — часть данных: любое геометрическое преобразование картинки надо
            применить и к рамке. Правильность легко проверить: отражения и повороты переставляют пиксели, но не меняют
            содержимое рамки, поэтому сумма внутри неё сохраняется. Crop может «отрезать» часть объекта — тогда рамку
            обрезают по кадру (а совсем пропавшие объекты в реальных пайплайнах удаляют).

            **Типичные ошибки:**
            - Отразили картинку, но не рамку — рамка указывает на фон (box_sum меняется после hflip).
            - x → W − 1 − x вместо W − x: это формула для индекса пикселя, а у рамки координаты по границам.
            - Поворот по часовой стрелке вместо против (`np.rot90(img, -1)`) или W после поворота вместо W до.
            - Не упорядочили координаты после отражения — x_min > x_max, площадь отрицательная.

            **Что вынести:** геометрию применяют синхронно к картинке, рамкам и маскам (Albumentations делает это сам);
            цветовые аугментации разметку не трогают. → шпаргалка «Data augmentation».
        """,
    },
    {
        "name": "diffusion_models",
        "title": "Диффузия: прямой процесс q(x_t | x₀) и шаг обратного DDPM",
        "data": data_diffusion,
        "task": """
            Расписание шума DDPM: T = 1000, βₜ линейно от 0.0001 до 0.02 (`np.linspace(1e-4, 0.02, 1000)`),
            шаги нумеруются с t = 1 (β₁ = 0.0001, β₁₀₀₀ = 0.02). Обозначения: αₜ = 1 − βₜ, ᾱₜ = α₁·α₂·…·αₜ.

            Файл `input.csv`: `t, k, x0, eps, eps_pred` — для шагов t ∈ {1, 10, 100, 250, 500, 1000} и 4 «пикселей»
            k = 1…4 исходной картинки: значение x₀, шум ε ~ N(0, 1) и предсказание шума сетью ε_θ(xₜ, t).

            Для каждой строки вычислите:

            1. ᾱₜ (`alpha_bar`).
            2. Зашумлённое значение «одним прыжком»: xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε.
            3. Оценку исходной картинки по предсказанному шуму: x̂₀ = (xₜ − √(1 − ᾱₜ) · ε_θ) / √ᾱₜ.
            4. Среднее обратного шага DDPM: μ = (xₜ − βₜ / √(1 − ᾱₜ) · ε_θ) / √αₜ
               (сэмпл x_{t−1} = μ + σₜ·z; шум z здесь не добавляем).

            ## Подсказки

            - Главное — индексация: `betas[t - 1]` — это βₜ, а ᾱₜ = `np.cumprod(alphas)[t - 1]`.
            - Самопроверка из лекции: √ᾱ₂₅₀ ≈ 0.72, √(1 − ᾱ₂₅₀) ≈ 0.69, а ᾱ₁₀₀₀ < 0.0001 — почти чистый шум.
            - Если бы ε_θ = ε, то x̂₀ = x₀ точно. Здесь сеть ошибается — посмотрите, как ошибка x̂₀ растёт с t:
              она умножается на √(1 − ᾱₜ)/√ᾱₜ.

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, k, alpha_bar, x_t, x0_hat, mu` — 24 строки в порядке файла (по t, внутри — по k);
            `alpha_bar` округлить до 6 знаков, остальное — до 4.
        """,
        "solution": """
            T = 1000


            def main():
                df = pd.read_csv('input.csv')
                betas = np.linspace(1e-4, 0.02, T)
                alphas = 1 - betas
                alpha_bar = np.cumprod(alphas)

                t = df['t'].to_numpy()
                b, a, ab = betas[t - 1], alphas[t - 1], alpha_bar[t - 1]   # шаги нумеруются с 1
                x0, eps, eps_pred = df['x0'], df['eps'], df['eps_pred']

                x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps               # q(x_t | x_0) одним прыжком
                x0_hat = (x_t - np.sqrt(1 - ab) * eps_pred) / np.sqrt(ab)
                mu = (x_t - b / np.sqrt(1 - ab) * eps_pred) / np.sqrt(a)     # среднее p_θ(x_{t−1} | x_t)

                # Проверка: при идеальной сети μ совпадает со средним апостериора q(x_{t−1} | x_t, x_0)
                ab_prev = np.where(t > 1, alpha_bar[np.maximum(t - 2, 0)], 1.0)
                mu_ideal = (x_t - b / np.sqrt(1 - ab) * eps) / np.sqrt(a)
                mu_post = np.sqrt(ab_prev) * b / (1 - ab) * x0 + np.sqrt(a) * (1 - ab_prev) / (1 - ab) * x_t
                assert np.allclose(mu_ideal, mu_post)
                print(pd.DataFrame({'t': t, 'signal': np.sqrt(ab), 'noise': np.sqrt(1 - ab)}).drop_duplicates().round(3))

                return pd.DataFrame({'t': t, 'k': df['k'], 'alpha_bar': ab.round(6), 'x_t': x_t.round(4),
                                     'x0_hat': x0_hat.round(4), 'mu': mu.round(4)})
        """,
        "review": """
            **Идея.** Прямой процесс не нужно моделировать шаг за шагом: x_t = √ᾱₜ·x₀ + √(1 − ᾱₜ)·ε — смесь картинки
            и шума с известными весами. Сеть учится предсказывать ε, и по этому предсказанию можно и «угадать» x₀,
            и сделать шаг назад. На больших t сигнала почти нет, поэтому малая ошибка ε_θ даёт огромную ошибку x̂₀ —
            генерация идёт маленькими шагами, а x̂₀ на практике обрезают до [−1, 1].

            **Типичные ошибки:**
            - Сдвиг индекса на 1 (`betas[t]` вместо `betas[t − 1]`) — на больших t ответ заметно расходится.
            - ᾱₜ как сумма или как αₜ вместо произведения.
            - Веса √ᾱ и √(1 − ᾱ) без корней — смесь уже не сохраняет дисперсию.
            - В μ поделили на √ᾱₜ вместо √αₜ или умножили ε_θ на √(1 − ᾱₜ) вместо βₜ/√(1 − ᾱₜ).

            **Что вынести:** обучение диффузии — регрессия шума по MSE на случайном t; ᾱₜ задаёт долю сигнала, и всё
            остальное (x_t, x̂₀, шаг DDPM) — простые формулы через неё. → шпаргалка «Diffusion модели».
        """,
    },
    {
        "name": "neural_style_transfer",
        "title": "Style transfer: матрица Грама, content loss и style loss",
        "data": data_style_transfer,
        "task": """
            Файл `input.csv` — карты признаков одного слоя CNN (C = 3 канала, 4 × 4) в длинном формате
            `map, channel, row, col, value` для нескольких изображений:

            - `content` — фотография (содержание), `style` — картина (стиль);
            - кандидаты-результаты: `copy`, `style_shuffled`, `blend`, `noise`.

            Для каждого кандидата F посчитайте:

            1. **Content loss**: L_c = Σ (F − F_content)² — сумма квадратов по всем 48 элементам.
            2. **Матрицу Грама**: разверните карту в матрицу M размера C × (H·W) и вычислите G = M·Mᵀ / (C·H·W).
            3. **Style loss**: L_s = Σᵢⱼ (G(F)ᵢⱼ − G(F_style)ᵢⱼ)² — сумма по 9 элементам матриц 3 × 3.
            4. **Итоговую потерю**: L = α·L_c + β·L_s при α = 1, β = 10.

            ## Подсказки

            - Порядок развёртки: строка M — один канал, столбцы — позиции 4 × 4 (`F.reshape(C, H*W)`).
            - Кандидат `style_shuffled` — карта стиля с перемешанными позициями. Каким должен выйти его style loss?
            - Соберите карту из длинной таблицы: отсортируйте по `channel, row, col` и сделайте `values.reshape(3, 4, 4)`;
              затем `G = M @ M.T / M.size`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `candidate, content_loss, style_loss, total_loss` — 4 строки в порядке `copy, style_shuffled,
            blend, noise`; значения округлить до 4 знаков.
        """,
        "solution": """
            ALPHA, BETA = 1.0, 10.0
            CANDIDATES = ['copy', 'style_shuffled', 'blend', 'noise']


            def gram(F):
                C, H, W = F.shape
                M = F.reshape(C, H * W)             # каналы × позиции
                return M @ M.T / (C * H * W)        # сумма по позициям — расположение «забывается»


            def main():
                df = pd.read_csv('input.csv').sort_values(['map', 'channel', 'row', 'col'])
                maps = {name: g['value'].to_numpy().reshape(3, 4, 4) for name, g in df.groupby('map')}
                G_style = gram(maps['style'])

                rows = []
                for name in CANDIDATES:
                    F = maps[name]
                    lc = np.sum((F - maps['content']) ** 2)
                    ls = np.sum((gram(F) - G_style) ** 2)
                    rows.append({'candidate': name, 'content_loss': lc, 'style_loss': ls,
                                 'total_loss': ALPHA * lc + BETA * ls})

                result = pd.DataFrame(rows)
                assert np.isclose(result.loc[1, 'style_loss'], 0)   # перестановка позиций не меняет Грама
                print("лучший кандидат:", result.loc[result['total_loss'].idxmin(), 'candidate'])
                return result.round(4)
        """,
        "review": """
            **Идея.** Содержание — это сами карты признаков: где что находится. Стиль — их статистика: какие каналы
            включаются вместе. Матрица Грама суммирует произведения каналов по всем позициям, поэтому расположение
            исчезает: перемешанная карта стиля имеет style loss ровно 0, хотя как «картинка» она бессмысленна.
            Итоговая потеря — компромисс: копия фотографии идеальна по содержанию, но плоха по стилю, а смесь выигрывает.

            **Типичные ошибки:**
            - Развернули карту как (H·W) × C и посчитали Mᵀ·M по позициям — получилась матрица 16 × 16, а не 3 × 3.
            - Забыли нормировку 1/(C·H·W) — style loss в 48² раз больше.
            - Content loss со средним вместо суммы или со сравнением Грама — это уже style loss.
            - Сборка карты без сортировки — каналы и позиции перепутаны.

            **Что вынести:** L_content сравнивает признаки, L_style — матрицы Грама; β/α задаёт баланс «стиль против
            содержания». → шпаргалка «Neural style transfer».
        """,
    },
    {
        "name": "pooling_layers",
        "title": "Max и average pooling со stride и padding",
        "data": data_pooling,
        "task": """
            Карта признаков 6 × 7 (один канал) задана в длинном формате `input.csv` (`row, col, value`).
            Файл `configs.csv` — 5 вариантов пулинга: `config_id, mode, kernel, stride, padding`.

            Выполните каждый пулинг (по правилам PyTorch):

            - размер выхода: ⌊(вход + 2p − k) / s⌋ + 1 по высоте и по ширине;
            - `max` — максимум в окне; паддинг для max pooling — это **−∞** (пустые клетки никогда не выигрывают);
            - `avg` — среднее в окне; паддинг нулями, и нули **входят** в среднее: делим всегда на k·k
              (`nn.AvgPool2d` с `count_include_pad=True` по умолчанию);
            - `global_avg` / `global_max` — среднее / максимум по всей карте, выход 1 × 1.

            ## Подсказки

            - Окно выхода (i, j) начинается в строке i·s − p и столбце j·s − p исходной карты.
            - Левый верхний угол карты целиком отрицательный: при паддинге нулями max pooling выдал бы там 0 — это ошибка.
            - Дополните карту: `np.pad(x, p, constant_values=-np.inf)` для max и `constant_values=0` для avg, затем
              возьмите окна `xp[i*s:i*s+k, j*s:j*s+k]`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `config_id, out_h, out_w, row, col, value` — для каждой конфигурации все клетки выхода
            построчно (row, затем col), конфигурации по порядку `config_id`; `value` округлить до 4 знаков.
        """,
        "solution": """
            def pool(x, mode, k, s, p):
                fill = -np.inf if mode == 'max' else 0.0
                xp = np.pad(x.astype(float), p, constant_values=fill)
                H = (x.shape[0] + 2 * p - k) // s + 1
                W = (x.shape[1] + 2 * p - k) // s + 1
                out = np.empty((H, W))
                for i in range(H):
                    for j in range(W):
                        win = xp[i * s:i * s + k, j * s:j * s + k]
                        out[i, j] = win.max() if mode == 'max' else win.sum() / (k * k)   # делим на k², вкл. паддинг
                return out


            def main():
                px = pd.read_csv('input.csv')
                x = px.pivot(index='row', columns='col', values='value').to_numpy()
                configs = pd.read_csv('configs.csv')

                rows = []
                for c in configs.itertuples():
                    if c.mode == 'global_avg':
                        out = np.array([[x.mean()]])
                    elif c.mode == 'global_max':
                        out = np.array([[x.max()]])
                    else:
                        out = pool(x, c.mode, c.kernel, c.stride, c.padding)
                    print(f"config {c.config_id} ({c.mode}): {x.shape} → {out.shape}")
                    for i in range(out.shape[0]):
                        for j in range(out.shape[1]):
                            rows.append({'config_id': c.config_id, 'out_h': out.shape[0], 'out_w': out.shape[1],
                                         'row': i, 'col': j, 'value': round(float(out[i, j]), 4)})
                return pd.DataFrame(rows)
        """,
        "review": """
            **Идея.** Пулинг сжимает карту признаков без обучаемых параметров: каждое окно → одно число. Размер выхода
            считается по той же формуле, что у свёртки. Детали паддинга важны: для максимума пустые клетки не должны
            «выигрывать» (поэтому −∞), а для среднего PyTorch по умолчанию считает нули паддинга полноправными
            элементами окна — у краёв среднее «тянется» к нулю.

            **Типичные ошибки:**
            - Паддинг нулями в max pooling — в отрицательном углу появляется 0.
            - В avg pooling делили на число реальных (не паддинговых) клеток — это `count_include_pad=False`.
            - Размер выхода с округлением вверх (ceil_mode) — лишняя строка или столбец.
            - Окна без шага (stride 1) — выход почти такого же размера, как вход.

            **Что вынести:** выход = ⌊(n + 2p − k)/s⌋ + 1; max pooling — «есть ли признак в районе», average — «сколько
            его в среднем»; global average pooling заменяет flatten перед классификатором. → шпаргалка «Пулинг-слои».
        """,
    },
    {
        "name": "pose_estimation",
        "title": "Оценка позы: OKS и PCK по ключевым точкам COCO",
        "data": data_pose,
        "task": """
            Модель оценки позы нашла 17 ключевых точек COCO для 4 людей. Файлы:

            - `input.csv` — `person_id, kp_id, name, x_gt, y_gt, v, x_pred, y_pred`: разметка, флаг видимости v
              (0 — не размечена, 1 — размечена, но закрыта, 2 — видна) и предсказание;
            - `persons.csv` — `person_id, bbox_w, bbox_h, area` (площадь сегментации человека, пикс²);
            - `sigmas.csv` — `kp_id, name, sigma`: «трудность» точки из COCO (σ глаза меньше, чем σ бедра).

            В метриках участвуют только **размеченные** точки (v > 0). Для каждого человека посчитайте:

            1. `n_labeled` — число точек с v > 0.
            2. `mean_error` — среднее евклидово расстояние dᵢ между предсказанием и разметкой (пиксели).
            3. `pck` — доля точек с dᵢ < 0.1 · max(bbox_w, bbox_h) (PCK@0.1 относительно размера рамки).
            4. `oks` = mean по точкам exp(−dᵢ² / (2 · s² · kᵢ²)), где s² = area, kᵢ = 2·σᵢ
               (как в `pycocotools`).

            ## Подсказки

            - Для v = 0 координаты разметки — нули, а предсказание может быть где угодно: такие точки просто выбросьте.
            - Нормировка на размер делает OKS сравнимым для большого и маленького человека: та же ошибка в пикселях
              у маленького человека штрафуется сильнее.
            - `d2 = (x_pred - x_gt)**2 + (y_pred - y_gt)**2`, `k2 = (2*sigma)**2`, `oks = np.exp(-d2 / (2*area*k2)).mean()`
              по размеченным точкам человека.

            ## Ожидаемый формат выходного файла

            CSV с колонками `person_id, n_labeled, mean_error, pck, oks` — 4 строки в порядке `person_id`; значения
            округлить до 4 знаков.
        """,
        "solution": """
            PCK_ALPHA = 0.1


            def main():
                kp = pd.read_csv('input.csv').merge(pd.read_csv('sigmas.csv')[['kp_id', 'sigma']], on='kp_id')
                persons = pd.read_csv('persons.csv').set_index('person_id')

                rows = []
                for pid, g in kp.groupby('person_id', sort=True):
                    g = g[g['v'] > 0]                                    # только размеченные точки
                    p = persons.loc[pid]
                    d2 = (g['x_pred'] - g['x_gt']) ** 2 + (g['y_pred'] - g['y_gt']) ** 2
                    d = np.sqrt(d2)
                    thr = PCK_ALPHA * max(p['bbox_w'], p['bbox_h'])
                    assert np.min(np.abs(d - thr)) > 1e-3                # нет точек ровно на пороге PCK
                    k2 = (2 * g['sigma']) ** 2
                    oks = np.exp(-d2 / (2 * p['area'] * k2)).mean()      # гауссова близость с учётом размера и σ
                    rows.append({'person_id': pid, 'n_labeled': len(g), 'mean_error': d.mean(),
                                 'pck': (d < thr).mean(), 'oks': oks})

                result = pd.DataFrame(rows)
                print(result.round(3).to_string(index=False))
                return result.round(4)
        """,
        "review": """
            **Идея.** Ошибку ключевых точек нельзя мерить просто в пикселях: 5 пикселей — пустяк для человека во весь кадр
            и провал для фигурки в 100 пикселей. OKS нормирует ошибку на размер человека (s² = area) и на трудность
            точки (σ): глаза размечают точно, бёдра — с разбросом, поэтому промах по глазу штрафуется сильнее.
            OKS играет в позе роль IoU в детекции: по нему считают AP. PCK проще — доля точек внутри радиуса.

            **Типичные ошибки:**
            - Учли точки с v = 0 — «улетевшие» предсказания невидимых точек резко портят все метрики.
            - kᵢ = σᵢ вместо 2σᵢ или s вместо s² — OKS выходит заметно ниже.
            - Порог PCK по площади, а не по max(w, h), или ≤ вместо < (здесь точек на пороге нет, но в общем случае важно).
            - Средний OKS как exp от средней ошибки — это не то же самое, что среднее exp.

            **Что вынести:** метрики позы нормируют ошибку на масштаб объекта; OKS — стандарт COCO, PCK/PCKh — простой
            и наглядный вариант. → шпаргалка «Оценка позы человека».
        """,
    },
    {
        "name": "resnet",
        "title": "Residual-блок ResNet с проекцией в skip-связи",
        "data": data_resnet,
        "task": """
            Реализуйте прямой проход BasicBlock ResNet **в режиме eval** на NumPy. Вход `input.csv` — тензор 2 × 5 × 5
            (`channel, row, col, value`). Веса в `weights.json` (раскладка PyTorch: свёртка — `(out, in, kH, kW)`,
            без смещений):

            ```
            основной путь:  x → conv1 (3×3, stride 2, padding 1, 2→4) → bn1 → ReLU → conv2 (3×3, stride 1, padding 1, 4→4) → bn2  = F(x)
            skip-связь:     x → proj (1×1, stride 2, padding 0, 2→4) → bn_proj                                           = S(x)
            выход:          y = ReLU(F(x) + S(x))
            ```

            - Свёртка — как `nn.Conv2d` (кросс-корреляция, без переворота ядра), паддинг нулями:
              out[o, i, j] = Σ_c Σ_a Σ_b W[o, c, a, b] · x_pad[c, i·s + a, j·s + b].
            - BatchNorm в eval: (x − running_mean) / √(running_var + eps) · gamma + beta, по каждому каналу; eps — в файле.

            ## Подсказки

            - Размеры: conv1 со stride 2 даёт ⌊(5 + 2 − 3)/2⌋ + 1 = 3, и proj должна дать те же 4 × 3 × 3 — иначе сложить
              F(x) и x нельзя. Именно поэтому в skip-связи стоит проекция, а не identity.
            - В режиме eval BN использует **сохранённые** running mean/var, а не статистику текущего входа.
            - Одна функция `conv2d(x, W, stride, pad)`: `xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))`, затем для
              каждой позиции `np.sum(W[o] * xp[:, i*s:i*s+k, j*s:j*s+k])`; BN — с `[:, None, None]` для broadcast.

            ## Ожидаемый формат выходного файла

            CSV с колонками `channel, row, col, residual, shortcut, out` (residual = F(x), shortcut = S(x), out = y) —
            36 строк в порядке channel, row, col; значения округлить до 4 знаков.
        """,
        "solution": """
            import json


            def conv2d(x, W, stride, pad):
                xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
                C_out, _, k, _ = W.shape
                H = (x.shape[1] + 2 * pad - k) // stride + 1
                Wd = (x.shape[2] + 2 * pad - k) // stride + 1
                out = np.zeros((C_out, H, Wd))
                for o in range(C_out):
                    for i in range(H):
                        for j in range(Wd):
                            out[o, i, j] = np.sum(W[o] * xp[:, i * stride:i * stride + k, j * stride:j * stride + k])
                return out


            def batchnorm_eval(x, p, eps):
                g, b, m, v = (np.array(p[key])[:, None, None] for key in ['gamma', 'beta', 'running_mean', 'running_var'])
                return (x - m) / np.sqrt(v + eps) * g + b


            def main():
                df = pd.read_csv('input.csv').sort_values(['channel', 'row', 'col'])
                x = df['value'].to_numpy().reshape(2, 5, 5)
                w = json.load(open('weights.json', encoding='utf-8'))
                eps = w['eps']

                out = np.maximum(0, batchnorm_eval(conv2d(x, np.array(w['conv1']), 2, 1), w['bn1'], eps))
                F = batchnorm_eval(conv2d(out, np.array(w['conv2']), 1, 1), w['bn2'], eps)       # остаток F(x)
                S = batchnorm_eval(conv2d(x, np.array(w['proj']), 2, 0), w['bn_proj'], eps)      # проекция x
                assert F.shape == S.shape == (4, 3, 3)
                y = np.maximum(0, F + S)

                n_params = sum(np.array(w[k]).size for k in ['conv1', 'conv2', 'proj']) + 3 * 2 * 4
                print(f"обучаемых параметров в блоке: {n_params}; нулевых выходов после ReLU: {(y == 0).sum()} из {y.size}")

                C, H, W = y.shape
                idx = np.indices((C, H, W)).reshape(3, -1)
                return pd.DataFrame({'channel': idx[0], 'row': idx[1], 'col': idx[2], 'residual': F.ravel(),
                                     'shortcut': S.ravel(), 'out': y.ravel()}).round(4)
        """,
        "review": """
            **Идея.** Блок ResNet учит не всё преобразование, а поправку F(x) к входу: y = F(x) + x. Градиент через
            сложение проходит без изменений (∂y/∂x = 1 + ∂F/∂x), поэтому глубокие сети обучаются. Когда блок меняет
            размер (stride 2) и число каналов, x приходится привести к той же форме — свёрткой 1×1 с тем же шагом.

            **Типичные ошибки:**
            - ReLU перед сложением, а не после (или вообще после bn2 в основном пути) — значения residual неверны.
            - Перевернули ядро (настоящая свёртка вместо кросс-корреляции PyTorch).
            - BN по статистике текущего тензора (режим train) вместо running mean/var.
            - Перепутали оси весов: в PyTorch `(out, in, kH, kW)`, а в Keras — `(kH, kW, in, out)`.

            **Что вынести:** BasicBlock = conv-bn-relu-conv-bn + shortcut, затем ReLU; при смене формы shortcut —
            проекция 1×1. → шпаргалка «ResNet».
        """,
    },
    {
        "name": "transfer_learning_cnn",
        "title": "Transfer learning: замороженный backbone и новая голова",
        "data": data_transfer,
        "task": """
            Вы взяли предобученную CNN, **заморозили** все её слои и прогнали через неё картинки трёх классов
            (`cat`, `dog`, `fox`). Файл `input.csv` — полученные эмбеддинги после global average pooling:
            `id, e1…e8, label, split` (45 картинок `train`, 15 — `test`).

            Обучите только новую голову-классификатор (это и есть feature extraction):

            1. Стандартизуйте `e1…e8` по train (`StandardScaler`).
            2. Обучите `LogisticRegression(C=1.0, max_iter=5000, tol=1e-8)` (softmax-регрессия на 3 класса) на train.
               Маленький `tol` нужен, чтобы ответ не зависел от деталей сходимости.
            3. Для каждой картинки test выдайте предсказанный класс и вероятности всех трёх классов.

            ## Подсказки

            - Backbone заморожен — его выходы фиксированы, поэтому эмбеддинги можно посчитать один раз и закэшировать.
              Обучается только голова: 8·3 + 3 = 27 параметров против ~11 млн у ResNet-18.
            - Порядок классов в `predict_proba` — `model.classes_` (по алфавиту: cat, dog, fox).
            - `scaler = StandardScaler().fit(X_train)`, `model.fit(scaler.transform(X_train), y_train)`,
              `proba = model.predict_proba(scaler.transform(X_test))`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `id, pred, p_cat, p_dog, p_fox` — 15 строк test в порядке `id`; вероятности округлить
            до 4 знаков.
        """,
        "solution": """
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            FEATURES = [f'e{k}' for k in range(1, 9)]


            def main():
                df = pd.read_csv('input.csv')
                train, test = df[df['split'] == 'train'], df[df['split'] == 'test']

                scaler = StandardScaler().fit(train[FEATURES])           # статистики — только по train
                head = LogisticRegression(C=1.0, max_iter=5000, tol=1e-8)
                head.fit(scaler.transform(train[FEATURES]), train['label'])

                proba = head.predict_proba(scaler.transform(test[FEATURES]))
                n_trainable = head.coef_.size + head.intercept_.size
                acc = (head.classes_[proba.argmax(axis=1)] == test['label'].to_numpy()).mean()
                print(f"обучаемых параметров головы: {n_trainable}; accuracy на test: {acc:.3f}")

                result = pd.DataFrame({'id': test['id'].to_numpy(), 'pred': head.classes_[proba.argmax(axis=1)]})
                for k, cls in enumerate(head.classes_):
                    result[f'p_{cls}'] = proba[:, k].round(4)
                return result
        """,
        "review": """
            **Идея.** Предобученная сеть уже умеет превращать картинку в информативный вектор признаков. При малом
            наборе данных её замораживают и учат только голову — по сути, логистическую регрессию на эмбеддингах.
            Это быстро (признаки считаются один раз), почти не переобучается и часто даёт сильный бейзлайн; следующий
            шаг — разморозить верхние блоки и дообучить с маленьким learning rate.

            **Типичные ошибки:**
            - Scaler обучен на train + test — утечка, вероятности немного другие.
            - Перепутан порядок классов в столбцах вероятностей (не по `classes_`).
            - Другие гиперпараметры (C, без стандартизации) — вероятности расходятся с эталоном.
            - Мысль «разморозим всё сразу со случайной головой» — её большие градиенты портят предобученные признаки.

            **Что вынести:** feature extraction = замороженный backbone + обучаемая голова; число обучаемых параметров —
            только у головы. → шпаргалка «Transfer learning для CNN».
        """,
    },
    {
        "name": "vision_transformers_vit",
        "title": "ViT: нарезка на патчи, линейная проекция, [CLS] и позиции",
        "data": data_vit,
        "task": """
            Изображение 6 × 6 с 2 каналами задано в длинном формате `input.csv` (`row, col, channel, value`).
            Веса в `weights.json`: матрица проекции `E` (18 × 4), смещение `b` (4), токен `cls` (4) и позиционные
            эмбеддинги `pos` (5 × 4).

            Постройте входную последовательность энкодера ViT с патчами P = 3:

            1. Нарежьте картинку на N = (6/3)·(6/3) = 4 патча 3 × 3 × 2. Порядок патчей — построчно по сетке:
               (0, 0), (0, 1), (1, 0), (1, 1).
            2. Каждый патч вытяните в вектор длины 18 в порядке **(строка внутри патча, столбец, канал)** — как
               `img[r:r+3, c:c+3, :].reshape(-1)` для массива формы (H, W, C).
            3. Эмбеддинг патча: zₖ = patchₖ · E + b; в начало добавьте токен [CLS] = `cls`.
            4. Прибавьте позиционные эмбеддинги: токен номер i (0 — CLS, 1…4 — патчи) получает `pos[i]`.

            ## Подсказки

            - Для ViT-B/16 на 224 × 224 то же самое даёт 196 патчей по 768 чисел и 197 токенов — посчитайте, чтобы
              почувствовать масштаб.
            - Соберите массив (H, W, C): отсортируйте по `row, col, channel` и `reshape(6, 6, 2)`.
            - Векторно, как в лекции: `img.reshape(2, 3, 2, 3, 2).transpose(0, 2, 1, 3, 4).reshape(4, 18)`; затем
              `tokens = np.vstack([cls, patches @ E + b]) + pos`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `token, patch_row, patch_col, z1, z2, z3, z4` — 5 строк (token = 0…4); для CLS
            `patch_row = patch_col = -1`; значения z округлить до 4 знаков.
        """,
        "solution": """
            import json

            P = 3


            def main():
                df = pd.read_csv('input.csv').sort_values(['row', 'col', 'channel'])
                img = df['value'].to_numpy().reshape(6, 6, 2)             # (H, W, C)
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                H, W, C = img.shape
                gh, gw = H // P, W // P

                patches = img.reshape(gh, P, gw, P, C).transpose(0, 2, 1, 3, 4).reshape(gh * gw, P * P * C)
                assert np.allclose(patches[1], img[0:3, 3:6, :].reshape(-1))   # патч (0, 1) — правый верхний

                tokens = np.vstack([w['cls'], patches @ w['E'] + w['b']]) + w['pos']   # [CLS] + патчи, + позиции
                print(f"патчей: {len(patches)}, токенов: {len(tokens)}, размерность: {tokens.shape[1]}")

                result = pd.DataFrame(tokens.round(4), columns=['z1', 'z2', 'z3', 'z4'])
                result.insert(0, 'token', np.arange(len(tokens)))
                result.insert(1, 'patch_row', [-1] + [k // gw for k in range(gh * gw)])
                result.insert(2, 'patch_col', [-1] + [k % gw for k in range(gh * gw)])
                return result
        """,
        "review": """
            **Идея.** ViT превращает картинку в «предложение»: патч — слово, линейная проекция — эмбеддинг слова. Так как
            self-attention не знает порядка, к каждому токену прибавляют позиционный эмбеддинг; отдельный обучаемый
            токен [CLS] собирает информацию со всей картинки и идёт в голову классификации. Линейная проекция патчей —
            это ровно свёртка с ядром P×P и шагом P.

            **Типичные ошибки:**
            - `img.reshape(4, 18)` без transpose — «патчи» оказываются полосками строк, а не квадратами.
            - Порядок развёртки (канал, строка, столбец) вместо (строка, столбец, канал) — другой вектор патча.
            - Забыли [CLS] или прибавили позиции до добавления CLS — эмбеддинги сдвинуты на одну позицию.
            - Не прибавили позиционный эмбеддинг к самому CLS.

            **Что вынести:** N = (H/P)·(W/P) токенов + 1 CLS; длина последовательности (и стоимость attention ∝ N²)
            быстро растёт при уменьшении патча. → шпаргалка «Vision Transformers (ViT)».
        """,
    },
    {
        "name": "yolo",
        "title": "YOLO: декодирование сетки и якорей, порог уверенности и NMS",
        "data": data_yolo,
        "task": """
            YOLO-подобный детектор (в стиле YOLOv3) для картинки 128 × 128 выдал «сырой» тензор на сетке 4 × 4
            (шаг stride = 32 пикселя) с 2 якорями на клетку. Файл `input.csv` — 32 строки
            `gy, gx, anchor, tx, ty, tw, th, obj, cls_person, cls_car` (все значения — логиты/сырые выходы).
            Параметры — в `config.json`: размеры якорей `anchors` (ширина, высота в пикселях), классы, пороги.

            Декодирование (σ — сигмоида):

            - центр: bx = (σ(tx) + gx) · stride, by = (σ(ty) + gy) · stride;
            - размер: bw = p_w · exp(tw), bh = p_h · exp(th), где (p_w, p_h) — якорь `anchor`;
            - класс — с максимальным логитом; уверенность score = σ(obj) · σ(логит этого класса);
            - рамка в углах: x1 = bx − bw/2, y1 = by − bh/2, x2 = bx + bw/2, y2 = by + bh/2, затем **обрезать** по
              картинке [0, 128].

            Постобработка:

            1. Оставить рамки со score **> 0.25** (`score_threshold`).
            2. NMS **по каждому классу отдельно**: по убыванию score берём лучшую рамку и удаляем рамки того же класса
               с IoU **> 0.5** с ней (IoU — по обрезанным рамкам, площадь (x2 − x1)·(y2 − y1)).

            ## Подсказки

            - Сигмоида держит центр внутри «своей» клетки, exp — делает размер положительным и масштабирует якорь.
            - Почти все 32 предсказания — фон с очень маленьким σ(obj). После порога остаётся горстка рамок, и NMS
              схлопывает дубли одного объекта из соседних клеток и якорей.
            - `cx = (sig(tx) + gx) * stride`, `w = anchors[anchor][0] * np.exp(tw)`; NMS — как в упражнении
              «IoU и NMS»: `order.pop(0)` и фильтр `iou <= 0.5`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `gy, gx, anchor, cls, score, x1, y1, x2, y2` — оставленные рамки по убыванию `score`;
            `score` округлить до 4 знаков, координаты — до 2.
        """,
        "solution": """
            import json


            def sigmoid(v):
                return 1 / (1 + np.exp(-v))


            def iou(a, b):
                w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
                return w * h / (area(a) + area(b) - w * h)


            def main():
                df = pd.read_csv('input.csv')
                cfg = json.load(open('config.json', encoding='utf-8'))
                s, size = cfg['stride'], cfg['image_size']
                anchors = np.array(cfg['anchors'], dtype=float)

                bx = (sigmoid(df['tx']) + df['gx']) * s                       # центр в пикселях
                by = (sigmoid(df['ty']) + df['gy']) * s
                bw = anchors[df['anchor'], 0] * np.exp(df['tw'])               # поправка к якорю
                bh = anchors[df['anchor'], 1] * np.exp(df['th'])
                logits = df[['cls_person', 'cls_car']].to_numpy()
                cls_idx = logits.argmax(axis=1)
                score = sigmoid(df['obj']) * sigmoid(logits.max(axis=1))

                det = df[['gy', 'gx', 'anchor']].copy()
                det['cls'] = np.array(cfg['classes'])[cls_idx]
                det['score'] = score
                det['x1'], det['y1'] = np.clip(bx - bw / 2, 0, size), np.clip(by - bh / 2, 0, size)
                det['x2'], det['y2'] = np.clip(bx + bw / 2, 0, size), np.clip(by + bh / 2, 0, size)

                assert np.min(np.abs(det['score'] - cfg['score_threshold'])) > 1e-3   # нет рамок на пороге
                det = det[det['score'] > cfg['score_threshold']]
                print(f"после порога: {len(det)} из {len(df)}")

                keep = []
                for _, g in det.groupby('cls'):
                    order = list(g.sort_values('score', ascending=False).index)
                    while order:
                        best = order.pop(0)
                        keep.append(best)
                        a = det.loc[best, ['x1', 'y1', 'x2', 'y2']].to_numpy()
                        ious = {i: iou(a, det.loc[i, ['x1', 'y1', 'x2', 'y2']].to_numpy()) for i in order}
                        assert all(abs(v - cfg['iou_threshold']) > 1e-3 for v in ious.values())
                        order = [i for i in order if ious[i] <= cfg['iou_threshold']]

                result = det.loc[keep].sort_values('score', ascending=False).reset_index(drop=True)
                result['score'] = result['score'].round(4)
                result[['x1', 'y1', 'x2', 'y2']] = result[['x1', 'y1', 'x2', 'y2']].round(2)
                return result
        """,
        "review": """
            **Идея.** Сеть YOLO не выдаёт рамки напрямую: каждая клетка сетки и каждый якорь дают сырые числа, которые
            надо декодировать. Сигмоида от tx, ty — смещение центра внутри клетки, exp(tw), exp(th) — масштаб якоря,
            objectness × вероятность класса — уверенность. Потом порог выбрасывает фон, а NMS по классам — дубли одного
            объекта, найденного соседними клетками и разными якорями.

            **Типичные ошибки:**
            - Забыли прибавить gx, gy — все рамки сбились в левый верхний угол.
            - bw = p_w · tw или exp(tw) без якоря — размеры бессмысленные.
            - Перепутали gx/gy (столбец/строку) — рамки отражены относительно диагонали.
            - Score только по objectness, без вероятности класса, или NMS без учёта класса — «человек» на машине пропадает.
            - Не обрезали рамку по картинке — машина у края даёт другой x2 и IoU.

            **Что вынести:** декодирование = сетка + якоря + sigmoid/exp; постобработка = порог + NMS по классам.
            → шпаргалка «Детекция объектов (YOLO)».
        """,
    },
    {
        "name": "gru",
        "title": "Ячейка GRU: прямой проход в раскладке PyTorch",
        "data": data_gru,
        "task": """
            Дана последовательность из 5 шагов `input.csv` (`t, x1, x2`) и веса однослойного GRU (вход D = 2,
            скрытое состояние H = 3) в `weights.json` в раскладке PyTorch `nn.GRU`: `W_ih` (9 × 2), `W_hh` (9 × 3),
            `b_ih` (9), `b_hh` (9), а также начальное состояние `h0` (3).

            Строки весов разбиты на 3 блока по H = 3 в порядке **r, z, n** (как в PyTorch): строки 0–2 — reset gate r,
            3–5 — update gate z, 6–8 — кандидат n. Для t = 1…5, начиная с h₀ из файла:

            - rₜ = σ(W_ir·xₜ + b_ir + W_hr·hₜ₋₁ + b_hr)
            - zₜ = σ(W_iz·xₜ + b_iz + W_hz·hₜ₋₁ + b_hz)
            - nₜ = tanh(W_in·xₜ + b_in + rₜ ⊙ (W_hn·hₜ₋₁ + b_hn))
            - hₜ = (1 − zₜ) ⊙ nₜ + zₜ ⊙ hₜ₋₁

            ## Подсказки

            - Обратите внимание на две детали PyTorch: r умножает **вместе со смещением** (W_hn·h + b_hn), а z ≈ 1
              означает «сохранить старое» (в лекции формула записана зеркально).
            - Нельзя просто сложить `W_ih @ x + W_hh @ h` и разрезать на 3 части, как в LSTM: блок n собирается отдельно.
            - `gi = W_ih @ x + b_ih`, `gh = W_hh @ h + b_hh`, `i_r, i_z, i_n = np.split(gi, 3)`, `h_r, h_z, h_n =
              np.split(gh, 3)`, затем `r = σ(i_r + h_r)`, `z = σ(i_z + h_z)`, `n = tanh(i_n + r * h_n)`.

            ## Ожидаемый формат выходного файла

            CSV с колонками `t, h1, h2, h3, z1, z2, z3` — 5 строк (t = 1…5); значения округлить до 4 знаков.
        """,
        "solution": """
            import json


            def sigmoid(x):
                return 1 / (1 + np.exp(-x))


            def main():
                df = pd.read_csv('input.csv')
                w = {k: np.array(v) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}
                h = w['h0'].copy()

                rows = []
                for t, x in zip(df['t'], df[['x1', 'x2']].to_numpy()):
                    gi = w['W_ih'] @ x + w['b_ih']               # вклад входа во все 3 блока
                    gh = w['W_hh'] @ h + w['b_hh']               # вклад состояния (со смещением b_hh)
                    i_r, i_z, i_n = np.split(gi, 3)              # порядок PyTorch: r, z, n
                    h_r, h_z, h_n = np.split(gh, 3)
                    r = sigmoid(i_r + h_r)
                    z = sigmoid(i_z + h_z)
                    n = np.tanh(i_n + r * h_n)                   # reset gate «гасит» вклад прошлого состояния
                    h = (1 - z) * n + z * h                      # z ≈ 1 → состояние почти не меняется
                    rows.append({'t': t, **{f'h{k + 1}': h[k] for k in range(3)}, **{f'z{k + 1}': z[k] for k in range(3)}})
                    print(f"t={t}: r = {r.round(3)}, z = {z.round(3)}")

                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** GRU — облегчённый LSTM: одно состояние h вместо пары (h, c) и два вентиля вместо трёх. Update gate z
            решает, какую долю старого состояния сохранить (интерполяция между hₜ₋₁ и кандидатом n), reset gate r —
            насколько прошлое участвует в построении кандидата. Сложение вместо перемножения даёт градиенту прямой путь
            назад по времени.

            **Типичные ошибки:**
            - Порядок блоков z, r, n (как пишут в некоторых статьях) — ответ неверен с первого шага.
            - r ⊙ (W_hn·h) + b_hn вместо r ⊙ (W_hn·h + b_hn) — небольшое, но систематическое расхождение.
            - hₜ = z ⊙ n + (1 − z) ⊙ hₜ₋₁ (формула из лекции) с весами PyTorch — роли z перевёрнуты.
            - Начали с h₀ = 0, а не с `h0` из файла.

            **Что вынести:** 3 блока весов = 3H строк; в PyTorch z — доля старого состояния, а reset применяется к
            W_hn·h + b_hn. → шпаргалка «GRU».
        """,
    },
    {
        "name": "rnn_basics",
        "title": "Простая RNN: прямой проход и BPTT-градиенты",
        "data": data_rnn,
        "task": """
            Простая (Elman) RNN: вход D = 2, скрытое состояние H = 3, скалярный выход на каждом шаге.
            Последовательность из 4 шагов и цели — в `input.csv` (`t, x1, x2, target`), веса — в `weights.json`:
            `W_xh` (3 × 2), `W_hh` (3 × 3), `b_h` (3), `W_hy` (1 × 3), `b_y` (1).

            Прямой проход (h₀ = 0):

            - hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ + b_h)
            - yₜ = W_hy·hₜ + b_y
            - L = ½ · Σₜ (yₜ − targetₜ)² — сумма по всем 4 шагам.

            Вычислите L и градиенты L по **всем** параметрам методом backpropagation through time.

            ## Подсказки

            - Веса общие для всех шагов, поэтому градиент по каждому весу — сумма вкладов от всех шагов.
            - Идите назад от t = 4 к t = 1 и держите «градиент, пришедший из будущего» dh_next (в конце он нулевой):
              dhₜ = W_hyᵀ·(yₜ − targetₜ) + dh_next; da = dhₜ ⊙ (1 − hₜ²); dh_next = W_hhᵀ·da.
            - Накопления: dW_hy += (yₜ − targetₜ)·hₜᵀ, db_y += (yₜ − targetₜ), dW_xh += da·xₜᵀ, dW_hh += da·hₜ₋₁ᵀ,
              db_h += da. Проверьте результат конечными разностями: (L(w + ε) − L(w − ε)) / 2ε.

            ## Ожидаемый формат выходного файла

            CSV с колонками `param, i, j, value` — 23 строки: сначала `loss` (i = j = 0), затем градиенты
            `W_xh`, `W_hh`, `b_h`, `W_hy`, `b_y` поэлементно по строкам (i — строка, j — столбец; для векторов `b_h`
            и `b_y` — i = индекс, j = 0); значения округлить до 4 знаков.
        """,
        "solution": """
            import json

            ORDER = ['W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']


            def forward(w, X, target):
                hs = [np.zeros(w['W_hh'].shape[0])]
                ys = []
                for x in X:
                    hs.append(np.tanh(w['W_hh'] @ hs[-1] + w['W_xh'] @ x + w['b_h']))   # одни веса на всех шагах
                    ys.append(float((w['W_hy'] @ hs[-1] + w['b_y'])[0]))
                ys = np.array(ys)
                return 0.5 * np.sum((ys - target) ** 2), hs, ys


            def main():
                df = pd.read_csv('input.csv')
                X, target = df[['x1', 'x2']].to_numpy(), df['target'].to_numpy()
                w = {k: np.array(v, dtype=float) for k, v in json.load(open('weights.json', encoding='utf-8')).items()}

                loss, hs, ys = forward(w, X, target)
                g = {k: np.zeros_like(v) for k, v in w.items()}
                dh_next = np.zeros_like(hs[0])
                for t in reversed(range(len(X))):              # BPTT: от последнего шага к первому
                    dy = ys[t] - target[t]
                    g['W_hy'] += dy * hs[t + 1][None, :]
                    g['b_y'] += dy
                    dh = w['W_hy'][0] * dy + dh_next            # градиент от выхода + «из будущего»
                    da = dh * (1 - hs[t + 1] ** 2)              # через tanh
                    g['W_xh'] += np.outer(da, X[t])
                    g['W_hh'] += np.outer(da, hs[t])
                    g['b_h'] += da
                    dh_next = w['W_hh'].T @ da

                # Проверка конечными разностями
                eps = 1e-6
                for k in ORDER:
                    for idx in np.ndindex(w[k].shape):
                        wp = {n: v.copy() for n, v in w.items()}; wp[k][idx] += eps
                        wm = {n: v.copy() for n, v in w.items()}; wm[k][idx] -= eps
                        num = (forward(wp, X, target)[0] - forward(wm, X, target)[0]) / (2 * eps)
                        assert abs(num - g[k][idx]) < 1e-6, (k, idx)
                print(f"loss = {loss:.4f}; градиенты совпали с численными")

                rows = [{'param': 'loss', 'i': 0, 'j': 0, 'value': loss}]
                for k in ORDER:
                    G = g[k].reshape(len(g[k]), -1) if g[k].ndim == 1 else g[k]
                    for i in range(G.shape[0]):
                        for j in range(G.shape[1]):
                            rows.append({'param': k, 'i': i, 'j': j, 'value': G[i, j]})
                return pd.DataFrame(rows).round(4)
        """,
        "review": """
            **Идея.** Развёрнутая во времени RNN — глубокая сеть с общими весами. BPTT — обычный backprop по этой
            развёртке: градиент состояния на шаге t складывается из вклада выхода yₜ и вклада, пришедшего из будущего
            через W_hhᵀ и производную tanh. Градиенты весов суммируются по шагам. Каждый шаг назад умножает сигнал на
            W_hhᵀ·diag(1 − h²) — отсюда затухание и взрыв градиента на длинных последовательностях.

            **Типичные ошибки:**
            - Взяли градиент только от последнего шага или забыли dh_next — веса W_hh, W_xh получают слишком мало.
            - Производная tanh как 1 − tanh(a) или по hₜ₋₁ вместо hₜ.
            - dW_hh += da·hₜᵀ вместо hₜ₋₁ᵀ — сдвиг на шаг.
            - Транспонированный градиент W_xh (2 × 3 вместо 3 × 2) — порядок строк в ответе не тот.

            **Что вынести:** BPTT = backprop по развёртке + сумма по шагам; проверяйте ручные градиенты конечными
            разностями. → шпаргалка «Основы RNN».
        """,
    },
]


def main():
    print("Генерация упражнений к лекциям DL / CV:")
    for seed, exercise in enumerate(EXERCISE_LIST, start=741):
        build(exercise, seed)


if __name__ == "__main__":
    main()
