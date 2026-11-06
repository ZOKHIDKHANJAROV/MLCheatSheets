# -*- coding: utf-8 -*-
"""Рисунки группы 8b (ИИ-агенты): rag, vector_search, agent_memory (SVG-схемы и график matplotlib)."""

import numpy as np
from xml.sax.saxutils import escape as _esc


# ---------------------------------------------------------------- SVG-помощники
def svg(pre, w, h, label, *parts):
    defs = (f'<defs><marker id="{pre}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/>'
            f'</marker></defs>')
    body = "".join(parts).replace("#@a", f"#{pre}-a")
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{_esc(label)}">{defs}{body}</svg>')


def T(x, y, s, cls="", a="middle", fs=None):
    at = f' class="{cls}"' if cls else ""
    if a:
        at += f' text-anchor="{a}"'
    if fs:
        at += f' style="font-size:{fs}px"'
    return f'<text x="{x}" y="{y}"{at}>{_esc(s)}</text>'


def B(x, y, w, h, c, lines, fs=None, rx=6, cls=""):
    if isinstance(lines, str):
        lines = [lines]
    out = f'<rect class="{c}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * 7.5 + 4
    for i, s in enumerate(lines):
        out += T(x + w / 2, round(y0 + 15 * i, 1), s, cls, fs=fs)
    return out


def A(x1, y1, x2, y2, cls="ln", dash=False, both=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    st = ' marker-start="url(#@a)"' if both else ""
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{d}{st} marker-end="url(#@a)"/>'


def L(x1, y1, x2, y2, cls="ln", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{d}/>'


def PA(d, cls="ln", arrow=True, dash=False):
    m = ' marker-end="url(#@a)"' if arrow else ""
    ds = ' stroke-dasharray="4 3"' if dash else ""
    return f'<path class="{cls}" d="{d}" fill="none"{ds}{m}/>'


def C(cx, cy, r, cls, s=None, tcls=""):
    out = f'<circle class="{cls}" cx="{cx}" cy="{cy}" r="{r}"/>'
    if s:
        out += T(cx, cy + 4, s, tcls)
    return out


def R(x, y, w, h, cls, rx=0):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'


def poly(pts, cls="ln"):
    return f'<polyline class="{cls}" fill="none" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'


# ---------------------------------------------------------------- RAG
RAG_PIPELINE = svg(
    "rag-p", 340, 232, "Пайплайн RAG: индексация и запрос",
    T(10, 14, "ИНДЕКСАЦИЯ (офлайн)", "mut", a="start", fs=10),
    B(10, 22, 68, 40, "c6", ["Документы"], fs=10),
    B(92, 22, 68, 40, "c1", ["Чанки"], fs=11),
    B(174, 22, 68, 40, "c3", ["Эмбеддинги"], fs=10),
    B(256, 22, 68, 40, "c5", ["Вектор.", "индекс"], fs=10),
    A(78, 42, 92, 42), A(160, 42, 174, 42), A(242, 42, 256, 42),
    PA("M290,62 V84 H126 V108", dash=True),
    T(206, 80, "индекс читается при запросе", "mut", fs=9),
    T(10, 100, "ЗАПРОС (онлайн)", "mut", a="start", fs=10),
    B(10, 108, 68, 40, "c6", ["Вопрос"], fs=11),
    B(92, 108, 68, 40, "c1", ["Поиск", "top-k"], fs=10),
    B(174, 108, 68, 40, "c3", ["Реранкинг"], fs=10),
    B(256, 108, 68, 40, "c5", ["Контекст", "+ вопрос"], fs=10),
    A(78, 128, 92, 128), A(160, 128, 174, 128), A(242, 128, 256, 128),
    B(256, 176, 68, 40, "c4", ["LLM"], fs=12),
    A(290, 148, 290, 176),
    B(174, 176, 68, 40, "c2", ["Ответ", "+ источники"], fs=10),
    A(256, 196, 242, 196),
    T(10, 192, "в prompt идут только", "mut", a="start", fs=10),
    T(10, 207, "3–5 лучших чанков", "mut", a="start", fs=10),
)


def _chunking():
    parts = [T(10, 14, "Документ из n токенов", "mut", a="start", fs=10), R(10, 20, 320, 20, "c6", 4)]
    for i in range(4):
        x, y = 10 + 73 * i, 52 + 34 * i
        parts.append(B(x, y, 100, 24, "c1" if i % 2 == 0 else "c3", [f"чанк {i + 1}"], fs=10))
    for i in range(3):
        y1, y2 = 52 + 34 * i, 52 + 34 * (i + 1) + 24
        parts.append(L(83 + 73 * i, y1, 83 + 73 * i, y2, "s4", dash=True))
        parts.append(L(110 + 73 * i, y1, 110 + 73 * i, y2, "s4", dash=True))
        parts.append(T(96.5 + 73 * i, y1 + 32, "overlap", "t4", fs=8))
    parts += [T(10, 198, "size = 100, overlap = 27, шаг = size − overlap = 73", "mut", a="start", fs=10),
              T(10, 213, "перекрытие не даёт разорвать мысль на границе", "mut", a="start", fs=10)]
    return svg("rag-c", 340, 222, "Чанкинг с перекрытием", *parts)


RAG_CHUNKS = _chunking()


def _lost_in_middle():
    pts = [(50, 40), (80, 75), (110, 100), (150, 115), (190, 118), (230, 105), (260, 82), (290, 55), (310, 40)]
    parts = [L(40, 20, 40, 160), L(40, 160, 322, 160),
             T(46, 14, "доля верных ответов", "mut", a="start", fs=10),
             poly(pts, "ln s1"),
             T(50, 176, "начало", "mut", fs=10), T(180, 176, "середина", "mut", fs=10), T(306, 176, "конец", "mut", fs=10),
             T(180, 140, "хуже всего — середина", "t4", fs=10),
             T(180, 196, "позиция нужного чанка в контексте", "mut", fs=10)]
    return svg("rag-l", 340, 204, "Эффект lost in the middle (схематично)", *parts)


RAG_LOST = _lost_in_middle()


# ---------------------------------------------------------------- vector_search
def _ivf():
    centers = [(70, 70), (250, 60), (80, 170), (240, 170)]
    probed = {0, 2}
    offs = [(-20, -10), (10, -22), (24, 6), (-8, 18), (-26, 14), (14, 24), (0, -2)]
    parts = []
    for i, (cx, cy) in enumerate(centers):
        parts.append(f'<circle class="{"c1" if i in probed else "s6"}" cx="{cx}" cy="{cy}" r="44"'
                     + ('' if i in probed else ' fill="none" stroke-dasharray="4 3"') + '/>')
    for i, (cx, cy) in enumerate(centers):
        for dx, dy in offs:
            parts.append(C(cx + dx * (1.1 if i % 2 else 0.9), cy + dy, 3, "t1" if i in probed else "t6"))
        parts.append(C(cx, cy, 5, "c5"))
    for i in probed:
        parts.append(L(130, 120, centers[i][0], centers[i][1], "s1", dash=True))
    parts += [C(130, 120, 6, "c4"), T(142, 116, "q", "t4", a="start", fs=13),
              T(10, 232, "nprobe = 2: смотрим 2 ближайшие ячейки из 4,", "mut", a="start", fs=10),
              T(10, 246, "остальные пропускаем. ● центроид, q — запрос", "mut", a="start", fs=10)]
    return svg("vs-i", 340, 254, "IVF: ячейки и nprobe", *parts)


VS_IVF = _ivf()

VS_GEOMETRY = svg(
    "vs-g", 340, 214, "Косинус, скалярное произведение и L2",
    A(40, 170, 200, 60, "ln s1"), A(40, 170, 290, 130, "ln s2"),
    L(200, 60, 290, 130, "s3", dash=True),
    PA("M89.4,136 A60,60 0 0 1 99.2,160.5", "s4", arrow=False),
    T(106, 150, "θ", "t4", a="start", fs=13),
    T(204, 52, "a", "t1", a="start", fs=13), T(296, 136, "b", "t2", a="start", fs=13),
    T(262, 88, "L2 = ‖a − b‖", "t3", a="start", fs=11),
    T(34, 186, "O", "mut", fs=10),
    T(10, 198, "cos θ — только угол; a·b = ‖a‖·‖b‖·cos θ", "mut", a="start", fs=10),
    T(10, 211, "после нормализации ‖a − b‖² = 2 − 2·cos θ", "mut", a="start", fs=10),
)


def recall_chart(fig):
    """recall@10 приближённого IVF-поиска против доли просмотренных векторов (nprobe меняется)."""
    rng = np.random.default_rng(0)
    n, d, n_cent, nlist = 8000, 32, 30, 40
    centers = rng.normal(size=(n_cent, d)) * 0.7
    X = centers[rng.integers(0, n_cent, n)] + rng.normal(size=(n, d))
    Q = centers[rng.integers(0, n_cent, 100)] + rng.normal(size=(100, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)

    cent = X[rng.choice(n, nlist, replace=False)].copy()
    for _ in range(10):
        cell = np.argmax(X @ cent.T - 0.5 * (cent ** 2).sum(1), axis=1)
        for j in range(nlist):
            if (cell == j).any():
                cent[j] = X[cell == j].mean(0)
    cell = np.argmax(X @ cent.T - 0.5 * (cent ** 2).sum(1), axis=1)
    exact = np.argsort(-(Q @ X.T), axis=1)[:, :10]

    probes, rec, frac = [1, 2, 3, 5, 8, 12, 20, 40], [], []
    for p in probes:
        r, f = [], []
        for qi, q in enumerate(Q):
            near = np.argsort(-(cent @ q - 0.5 * (cent ** 2).sum(1)))[:p]
            cand = np.flatnonzero(np.isin(cell, near))
            top = cand[np.argsort(-(X[cand] @ q))[:10]]
            r.append(len(set(top) & set(exact[qi])) / 10)
            f.append(len(cand) / n)
        rec.append(np.mean(r))
        frac.append(100 * np.mean(f))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(frac, rec, "o-", lw=2)
    for p, x, y in zip(probes, frac, rec):
        if p in (1, 3, 8, 40):
            ax.annotate(f"nprobe={p}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=7)
    ax.axhline(0.95, color="#8a96a3", ls="--", lw=1)
    ax.set_xlabel("доля просмотренных векторов, %")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0, 1.05)
    ax.set_title("IVF: больше ячеек — выше recall, ниже скорость", fontsize=9)
    fig.set_size_inches(4.2, 3.2)
    fig.tight_layout()


# ---------------------------------------------------------------- agent_memory
MEM_ARCH = svg(
    "am-a", 340, 246, "Кратковременная и долговременная память агента",
    T(85, 14, "Кратковременная память", "mut", fs=10),
    R(10, 20, 150, 200, "s1", 8),
    B(20, 30, 130, 36, "c6", ["Системная инструкция"], fs=10),
    B(20, 72, 130, 36, "c5", ["Сводка истории"], fs=10),
    B(20, 114, 130, 36, "c2", ["Недавние сообщения"], fs=10),
    B(20, 156, 130, 36, "c3", ["Найденные воспоминания"], fs=10),
    T(85, 211, "контекстное окно", "mut", fs=9),
    T(270, 14, "Долговременная память", "mut", fs=10),
    B(210, 36, 120, 50, "c3", ["Эпизоды", "(вектор. хранилище)"], fs=10),
    B(210, 100, 120, 50, "c5", ["Факты", "(база ключ — значение)"], fs=10),
    B(210, 164, 120, 50, "c4", ["Правила и навыки", "(процедурная)"], fs=10),
    L(190, 61, 190, 189), L(190, 61, 210, 61), L(190, 125, 210, 125), L(190, 189, 210, 189),
    A(160, 110, 190, 110), T(175, 104, "запись", "mut", fs=9),
    A(190, 140, 160, 140), T(175, 154, "чтение", "mut", fs=9),
    T(170, 238, "в окно — нужное сейчас, в хранилище — важное надолго", "mut", fs=10),
)

MEM_TYPES = svg(
    "am-t", 340, 252, "Типы долговременной памяти и забывание",
    B(10, 10, 320, 56, "c3", ["Эпизодическая: «что было»", "12 мая запуск упал — не хватило памяти"], fs=11),
    B(10, 76, 320, 56, "c5", ["Семантическая: «что верно»", "пользователь пишет на Python, пояс UTC+3"], fs=11),
    B(10, 142, 320, 56, "c4", ["Процедурная: «как делать»", "для отчёта: выгрузка → проверка пропусков"], fs=11),
    T(170, 224, "забывание: удаляем записи с минимальной ценностью", "mut", fs=10),
    T(170, 240, "ценность = a · важность + b · свежесть", "mut", fs=10),
)


VISUALS = {
    "rag": [
        {"title": "Пайплайн RAG", "svg": RAG_PIPELINE,
         "caption": "Офлайн документы режутся на чанки, кодируются эмбеддингами и попадают в индекс. На запросе вопрос ищет "
                    "кандидатов, реранкер выбирает лучшие, из них собирается контекст — и LLM отвечает со ссылками на источники."},
        {"title": "Чанкинг с перекрытием", "svg": RAG_CHUNKS,
         "caption": "Окно размера size сдвигается с шагом size − overlap. Перекрытие повторяет границу соседних чанков, "
                    "поэтому мысль, оказавшаяся на стыке, целиком попадает хотя бы в один из них."},
        {"title": "«Lost in the middle»", "svg": RAG_LOST,
         "caption": "Схема: модели обычно лучше используют информацию из начала и конца длинного контекста, чем из середины. "
                    "Поэтому самые релевантные чанки ставят по краям, а k не раздувают. Величина эффекта зависит от модели."},
    ],
    "vector_search": [
        {"title": "IVF: поиск по ячейкам", "svg": VS_IVF,
         "caption": "База разбита на ячейки вокруг центроидов. Запрос q сравнивается с центроидами, и перебираются только "
                    "векторы в nprobe ближайших ячейках. Сосед в соседней, непросмотренной ячейке будет потерян."},
        {"title": "Косинус, dot и L2", "svg": VS_GEOMETRY,
         "caption": "Косинус зависит только от угла θ, скалярное произведение — ещё и от длин векторов, L2 — расстояние между "
                    "концами. Для векторов единичной длины порядок соседей у всех трёх мер одинаков."},
        {"title": "Recall против скорости", "chart": recall_chart,
         "caption": "Игрушечный пример на искусственных кластерах (numpy). Чем больше ячеек просматривается (nprobe), тем выше "
                    "recall@10 относительно точного перебора и тем больше векторов приходится сравнить. Выбирают точку под целевой recall."},
    ],
    "agent_memory": [
        {"title": "Кратковременная и долговременная память", "svg": MEM_ARCH,
         "caption": "В контекстное окно попадают инструкция, сводка старой истории, недавние сообщения и найденные воспоминания. "
                    "Важное записывается в долговременное хранилище и поднимается обратно по запросу."},
        {"title": "Типы памяти", "svg": MEM_TYPES,
         "caption": "Эпизодическая память хранит события, семантическая — факты и предпочтения, процедурная — способы действовать. "
                    "Чтобы хранилище не зарастало шумом, записи с низкой ценностью (важность и свежесть) вытесняются."},
    ],
}
