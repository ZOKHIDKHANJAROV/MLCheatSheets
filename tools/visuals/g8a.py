# -*- coding: utf-8 -*-
"""Рисунки группы 8a: ai_agents_basics, tool_use_function_calling, react_agents (SVG-схемы)."""

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


# =============================================================== 1. ai_agents_basics
AGENT_LOOP = svg(
    "agbasics", 340, 300, "Цикл ИИ-агента: восприятие, рассуждение, действие, наблюдение",
    B(20, 14, 120, 44, "c1", ["1. Восприятие", "цель, история"]),
    B(200, 14, 120, 44, "c5", ["2. Рассуждение", "LLM решает"]),
    B(200, 170, 120, 44, "c3", ["3. Действие", "вызов инструмента"]),
    B(20, 170, 120, 44, "c2", ["4. Наблюдение", "результат в контекст"]),
    A(140, 36, 198, 36),
    A(260, 58, 260, 168),
    A(198, 192, 142, 192),
    A(80, 168, 80, 60),
    B(95, 92, 150, 50, "c6", ["Память + инструменты", "контекст, внешний мир"]),
    A(260, 142, 245, 122, dash=True),
    T(170, 248, "Цикл повторяется, пока не выдан финальный ответ", "mut", fs=11),
    B(60, 258, 220, 30, "c4", "Лимит шагов / бюджета → стоп"),
)

# =============================================================== 2. tool_use_function_calling
_XM, _XA, _XT = 60, 190, 320
_lifelines = [
    B(10, 6, 100, 30, "c5", "Модель"),
    B(140, 6, 100, 30, "c1", "Приложение"),
    B(270, 6, 100, 30, "c3", "Инструмент"),
    L(_XM, 38, _XM, 262, "ln s6", dash=True),
    L(_XA, 38, _XA, 262, "ln s6", dash=True),
    L(_XT, 38, _XT, 262, "ln s6", dash=True),
]
_steps = [
    (66, _XA, _XM, "messages + схемы", False),
    (102, _XM, _XA, "tool_call: имя, args", False),
    (138, _XA, _XT, "валидация, вызов", False),
    (174, _XT, _XA, "результат / ошибка", True),
    (210, _XA, _XM, "tool_result (id)", False),
    (246, _XM, _XA, "финальный текст", False),
]
TOOL_CYCLE = svg(
    "toolcall", 380, 290, "Последовательность вызова инструмента",
    *_lifelines,
    *[A(x1, y, x2, y, dash=d) + T((x1 + x2) / 2, y - 5, txt, "mut", fs=10) for y, x1, x2, txt, d in _steps],
    T(190, 282, "шаги 2–5 повторяются до финального текста или max_steps", "mut", fs=10),
)

# =============================================================== 3. react_agents
REACT_LOOP = svg(
    "reactloop", 340, 270, "Цикл ReAct: Thought, Action, Observation",
    B(10, 8, 90, 30, "c2", "Final Answer"),
    A(55, 70, 55, 40),
    B(10, 70, 90, 42, "c1", ["Thought", "рассуждение"]),
    B(125, 70, 90, 42, "c3", ["Action", "tool[аргумент]"]),
    B(240, 70, 90, 42, "c2", ["Observation", "результат"]),
    A(100, 91, 123, 91),
    A(215, 91, 238, 91),
    PA("M285,112 L285,144 L55,144 L55,114"),
    T(55, 126, "модель", "mut", fs=10),
    T(170, 126, "модель → программа", "mut", fs=10),
    T(285, 126, "только среда", "mut", fs=10),
    T(170, 160, "повторять, пока не будет ответа", "mut", fs=11),
    B(10, 176, 320, 76, "c4",
      ["Остановка и защита", "лимит шагов · бюджет токенов", "повтор того же Action → зацикливание"]),
    T(62, 58, "найден ответ", "mut", "start", 10),
)

VISUALS = {
    "ai_agents_basics": [
        {"title": "Цикл ИИ-агента", "svg": AGENT_LOOP,
         "caption": "Агент повторяет цикл «восприятие → рассуждение → действие → наблюдение»: LLM выбирает шаг, "
                    "инструмент выполняет его в мире, результат возвращается в контекст. Лимит шагов не даёт циклу быть бесконечным."},
    ],
    "tool_use_function_calling": [
        {"title": "Вызов инструмента шаг за шагом", "svg": TOOL_CYCLE,
         "caption": "Модель только просит вызвать функцию (tool_call), а приложение проверяет аргументы, выполняет её и "
                    "возвращает tool_result с тем же id. Цикл повторяется, пока модель не ответит текстом."},
    ],
    "react_agents": [
        {"title": "Цикл ReAct", "svg": REACT_LOOP,
         "caption": "Модель пишет Thought и Action, программа выполняет действие и добавляет Observation. "
                    "Цикл заканчивается финальным ответом либо ограничителями: лимит шагов, бюджет, повторы действий."},
    ],
}
