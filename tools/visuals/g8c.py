# -*- coding: utf-8 -*-
"""Рисунки группы 8c: ИИ-агенты — agent_planning, multi_agent_systems, mcp_protocol (SVG-схемы)."""

import math
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


def C(cx, cy, r, cls, s=None, tcls=""):
    out = f'<circle class="{cls}" cx="{cx}" cy="{cy}" r="{r}"/>'
    if s:
        out += T(cx, cy + 4, s, tcls)
    return out


def edge(p, q, rp, rq, cls="ln", both=False):
    """Стрелка между центрами двух кругов, укороченная до их границ."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    a = (round(p[0] + ux * (rp + 1), 1), round(p[1] + uy * (rp + 1), 1))
    b = (round(q[0] - ux * (rq + 3), 1), round(q[1] - uy * (rq + 3), 1))
    return A(a[0], a[1], b[0], b[1], cls, both=both)


# =============================================================== 1. agent_planning
_N = 64   # ширина узла DAG
_H = 36
PLANNING = svg(
    "agplan", 345, 300, "DAG подзадач с критическим путём и бюджетом токенов",
    T(4, 14, "План: подзадачи и зависимости (длительность в шагах)", "", "start", 12),
    # рёбра
    A(68, 123, 94, 53, "ln s6"), A(68, 123, 94, 123, "ln s3"), A(68, 123, 94, 193, "ln s6"),
    A(158, 53, 184, 88, "ln s6"), A(158, 123, 184, 88, "ln s3"),
    A(248, 88, 274, 121, "ln s3"), A(158, 193, 274, 127, "ln s6"),
    # узлы
    B(4, 105, _N, _H, "c3", ["источники", "1 шаг"], fs=11),
    B(94, 35, _N, _H, "c1", ["чтение A", "2 шага"], fs=11),
    B(94, 105, _N, _H, "c3", ["чтение B", "4 шага"], fs=11),
    B(94, 175, _N, _H, "c1", ["введение", "2 шага"], fs=11),
    B(184, 70, _N, _H, "c3", ["сверка", "2 шага"], fs=11),
    B(274, 105, _N, _H, "c3", ["отчёт", "3 шага"], fs=11),
    # легенда
    B(4, 232, 14, 10, "c3", "", rx=2),
    T(24, 241, "критический путь: slack = 0, T = 1+4+2+3 = 10", "", "start", 11),
    B(4, 248, 14, 10, "c1", "", rx=2),
    T(24, 257, "есть запас: A и введение можно сдвинуть или вести параллельно", "", "start", 10.5),
    # бюджет
    T(4, 280, "Бюджет токенов:", "mut", "start", 11),
    f'<rect class="c6" x="104" y="270" width="236" height="12" rx="3"/>',
    f'<rect class="c2" x="104" y="270" width="170" height="12" rx="3"/>',
    T(189, 280, "потрачено", "", "middle", 10),
    T(307, 280, "остаток", "mut", "middle", 10),
)

# =============================================================== 2. multi_agent_systems
_P = (55, 50)       # оркестратор
_W = [(18, 108), (55, 120), (92, 108)]
_CH = [(170, 40), (170, 80), (170, 120), (170, 160)]
_M = [(250, 52), (320, 52), (320, 138), (250, 138)]
MULTI = svg(
    "agmulti", 340, 250, "Три схемы координации четырёх агентов и число сообщений за раунд",
    # supervisor
    *[edge(_P, w, 13, 10, both=True) for w in _W],
    C(_P[0], _P[1], 13, "c5", "О"),
    *[C(x, y, 10, "c1", str(i)) for i, (x, y) in enumerate(_W, start=1)],
    T(55, 192, "Supervisor", "", fs=12),
    T(55, 209, "2k = 6", "t3", fs=12),
    # pipeline
    *[edge(_CH[i], _CH[i + 1], 11, 11) for i in range(3)],
    *[C(x, y, 11, "c1", str(i)) for i, (x, y) in enumerate(_CH, start=1)],
    T(170, 192, "Pipeline", "", fs=12),
    T(170, 209, "n − 1 = 3", "t3", fs=12),
    # все-со-всеми
    *[edge(_M[i], _M[j], 11, 11, "ln s4", both=True) for i in range(4) for j in range(i + 1, 4)],
    *[C(x, y, 11, "c4", str(i)) for i, (x, y) in enumerate(_M, start=1)],
    T(285, 192, "Все-со-всеми", "", fs=12),
    T(285, 209, "n(n − 1) = 12", "t4", fs=12),
    T(170, 238, "n = 4 агента; стрелка — сообщение (у mesh в обе стороны)", "mut", fs=10),
)

# =============================================================== 3. mcp_protocol: N×M
_APPS = ["Чат", "IDE", "Агент"]
_SRV = ["Файлы", "База", "Трекер"]
_ay = [24, 54, 84]
_by = [190, 220, 250]
MCP_NM = svg(
    "mcpnm", 340, 300, "Без стандарта нужно N×M коннекторов, с MCP — N+M",
    T(170, 14, "Без стандарта: N × M = 9 коннекторов", "t4", fs=12),
    *[L(78, ya + 11, 262, yb + 11, "ln s6") for ya in _ay for yb in _ay],
    *[B(8, y, 70, 22, "c1", n) for y, n in zip(_ay, _APPS)],
    *[B(262, y, 70, 22, "c3", n) for y, n in zip(_ay, _SRV)],
    L(8, 140, 332, 140, "ln s6", dash=True),
    T(170, 170, "С MCP: N + M = 6 реализаций", "t2", fs=12),
    *[A(78, y + 11, 138, 220, "ln s1") for y in _by],
    *[A(202, 220, 262, y + 11, "ln s3") for y in _by],
    *[B(8, y, 70, 22, "c1", n) for y, n in zip(_by, _APPS)],
    B(138, 190, 64, 60, "c5", ["MCP", "клиент–", "сервер"], fs=11),
    *[B(262, y, 70, 22, "c3", n) for y, n in zip(_by, _SRV)],
    T(170, 292, "каждая сторона реализует протокол один раз", "mut", fs=10.5),
)

# =============================================================== 3b. mcp_protocol: архитектура
MCP_ARCH = svg(
    "mcparch", 340, 310, "Архитектура MCP: host, клиенты, серверы и три типа возможностей",
    B(6, 8, 160, 190, "c6", [""], rx=8),
    T(86, 26, "Host (приложение)", "", fs=12),
    B(20, 36, 132, 28, "c1", "LLM"),
    L(26, 64, 26, 168, "ln"),
    A(26, 112, 40, 112), A(26, 168, 40, 168),
    B(40, 96, 112, 32, "c5", "MCP-клиент 1"),
    B(40, 152, 112, 32, "c5", "MCP-клиент 2"),
    A(152, 112, 202, 112, both=True),
    A(152, 168, 202, 168, both=True),
    T(178, 103, "JSON-RPC", "mut", fs=9),
    B(202, 92, 132, 40, "c2", ["Сервер A", "stdio · локально"], fs=11),
    B(202, 148, 132, 40, "c2", ["Сервер B", "HTTP · удалённо"], fs=11),
    T(170, 226, "Сервер открывает три вида возможностей", "", fs=12),
    B(6, 238, 104, 52, "c3", ["tools", "действия", "решает модель"], fs=10.5),
    B(118, 238, 104, 52, "c1", ["resources", "данные", "решает приложение"], fs=10.5),
    B(230, 238, 104, 52, "c4", ["prompts", "шаблоны", "выбирает человек"], fs=10.5),
    T(170, 304, "решения о вызовах принимает host, а не сервер", "mut", fs=10.5),
)

VISUALS = {
    "agent_planning": [
        {"title": "DAG подзадач и бюджет", "svg": PLANNING,
         "caption": "План агента — граф зависимостей. Длина критического пути (здесь 10 шагов) задаёт минимальное время; "
                    "подзадачи с запасом можно вести параллельно. Бюджет токенов ограничивает весь план целиком."},
    ],
    "multi_agent_systems": [
        {"title": "Схемы координации", "svg": MULTI,
         "caption": "Число сообщений за раунд: у supervisor оно растёт линейно (2k), у конвейера тоже (n − 1), у схемы "
                    "«все-со-всеми» квадратично — n(n − 1). Каждое сообщение — токены, которые кто-то читает и оплачивает."},
    ],
    "mcp_protocol": [
        {"title": "Интеграции: до и после", "svg": MCP_NM,
         "caption": "Без стандарта каждое приложение пишет коннектор к каждой системе: N × M. С общим протоколом "
                    "приложение реализует клиент, а система — сервер, и интеграций становится N + M."},
        {"title": "Архитектура MCP", "svg": MCP_ARCH,
         "caption": "Host держит по одному MCP-клиенту на каждый сервер. Клиент и сервер обмениваются сообщениями JSON-RPC "
                    "через stdio или HTTP; сервер предлагает tools, resources и prompts."},
    ],
}
