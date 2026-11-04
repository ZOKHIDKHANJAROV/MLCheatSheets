# -*- coding: utf-8 -*-
"""Рисунки для группы g3 (glm … label_propagation): SVG-схемы и графики matplotlib."""

import numpy as np

PAL = ["#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"]
BLUE, GREEN, ORANGE, RED, VIOLET, GRAY = PAL


# ---------------------------------------------------------------- SVG-помощники
def _e(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _n(v):
    return f"{round(float(v), 1):g}"


def svg(p, w, h, label, parts):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_e(label)}">'
            f'<defs><marker id="{p}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            + "".join(parts) + "</svg>")


def T(x, y, s, cls="", fs=None, a="middle", b=False, rot=None):
    st = []
    if fs:
        st.append(f"font-size:{fs}px")
    if b:
        st.append("font-weight:bold")
    at = f' class="{cls}"' if cls else ""
    if st:
        at += f' style="{";".join(st)}"'
    if a != "start":
        at += f' text-anchor="{a}"'
    if rot is not None:
        at += f' transform="rotate({rot} {_n(x)} {_n(y)})"'
    return f'<text x="{_n(x)}" y="{_n(y)}"{at}>{_e(s)}</text>'


def R(x, y, w, h, cls="c6", rx=6, sw=None, dash=False):
    st = []
    if sw:
        st.append(f"stroke-width:{sw}")
    at = f' style="{";".join(st)}"' if st else ""
    if dash:
        at += ' stroke-dasharray="4 3"'
    return f'<rect class="{cls}" x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{rx}"{at}/>'


def B(x, y, w, h, cls, lines, fs=12, rx=6, sw=None, dash=False):
    """Прямоугольник с центрированными строками; строка — str или (str, класс, размер)."""
    if isinstance(lines, (str, tuple)):
        lines = [lines]
    out = [R(x, y, w, h, cls, rx, sw, dash)]
    specs = [(s, "", fs) if isinstance(s, str) else s for s in lines]
    lh = [f + 3 for _, _, f in specs]
    total = sum(lh) - 3
    yy = y + h / 2 - total / 2
    for (s, c, f), l in zip(specs, lh):
        out.append(T(x + w / 2, yy + f * 0.82, s, c, f if f != 12 else None))
        yy += l
    return "".join(out)


def L(x1, y1, x2, y2, p=None, cls="ln", dash=False, start=False, sw=None):
    at = ""
    if p:
        at += f' marker-end="url(#{p}-a)"'
        if start:
            at += f' marker-start="url(#{p}-a)"'
    if dash:
        at += ' stroke-dasharray="4 3"'
    if sw:
        at += f' style="stroke-width:{sw}"'
    return f'<line class="{cls}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}"{at}/>'


def PA(d, p=None, cls="ln", dash=False, sw=None):
    at = f' marker-end="url(#{p}-a)"' if p else ""
    if dash:
        at += ' stroke-dasharray="4 3"'
    if sw:
        at += f' style="stroke-width:{sw}"'
    return f'<path class="{cls}" d="{d}"{at}/>'


def C(cx, cy, r, cls="c6", sw=None, dash=False):
    at = f' style="stroke-width:{sw}"' if sw else ""
    if dash:
        at += ' stroke-dasharray="3 2"'
    return f'<circle class="{cls}" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}"{at}/>'


def N(cx, cy, r, cls, label="", fs=12, tcls="", sw=None):
    return C(cx, cy, r, cls, sw) + (T(cx, cy + fs * 0.36, label, tcls, fs if fs != 12 else None) if label else "")


def E(x1, y1, x2, y2, r1=14, r2=14, p=None, cls="ln", dash=False, sw=None):
    """Ребро между центрами узлов, укороченное на радиусы."""
    dx, dy = x2 - x1, y2 - y1
    d = (dx * dx + dy * dy) ** 0.5
    r2 = r2 + (2 if p else 0)
    return L(x1 + dx * r1 / d, y1 + dy * r1 / d, x2 - dx * r2 / d, y2 - dy * r2 / d, p, cls, dash, sw=sw)


# ======================================================================= SVG
# --- glm
def _glm():
    p = "glm"
    s = [T(155, 16, "линейный предиктор", "mut", 10), T(277, 16, "обратная связь g⁻¹", "mut", 10),
         B(15, 24, 62, 34, "c6", "X"), L(77, 41, 106, 41, p),
         B(108, 24, 94, 34, "c1", "η = Xβ"), L(202, 41, 226, 41, p),
         B(228, 24, 98, 34, "c2", "μ = g⁻¹(η)"), L(277, 58, 277, 90, p),
         B(200, 92, 128, 32, "c3", "y ~ семейство(μ)"),
         T(110, 113, "E[y | X] = μ"),
         T(20, 152, "Семейство", "mut", 11, "start", True), T(125, 152, "g(μ)", "mut", 11, "start", True),
         T(240, 152, "μ = g⁻¹(η)", "mut", 11, "start", True),
         L(15, 159, 328, 159, cls="ln s6")]
    rows = [("Gaussian", "μ", "η", "t1"), ("Binomial", "log(μ/(1−μ))", "1/(1+e⁻η)", "t2"),
            ("Poisson", "log μ", "e^η", "t3"), ("Gamma", "1/μ", "1/η", "t4")]
    for i, (f, g, gi, c) in enumerate(rows):
        y = 178 + 22 * i
        s += [T(20, y, f, c, a="start", b=True), T(125, y, g, a="start"), T(240, y, gi, a="start")]
    return svg(p, 340, 252, "Схема GLM", s)


# --- gnn_applications
def _gnna():
    p = "gnna"
    pts = [(35, 40), (85, 28), (108, 72), (68, 96), (24, 86)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 3)]
    s = [E(*pts[a], *pts[b], 11, 11) for a, b in edges]
    s += [N(x, y, 11, "c1" if i in (0, 2) else "c2") for i, (x, y) in enumerate(pts)]
    s += [T(66, 13, "граф G = (V, E), X", "mut", 10),
          L(122, 62, 143, 62, p),
          B(145, 38, 110, 48, "c5", ["GNN", ("message passing", "mut", 10)]),
          L(255, 62, 274, 62, p), T(304, 66, "hᵥ ∈ ℝᵈ")]
    heads = [(60, "Node", "softmax(W·hᵥ)", "класс узла"),
             (170, "Link", "σ(hᵤ · hᵥ)", "есть ли ребро"),
             (280, "Graph", "pool → MLP", "класс графа")]
    second = ["classification", "prediction", "classification"]
    for (cx, a, f, ex), sec in zip(heads, second):
        s += [L(200, 86, cx, 140, p), B(cx - 52, 142, 104, 40, "c3", [(a, "", 11), (sec, "", 11)]),
              T(cx, 202, f, fs=11), T(cx, 220, ex, "mut", 10)]
    s.append(T(170, 242, "Cora · соцсети, рекомендации · молекулы", "mut", 10))
    return svg(p, 340, 250, "Задачи GNN", s)


# --- gpt_architectures
def _gpt():
    p = "gpt"
    cx, bx, bw = 110, 30, 160
    s = [R(10, 90, 196, 214, "c6", 8, dash=True), T(214, 100, "× N", a="start", b=True),
         T(cx, 362, "токены x₁ … xₜ₋₁"), L(cx, 350, cx, 344, p),
         B(bx, 316, bw, 26, "c6", "Token + Pos embedding", 11), L(cx, 316, cx, 292, p),
         B(bx, 268, bw, 22, "c6", "LayerNorm", 11), L(cx, 268, cx, 256, p),
         B(bx, 228, bw, 26, "c1", "Masked Self-Attention", 11), L(cx, 228, cx, 218, p),
         N(cx, 208, 8, "c6", "+"), L(cx, 200, cx, 190, p),
         B(bx, 166, bw, 22, "c6", "LayerNorm", 11), L(cx, 166, cx, 154, p),
         B(bx, 126, bw, 26, "c2", "FFN (MLP)", 11), L(cx, 126, cx, 116, p),
         N(cx, 106, 8, "c6", "+"), L(cx, 98, cx, 80, p),
         B(bx, 56, bw, 22, "c6", "LayerNorm", 11), L(cx, 56, cx, 44, p),
         B(bx, 16, bw, 26, "c3", "Linear + softmax", 11),
         T(196, 34, "→ P(xₜ | x<ₜ)", a="start"),
         PA("M110,304 H20 V208 H100", p, "ln s6"), PA("M110,196 H200 V106 H120", p, "ln s6"),
         T(280, 140, "Causal mask", b=True)]
    n, c0x, c0y, cs = 5, 235, 150, 18
    for i in range(n):
        for j in range(n):
            s.append(R(c0x + j * cs, c0y + i * cs, cs - 2, cs - 2, "c1" if j <= i else "c6", 2))
    s += [T(280, 258, "строка t видит ≤ t", "mut", 10), T(280, 274, "residual — серые", "mut", 10)]
    return svg(p, 340, 370, "Архитектура GPT", s)


# --- gradient_boosting_hyperparameters
def _gbh():
    p = "gbh"
    s = [T(170, 20, "F(x) = F₀ + η·h₁(x) + η·h₂(x) + … + η·h_M(x)"),
         T(170, 40, "η = learning_rate,  M = n_estimators", "mut", 11),
         T(70, 62, "данные для дерева", "mut", 10)]
    rows, cols = {0, 2, 3, 5}, {1, 3, 4}
    for i in range(6):
        for j in range(5):
            s.append(R(30 + j * 16, 80 + i * 14, 15, 13, "c1" if (i in rows and j in cols) else "c6", 1))
    s += [T(22, 80 + i * 14 + 11, "•", "t1") for i in sorted(rows)]
    s += [T(30 + j * 16 + 7.5, 77, "•", "t1") for j in sorted(cols)]
    s += [T(70, 180, "subsample: доля строк", "mut", 10), T(70, 194, "colsample_bytree:", "mut", 10),
          T(70, 207, "доля признаков", "mut", 10)]
    # дерево
    s += [T(235, 70, "split, если gain > gamma", "mut", 10),
          L(235, 102, 185, 128), L(235, 102, 285, 178), L(185, 152, 155, 178), L(185, 152, 215, 178),
          B(200, 78, 70, 24, "c3", "xⱼ < t", 11), B(150, 128, 70, 24, "c3", "xₖ < s", 11),
          B(135, 178, 40, 22, "c2", "w₁", 11), B(195, 178, 40, 22, "c2", "w₂", 11),
          B(265, 178, 40, 22, "c2", "w₃", 11),
          L(322, 80, 322, 198, p, start=True), T(334, 216, "max_depth = 2", "mut", 10, "end"),
          T(170, 240, "листья w: штраф reg_alpha (L1), reg_lambda (L2)", fs=10),
          T(170, 256, "min_child_weight: мин. сумма hessian в листе", fs=10),
          T(170, 278, "малый η → нужно больше деревьев M", "t3", 11, b=True)]
    return svg(p, 340, 288, "Где действуют гиперпараметры бустинга", s)


# --- gradient_boosting_regression
def _gbr():
    p = "gbr"
    s = [B(85, 16, 170, 30, "c6", "F₀(x) = mean(y)"), L(170, 46, 170, 68, p),
         B(60, 70, 220, 30, "c3", "остатки rᵢ = yᵢ − Fₘ₋₁(xᵢ)"), L(170, 100, 170, 122, p),
         T(178, 116, "= антиградиент MSE", "mut", 10, "start"),
         B(55, 124, 230, 30, "c2", "дерево hₘ: обучить на (xᵢ, rᵢ)"), L(170, 154, 170, 176, p),
         B(70, 178, 200, 30, "c1", "Fₘ = Fₘ₋₁ + ν · hₘ"),
         PA("M270,193 H305 V85 H282", p), T(310, 142, "m+1", fs=11, a="start"),
         T(170, 238, "ν — learning_rate (shrinkage)", "mut", 11),
         T(170, 256, "мелкие деревья: max_depth 3–5", "mut", 11),
         T(170, 274, "subsample < 1 → stochastic GB", "mut", 11)]
    return svg(p, 340, 284, "Шаги градиентного бустинга", s)


# --- grammatical_inference
def _gi():
    p = "gi"
    s = [T(170, 18, "RPNI:  E⁺ = {a, aa, aaa},   E⁻ = {ε}"),
         T(10, 42, "1. Prefix tree acceptor (PTA)", "mut", 11, "start")]
    xs = [40, 120, 200, 280]
    s.append(L(8, 75, 22, 75, p))
    for i, x in enumerate(xs):
        s.append(N(x, 75, 16, "c1" if i else "c6", f"q{i}"))
        if i:
            s.append(C(x, 75, 12, "s1"))
        if i < 3:
            s += [E(x, 75, xs[i + 1], 75, 16, 16, p), T((x + xs[i + 1]) / 2, 68, "a", "t3")]
    s += [T(10, 124, "2. Сливаем q1, q2, q3 — E⁻ не нарушен", "mut", 11, "start"),
          T(10, 142, "слить q0 с q1 нельзя: принял бы ε ∈ E⁻", "t4", 11, "start"),
          L(78, 195, 92, 195, p), N(110, 195, 16, "c6", "q0"),
          N(220, 195, 16, "c1", "q1"), C(220, 195, 12, "s1"), T(220, 226, "q1 ≡ q2 ≡ q3", "mut", 10),
          E(110, 195, 220, 195, 16, 16, p), T(165, 188, "a", "t3"),
          PA("M212,181 C198,148 242,148 228,181", p), T(220, 152, "a", "t3"),
          T(300, 200, "L = a⁺", b=True)]
    return svg(p, 340, 236, "RPNI: PTA и слияние состояний", s)


# --- graph_algorithms
def _ga():
    p = "ga"
    pos = {"A": (40, 100), "B": (120, 60), "C": (120, 140), "D": (200, 60), "E": (200, 140), "F": (280, 100)}
    lvl = {"A": "c1", "B": "c2", "C": "c2", "D": "c3", "E": "c3", "F": "c5"}
    edges = ["AB", "AC", "BD", "CD", "CE", "DF", "EF"]
    s = [T(x, 24, f"d={i}", "mut", 11) for i, x in enumerate([40, 120, 200, 280])]
    s += [E(*pos[a], *pos[b], 15, 15) for a, b in edges]
    s += [N(*pos[k], 15, lvl[k], k) for k in pos]
    s += [T(14, 192, "BFS (очередь):", "t1", a="start", b=True), T(200, 192, "A B C D E F", a="start"),
          T(14, 212, "DFS (стек):", "t5", a="start", b=True), T(200, 212, "A B D C E F", a="start"),
          T(170, 236, "BFS: уровни d = кратчайший путь по числу рёбер", "mut", 10)]
    return svg(p, 320, 246, "BFS и DFS на графе", s)


# --- graph_based_ssl
def _gssl():
    p = "gssl"
    base = [(20, 25), (55, 15), (40, 55), (15, 80), (100, 40), (130, 25), (115, 75), (140, 95)]
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (0, 3), (4, 5), (4, 6), (5, 6), (6, 7)]
    s = [T(85, 36, "до: 2 метки", b=True), T(255, 36, "после", b=True)]
    for ox, after in [(15, False), (185, True)]:
        pts = [(x + ox, y + 50) for x, y in base]
        s += [E(*pts[a], *pts[b], 9, 9) for a, b in edges]
        s.append(E(*pts[2], *pts[4], 9, 9, cls="ln s6", dash=True))
        for i, (x, y) in enumerate(pts):
            if i == 1:
                s.append(N(x, y, 9, "c1", sw=3))
            elif i == 7:
                s.append(N(x, y, 9, "c4", sw=3))
            elif after:
                s.append(N(x, y, 9, "c1" if i < 4 else "c4"))
            else:
                s.append(N(x, y, 9, "c6", "?", 10))
    s += [L(162, 112, 180, 112, p),
          T(170, 196, "wᵢⱼ = exp(−‖xᵢ − xⱼ‖² / 2σ²)"),
          T(170, 218, "P = D⁻¹W,  Y⁽ᵗ⁺¹⁾ = αPY⁽ᵗ⁾ + (1−α)Y⁽⁰⁾"),
          T(170, 240, "гладкость: соседи получают похожие метки", "mut", 11),
          T(170, 258, "пунктир — слабое ребро между кластерами", "mut", 11)]
    return svg(p, 340, 268, "Распространение меток по графу", s)


# --- graph_kernels
def _gk():
    p = "gk"
    shape = [(85, 80), (30, 80), (130, 45), (130, 115)]
    edges = [(0, 1), (0, 2), (0, 3), (2, 3)]
    cls = {"A": "c1", "B": "c2", "C": "c3", "D": "c5", "E": "c4"}
    s = [T(80, 20, "метки ℓ₀", b=True), T(260, 20, "метки ℓ₁ (h = 1)", b=True)]
    for ox, labs in [(0, "ABBB"), (180, "CDEE")]:
        pts = [(x + ox, y) for x, y in shape]
        s += [E(*pts[a], *pts[b], 14, 14) for a, b in edges]
        s += [N(x, y, 14, cls[l], l) for (x, y), l in zip(pts, labs)]
    s += [L(152, 80, 192, 80, p),
          T(170, 150, "(своя метка | мультимножество соседей)", "mut", 10),
          T(170, 170, "A | B,B,B → C"), T(170, 188, "B | A → D"), T(170, 206, "B | A,B → E"),
          T(84, 268, "φ(G):", a="end", b=True)]
    counts = [("A", 1), ("B", 3), ("C", 1), ("D", 1), ("E", 2)]
    for i, (l, c) in enumerate(counts):
        x = 100 + 30 * i
        s += [R(x, 290 - 14 * c, 20, 14 * c, cls[l], 2), T(x + 10, 304, l, fs=11),
              T(x + 10, 286 - 14 * c, str(c), "mut", 10)]
    s += [L(94, 290, 254, 290, cls="ln s6"),
          T(170, 326, "K(G₁, G₂) = ⟨φ(G₁), φ(G₂)⟩"),
          T(170, 344, "счётчики меток всех итераций 0…h", "mut", 10)]
    return svg(p, 340, 352, "Итерация Weisfeiler-Lehman", s)


# --- graph_theory_ml
def _gtm():
    p = "gtm"
    pos = [(40, 55), (120, 55), (80, 115), (80, 175)]
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    s = [E(*pos[a], *pos[b], 14, 14) for a, b in edges]
    s += [N(x, y, 14, "c1", str(i + 1)) for i, (x, y) in enumerate(pos)]
    A = [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
    x0, y0, cs = 180, 52, 24
    s += [T(228, 22, "A (смежность)", b=True, fs=11), T(304, 22, "deg", b=True, fs=11)]
    for i in range(4):
        s += [T(x0 + i * cs + 12, 46, str(i + 1), "mut", 10), T(x0 - 8, y0 + i * cs + 16, str(i + 1), "mut", 10)]
        for j in range(4):
            s += [R(x0 + j * cs, y0 + i * cs, cs - 2, cs - 2, "c1" if A[i][j] else "c6", 2),
                  T(x0 + j * cs + 11, y0 + i * cs + 16, str(A[i][j]), fs=11)]
        s.append(T(304, y0 + i * cs + 16, str(sum(A[i])), "t3", b=True))
    s += [T(170, 212, "L = D − A,   0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ"),
          T(170, 230, "λ₂ (Fiedler) > 0 ⇔ граф связен", "mut", 11)]
    return svg(p, 340, 240, "Граф, матрица смежности и степени", s)


# --- hdbscan
def _hdb():
    p = "hdb"
    steps = [("c1", "core_k(x): расстояние до k-го соседа"),
             ("c1", "d_mreach(a,b) = max(core_k(a), core_k(b), d(a,b))"),
             ("c2", "MST по d_mreach → single linkage tree"),
             ("c3", "Condensed tree: ветви < min_cluster_size → шум"),
             ("c5", "Стабильные кластеры (EOM); шум = −1")]
    s = []
    for i, (c, t) in enumerate(steps):
        y = 12 + 48 * i
        s += [B(28, y, 304, 32, c, t, 11), N(16, y + 16, 9, "c6", str(i + 1), 10)]
        if i < 4:
            s.append(L(180, y + 32, 180, y + 46, p))
    s += [T(170, 262, "λ = 1/d;  стабильность = Σₚ (λₚ − λ_birth)", "mut", 11),
          T(170, 280, "плотные области «живут» дольше по λ", "mut", 10)]
    return svg(p, 340, 290, "Шаги HDBSCAN", s)


# --- hierarchical_rl
def _hrl():
    p = "hrl"
    s = [B(60, 16, 170, 34, "c5", "π_hi(g|s): менеджер", 11),
         B(60, 118, 170, 34, "c1", "π_lo(a|s,g): исполнитель", 11),
         B(60, 220, 170, 32, "c6", "Среда"),
         L(120, 50, 120, 116, p), T(112, 82, "подцель g", a="end"), T(112, 97, "раз в k шагов", "mut", 10, "end"),
         L(120, 152, 120, 218, p), T(112, 190, "действие aₜ", a="end"),
         L(175, 220, 175, 154, p), T(183, 184, "sₜ, rᵢₙ(s, g)", a="start", fs=11),
         T(183, 200, "внутренняя", "mut", 10, "start"),
         PA("M230,236 H300 V33 H232", p), T(296, 88, "sₜ, Σ R", a="end", fs=11),
         T(296, 102, "внешняя", "mut", 10, "end"),
         T(170, 274, "«дойти до кухни» → шаги к кухне", "mut", 11)]
    return svg(p, 340, 284, "Двухуровневая иерархия HRL", s)


# --- hierarchical_rl_detailed
def _hrld():
    p = "hrld"
    s = [T(170, 20, "Option o = (I, π, β)", b=True),
         T(8, 63, "опции", "mut", 11, "start"), T(8, 108, "шаги", "mut", 11, "start"),
         B(70, 46, 120, 24, "c5", "o₁: к двери", 11), B(190, 46, 120, 24, "c3", "o₂: открыть дверь", 11),
         T(130, 88, "π₁ выбирает aₜ", "mut", 10), T(250, 88, "π₂ выбирает aₜ", "mut", 10)]
    for i in range(12):
        s.append(R(72 + 20 * i, 96, 16, 16, "c5" if i < 6 else "c3", 2))
    s += [L(70, 130, 318, 130, p), T(326, 134, "t", a="start")]
    for x, lab in [(70, "s₀ ∈ I₁"), (190, "β₁(s) = 1"), (310, "β₂ = 1")]:
        s += [L(x, 124, x, 136), T(x if x < 300 else 330, 150, lab, "mut", 10, "middle" if x < 300 else "end")]
    s += [T(170, 182, "Q(s,o) ← Q(s,o) + α[R + γᵏ maxₒ′ Q(s′,o′) − Q(s,o)]", fs=11),
          T(170, 202, "R — награда за k шагов опции (SMDP)", "mut", 11),
          T(170, 220, "I — где можно начать, β — когда остановиться", "mut", 10)]
    return svg(p, 340, 232, "Временная структура options", s)


# --- hoeffding_trees
def _ht():
    p = "ht"
    s = [B(40, 12, 260, 30, "c6", "(x, y) → лист: обновить счётчики", 11),
         L(170, 42, 170, 60, p),
         B(40, 62, 260, 30, "c3", "накоплено grace_period примеров?", 11),
         PA("M300,77 H318 V27 H302", p), T(312, 52, "нет", fs=10, a="start"),
         L(170, 92, 170, 110, p), T(178, 106, "да", fs=10, a="start"),
         B(40, 112, 260, 44, "c1", [("ΔG = G(X_a) − G(X_b)  (лучший − второй)", "", 11),
                                    ("ε = √(R² ln(1/δ) / 2n)", "", 11)]),
         L(170, 156, 170, 174, p),
         B(70, 176, 200, 30, "c3", "ΔG > ε  или  ε < τ ?", 12),
         L(130, 206, 105, 230, p), T(108, 220, "да", fs=10, a="end"),
         L(210, 206, 240, 230, p), T(234, 220, "нет", fs=10, a="start"),
         B(40, 232, 130, 30, "c2", "split по X_a", 11), B(190, 232, 110, 30, "c6", "ждать данных", 11),
         T(170, 284, "τ — tie-break при почти равных атрибутах", "mut", 10)]
    return svg(p, 340, 294, "Решение о разбиении в VFDT", s)


# --- hugging_face_transformers
def _hf():
    p = "hf"
    chain = [("c6", "текст: «Отличный фильм!»"), ("c1", "AutoTokenizer"),
             ("c6", "input_ids, attention_mask"), ("c2", "AutoModelFor…"),
             ("c6", "logits → softmax"), ("c3", "{label: POSITIVE, score: 0.99}")]
    s = []
    for i, (c, t) in enumerate(chain):
        y = 10 + 46 * i
        s.append(B(30, y, 200, 30, c, t, 11))
        if i < 5:
            s.append(L(130, y + 30, 130, y + 44, p))
    s += [B(252, 86, 80, 90, "c5", [("Model", "", 11), ("Hub", "", 11), ("from_", "mut", 10),
                                    ("pretrained()", "mut", 10)]),
          L(252, 110, 232, 71, p, dash=True), L(252, 152, 232, 163, p, dash=True),
          T(170, 290, "pipeline() делает шаги 2–5 сразу", "mut", 10)]
    return svg(p, 340, 298, "Поток данных в Transformers", s)


# --- hybrid_recommender_systems
def _hyb():
    p = "hyb"
    s = [T(10, 20, "1. Weighted", "t5", a="start", b=True),
         B(10, 30, 80, 24, "c1", "CF", 11), B(10, 62, 80, 24, "c2", "Content", 11),
         L(90, 42, 123, 52, p), L(90, 74, 123, 64, p),
         B(125, 42, 130, 32, "c5", "w₁·s_CF + w₂·s_CB", 11), L(255, 58, 278, 58, p),
         B(280, 46, 50, 24, "c6", "топ-N", 11),
         T(10, 112, "2. Switching", "t3", a="start", b=True),
         B(10, 122, 130, 32, "c3", "новый пользователь?", 11),
         L(140, 132, 188, 130, p), T(164, 124, "да", fs=10),
         L(140, 146, 188, 162, p), T(158, 168, "нет", fs=10),
         B(190, 118, 110, 24, "c2", "Content-based", 11), B(190, 150, 110, 24, "c1", "CF", 11),
         T(10, 200, "3. Cascade", "t1", a="start", b=True),
         B(10, 210, 100, 36, "c1", [("CF:", "", 11), ("1000 кандидатов", "", 11)]),
         L(110, 228, 123, 228, p),
         B(125, 210, 100, 36, "c2", [("Content:", "", 11), ("rerank → 50", "", 11)]),
         L(225, 228, 238, 228, p),
         B(240, 210, 90, 36, "c6", [("бизнес-", "", 11), ("правила → 10", "", 11)]),
         T(170, 272, "цель: холодный старт + разреженность", "mut", 11)]
    return svg(p, 340, 282, "Типы гибридизации", s)


# --- hyperparameter_strategies
def _hps():
    p = "hps"
    s = [T(170, 20, "Successive Halving (η = 3)", b=True)]
    rows = [(9, 18, {1, 4, 7}, "9 × r"), (3, 58, {1}, "3 × 3r"), (1, 178, {0}, "1 × 9r")]
    for k, (n, w, best, lab) in enumerate(rows):
        y = 40 + 60 * k
        s.append(T(8, y + 17, f"раунд {k}", "mut", 11, "start"))
        for i in range(n):
            s.append(R(60 + i * (w + 2), y, w, 26, "c2" if i in best else "c6", 3))
        s.append(T(248, y + 17, lab, fs=11, a="start"))
        if k < 2:
            s += [L(150, y + 28, 150, y + 58, p), T(158, y + 47, "лучшие 1/3", "mut", 10, "start")]
    s += [T(150, 177, "лучшая конфигурация", fs=11),
          T(170, 220, "каждый раунд — одинаковый бюджет 9r", "mut", 11),
          T(170, 238, "Hyperband: несколько таких сеток", "mut", 11),
          T(170, 256, "с разным стартовым числом конфигураций", "mut", 11)]
    return svg(p, 340, 266, "Successive Halving", s)


# --- hypothesis_testing
def _hyp():
    p = "hyp"
    s = [B(105, 10, 130, 28, "c6", "Что сравниваем?"),
         L(150, 38, 75, 60, p), L(190, 38, 250, 60, p),
         B(15, 62, 120, 28, "c5", "категории"), B(175, 62, 150, 28, "c3", "числовые"),
         L(75, 90, 75, 110, p), B(15, 112, 120, 28, "c5", "χ²-тест"),
         L(250, 90, 218, 150, p),
         T(155, 166, "независимые", fs=11, b=True), T(277, 166, "парные", fs=11, b=True),
         T(8, 193, "норм.", "mut", 11, "start"), T(8, 229, "не норм.", "mut", 11, "start"),
         B(95, 174, 120, 30, "c1", "t-тест (Welch)", 11), B(220, 174, 115, 30, "c1", "парный t-тест", 11),
         B(95, 210, 120, 30, "c2", "Mann-Whitney U", 11), B(220, 210, 115, 30, "c2", "Wilcoxon", 11),
         T(170, 264, "p < α → отвергаем H₀ (обычно α = 0.05)", fs=11),
         T(170, 282, "p-value — вероятность таких данных при H₀", "mut", 10)]
    return svg(p, 340, 292, "Выбор статистического теста", s)


# --- image_classification_dl
def _icd():
    p = "icd"
    s = []
    for title, cy, colors in [("1) Feature extraction: backbone заморожен", 66, ["c6"] * 4),
                              ("2) Fine-tuning: верхние блоки, LR ≈ 1e-5", 176, ["c6", "c6", "c3", "c3"])]:
        s += [T(10, cy - 44, title, fs=11, a="start", b=True),
              B(10, cy - 15, 30, 30, "c6", "img", 10), L(40, cy, 52, cy, p)]
        for i, h in enumerate([60, 50, 40, 30]):
            s.append(R(55 + 36 * i, cy - h / 2, 28, h, colors[i], 3))
        s += [L(199, cy, 210, cy, p), B(212, cy - 14, 34, 28, "c6", "GAP", 10), L(246, cy, 256, cy, p),
              B(258, cy - 18, 72, 36, "c2", [("Dense +", "", 11), ("softmax", "", 11)])]
    s += [T(122, 112, "pretrained (ImageNet)", "mut", 10),
          T(87, 222, "заморожено", "mut", 10), T(162, 222, "дообучаем", "mut", 10),
          R(10, 238, 12, 12, "c6", 2), T(26, 248, "заморожено", fs=11, a="start"),
          R(110, 238, 12, 12, "c3", 2), T(126, 248, "дообучение", fs=11, a="start"),
          R(210, 238, 12, 12, "c2", 2), T(226, 248, "новая голова", fs=11, a="start")]
    return svg(p, 340, 258, "Transfer learning в два этапа", s)


# --- image_text_models
def _itm():
    p = "itm"
    sub = "₁₂₃"
    s = [T(210, 22, "«кот», «собака», «машина»", "mut", 11),
         B(130, 32, 160, 30, "c2", "Text encoder (Transformer)", 11), L(210, 62, 210, 88, p),
         T(55, 140, "картинки", "mut", 11),
         B(10, 150, 90, 40, "c1", [("Image", "", 11), ("encoder", "", 11)]),
         T(55, 208, "ViT / ResNet", "mut", 10), L(100, 170, 118, 170, p)]
    for j in range(3):
        s.append(T(170 + 40 * j, 104, f"T{sub[j]}", "t2", b=True))
        s.append(T(136, 134 + 40 * j, f"I{sub[j]}", "t1", b=True))
        for i in range(3):
            s += [R(150 + 40 * j, 110 + 40 * i, 38, 38, "c2" if i == j else "c6", 3),
                  T(169 + 40 * j, 133 + 40 * i, f"I{sub[i]}·T{sub[j]}", fs=10)]
    s += [T(210, 252, "InfoNCE: диагональ ↑, остальное ↓", fs=11),
          T(210, 272, "sᵢⱼ = cos(Iᵢ, Tⱼ) / τ,  τ ≈ 0.07", fs=11),
          T(210, 290, "zero-shot: класс = argmaxⱼ sᵢⱼ", "mut", 10)]
    return svg(p, 340, 300, "Контрастное обучение CLIP", s)


# --- image_to_image_translation
def _i2i():
    p = "i2i"
    s = [B(10, 40, 70, 34, "c6", "x (скетч)", 11), L(80, 57, 108, 57, p),
         B(110, 32, 100, 50, "c1", ["G: U-Net", ("skip connections", "mut", 10)]), L(210, 57, 248, 57, p),
         B(250, 40, 80, 34, "c6", "G(x)"),
         B(10, 120, 70, 34, "c2", "y (фото)", 11),
         PA("M30,74 V106 H140 V128", p, "ln s6", dash=True),
         L(80, 140, 118, 148, p), L(290, 74, 232, 146, p),
         B(120, 130, 110, 46, "c4", ["D: PatchGAN", ("пара (x, ·)", "mut", 10)]),
         L(230, 172, 253, 196, p)]
    real = [[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1]]
    for i in range(4):
        for j in range(4):
            s.append(R(255 + 12 * j, 192 + 12 * i, 11, 11, "c2" if real[i][j] else "c4", 1))
    s += [T(279, 258, "real/fake по патчам", "mut", 10),
          T(10, 206, "L = L_cGAN(G, D)", a="start"), T(22, 226, "+ λ·‖y − G(x)‖₁", a="start"),
          T(10, 246, "λ ≈ 100: L1 держит структуру", "mut", 10, "start"),
          T(170, 280, "CycleGAN (непарные): + λ_cyc·‖F(G(x)) − x‖₁", "mut", 10)]
    return svg(p, 340, 290, "Pix2Pix: генератор и PatchGAN", s)


# --- imbalanced_algorithms
def _imba():
    p = "imba"
    s = [T(170, 14, "K подвыборок: |−| = |+| = N₊", "mut", 11),
         B(10, 22, 240, 22, "c1", "мажоритарный класс (−)", 11), B(262, 22, 30, 22, "c4", "+", 11),
         T(298, 37, "N₊", a="start", fs=11)]
    srcs = [50, 130, 210]
    for k in range(3):
        x0 = 15 + 105 * k
        s += [L(srcs[k], 44, x0 + 25, 76, p, "ln s1", dash=True),
              L(277, 44, x0 + 75, 76, p, "ln s4", dash=True),
              B(x0, 78, 50, 26, "c1", "−", 11), B(x0 + 50, 78, 50, 26, "c4", "+", 11),
              L(x0 + 50, 104, x0 + 50, 126, p),
              B(x0 + 15, 128, 70, 28, "c5", f"h{'₁₂₃'[k]}", 11),
              L(x0 + 50, 156, 170, 182, p)]
    s += [B(80, 184, 180, 30, "c2", "среднее / голосование", 11),
          L(170, 214, 170, 228, p), T(170, 244, "ŷ = sign(Σₖ hₖ(x))", fs=11),
          T(170, 266, "Easy Ensemble: AdaBoost на каждой подвыборке", "mut", 10)]
    return svg(p, 340, 276, "Easy Ensemble", s)


# --- imbalanced_metrics
def _imbm():
    p = "imbm"
    x0, y0, cw, ch = 100, 50, 80, 50
    cells = [("TN", "9850", "c6"), ("FP", "50", "c3"), ("FN", "30", "c4"), ("TP", "70", "c2")]
    s = [T(140, 42, "Pred: N", b=True, fs=11), T(220, 42, "Pred: P", b=True, fs=11),
         T(92, 79, "True: N", a="end", b=True, fs=11), T(92, 129, "True: P", a="end", b=True, fs=11)]
    for k, (a, n, c) in enumerate(cells):
        i, j = divmod(k, 2)
        x, y = x0 + j * cw, y0 + i * ch
        s += [R(x, y, cw - 2, ch - 2, c, 3), T(x + cw / 2 - 1, y + 21, a, b=True),
              T(x + cw / 2 - 1, y + 37, n, "mut", 11)]
    s += [R(176, 46, 86, 106, "s1", 6, sw=2, dash=True), R(96, 98, 170, 56, "s2", 6, sw=2, dash=True),
          T(300, 62, "Precision", "t1", 11, "middle"), T(300, 76, "(столбец)", "mut", 10),
          T(48, 158, "Recall (строка)", "t2", 10),
          T(170, 186, "Precision = TP/(TP+FP) = 70/120 ≈ 0.58", "t1"),
          T(170, 206, "Recall = TP/(TP+FN) = 70/100 = 0.70", "t2"),
          T(170, 226, "F1 = 2PR/(P+R) ≈ 0.64"),
          T(170, 246, "Accuracy = 9920/10000 = 0.992 — обманчиво", "t4", 11)]
    return svg(p, 340, 256, "Матрица ошибок и метрики", s)


# --- imitation_learning
def _il():
    p = "il"
    s = [B(60, 12, 220, 30, "c5", "демонстрации эксперта π*", 11), L(170, 42, 170, 60, p),
         B(90, 62, 160, 30, "c6", "D = {(s, a*)}"), L(170, 92, 170, 110, p),
         B(60, 112, 220, 30, "c1", "обучить π на D (supervised)", 11),
         L(240, 142, 255, 170, p),
         B(180, 172, 150, 40, "c3", [("запустить π:", "", 11), ("s ~ d_π", "", 11)]),
         L(180, 192, 162, 192, p),
         B(10, 172, 150, 40, "c5", [("эксперт даёт", "", 11), ("a* = π*(s)", "", 11)]),
         PA("M30,172 V77 H88", p, "ln s2"), T(36, 104, "D ← D ∪ новые", "t2", 10, "start"),
         T(170, 236, "BC — только верхняя цепочка: ошибки копятся", "t1", 10),
         T(170, 254, "DAgger — весь цикл: π учится на своих состояниях", "t2", 10)]
    return svg(p, 340, 264, "Behavioral Cloning и DAgger", s)


# --- inception_efficientnet
def _incep():
    p = "incep"
    xs = [50, 130, 210, 290]
    rowa = [("1×1", "c1"), ("1×1", "c1"), ("1×1", "c1"), ("max pool", "c3")]
    rowb = [None, ("3×3", "c2"), ("5×5", "c5"), ("1×1", "c1")]
    s = [B(70, 12, 200, 28, "c6", "предыдущий слой")]
    for x, a, b in zip(xs, rowa, rowb):
        s += [L(170, 40, x, 68, p), B(x - 35, 70, 70, 26, a[1], a[0], 11)]
        if b:
            s += [L(x, 96, x, 128, p), B(x - 35, 130, 70, 26, b[1], b[0], 11), L(x, 156, x, 196, p)]
        else:
            s.append(L(x, 96, x, 196, p))
    s += [B(20, 198, 300, 28, "c6", "Concat по каналам"),
          T(170, 248, "1×1 перед 3×3 / 5×5 сжимает число каналов", "mut", 11)]
    return svg(p, 340, 258, "Inception-модуль", s)


# --- incremental_learning
def _inc():
    p = "inc"
    s = [L(10, 232, 322, 232, p), L(10, 232, 10, 18, p), T(318, 248, "θ₁", a="end"), T(22, 22, "θ₂", a="start"),
         '<ellipse class="s1" cx="135" cy="170" rx="115" ry="30" style="stroke-width:2"/>',
         '<ellipse class="s2" cx="255" cy="130" rx="32" ry="70" style="stroke-width:2"/>',
         T(62, 175, "задача A", "t1", 11), T(255, 52, "задача B", "t2", 11),
         L(135, 170, 262, 131, p, "ln s4"), L(135, 170, 235, 168, p, "ln s2", sw=2),
         C(135, 170, 4, "c1"), T(135, 190, "θ*_A", fs=11),
         C(265, 130, 4, "c2"), T(262, 118, "θ*_B", fs=11),
         C(238, 168, 4, "c2"),
         T(165, 128, "SGD: забывает A", "t4", 11), T(188, 186, "EWC", "t2", 11, b=True),
         T(170, 262, "L = L_B(θ) + Σᵢ (λ/2)·Fᵢ·(θᵢ − θ*_A,ᵢ)²"),
         T(170, 280, "Fᵢ — Fisher: насколько вес i важен для A", "mut", 10)]
    return svg(p, 340, 290, "Геометрия EWC", s)


# --- inductive_logic_programming
def _ilp():
    p = "ilp"
    s = [B(10, 10, 160, 56, "c6", [("B: фоновые знания", "", 11), ("parent(tom, bob)", "mut", 11),
                                  ("parent(bob, ann)", "mut", 11)]),
         B(10, 72, 160, 28, "c2", "E⁺: grandparent(tom, ann)", 10),
         B(10, 106, 160, 28, "c4", "E⁻: grandparent(tom, bob)", 10),
         L(170, 38, 198, 62, p), L(170, 86, 198, 76, p), L(170, 120, 198, 90, p),
         B(200, 50, 130, 48, "c5", ["ILP", ("FOIL, Progol, Aleph", "mut", 10)]),
         L(265, 98, 265, 150, p),
         B(10, 152, 320, 30, "c1", "H: grandparent(X,Y) :- parent(X,Z), parent(Z,Y).", 11),
         T(14, 210, "Специализация (от общего к частному):", "mut", 11, "start"),
         T(14, 230, "grandparent(X,Y).  — покрывает E⁻", "t4", 11, "start"),
         T(14, 250, "… :- parent(X,Z).  — всё ещё покрывает E⁻", "t4", 11, "start"),
         T(14, 270, "… :- parent(X,Z), parent(Z,Y).  ✓ E⁺, не E⁻", "t2", 11, "start")]
    return svg(p, 340, 282, "ILP: вход, гипотеза и поиск", s)


# --- influence_maximization
def _im():
    p = "im"
    pos = {"S": (40, 110), "a": (120, 50), "b": (120, 170), "c": (200, 110), "d": (280, 50), "e": (280, 170)}
    ok = [("S", "a", "0.4"), ("a", "c", "0.5"), ("c", "e", "0.4")]
    fail = [("S", "b", "0.2"), ("a", "d", "0.3"), ("c", "d", "0.2")]
    s = []
    for u, v, pr in ok:
        s.append(E(*pos[u], *pos[v], 15, 15, p, sw=2))
    for u, v, pr in fail:
        s.append(E(*pos[u], *pos[v], 15, 15, p, "ln s6", dash=True))
    for u, v, pr in ok + fail:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        d = (dx * dx + dy * dy) ** 0.5
        s.append(T(mx - dy / d * 11, my + dx / d * 11 + 4, pr, "mut", 10))
    style = {"S": ("c4", "t=0"), "a": ("c3", "t=1"), "c": ("c3", "t=2"), "e": ("c3", "t=3"),
             "b": ("c6", ""), "d": ("c6", "")}
    for k, (x, y) in pos.items():
        c, lab = style[k]
        s.append(N(x, y, 15, c, k))
        if lab:
            s.append(T(x, y - 20 if k in "ad" else y + 28, lab, "mut", 10))
    s += [T(170, 214, "сплошная — активация, пунктир — неудача", fs=11),
          T(170, 232, "новый активный u: одна попытка на соседа v", "mut", 11),
          T(170, 250, "σ(S) — ожидаемое число активных узлов", "mut", 10),
          T(170, 266, "greedy (субмодулярность): ≥ (1 − 1/e)·OPT", "mut", 10)]
    return svg(p, 340, 274, "Independent Cascade", s)


# --- information_bottleneck_theory
def _ib():
    p = "ib"
    s = [T(170, 22, "марковская цепь: Y → X → T → Ŷ", "mut", 11),
         PA("M50,80 V70 H170 V80", cls="ln s3", sw=2), T(110, 62, "I(X;T) → min (сжатие)", "t3", 11),
         B(20, 100, 60, 40, "c6", "X"), L(80, 120, 138, 120, p), T(109, 112, "p(t|x)", "mut", 10),
         B(140, 100, 60, 40, "c5", "T"), L(200, 120, 258, 120, p), T(229, 112, "p(ŷ|t)", "mut", 10),
         B(260, 100, 60, 40, "c2", "Ŷ"),
         PA("M170,152 V162 H290 V152", cls="ln s2", sw=2), T(230, 180, "I(T;Y) → max (предсказание)", "t2", 11),
         T(170, 212, "L_IB = I(T;Y) − β·I(T;X)  → max"),
         T(170, 234, "β ↑ — сильнее сжатие (вплоть до недообучения)", "mut", 10)]
    return svg(p, 340, 246, "Information Bottleneck", s)


# --- intrusion_detection
def _ids():
    p = "ids"
    s = [B(60, 10, 220, 30, "c6", "сетевой трафик / логи хостов", 11), L(170, 40, 170, 58, p),
         B(30, 60, 280, 30, "c6", "признаки потоков: bytes, duration, flags", 11),
         L(120, 90, 90, 118, p), L(220, 90, 250, 118, p),
         B(15, 120, 145, 44, "c3", [("Signature-based", "", 11), ("правила, известные атаки", "mut", 10)]),
         B(180, 120, 145, 44, "c1", [("Anomaly-based (ML)", "", 11), ("модель нормы → zero-day", "mut", 10)]),
         L(90, 164, 140, 188, p), L(250, 164, 200, 188, p),
         B(70, 190, 200, 30, "c5", "Hybrid: оценка → alert?", 11), L(170, 220, 170, 238, p),
         B(70, 240, 200, 30, "c4", "SOC: реакция, блокировка", 11)]
    return svg(p, 340, 280, "Конвейер IDS", s)


# --- inverse_rl
def _irl():
    p = "irl"
    s = [T(140, 22, "RL:   R → π*", a="start"), T(140, 42, "IRL:  π_E (демонстрации) → R", a="start"),
         B(10, 10, 110, 34, "c5", "D эксперта → μ_E", 11), L(60, 44, 60, 118, p),
         B(150, 60, 150, 30, "c3", "R(s) = wᵀφ(s)", 11), L(260, 90, 260, 118, p),
         B(190, 120, 140, 30, "c1", "RL: найти π для R", 11), L(260, 150, 230, 178, p),
         B(100, 180, 160, 30, "c6", "μ_π = E_π[Σ γᵗ φ(sₜ)]", 11), L(130, 180, 100, 152, p),
         B(10, 120, 140, 30, "c2", "w ← w + α(μ_E − μ_π)", 11), L(150, 122, 168, 92, p),
         T(170, 236, "цель: μ_π ≈ μ_E (feature matching)", "mut", 11),
         T(170, 256, "MaxEnt IRL: π(a|s) ∝ exp(Q(s,a)/α)", "mut", 10),
         T(170, 272, "GAIL: R(s,a) = log D − log(1 − D)", "mut", 10)]
    return svg(p, 340, 282, "Цикл IRL с feature matching", s)


# --- jax_flax
def _jax():
    p = "jax"
    tr = [("c1", "grad(f): ∇f"), ("c2", "jit(f): компиляция XLA"),
          ("c3", "vmap(f): по батчу"), ("c5", "pmap(f): по устройствам")]
    s = [B(10, 62, 70, 32, "c6", "f(x)")]
    for i, (c, t) in enumerate(tr):
        y = 10 + 36 * i
        s += [L(80, 78, 118, y + 14, p), B(120, y, 210, 28, c, t, 11)]
    s += [T(170, 170, "jit(vmap(grad(loss)))  — композиция", fs=11),
          T(10, 192, "Flax:", "t5", 11, "start", True),
          B(10, 200, 100, 30, "c1", "init(key, x)", 11), L(110, 215, 133, 215, p),
          B(135, 200, 80, 30, "c3", "params", 11), L(215, 215, 230, 215, p),
          B(232, 200, 100, 30, "c2", "apply(params, x)", 10),
          T(170, 250, "код (Module) и состояние (params) разделены", "mut", 10),
          T(170, 268, "PRNG: key, sub = jax.random.split(key)", "mut", 10)]
    return svg(p, 340, 278, "Преобразования JAX и Flax", s)


# --- kalman_filters
def _kf():
    p = "kf"
    s = [T(160, 12, "x̂₀, P₀", fs=11), L(160, 16, 160, 36, p),
         B(30, 38, 260, 62, "c1", [("Predict (прогноз)", "", 12), ("x̂ₖ⁻ = F·x̂ₖ₋₁ + B·uₖ", "", 12),
                                    ("Pₖ⁻ = F·Pₖ₋₁·Fᵀ + Q", "", 12)]),
         L(160, 100, 160, 136, p),
         B(30, 138, 260, 88, "c2", [("Update (коррекция)", "", 12), ("Kₖ = Pₖ⁻Hᵀ(HPₖ⁻Hᵀ + R)⁻¹", "", 12),
                                     ("x̂ₖ = x̂ₖ⁻ + Kₖ(zₖ − H·x̂ₖ⁻)", "", 12), ("Pₖ = (I − KₖH)·Pₖ⁻", "", 12)]),
         PA("M290,182 H318 V69 H292", p), T(330, 126, "k → k+1", "mut", 11, rot=90),
         L(160, 262, 160, 228, p), T(160, 278, "измерение zₖ", fs=11)]
    return svg(p, 340, 286, "Цикл фильтра Калмана", s)


# --- kernel_cca
def _kcca():
    p = "kcca"
    s = [B(10, 30, 80, 34, "c1", "X (вид 1)", 11), L(90, 47, 113, 47, p),
         B(115, 30, 100, 34, "c6", "Kₓ = k(xᵢ, xⱼ)", 11), L(215, 47, 238, 47, p),
         B(240, 30, 90, 34, "c1", "u = Kₓα", 11),
         B(10, 130, 80, 34, "c2", "Y (вид 2)", 11), L(90, 147, 113, 147, p),
         B(115, 130, 100, 34, "c6", "K_y = k(yᵢ, yⱼ)", 11), L(215, 147, 238, 147, p),
         B(240, 130, 90, 34, "c2", "v = K_yβ", 11),
         L(285, 68, 285, 126, p, start=True), T(276, 101, "max corr(u, v)", "t5", 11, "end"),
         T(170, 196, "ρ = αᵀKₓK_yβ / √(αᵀ(Kₓ+κI)²α · βᵀ(K_y+κI)²β)", fs=11),
         T(170, 218, "κ > 0 обязателен: иначе ρ = 1 (переобучение)", "mut", 11),
         T(170, 238, "kernel trick: φ и ψ явно не вычисляем", "mut", 10)]
    return svg(p, 340, 248, "Kernel CCA", s)


# --- kernel_pca
def _kpca_pts():
    from sklearn.decomposition import KernelPCA
    a_in, a_out = np.linspace(0, 2 * np.pi, 8, endpoint=False), np.linspace(0, 2 * np.pi, 14, endpoint=False)
    X = np.r_[np.c_[0.3 * np.cos(a_in), 0.3 * np.sin(a_in)], np.c_[np.cos(a_out), np.sin(a_out)]]
    Z = KernelPCA(2, kernel="rbf", gamma=2).fit_transform(X)
    return X, Z


def _kpca():
    p = "kpca"
    X, Z = _kpca_pts()
    s = [T(75, 18, "исходное пространство", "mut", 10), T(262, 18, "проекция на PC₁, PC₂", "mut", 10),
         L(135, 80, 183, 80, p), T(159, 72, "φ / k", fs=11),
         L(200, 140, 328, 140, cls="ln s6"), T(326, 154, "PC₁", "mut", 10, "end")]
    for k, (x, y) in enumerate(X):
        s.append(C(75 + 50 * x, 80 - 50 * y, 3.5, "t1" if k < 8 else "t4"))
    zx = (Z[:, 0] - Z[:, 0].min()) / np.ptp(Z[:, 0])
    zy = Z[:, 1] / (np.abs(Z[:, 1]).max() + 1e-9)
    for k in range(len(Z)):
        s.append(C(210 + 110 * zx[k], 82 - 45 * zy[k], 3.5, "t1" if k < 8 else "t4"))
    s += [T(10, 176, "1. K = [k(xᵢ, xⱼ)], RBF: exp(−γ‖xᵢ − xⱼ‖²)", fs=11, a="start"),
          T(10, 196, "2. центрировать: K̃ = K − 1ₙK − K1ₙ + 1ₙK1ₙ", fs=11, a="start"),
          T(10, 216, "3. K̃αₖ = λₖαₖ — собственные векторы", fs=11, a="start"),
          T(10, 236, "4. проекция: zₖ(x) = Σᵢ αₖᵢ k(xᵢ, x)", fs=11, a="start")]
    return svg(p, 340, 248, "Kernel PCA: идея и шаги", s)


# --- label_propagation
def _lp():
    p = "lp"
    rows = [("t=0", [1, .5, .5, .5, .5, 0]), ("t=1", [1, .75, .5, .5, .25, 0]),
            ("t=2", [1, .75, .625, .375, .25, 0]), ("t=∞", [1, .8, .6, .4, .2, 0])]
    s = [T(170, 16, "P(синий) в цепочке; концы размечены (clamp)", fs=11)]
    for r, (lab, vals) in enumerate(rows):
        y = 44 + 48 * r
        s.append(T(10, y + 4, lab, "mut", 11, "start"))
        for i in range(5):
            s.append(E(60 + 50 * i, y, 110 + 50 * i, y, 15, 15))
        for i, v in enumerate(vals):
            c = "c1" if v > 0.5 else ("c4" if v < 0.5 else "c6")
            txt = str(int(v)) if i in (0, 5) else (f"{v:.3f}" if v in (0.625, 0.375) else f"{v:.2f}")
            s.append(N(60 + 50 * i, y, 15, c, txt, 9 if len(txt) > 4 else 10, sw=3 if i in (0, 5) else None))
    s += [T(170, 230, "yᵢ ← Σⱼ wᵢⱼ yⱼ / Σⱼ wᵢⱼ;  размеченные фиксируем", fs=11),
          T(170, 248, "предел — гармоническая функция: 0.8, 0.6, 0.4, 0.2", "mut", 10)]
    return svg(p, 340, 258, "Итерации Label Propagation", s)


# ======================================================================= графики
def ch_glm(fig):
    from sklearn.linear_model import PoissonRegressor, LinearRegression
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 3, 150)
    y = rng.poisson(np.exp(0.3 + 0.8 * x))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=9, alpha=0.45, color=GRAY, label="данные (счётчики)")
    g = np.linspace(0, 3, 200)[:, None]
    pr = PoissonRegressor(alpha=0).fit(x[:, None], y)
    ols = LinearRegression().fit(x[:, None], y)
    ax.plot(g, pr.predict(g), color=RED, lw=2, label="Poisson GLM: μ = exp(Xβ)")
    ax.plot(g, ols.predict(g), color=BLUE, lw=2, ls="--", label="OLS: μ = Xβ")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_title("Log-link держит μ > 0 и ловит рост", fontsize=10)
    ax.legend(fontsize=8)


def ch_gpt(fig):
    z = np.array([3.0, 2.2, 1.5, 0.8, 0.3, -0.5])
    ax = fig.add_subplot(111)
    w = 0.26
    for k, (t, c) in enumerate([(0.5, BLUE), (1.0, GREEN), (2.0, ORANGE)]):
        pr = np.exp(z / t) / np.exp(z / t).sum()
        ax.bar(np.arange(6) + (k - 1) * w, pr, w, color=c, label=f"T = {t}")
    ax.set_xticks(range(6), [f"т{i + 1}" for i in range(6)])
    ax.set_xlabel("токены (по убыванию логита)"), ax.set_ylabel("P(токен)")
    ax.set_title("softmax(z / T): температура генерации", fontsize=10)
    ax.legend(fontsize=8)


def ch_gbh(fig):
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor
    X, y = make_friedman1(500, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = X[:350], X[350:], y[:350], y[350:]
    ax = fig.add_subplot(111)
    for lr, c in [(0.5, RED), (0.1, BLUE), (0.03, GREEN)]:
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=lr, max_depth=3, random_state=0).fit(Xtr, ytr)
        err = [np.mean((yte - p) ** 2) for p in m.staged_predict(Xte)]
        ax.plot(range(1, 301), err, color=c, lw=1.8, label=f"learning_rate = {lr}")
        i = int(np.argmin(err))
        ax.plot(i + 1, err[i], "o", color=c, ms=5)
    ax.set_ylim(0, 12)
    ax.set_xlabel("n_estimators"), ax.set_ylabel("MSE на тесте")
    ax.set_title("Меньше learning_rate → больше деревьев", fontsize=10)
    ax.legend(fontsize=8)


def ch_gbr(fig):
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 6, 120))
    y = np.sin(x) + 0.3 * x + rng.normal(0, 0.2, x.size)
    m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0)
    m.fit(x[:, None], y)
    g = np.linspace(0, 6, 400)[:, None]
    st = list(m.staged_predict(g))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, color=GRAY, alpha=0.5)
    for k, c in [(1, ORANGE), (10, GREEN), (80, BLUE)]:
        ax.plot(g, st[k - 1], color=c, lw=2, label=f"M = {k}")
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_title("F_M(x) после M деревьев (ν = 0.1)", fontsize=10)
    ax.legend(fontsize=8)


def ch_gssl(fig):
    from sklearn.datasets import make_moons
    from sklearn.neighbors import kneighbors_graph
    from sklearn.svm import SVC
    X, y = make_moons(300, noise=0.1, random_state=0)
    W = kneighbors_graph(X, 7, include_self=False).toarray()
    W = np.maximum(W, W.T)
    Lap = np.diag(W.sum(1)) - W
    ns = [1, 2, 3, 5, 10, 20]
    acc_lp, acc_svm = [], []
    for n in ns:
        a1, a2 = [], []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            idx = np.r_[rng.choice(np.where(y == 0)[0], n, replace=False),
                        rng.choice(np.where(y == 1)[0], n, replace=False)]
            u = np.setdiff1d(np.arange(len(y)), idx)
            # гармоническое решение Label Propagation: f_u = L_uu⁻¹ W_ul y_l
            f = np.linalg.solve(Lap[np.ix_(u, u)] + 1e-9 * np.eye(len(u)), W[np.ix_(u, idx)] @ y[idx])
            a1.append(np.mean((f > 0.5) == y[u]))
            a2.append(np.mean(SVC(gamma=1.0).fit(X[idx], y[idx]).predict(X[u]) == y[u]))
        acc_lp.append(np.mean(a1)), acc_svm.append(np.mean(a2))
    ax = fig.add_subplot(111)
    ax.plot(ns, acc_lp, "o-", color=GREEN, lw=2, label="Label Propagation (kNN-граф)")
    ax.plot(ns, acc_svm, "s--", color=GRAY, lw=2, label="SVM только на метках")
    ax.set_xscale("log"), ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("размеченных точек на класс"), ax.set_ylabel("accuracy на неразмеченных")
    ax.set_title("Two moons: граф помогает при малом числе меток", fontsize=10)
    ax.legend(fontsize=8)


def ch_gtm(fig):
    rng = np.random.default_rng(3)
    n = 20
    lab = np.r_[np.zeros(n), np.ones(n)].astype(int)
    P = np.where(lab[:, None] == lab[None, :], 0.35, 0.03)
    A = np.triu((rng.random((2 * n, 2 * n)) < P).astype(float), 1)
    A = A + A.T
    Lap = np.diag(A.sum(1)) - A
    w, V = np.linalg.eigh(Lap)
    f = V[:, 1]
    ax = fig.add_subplot(111)
    for c, col, name in [(0, BLUE, "сообщество 1"), (1, RED, "сообщество 2")]:
        idx = np.where(lab == c)[0]
        ax.scatter(idx, f[idx], color=col, s=22, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("номер узла"), ax.set_ylabel("компонента Fiedler-вектора")
    ax.set_title(f"Знак Fiedler-вектора делит граф (λ₂ = {w[1]:.2f})", fontsize=10)
    ax.legend(fontsize=8)


def ch_hdb(fig):
    from sklearn.cluster import HDBSCAN
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal([0, 0], 0.25, (120, 2)), rng.normal([3, 0.5], 0.7, (150, 2)),
              rng.normal([1.0, 3.2], 0.45, (100, 2)), rng.uniform([-2, -2.5], [6, 5.5], (60, 2))]
    lab = HDBSCAN(min_cluster_size=15, copy=True).fit_predict(X)
    ax = fig.add_subplot(111)
    noise = lab == -1
    ax.scatter(X[noise, 0], X[noise, 1], s=10, color=GRAY, marker="x", label="шум (−1)")
    for k in range(lab.max() + 1):
        m = lab == k
        ax.scatter(X[m, 0], X[m, 1], s=9, color=PAL[[0, 1, 2, 4, 3][k % 5]], label=f"кластер {k}")
    ax.set_title("HDBSCAN: кластеры разной плотности", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([]), ax.set_yticks([])


def ch_ht(fig):
    n = np.logspace(1.5, 4, 200)
    ax = fig.add_subplot(111)
    for d, c in [(0.1, ORANGE), (0.01, BLUE), (1e-4, VIOLET)]:
        ax.plot(n, np.sqrt(np.log(1 / d) / (2 * n)), color=c, lw=2, label=f"δ = {d:g}")
    ax.axhline(0.05, color=RED, ls="--", lw=1.5)
    ax.text(40, 0.058, "ΔG = 0.05", color=RED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("n — примеров в листе"), ax.set_ylabel("ε")
    ax.set_title("Hoeffding bound ε(n), R = 1: split, когда ε < ΔG", fontsize=10)
    ax.legend(fontsize=8)


def ch_hps(fig):
    g = lambda x: np.exp(-((x - 0.37) / 0.07) ** 2)
    xs = np.linspace(0, 1, 400)
    rng = np.random.default_rng(4)
    designs = [("Grid 3×3: всего 3 разных значения x", np.repeat([0.1, 0.5, 0.9], 3)),
               ("Random 9: 9 разных значений x", rng.uniform(0, 1, 9))]
    for k, (title, pts) in enumerate(designs):
        ax = fig.add_subplot(2, 1, k + 1)
        ax.plot(xs, g(xs), color=GRAY, lw=1.5, label="качество от важного x")
        ax.scatter(pts, g(pts), color=[BLUE, ORANGE][k], s=30, zorder=3, label="испытания")
        b = pts[np.argmax(g(pts))]
        ax.scatter([b], [g(b)], s=110, facecolor="none", edgecolor=RED, lw=1.5, zorder=4, label="лучшее")
        ax.set_title(title, fontsize=10), ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("важный гиперпараметр x"), ax.legend(fontsize=7, loc="upper right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def ch_hyp(fig):
    from scipy import stats
    x = np.linspace(-4, 6.5, 500)
    mu1, crit = 2.5, stats.norm.ppf(0.95)
    h0, h1 = stats.norm.pdf(x), stats.norm.pdf(x, mu1)
    ax = fig.add_subplot(111)
    ax.plot(x, h0, color=BLUE, lw=2, label="H₀")
    ax.plot(x, h1, color=GREEN, lw=2, label="H₁")
    ax.fill_between(x, h0, where=x >= crit, color=RED, alpha=0.5, label="α (ошибка I рода)")
    ax.fill_between(x, h1, where=x < crit, color=ORANGE, alpha=0.4, label="β (ошибка II рода)")
    ax.fill_between(x, h1, where=x >= crit, color=GREEN, alpha=0.15,
                    label=f"мощность 1−β = {1 - stats.norm.cdf(crit, mu1):.2f}")
    ax.axvline(crit, color="k", ls="--", lw=1)
    ax.text(crit + 0.1, 0.5, "критическое\nзначение", fontsize=8)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("статистика теста"), ax.set_yticks([])
    ax.set_title("Односторонний тест, α = 0.05", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def ch_imba(fig):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1.2, (500, 2)), rng.normal(2.2, 0.7, (25, 2))]
    y = np.r_[np.zeros(500), np.ones(25)]
    ax = fig.add_subplot(111)
    ax.scatter(*X[y == 0].T, s=6, color=GRAY, alpha=0.5, label="мажоритарный (500)")
    ax.scatter(*X[y == 1].T, s=16, color=RED, label="минорный (25)")
    g = np.linspace(-4, 5, 50)
    for cw, c, ls, name in [(None, BLUE, "--", "без весов"), ("balanced", GREEN, "-", "class_weight='balanced'")]:
        m = LogisticRegression(class_weight=cw).fit(X, y)
        w, b = m.coef_[0], m.intercept_[0]
        ax.plot(g, -(w[0] * g + b) / w[1], color=c, ls=ls, lw=2, label=name)
        rec = np.mean(m.predict(X[y == 1]) == 1)
        print_ = f"{name}: recall {rec:.2f}"
    ax.set_xlim(-4, 5), ax.set_ylim(-4, 5)
    ax.set_title("Веса классов сдвигают границу к мажоритарному", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")


def ch_imbm(fig):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    rng = np.random.default_rng(0)
    y = (rng.random(8000) < 0.02).astype(int)
    X = rng.normal(0, 1, (8000, 5)) + y[:, None] * np.r_[1.5, 1.5, 0, 0, 0]
    Xtr, Xte, ytr, yte = X[:4000], X[4000:], y[:4000], y[4000:]
    s = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, s)
    pr, rc, _ = precision_recall_curve(yte, s)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC-AUC = {roc_auc_score(yte, s):.2f}")
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
    ax.set_xlabel("FPR"), ax.set_ylabel("TPR (recall)"), ax.legend(fontsize=8, loc="lower right")
    ax.set_title("ROC выглядит оптимистично", fontsize=10)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(rc, pr, color=RED, lw=2, label=f"PR-AUC (AP) = {average_precision_score(yte, s):.2f}")
    ax2.axhline(yte.mean(), color=GRAY, ls="--", lw=1, label=f"случайный = доля «+» ≈ {yte.mean():.3f}")
    ax2.set_xlabel("recall"), ax2.set_ylabel("precision"), ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("PR-кривая честнее при ~2% позитивов", fontsize=10)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def ch_il(fig):
    T_ = np.arange(1, 101)
    eps = 0.01
    ax = fig.add_subplot(111)
    ax.plot(T_, eps * T_ ** 2, color=RED, lw=2, label="BC: O(ε·T²)")
    ax.plot(T_, eps * T_, color=GREEN, lw=2, label="DAgger: O(ε·T)")
    ax.set_xlabel("горизонт T (шагов)"), ax.set_ylabel("ожидаемое число ошибок")
    ax.set_title("Накопление ошибок, ε = 0.01 (оценки Ross et al.)", fontsize=10)
    ax.legend(fontsize=8)


def ch_incep(fig):
    phi = np.linspace(0, 7, 100)
    ax = fig.add_subplot(111)
    for base, c, name in [(1.2, BLUE, "depth α^φ (α = 1.2)"), (1.1, GREEN, "width β^φ (β = 1.1)"),
                          (1.15, ORANGE, "resolution γ^φ (γ = 1.15)"),
                          (1.2 * 1.1 ** 2 * 1.15 ** 2, RED, "FLOPS (αβ²γ²)^φ ≈ 2^φ")]:
        ax.plot(phi, base ** phi, color=c, lw=2, label=name)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("φ — коэффициент масштабирования"), ax.set_ylabel("множитель")
    ax.set_title("Compound scaling EfficientNet", fontsize=10)
    ax.legend(fontsize=7)


def ch_im(fig):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    rng = np.random.default_rng(0)
    n, m = 300, 2
    edges = [(0, 1), (1, 2), (0, 2)]
    deg = np.zeros(n)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    for v in range(3, n):
        t = rng.choice(v, m, replace=False, p=deg[:v] / deg[:v].sum())
        for u in t:
            edges.append((v, u)); deg[v] += 1; deg[u] += 1
    E_ = np.array(edges)
    S, p = 300, 0.1
    labs = np.empty((S, n), int)
    sizes = []
    for k in range(S):
        keep = E_[rng.random(len(E_)) < p]
        g = csr_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        labs[k] = lab
        sizes.append(np.bincount(lab, minlength=n))
    sizes = np.array(sizes)
    rows = np.arange(S)[:, None]

    def spread(seq):
        out, cov = [], np.zeros((S, n), bool)
        tot = 0.0
        for v in seq:
            l = labs[:, v]
            new = ~cov[np.arange(S), l]
            tot += np.mean(sizes[np.arange(S), l] * new)
            cov[np.arange(S), l] = True
            out.append(tot)
        return out

    K = 10
    cov = np.zeros((S, n), bool)
    greedy = []
    for _ in range(K):
        gain = (sizes[rows, labs] * ~cov[rows, labs]).mean(0)
        gain[greedy] = -1
        v = int(np.argmax(gain))
        greedy.append(v)
        cov[np.arange(S), labs[:, v]] = True
    ks = np.arange(1, K + 1)
    ax = fig.add_subplot(111)
    ax.plot(ks, spread(greedy), "o-", color=RED, lw=2, label="greedy")
    ax.plot(ks, spread(list(np.argsort(-deg)[:K])), "s--", color=BLUE, lw=2, label="top degree")
    ax.plot(ks, spread(list(rng.choice(n, K, replace=False))), "^:", color=GRAY, lw=2, label="случайные")
    ax.set_xlabel("k — размер seed set"), ax.set_ylabel("σ(S): ожидаемый охват")
    ax.set_title("IC, p = 0.1, граф BA (n = 300)", fontsize=10)
    ax.legend(fontsize=8)


def ch_ib(fig):
    ix = np.linspace(0, 4, 300)
    ax = fig.add_subplot(111)
    for rho, c in [(0.95, BLUE), (0.8, GREEN), (0.6, ORANGE)]:
        iy = -0.5 * np.log(1 - rho ** 2 * (1 - np.exp(-2 * ix)))
        ax.plot(ix, iy, color=c, lw=2, label=f"ρ(X,Y) = {rho}")
        ax.axhline(-0.5 * np.log(1 - rho ** 2), color=c, ls=":", lw=1)
    ax.text(2.2, 0.08, "достижимо под кривой", fontsize=8, color=GRAY)
    ax.set_xlabel("I(T;X), нат — сложность T"), ax.set_ylabel("I(T;Y), нат")
    ax.set_title("IB-кривая (гауссов случай); пунктир — I(X;Y)", fontsize=10)
    ax.legend(fontsize=8, loc="center right")


def ch_ids(fig):
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(0)
    tr = rng.normal(0, 1, (1500, 6))
    te_n = rng.normal(0, 1, (600, 6))
    te_a = rng.normal(0, 1, (60, 6)) + np.r_[3, 2.5, 0, 0, 2, 0]
    iso = IsolationForest(contamination=0.05, random_state=0).fit(tr)
    sn, sa = -iso.score_samples(te_n), -iso.score_samples(te_a)
    thr = -iso.offset_
    ax = fig.add_subplot(111)
    bins = np.linspace(min(sn.min(), sa.min()), max(sn.max(), sa.max()), 35)
    ax.hist(sn, bins, color=BLUE, alpha=0.6, density=True, label="нормальный трафик")
    ax.hist(sa, bins, color=RED, alpha=0.6, density=True, label="атаки")
    ax.axvline(thr, color="k", ls="--", lw=1.2)
    top = ax.get_ylim()[1] * 1.5
    ax.set_ylim(0, top)
    ax.text(thr + 0.004, top * 0.62, "порог\n(5%)", fontsize=8)
    ax.set_xlabel("anomaly score (−score_samples)"), ax.set_ylabel("плотность")
    ax.set_title("Isolation Forest, обучен только на норме", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=1)


def ch_kf(fig):
    rng = np.random.default_rng(2)
    n, dt = 60, 1.0
    F = np.array([[1, dt], [0, 1]]); H = np.array([[1.0, 0]])
    Q = 0.05 * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]); R = np.array([[9.0]])
    tt = np.arange(n)
    xs = 12 * np.sin(2 * np.pi * tt / 60) + 0.1 * tt      # плавно маневрирующая цель
    zs = xs + rng.normal(0, 3, n)
    xh, P = np.array([0.0, 0.0]), np.eye(2) * 10
    est, sd = [], []
    for z in zs:
        xh, P = F @ xh, F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        xh = xh + (K @ (np.array([z]) - H @ xh)).ravel()
        P = (np.eye(2) - K @ H) @ P
        est.append(xh[0]); sd.append(np.sqrt(P[0, 0]))
    t = np.arange(n); est, sd = np.array(est), np.array(sd)
    ax = fig.add_subplot(111)
    ax.scatter(t, zs, s=10, color=GRAY, label="измерения zₖ (σ = 3)")
    ax.plot(t, xs, color="k", lw=1.2, ls="--", label="истинная позиция")
    ax.plot(t, est, color=BLUE, lw=2, label="оценка x̂ₖ")
    ax.fill_between(t, est - 2 * sd, est + 2 * sd, color=BLUE, alpha=0.15, label="±2√Pₖ")
    ax.set_xlabel("шаг k"), ax.set_ylabel("позиция")
    ax.set_title("Калман: модель постоянной скорости", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")


def ch_kcca(fig):
    rng = np.random.default_rng(0)

    def views(n):
        t = rng.uniform(-1, 1, n)
        return (t + rng.normal(0, 0.1, n))[:, None], (t ** 2 + rng.normal(0, 0.05, n))[:, None]

    Xtr, Ytr = views(200)
    Xte, Yte = views(200)
    rbf = lambda A, B_, g: np.exp(-g * (A - B_.T) ** 2)
    n, kap = 200, 0.5
    Hc = np.eye(n) - 1.0 / n
    Kx, Ky = Hc @ rbf(Xtr, Xtr, 2.0) @ Hc, Hc @ rbf(Ytr, Ytr, 2.0) @ Hc
    Rx, Ry = Kx + kap * np.eye(n), Ky + kap * np.eye(n)
    M = np.linalg.solve(Rx, Ky) @ np.linalg.solve(Ry, Kx)
    w, V = np.linalg.eig(M)
    a = np.real(V[:, np.argmax(np.real(w))])
    b = np.linalg.solve(Ry, Kx @ a)

    def proj(Xn, Xt, coef):
        k = rbf(Xn, Xt, 2.0)
        k = k - k.mean(1, keepdims=True) - rbf(Xt, Xt, 2.0).mean(0) + rbf(Xt, Xt, 2.0).mean()
        return k @ coef

    u, v = proj(Xte, Xtr, a), proj(Yte, Ytr, b)
    r_lin = abs(np.corrcoef(Xte[:, 0], Yte[:, 0])[0, 1])
    r_k = abs(np.corrcoef(u, v)[0, 1])
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(Xte, Yte, s=8, color=GRAY)
    ax.set_xlabel("x (вид 1)"), ax.set_ylabel("y (вид 2)")
    ax.set_title(f"Исходные виды: |линейная corr| = {r_lin:.2f}", fontsize=10)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(u * np.sign(np.corrcoef(u, v)[0, 1]), v, s=8, color=VIOLET)
    ax2.set_xlabel("u = f(x)"), ax2.set_ylabel("v = g(y)")
    ax2.set_title(f"Kernel CCA (RBF, тест): corr(u, v) = {r_k:.2f}", fontsize=10)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def ch_kpca(fig):
    from sklearn.datasets import make_circles
    from sklearn.decomposition import PCA, KernelPCA
    X, y = make_circles(300, factor=0.3, noise=0.05, random_state=0)
    for k, (title, Z) in enumerate([("PCA: кольца не разделяются", PCA(2).fit_transform(X)),
                                    ("Kernel PCA (RBF, γ = 10): разделились по PC₁",
                                     KernelPCA(2, kernel="rbf", gamma=10).fit_transform(X))]):
        ax = fig.add_subplot(2, 1, k + 1)
        ax.scatter(*Z[y == 0].T, s=8, color=RED, label="внешнее кольцо")
        ax.scatter(*Z[y == 1].T, s=8, color=BLUE, label="внутреннее кольцо")
        ax.set_title(title, fontsize=10), ax.set_xlabel("PC₁"), ax.set_ylabel("PC₂")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def ch_lp(fig):
    from sklearn.datasets import make_moons
    from sklearn.semi_supervised import LabelPropagation
    X, y = make_moons(300, noise=0.08, random_state=0)
    yl = np.full(len(y), -1)
    i0, i1 = np.where(y == 0)[0][0], np.where(y == 1)[0][0]
    yl[[i0, i1]] = y[[i0, i1]]
    m = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=2000).fit(X, yl)
    pred = m.transduction_
    acc = np.mean(pred[yl == -1] == y[yl == -1])
    ax = fig.add_subplot(111)
    ax.scatter(*X[pred == 0].T, s=9, color=BLUE, alpha=0.6)
    ax.scatter(*X[pred == 1].T, s=9, color=RED, alpha=0.6)
    ax.scatter(*X[[i0]].T, s=180, marker="*", color=BLUE, edgecolor="k", label="метка класса 0")
    ax.scatter(*X[[i1]].T, s=180, marker="*", color=RED, edgecolor="k", label="метка класса 1")
    ax.set_title(f"1 метка на класс → accuracy {acc:.2f} (kNN, k = 7)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xticks([]), ax.set_yticks([])


# ======================================================================= VISUALS
def _sv(title, fn, caption):
    return {"title": title, "svg": fn(), "caption": caption}


def _ch(title, fn, caption):
    return {"title": title, "chart": fn, "caption": caption}


VISUALS = {
    "glm": [
        _sv("Из чего состоит GLM", _glm,
            "Линейный предиктор η = Xβ переводится в среднее μ обратной функцией связи, а y берётся из выбранного "
            "семейства. Пара «семейство + связь» определяет модель."),
        _ch("Poisson-регрессия против OLS", ch_glm,
            "Для счётных данных log-link даёт экспоненциальный рост и всегда положительное μ; прямая OLS уходит в "
            "отрицательные значения слева."),
    ],
    "gnn_applications": [
        _sv("Одни эмбеддинги — три задачи", _gnna,
            "GNN строит эмбеддинги узлов hᵥ передачей сообщений; дальше меняется только голова: классификатор узла, "
            "скор пары узлов или pooling всего графа."),
    ],
    "gpt_architectures": [
        _sv("Decoder-only блок GPT", _gpt,
            "Pre-LN блок: LayerNorm → masked self-attention → residual, затем LayerNorm → FFN → residual, повторённый "
            "N раз. Causal mask разрешает токену t смотреть только на позиции ≤ t."),
        _ch("Температура при генерации", ch_gpt,
            "При T < 1 распределение заостряется к самому вероятному токену, при T > 1 выравнивается — текст "
            "становится разнообразнее, но менее связным."),
    ],
    "gradient_boosting_hyperparameters": [
        _sv("Где действует каждый гиперпараметр", _gbh,
            "learning_rate и n_estimators задают сумму деревьев, subsample/colsample — данные для каждого дерева, "
            "а max_depth, gamma, min_child_weight и reg_* — форму самого дерева."),
        _ch("learning_rate против n_estimators", ch_gbh,
            "Большой шаг за десяток деревьев выходит на плато с худшей ошибкой; меньший шаг сходится медленнее, "
            "зато глубже. Точка — минимум каждой кривой."),
    ],
    "gradient_boosting_regression": [
        _sv("Цикл градиентного бустинга", _gbr,
            "Каждое новое дерево обучается на остатках текущей модели (антиградиенте MSE) и добавляется с "
            "множителем ν."),
        _ch("Как растёт приближение", ch_gbr,
            "Одно дерево даёт грубую ступеньку, десять — общий тренд, восемьдесят — форму зависимости; при "
                    "ещё большем M модель начнёт подгоняться под шум (поможет early stopping)."),
    ],
    "grammatical_inference": [
        _sv("RPNI на маленьком примере", _gi,
            "Из положительных примеров строится дерево префиксов, затем состояния сливаются, пока автомат не "
            "начинает принимать отрицательные примеры. Получается обобщение a⁺."),
    ],
    "graph_algorithms": [
        _sv("Порядок обхода BFS и DFS", _ga,
            "BFS идёт слоями по расстоянию d от A, поэтому находит кратчайшие пути в невзвешенном графе; DFS "
            "уходит вглубь (соседи по алфавиту) и возвращается назад."),
    ],
    "graph_based_ssl": [
        _sv("Метки текут по графу сходства", _gssl,
            "Две известные метки распространяются по плотным связям внутри кластеров; слабое ребро между "
            "кластерами почти не переносит метку."),
        _ch("Сколько меток нужно", ch_gssl,
            "Когда меток мало, графовый метод использует структуру неразмеченных точек и сильно обгоняет "
            "SVM, обученный только на размеченных. Среднее по 8 случайным выборкам."),
    ],
    "graph_kernels": [
        _sv("Одна итерация Weisfeiler-Lehman", _gk,
            "Каждый узел получает новую метку из своей метки и отсортированного мультимножества соседей. "
            "Граф описывается гистограммой меток, а ядро — скалярное произведение гистограмм."),
    ],
    "graph_theory_ml": [
        _sv("Граф в матричном виде", _gtm,
            "Матрица смежности A симметрична для неориентированного графа, степень — сумма строки. Из них строится "
            "лапласиан L = D − A — основа спектральных методов."),
        _ch("Спектральное разбиение", ch_gtm,
            "Знак второго собственного вектора лапласиана (Fiedler) почти идеально разделяет два плотных "
            "сообщества со слабыми связями между ними."),
    ],
    "hdbscan": [
        _sv("Алгоритм HDBSCAN по шагам", _hdb,
            "Расстояния «растягиваются» в разреженных областях (mutual reachability), по ним строится MST и иерархия; "
            "из сжатого дерева выбираются самые стабильные кластеры."),
        _ch("Кластеры разной плотности", ch_hdb,
            "Один min_cluster_size находит и плотный, и рыхлый кластер, а разреженный фон уходит в шум — с одним "
            "eps у DBSCAN так не получится."),
    ],
    "hierarchical_rl": [
        _sv("Менеджер и исполнитель", _hrl,
            "Верхний уровень раз в k шагов выбирает подцель по внешней награде; нижний выдаёт примитивные "
            "действия и учится на внутренней награде за достижение подцели."),
    ],
    "hierarchical_rl_detailed": [
        _sv("Options во времени", _hrld,
            "Опция стартует в состоянии из I, действует своей политикой π и завершается по β. Над опциями "
            "работает SMDP-Q-learning с дисконтом γᵏ за k шагов."),
    ],
    "hoeffding_trees": [
        _sv("Когда VFDT делает split", _ht,
            "Лист копит статистики; раз в grace_period примеров разница выигрышей двух лучших атрибутов сравнивается "
            "с границей Хёффдинга ε. При почти равных атрибутах split делается, когда ε < τ."),
        _ch("Граница Хёффдинга от числа примеров", ch_ht,
            "ε убывает как 1/√n: чем меньше δ (выше уверенность), тем больше примеров нужно, чтобы разница "
            "ΔG = 0.05 стала значимой."),
    ],
    "hugging_face_transformers": [
        _sv("Что происходит внутри pipeline()", _hf,
            "Токенизатор превращает текст в input_ids и attention_mask, модель выдаёт logits, постобработка — "
            "метку и вероятность. Токенизатор и веса берутся из Hub по одному имени модели."),
    ],
    "hybrid_recommender_systems": [
        _sv("Три способа гибридизации", _hyb,
            "Weighted смешивает скоры, Switching выбирает метод по ситуации (например, холодный старт), Cascade "
            "последовательно сужает список кандидатов."),
    ],
    "hyperparameter_strategies": [
        _sv("Successive Halving", _hps,
            "Много конфигураций получают маленький бюджет, лучшая треть переходит дальше с утроенным бюджетом. "
            "Hyperband перебирает несколько таких сеток."),
        _ch("Почему Random лучше Grid", ch_hps,
            "Если важен один параметр, сетка 3×3 проверяет лишь 3 его значения, а 9 случайных точек — 9 разных; "
            "шанс попасть в узкий пик выше."),
    ],
    "hypothesis_testing": [
        _sv("Какой тест выбрать", _hyp,
            "Категориальные признаки — χ²; для чисел выбор зависит от нормальности и от того, связаны ли выборки "
            "(парные измерения, одни и те же фолды CV)."),
        _ch("Ошибки I и II рода", ch_hyp,
            "α — хвост H₀ за критическим значением, β — часть H₁ до него. Сдвиг порога уменьшает одну ошибку "
            "за счёт другой; мощность растёт с размером эффекта и выборки."),
    ],
    "image_classification_dl": [
        _sv("Transfer learning в два этапа", _icd,
            "Сначала обучается только новая голова поверх замороженного backbone, затем размораживаются верхние "
            "блоки и всё дообучается с малым learning rate."),
    ],
    "image_text_models": [
        _sv("Контрастное обучение CLIP", _itm,
            "Два энкодера дают эмбеддинги картинок и подписей; в матрице сходств батча правильные пары лежат на "
            "диагонали, и InfoNCE тянет их вверх, а остальные вниз."),
    ],
    "image_to_image_translation": [
        _sv("Pix2Pix: G и PatchGAN", _i2i,
            "U-Net переводит x в G(x); дискриминатор видит пару (x, y) или (x, G(x)) и выдаёт карту real/fake по "
            "патчам. L1-слагаемое привязывает результат к целевому y."),
    ],
    "imbalanced_algorithms": [
        _sv("Easy Ensemble", _imba,
            "Мажоритарный класс случайно делится на подвыборки размера N₊, каждая объединяется со всеми "
            "минорными объектами; ансамбль моделей усредняется, и данные не теряются."),
        _ch("class_weight='balanced'", ch_imba,
            "Без весов граница прижимается к минорному классу и почти все его объекты теряются; веса сдвигают её, "
            "повышая recall ценой лишних срабатываний."),
    ],
    "imbalanced_metrics": [
        _sv("Матрица ошибок и метрики", _imbm,
            "Precision считается по столбцу предсказанных «+», recall — по строке истинных «+». При 1% позитивов "
            "accuracy 0.992 скрывает, что 30% позитивов пропущено."),
        _ch("ROC против PR при дисбалансе", ch_imbm,
            "Высокий ROC-AUC держится на огромном числе TN; PR-кривая показывает, что precision быстро падает. "
            "Базовая линия PR равна доле позитивов, а не 0.5."),
    ],
    "imitation_learning": [
        _sv("BC и цикл DAgger", _il,
            "Behavioral Cloning учится только на демонстрациях. DAgger запускает обученную политику, просит "
            "эксперта разметить её собственные состояния и добавляет их в D."),
        _ch("Почему BC накапливает ошибки", ch_il,
            "Теоретические оценки: ошибка BC растёт квадратично по горизонту из-за сдвига распределения, у DAgger — "
            "линейно."),
    ],
    "inception_efficientnet": [
        _sv("Inception-модуль", _incep,
            "Четыре параллельные ветви ловят паттерны разного масштаба и склеиваются по каналам; 1×1 свёртки перед "
            "дорогими 3×3 и 5×5 снижают вычисления."),
        _ch("Compound scaling", ch_incep,
            "Глубина, ширина и разрешение растут вместе по степеням φ; при α·β²·γ² ≈ 2 каждый шаг φ примерно "
            "удваивает FLOPS."),
    ],
    "incremental_learning": [
        _sv("Почему EWC не забывает", _inc,
            "Обычный SGD уходит в минимум задачи B и покидает область низкого loss A. EWC штрафует сдвиг важных "
            "по Fisher весов и находит точку, хорошую для обеих задач."),
    ],
    "inductive_logic_programming": [
        _sv("ILP на примере grandparent", _ilp,
            "Из фоновых фактов и примеров ILP ищет правило, покрывающее E⁺ и не покрывающее E⁻: начинает с общего "
            "клоза и добавляет литералы, пока негативы не исключены."),
    ],
    "influence_maximization": [
        _sv("Independent Cascade", _im,
            "Каждый недавно активированный узел один раз пытается активировать каждого соседа с вероятностью "
            "ребра; процесс идёт, пока появляются новые активные."),
        _ch("Greedy против эвристик", ch_im,
            "Greedy по Монте-Карло-оценке σ(S) не хуже выбора хабов и намного лучше случайных узлов; на BA-графе "
            "хабы — сильная эвристика. Каждый следующий seed даёт меньший прирост (субмодулярность)."),
    ],
    "information_bottleneck_theory": [
        _sv("Цель Information Bottleneck", _ib,
            "Представление T должно забыть как можно больше об X и сохранить как можно больше о Y; β задаёт "
            "компромисс между сжатием и предсказанием."),
        _ch("Кривая IB", ch_ib,
            "Для гауссовых X, Y кривая известна точно: I(T;Y) растёт с I(T;X) и насыщается на I(X;Y). Точки над "
            "кривой недостижимы ни одним представлением."),
    ],
    "intrusion_detection": [
        _sv("Конвейер обнаружения вторжений", _ids,
            "Из трафика извлекаются признаки потоков; сигнатурная ветка ловит известные атаки, ML-модель нормы — "
            "новые отклонения, а гибридная оценка решает, поднимать ли alert."),
        _ch("Anomaly-based детектор", ch_ids,
            "Isolation Forest обучен только на нормальном трафике; атаки получают высокий anomaly score, но хвосты "
            "перекрываются — порог задаёт баланс FP и FN."),
    ],
    "inverse_rl": [
        _sv("Цикл IRL (feature matching)", _irl,
            "Награда ищется как wᵀφ(s): под текущую R решается RL, ожидания признаков политики сравниваются с "
            "экспертными, и w сдвигается в сторону эксперта."),
    ],
    "jax_flax": [
        _sv("Преобразования функций и Flax", _jax,
            "JAX превращает чистую функцию в новую: градиент, скомпилированную, векторизованную, параллельную — "
            "и их можно комбинировать. Во Flax параметры живут отдельно от кода модели."),
    ],
    "kalman_filters": [
        _sv("Цикл predict → update", _kf,
            "Прогноз двигает оценку моделью F и увеличивает неопределённость на Q; измерение zₖ корректирует её "
            "с весом Kₖ, который больше, когда R мало относительно P."),
        _ch("Фильтр на зашумлённых измерениях", ch_kf,
            "Оценка проходит гораздо ближе к истинной траектории, чем сырые измерения, а коридор ±2σ быстро "
            "сужается после старта с большим P₀."),
    ],
    "kernel_cca": [
        _sv("Схема Kernel CCA", _kcca,
            "Каждый вид проходит через свою матрицу ядра; ищутся коэффициенты α и β, при которых проекции u и v "
            "максимально коррелированы. Регуляризация κ спасает от тривиального ρ = 1."),
        _ch("Нелинейная связь двух видов", ch_kcca,
            "Связь y ≈ x² не видна линейной корреляцией, а Kernel CCA находит f и g, сильно коррелированные даже "
            "на тестовых данных."),
    ],
    "kernel_pca": [
        _sv("Kernel PCA: идея и шаги", _kpca,
            "Вместо ковариации раскладывается центрированная матрица ядра; концентрические кольца, неразделимые "
            "линейно, расходятся по первой компоненте (точки посчитаны KernelPCA)."),
        _ch("PCA против Kernel PCA", ch_kpca,
            "Линейный PCA лишь поворачивает кольца; RBF Kernel PCA разносит внутреннее и внешнее кольцо по PC₁."),
    ],
    "label_propagation": [
        _sv("Итерации на цепочке", _lp,
            "Каждый узел берёт среднее меток соседей, размеченные концы зафиксированы. Значения сходятся к "
            "линейной интерполяции 0.8 … 0.2."),
        _ch("Two moons с двумя метками", ch_lp,
            "По kNN-графу метка проходит вдоль каждой «луны», поэтому одной метки на класс хватает почти для "
            "всех точек."),
    ],
}
