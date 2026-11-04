# -*- coding: utf-8 -*-
"""Рисунки для шпаргалок группы g5 (multi_task_learning … optimizers_advanced)."""

import numpy as np

PAL = ["#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"]
BLUE, GREEN, ORANGE, RED, PURPLE, GREY = PAL


# ---------------------------------------------------------------- помощники SVG
def _e(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, cls="", a="middle", fs=None, extra=""):
    c = f' class="{cls}"' if cls else ""
    f = f' font-size="{fs}"' if fs else ""
    return f'<text x="{x:g}" y="{y:g}" text-anchor="{a}"{c}{f}{extra}>{_e(s)}</text>'


def R(x, y, w, h, cls, rx=6, extra=""):
    return f'<rect class="{cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}"{extra}/>'


def B(x, y, w, h, cls, *lines, fs=12, rx=6):
    """Прямоугольник с центрированными строками. Строка — str или (str, класс, размер)."""
    out = [R(x, y, w, h, cls, rx)]
    items = [(l, "", fs) if isinstance(l, str) else l for l in lines]
    lh = max(i[2] for i in items) + 4 if items else 0
    y0 = y + h / 2 - (len(items) - 1) * lh / 2
    for k, (s, c, f) in enumerate(items):
        out.append(T(x + w / 2, y0 + k * lh + f * 0.36, s, c, fs=f if f != 12 else None))
    return "".join(out)


def C(cx, cy, r, cls, label=None, fs=None):
    out = f'<circle class="{cls}" cx="{cx:g}" cy="{cy:g}" r="{r:g}"/>'
    if label is not None:
        f = fs or 12
        out += T(cx, cy + f * 0.36, label, fs=fs)
    return out


def A(p, x1, y1, x2, y2, cls="ln", dash=False, both=False, head=True):
    d = ' stroke-dasharray="4 3"' if dash else ""
    m = f' marker-end="url(#{p}-a)"' if head else ""
    if both:
        m += f' marker-start="url(#{p}-a)"'
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"{d}{m}/>'


def P(p, d, cls="ln", dash=False, head=True):
    da = ' stroke-dasharray="4 3"' if dash else ""
    m = f' marker-end="url(#{p}-a)"' if head else ""
    return f'<path class="{cls}" d="{d}"{da}{m}/>'


def L(x1, y1, x2, y2, cls="ln", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"{d}/>'


def svg(p, w, h, label, *parts):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_e(label)}">'
            f'<defs><marker id="{p}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            + "".join(parts) + "</svg>")


# ================================================================ multi_task_learning
p = "mtl"
MTL = svg(p, 360, 262, "Hard parameter sharing",
          B(15, 107, 52, 36, "c6", "x"),
          A(p, 67, 125, 98, 125),
          B(100, 87, 90, 76, "c1", "Общий", "backbone", "θ_shared"),
          T(145, 183, "градиенты", "mut", fs=10), T(145, 197, "всех задач", "mut", fs=10),
          A(p, 190, 125, 220, 45), A(p, 190, 125, 220, 125), A(p, 190, 125, 220, 205),
          B(222, 27, 78, 36, "c2", "голова θ₁"),
          B(222, 107, 78, 36, "c3", "голова θ₂"),
          B(222, 187, 78, 36, "c5", "голова θ_K"),
          A(p, 300, 45, 318, 45), A(p, 300, 125, 318, 125), A(p, 300, 205, 318, 205),
          T(322, 49, "L₁", a="start"), T(322, 129, "L₂", a="start"), T(322, 209, "L_K", a="start"),
          T(261, 170, "⋮", "mut"),
          T(180, 250, "L = Σₖ wₖ · Lₖ(θ_shared, θₖ)"))

# ================================================================ multi_task_learning_deep_dive
p = "mtld"
O = (150, 180)
MTLD = svg(p, 340, 262, "PCGrad: проекция конфликтующего градиента",
           T(170, 18, "PCGrad: конфликт g₁ · g₂ < 0"),
           L(175.5, 69.5, 139.5, 225.5, "ln s6", dash=True),
           T(134, 224, "⊥ g₂", "mut", a="end", fs=11),
           A(p, 150, 180, 90, 70, "ln s1"),
           A(p, 150, 180, 280, 210, "ln s2"),
           A(p, 150, 180, 171, 89, "ln s5"),
           L(90, 70, 171, 89, "ln s6", dash=True),
           T(80, 64, "g₁", "t1", a="end"), T(288, 216, "g₂", "t2", a="start"),
           T(180, 84, "g₁′", "t5", a="start"),
           T(126, 68, "−proj", "mut", fs=10),
           C(150, 180, 3, "ah"),
           T(170, 252, "g₁′ = g₁ − (g₁·g₂ / ‖g₂‖²) · g₂"))

# ================================================================ multi_view_learning
p = "mvl"
MVL = svg(p, 360, 215, "Co-training",
          B(15, 70, 80, 40, "c3", "L", ("с метками", "", 11)),
          B(135, 28, 100, 36, "c1", "h₁ ( x⁽¹⁾ )"),
          B(135, 116, 100, 36, "c2", "h₂ ( x⁽²⁾ )"),
          A(p, 95, 82, 133, 50), A(p, 95, 98, 133, 132),
          T(110, 58, "обучение", "mut", fs=10, a="end"),
          B(268, 72, 80, 36, "c6", ("прогноз", "", 11), ("на U", "", 11)),
          A(p, 235, 46, 268, 82), A(p, 235, 134, 268, 98),
          P(p, "M308,108 V190 H55 V112", "ln s3"),
          T(180, 205, "топ-k уверенных → псевдо-метки в L", "mut", fs=11),
          T(185, 94, "view 1 ≠ view 2", "mut", fs=10))

# ================================================================ multimodal_transformers
p = "mmt"
MMT = svg(p, 360, 250, "Cross-modal attention",
          B(15, 205, 140, 32, "c1", "Текст: токены"),
          B(205, 205, 140, 32, "c2", "Изображение: патчи"),
          B(50, 145, 70, 28, "c1", "W_Q"),
          B(240, 145, 70, 28, "c2", "W_K, W_V"),
          A(p, 85, 205, 85, 175), A(p, 275, 205, 275, 175),
          B(90, 75, 180, 36, "c5", "softmax(QKᵀ/√d)·V"),
          A(p, 85, 145, 140, 113), A(p, 275, 145, 220, 113),
          T(100, 130, "Q", "t1", a="end"), T(262, 130, "K, V", "t2", a="start"),
          B(90, 15, 180, 30, "c6", "Add & Norm → FFN"),
          A(p, 180, 75, 180, 47),
          P(p, "M30,205 V30 H88", "ln s6", dash=True),
          T(36, 120, "residual", "mut", fs=10, a="start"),
          T(312, 27, "текст,", "mut", fs=10), T(312, 41, "знающий", "mut", fs=10),
          T(312, 55, "картинку", "mut", fs=10))

# ================================================================ multiple_regression
p = "mreg"
MREG = svg(p, 340, 260, "Геометрия OLS",
           '<polygon class="c6" points="30,190 230,230 320,160 120,120"/>',
           T(272, 200, "span(X)", "mut", fs=11),
           A(p, 90, 175, 150, 205, "ln s6"), A(p, 90, 175, 150, 140, "ln s6"),
           T(156, 214, "x₁", "mut", a="start", fs=11), T(150, 134, "x₂", "mut", fs=11),
           A(p, 90, 175, 189, 47, "ln s1"),
           A(p, 90, 175, 188, 175, "ln s2"),
           L(190, 175, 190, 45, "ln s4", dash=True),
           '<polyline class="ln" points="190,163 201.8,165.4 201.8,177.4"/>',
           C(190, 45, 3, "t1"), C(190, 175, 3, "t2"),
           T(198, 45, "y", "t1", a="start"),
           T(196, 112, "e = y − ŷ", "t4", a="start"),
           T(184, 192, "ŷ = Xβ̂", "t2", a="middle"),
           T(170, 250, "Xᵀ(y − Xβ̂) = 0 ⇒ β̂ = (XᵀX)⁻¹Xᵀy"))

# ================================================================ multitask_rl
p = "mtrl"
MTRL = svg(p, 360, 258, "Distral",
           C(180, 120, 32, "c5", "π₀", fs=16),
           T(220, 150, "дистиллированная", "mut", a="start", fs=10),
           B(20, 40, 90, 40, "c1", "π₁", ("задача 1", "", 11)),
           B(250, 40, 90, 40, "c2", "π₂", ("задача 2", "", 11)),
           B(135, 190, 90, 40, "c3", "π_K", ("задача K", "", 11)),
           A(p, 110, 70, 153, 103, "ln s6", dash=True, both=True),
           A(p, 250, 70, 207, 103, "ln s6", dash=True, both=True),
           A(p, 180, 190, 180, 154, "ln s6", dash=True, both=True),
           T(122, 100, "KL", "mut", a="end", fs=11), T(238, 100, "KL", "mut", a="start", fs=11),
           T(188, 176, "KL", "mut", a="start", fs=11),
           T(180, 250, "πᵢ: max E[Rᵢ] − α·KL(πᵢ ‖ π₀)"))

# ================================================================ multivariate_time_series
p = "mvts"


def _vars_cols():
    out = []
    for x, lab in [(70, "t−2"), (180, "t−1"), (290, "t")]:
        out.append(T(x, 25, lab, "mut", fs=11))
        out.append(C(x, 62, 22, "c1", "x₁"))
        out.append(C(x, 150, 22, "c2", "x₂"))
    for x0, last in [(70, False), (180, True)]:
        x1 = x0 + 110
        c_cross = ("ln s3", "ln s5") if last else ("ln s6", "ln s6")
        out.append(A(p, x0 + 22, 62, x1 - 24, 62))
        out.append(A(p, x0 + 22, 150, x1 - 24, 150))
        out.append(A(p, x0 + 17, 76, x1 - 19, 136, c_cross[0]))   # x1 → x2
        out.append(A(p, x0 + 17, 136, x1 - 19, 76, c_cross[1]))   # x2 → x1
    out += [T(235, 54, "a₁₁", "mut", fs=11), T(235, 168, "a₂₂", "mut", fs=11),
            T(240, 143, "a₂₁", "t3", fs=11), T(240, 77, "a₁₂", "t5", fs=11)]
    return "".join(out)


MVTS = svg(p, 360, 232, "VAR(1)",
           _vars_cols(),
           T(180, 200, "xₜ = c + A₁xₜ₋₁ + … + Aₚxₜ₋ₚ + εₜ"),
           T(180, 220, "A₁ = [[a₁₁, a₁₂], [a₂₁, a₂₂]]", "mut", fs=11))

# ================================================================ musical_neural_networks
p = "rvq"
RVQ = svg(p, 360, 318, "Residual vector quantization",
          T(60, 56, "x"), A(p, 72, 52, 118, 52),
          B(120, 36, 110, 32, "c1", "кодбук 1"),
          A(p, 230, 52, 262, 52), T(268, 56, "q₁ (k₁)", a="start"),
          A(p, 60, 62, 60, 87), C(60, 100, 11, "c6", "−"),
          P(p, "M175,68 V100 H73"),
          A(p, 60, 111, 60, 134), T(60, 150, "r₁"),
          A(p, 72, 146, 118, 146),
          B(120, 130, 110, 32, "c2", "кодбук 2"),
          A(p, 230, 146, 262, 146), T(268, 150, "q₂ (k₂)", a="start"),
          A(p, 60, 156, 60, 181), C(60, 194, 11, "c6", "−"),
          P(p, "M175,162 V194 H73"),
          A(p, 60, 205, 60, 228), T(60, 244, "r₂"),
          A(p, 72, 240, 118, 240),
          B(120, 224, 110, 32, "c3", "кодбук 3"),
          A(p, 230, 240, 262, 240), T(268, 244, "q₃ (k₃)", a="start"),
          T(180, 288, "x̂ = q₁ + q₂ + q₃ ≈ x"),
          T(180, 308, "токены кадра: (k₁, k₂, k₃)", "mut", fs=11))

# ================================================================ naive_bayes_bayesian_method
p = "nbb"
_nb = [C(170, 45, 24, "c3", "y")]
for cx, lab in [(50, "x₁"), (120, "x₂"), (190, "x₃"), (290, "xₙ")]:
    dx, dy = cx - 170, 150 - 45
    n_ = (dx * dx + dy * dy) ** 0.5
    _nb.append(A(p, 170 + 24 * dx / n_, 45 + 24 * dy / n_, cx - 24 * dx / n_, 150 - 24 * dy / n_))
    _nb.append(C(cx, 150, 22, "c1", lab))
NBB = svg(p, 340, 228, "Наивный Байес как граф",
          *_nb, T(240, 154, "…"),
          T(202, 40, "prior P(y)", "mut", a="start", fs=11),
          T(88, 95, "P(xᵢ|y)", "mut", a="end", fs=11),
          T(170, 194, "xᵢ независимы при условии y", "mut", fs=11),
          T(170, 216, "P(y|x) ∝ P(y) · ∏ᵢ P(xᵢ|y)"))

# ================================================================ nas
p = "nas"
_ops = ["conv 3×3", "max pool", "skip", "none"]
_nas = []
for k, op in enumerate(_ops):
    y = 190 + 32 * k
    _nas.append(B(130, y, 100, 24, ["c1", "c2", "c3", "c6"][k], f"p{'₁₂₃₄'[k]}·{op}", fs=11))
    _nas.append(A(p, 58, 250, 128, y + 12))
    _nas.append(L(230, y + 12, 259, 250))
NAS = svg(p, 360, 362, "NAS и DARTS",
          B(15, 15, 120, 40, "c6", "Search space", ("варианты A", "mut", 11)),
          B(225, 15, 120, 40, "c1", "Search strategy", ("RL, эволюция, ∇", "mut", 11)),
          B(225, 95, 120, 40, "c2", "Performance", "estimation"),
          A(p, 135, 35, 223, 35),
          A(p, 265, 55, 265, 93), T(258, 79, "архитектура a", "mut", a="end", fs=10),
          A(p, 305, 95, 305, 57), T(312, 79, "оценка", "mut", a="start", fs=10),
          T(180, 172, "DARTS: смешанная операция на ребре", fs=12),
          C(40, 250, 18, "c5", "xᵢ"), C(322, 250, 18, "c5", "xⱼ"),
          *_nas,
          C(270, 250, 11, "c6", "Σ"), A(p, 281, 250, 302, 250),
          T(180, 335, "x̄ⱼ = Σₒ softmax(α)ₒ · o(xᵢ)"),
          T(180, 353, "после поиска: оставить argmax α", "mut", fs=11))

# ================================================================ neural_architecture_patterns
p = "nap"
NAP = svg(p, 340, 292, "Bottleneck residual block",
          T(150, 20, "x (256 каналов)"),
          A(p, 150, 26, 150, 40),
          B(75, 40, 150, 30, "c1", "1×1 conv: 256→64"),
          A(p, 150, 70, 150, 93),
          B(75, 95, 150, 30, "c2", "3×3 conv: 64→64"),
          A(p, 150, 125, 150, 148),
          B(75, 150, 150, 30, "c1", "1×1 conv: 64→256"),
          A(p, 150, 180, 150, 200),
          C(150, 215, 13, "c3", "+"),
          P(p, "M150,31 H290 V215 H165", "ln s3"),
          T(258, 125, "identity", "t3", fs=11),
          T(72, 59, "сжатие", "mut", a="end", fs=10),
          T(72, 114, "обработка", "mut", a="end", fs=10),
          T(72, 169, "расширение", "mut", a="end", fs=10),
          A(p, 150, 228, 150, 255),
          T(150, 272, "y = ReLU(F(x) + x)"))

# ================================================================ neural_network_serialization
p = "nns"
_fmts = [("c2", "state_dict (.pt)", "только веса, нужен код"),
         ("c4", "torch.save(model)", "весь объект через pickle"),
         ("c1", "TorchScript (.pt)", "граф без Python, C++"),
         ("c3", "ONNX (.onnx)", "кросс-фреймворк, ORT")]
_nns = []
for k, (c, a_, b_) in enumerate(_fmts):
    y = 15 + 60 * k
    _nns.append(B(165, y, 180, 44, c, a_, (b_, "mut", 11)))
    _nns.append(A(p, 110, 135, 163, y + 22))
NNS = svg(p, 360, 284, "Форматы сохранения",
          B(15, 110, 95, 50, "c5", "nn.Module", ("в памяти", "mut", 11)),
          *_nns,
          T(180, 262, "в checkpoint: + optimizer, epoch,", "mut", fs=11),
          T(180, 277, "препроцессинг и метаданные", "mut", fs=11))


# ================================================================ neural_networks_astronomy
def _lc(x0, x1, y0, dips, depth=14, w=7):
    pts = []
    for x in range(x0, x1 + 1, 2):
        y = y0
        for d in dips:
            if abs(x - d) < w:
                y = y0 + depth * (1 - ((x - d) / w) ** 2)
        pts.append(f"{x},{y:.1f}")
    return f'<polyline class="ln s1" points="{" ".join(pts)}"/>'


p = "astr"
ASTR = svg(p, 360, 322, "Поиск транзитов",
           T(180, 16, "кривая блеска звезды", "mut", fs=11),
           _lc(30, 330, 32, [80, 180, 280]),
           A(p, 180, 52, 180, 78),
           B(60, 80, 240, 30, "c1", "detrend + поиск периода (BLS)"),
           A(p, 180, 110, 180, 128),
           B(60, 130, 240, 30, "c6", "phase folding по периоду P"),
           _lc(110, 250, 180, [180], depth=18, w=22),
           T(253, 196, "глубина ≈ (Rp/R*)²", "mut", a="start", fs=10),
           A(p, 180, 204, 180, 223),
           B(60, 225, 240, 30, "c5", "1D CNN (global + local view)"),
           A(p, 150, 255, 100, 278), A(p, 210, 255, 260, 278),
           B(40, 280, 120, 30, "c2", "планета"),
           B(200, 280, 120, 30, "c4", "ложное: EB, шум"))

# ================================================================ neural_networks_biology
p = "bio"
BIO = svg(p, 360, 298, "AlphaFold2",
          B(60, 10, 240, 30, "c6", "аминокислотная последовательность"),
          A(p, 140, 40, 90, 68), A(p, 220, 40, 270, 68),
          B(15, 70, 150, 36, "c1", "MSA: гомологи"),
          B(195, 70, 150, 36, "c3", "pair: пары (i, j)"),
          A(p, 90, 106, 90, 133), A(p, 270, 106, 270, 133),
          B(15, 135, 330, 50, "c5", "Evoformer × 48", ("обмен MSA ↔ pair", "mut", 11)),
          A(p, 180, 185, 180, 213),
          B(60, 215, 240, 34, "c2", "Structure module (IPA)"),
          P(p, "M60,232 H25 V187", "ln s6", dash=True),
          T(30, 207, "recycling", "mut", a="start", fs=10),
          A(p, 180, 249, 180, 266),
          T(180, 284, "3D-координаты атомов + pLDDT"))

# ================================================================ neural_networks_chemistry
p = "chem"
_atoms = {"a": (40, 80, "C"), "v": (95, 125, "C"), "o": (95, 195, "O"), "c": (150, 80, "C"), "n": (205, 125, "N")}
_bonds = [("a", "v"), ("v", "o"), ("v", "c"), ("c", "n")]
_ch = []
for u, w in _bonds:
    (x1, y1, _), (x2, y2, _) = _atoms[u], _atoms[w]
    _ch.append(L(x1, y1, x2, y2))
_ch.append(L(91, 155, 91, 165, "ln"))  # двойная связь C=O (вторая черта)
for u in ["a", "o", "c"]:  # сообщения от соседей к v
    x1, y1, _ = _atoms[u]
    x2, y2, _ = _atoms["v"]
    dx, dy = x2 - x1, y2 - y1
    n_ = (dx * dx + dy * dy) ** 0.5
    _ch.append(A(p, x1 + 19 * dx / n_, y1 + 19 * dy / n_, x2 - 19 * dx / n_, y2 - 19 * dy / n_, "ln s3"))
for k, (x, y, s) in _atoms.items():
    _ch.append(C(x, y, 16, "c3" if k == "v" else "c6", s))
CHEM = svg(p, 360, 275, "Message passing на молекуле",
           *_ch[0:4], *_ch[5:],
           T(118, 125, "h_v", "t3", a="start", fs=11),
           T(103, 165, "e_uv", "mut", a="start", fs=10),
           T(150, 45, "соседи шлют сообщения", "mut", fs=10),
           B(250, 103, 100, 44, "c2", "readout Σ h_v", ("→ свойство ŷ", "", 11), fs=11),
           A(p, 223, 125, 248, 125),
           T(180, 238, "m_v = Σ_{u∈N(v)} M(h_v, h_u, e_uv)"),
           T(180, 258, "h_v ← U(h_v, m_v),  × T слоёв"))

# ================================================================ neural_networks_medicine
p = "med"
_tiles = "".join(R(x, 32, 26, 26, "c3", 3) for x in (125, 160, 195, 230))
MED = svg(p, 360, 292, "Attention-MIL для срезов",
          R(20, 15, 60, 60, "c6", 4),
          L(40, 15, 40, 75, "ln s6"), L(60, 15, 60, 75, "ln s6"),
          L(20, 35, 80, 35, "ln s6"), L(20, 55, 80, 55, "ln s6"),
          T(50, 92, "WSI", "mut", fs=11),
          A(p, 84, 45, 120, 45),
          T(195, 24, "тайлы 256×256", "mut", fs=10), _tiles,
          T(272, 50, "…"), T(298, 50, "× N", "mut", a="start", fs=11),
          A(p, 195, 60, 195, 83),
          B(70, 85, 220, 28, "c1", "CNN-энкодер тайлов"),
          A(p, 180, 113, 180, 127),
          T(180, 143, "h₁, h₂, …, h_N"),
          A(p, 180, 149, 180, 161),
          B(70, 163, 220, 28, "c5", "attention-веса aᵢ, Σ aᵢ = 1"),
          A(p, 180, 191, 180, 206),
          B(70, 208, 220, 28, "c2", "z = Σᵢ aᵢ · hᵢ"),
          A(p, 180, 236, 180, 251),
          B(70, 253, 220, 28, "c3", "классификатор → диагноз"))

# ================================================================ neural_networks_physics
p = "pinn"
PINN = svg(p, 360, 280, "PINN",
           B(15, 100, 60, 34, "c6", "(x, t)"),
           A(p, 75, 117, 98, 117),
           B(100, 90, 80, 54, "c1", "нейросеть", "u_θ(x, t)"),
           A(p, 180, 117, 208, 117),
           B(210, 90, 135, 54, "c5", "autodiff:", "u_t, u_x, u_xx"),
           A(p, 140, 144, 100, 183), A(p, 277, 144, 270, 183),
           B(15, 185, 160, 40, "c2", "L_data", ("u ≈ наблюдения", "mut", 11)),
           B(195, 185, 150, 40, "c3", "L_PDE", ("невязка ‖N[u]‖²", "mut", 11)),
           T(180, 22, "точки коллокации (x, t) — без меток", "mut", fs=11),
           T(180, 250, "L = L_data + λ·L_PDE (+ L_BC, L_IC)"),
           T(180, 270, "напр. N[u] = u_t + u·u_x − ν·u_xx", "mut", fs=11))

# ================================================================ neural_tangent_kernel
p = "ntk"
NTKS = svg(p, 360, 250, "NTK и lazy training",
           T(90, 20, "ядро = скалярное", "mut", fs=11), T(90, 36, "произведение ∇θ f", "mut", fs=11),
           A(p, 50, 170, 140, 70, "ln s1"), A(p, 50, 170, 165, 140, "ln s2"),
           C(50, 170, 3, "ah"),
           T(120, 62, "∇θ f(x)", "t1", a="end", fs=11), T(150, 160, "∇θ f(x′)", "t2", a="middle", fs=11),
           '<path class="ln s6" d="M75,142 A35,35 0 0 1 84,159"/>',
           L(185, 15, 185, 195, "ln s6", dash=True),
           T(270, 20, "пространство весов θ", "mut", fs=11),
           '<circle class="ln s1" cx="250" cy="120" r="24" stroke-dasharray="3 3"/>',
           C(250, 120, 3, "ah"), T(250, 112, "θ₀", fs=11),
           P(p, "M250,120 q6,8 12,6 t10,6", "ln s1"),
           P(p, "M250,120 q20,-10 35,-35 t40,-35", "ln s4"),
           T(250, 162, "широкая: ‖θ−θ₀‖→0", "t1", fs=10),
           T(335, 76, "узкая", "t4", a="middle", fs=10),
           T(180, 218, "Θ(x, x′) = ⟨∇θ f(x; θ), ∇θ f(x′; θ)⟩"),
           T(180, 240, "f(x; θ) ≈ f(x; θ₀) + ∇θ f(x; θ₀)ᵀ(θ − θ₀)", fs=11))

# ================================================================ neuro_symbolic_ai_systems
p = "nsy"
NSY = svg(p, 360, 268, "Нейро-символьный конвейер",
          B(90, 15, 180, 32, "c6", "сырые данные (пиксели)"),
          A(p, 180, 47, 180, 70),
          B(90, 72, 180, 42, "c1", "нейросеть: восприятие", ("P(символ | вход)", "mut", 11)),
          A(p, 180, 114, 180, 133),
          B(90, 135, 180, 32, "c3", "символы и факты"),
          A(p, 180, 167, 180, 188),
          B(90, 190, 180, 42, "c5", "символьный вывод", ("правила, логика", "mut", 11)),
          A(p, 180, 232, 180, 246),
          T(180, 262, "ответ + объяснение (цепочка правил)", fs=11),
          P(p, "M270,211 H320 V93 H272", "ln s6", dash=True),
          T(334, 152, "обратная связь", "mut", fs=10, extra=' transform="rotate(-90 334 152)"'))

# ================================================================ neuromorphic_computing
p = "nmc"
_cores = "".join(B(x, y, 54, 40, "c2", ("нейроны", "", 10), ("+ память", "", 10))
                 for x, y in [(200, 40), (286, 40), (200, 120), (286, 120)])
NMC = svg(p, 360, 228, "Von Neumann и нейроморфная архитектура",
          T(90, 22, "Von Neumann"), T(270, 22, "Нейроморфная"),
          B(30, 40, 120, 40, "c1", "CPU (такт)"),
          B(30, 140, 120, 40, "c6", "Память"),
          A(p, 55, 82, 55, 138, both=True), A(p, 125, 82, 125, 138, both=True),
          T(90, 107, "узкое", "t4", fs=11), T(90, 121, "место", "t4", fs=11),
          T(90, 202, "тактово, синхронно,", "mut", fs=10),
          T(90, 216, "энергия на пересылку", "mut", fs=10),
          L(180, 30, 180, 215, "ln s6", dash=True),
          _cores,
          A(p, 255, 60, 285, 60, "ln s3", dash=True, both=True),
          A(p, 255, 140, 285, 140, "ln s3", dash=True, both=True),
          A(p, 227, 81, 227, 119, "ln s3", dash=True, both=True),
          A(p, 313, 81, 313, 119, "ln s3", dash=True, both=True),
          T(270, 104, "спайки", "t3", fs=10),
          T(270, 202, "асинхронно, event-driven,", "mut", fs=10),
          T(270, 216, "память рядом с нейронами", "mut", fs=10))

# ================================================================ nlp_tasks_ner_classification_summarization
p = "ner"
_tok = ["Илон", "Маск", "основал", "SpaceX", "в", "2002"]
_tag = [("B-PER", "c1"), ("I-PER", "c1"), ("O", "c6"), ("B-ORG", "c2"), ("O", "c6"), ("B-DATE", "c3")]
_ner = []
for k, (tk, (tg, c)) in enumerate(zip(_tok, _tag)):
    x = 9 + 58 * k
    _ner.append(B(x, 15, 52, 28, "c6", tk, fs=11))
    _ner.append(A(p, x + 26, 43, x + 26, 58))
    _ner.append(A(p, x + 26, 86, x + 26, 101))
    _ner.append(B(x, 103, 52, 28, c, tg, fs=11))
NER = svg(p, 360, 196, "NER как token classification",
          *_ner[:2], *_ner[4:6], *_ner[8:10], *_ner[12:14], *_ner[16:18], *_ner[20:22],
          B(9, 60, 342, 26, "c5", "BERT → softmax по тегам для каждого токена", fs=11),
          *_ner[2:4], *_ner[6:8], *_ner[10:12], *_ner[14:16], *_ner[18:20], *_ner[22:24],
          L(12, 139, 116, 139, "ln s1"), L(186, 139, 232, 139, "ln s2"), L(302, 139, 348, 139, "ln s3"),
          T(64, 156, "PER: Илон Маск", "t1", fs=11), T(209, 156, "ORG: SpaceX", "t2", fs=11),
          T(325, 156, "DATE: 2002", "t3", fs=11),
          T(180, 186, "BIO: B — начало, I — продолжение, O — вне сущности", "mut", fs=10))

# ================================================================ node_ensemble
p = "node"
NODE = svg(p, 360, 262, "Oblivious decision tree",
           B(130, 20, 100, 30, "c1", "x₃ > b₁"),
           B(40, 95, 100, 30, "c3", "x₇ > b₂"),
           B(220, 95, 100, 30, "c3", "x₇ > b₂"),
           A(p, 165, 50, 100, 93), A(p, 195, 50, 260, 93),
           T(118, 68, "1 − c₁", "mut", a="end", fs=11), T(242, 68, "c₁", "mut", a="start", fs=11),
           *[A(p, x0, 125, x1, 173) for x0, x1 in [(75, 45), (105, 135), (255, 225), (285, 315)]],
           T(52, 152, "1−c₂", "mut", a="end", fs=10), T(128, 152, "c₂", "mut", a="start", fs=10),
           T(232, 152, "1−c₂", "mut", a="end", fs=10), T(308, 152, "c₂", "mut", a="start", fs=10),
           *[B(x, 175, 60, 28, "c2", s) for x, s in [(15, "R₀₀"), (105, "R₀₁"), (195, "R₁₀"), (285, "R₁₁")]],
           T(240, 30, "один признак", "t3", a="start", fs=10), T(240, 43, "на уровень", "t3", a="start", fs=10),
           T(180, 228, "h(x) = Σ Rₗ · wₗ,   w₁₁ = c₁ · c₂"),
           T(180, 248, "cₖ = σ(α(x − bₖ)) или entmax", "mut", fs=11))

# ================================================================ non_metric_criteria
p = "nmcr"
NMCR = svg(p, 360, 282, "Выбор модели по неметрическим критериям",
           B(10, 15, 180, 34, "c3", "Нужна объяснимость?"),
           B(225, 15, 125, 34, "c2", "линейная, дерево"),
           B(10, 85, 180, 34, "c3", "Жёсткий latency/память?"),
           B(225, 85, 125, 34, "c2", "компактная модель"),
           B(10, 155, 180, 34, "c3", "Табличные данные?"),
           B(225, 155, 125, 34, "c2", "бустинг / RF"),
           B(10, 225, 180, 34, "c1", "нейросеть + сжатие"),
           *[A(p, 190, y, 223, y) for y in (32, 102, 172)],
           *[T(207, y - 8, "да", "mut", fs=10) for y in (32, 102, 172)],
           *[A(p, 100, y, 100, y + 34) for y in (49, 119, 189)],
           *[T(106, y + 21, "нет", "mut", a="start", fs=10) for y in (49, 119, 189)],
           T(288, 240, "ориентир,", "mut", fs=10), T(288, 254, "не правило", "mut", fs=10))

# ================================================================ nonparametric_regression
p = "npr"


def _nw_svg():
    xs = [40, 62, 85, 104, 128, 146, 165, 178, 196, 214, 232, 255, 276, 300, 322]
    ys = [140, 128, 120, 102, 96, 82, 90, 70, 78, 66, 82, 95, 110, 118, 132]
    x0, h = 180, 35
    w = [np.exp(-0.5 * ((x - x0) / h) ** 2) for x in xs]
    yhat = sum(wi * yi for wi, yi in zip(w, ys)) / sum(w)
    out = [L(25, 170, 340, 170, "ln s6")]
    bell = " ".join(f"{x},{232 - 45 * np.exp(-0.5 * ((x - x0) / h) ** 2):.1f}" for x in range(60, 302, 4))
    out.append(f'<polyline class="ln s3" points="{bell}"/>')
    out.append(L(60, 232, 300, 232, "ln s6"))
    out.append(L(x0, 50, x0, 232, "ln s6", dash=True))
    yb = 232 - 45 * np.exp(-0.5)
    out.append(A(p, x0, yb, x0 + h - 1, yb, "ln s3"))
    out.append(T(x0 + h / 2, yb + 13, "h", "t3", fs=11))
    out.append(T(300, 222, "K(·)", "t3", a="start", fs=11))
    for x, y, wi in zip(xs, ys, w):
        out.append(f'<circle class="c1" cx="{x}" cy="{y}" r="{2 + 5 * wi:.1f}"/>')
    out.append(f'<circle class="c4" cx="{x0}" cy="{yhat:.1f}" r="5"/>')
    out.append(T(x0 + 8, yhat + 20, "ŷ(x₀)", "t4", a="start", fs=11))
    out.append(T(x0 + 6, 184, "x₀", a="start", fs=11))
    return "".join(out)


NPR = svg(p, 360, 268, "Nadaraya-Watson",
          T(180, 22, "размер точки = вес K((x₀ − xᵢ)/h)", "mut", fs=11),
          _nw_svg(),
          T(180, 258, "ŷ(x₀) = Σ K((x₀−xᵢ)/h)·yᵢ / Σ K((x₀−xᵢ)/h)", fs=11))

# ================================================================ normalizing_flows
p = "nf"
NF = svg(p, 360, 272, "Affine coupling layer",
         B(40, 15, 90, 30, "c1", "x₁"), B(230, 15, 90, 30, "c2", "x₂"),
         A(p, 85, 45, 85, 198),
         A(p, 85, 112, 113, 112),
         B(115, 95, 130, 34, "c5", "сеть: s(x₁), t(x₁)", fs=11),
         A(p, 245, 104, 261, 104), T(252, 98, "s", "t5", fs=11),
         P(p, "M230,129 V155 H261", "ln"), T(222, 150, "t", "t5", a="end", fs=11),
         A(p, 275, 45, 275, 90),
         C(275, 104, 13, "c6", "⊙"), T(292, 100, "exp(s)", "mut", a="start", fs=10),
         A(p, 275, 117, 275, 141),
         C(275, 155, 13, "c6", "+"),
         A(p, 275, 168, 275, 198),
         B(40, 200, 90, 30, "c1", "y₁ = x₁"), B(230, 200, 90, 30, "c2", "y₂"),
         T(180, 248, "y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁),  log|det J| = Σ s(x₁)", fs=11),
         T(180, 265, "обратно: x₂ = (y₂ − t(y₁)) ⊙ exp(−s(y₁))", "mut", fs=11))

# ================================================================ numpy_ml
p = "npb"


def _cell(x, y, v, cls, virt=False):
    d = ' stroke-dasharray="3 2"' if virt else ""
    return (f'<rect class="{cls}" x="{x}" y="{y}" width="30" height="30"{d}/>'
            + T(x + 15, y + 19, v, "mut" if virt else "", fs=12))


_np = []
for i, v in enumerate([1, 2, 3]):
    _np.append(_cell(20, 50 + 30 * i, v, "c1"))
    for j in (1, 2):
        _np.append(_cell(20 + 30 * j, 50 + 30 * i, v, "c6", True))
for j, v in enumerate([10, 20, 30]):
    _np.append(_cell(145 + 30 * j, 50, v, "c2"))
    for i in (1, 2):
        _np.append(_cell(145 + 30 * j, 50 + 30 * i, v, "c6", True))
for i, a_ in enumerate([1, 2, 3]):
    for j, b_ in enumerate([10, 20, 30]):
        _np.append(_cell(265 + 30 * j, 50 + 30 * i, a_ + b_, "c3"))
NPB = svg(p, 360, 205, "Broadcasting",
          T(180, 25, "col (3, 1) + row (1, 3) → (3, 3)"),
          *_np,
          T(125, 99, "+", fs=16), T(250, 99, "=", fs=16),
          T(65, 165, "(3, 1)", "t1"), T(190, 165, "(1, 3)", "t2"), T(310, 165, "(3, 3)", "t3"),
          T(180, 192, "оси сравниваются справа: равны или одна = 1", "mut", fs=10))

# ================================================================ object_tracking_video
p = "trk"
TRK = svg(p, 360, 300, "SORT",
          B(110, 10, 140, 30, "c6", "кадр t"),
          A(p, 150, 40, 100, 63),
          B(20, 65, 150, 40, "c1", "детектор (YOLO)", ("боксы D_t", "mut", 11)),
          B(190, 65, 150, 40, "c3", "Kalman: predict", ("треки T_t|t−1", "mut", 11)),
          A(p, 95, 105, 140, 133), A(p, 265, 105, 220, 133),
          B(60, 135, 240, 40, "c5", "сопоставление: IoU + Hungarian", ("DeepSORT: + appearance", "mut", 11)),
          A(p, 130, 175, 64, 208), A(p, 180, 175, 180, 208), A(p, 230, 175, 296, 208),
          B(10, 210, 105, 40, "c2", ("совпали:", "", 11), ("Kalman update", "", 11)),
          B(127, 210, 106, 40, "c2", ("новая детекция", "", 10), ("новый трек", "", 11)),
          B(244, 210, 106, 40, "c4", ("трек без пары:", "", 11), ("age+1 → удаление", "", 10)),
          P(p, "M62,250 V275 H352 V85 H342", "ln s6", dash=True),
          T(200, 291, "следующий кадр: t → t+1", "mut", fs=10))

# ================================================================ online_learning
p = "onl"
ONL = svg(p, 360, 262, "Протокол online learning",
          B(120, 10, 120, 30, "c6", "получить xₜ"),
          B(230, 90, 120, 44, "c1", "прогноз", "ŷₜ = f(xₜ; wₜ)"),
          B(115, 185, 130, 44, "c3", "узнать yₜ,", "потеря ℓ(ŷₜ, yₜ)"),
          B(10, 90, 120, 44, "c2", "обновить", "w ← w − ηₜ∇ℓ"),
          P(p, "M240,25 Q290,25 290,88"),
          P(p, "M290,134 Q290,207 247,207"),
          P(p, "M115,207 Q70,207 70,136"),
          P(p, "M70,90 Q70,25 118,25"),
          T(180, 107, "тест →", "mut", fs=10), T(180, 123, "обучение", "mut", fs=10),
          T(180, 252, "Regret_T = Σₜ ℓₜ(wₜ) − min_w Σₜ ℓₜ(w)", fs=11))

# ================================================================ onnx_format
p = "onx"
ONX = svg(p, 360, 290, "ONNX: экспорт и запуск",
          B(10, 20, 95, 28, "c5", "PyTorch"), B(10, 58, 95, 28, "c3", "TF / Keras"),
          B(10, 96, 95, 28, "c1", "sklearn"),
          A(p, 105, 34, 138, 70), A(p, 105, 72, 138, 80), A(p, 105, 110, 138, 90),
          T(122, 17, "экспорт", "mut", fs=10),
          B(140, 40, 100, 80, "c6", "model.onnx", ("граф + веса", "mut", 11), ("protobuf, opset", "mut", 10)),
          A(p, 240, 80, 263, 80),
          B(265, 55, 85, 50, "c2", "ONNX", "Runtime"),
          A(p, 290, 105, 206, 163), A(p, 307, 105, 266, 163), A(p, 325, 105, 327, 163),
          B(180, 165, 52, 28, "c6", "CPU", fs=11), B(240, 165, 52, 28, "c6", "CUDA", fs=11),
          B(300, 165, 55, 28, "c6", "TensorRT", fs=11),
          T(268, 210, "Execution Providers", "mut", fs=10),
          T(180, 240, "внутри: nodes (операторы) + initializers (веса)", "mut", fs=10),
          B(30, 252, 70, 26, "c1", "Conv", fs=11), B(145, 252, 70, 26, "c1", "Relu", fs=11),
          B(260, 252, 70, 26, "c1", "Gemm", fs=11),
          A(p, 100, 265, 143, 265), A(p, 215, 265, 258, 265))

# ================================================================ optics
p = "opt"
OPTS = svg(p, 340, 262, "OPTICS: core и reachability distance",
           '<circle class="c1" cx="130" cy="130" r="50" fill-opacity="0.5"/>',
           *[C(x, y, 4, "t6") for x, y in [(110, 150), (105, 112), (115, 165), (122, 98), (95, 135)]],
           C(173.3, 105, 4, "t6"),
           A(p, 130, 130, 171, 106, "ln s1"),
           T(150, 128, "core", "t1", a="start", fs=10),
           C(130, 130, 5, "t4"), T(124, 148, "p", "t4", a="end"),
           C(155, 155, 4, "t2"), T(162, 168, "o₁", "t2", a="start"),
           C(280, 175, 4, "t3"), T(288, 180, "o₂", "t3", a="start"),
           L(130, 130, 276, 174, "ln s3", dash=True),
           T(225, 145, "d(p, o₂)", "t3", fs=11),
           T(130, 60, "радиус = core-dist(p)", "t1", fs=11),
           T(60, 205, "reach(o₁) = core-dist(p)", "t2", a="start", fs=11),
           T(60, 222, "reach(o₂) = d(p, o₂)", "t3", a="start", fs=11),
           T(170, 252, "reach-dist(o, p) = max(core-dist(p), d(p, o))", fs=11))

# ================================================================ optimization_based_meta_learning
p = "maml"
_th, _opt = (170, 130), [(50, 45), (300, 55), (190, 210)]
_mm = []
for k, (ox, oy) in enumerate(_opt):
    tx, ty = _th[0] + 0.55 * (ox - _th[0]), _th[1] + 0.55 * (oy - _th[1])
    c = ["1", "2", "3"][k]
    _mm.append(A(p, 170, 130, tx, ty, f"ln s{c}", dash=True))
    _mm.append(L(tx, ty, ox, oy, "ln s6", dash=True))
    _mm.append(C(ox, oy, 5, f"t{c}"))
    _mm.append(C(tx, ty, 3, f"t{c}"))
MAML = svg(p, 340, 282, "MAML",
           *_mm,
           T(42, 38, "θ*₁", "t1", a="middle", fs=11), T(300, 46, "θ*₂", "t2", fs=11),
           T(204, 220, "θ*₃", "t3", a="start", fs=11),
           T(96, 92, "θ′₁", "t1", a="end", fs=11), T(248, 100, "θ′₂", "t2", a="start", fs=11),
           T(170, 178, "θ′₃", "t3", a="end", fs=11),
           C(170, 130, 6, "c5"), T(150, 140, "θ", "t5", a="end"),
           A(p, 170, 130, 177, 110, "ln s5"), T(184, 116, "β", "t5", a="start", fs=11),
           T(170, 248, "θ′ᵢ = θ − α∇θ L_τᵢ(θ)   (inner, support)", fs=11),
           T(170, 268, "θ ← θ − β∇θ Σᵢ L_τᵢ(θ′ᵢ)   (outer, query)", fs=11))

# ================================================================ optimizers_advanced
p = "opta"
_steps = [("c6", "SGD", "θ ← θ − η·g"),
          ("c1", "AdaGrad", "G ← G + g²;  θ ← θ − η·g / √(G + ε)"),
          ("c2", "RMSProp", "v ← βv + (1−β)g²;  θ ← θ − η·g / √(v + ε)"),
          ("c3", "Adam", "m ← β₁m + (1−β₁)g;  θ ← θ − η·m̂ / (√v̂ + ε)"),
          ("c5", "AdamW", "θ ← θ − η·(m̂ / (√v̂ + ε) + λθ)")]
_notes = ["адаптивный шаг на параметр", "сумма → EMA: шаг не гаснет", "+ импульс m, bias correction",
          "weight decay отдельно от ∇"]
_oa = []
for k, (c, n_, f_) in enumerate(_steps):
    y = 10 + 63 * k
    _oa.append(B(20, y, 320, 38, c, n_, (f_, "mut", 11), fs=12))
    if k < 4:
        _oa.append(A(p, 60, y + 38, 60, y + 61))
        _oa.append(T(72, y + 54, _notes[k], "mut", a="start", fs=10))
OPTA = svg(p, 360, 312, "Эволюция адаптивных оптимизаторов", *_oa)


# ================================================================ графики
def pareto_chart(fig):
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(0)
    th = rng.uniform(-0.3, 1.3, size=(600, 2))
    L1 = (th ** 2).sum(1)
    L2 = (th[:, 0] - 1) ** 2 + 4 * (th[:, 1] - 1) ** 2
    ax.scatter(L1, L2, s=5, color=GREY, alpha=0.35, label="случайные θ")
    w = np.linspace(0, 1, 200)
    t1 = 1 - w
    t2 = 4 * (1 - w) / (w + 4 * (1 - w))
    f1 = t1 ** 2 + t2 ** 2
    f2 = (t1 - 1) ** 2 + 4 * (t2 - 1) ** 2
    ax.plot(f1, f2, color=BLUE, lw=2.2, label="фронт Парето")
    for wi, c in [(0.2, GREEN), (0.5, ORANGE), (0.8, RED)]:
        a = 1 - wi
        b = 4 * (1 - wi) / (wi + 4 * (1 - wi))
        ax.scatter([a * a + b * b], [(a - 1) ** 2 + 4 * (b - 1) ** 2], s=40, color=c, zorder=3,
                   label=f"min {wi}·L₁ + {1 - wi:.1f}·L₂")
    ax.set_xlim(0, 3), ax.set_ylim(0, 5)
    ax.set_xlabel("L₁ (задача 1)"), ax.set_ylabel("L₂ (задача 2)")
    ax.set_title("Веса задач выбирают точку на фронте Парето", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def cca_chart(fig):
    rng = np.random.default_rng(1)
    n = 300
    z = rng.normal(size=n)
    X1 = np.c_[0.5 * z, 0.9 * z, 0.7 * z] + rng.normal(size=(n, 3))
    X2 = np.c_[0.5 * z, -0.9 * z, 0.7 * z, 0 * z] + rng.normal(size=(n, 4))
    A1, A2 = X1 - X1.mean(0), X2 - X2.mean(0)

    def isqrt(S):
        e, V = np.linalg.eigh(S)
        return V @ np.diag(e ** -0.5) @ V.T

    W1, W2 = isqrt(A1.T @ A1 / n), isqrt(A2.T @ A2 / n)
    U_, _, Vt = np.linalg.svd(W1 @ (A1.T @ A2 / n) @ W2)
    u, v = A1 @ W1 @ U_[:, 0], A2 @ W2 @ Vt[0]
    if np.corrcoef(u, v)[0, 1] < 0:
        v = -v
    ax = fig.add_subplot(2, 1, 1)
    r0 = np.corrcoef(X1[:, 0], X2[:, 0])[0, 1]
    ax.scatter(X1[:, 0], X2[:, 0], s=6, alpha=0.5, color=GREY)
    ax.set_title(f"Сырые признаки: corr = {r0:.2f}", fontsize=9)
    ax.set_xlabel("признак view 1"), ax.set_ylabel("признак view 2")
    ax2 = fig.add_subplot(2, 1, 2)
    r1 = np.corrcoef(u, v)[0, 1]
    ax2.scatter(u, v, s=6, alpha=0.5, color=BLUE)
    ax2.set_title(f"После CCA: corr(u, v) = {r1:.2f}", fontsize=9)
    ax2.set_xlabel("u = a·x⁽¹⁾"), ax2.set_ylabel("v = b·x⁽²⁾")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def vif_chart(fig):
    ax = fig.add_subplot(111)
    r = np.linspace(0, 0.99, 300)
    ax.plot(r, 1 / (1 - r ** 2), color=BLUE, lw=2, label="VIF = 1 / (1 − R²)")
    ax.plot(r, 1 / np.sqrt(1 - r ** 2), color=GREEN, lw=1.6, ls="--", label="рост SE(β̂) = √VIF")
    for v, c in [(5, ORANGE), (10, RED)]:
        ax.axhline(v, color=c, lw=1, ls=":")
        rr = np.sqrt(1 - 1 / v)
        ax.annotate(f"VIF={v} при |ρ|≈{rr:.2f}", (rr, v), xytext=(0.05, v * 1.25), fontsize=8, color=c)
    ax.set_yscale("log")
    ax.set_xlabel("корреляция двух признаков |ρ| (R²ᵢ = ρ²)")
    ax.set_ylabel("во сколько раз")
    ax.set_title("Мультиколлинеарность раздувает дисперсию", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def granger_chart(fig):
    rng = np.random.default_rng(3)
    n = 400
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for t in range(1, n):
        x1[t] = 0.6 * x1[t - 1] + rng.normal()
        x2[t] = 0.3 * x2[t - 1] + (0.9 * x1[t - 3] if t >= 3 else 0) + 0.6 * rng.normal()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(x1[:120], color=BLUE, lw=1.2, label="x₁")
    ax.plot(x2[:120], color=ORANGE, lw=1.2, label="x₂ (ведомый)")
    ax.set_title("x₂ повторяет x₁ с запаздыванием 3 шага", fontsize=9)
    ax.set_xlabel("t"), ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax2 = fig.add_subplot(2, 1, 2)
    lags = np.arange(-8, 9)
    cc = [np.corrcoef(x1[8 - k:n - 8 - k], x2[8:n - 8])[0, 1] for k in lags]
    ax2.bar(lags, cc, color=[RED if k == 3 else GREY for k in lags])
    ax2.set_xlabel("лаг k: corr(x₁ₜ₋ₖ, x₂ₜ)"), ax2.set_ylabel("корреляция")
    ax2.set_title("Кросс-корреляция: пик при k = 3", fontsize=9)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def rvq_chart(fig):
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 8)) * np.linspace(2.0, 0.5, 8)
    r = X.copy()
    err = [1.0]
    for s in range(6):
        km = KMeans(16, n_init=1, random_state=s, max_iter=50).fit(r)
        r = r - km.cluster_centers_[km.labels_]
        err.append((r ** 2).sum() / (X ** 2).sum())
    ax = fig.add_subplot(111)
    ax.plot(range(7), err, "o-", color=BLUE, lw=2)
    for k in range(1, 7):
        ax.annotate(f"{4 * k} бит", (k, err[k]), xytext=(k + 0.05, err[k] + 0.04), fontsize=8, color=GREY)
    ax.set_xlabel("число кодбуков (по 16 кодов = 4 бита)")
    ax.set_ylabel("‖x − x̂‖² / ‖x‖²")
    ax.set_title("RVQ: каждый кодбук уменьшает остаток", fontsize=9)
    ax.set_ylim(0, 1.08)


def beta_chart(fig):
    from scipy.stats import beta
    ax = fig.add_subplot(111)
    t = np.linspace(0, 1, 400)
    for (a, b), lab, c in [((2, 2), "prior Beta(2, 2)", GREY),
                           ((5, 3), "после 4 бросков (3 «да»)", ORANGE),
                           ((23, 11), "после 34 бросков (24 «да»)", BLUE)]:
        ax.plot(t, beta.pdf(t, a, b), color=c, lw=2, label=lab)
    ax.axvline(0.7, color=RED, ls=":", lw=1.2, label="истинное θ = 0.7")
    ax.set_xlabel("θ = P(«да»)"), ax.set_ylabel("плотность")
    ax.set_title("Posterior сужается с ростом данных", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def nas_chart(fig):
    ax = fig.add_subplot(111)
    names = ["Evolution", "RL (NASNet)", "Random", "Gradient (DARTS)", "Weight sharing"]
    vals = [3150, 1800, 100, 4, 0.5]
    cols = [RED, ORANGE, GREY, BLUE, GREEN]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, i, f"{v:g}" + ("+" if v == 100 else ""), va="center", fontsize=8)
    ax.set_xscale("log"), ax.set_xlim(0.2, 3e4)
    ax.set_xlabel("GPU-days (лог. шкала)")
    ax.set_title("Стоимость поиска по стратегиям", fontsize=9)
    ax.grid(axis="y", visible=False)


def sepconv_chart(fig):
    ax = fig.add_subplot(111)
    Cc = np.array([16, 32, 64, 128, 256, 512])
    k = 3
    std = k * k * Cc * Cc
    sep = k * k * Cc + Cc * Cc
    ax.plot(Cc, std, "o-", color=RED, lw=2, label="обычная conv: k²·C_in·C_out")
    ax.plot(Cc, sep, "o-", color=GREEN, lw=2, label="depthwise + pointwise: k²·C_in + C_in·C_out")
    for c_, a, b in zip(Cc[1::2], std[1::2], sep[1::2]):
        ax.annotate(f"×{a / b:.1f}", (c_, b), xytext=(c_ * 0.8, b / 3.2), fontsize=8, color=GREEN)
    ax.set_xscale("log", base=2), ax.set_yscale("log")
    ax.set_xlabel("каналы C_in = C_out (k = 3)"), ax.set_ylabel("параметров")
    ax.set_title("Separable conv: в ~k² раз меньше параметров", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def transit_chart(fig):
    rng = np.random.default_rng(2)
    t = np.arange(0, 27, 0.02)
    P_, dur, depth = 3.7, 0.16, 0.01
    ph = ((t - 1.0 + P_ / 2) % P_) - P_ / 2
    flux = 1 - depth * (np.abs(ph) < dur / 2) + 0.003 * np.sin(2 * np.pi * t / 11) + rng.normal(0, 0.0025, t.size)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, flux, ".", ms=1.5, color=BLUE)
    ax.set_title("Кривая блеска: провалы каждые P = 3.7 сут", fontsize=9)
    ax.set_xlabel("время, сут"), ax.set_ylabel("поток")
    ax2 = fig.add_subplot(2, 1, 2)
    detr = flux - 0.003 * np.sin(2 * np.pi * t / 11)
    ax2.plot(ph * 24, detr, ".", ms=2, color=GREY, alpha=0.6)
    bins = np.linspace(-12, 12, 49)
    idx = np.digitize(ph * 24, bins)
    m = [detr[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))]
    ax2.plot((bins[:-1] + bins[1:]) / 2, m, color=RED, lw=2, label="среднее по фазе")
    ax2.set_xlim(-12, 12)
    ax2.set_title("После detrend и phase folding: глубина ≈ 1%", fontsize=9)
    ax2.set_xlabel("фаза, ч от центра транзита"), ax2.set_ylabel("поток")
    ax2.legend(fontsize=7, loc="lower left")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def ppv_chart(fig):
    ax = fig.add_subplot(111)
    prev = np.logspace(-3, np.log10(0.5), 300)
    for sens, spec, c in [(0.90, 0.95, RED), (0.90, 0.99, BLUE)]:
        ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
        ax.plot(prev, ppv, color=c, lw=2, label=f"sens {sens}, spec {spec}")
    ax.scatter([0.01], [0.154], color=RED, zorder=3)
    ax.annotate("1%: PPV = 0.154", (0.01, 0.154), xytext=(0.013, 0.05), fontsize=8, color=RED)
    ax.set_xscale("log")
    ax.set_xlabel("распространённость болезни"), ax.set_ylabel("PPV (precision)")
    ax.set_title("При редкой болезни большинство «+» ложные", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def ntk_chart(fig):
    rng = np.random.default_rng(0)
    n = 20
    x = np.linspace(-2, 2, n)
    y = np.sin(1.5 * x)
    widths = [16, 64, 256, 1024, 4096]
    dth, dk = [], []
    for m in widths:
        w, b, a = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m)
        s = 1 / np.sqrt(m)

        def fj(w, b, a):
            z = np.outer(x, w) + b
            h = np.maximum(z, 0)
            d = (z > 0) * a * s
            return h @ a * s, np.hstack([d * x[:, None], d, h * s])

        th0 = np.concatenate([w, b, a])
        f, J = fj(w, b, a)
        K0 = J @ J.T
        for _ in range(300):
            f, J = fj(w, b, a)
            g = J.T @ (f - y) / n
            w, b, a = w - 0.5 * g[:m], b - 0.5 * g[m:2 * m], a - 0.5 * g[2 * m:]
        f, J = fj(w, b, a)
        dth.append(np.linalg.norm(np.concatenate([w, b, a]) - th0) / np.linalg.norm(th0))
        dk.append(np.linalg.norm(J @ J.T - K0) / np.linalg.norm(K0))
    ax = fig.add_subplot(111)
    ax.loglog(widths, dth, "o-", color=BLUE, lw=2, label="‖θ_T − θ₀‖ / ‖θ₀‖")
    ax.loglog(widths, dk, "s-", color=ORANGE, lw=2, label="‖Θ_T − Θ₀‖ / ‖Θ₀‖")
    ref = dk[0] * (np.array(widths) / widths[0]) ** -0.5
    ax.loglog(widths, ref, ":", color=GREY, label="∝ 1/√m")
    ax.set_xlabel("ширина скрытого слоя m"), ax.set_ylabel("относительное изменение")
    ax.set_title("Чем шире сеть, тем меньше меняются веса и ядро", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")


def lif_chart(fig):
    dt, tau, vth = 0.1, 10.0, 1.0
    t = np.arange(0, 120, dt)
    I = np.where(t < 20, 0, np.where(t < 60, 0.9, 1.6))
    V = np.zeros_like(t)
    spikes = []
    for k in range(1, t.size):
        V[k] = V[k - 1] + dt / tau * (-V[k - 1] + I[k])
        if V[k] >= vth:
            spikes.append(t[k])
            V[k] = 0.0
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2])
    ax = fig.add_subplot(gs[0])
    ax.plot(t, I, color=ORANGE, lw=2)
    ax.set_ylabel("R·I(t)"), ax.set_title("Leaky Integrate-and-Fire", fontsize=9)
    ax.set_xticklabels([])
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, V, color=BLUE, lw=1.6, label="V(t)")
    ax2.axhline(vth, color=RED, ls="--", lw=1, label="порог V_th")
    ax2.vlines(spikes, 1.05, 1.3, color=PURPLE, lw=1.5, label="спайки")
    ax2.set_ylim(-0.05, 1.4)
    ax2.set_xlabel("время, мс"), ax2.set_ylabel("потенциал V")
    ax2.text(2, 1.12, "20–60 мс: R·I < V_th,\nпорог не достигнут", fontsize=8, color=GREY)
    ax2.legend(fontsize=7, loc="lower right")
    fig.set_size_inches(4.2, 4.6)
    fig.tight_layout()


def softsplit_chart(fig):
    ax = fig.add_subplot(111)
    xx = np.linspace(-3, 3, 400)
    for a, c in [(1, GREY), (3, ORANGE), (10, BLUE)]:
        ax.plot(xx, 1 / (1 + np.exp(-a * xx)), color=c, lw=2, label=f"α = {a}")
    ax.step(xx, (xx > 0).astype(float), where="mid", color=RED, ls="--", lw=1.2, label="жёсткий split")
    ax.set_xlabel("x − b"), ax.set_ylabel("c = σ(α(x − b))")
    ax.set_title("Soft split: α задаёт жёсткость порога", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def train_time_chart(fig):
    ax = fig.add_subplot(111)
    names = ["SVM (kernel)", "Deep Neural Net", "Random Forest", "XGBoost", "Logistic Regr.", "Linear Regr."]
    vals = [1000, 500, 50, 40, 2, 1]
    cols = [RED, RED, ORANGE, ORANGE, GREEN, GREEN]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, i, f"{v}x", va="center", fontsize=8)
    ax.set_xscale("log"), ax.set_xlim(0.5, 5000)
    ax.set_xlabel("время обучения относительно линейной (лог.)")
    ax.set_title("Скорость обучения — тоже критерий", fontsize=9)
    ax.grid(axis="y", visible=False)


def nw_chart(fig):
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(0, 10, 90))
    y = np.sin(x) + 0.15 * x + rng.normal(0, 0.35, x.size)
    g = np.linspace(0, 10, 300)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, color=GREY, alpha=0.6)
    ax.plot(g, np.sin(g) + 0.15 * g, color="black", lw=1, ls=":", label="истинная f")
    for h, c, lab in [(0.12, RED, "h = 0.12: переобучение"), (0.6, BLUE, "h = 0.6"),
                      (3.0, ORANGE, "h = 3: недообучение")]:
        W = np.exp(-0.5 * ((g[:, None] - x[None]) / h) ** 2)
        ax.plot(g, W @ y / W.sum(1), color=c, lw=1.8, label=lab)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_title("Nadaraya-Watson: bandwidth h = bias ↔ variance", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def flow1d_chart(fig):
    from scipy.stats import norm
    ax = fig.add_subplot(111)
    z = np.linspace(-4, 4, 800)
    fz = z + 1.2 * np.tanh(3 * z)
    dfz = 1 + 3.6 / np.cosh(3 * z) ** 2
    ax.plot(z, norm.pdf(z), color=GREY, lw=1.6, ls="--", label="p_z: N(0, 1)")
    ax.plot(fz, norm.pdf(z) / dfz, color=BLUE, lw=2.2, label="p_x(x) = p_z(z) / |f′(z)|")
    rng = np.random.default_rng(0)
    zs = rng.normal(size=20000)
    ax.hist(zs + 1.2 * np.tanh(3 * zs), bins=80, density=True, color=BLUE, alpha=0.2, label="сэмплы x = f(z)")
    ax.set_xlim(-4.5, 4.5)
    ax.set_xlabel("значение"), ax.set_ylabel("плотность")
    ax.set_title("f(z) = z + 1.2·tanh(3z) растягивает центр", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def drift_chart(fig):
    rng = np.random.default_rng(5)
    T_, half = 6000, 3000
    X = rng.normal(size=(T_, 2))
    wt = np.where(np.arange(T_)[:, None] < half, [2.0, -1.0], [-1.0, 2.0])
    y = ((X * wt).sum(1) + 0.3 * rng.normal(size=T_) > 0).astype(float)
    ax = fig.add_subplot(111)
    for mode, c, lab in [("const", BLUE, "η = 0.05 (постоянный)"), ("decay", ORANGE, "ηₜ = 0.5/√t")]:
        w = np.zeros(2)
        hit = np.empty(T_)
        for t in range(T_):
            pr = 1 / (1 + np.exp(-X[t] @ w))
            hit[t] = (pr > 0.5) == y[t]
            eta = 0.05 if mode == "const" else 0.5 / np.sqrt(t + 1)
            w -= eta * (pr - y[t]) * X[t]
        roll = np.convolve(hit, np.ones(200) / 200, mode="valid")
        ax.plot(np.arange(roll.size) + 200, roll, color=c, lw=1.6, label=lab)
    ax.axvline(half, color=RED, ls="--", lw=1)
    ax.text(half + 100, 0.08, "concept drift", color=RED, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("шаг t"), ax.set_ylabel("prequential accuracy (окно 200)")
    ax.set_title("Постоянный шаг быстрее адаптируется к дрейфу", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")


def optics_chart(fig):
    from sklearn.cluster import OPTICS
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal([-4, -1], 0.3, (120, 2)), rng.normal([0, 3], 1.0, (120, 2)),
                   rng.normal([4, -1], 0.6, (100, 2)), rng.uniform([-7, -4], [7, 7], (30, 2))])
    o = OPTICS(min_samples=15, xi=0.1, min_cluster_size=0.15).fit(X)
    lab = o.labels_
    cols = np.array(PAL[:5])

    def colr(l):
        return [cols[k % 5] if k >= 0 else "#bbbbbb" for k in l]

    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], s=6, c=colr(lab))
    ax.set_title("Кластеры разной плотности (серые — шум)", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])
    ax2 = fig.add_subplot(2, 1, 2)
    r = o.reachability_[o.ordering_]
    r = np.where(np.isinf(r), np.nan, r)
    ax2.bar(np.arange(r.size), r, width=1.0, color=colr(lab[o.ordering_]))
    ax2.set_xlabel("точки в порядке обхода OPTICS"), ax2.set_ylabel("reachability")
    ax2.set_title("Reachability plot: «долины» = кластеры", fontsize=9)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


def optim_chart(fig):
    ax = fig.add_subplot(2, 1, 1)
    t = np.arange(1, 201)
    eta = 0.01
    ax.plot(t, eta / np.sqrt(t), color=BLUE, lw=2, label="AdaGrad: η/√(Σg²)")
    ax.plot(t, eta / np.sqrt(1 - 0.9 ** t), color=GREEN, lw=2, label="RMSProp (β=0.9): η/√E[g²]")
    ax.set_ylim(0, 0.035)
    ax.set_xlabel("шаг t (|g| = 1 всегда)"), ax.set_ylabel("эффективный шаг")
    ax.set_title("AdaGrad гасит шаг, RMSProp — нет", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax2 = fig.add_subplot(2, 1, 2)
    A_ = np.array([1.0, 20.0])

    def run(kind, lr, n=80):
        th = np.array([-4.0, 1.0])
        G = np.zeros(2); m = np.zeros(2); v = np.zeros(2)
        path = [th.copy()]
        for k in range(1, n + 1):
            g = A_ * th
            if kind == "sgd":
                th = th - lr * g
            elif kind == "adagrad":
                G += g * g; th = th - lr * g / np.sqrt(G + 1e-8)
            elif kind == "rmsprop":
                v = 0.9 * v + 0.1 * g * g; th = th - lr * g / np.sqrt(v + 1e-8)
            else:
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
                th = th - lr * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
            path.append(th.copy())
        return np.array(path)

    xx, yy = np.meshgrid(np.linspace(-4.5, 1, 100), np.linspace(-1.5, 1.5, 100))
    ax2.contour(xx, yy, 0.5 * (xx ** 2 + 20 * yy ** 2), levels=12, colors="#cccccc", linewidths=0.7)
    for kind, lr, c, lab, ls, lw in [("sgd", 0.09, GREY, "SGD", "-", 1), ("rmsprop", 0.1, GREEN, "RMSProp", "-", 1),
                                     ("adagrad", 0.6, BLUE, "AdaGrad", "--", 1.6), ("adam", 0.1, ORANGE, "Adam", "-", 1)]:
        pth = run(kind, lr)
        ax2.plot(pth[:, 0], pth[:, 1], marker=".", ms=2.5, ls=ls, lw=lw, color=c, label=lab)
    ax2.scatter([0], [0], marker="*", s=80, color=RED, zorder=3)
    ax2.set_xlabel("θ₁"), ax2.set_ylabel("θ₂")
    ax2.set_title("f = ½(θ₁² + 20θ₂²), 80 шагов", fontsize=9)
    ax2.legend(fontsize=7, loc="lower right", ncol=2)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ VISUALS
VISUALS = {
    "multi_task_learning": [
        {"title": "Hard parameter sharing", "svg": MTL,
         "caption": "Общий backbone получает градиенты всех K задач, у каждой задачи — своя голова и свой лосс Lₖ; "
                    "веса wₖ решают, какая задача сильнее тянет общие слои."},
    ],
    "multi_task_learning_deep_dive": [
        {"title": "PCGrad: конфликт градиентов", "svg": MTLD,
         "caption": "Если cos(g₁, g₂) < 0, PCGrad убирает из g₁ компоненту вдоль g₂ — остаётся проекция g₁′ "
                    "на плоскость, перпендикулярную g₂, и шаг больше не вредит задаче 2."},
        {"title": "Веса задач и фронт Парето", "chart": pareto_chart,
         "caption": "Каждый набор весов wₖ даёт одну Парето-оптимальную точку: улучшить одну задачу можно только "
                    "ценой другой. MGDA ищет такую точку без ручного подбора весов."},
    ],
    "multi_view_learning": [
        {"title": "Co-training", "svg": MVL,
         "caption": "Две модели учатся на разных views; уверенные прогнозы каждой на неразмеченных данных "
                    "становятся псевдо-метками в L. Работает, когда views достаточны и условно независимы."},
        {"title": "CCA находит общий сигнал", "chart": cca_chart,
         "caption": "Отдельные признаки двух views коррелируют слабо, но линейные проекции u и v, найденные CCA, "
                    "вытаскивают общую латентную переменную — корреляция резко растёт."},
    ],
    "multimodal_transformers": [
        {"title": "Cross-modal attention", "svg": MMT,
         "caption": "Query берутся из одной модальности (текст), Key и Value — из другой (патчи изображения): "
                    "каждый токен текста собирает нужные ему фрагменты картинки. Так устроены dual-stream модели."},
    ],
    "multiple_regression": [
        {"title": "Геометрия OLS", "svg": MREG,
         "caption": "ŷ = Xβ̂ — ортогональная проекция y на пространство столбцов X; остаток e перпендикулярен "
                    "всем столбцам, отсюда нормальные уравнения XᵀXβ̂ = Xᵀy."},
        {"title": "VIF и корреляция признаков", "chart": vif_chart,
         "caption": "VIF растёт как 1/(1 − R²ᵢ): при |ρ| ≈ 0.9 дисперсия β̂ уже в 5 раз больше, а стандартная "
                    "ошибка — в √VIF раз. Порог 5–10 — сигнал мультиколлинеарности."},
    ],
    "multitask_rl": [
        {"title": "Distral: общая политика π₀", "svg": MTRL,
         "caption": "Каждая задача учит свою политику πᵢ, но штраф KL(πᵢ ‖ π₀) держит её рядом с общей π₀, "
                    "а π₀ дистиллируется из всех πᵢ — так знания переносятся между задачами."},
    ],
    "multivariate_time_series": [
        {"title": "VAR(1): связи во времени и между рядами", "svg": MVTS,
         "caption": "Каждая переменная в момент t зависит от прошлого всех переменных: диагональ A — "
                    "собственная инерция, внедиагональные a₁₂, a₂₁ — перекрёстное влияние (основа теста Грейнджера)."},
        {"title": "Запаздывающее влияние", "chart": granger_chart,
         "caption": "x₂ зависит от x₁ с лагом 3, поэтому кросс-корреляция имеет пик при k = 3; прошлое x₁ "
                    "улучшает прогноз x₂ — x₁ Granger-причина x₂."},
    ],
    "musical_neural_networks": [
        {"title": "Residual vector quantization (RVQ)", "svg": RVQ,
         "caption": "Первый кодбук грубо квантует эмбеддинг кадра, каждый следующий — остаток предыдущего; "
                    "индексы k₁, k₂, k₃ — токены, которые генерирует трансформер (EnCodec, MusicGen)."},
        {"title": "Ошибка RVQ и число кодбуков", "chart": rvq_chart,
         "caption": "Каждый кодбук из 16 кодов добавляет 4 бита на кадр и уменьшает ошибку реконструкции — "
                    "число кодбуков регулирует баланс битрейта и качества."},
    ],
    "naive_bayes_bayesian_method": [
        {"title": "Наивное предположение как граф", "svg": NBB,
         "caption": "Класс y — единственный родитель всех признаков: при известном y признаки независимы, "
                    "поэтому P(x|y) распадается в произведение P(xᵢ|y)."},
        {"title": "Байесовское обновление", "chart": beta_chart,
         "caption": "Prior Beta(2, 2) после наблюдений превращается в posterior Beta(2 + «да», 2 + «нет»): "
                    "пик смещается к истинному θ, а распределение сужается."},
    ],
    "nas": [
        {"title": "Компоненты NAS и DARTS", "svg": NAS,
         "caption": "Стратегия выбирает архитектуру из search space, оценка возвращает её качество. DARTS делает "
                    "выбор дифференцируемым: на ребре — смесь операций с весами softmax(α), затем берётся argmax."},
        {"title": "Стоимость поиска", "chart": nas_chart,
         "caption": "Эволюция и RL требуют тысячи GPU-дней, градиентный поиск и weight sharing — единицы и доли: "
                    "разница в 3–4 порядка (данные из таблицы шпаргалки)."},
    ],
    "neural_architecture_patterns": [
        {"title": "Bottleneck + skip connection", "svg": NAP,
         "caption": "1×1 свёртки сжимают 256 каналов до 64 и разворачивают обратно, дорогая 3×3 работает на узком "
                    "тензоре; identity-путь x даёт градиенту прямую дорогу в глубокой сети."},
        {"title": "Depthwise separable: экономия параметров", "chart": sepconv_chart,
         "caption": "Обычная свёртка растёт как C², separable — почти так же, но с множителем 1 вместо k²: "
                    "при k = 3 и большом C выигрыш стремится к ~9 раз (MobileNet, Xception)."},
    ],
    "neural_network_serialization": [
        {"title": "Что и куда сохранять", "svg": NNS,
         "caption": "state_dict — рекомендуемый способ (нужен код класса), pickle целой модели ломается при "
                    "рефакторинге и небезопасен; TorchScript и ONNX уносят граф для деплоя без Python."},
    ],
    "neural_networks_astronomy": [
        {"title": "Конвейер поиска экзопланет", "svg": ASTR,
         "caption": "Кривую блеска очищают от тренда, ищут период (BLS), сворачивают по фазе и подают в 1D CNN, "
                    "которая отличает планету от затменных двойных (EB) и артефактов."},
        {"title": "Транзит в данных", "chart": transit_chart,
         "caption": "В сыром ряду провал 1% едва виден на фоне шума и переменности звезды; после detrend и "
                    "свёртки по периоду все транзиты накладываются и форма провала проявляется."},
    ],
    "neural_networks_biology": [
        {"title": "Архитектура AlphaFold2", "svg": BIO,
         "caption": "Evoformer обменивается информацией между MSA (эволюция) и парным представлением (контакты "
                    "остатков), Structure module строит 3D-координаты; результат прогоняется повторно (recycling)."},
    ],
    "neural_networks_chemistry": [
        {"title": "Message passing на молекуле", "svg": CHEM,
         "caption": "Атомы — вершины, связи — рёбра с признаками e_uv. На каждом слое атом собирает сообщения "
                    "соседей и обновляет h_v; readout агрегирует все атомы в предсказание свойства."},
    ],
    "neural_networks_medicine": [
        {"title": "Attention-MIL для гистологии", "svg": MED,
         "caption": "Метка известна только для всего среза: тайлы кодируются CNN, attention даёт каждому вес aᵢ, "
                    "взвешенная сумма классифицируется. Веса aᵢ заодно подсвечивают подозрительные участки."},
        {"title": "PPV и распространённость", "chart": ppv_chart,
         "caption": "Та же модель (sens 0.90, spec 0.95) при распространённости 1% даёт лишь 15% верных "
                    "положительных ответов; высокая специфичность критична для скрининга."},
    ],
    "neural_networks_physics": [
        {"title": "Physics-informed neural network", "svg": PINN,
         "caption": "Сеть u_θ(x, t) учится сразу на данных и на уравнении: производные берутся автодиффом, "
                    "невязка PDE в точках коллокации входит в лосс вместе с граничными условиями."},
    ],
    "neural_tangent_kernel": [
        {"title": "NTK и lazy training", "svg": NTKS,
         "caption": "NTK — скалярное произведение градиентов по весам для двух входов. В широкой сети веса "
                    "почти не уходят от θ₀, поэтому сеть ≈ своя линеаризация, а ядро Θ постоянно."},
        {"title": "Ширина сети и изменение весов", "chart": ntk_chart,
         "caption": "Двухслойная ReLU-сеть в NTK-параметризации, 300 шагов GD: относительное изменение весов "
                    "и ядра падает примерно как 1/√m — это и есть lazy-режим."},
    ],
    "neuro_symbolic_ai_systems": [
        {"title": "Нейро-символьный конвейер", "svg": NSY,
         "caption": "Нейросеть переводит сырые данные в символы с вероятностями, символьный модуль делает "
                    "логический вывод и объясняет ответ; результат вывода может служить сигналом для обучения сети."},
    ],
    "neuromorphic_computing": [
        {"title": "Von Neumann vs нейроморфная архитектура", "svg": NMC,
         "caption": "В классической архитектуре данные постоянно ездят между CPU и памятью; в нейроморфной "
                    "память лежит рядом с нейронами, а ядра обмениваются редкими спайками только по событию."},
        {"title": "Нейрон LIF", "chart": lif_chart,
         "caption": "Потенциал интегрирует ток и утекает к покою; при R·I < V_th порог не достигается, при большем "
                    "токе нейрон спайкует и сбрасывается — частота спайков кодирует силу входа."},
    ],
    "nlp_tasks_ner_classification_summarization": [
        {"title": "NER как token classification", "svg": NER,
         "caption": "BERT выдаёт тег для каждого токена по схеме BIO; соседние B- и I- теги одного типа "
                    "склеиваются в сущность. Метрика — F1 по целым сущностям."},
    ],
    "node_ensemble": [
        {"title": "Oblivious decision tree", "svg": NODE,
         "caption": "На каждом уровне все узлы проверяют один и тот же признак и порог, поэтому дерево глубины d "
                    "— это d решений и 2ᵈ листьев; в NODE вес листа — произведение мягких решений cₖ."},
        {"title": "Soft split", "chart": softsplit_chart,
         "caption": "Сигмоида с крутизной α заменяет жёсткий порог: при малом α обе ветви получают вклад и "
                    "градиент течёт к порогу b, при большом α split почти жёсткий."},
    ],
    "non_metric_criteria": [
        {"title": "Выбор модели по ограничениям", "svg": NMCR,
         "caption": "Упрощённая схема: сначала проверяют жёсткие требования (объяснимость, latency, память), "
                    "и только потом сравнивают модели по метрикам качества."},
        {"title": "Относительное время обучения", "chart": train_time_chart,
         "caption": "Ориентиры из таблицы шпаргалки: между линейной моделью и kernel SVM — три порядка "
                    "по времени обучения при сопоставимом размере данных."},
    ],
    "nonparametric_regression": [
        {"title": "Ядерная регрессия Nadaraya-Watson", "svg": NPR,
         "caption": "Прогноз в x₀ — взвешенное среднее yᵢ; вес задаёт ядро K, ширина h решает, сколько соседей "
                    "реально влияют на ответ."},
        {"title": "Выбор bandwidth", "chart": nw_chart,
         "caption": "Малое h повторяет шум (высокая variance), большое h сглаживает форму (высокий bias); "
                    "h подбирают кросс-валидацией."},
    ],
    "normalizing_flows": [
        {"title": "Affine coupling layer (RealNVP)", "svg": NF,
         "caption": "x₁ проходит без изменений и задаёт s и t для x₂; якобиан треугольный, поэтому log det = Σ s, "
                    "а обращение не требует обращать сами сети s и t."},
        {"title": "Замена переменных в 1D", "chart": flow1d_chart,
         "caption": "Где f растягивает пространство (|f′| велик, около 0), плотность падает: гауссиана "
                    "превращается в бимодальное распределение, и формула точно совпадает с гистограммой сэмплов."},
    ],
    "numpy_ml": [
        {"title": "Broadcasting", "svg": NPB,
         "caption": "Оси размера 1 «растягиваются» без копирования (пунктир — виртуальные копии): "
                    "(3, 1) + (1, 3) даёт матрицу (3, 3) всех попарных сумм."},
    ],
    "object_tracking_video": [
        {"title": "Цикл SORT / DeepSORT", "svg": TRK,
         "caption": "Kalman предсказывает положение треков, детекции сопоставляются с ними по IoU венгерским "
                    "алгоритмом; несопоставленные детекции открывают треки, а треки без детекций стареют и удаляются."},
    ],
    "online_learning": [
        {"title": "Протокол online learning", "svg": ONL,
         "caption": "На каждом шаге модель сначала прогнозирует, потом узнаёт ответ и обновляется — так считается "
                    "prequential-оценка. Качество алгоритма меряют регретом относительно лучшей фиксированной модели."},
        {"title": "Адаптация к дрейфу", "chart": drift_chart,
         "caption": "Online-логрегрессия на потоке, где в t = 3000 меняется зависимость: с постоянным шагом модель "
                    "быстро переучивается, а с затухающим ηₜ ∝ 1/√t восстанавливается гораздо медленнее."},
    ],
    "onnx_format": [
        {"title": "ONNX: от фреймворка к runtime", "svg": ONX,
         "caption": "Модель из любого фреймворка экспортируется в граф операторов с весами (protobuf); ONNX Runtime "
                    "оптимизирует граф и исполняет его на CPU, GPU или TensorRT через Execution Providers."},
    ],
    "optics": [
        {"title": "Core и reachability distance", "svg": OPTS,
         "caption": "core-dist(p) — радиус до min_samples-го соседа. Для близкой точки o₁ reachability равна "
                    "core-dist(p), для далёкой o₂ — обычному расстоянию: внутри плотных областей расстояния «сглажены»."},
        {"title": "Reachability plot", "chart": optics_chart,
         "caption": "Точки в порядке обхода: долины на графике — кластеры, пики — переходы между ними. Плотный "
                    "кластер даёт глубокую долину, рыхлый — мелкую, и один параметр eps не нужен."},
    ],
    "optimization_based_meta_learning": [
        {"title": "MAML: внутренний и внешний цикл", "svg": MAML,
         "caption": "Из общей инициализации θ несколько шагов на support set приводят к θ′ᵢ для каждой задачи; "
                    "внешний шаг сдвигает θ так, чтобы после адаптации лосс на query set был минимален."},
    ],
    "optimizers_advanced": [
        {"title": "От SGD к AdamW", "svg": OPTA,
         "caption": "AdaGrad делит шаг на накопленную сумму g², RMSProp заменяет сумму скользящим средним, "
                    "Adam добавляет импульс и bias correction, AdamW выносит weight decay из градиентного шага."},
        {"title": "Эффективный шаг и траектории", "chart": optim_chart,
         "caption": "При постоянном градиенте шаг AdaGrad убывает как 1/√t, у RMSProp — стабилизируется. "
                    "На вытянутой параболе адаптивные методы выравнивают масштаб осей, а SGD колеблется по крутой оси."},
    ],
}
