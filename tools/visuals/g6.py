# -*- coding: utf-8 -*-
"""Рисунки для шпаргалок группы g6 (optuna_hyperopt … social_network_analysis).

SVG собираются небольшими помощниками (_svg, _box, _t, _ar …): цвета только через классы theme.css,
размер шрифта — через style (правило `.mlcs-visual svg text {font-size:12px}` перебивает атрибут font-size).
Графики — matplotlib, фиксированный RNG, только numpy / scipy / sklearn.
"""

import numpy as np
from html import escape as _esc

B, G, O, R, V, GR = "#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"


def _n(v):
    return f"{round(float(v), 1):g}"


def _svg(topic, w, h, label, *parts):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(label)}">'
            f'<defs><marker id="{topic}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            + "".join(parts) + "</svg>")


def _t(x, y, s, cls=None, a="middle", fs=None, bold=False, mono=False, extra=""):
    at = f' class="{cls}"' if cls else ""
    if a != "start":
        at += f' text-anchor="{a}"'
    st = []
    if fs:
        st.append(f"font-size:{fs}px")
    if bold:
        st.append("font-weight:600")
    if mono:
        st.append("font-family:Consolas,Menlo,monospace")
    if st:
        at += f' style="{";".join(st)}"'
    return f'<text x="{_n(x)}" y="{_n(y)}"{at}{extra}>{_esc(s, quote=False)}</text>'


def _box(x, y, w, h, cls, lines, fs=None, rx=6, mono=False, bold_first=False):
    """Прямоугольник с центрированными строками; строка может быть (текст, класс)."""
    if isinstance(lines, str):
        lines = [lines]
    f = fs or 12
    out = [f'<rect class="{cls}" x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{rx}"/>']
    lh = f + 3
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + f * 0.36
    for i, s in enumerate(lines):
        c = None
        if isinstance(s, tuple):
            s, c = s
        out.append(_t(x + w / 2, y0 + i * lh, s, cls=c, fs=fs, mono=mono, bold=bold_first and i == 0))
    return "".join(out)


def _ar(topic, x1, y1, x2, y2, cls="ln", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return (f'<line class="{cls}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}"{d} '
            f'marker-end="url(#{topic}-a)"/>')


def _ln(x1, y1, x2, y2, cls="ln", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line class="{cls}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}"{d}/>'


def _path(d, cls="ln", topic=None, dash=False):
    m = f' marker-end="url(#{topic}-a)"' if topic else ""
    ds = ' stroke-dasharray="4 3"' if dash else ""
    return f'<path class="{cls}" d="{d}"{ds}{m}/>'


def _poly(xs, ys, cls="ln", extra=""):
    pts = " ".join(f"{_n(a)},{_n(b)}" for a, b in zip(xs, ys))
    return f'<polyline class="{cls}" fill="none" points="{pts}"{extra}/>'


def _c(cx, cy, r, cls):
    return f'<circle class="{cls}" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}"/>'


def _rect(x, y, w, h, cls, rx=0, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<rect class="{cls}" x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{rx}"{d}/>'


VISUALS = {}

# ───────────────────────────── optuna_hyperopt ─────────────────────────────
_k = "optuna_hyperopt"
OPTUNA_SVG = _svg(
    _k, 340, 285, "Цикл study.optimize в Optuna",
    _box(70, 10, 200, 40, "c6", ["Study: история trials", ("direction='maximize'", "mut")], fs=11),
    _box(15, 82, 140, 44, "c1", ["Sampler (TPE)", "предлагает параметры"], fs=11),
    _box(185, 82, 140, 44, "c2", ["objective(trial)", "suggest → fit → score"], fs=11),
    _box(185, 158, 140, 44, "c3", ["report(score, step)", "should_prune()?"], fs=11),
    _box(15, 232, 140, 40, "c4", ["TrialPruned", "испытание прервано"], fs=11),
    _box(185, 232, 140, 40, "c2", ["return score", "после всех step"], fs=11),
    _ar(_k, 110, 50, 90, 80), _t(80, 68, "история", "mut", fs=11, a="end"),
    _ar(_k, 155, 104, 183, 104),
    _ar(_k, 255, 126, 255, 156),
    _ar(_k, 200, 202, 130, 230), _t(150, 214, "да", "t4", fs=11, bold=True),
    _ar(_k, 255, 202, 255, 230), _t(262, 220, "нет", "t2", fs=11, a="start", bold=True),
    _path("M15,252 H6 V30 H68", "ln s6", _k, dash=True),
    _path("M325,252 H334 V30 H272", "ln s6", _k, dash=True),
)


def optuna_chart(fig):
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, 40)
    loss = 0.5 * (x - 0.8) ** 2 + 0.3 * np.sin(3 * x) + rng.normal(0, 0.15, 40)
    thr = np.quantile(loss, 0.25)
    gd = loss <= thr
    grid = np.linspace(-3, 3, 300)
    lx, gx = gaussian_kde(x[gd])(grid), gaussian_kde(x[~gd])(grid)
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(x[~gd], loss[~gd], s=14, color=GR, label="плохие trials")
    ax.scatter(x[gd], loss[gd], s=20, color=G, label="лучшие γ = 25%")
    ax.axhline(thr, color=O, ls="--", lw=1.2, label="порог y*")
    ax.set_ylabel("loss (минимизируем)")
    ax.set_title("40 завершённых trials", fontsize=9)
    ax.legend(fontsize=7, loc="upper center")
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    ax2.plot(grid, lx, color=G, lw=2, label="l(x) — лучшие")
    ax2.plot(grid, gx, color=GR, lw=2, label="g(x) — остальные")
    ratio = lx / gx
    ax2.plot(grid, ratio / ratio.max() * max(lx.max(), gx.max()), color=B, lw=1.5, ls="--",
             label="l(x) / g(x) (масштаб)")
    xb = grid[np.argmax(ratio)]
    ax2.axvline(xb, color=R, lw=1.2)
    ax2.text(xb - 0.08, 0.02, "следующее x", color=R, fontsize=8, ha="right")
    ax2.set_xlabel("гиперпараметр x"), ax2.set_ylabel("плотность")
    ax2.set_title("TPE: выбрать x с максимумом l(x) / g(x)", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS[_k] = [
    {"title": "Цикл оптимизации Optuna", "svg": OPTUNA_SVG,
     "caption": "Sampler (по умолчанию TPE) берёт историю study и предлагает параметры; внутри objective "
                "промежуточный score сообщается через trial.report, и pruner обрывает бесперспективные trials."},
    {"title": "Как TPE выбирает следующую точку", "chart": optuna_chart,
     "caption": "TPE делит trials на лучшие и остальные, строит две плотности l(x) и g(x) и пробует x, где "
                "отношение l/g максимально, — там хорошие результаты встречались чаще плохих."},
]

# ───────────────────────────── outliers_handling ─────────────────────────────
_k = "outliers_handling"
OUTLIERS_SVG = _svg(
    _k, 340, 200, "Box plot и границы IQR",
    _ln(10, 80, 330, 80, "ln s6"),
    _ln(40, 70, 40, 108, "ln s3", dash=True), _ln(280, 70, 280, 108, "ln s3", dash=True),
    _ln(62, 80, 130, 80), _ln(190, 80, 255, 80), _ln(62, 70, 62, 90), _ln(255, 70, 255, 90),
    _rect(130, 60, 60, 40, "c1", rx=2), _ln(156, 60, 156, 100, "ln s4"),
    _t(130, 50, "Q1", fs=12), _t(190, 50, "Q3", fs=12), _t(156, 30, "медиана", "t4", fs=11),
    _ln(156, 34, 156, 58, "ln s4"),
    _c(22, 80, 4, "t4"), _c(300, 80, 4, "t4"), _c(318, 80, 4, "t4"),
    _t(22, 64, "выброс", "t4", fs=11), _t(309, 64, "выбросы", "t4", fs=11),
    _path("M130,110 V116 H190 V110", "ln"), _t(160, 132, "IQR = Q3 − Q1", fs=12),
    _t(40, 154, "Q1 − 1.5·IQR", "t3", fs=11), _t(280, 154, "Q3 + 1.5·IQR", "t3", fs=11),
    _t(105, 118, "ус", "mut", fs=10), _t(222, 118, "ус", "mut", fs=10),
    _t(170, 186, "Z-score: z = (x − μ) / σ,  |z| > 3 → выброс", "mut", fs=11),
)


def outliers_chart(fig):
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.lognormal(3, 0.35, 400), [70, 85, 105, 4]])
    q1, q3 = np.percentile(x, [25, 75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    mu, sd = x.mean(), x.std()
    n_iqr = int(((x < lo) | (x > hi)).sum())
    n_z = int((np.abs(x - mu) / sd > 3).sum())
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=60, color=GR, alpha=0.7)
    ax.axvline(lo, color=G, ls="--", lw=1.5)
    ax.axvline(hi, color=G, ls="--", lw=1.5, label=f"IQR-границы: {n_iqr} выбросов")
    ax.axvline(mu + 3 * sd, color=R, ls="-.", lw=1.5, label=f"μ + 3σ: {n_z} выбросов")
    ax.set_xlabel("значение признака"), ax.set_ylabel("частота")
    ax.set_title("На скошенных данных IQR и Z-score расходятся", fontsize=9)
    ax.legend(fontsize=7)


VISUALS[_k] = [
    {"title": "Box plot и правило 1.5·IQR", "svg": OUTLIERS_SVG,
     "caption": "Ящик — от Q1 до Q3, линия внутри — медиана; усы доходят до крайних точек внутри границ "
                "Q1 − 1.5·IQR и Q3 + 1.5·IQR, всё, что за ними, рисуется точками-выбросами."},
    {"title": "IQR против Z-score", "chart": outliers_chart,
     "caption": "Выбросы сами раздувают σ, поэтому порог μ + 3σ уезжает вправо и пропускает часть хвоста; "
                "квартили устойчивее. Для скошенных данных IQR (или логарифм) надёжнее Z-score."},
]

# ───────────────────────────── pandas_ml ─────────────────────────────
_k = "pandas_ml"


def _pandas_svg():
    p = [_t(55, 18, "df", bold=True), _t(165, 18, "split", bold=True), _t(290, 18, "apply + combine", bold=True)]
    p += [_rect(10, 26, 45, 20, "c6"), _rect(55, 26, 45, 20, "c6"),
          _t(32.5, 40, "city", fs=11, bold=True), _t(77.5, 40, "salary", fs=11, bold=True)]
    rows = [("A", 10), ("B", 20), ("A", 30), ("B", 40), ("A", 20)]
    for i, (c, s) in enumerate(rows):
        cl = "c1" if c == "A" else "c3"
        y = 46 + 20 * i
        p += [_rect(10, y, 45, 20, cl), _rect(55, y, 45, 20, cl), _t(32.5, y + 14, c), _t(77.5, y + 14, str(s))]
    for j, (s, y) in enumerate([(10, 36), (30, 56), (20, 76)]):
        p += [_rect(135, y, 60, 20, "c1"), _t(165, y + 14, f"A   {s}")]
    for s, y in [(20, 112), (40, 132)]:
        p += [_rect(135, y, 60, 20, "c3"), _t(165, y + 14, f"B   {s}")]
    p += [_ar(_k, 104, 96, 131, 96)]
    p += [_rect(250, 56, 80, 22, "c1"), _t(290, 71, "A   20"), _rect(250, 122, 80, 22, "c3"), _t(290, 137, "B   30")]
    p += [_ar(_k, 197, 66, 247, 67), _ar(_k, 197, 132, 247, 133),
          _t(222, 58, "mean", "mut", fs=10), _t(222, 124, "mean", "mut", fs=10)]
    p += [_t(10, 176, "groupby('city')['salary'].transform('mean')", "mut", a="start", fs=10, mono=True)]
    for i, (c, _) in enumerate(rows):
        cl = "c1" if c == "A" else "c3"
        p += [_rect(10 + 64 * i, 184, 60, 22, cl), _t(40 + 64 * i, 199, "20" if c == "A" else "30")]
    p += [_t(170, 222, "transform сохраняет число строк df", "mut", fs=11)]
    return _svg(_k, 340, 230, "groupby: split-apply-combine и transform", *p)


VISUALS[_k] = [
    {"title": "groupby: split → apply → combine", "svg": _pandas_svg(),
     "caption": "groupby делит строки по ключу, считает агрегат в каждой группе и склеивает результат — одна "
                "строка на группу. transform возвращает агрегат на место каждой исходной строки, поэтому его "
                "можно сразу записать новым признаком."},
]

# ───────────────────────────── pca_method ─────────────────────────────
_k = "pca_method"


def _pca_svg():
    rng = np.random.default_rng(3)
    ang = np.deg2rad(-28)
    u = np.array([np.cos(ang), np.sin(ang)])
    v = np.array([-u[1], u[0]])
    z = rng.normal(size=(70, 2)) * [48, 14]
    z = np.clip(z, [-120, -40], [120, 40])
    pts = np.array([165, 120]) + z[:, :1] * u + z[:, 1:] * v
    m = pts.mean(0)
    w, E = np.linalg.eigh(np.cov(pts.T))
    e1, e2 = E[:, 1], E[:, 0]
    s1, s2 = np.sqrt(w[1]), np.sqrt(w[0])
    if e1[0] < 0:
        e1 = -e1
    if e2[1] > 0:
        e2 = -e2
    d = pts - m
    c1, c2 = d @ e1, d @ e2
    cand = np.where((np.abs(c1) < 1.2 * s1) & (c2 < 0))[0]
    i = cand[np.argmax(-c2[cand])]
    pr = m + c1[i] * e1
    a1, b1 = m - 2.6 * s1 * e1, m + 2.6 * s1 * e1
    t2 = m + 3.2 * s2 * e2
    p = [_c(x, y, 2.6, "t6") for x, y in pts]
    p += [_ln(*a1, *m, "ln s1"), _ar(_k, *m, *b1, "ln s1"), _ar(_k, *m, *t2, "ln s2")]
    p += [_ln(*pts[i], *pr, "ln s4", dash=True), _c(*pts[i], 4, "t4"), _c(*pr, 4, "t4")]
    p += [_t(b1[0] + 4, b1[1] - 8, "PC1", "t1", bold=True), _t(t2[0] - 4, t2[1] - 8, "PC2", "t2", bold=True)]
    p += [_t(pr[0] + 10, pr[1] + 18, "проекция", "t4", a="start", fs=11)]
    p += [_t(170, 232, "PC1 — направление максимальной дисперсии", "mut", fs=11),
          _t(170, 250, "PC2 ⟂ PC1, берёт остаток дисперсии", "mut", fs=11)]
    return _svg(_k, 340, 260, "Главные компоненты облака точек", *p)


def pca_chart(fig):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    X = load_digits().data
    r = PCA().fit(X).explained_variance_ratio_
    cum = np.cumsum(r)
    k = int(np.searchsorted(cum, 0.95) + 1)
    ax = fig.add_subplot(1, 1, 1)
    n = np.arange(1, 41)
    ax.bar(n, r[:40], color=B, alpha=0.75, label="explained_variance_ratio_")
    ax.plot(n, cum[:40], color=O, lw=2, marker="o", ms=2.5, label="накопленная доля")
    ax.axhline(0.95, color=R, ls="--", lw=1)
    ax.axvline(k, color=R, ls=":", lw=1)
    ax.text(k - 0.8, 0.5, f"95% при k = {k}", color=R, fontsize=8, ha="right")
    ax.set_xlabel("номер компоненты"), ax.set_ylabel("доля дисперсии")
    ax.set_title("digits, 64 признака: доля дисперсии", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right")


VISUALS[_k] = [
    {"title": "Геометрия PCA", "svg": _pca_svg(),
     "caption": "PC1 проходит вдоль самой длинной оси облака, PC2 перпендикулярна ей. Новый признак — "
                "координата проекции точки на компоненту; отбросив PC2, теряем только малую дисперсию поперёк."},
    {"title": "Сколько компонент оставить", "chart": pca_chart,
     "caption": "Первые компоненты объясняют больше всего дисперсии, дальше вклад быстро падает. "
                "PCA(n_components=0.95) берёт минимальное k, при котором накопленная доля достигает 95%."},
]

# ───────────────────────────── perceptron ─────────────────────────────
_k = "perceptron"
PERCEPTRON_SVG = _svg(
    _k, 340, 200, "Схема искусственного нейрона",
    _c(30, 50, 15, "c1"), _c(30, 100, 15, "c1"), _c(30, 150, 15, "c1"),
    _t(30, 54, "x₁"), _t(30, 104, "x₂"), _t(30, 154, "x₃"),
    _ar(_k, 45, 53, 153, 90), _ar(_k, 45, 100, 150, 100), _ar(_k, 45, 147, 153, 110),
    _t(98, 62, "w₁", "t3", fs=12), _t(98, 94, "w₂", "t3", fs=12), _t(98, 146, "w₃", "t3", fs=12),
    _c(175, 100, 24, "c3"), _t(175, 106, "Σ", fs=18),
    _t(175, 28, "b", "t5", bold=True), _ar(_k, 175, 34, 175, 74, "ln s5"),
    _ar(_k, 199, 100, 228, 100),
    _box(230, 80, 50, 40, "c2", "f", fs=14),
    _ar(_k, 280, 100, 315, 100), _t(327, 104, "y"),
    _t(175, 148, "z = Σ wᵢxᵢ + b", "mut", fs=11), _t(255, 140, "y = f(z)", "mut", fs=11),
    _t(30, 186, "входы", "mut", fs=11), _t(175, 186, "линейная часть", "mut", fs=11),
    _t(262, 186, "активация", "mut", fs=11),
)


def perceptron_chart(fig):
    z = np.linspace(-4, 4, 400)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=B, lw=2, label="Sigmoid: (0, 1)")
    ax.plot(z, np.tanh(z), color=G, lw=2, label="Tanh: (−1, 1)")
    ax.plot(z, np.maximum(0, z), color=O, lw=2, label="ReLU: max(0, z)")
    ax.plot(z, np.where(z > 0, z, 0.1 * z), color=V, lw=1.5, ls="--", label="Leaky ReLU (α = 0.1)")
    ax.set_ylim(-1.5, 3), ax.set_xlabel("z"), ax.set_ylabel("f(z)")
    ax.set_title("Функции активации", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


VISUALS[_k] = [
    {"title": "Устройство нейрона", "svg": PERCEPTRON_SVG,
     "caption": "Каждый вход умножается на свой вес, суммы складываются со смещением b в z, "
                "а нелинейная активация f превращает z в выход y."},
    {"title": "Формы функций активации", "chart": perceptron_chart,
     "caption": "Sigmoid и tanh насыщаются на краях (градиент → 0), ReLU не насыщается при z > 0, "
                "но обнуляет отрицательные z; Leaky ReLU оставляет им малый наклон (обычно α = 0.01)."},
]

# ───────────────────────────── pose_estimation ─────────────────────────────
_k = "pose_estimation"


def _pose_svg():
    kp = {"нос": (90, 38), "шея": (90, 66), "rsh": (60, 70), "lsh": (120, 70), "rel": (46, 110), "rwr": (40, 148),
          "lel": (140, 106), "lwr": (152, 142), "rhip": (74, 150), "lhip": (106, 150), "rkn": (70, 200),
          "lkn": (112, 200), "ran": (66, 250), "lan": (118, 250)}
    bones = [("нос", "шея"), ("шея", "rsh"), ("шея", "lsh"), ("rsh", "rel"), ("rel", "rwr"), ("lsh", "lel"),
             ("lel", "lwr"), ("шея", "rhip"), ("шея", "lhip"), ("rhip", "lhip"), ("rhip", "rkn"), ("rkn", "ran"),
             ("lhip", "lkn"), ("lkn", "lan")]
    p = [_ln(*kp[a], *kp[b], "ln s6") for a, b in bones]
    p += [_c(x, y, 4.5, "c1") for x, y in kp.values()]
    p += [_c(84, 32, 2, "t1"), _c(96, 32, 2, "t1")]
    # угол в локте: между векторами локоть→плечо и локоть→запястье
    ex, ey = kp["lel"]
    a1 = np.arctan2(kp["lsh"][1] - ey, kp["lsh"][0] - ex)
    a2 = np.arctan2(kp["lwr"][1] - ey, kp["lwr"][0] - ex)
    rr = 14
    p += [_path(f"M{_n(ex + rr * np.cos(a1))},{_n(ey + rr * np.sin(a1))} "
                f"A{rr},{rr} 0 0 0 {_n(ex + rr * np.cos(a2))},{_n(ey + rr * np.sin(a2))}", "ln s4")]
    p += [_t(ex - 22, ey + 4, "θ", "t4", bold=True)]
    lab = [("нос", "нос", -10, 4, "end"), ("lsh", "плечо", 8, -4, "start"), ("lel", "локоть", 9, 0, "start"),
           ("lwr", "запястье", 8, 8, "start"), ("lhip", "бедро", 9, 4, "start"), ("lkn", "колено", 9, 4, "start"),
           ("lan", "лодыжка", 9, 4, "start")]
    for key, s, dx, dy, a in lab:
        x, y = kp[key]
        p.append(_t(x + dx, y + dy, s, "mut", a=a, fs=10))
    steps = [("c6", ["изображение"]), ("c1", ["CNN"]), ("c3", ["heatmaps", "1 карта = 1 точка"]),
             ("c5", ["argmax карты", "→ (x, y), conf"]), ("c2", ["conf > 0.1 ?", "точка найдена"])]
    y = 14
    for i, (cl, lines) in enumerate(steps):
        h = 30 if len(lines) == 1 else 40
        p.append(_box(220, y, 112, h, cl, lines, fs=11))
        if i < len(steps) - 1:
            p.append(_ar(_k, 276, y + h, 276, y + h + 16))
        y += h + 16
    p += [_t(90, 278, "keypoints + скелет", "mut", fs=11)]
    return _svg(_k, 340, 290, "Скелет keypoints и конвейер heatmap", *p)


VISUALS[_k] = [
    {"title": "Keypoints и как их находят", "svg": _pose_svg(),
     "caption": "Сеть выдаёт по одной карте вероятностей (heatmap) на каждую ключевую точку; координата — "
                "максимум карты (cv2.minMaxLoc), слабые точки отбрасываются по порогу conf. "
                "По тройкам точек считают углы суставов, например θ в локте."},
]

# ───────────────────────────── positional_encoding ─────────────────────────────
_k = "positional_encoding"


def _pe_svg():
    p = [_t(170, 18, "PE(pos, 2i) = sin(pos / 10000^(2i/d))", fs=11, mono=True),
         _t(170, 34, "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))", fs=11, mono=True)]
    X0, DX = 78, 26
    rows = [(1.0, "ω = 1", 70), (10000 ** (-2 / 16), "ω ≈ 0.32", 112), (10000 ** (-4 / 16), "ω = 0.1", 154)]
    pos = np.linspace(0, 9, 120)
    for w, lab, cy in rows:
        p.append(_ln(X0, cy, X0 + 9 * DX, cy, "ln s6"))
        p.append(_poly(X0 + pos * DX, cy - 15 * np.sin(w * pos), "ln s1"))
        for k in range(10):
            cl = "t4" if k == 4 else "t1"
            p.append(_c(X0 + k * DX, cy - 15 * np.sin(w * k), 3.2 if k == 4 else 2.2, cl))
        p.append(_t(8, cy + 4, lab, "mut", a="start", fs=11))
    p.append(_ln(X0 + 4 * DX, 48, X0 + 4 * DX, 172, "ln s4", dash=True))
    for k in range(10):
        p.append(_t(X0 + k * DX, 186, str(k), "mut", fs=10))
    p.append(_t(X0 - 14, 186, "pos", "mut", a="end", fs=10))
    p += [_box(8, 200, 120, 30, "c1", "Embedding(token)", fs=11), _t(138, 220, "+", bold=True),
          _box(148, 200, 72, 30, "c3", "PE(pos)", fs=11), _t(230, 220, "=", bold=True),
          _box(240, 200, 92, 30, "c2", "вход слоя", fs=11)]
    return _svg(_k, 340, 240, "Синусоидальное позиционное кодирование", *p)


def pe_chart(fig):
    L, d = 60, 64
    pos = np.arange(L)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((L, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax.grid(False)
    ax.set_xlabel("измерение эмбеддинга (d_model = 64)"), ax.set_ylabel("позиция pos")
    ax.set_title("Sinusoidal PE: слева частые волны, справа медленные", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


VISUALS[_k] = [
    {"title": "Как устроено синусоидальное PE", "svg": _pe_svg(),
     "caption": "Каждая пара измерений — синус/косинус своей частоты, от ω = 1 до очень медленных. Вектор PE "
                "позиции — значения всех волн в этой точке (красная линия, pos = 4); он прибавляется к эмбеддингу токена."},
    {"title": "Матрица PE", "chart": pe_chart,
     "caption": "Младшие измерения меняются от позиции к позиции быстро, старшие — почти константы. "
                "Набор частот даёт каждой позиции уникальный код, а сдвиг на k — линейное преобразование."},
]

# ───────────────────────────── ppo_reinforcement_learning ─────────────────────────────
_k = "ppo_reinforcement_learning"
PPO_SVG = _svg(
    _k, 340, 270, "Итерация PPO",
    _box(40, 10, 260, 44, "c1", ["1. Rollout политикой π_old", ("N сред × T шагов → (s, a, r, log π_old, V)", "mut")], fs=11),
    _ar(_k, 170, 54, 170, 72),
    _box(40, 74, 260, 44, "c3", ["2. GAE: преимущества Aₜ", ("цели критика Rₜ = Aₜ + V(sₜ)", "mut")], fs=11),
    _ar(_k, 170, 118, 170, 136),
    _box(40, 138, 260, 56, "c2", ["3. K эпох мини-батчей, Adam", "rₜ = π_θ(a|s) / π_old(a|s)",
                                  ("L = −L_CLIP + c₁·L_VF − c₂·H", "mut")], fs=11),
    _ar(_k, 170, 194, 170, 212),
    _box(40, 214, 260, 34, "c6", "4. π_old ← π_θ, батч выбросить", fs=11),
    _path("M40,231 H14 V32 H38", "ln s6", _k, dash=True),
    _t(170, 264, "L_CLIP = E[min(rₜAₜ, clip(rₜ, 1−ε, 1+ε)·Aₜ)]", fs=11, mono=True),
)


def ppo_chart(fig):
    eps = 0.2
    r = np.linspace(0, 2, 400)
    for n, A, c in [(1, 1.0, G), (2, -1.0, R)]:
        ax = fig.add_subplot(2, 1, n)
        L = np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)
        ax.plot(r, r * A, color=GR, lw=1, ls=":", label="r·A без клипа")
        ax.plot(r, L, color=c, lw=2.2, label="L_CLIP")
        for v in (1 - eps, 1 + eps):
            ax.axvline(v, color=O, ls="--", lw=1)
        ax.plot([1], [A], "o", color="black", ms=4)
        ax.set_title(f"A {'> 0' if A > 0 else '< 0'}: " +
                     ("выгода обрезана при r > 1+ε" if A > 0 else "выгода обрезана при r < 1−ε"), fontsize=9)
        ax.set_ylabel("L_CLIP")
        ax.legend(fontsize=7, loc="upper left" if A > 0 else "upper right")
    ax.set_xlabel("ratio r = π_θ / π_old   (ε = 0.2)")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS[_k] = [
    {"title": "Одна итерация PPO", "svg": PPO_SVG,
     "caption": "Данные собирает старая политика, по ним считаются GAE-преимущества, затем несколько эпох "
                "SGD по clipped-целевой функции; после обновления батч выбрасывается — PPO on-policy."},
    {"title": "Clipped objective", "chart": ppo_chart,
     "caption": "За пределами [1−ε, 1+ε] график плоский — градиента нет, политике незачем уходить дальше. "
                "Ухудшения (A < 0 при r > 1) не обрезаются: min берёт пессимистичную оценку."},
]

# ───────────────────────────── production_ml_best_practices ─────────────────────────────
_k = "production_ml_best_practices"


def _prod_svg():
    L = [("c1", ["Данные", "Feature Store"]), ("c2", ["Обучение", "эксперименты"]),
         ("c5", ["Версионирование", "модель, данные, код"]), ("c3", ["CI/CD", "тесты, валидация"])]
    Rr = [("c4", ["Триггер retrain", "drift, падение KPI"]), ("c6", ["Мониторинг", "data / concept drift"]),
          ("c1", ["Serving", "batch / API / stream"]), ("c2", ["Деплой", "canary, A/B, shadow"])]
    ys = [12, 74, 136, 198]
    p = []
    for (cl, lines), y in zip(L, ys):
        p.append(_box(12, y, 136, 42, cl, lines, fs=11))
    for (cl, lines), y in zip(Rr, ys):
        p.append(_box(192, y, 136, 42, cl, lines, fs=11))
    for y in ys[:-1]:
        p.append(_ar(_k, 80, y + 42, 80, y + 60))
        p.append(_ar(_k, 260, y + 60, 260, y + 44))
    p.append(_ar(_k, 148, 219, 190, 219))
    p.append(_ar(_k, 192, 33, 150, 33, "ln s4"))
    p.append(_t(170, 262, "цикл не заканчивается после деплоя", "mut", fs=11))
    return _svg(_k, 340, 270, "Жизненный цикл ML-модели в продакшне", *p)


VISUALS[_k] = [
    {"title": "Жизненный цикл модели в продакшне", "svg": _prod_svg(),
     "caption": "Слева — путь к релизу: данные, обучение, версии, автоматические проверки. Справа — жизнь "
                "после релиза: постепенный деплой, serving, мониторинг дрейфа и возврат к переобучению."},
]

# ───────────────────────────── program_synthesis_ml ─────────────────────────────
_k = "program_synthesis_ml"
PS_SVG = _svg(
    _k, 340, 245, "Цикл генерация-проверка в синтезе программ",
    _box(20, 10, 300, 44, "c1", ["Спецификация: I/O-примеры", ("\"John Smith\" → \"Smith, J.\"", "mut")], fs=11),
    _ar(_k, 90, 54, 90, 88), _ar(_k, 255, 54, 255, 88),
    _box(20, 90, 140, 50, "c3", ["Генератор", "перебор / нейросеть"], fs=11),
    _box(190, 90, 130, 50, "c5", ["Верификатор", "примеры / SMT (Z3)"], fs=11),
    _ar(_k, 160, 106, 188, 106), _t(175, 80, "кандидат", "mut", fs=10),
    _ar(_k, 190, 126, 162, 126, "ln s4"), _t(175, 156, "✗ контрпример", "t4", fs=11),
    _ar(_k, 255, 140, 255, 178, "ln s2"), _t(262, 163, "✓", "t2", a="start", bold=True),
    _box(190, 180, 130, 44, "c2", ["Программа", ("ранжирование", "mut")], fs=11),
    _t(90, 182, "только в рамках DSL;", "mut", fs=10), _t(90, 196, "число программ растёт", "mut", fs=10),
    _t(90, 210, "экспоненциально", "mut", fs=10),
)


def ps_chart(fig):
    t, u, b = 4, 2, 3
    N = [float(t)]
    for _ in range(5):
        N.append(t + u * N[-1] + b * N[-1] ** 2)
    ax = fig.add_subplot(1, 1, 1)
    d = np.arange(6)
    ax.bar(d, N, color=V, alpha=0.8)
    ax.set_yscale("log")
    for x, n in zip(d, N):
        ax.text(x, n * 1.6, f"{n:.0e}" if n > 1e4 else f"{n:.0f}", ha="center", fontsize=7)
    ax.set_xlabel("максимальная глубина программы d"), ax.set_ylabel("число программ (log)")
    ax.set_ylim(1, 1e40)
    ax.set_title("DSL: 4 терминала, 2 унарные и 3 бинарные операции", fontsize=9)


VISUALS[_k] = [
    {"title": "Генерация и проверка программ", "svg": PS_SVG,
     "caption": "Генератор предлагает программу на DSL, верификатор проверяет её на спецификации; "
                "неудача даёт контрпример или отсекает ветку перебора, успех — кандидат для ранжирования."},
    {"title": "Взрыв пространства поиска", "chart": ps_chart,
     "caption": "N(d) = t + u·N(d−1) + b·N(d−1)²: уже к глубине 4–5 программ больше 10¹⁷, поэтому "
                "нужны pruning, ограниченные DSL и нейросетевые подсказки, а не слепой перебор."},
]

# ───────────────────────────── prompt_engineering ─────────────────────────────
_k = "prompt_engineering"


def _prompt_svg():
    parts = [("c5", "Role (system)", "«Ты опытный ML-инженер»"), ("c1", "Instruction", "что сделать: «Объясни GD»"),
             ("c3", "Context", "аудитория, ограничения"), ("c6", "Few-shot примеры", "вход → ответ, 2–5 штук"),
             ("c2", "Input Data", "{user_review}"), ("c4", "Output Format", "JSON / ≤ 150 слов")]
    p = []
    for i, (cl, name, ex) in enumerate(parts):
        y = 8 + i * 42
        p += [_rect(10, y, 205, 36, cl, rx=6), _t(20, y + 15, name, a="start", fs=12, bold=True),
              _t(20, y + 30, ex, "mut", a="start", fs=11)]
    p += [_path("M219,10 h6 v248 h-6", "ln"), _ar(_k, 228, 134, 248, 134),
          _box(250, 112, 80, 44, "c1", ["LLM"], fs=13),
          _ar(_k, 290, 156, 290, 196), _box(250, 198, 80, 40, "c2", ["ответ"], fs=12),
          _t(290, 92, "один prompt", "mut", fs=10)]
    return _svg(_k, 340, 262, "Из чего состоит prompt", *p)


VISUALS[_k] = [
    {"title": "Анатомия prompt", "svg": _prompt_svg(),
     "caption": "Хороший prompt собирается из блоков: роль, чёткая инструкция, контекст, примеры (few-shot), "
                "сами данные и требуемый формат ответа. Чем явнее каждый блок, тем предсказуемее ответ."},
]

# ───────────────────────────── prophet_time_series ─────────────────────────────
_k = "prophet_time_series"


def _prophet_svg():
    t = np.linspace(0, 1, 160)
    X0, W = 92, 236
    xs = X0 + t * W
    cp = 0.55
    g = np.where(t < cp, 0.6 * t, 0.6 * cp + 1.6 * (t - cp))
    s = 0.12 * np.sin(2 * np.pi * 4 * t) + 0.05 * np.sin(2 * np.pi * 8 * t + 1)
    hol = np.zeros_like(t)
    for c in (0.3, 0.8):
        hol += 0.25 * np.exp(-((t - c) / 0.012) ** 2)
    rows = [("g(t)", "тренд", g, 42, "s1"), ("s(t)", "сезонность", s, 98, "s2"), ("h(t)", "праздники", hol, 150, "s3")]
    p = [_t(170, 16, "y(t) = g(t) + s(t) + h(t) + εₜ", fs=12, bold=True)]
    for name, lab, f, cy, cl in rows:
        f0 = f - (f.max() + f.min()) / 2
        p += [_t(10, cy, name, a="start", fs=12, bold=True), _t(10, cy + 14, lab, "mut", a="start", fs=10),
              _ln(X0, cy, X0 + W, cy, "ln s6", dash=True), _poly(xs, cy - 40 * f0 / max(np.ptp(f), 0.6), cl)]
    p += [_ln(X0 + cp * W, 22, X0 + cp * W, 68, "ln s4", dash=True), _t(X0 + cp * W + 4, 66, "changepoint", "t4",
                                                                        a="start", fs=11)]
    p += [_t(80, 103, "+", bold=True, fs=14), _t(80, 155, "+", bold=True, fs=14), _ln(10, 186, 330, 186, "ln s6"),
          _t(80, 227, "=", bold=True, fs=14)]
    y = g + s + hol
    y0 = y - (y.max() + y.min()) / 2
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.03, t.size)
    p += [_t(10, 222, "y(t)", a="start", fs=12, bold=True), _t(10, 236, "прогноз", "mut", a="start", fs=10),
          _poly(xs, 222 - 28 * (y0 + noise) / np.ptp(y) * 1.6, "ln s6"), _poly(xs, 222 - 28 * y0 / np.ptp(y) * 1.6, "s5",
                                                                                extra=' stroke-width="2"')]
    return _svg(_k, 340, 260, "Аддитивная модель Prophet", *p)


def prophet_chart(fig):
    rng = np.random.default_rng(6)
    P = 365.25
    t = np.arange(0, 365)
    true = np.where((t > 330) | (t < 10), 1.5, 0) + 0.8 * np.sin(2 * np.pi * t / P) + 0.3 * np.cos(4 * np.pi * t / P)
    y = true + rng.normal(0, 0.25, t.size)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(t, y, s=4, color=GR, alpha=0.5, label="данные за год")

    def fourier(N):
        cols = [np.ones_like(t, dtype=float)]
        for n in range(1, N + 1):
            cols += [np.cos(2 * np.pi * n * t / P), np.sin(2 * np.pi * n * t / P)]
        X = np.column_stack(cols)
        return X @ np.linalg.lstsq(X, y, rcond=None)[0]

    for N, c in [(1, O), (3, B), (10, R)]:
        ax.plot(t, fourier(N), color=c, lw=2 if N != 1 else 1.5, label=f"fourier_order = {N}")
    ax.set_xlabel("день года"), ax.set_ylabel("s(t)")
    ax.set_title("Годовая сезонность рядом Фурье разного порядка", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.set_ylim(-2, 3.4)


VISUALS[_k] = [
    {"title": "Декомпозиция Prophet", "svg": _prophet_svg(),
     "caption": "Прогноз — сумма кусочно-линейного тренда с точками излома (changepoints), периодической "
                "сезонности и всплесков в праздники; каждую компоненту можно посмотреть отдельно."},
    {"title": "Порядок ряда Фурье", "chart": prophet_chart,
     "caption": "fourier_order задаёт число гармоник: при N = 1 видна только плавная волна, при N = 10 "
                "сезонность ловит и резкий пик в конце года (по умолчанию для yearly — 10), но рискует переобучиться."},
]

# ───────────────────────────── pseudo_labeling ─────────────────────────────
_k = "pseudo_labeling"
PSEUDO_SVG = _svg(
    _k, 340, 255, "Цикл pseudo-labeling",
    _box(12, 12, 140, 44, "c1", ["Размеченные L", ("мало меток", "mut")], fs=11),
    _box(188, 12, 140, 44, "c2", ["Обучить модель", ("на L + псевдо-метках", "mut")], fs=11),
    _box(12, 92, 140, 44, "c6", ["Неразмеченные U", ("много данных", "mut")], fs=11),
    _box(188, 92, 140, 44, "c5", ["Предсказать U", ("predict_proba", "mut")], fs=11),
    _box(188, 172, 140, 44, "c3", ["max P ≥ порог?", ("например 0.9", "mut")], fs=11),
    _box(12, 172, 140, 44, "c4", ["Добавить в L", ("с псевдо-метками", "mut")], fs=11),
    _ar(_k, 152, 34, 186, 34), _ar(_k, 258, 56, 258, 90), _ar(_k, 152, 114, 186, 114),
    _ar(_k, 258, 136, 258, 170), _ar(_k, 188, 194, 154, 194), _t(170, 187, "да", "t2", fs=11, bold=True),
    _path("M12,194 H5 V34 H10", "ln s6", _k, dash=True), _t(82, 158, "повторить", "mut", fs=10),
    _ar(_k, 258, 216, 258, 232), _t(258, 246, "нет → остаются в U", "mut", fs=11),
)


def pseudo_chart(fig):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(2200, n_features=10, n_informative=5, class_sep=0.9, random_state=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    lab, unl = idx[:60], idx[60:]
    m = LogisticRegression(max_iter=1000).fit(X[lab], y[lab])
    P = m.predict_proba(X[unl])
    conf, pred = P.max(1), P.argmax(1)
    th = np.linspace(0.5, 0.99, 50)
    share = [(conf >= t).mean() for t in th]
    acc = [(pred[conf >= t] == y[unl][conf >= t]).mean() for t in th]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(th, acc, color=G, lw=2, label="точность псевдо-меток")
    ax.plot(th, share, color=B, lw=2, label="доля U, взятая в обучение")
    ax.axvline(0.9, color=O, ls="--", lw=1)
    ax.text(0.905, 0.1, "0.9", color=O, fontsize=8)
    ax.set_xlabel("порог уверенности"), ax.set_ylim(0, 1.02)
    ax.set_title("Порог: качество меток против их количества", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")


VISUALS[_k] = [
    {"title": "Цикл pseudo-labeling", "svg": PSEUDO_SVG,
     "caption": "Модель, обученная на малом L, размечает U; в обучение попадают только уверенные предсказания, "
                "остальные ждут следующей итерации. Цикл повторяют, пока U не исчерпан или качество не перестало расти."},
    {"title": "Выбор порога уверенности", "chart": pseudo_chart,
     "caption": "Чем выше порог, тем точнее псевдо-метки, но тем меньше их берётся. Слишком низкий порог "
                "заливает в обучение ошибки (confirmation bias), слишком высокий почти ничего не добавляет."},
]

# ───────────────────────────── pytorch_full_guide ─────────────────────────────
_k = "pytorch_full_guide"


def _torch_svg():
    steps = [("c6", "for x, y in train_loader:", ""), ("c6", "x, y = x.to(device), y.to(device)", ""),
             ("c1", "y_hat = model(x)", "forward"), ("c3", "loss = criterion(y_hat, y)", "скаляр"),
             ("c4", "optimizer.zero_grad()", "∇ = 0"), ("c4", "loss.backward()", "∂L/∂θ"),
             ("c2", "optimizer.step()", "θ ← θ − η∇")]
    p = [_t(170, 14, "model.train()  ·  for epoch in range(N):", "mut", fs=11, mono=True)]
    for i, (cl, code, note) in enumerate(steps):
        y = 24 + i * 40
        p.append(_box(40, y, 228, 28, cl, code, fs=11, mono=True))
        if i < len(steps) - 1:
            p.append(_ar(_k, 154, y + 28, 154, y + 39))
        if note:
            p.append(_t(274, y + 18, note, "mut", a="start", fs=11))
    p.append(_path("M40,278 H26 V38 H38", "ln s6", _k, dash=True))
    p.append(_t(15, 158, "следующий batch", "mut", fs=10, extra=' transform="rotate(-90 15 158)"'))
    return _svg(_k, 340, 300, "Цикл обучения в PyTorch", *p)


def torch_chart(fig):
    ep = np.arange(0, 50)
    lr0 = 0.1
    step = lr0 * 0.1 ** (ep // 10)
    expo = lr0 * 0.9 ** ep
    cos = 0 + (lr0 - 0) * (1 + np.cos(np.pi * ep / 50)) / 2
    tot, pct = 50, 0.3
    init, final = lr0 / 25, lr0 / 25 / 1e4
    up = int(pct * tot)
    one = np.where(ep < up, init + (lr0 - init) * (1 - np.cos(np.pi * ep / up)) / 2,
                   final + (lr0 - final) * (1 + np.cos(np.pi * (ep - up) / (tot - 1 - up))) / 2)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ep, step, color=B, lw=2, drawstyle="steps-post", label="StepLR(10, γ=0.1)")
    ax.plot(ep, expo, color=G, lw=2, label="ExponentialLR(γ=0.9)")
    ax.plot(ep, cos, color=O, lw=2, label="CosineAnnealingLR(T_max=50)")
    ax.plot(ep, one, color=V, lw=2, ls="--", label="OneCycleLR(max_lr=0.1)")
    ax.set_yscale("log"), ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel("эпоха"), ax.set_ylabel("learning rate (log)")
    ax.set_title("Learning rate schedulers, lr₀ = 0.1", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")


VISUALS[_k] = [
    {"title": "Цикл обучения", "svg": _torch_svg(),
     "caption": "Порядок внутри батча фиксирован: forward → loss → zero_grad → backward → step. "
                "Без zero_grad градиенты прошлых батчей накапливаются, потому что backward их прибавляет."},
    {"title": "Расписания learning rate", "chart": torch_chart,
     "caption": "StepLR снижает lr ступенями, ExponentialLR — в γ раз каждую эпоху, Cosine плавно уводит к нулю, "
                "OneCycle сначала разгоняет lr до max_lr, затем гасит. Ось y логарифмическая."},
]

# ───────────────────────────── pytorch_lightning ─────────────────────────────
_k = "pytorch_lightning"
PL_SVG = _svg(
    _k, 340, 262, "LightningModule, DataModule и Trainer",
    _rect(10, 10, 172, 130, "c1", rx=8), _t(96, 30, "LightningModule", bold=True),
    *[_t(22, 52 + 20 * i, s, a="start", fs=11, mono=True) for i, s in
      enumerate(["forward(x)", "training_step()", "validation_step()", "configure_optimizers()"])],
    _t(96, 132, "что считать", "mut", fs=10),
    _rect(192, 10, 138, 130, "c3", rx=8), _t(261, 30, "DataModule", bold=True),
    *[_t(202, 52 + 20 * i, s, a="start", fs=11, mono=True) for i, s in
      enumerate(["setup()", "train_dataloader()", "val_dataloader()"])],
    _t(261, 132, "откуда данные", "mut", fs=10),
    _ar(_k, 96, 140, 96, 172), _ar(_k, 261, 140, 261, 172),
    _box(10, 174, 320, 50, "c2", ["trainer.fit(model, dm)", ("эпохи, backward, step, devices, AMP, callbacks", "mut")],
         fs=11, bold_first=True),
    _t(170, 248, "вы пишете «что», Trainer делает «как»", "mut", fs=11),
)

VISUALS[_k] = [
    {"title": "Кто за что отвечает", "svg": PL_SVG,
     "caption": "В LightningModule остаётся только наука: модель, шаг обучения/валидации и оптимизатор; "
                "DataModule отдаёт загрузчики. Цикл эпох, zero_grad/backward/step, перенос на GPU, "
                "multi-GPU, mixed precision и логирование берёт на себя Trainer."},
]

# ───────────────────────────── quantum_algorithms_ml ─────────────────────────────
_k = "quantum_algorithms_ml"


def _bloch_svg():
    cx, cy, r, ry, rz = 100, 120, 80, 22, 77
    psi0 = np.deg2rad(225)

    def proj(th, ph):
        return (cx + r * np.sin(th) * np.cos(ph + psi0),
                cy - ry * np.sin(th) * np.sin(ph + psi0) - rz * np.cos(th))
    sx, sy = proj(np.deg2rad(50), np.deg2rad(60))
    px, py = proj(np.pi / 2, 0)
    ang = np.arctan2(sx - cx, cy - sy)
    p = [_c(cx, cy, r, "s6"),
         f'<ellipse class="ln s6" cx="{cx}" cy="{cy}" rx="{r}" ry="{ry}" stroke-dasharray="4 3"/>',
         _ln(cx, cy - rz, cx, cy + rz, "ln s6", dash=True), _ln(cx, cy, px, py, "ln s6", dash=True),
         _c(cx, cy - rz, 4, "t1"), _c(cx, cy + rz, 4, "t4"),
         _t(cx, cy - r - 8, "|0⟩", "t1", bold=True), _t(cx, cy + r + 18, "|1⟩", "t4", bold=True),
         _c(px, py, 3.5, "t2"), _t(px - 6, py + 16, "|+⟩", "t2", a="end", fs=11),
         _ar(_k, cx, cy, sx, sy, "ln s5"), _t(sx + 6, sy - 4, "|ψ⟩", "t5", a="start", bold=True),
         _path(f"M{cx},{cy - 28} A28,28 0 0 1 {_n(cx + 28 * np.sin(ang))},{_n(cy - 28 * np.cos(ang))}", "ln s5"),
         _t(cx + 5, cy - 34, "θ", "t5", fs=11, a="start")]
    rx0 = 200
    for i, s in enumerate(["|ψ⟩ = α|0⟩ + β|1⟩", "|α|² + |β|² = 1", "P(0) = |α|²", "P(1) = |β|²"]):
        p.append(_t(rx0, 50 + 22 * i, s, a="start", fs=12 if i == 0 else 11, bold=i == 0))
    p.append(_t(rx0, 152, "H|0⟩ = (|0⟩ + |1⟩)/√2", a="start", fs=11))
    p.append(_t(rx0, 168, "= |+⟩ — суперпозиция", "mut", a="start", fs=10))
    # Bell-цепь
    y0, y1 = 236, 268
    p += [_t(10, y0 + 4, "q₀ |0⟩", a="start", fs=11), _t(10, y1 + 4, "q₁ |0⟩", a="start", fs=11),
          _ln(56, y0, 200, y0), _ln(56, y1, 200, y1),
          _box(80, y0 - 12, 26, 24, "c1", "H", fs=12, rx=3),
          _c(150, y0, 4, "ah"), _ln(150, y0, 150, y1 + 9), _c(150, y1, 9, "s6"),
          _ln(141, y1, 159, y1), _t(206, y0 + 20, "→ (|00⟩ + |11⟩)/√2", a="start", fs=11),
          _t(150, y0 - 18, "CNOT", "mut", fs=10)]
    return _svg(_k, 340, 290, "Кубит на сфере Блоха и цепь Белла", *p)


def grover_chart(fig):
    N = 64
    th = np.arcsin(1 / np.sqrt(N))
    k = np.arange(0, 15)
    pk = np.sin((2 * k + 1) * th) ** 2
    kopt = int(np.floor(np.pi / (4 * th)))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(k, pk, color=[R if x == kopt else B for x in k], alpha=0.85)
    ax.axhline(1 / N, color=GR, ls="--", lw=1, label="случайная догадка: 1/N")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(k[::2])
    ax.text(kopt, pk[kopt] + 0.03, f"k* = {kopt}", ha="center", color=R, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("число итераций Гровера k"), ax.set_ylabel("P(найти ответ)")
    ax.set_title("Гровер, N = 64: P = sin²((2k+1)θ), sin θ = 1/√N", fontsize=9)


VISUALS[_k] = [
    {"title": "Кубит, суперпозиция и запутанность", "svg": _bloch_svg(),
     "caption": "Состояние кубита — точка на сфере Блоха: полюса — |0⟩ и |1⟩, экватор — равные суперпозиции, "
                "как |+⟩ после H. Внизу: H и CNOT создают запутанное состояние Белла."},
    {"title": "Амплитудное усиление Гровера", "chart": grover_chart,
     "caption": "Каждая итерация поворачивает состояние к ответу: вероятность растёт до ≈1 за ≈(π/4)√N шагов "
                "и затем снова падает, поэтому число итераций надо выбирать, а не «чем больше, тем лучше»."},
]

# ───────────────────────────── quantum_classical_enhancement ─────────────────────────────
_k = "quantum_classical_enhancement"
QCE_SVG = _svg(
    _k, 340, 255, "Гибридный квантово-классический цикл",
    _t(170, 18, "квантовая часть (QPU)", "t5", fs=11, bold=True),
    _rect(8, 26, 324, 74, "s5", rx=8, dash=True),
    _box(18, 42, 92, 42, "c5", ["Feature map", "U(x)"], fs=11),
    _box(124, 42, 92, 42, "c5", ["Ansatz", "V(θ)"], fs=11),
    _box(230, 42, 92, 42, "c5", ["Измерение", "⟨Z⟩, shots"], fs=11),
    _ar(_k, 110, 63, 122, 63), _ar(_k, 216, 63, 228, 63),
    _rect(8, 140, 324, 80, "s1", rx=8, dash=True),
    _box(18, 158, 92, 42, "c1", ["Данные x", "классические"], fs=11),
    _box(124, 158, 92, 42, "c1", ["Оптимизатор", "COBYLA / SPSA"], fs=11),
    _box(230, 158, 92, 42, "c1", ["Cost C(θ)", "loss"], fs=11),
    _ar(_k, 64, 158, 64, 86), _ar(_k, 276, 84, 276, 156), _ar(_k, 230, 179, 218, 179),
    _ar(_k, 170, 158, 170, 86, "ln s3"), _t(176, 124, "новые θ", "t3", a="start", fs=11),
    _t(170, 240, "классическая часть (CPU)", "t1", fs=11, bold=True),
)


def qce_chart(fig):
    N = np.logspace(1, 8, 100)
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(N, N / 2, color=GR, lw=2, label="классический перебор: N/2")
    ax.loglog(N, np.pi / 4 * np.sqrt(N), color=V, lw=2, label="Гровер: (π/4)·√N")
    ax.set_xlabel("размер базы N"), ax.set_ylabel("число обращений к оракулу")
    ax.set_title("Квадратичное ускорение поиска", fontsize=9)
    ax.legend(fontsize=7)


VISUALS[_k] = [
    {"title": "Гибридный вариационный цикл", "svg": QCE_SVG,
     "caption": "Квантовый процессор только готовит состояние U(x), применяет параметрическую цепь V(θ) и "
                "измеряет; стоимость и подбор θ считает классический оптимизатор. Так устроены VQE, QAOA и VQC."},
    {"title": "Гровер против перебора", "chart": qce_chart,
     "caption": "При N = 10⁸ перебор в среднем требует 5·10⁷ запросов, Гровер — около 8·10³. Ускорение "
                "квадратичное, а не экспоненциальное, и съедается накладными расходами на загрузку данных."},
]

# ───────────────────────────── quantum_neural_networks ─────────────────────────────
_k = "quantum_neural_networks"


def _vqc_svg():
    ys = [56, 104, 152]
    p = [_t(80, 24, "кодирование", "mut", fs=10), _t(150, 24, "параметры θ", "mut", fs=10),
         _t(218, 24, "запутывание", "mut", fs=10), _t(286, 24, "измерение", "mut", fs=10)]
    for i, y in enumerate(ys):
        p += [_t(8, y + 4, "|0⟩", a="start"), _ln(32, y, 268, y),
              _box(56, y - 14, 48, 28, "c1", f"RY(x{'₁₂₃'[i]})", fs=11, rx=3),
              _box(126, y - 14, 48, 28, "c3", f"RY(θ{'₁₂₃'[i]})", fs=11, rx=3),
              _box(268, y - 14, 36, 28, "c2", "⟨Z⟩", fs=11, rx=3)]
    for xc, a, b in [(200, ys[0], ys[1]), (236, ys[1], ys[2])]:
        p += [_c(xc, a, 4, "ah"), _ln(xc, a, xc, b + 8), _c(xc, b, 8, "s6"),
              _ln(xc - 8, b, xc + 8, b)]
    p += [_path("M286,166 V192 H150 V170", "ln s4", _k, dash=True),
          _t(218, 210, "loss → ∂L/∂θ (parameter-shift) → θ", "t4", fs=11)]
    return _svg(_k, 340, 222, "Вариационная квантовая цепь", *p)


def qnn_chart(fig):
    th = np.linspace(-np.pi, np.pi, 300)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(th, np.cos(th), color=B, lw=2, label="f(θ) = ⟨Z⟩ = cos θ  (RY(θ)|0⟩)")
    ax.plot(th, -np.sin(th), color=GR, lw=1.5, ls="--", label="точная производная −sin θ")
    pts = np.linspace(-2.5, 2.5, 9)
    ps = (np.cos(pts + np.pi / 2) - np.cos(pts - np.pi / 2)) / 2
    ax.plot(pts, ps, "o", color=R, ms=5, label="[f(θ+π/2) − f(θ−π/2)] / 2")
    ax.set_xlabel("θ"), ax.set_ylim(-1.3, 2.0)
    ax.set_title("Parameter-shift rule даёт точный градиент", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


VISUALS[_k] = [
    {"title": "Схема VQC", "svg": _vqc_svg(),
     "caption": "Данные кодируются углами поворотов RY(x), обучаемые повороты RY(θ) и CNOT образуют слой; "
                "его повторяют несколько раз. Измерения ⟨Z⟩ дают выход, а градиент по θ считается на том же железе."},
    {"title": "Parameter-shift rule", "chart": qnn_chart,
     "caption": "Для гейтов-вращений градиент — разность двух запусков цепи со сдвигом ±π/2, и он точный, "
                "а не приближение конечными разностями (точки ложатся ровно на −sin θ)."},
]

# ───────────────────────────── random_forest_regression ─────────────────────────────
_k = "random_forest_regression"


def _tree(cx, y, cl):
    p = [_ln(cx, y, cx - 22, y + 24, "ln s6"), _ln(cx, y, cx + 22, y + 24, "ln s6")]
    for dx in (-22, 22):
        p += [_ln(cx + dx, y + 24, cx + dx - 10, y + 46, "ln s6"), _ln(cx + dx, y + 24, cx + dx + 10, y + 46, "ln s6")]
    p += [_c(cx, y, 6, cl), _c(cx - 22, y + 24, 5, cl), _c(cx + 22, y + 24, 5, cl)]
    p += [_c(cx + dx, y + 46, 4, "c2") for dx in (-32, -12, 12, 32)]
    return p


def _rf_svg():
    p = [_box(95, 8, 150, 32, "c6", "Данные D (n строк)", fs=11)]
    cols = [(56, "D₁", "T₁"), (170, "D₂", "T₂"), (284, "D_B", "T_B")]
    for cx, d, tname in cols:
        p += [_ar(_k, 170, 40, cx, 66), _box(cx - 46, 68, 92, 32, "c1", [f"{d} bootstrap"], fs=11)]
        p += [_ar(_k, cx, 100, cx, 116)] + _tree(cx, 124, "c3") + [_t(cx + 38, 130, tname, "t3", fs=11, bold=True)]
        p += [_ar(_k, cx, 176, 170 + (cx - 170) * 0.3, 202)]
    p += [_t(227, 150, "…", bold=True, fs=14)]
    p += [_box(70, 204, 200, 32, "c2", "ŷ(x) = (1/B)·Σ Tᵦ(x)", fs=12)]
    p += [_t(170, 256, "в каждом узле — max_features случайных признаков", "mut", fs=10),
          _t(170, 271, "≈ 37% строк вне bootstrap → OOB-оценка", "mut", fs=10)]
    return _svg(_k, 340, 280, "Bagging деревьев в Random Forest", *p)


def rf_chart(fig):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 6, 90))
    y = np.sin(x) + rng.normal(0, 0.3, x.size)
    g = np.linspace(0, 6, 400)[:, None]
    tree = DecisionTreeRegressor(min_samples_leaf=3, random_state=0).fit(x[:, None], y)
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=0).fit(x[:, None], y)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=9, color=GR, alpha=0.7, label="данные")
    ax.plot(g, np.sin(g), color="black", lw=1, ls=":", label="истинная sin x")
    ax.plot(g, tree.predict(g), color=O, lw=1.2, label="одно дерево")
    ax.plot(g, rf.predict(g), color=B, lw=2, label="Random Forest, 150 деревьев")
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.set_title("Усреднение деревьев снижает дисперсию", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")


VISUALS[_k] = [
    {"title": "Как устроен Random Forest", "svg": _rf_svg(),
     "caption": "Каждое дерево учится на своей bootstrap-выборке и в каждом узле видит лишь случайную часть "
                "признаков, поэтому деревья слабо коррелированы; прогноз регрессии — среднее по деревьям."},
    {"title": "Одно дерево против леса", "chart": rf_chart,
     "caption": "Одно дерево (min_samples_leaf=3 у обоих) подгоняется под шум и даёт рваные ступеньки; среднее "
                "150 деревьев на bootstrap-выборках гораздо ближе к истинной зависимости. За пределами данных прогноз леса — константа."},
]

# ───────────────────────────── regex_finite_state ─────────────────────────────
_k = "regex_finite_state"
REGEX_SVG = _svg(
    _k, 340, 190, "Конечный автомат для регулярного выражения ab*c",
    _t(170, 20, "regex:  ab*c", fs=13, bold=True, mono=True),
    _ar(_k, 8, 100, 36, 100), _t(10, 90, "start", "mut", a="start", fs=10),
    _c(60, 100, 22, "c1"), _t(60, 105, "q0"),
    _c(170, 100, 22, "c3"), _t(170, 105, "q1"),
    _c(280, 100, 22, "c2"), _c(280, 100, 17, "s2"), _t(280, 105, "q2"),
    _ar(_k, 82, 100, 146, 100), _t(114, 92, "a", fs=13, bold=True, mono=True),
    _ar(_k, 192, 100, 256, 100), _t(224, 92, "c", fs=13, bold=True, mono=True),
    _path("M158,81 C140,38 200,38 182,81", "ln", _k), _t(170, 38, "b", fs=13, bold=True, mono=True),
    _t(280, 140, "принимающее", "t2", fs=11),
    _t(170, 160, "abbc:  q0 →a q1 →b q1 →b q1 →c q2  ✓", fs=11, mono=True),
    _t(170, 180, "нет перехода по символу → строка отвергнута", "mut", fs=10),
)

VISUALS[_k] = [
    {"title": "Regex как конечный автомат", "svg": REGEX_SVG,
     "caption": "Каждый символ входа переводит автомат по одному ребру; b* превращается в петлю на q1. "
                "Строка подходит, если после последнего символа автомат стоит в принимающем состоянии (двойной круг)."},
]

# ───────────────────────────── regularization ─────────────────────────────
_k = "regularization"


def _reg_panel(ox, oy, t, kind, center, Q):
    """Оптимум квадратичной потери на границе ограничения и эллипс-линия уровня через него (математич. y вверх)."""
    Qi = np.array(Q)
    c = np.array(center)
    ang = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    if kind == "l1":
        u = np.stack([np.cos(ang), np.sin(ang)], 1)
        bnd = t * u / np.abs(u).sum(1, keepdims=True)
    else:
        bnd = t * np.stack([np.cos(ang), np.sin(ang)], 1)
    d = bnd - c
    lv = np.einsum("ij,jk,ik->i", d, Qi, d)
    w = bnd[np.argmin(lv)]
    level = lv.min()
    ev, evec = np.linalg.eigh(Qi)
    rot = np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
    p = []
    for f, cl in [(level, "s3"), (level * 0.4, "s3")]:
        a, b = np.sqrt(f / ev[0]), np.sqrt(f / ev[1])
        p.append(f'<ellipse class="ln {cl}" cx="{_n(ox + c[0])}" cy="{_n(oy - c[1])}" rx="{_n(a)}" ry="{_n(b)}" '
                 f'transform="rotate({_n(-rot)} {_n(ox + c[0])} {_n(oy - c[1])})"/>')
    p = [_ln(ox - 70, oy, ox + 70, oy, "ln s6"), _ln(ox, oy + 62, ox, oy - 100, "ln s6")] + p
    if kind == "l1":
        p.insert(2, f'<polygon class="c1" points="{ox + t},{oy} {ox},{oy - t} {ox - t},{oy} {ox},{oy + t}" '
                    f'fill-opacity="0.8"/>')
    else:
        p.insert(2, f'<circle class="c1" cx="{ox}" cy="{oy}" r="{t}" fill-opacity="0.8"/>')
    p += [_c(ox + c[0], oy - c[1], 3.5, "t3"), _t(ox + c[0] + 6, oy - c[1] + 4, "ŵ", "t3", a="start", bold=True),
          _c(ox + w[0], oy - w[1], 4.5, "t4")]
    return p, w


def _reg_svg():
    Q = [[1.0, 0.6], [0.6, 2.0]]
    p1, w1 = _reg_panel(85, 140, 36, "l1", (22, 48), Q)
    p2, w2 = _reg_panel(255, 140, 36, "l2", (22, 48), Q)
    assert abs(w1[0]) < 0.5 and min(w2) > 10  # L1 — касание в вершине, L2 — нет
    p = p1 + p2
    p += [_t(85, 18, "L1 (Lasso): |w₁| + |w₂| ≤ t", fs=11, bold=True),
          _t(255, 18, "L2 (Ridge): w₁² + w₂² ≤ t²", fs=11, bold=True),
          _t(85 + w1[0] - 8, 140 - w1[1] + 4, "w₁ = 0", "t4", a="end", fs=11),
          _t(152, 150, "w₁", "mut", fs=10), _t(322, 150, "w₁", "mut", fs=10),
          _t(78, 50, "w₂", "mut", a="end", fs=10), _t(248, 50, "w₂", "mut", a="end", fs=10),
          _t(85, 222, "касание в вершине →", "mut", fs=10), _t(85, 236, "часть весов ровно 0", "mut", fs=10),
          _t(255, 222, "касание на окружности →", "mut", fs=10), _t(255, 236, "веса малы, но не 0", "mut", fs=10)]
    return _svg(_k, 340, 244, "Геометрия L1 и L2 регуляризации", *p)


def reg_chart(fig):
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import lasso_path, Ridge
    X, y = load_diabetes(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    y = y - y.mean()
    al, coefs, _ = lasso_path(X, y, alphas=np.logspace(-1.5, 1.7, 60))
    ax = fig.add_subplot(2, 1, 1)
    for c, col in zip(coefs, (B, G, O, R, V, GR, B, G, O, R)):
        ax.plot(al, c, color=col, lw=1.4)
    ax.set_xscale("log"), ax.invert_xaxis()
    ax.set_title("Lasso: коэффициенты по очереди обнуляются", fontsize=9), ax.set_ylabel("вес")
    ar = np.logspace(-1, 4.5, 60)
    rc = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in ar]).T
    ax2 = fig.add_subplot(2, 1, 2)
    for c, col in zip(rc, (B, G, O, R, V, GR, B, G, O, R)):
        ax2.plot(ar, c, color=col, lw=1.4)
    ax2.set_xscale("log"), ax2.invert_xaxis()
    ax2.set_title("Ridge: веса плавно сжимаются к 0, но не равны 0", fontsize=9)
    ax2.set_xlabel("α (сила регуляризации) — растёт влево"), ax2.set_ylabel("вес")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS[_k] = [
    {"title": "Почему L1 даёт разреженность", "svg": _reg_svg(),
     "caption": "Решение — точка, где линия уровня потерь (эллипс вокруг ŵ) впервые касается допустимой области. "
                "У ромба L1 касание чаще приходится на вершину, где один вес равен нулю; у круга L2 — нет."},
    {"title": "Пути коэффициентов (diabetes)", "chart": reg_chart,
     "caption": "С ростом α Lasso выключает признаки один за другим — встроенный отбор признаков; "
                "Ridge уменьшает все веса одновременно и сохраняет их ненулевыми."},
]

# ───────────────────────────── reservoir_sampling ─────────────────────────────
_k = "reservoir_sampling"


def _reservoir_svg():
    p = [_t(170, 16, "поток: размер n заранее неизвестен", "mut", fs=11)]
    for i in range(1, 11):
        x = 15 + 28 * (i - 1)
        cl = "c6" if i < 7 else ("c3" if i == 7 else "s6")
        p += [_rect(x, 26, 24, 24, cl, rx=3), _t(x + 12, 43, f"x{i}" if i < 10 else "x10", fs=11)]
    p += [_t(305, 43, "…", bold=True), _t(195, 64, "i = 7", "t3", fs=11, bold=True)]
    p += [_t(45, 132, "reservoir", "mut", fs=11), _t(45, 148, "k = 3", "mut", fs=11)]
    for j, v in enumerate(["x2", "x5", "x4"]):
        x = 92 + 64 * j
        p += [_box(x, 118, 52, 34, "c1", v, fs=12), _t(x + 26, 168, f"слот {j + 1}", "mut", fs=10)]
    p += [_ar(_k, 195, 70, 185, 114, "ln s3", dash=True), _t(204, 96, "p = k/i = 3/7", "t3", a="start", fs=11)]
    p += [_t(12, 192, "i ≤ k: просто положить в reservoir", a="start", fs=11),
          _t(12, 210, "i > k: j = randint(1, i); если j ≤ k → R[j] = xᵢ", a="start", fs=11),
          _t(12, 228, "итог: P(xᵢ в выборке) = k/n для каждого i", "t2", a="start", fs=11, bold=True)]
    return _svg(_k, 340, 238, "Algorithm R: reservoir sampling", *p)


def reservoir_chart(fig):
    rng = np.random.default_rng(8)
    n, k, runs = 30, 5, 20000
    cnt = np.zeros(n)
    for _ in range(runs):
        res = list(range(k))
        js = rng.integers(0, np.arange(k + 1, n + 1))  # j ~ U{0..i-1} для i = k+1..n
        for i, j in zip(range(k, n), js):
            if j < k:
                res[j] = i
        cnt[res] += 1
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(1, n + 1), cnt / runs, color=B, alpha=0.8, label="частота попадания")
    ax.axhline(k / n, color=R, ls="--", lw=1.5, label=f"k/n = {k}/{n}")
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("номер элемента в потоке"), ax.set_ylabel("P(в выборке)")
    ax.set_title(f"{runs} прогонов Algorithm R, n = {n}, k = {k}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


VISUALS[_k] = [
    {"title": "Algorithm R по шагам", "svg": _reservoir_svg(),
     "caption": "Первые k элементов кладутся в reservoir целиком; i-й элемент заменяет случайный слот с "
                "вероятностью k/i. Память — O(k), один проход, длину потока знать не нужно."},
    {"title": "Проверка равномерности", "chart": reservoir_chart,
     "caption": "Ранние элементы чаще попадают в reservoir, но и чаще вытесняются; в итоге каждый из n "
                "элементов оказывается в выборке с одной и той же вероятностью k/n."},
]

# ───────────────────────────── risk_management ─────────────────────────────
_k = "risk_management"
RISK_SVG = _svg(
    _k, 340, 210, "Ожидаемые потери EL = PD · LGD · EAD",
    _t(170, 18, "Кредитный риск: три модели", "mut", fs=11),
    _box(10, 30, 96, 58, "c4", ["PD", ("вероятность", "mut"), ("дефолта", "mut")], fs=11, bold_first=True),
    _t(117, 64, "×", bold=True, fs=16),
    _box(128, 30, 96, 58, "c3", ["LGD", ("доля потерь", "mut"), ("при дефолте", "mut")], fs=11, bold_first=True),
    _t(235, 64, "×", bold=True, fs=16),
    _box(246, 30, 86, 58, "c1", ["EAD", ("сумма под", "mut"), ("риском", "mut")], fs=11, bold_first=True),
    _t(58, 106, "классификация", "mut", fs=10), _t(176, 106, "регрессия", "mut", fs=10),
    _t(289, 106, "регрессия", "mut", fs=10),
    _ar(_k, 58, 112, 140, 136), _ar(_k, 176, 112, 170, 134), _ar(_k, 289, 112, 200, 136),
    _box(70, 138, 200, 32, "c2", "EL = PD · LGD · EAD", fs=13),
    _t(170, 194, "2% · 45% · 1 000 000 = 9 000", fs=12, mono=True),
)


def risk_chart(fig):
    from scipy import stats
    rng = np.random.default_rng(9)
    ret = stats.t.rvs(df=4, size=20000, random_state=rng) * 0.01
    pnl = ret * 1000
    var = -np.percentile(pnl, 5)
    es = -pnl[pnl <= -var].mean()
    ax = fig.add_subplot(1, 1, 1)
    bins = np.linspace(-60, 60, 121)
    h, e = np.histogram(pnl, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    ax.bar(c, h, width=e[1] - e[0], color=np.where(c <= -var, R, GR), alpha=0.8)
    ax.axvline(-var, color=O, lw=2, label=f"VaR 95% = {var:.0f} тыс.")
    ax.axvline(-es, color=R, lw=2, ls="--", label=f"ES 95% = {es:.0f} тыс.")
    ax.set_xlabel("дневной P&L, тыс. (портфель 1 млн)"), ax.set_ylabel("частота")
    ax.set_title("Хвост потерь: VaR — порог, ES — среднее за ним", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


VISUALS[_k] = [
    {"title": "Ожидаемые кредитные потери", "svg": RISK_SVG,
     "caption": "Ожидаемые потери по кредиту раскладываются на три модели: вероятность дефолта (PD), долю, "
                "которую не вернут (LGD), и сумму под риском в момент дефолта (EAD)."},
    {"title": "VaR и Expected Shortfall", "chart": risk_chart,
     "caption": "VaR 95% — потеря, которую в 95% дней не превысим; он ничего не говорит о том, насколько плохо в "
                "оставшихся 5%. ES усредняет красный хвост и при тяжёлых хвостах заметно больше VaR."},
]

# ───────────────────────────── rnn_basics ─────────────────────────────
_k = "rnn_basics"


def _rnn_svg():
    xs = [80, 170, 260]
    names = [("hₜ₋₁", "xₜ₋₁", "yₜ₋₁"), ("hₜ", "xₜ", "yₜ"), ("hₜ₊₁", "xₜ₊₁", "yₜ₊₁")]
    p = [_ar(_k, 10, 115, 48, 115), _t(14, 106, "hₜ₋₂", "mut", a="start", fs=10)]
    for i, (cx, (h, x, y)) in enumerate(zip(xs, names)):
        p += [_box(cx - 30, 95, 60, 40, "c1", [h, ("tanh", "mut")], fs=11),
              _c(cx, 185, 14, "c3"), _t(cx, 189, x, fs=11),
              _ar(_k, cx, 171, cx, 137), _ar(_k, cx, 95, cx, 58),
              _c(cx, 44, 14, "c2"), _t(cx, 48, y, fs=11)]
        if i < 2:
            p.append(_ar(_k, cx + 30, 115, cx + 58, 115))
    p += [_ar(_k, 290, 115, 328, 115)]
    p += [_t(125, 90, "W_hh", "t1", fs=11), _t(215, 90, "W_hh", "t1", fs=11),
          _t(176, 158, "W_xh", "t3", a="start", fs=11), _t(176, 76, "W_hy", "t2", a="start", fs=11)]
    p += [_t(170, 222, "hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ + b_h)", fs=11, mono=True),
          _t(170, 240, "одни и те же веса на каждом шаге t", "mut", fs=10)]
    return _svg(_k, 340, 248, "Развёрнутая во времени RNN", *p)


def rnn_chart(fig):
    rng = np.random.default_rng(10)
    n, T = 32, 60
    ax = fig.add_subplot(1, 1, 1)
    for rho, c in [(0.5, R), (0.9, O), (1.2, B)]:
        W = rng.normal(size=(n, n))
        W *= rho / np.max(np.abs(np.linalg.eigvals(W)))
        U = rng.normal(size=(n, 4)) * 0.5
        h, Js = np.zeros(n), []
        for _ in range(T):
            h = np.tanh(W @ h + U @ rng.normal(size=4))
            Js.append((1 - h ** 2)[:, None] * W)
        M, norms = np.eye(n), []
        for J in reversed(Js):
            M = M @ J
            norms.append(np.linalg.norm(M, 2))
        ax.semilogy(np.arange(1, T + 1), norms, color=c, lw=2, label=f"спектр. радиус W_hh = {rho}")
    ax.set_xlabel("сколько шагов назад (T − t)"), ax.set_ylabel("‖∂h_T / ∂h_t‖")
    ax.set_title("Градиент через время затухает экспоненциально", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")


VISUALS[_k] = [
    {"title": "RNN, развёрнутая во времени", "svg": _rnn_svg(),
     "caption": "Скрытое состояние hₜ передаётся на следующий шаг и смешивается с новым входом; "
                "матрицы W_hh, W_xh, W_hy общие для всех шагов, поэтому сеть работает с последовательностью любой длины."},
    {"title": "Затухающий градиент", "chart": rnn_chart,
     "caption": "Градиент к далёким шагам — произведение якобианов diag(1 − h²)·W_hh; при малом спектральном "
                "радиусе он гаснет за десяток шагов, а насыщение tanh гасит его даже при радиусе > 1. Отсюда LSTM и GRU."},
]

# ───────────────────────────── rnn_lstm_nlp ─────────────────────────────
_k = "rnn_lstm_nlp"


def _lstm_nlp_svg():
    p = [_box(40, 8, 260, 26, "c6", "\"I love this movie\"", fs=12),
         _ar(_k, 170, 34, 170, 44),
         _box(40, 46, 260, 26, "c6", "tokenize + pad: [0, 12, 5, 3, 48]", fs=11, mono=True),
         _ar(_k, 170, 72, 170, 82),
         _box(40, 84, 260, 26, "c1", "Embedding: id → вектор (dim 128)", fs=11),
         _ar(_k, 170, 110, 170, 118)]
    xs = [49 + 52 * i for i in range(5)]
    for i, x in enumerate(xs):
        p += [_box(x, 122, 40, 24, "c2", "LSTM", fs=11, rx=4), _box(x, 158, 40, 24, "c3", "LSTM", fs=11, rx=4)]
        if i < 4:
            p += [_ar(_k, x + 40, 134, x + 51, 134), _ar(_k, x + 52, 170, x + 41, 170)]
    p += [_t(22, 138, "→", "t2", bold=True, fs=14), _t(22, 174, "←", "t3", bold=True, fs=14)]
    p += [_path("M297,134 H318 V209 H252", "ln s2", _k), _path("M49,170 H34 V209 H88", "ln s3", _k)]
    p += [_t(324, 128, "h→", "t2", a="end", fs=11), _t(30, 196, "h←", "t3", a="end", fs=11)]
    p += [_box(90, 196, 160, 26, "c5", "concat [h→ ; h←]", fs=11),
          _ar(_k, 170, 222, 170, 232), _box(90, 234, 160, 26, "c4", "Dense + sigmoid → P(+)", fs=11)]
    return _svg(_k, 340, 268, "BiLSTM для классификации текста", *p)


VISUALS[_k] = [
    {"title": "Конвейер классификации текста на BiLSTM", "svg": _lstm_nlp_svg(),
     "caption": "Текст превращается в индексы и выравнивается padding-ом, Embedding даёт векторы слов. "
                "Прямой LSTM читает слева направо, обратный — справа налево; их последние состояния "
                "склеиваются и идут в классификатор."},
]

# ───────────────────────────── rnn_regularization ─────────────────────────────
_k = "rnn_regularization"


def _rnnreg_svg():
    rng = np.random.default_rng(11)
    nh, nt, cw, ch = 5, 6, 21, 18
    std = rng.random((nh, nt)) < 0.4
    var_rows = np.array([False, True, False, False, True])
    p = []
    for ox, title, sub, mask in [(20, "Стандартный", "новая маска на каждом t", std),
                                 (190, "Variational", "одна маска на все t", np.repeat(var_rows[:, None], nt, 1))]:
        w = nt * cw
        p += [_t(ox + w / 2, 18, title, bold=True), _t(ox + w / 2, 34, sub, "mut", fs=10)]
        for i in range(nh):
            for j in range(nt):
                x, y = ox + j * cw, 46 + i * ch
                if mask[i, j]:
                    p += [_rect(x, y, cw, ch, "c4"), _t(x + cw / 2, y + 13, "×", "t4", fs=11, bold=True)]
                else:
                    p.append(_rect(x, y, cw, ch, "c1"))
        p += [_ar(_k, ox, 150, ox + w, 150), _t(ox + w / 2, 166, "время t", "mut", fs=10)]
    p += [_t(10, 100, "h", "mut", a="start", fs=11)]
    p += [_rect(20, 182, 14, 12, "c1"), _t(40, 192, "нейрон сохранён", a="start", fs=11),
          _rect(160, 182, 14, 12, "c4"), _t(180, 192, "обнулён dropout", a="start", fs=11),
          _t(170, 216, "variational: память не рвётся между шагами", "mut", fs=11)]
    return _svg(_k, 340, 226, "Стандартный и variational dropout в RNN", *p)


VISUALS[_k] = [
    {"title": "Маски dropout во времени", "svg": _rnnreg_svg(),
     "caption": "Строки — нейроны скрытого состояния, столбцы — шаги времени. Обычный dropout каждый шаг "
                "выключает разные нейроны и разрушает передачу памяти; variational dropout выключает одни и те же "
                "нейроны на всей последовательности."},
]

# ───────────────────────────── rule_based_ml_systems ─────────────────────────────
_k = "rule_based_ml_systems"
RULE_SVG = _svg(
    _k, 340, 252, "Последовательная гибридная система: правила и ML",
    _box(55, 8, 120, 28, "c6", "Вход x", fs=12),
    _ar(_k, 115, 36, 115, 54),
    _box(20, 56, 190, 42, "c3", ["Правила: IF условие", "THEN действие"], fs=11),
    _ar(_k, 210, 77, 236, 77), _t(223, 70, "да", "t2", fs=11, bold=True),
    _box(238, 56, 92, 42, "c4", ["решение", "по правилу"], fs=11),
    _ar(_k, 115, 98, 115, 124), _t(122, 115, "нет", "mut", a="start", fs=11),
    _box(20, 126, 190, 42, "c1", ["ML-модель", ("predict_proba(x)", "mut")], fs=11),
    _ar(_k, 115, 168, 115, 194),
    _box(20, 196, 190, 42, "c5", ["Проверка правилами", ("бизнес-ограничения", "mut")], fs=11),
    _ar(_k, 210, 217, 236, 217),
    _box(238, 196, 92, 42, "c2", ["итоговое", "решение"], fs=11),
    _ar(_k, 284, 98, 284, 194),
)


def fuzzy_chart(fig):
    x = np.linspace(0, 10, 501)

    def trimf(x, a, b, c):
        left = np.where(b > a, (x - a) / max(b - a, 1e-9), 1.0)
        right = np.where(c > b, (c - x) / max(c - b, 1e-9), 1.0)
        return np.clip(np.minimum(np.where(x <= b, left, right), 1), 0, 1)

    ax = fig.add_subplot(1, 1, 1)
    for (a, b, c), name, col in [((0, 0, 5), "low [0, 0, 5]", B), ((0, 5, 10), "medium [0, 5, 10]", G),
                                 ((5, 10, 10), "high [5, 10, 10]", R)]:
        ax.plot(x, trimf(x, a, b, c), color=col, lw=2, label=name)
    ax.axvline(7, color=GR, ls="--", lw=1)
    ax.plot([7, 7], [0.6, 0.4], "o", color="black", ms=4)
    ax.text(7.15, 0.62, "μ_medium = 0.6", fontsize=8)
    ax.text(7.15, 0.33, "μ_high = 0.4", fontsize=8)
    ax.set_xlabel("температура x"), ax.set_ylabel("степень принадлежности μ")
    ax.set_title("Нечёткие множества fuzz.trimf", fontsize=9)
    ax.set_ylim(0, 1.38)
    ax.legend(fontsize=7, loc="upper center", ncol=3)


VISUALS[_k] = [
    {"title": "Правила + ML в одной системе", "svg": RULE_SVG,
     "caption": "Жёсткие правила обрабатывают очевидные и регуляторные случаи сразу, остальное решает "
                "ML-модель, а её ответ проходит финальную проверку бизнес-правилами."},
    {"title": "Нечёткие функции принадлежности", "chart": fuzzy_chart,
     "caption": "В fuzzy logic значение принадлежит нескольким множествам частично: x = 7 — это medium на 0.6 "
                "и high на 0.4. Правила IF … THEN срабатывают с такой же частичной силой."},
]

# ───────────────────────────── self_attention ─────────────────────────────
_k = "self_attention"


def _sa_svg():
    p = [_box(125, 8, 90, 28, "c6", "X  (n×d)", fs=12),
         _ar(_k, 150, 36, 60, 64), _ar(_k, 170, 36, 170, 64), _ar(_k, 190, 36, 280, 64),
         _box(20, 66, 82, 30, "c1", "Q = X·W_Q", fs=11), _box(129, 66, 82, 30, "c3", "K = X·W_K", fs=11),
         _box(238, 66, 82, 30, "c2", "V = X·W_V", fs=11),
         _ar(_k, 61, 96, 95, 128), _ar(_k, 170, 96, 150, 128),
         _box(20, 130, 190, 34, "c5", "S = Q·Kᵀ / √d_k + M", fs=12),
         _ar(_k, 115, 164, 115, 182),
         _box(20, 184, 190, 34, "c4", ["A = softmax(S)", ("по строкам, Σⱼ aᵢⱼ = 1", "mut")], fs=11),
         _ar(_k, 115, 218, 150, 248),
         _path("M305,96 V263 H262", "ln s2", _k),
         _box(110, 248, 150, 30, "c1", "Z = A·V  (n×d_v)", fs=12)]
    # causal mask
    ox, oy, s = 222, 136, 17
    p += [_t(ox + 2 * s, oy - 6, "causal M", "mut", fs=10)]
    for i in range(4):
        for j in range(4):
            if j <= i:
                p.append(_rect(ox + j * s, oy + i * s, s, s, "c1"))
            else:
                p += [_rect(ox + j * s, oy + i * s, s, s, "c6"), _t(ox + j * s + s / 2, oy + i * s + 12, "−∞", "mut",
                                                                  fs=10)]
    return _svg(_k, 340, 286, "Scaled dot-product self-attention", *p)


def sa_chart(fig):
    rng = np.random.default_rng(12)
    dks = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    n, reps = 32, 200
    raw, sc = [], []
    for dk in dks:
        q = rng.normal(size=(reps, 1, dk))
        k = rng.normal(size=(reps, n, dk))
        s = (q * k).sum(-1)
        for arr, z in [(raw, s), (sc, s / np.sqrt(dk))]:
            e = np.exp(z - z.max(1, keepdims=True))
            arr.append((e / e.sum(1, keepdims=True)).max(1).mean())
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(dks, raw, "o-", color=R, lw=2, base=2, label="softmax(q·k) — без масштаба")
    ax.semilogx(dks, sc, "o-", color=B, lw=2, base=2, label="softmax(q·k / √d_k)")
    ax.axhline(1 / n, color=GR, ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("d_k"), ax.set_ylabel("средний максимальный вес")
    ax.set_title(f"Без √d_k softmax почти one-hot (n = {n})", fontsize=9)
    ax.legend(fontsize=7, loc="center right")


VISUALS[_k] = [
    {"title": "Scaled dot-product attention", "svg": _sa_svg(),
     "caption": "Q, K и V — три проекции одной и той же X. Матрица оценок n×n масштабируется на √d_k, маска "
                "добавляет −∞ к запрещённым парам (causal — будущее), softmax по строкам даёт веса для смеси строк V."},
    {"title": "Зачем делить на √d_k", "chart": sa_chart,
     "caption": "Дисперсия q·k растёт как d_k, и без масштаба softmax почти весь вес отдаёт одному токену — "
                "градиенты пропадают. После деления распределение весов остаётся мягким при любом d_k."},
]

# ───────────────────────────── self_training ─────────────────────────────
_k = "self_training"


def _st_svg():
    X0, W = 90, 238
    its = [(0.10, 0.0), (0.10, 0.25), (0.35, 0.20), (0.55, 0.15)]
    p = [_rect(12, 10, 14, 12, "c1"), _t(31, 20, "L (истинные)", a="start", fs=11),
         _rect(118, 10, 14, 12, "c5"), _t(137, 20, "новые псевдо", a="start", fs=11),
         _rect(232, 10, 14, 12, "c6"), _t(251, 20, "U (без меток)", a="start", fs=11)]
    for i, (l, new) in enumerate(its):
        y = 40 + i * 40
        u = 1 - l - new
        p.append(_t(10, y + 17, f"итерация {i}", a="start", fs=11))
        x = X0
        for frac, cl, txt in [(l, "c1", f"{int(round(l * 100))}%"), (new, "c5", f"+{int(round(new * 100))}%"),
                              (u, "c6", f"{int(round(u * 100))}%")]:
            if frac <= 0:
                continue
            p.append(_rect(x, y, W * frac, 26, cl))
            if W * frac > 30:
                p.append(_t(x + W * frac / 2, y + 17, txt, fs=11))
            x += W * frac
    p += [_t(170, 216, "SelfTrainingClassifier(base, threshold=0.75)", fs=11, mono=True),
          _t(170, 232, "y = −1 для неразмеченных; цикл до max_iter или пока U не пуст", "mut", fs=10)]
    return _svg(_k, 340, 240, "Рост размеченной выборки при self-training", *p)


def st_chart(fig):
    from sklearn.datasets import make_moons
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    X, y = make_moons(300, noise=0.12, random_state=1)
    rng = np.random.default_rng(4)
    lab = np.concatenate([rng.choice(np.where(y == c)[0], 4, replace=False) for c in (0, 1)])
    yy = np.full_like(y, -1)
    yy[lab] = y[lab]
    base = GaussianProcessClassifier(10.0 * RBF(0.3), optimizer=None)
    sup = GaussianProcessClassifier(10.0 * RBF(0.3), optimizer=None).fit(X[lab], y[lab])
    st = SelfTrainingClassifier(base, criterion="k_best", k_best=10, max_iter=None).fit(X, yy)
    gx, gy = np.meshgrid(np.linspace(-1.5, 2.5, 160), np.linspace(-1, 1.5, 120))
    G2 = np.c_[gx.ravel(), gy.ravel()]
    for n, (m, title) in enumerate([(sup, "только 8 меток: acc = {:.2f}"),
                                    (st, "self-training, k_best=10: acc = {:.2f}")], 1):
        ax = fig.add_subplot(2, 1, n)
        ax.contourf(gx, gy, m.predict(G2).reshape(gx.shape), levels=[-0.5, 0.5, 1.5], colors=[B, O], alpha=0.15)
        ax.scatter(X[:, 0], X[:, 1], s=5, color=GR, alpha=0.5)
        ax.scatter(X[lab, 0], X[lab, 1], s=40, c=[B if t == 0 else O for t in y[lab]], edgecolor="black")
        ax.set_title(title.format((m.predict(X) == y).mean()), fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS[_k] = [
    {"title": "Как растёт L по итерациям", "svg": _st_svg(),
     "caption": "На каждой итерации уверенные предсказания (P ≥ threshold) переносятся из U в L как псевдо-метки, "
                "и модель переобучается на расширенной выборке. Ошибочные псевдо-метки тоже остаются в L навсегда."},
    {"title": "Две луны: 8 меток и 292 без меток", "chart": st_chart,
     "caption": "По восьми точкам классификатор (GP) проводит грубую границу; self-training на каждой итерации "
                "добавляет 10 самых уверенных точек, и граница огибает структуру данных. Работает, когда классы разделены областями малой плотности."},
]

# ───────────────────────────── sequence_aware_recommendations ─────────────────────────────
_k = "sequence_aware_recommendations"
SEQREC_SVG = _svg(
    _k, 340, 236, "Next-item prediction в сессии",
    _t(170, 14, "сессия пользователя", "mut", fs=10),
    _box(20, 22, 50, 30, "c6", "A", fs=12), _box(95, 22, 50, 30, "c6", "B", fs=12),
    _box(170, 22, 50, 30, "c6", "C", fs=12), _box(245, 22, 50, 30, "s4", "?", fs=13),
    _ar(_k, 70, 37, 93, 37), _ar(_k, 145, 37, 168, 37), _ar(_k, 220, 37, 243, 37),
    _t(45, 66, "кроссовки", "mut", fs=10), _t(120, 66, "носки", "mut", fs=10), _t(195, 66, "шорты", "mut", fs=10),
    _t(312, 41, "t →", "mut", a="start", fs=10),
    _ar(_k, 188, 72, 110, 108, "ln s3"),
    _ar(_k, 45, 72, 230, 108, "ln s5", dash=True), _ar(_k, 120, 72, 240, 108, "ln s5", dash=True),
    _ar(_k, 205, 72, 255, 108, "ln s5", dash=True),
    _box(12, 110, 150, 46, "c3", ["Markov, 1-й порядок", "P(i | C)"], fs=11),
    _box(178, 110, 150, 46, "c5", ["GRU4Rec / SASRec", "P(i | A, B, C)"], fs=11),
    _ar(_k, 87, 156, 140, 180), _ar(_k, 253, 156, 200, 180),
    _box(90, 182, 160, 44, "c2", ["top-k следующих", ("Recall@k, MRR, NDCG", "mut")], fs=11),
)

VISUALS[_k] = [
    {"title": "Предсказание следующего item", "svg": SEQREC_SVG,
     "caption": "Марковская цепь смотрит только на последний item сессии, а RNN и self-attention модели "
                "сжимают всю историю в состояние. Ответ — ранжированный список кандидатов на следующий шаг."},
]

# ───────────────────────────── shap_neural_networks ─────────────────────────────
_k = "shap_neural_networks"


def _shap_svg():
    base = 0.30
    feats = [("возраст = 45", 0.15), ("доход = 80k", 0.10), ("долг = 3", -0.08), ("стаж = 7", 0.05)]
    X0, S = 110, 525.0

    def px(v):
        return X0 + (v - 0.2) * S

    p = []
    cur = base
    for i, (name, phi) in enumerate(feats):
        y = 30 + i * 36
        a, b = sorted([cur, cur + phi])
        cl = "c4" if phi > 0 else "c1"
        p += [_t(8, y + 15, name, a="start", fs=11), _rect(px(a), y, px(b) - px(a), 22, cl, rx=2)]
        lab = f"{phi:+.2f}"
        if px(b) - px(a) > 36:
            p.append(_t((px(a) + px(b)) / 2, y + 15, lab, fs=11))
        else:
            p.append(_t(px(b) + 4, y + 15, lab, "t4" if phi > 0 else "t1", a="start", fs=11))
        if i < len(feats) - 1:
            p.append(_ln(px(cur + phi), y + 22, px(cur + phi), y + 36, "ln s6"))
        cur += phi
    fx = cur
    p += [_ln(px(base), 22, px(base), 172, "ln s6", dash=True), _ln(px(fx), 140, px(fx), 192, "ln s6", dash=True),
          _t(px(base), 188, f"E[f(X)] = {base:.2f}", "mut", fs=11), _t(px(fx), 206, f"f(x) = {fx:.2f}", fs=11, bold=True),
          _t(170, 230, "f(x) = E[f(X)] + Σ φᵢ", fs=12, bold=True),
          _rect(10, 240, 12, 10, "c4"), _t(26, 249, "повышает", a="start", fs=11),
          _rect(100, 240, 12, 10, "c1"), _t(116, 249, "понижает прогноз", a="start", fs=11)]
    return _svg(_k, 340, 258, "SHAP waterfall для одного предсказания", *p)


VISUALS[_k] = [
    {"title": "Waterfall: из базового значения в прогноз", "svg": _shap_svg(),
     "caption": "Объяснение начинается со среднего прогноза E[f(X)], каждый признак сдвигает его на своё φᵢ; "
                "сумма всех сдвигов точно равна f(x) − E[f(X)] (свойство эффективности)."},
]

# ───────────────────────────── shape_constraints ─────────────────────────────
_k = "shape_constraints"


def _shapes_svg():
    t = np.linspace(0, 1, 60)
    cells = [("Монотонность ↑", "∂f/∂x ≥ 0", 0.15 + 0.7 * (t + 0.15 * np.sin(5 * t)) / 1.0, "s1"),
             ("Выпуклость", "∂²f/∂x² ≥ 0", 0.1 + 0.8 * t ** 2.5, "s3"),
             ("Вогнутость", "∂²f/∂x² ≤ 0", 0.1 + 0.8 * np.sqrt(t), "s2"),
             ("Унимодальность", "рост до пика, затем спад", 0.1 + 0.8 * np.exp(-((t - 0.45) / 0.22) ** 2), "s5")]
    p = []
    for idx, (title, f, y, cl) in enumerate(cells):
        ox, oy = 12 + (idx % 2) * 166, 8 + (idx // 2) * 118
        w, h = 150, 62
        p += [_t(ox + w / 2, oy + 12, title, fs=11, bold=True),
              _ln(ox + 8, oy + 22 + h, ox + w, oy + 22 + h, "ln s6"), _ln(ox + 8, oy + 22, ox + 8, oy + 22 + h, "ln s6"),
              _poly(ox + 10 + t * (w - 14), oy + 22 + h - y * h, cl, extra=' stroke-width="2.2"'),
              _t(ox + w / 2, oy + 102, f, "mut", fs=10)]
    return _svg(_k, 340, 240, "Виды shape constraints", *p)


def shapes_chart(fig):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(13)
    x = np.sort(np.concatenate([rng.uniform(0, 4, 70), rng.uniform(6, 10, 70), rng.uniform(4, 6, 6)]))
    y = np.log1p(x) * 2 + rng.normal(0, 0.6, x.size)
    g = np.linspace(0, 10, 400)
    kw = dict(max_iter=150, learning_rate=0.1, min_samples_leaf=5, random_state=0)
    free = HistGradientBoostingRegressor(**kw).fit(x[:, None], y)
    mono = HistGradientBoostingRegressor(monotonic_cst=[1], **kw).fit(x[:, None], y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=7, color=GR, alpha=0.6, label="данные (зависимость растёт)")
    ax.plot(g, free.predict(g[:, None]), color=R, lw=1.4, label="бустинг без ограничений")
    ax.plot(g, mono.predict(g[:, None]), color=B, lw=2, label="бустинг, monotonic_cst=[1]")
    ax.plot(g, iso.predict(g), color=G, lw=1.6, ls="--", label="IsotonicRegression")
    ax.set_xlabel("x"), ax.set_ylabel("f(x)")
    ax.set_title("Монотонность убирает ложные провалы", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


VISUALS[_k] = [
    {"title": "Какие формы можно потребовать", "svg": _shapes_svg(),
     "caption": "Ограничение формы — априорное знание о f: не убывает, выпукла, вогнута или имеет один пик. "
                "Модель ищет решение только среди функций такой формы."},
    {"title": "Монотонный бустинг и изотоническая регрессия", "chart": shapes_chart,
     "caption": "Без ограничения бустинг подстраивается под шум и местами «падает», хотя по смыслу зависимость "
                "растёт; monotonic_cst и IsotonicRegression гарантируют неубывающий прогноз."},
]

# ───────────────────────────── simulated_annealing ─────────────────────────────
_k = "simulated_annealing"


def _sa_energy(x):
    return 0.55 + 0.25 * np.cos(3.1 * x) + 0.18 * (x - 2.3) ** 2 / 4 - 0.12 * np.exp(-((x - 2.9) / 0.4) ** 2)


def _anneal_svg():
    xs = np.linspace(0, 4.6, 200)
    E = _sa_energy(xs)
    X0, W, Y0, H = 20, 300, 30, 150

    def P(x, e):
        return X0 + x / 4.6 * W, Y0 + H - (e - E.min()) / np.ptp(E) * (H - 20)

    sx, sy = P(xs, E)
    i_loc = np.argmin(E[:60])
    i_glob = np.argmin(E)
    i_top = 60 + np.argmax(E[60:i_glob])
    lx, ly = P(xs[i_loc], E[i_loc])
    gx, gy = P(xs[i_glob], E[i_glob])
    tx, ty = P(xs[i_top], E[i_top])
    p = [_poly(sx, sy, "s6", extra=' stroke-width="2"'),
         _c(lx, ly - 8, 7, "c3"), _c(gx, gy - 7, 5, "t2"),
         _path(f"M{_n(lx + 6)},{_n(ly - 16)} Q{_n((lx + tx) / 2)},{_n(ty - 50)} {_n(tx)},{_n(ty - 12)}", "ln s4", _k,
               dash=True),
         _t((lx + tx) / 2, ty - 44, "ΔE > 0", "t4", fs=11, bold=True),
         _t(lx, ly + 18, "локальный", "mut", fs=10), _t(lx, ly + 30, "минимум", "mut", fs=10),
         _t(gx, gy + 14, "глобальный", "t2", fs=11)]
    p += [_t(170, 18, "E(x)", "mut", fs=11),
          _t(12, 206, "ΔE ≤ 0 → принять всегда", a="start", fs=11),
          _t(12, 224, "ΔE > 0 → принять с P = exp(−ΔE / T)", a="start", fs=11, bold=True),
          _t(12, 244, "T велика: исследование  ·  T → 0: жадный спуск", "mut", a="start", fs=10)]
    return _svg(_k, 340, 254, "Simulated annealing выбирается из локального минимума", *p)


def anneal_chart(fig):
    ax = fig.add_subplot(2, 1, 1)
    dE = np.linspace(0, 10, 300)
    for T, c in [(0.5, R), (1, O), (10, B)]:
        ax.plot(dE, np.exp(-dE / T), color=c, lw=2, label=f"T = {T}")
    ax.set_xlabel("ухудшение ΔE"), ax.set_ylabel("P(принять)")
    ax.set_title("Критерий Метрополиса: P = exp(−ΔE / T)", fontsize=9)
    ax.legend(fontsize=7)
    ax2 = fig.add_subplot(2, 1, 2)
    k = np.arange(0, 201)
    T0 = 10
    ax2.plot(k, T0 * 0.97 ** k, color=B, lw=2, label="геометрическое α = 0.97")
    ax2.plot(k, np.maximum(T0 - 0.05 * k, 0), color=G, lw=2, label="линейное η = 0.05")
    ax2.plot(k[1:], T0 / (1 + k[1:]), color=O, lw=2, label="Коши T0/(1+k)")
    ax2.plot(k[1:], T0 * np.log(2) / np.log(1 + k[1:]), color=V, lw=2, ls="--", label="логарифмическое c/log(1+k)")
    ax2.set_ylim(0, 10.5)
    ax2.set_xlabel("итерация k"), ax2.set_ylabel("T")
    ax2.set_title("Расписания охлаждения, T0 = 10", fontsize=9)
    ax2.legend(fontsize=7)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS[_k] = [
    {"title": "Шаг «в гору»", "svg": _anneal_svg(),
     "caption": "Hill climbing застрял бы в локальном минимуме. SA иногда принимает ухудшение ΔE > 0 и "
                "переваливает через барьер; чем ниже температура, тем реже это происходит."},
    {"title": "Вероятность принятия и охлаждение", "chart": anneal_chart,
     "caption": "Малые ухудшения принимаются легче больших, а высокая T делает почти любой шаг допустимым. "
                "Геометрическое расписание — стандарт; логарифмическое гарантирует сходимость, но остывает крайне медленно."},
]

# ───────────────────────────── sklearn_overview ─────────────────────────────
_k = "sklearn_overview"
SK_SVG = _svg(
    _k, 340, 256, "Pipeline внутри GridSearchCV",
    _t(170, 16, "GridSearchCV(pipe, param_grid, cv=5)", fs=11, mono=True, bold=True),
    _rect(8, 24, 324, 186, "s6", rx=8, dash=True),
    _t(110, 44, "Pipeline", "mut", fs=11), _t(232, 44, "train", "t2", fs=11, bold=True),
    _t(298, 44, "test", "t1", fs=11, bold=True),
    _box(35, 54, 150, 36, "c3", "StandardScaler", fs=12),
    _box(35, 110, 150, 36, "c3", "PCA(n_components)", fs=12),
    _box(35, 166, 150, 36, "c1", "RandomForest…", fs=12),
    _ar(_k, 110, 90, 110, 108), _ar(_k, 110, 146, 110, 164),
    _t(232, 76, "fit_transform", "t2", fs=11), _t(298, 76, "transform", "t1", fs=11),
    _t(232, 132, "fit_transform", "t2", fs=11), _t(298, 132, "transform", "t1", fs=11),
    _t(232, 188, "fit", "t2", fs=11), _t(298, 188, "predict", "t1", fs=11),
    _t(170, 232, "единый API: fit / transform / predict / score", "mut", fs=11),
    _t(170, 249, "лучшие параметры → best_estimator_", "mut", fs=11),
)

VISUALS[_k] = [
    {"title": "Pipeline и единый API", "svg": SK_SVG,
     "caption": "На обучении каждый transformer делает fit_transform, финальная модель — fit; на тесте те же "
                "шаги только transform и predict, поэтому статистики не утекают из теста. GridSearchCV перебирает "
                "параметры всего Pipeline с кросс-валидацией."},
]

# ───────────────────────────── smote_imbalanced ─────────────────────────────
_k = "smote_imbalanced"


def _smote_svg():
    rng = np.random.default_rng(14)
    maj = rng.normal(size=(45, 2)) * [60, 38] + [200, 105]
    maj = maj[(maj[:, 0] > 120) & (maj[:, 0] < 330) & (maj[:, 1] > 25) & (maj[:, 1] < 175)]
    mino = np.array([[85, 105], [40, 70], [130, 80], [70, 155], [25, 150], [150, 165], [20, 40]])
    xi, nn = mino[0], [1, 2, 3]
    xn = mino[2]
    xnew = xi + 0.4 * (xn - xi)
    r = np.linalg.norm(mino[nn] - xi, axis=1).max() + 10
    p = [_c(x, y, 3, "t6") for x, y in maj if x > 165]
    p.append(_c(xi[0], xi[1], r, "s4").replace("/>", ' stroke-dasharray="4 3"/>'))
    p += [_ln(*xi, *mino[j], "ln s6", dash=True) for j in nn]
    p += [_ln(*xi, *xn, "ln s4")]
    p += [_c(x, y, 4.5, "t4") for x, y in mino]
    p += [_c(*xnew, 5, "c3"), _c(*xi, 7, "s4")]
    p += [_t(xi[0] - 8, xi[1] + 16, "xᵢ", "t4", a="end", bold=True),
          _t(xn[0] - 2, xn[1] - 10, "x_nn", "t4", a="end", fs=11),
          _t(xnew[0] + 4, xnew[1] + 20, "x_new", "t3", a="start", fs=11, bold=True),
          _t(xi[0], xi[1] - r - 7, "k = 3 ближайших", "t4", fs=11),
          _t(255, 22, "мажорный класс", "mut", fs=10)]
    p += [_t(170, 202, "x_new = xᵢ + λ·(x_nn − xᵢ),  λ ~ U(0, 1)", fs=11, mono=True),
          _t(170, 220, "соседи ищутся только внутри минорного класса", "mut", fs=10)]
    return _svg(_k, 340, 228, "Как SMOTE создаёт синтетический пример", *p)


def smote_chart(fig):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(15)
    Xa = rng.normal(size=(400, 2)) * [1.2, 1.0]
    Xb = rng.normal(size=(20, 2)) * 0.45 + [2.2, 1.6]
    nn = NearestNeighbors(n_neighbors=6).fit(Xb)
    _, idx = nn.kneighbors(Xb)
    base = rng.integers(0, len(Xb), 380)
    nb = idx[base, rng.integers(1, 6, 380)]
    syn = Xb[base] + rng.random((380, 1)) * (Xb[nb] - Xb[base])
    for n, (title, extra) in enumerate([("До: 400 : 20", None), ("После SMOTE (k = 5): 400 : 400", syn)], 1):
        ax = fig.add_subplot(2, 1, n)
        ax.scatter(Xa[:, 0], Xa[:, 1], s=5, color=GR, alpha=0.5, label="класс 0")
        if extra is not None:
            ax.scatter(extra[:, 0], extra[:, 1], s=6, color=O, alpha=0.6, label="синтетические")
        ax.scatter(Xb[:, 0], Xb[:, 1], s=16, color=R, label="класс 1 (исходные)")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-4, 4.5), ax.set_ylim(-3.5, 3.5)
        ax.legend(fontsize=7, loc="lower left")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS[_k] = [
    {"title": "Интерполяция SMOTE", "svg": _smote_svg(),
     "caption": "Для точки минорного класса берётся один из k ближайших соседей того же класса, и новая точка "
                "ставится в случайное место отрезка между ними — это не копия, а интерполяция."},
    {"title": "До и после SMOTE", "chart": smote_chart,
     "caption": "Синтетические точки заполняют выпуклую область минорного класса и выравнивают баланс. "
                "Применять только к train внутри CV (imblearn Pipeline), иначе синтетика утечёт в валидацию."},
]

# ───────────────────────────── social_network_analysis ─────────────────────────────
_k = "social_network_analysis"
_SNA_POS = {"L0": (75, 112), "L1": (38, 66), "L2": (108, 58), "L3": (30, 158), "L4": (112, 166), "L5": (22, 112),
            "B": (170, 112), "R0": (232, 70), "R1": (292, 58), "R2": (310, 122), "R3": (282, 176), "R4": (228, 154)}
_SNA_EDGES = [("L0", "L1"), ("L0", "L2"), ("L0", "L3"), ("L0", "L4"), ("L0", "L5"), ("L1", "L2"), ("L1", "L5"),
              ("L3", "L4"), ("L3", "L5"), ("L2", "B"), ("L4", "B"), ("B", "R0"), ("B", "R4"), ("R0", "R1"),
              ("R1", "R2"), ("R0", "R2"), ("R2", "R3"), ("R3", "R4"), ("R2", "R4")]


def _betweenness(nodes, edges):
    """Betweenness (Brandes) для невзвешенного неориентированного графа."""
    adj = {v: [] for v in nodes}
    for a, b in edges:
        adj[a].append(b), adj[b].append(a)
    bc = dict.fromkeys(nodes, 0.0)
    for s in nodes:
        S, P, sig, dist, Q = [], {v: [] for v in nodes}, dict.fromkeys(nodes, 0), dict.fromkeys(nodes, -1), [s]
        sig[s], dist[s] = 1, 0
        while Q:
            v = Q.pop(0)
            S.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sig[w] += sig[v]
                    P[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += sig[v] / sig[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return {v: x / 2 for v, x in bc.items()}


def _sna_svg():
    nodes = list(_SNA_POS)
    deg = {v: sum(v in e for e in _SNA_EDGES) for v in nodes}
    bc = _betweenness(nodes, _SNA_EDGES)
    hub, bridge = max(deg, key=deg.get), max(bc, key=bc.get)
    p = [_ln(*_SNA_POS[a], *_SNA_POS[b], "ln s6") for a, b in _SNA_EDGES]
    for v, (x, y) in _SNA_POS.items():
        cl = "c1" if v.startswith("L") else ("c2" if v.startswith("R") else "c3")
        p.append(_c(x, y, 13 if v in (hub, bridge) else 9, cl))
    hx, hy = _SNA_POS[hub]
    bx, by = _SNA_POS[bridge]
    p += [_t(hx, hy + 4, str(deg[hub]), fs=11, bold=True), _t(bx, by + 4, "B", fs=11, bold=True)]
    p += [_t(90, 200, f"хаб: degree = {deg[hub]} (макс.)", "t1", fs=11),
          _t(bx, by - 24, "мост", "t3", fs=11, bold=True), _t(248, 200, f"мост B: betweenness = {bc[bridge]:.0f} (макс.)", "t3", fs=11),
          _t(70, 20, "сообщество 1", "t1", fs=11, bold=True), _t(268, 20, "сообщество 2", "t2", fs=11, bold=True),
          _t(170, 222, "degree — сколько связей; betweenness — сколько", "mut", fs=10),
          _t(170, 236, "кратчайших путей проходит через узел", "mut", fs=10)]
    return _svg(_k, 340, 244, "Центральности узлов и сообщества", *p)


def sna_chart(fig):
    rng = np.random.default_rng(16)
    n, m = 3000, 2
    targets, deg = [], np.zeros(n, int)
    rep = [0, 1, 1, 2, 2, 0]
    deg[:3] = 2
    for v in range(3, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rep[rng.integers(len(rep))])
        for u in chosen:
            rep += [u, v]
            deg[u] += 1
            deg[v] += 1
    p = 2 * m / n
    ne = rng.binomial(n * (n - 1) // 2, p)
    a, b = rng.integers(0, n, ne), rng.integers(0, n, ne)
    ok = a != b
    deg_er = np.bincount(np.concatenate([a[ok], b[ok]]), minlength=n)
    ax = fig.add_subplot(1, 1, 1)
    for d, col, lab in [(deg, R, "Барабаши–Альберт (m = 2)"), (deg_er, B, "Эрдёш–Реньи, тот же средний k")]:
        k, c = np.unique(d[d > 0], return_counts=True)
        ax.loglog(k, c / n, "o", ms=3.5, color=col, label=lab)
    ax.set_xlabel("степень узла k"), ax.set_ylabel("P(k)")
    ax.set_title(f"Распределение степеней, n = {n}", fontsize=9)
    ax.legend(fontsize=7)


VISUALS[_k] = [
    {"title": "Центральности и сообщества", "svg": _sna_svg(),
     "caption": "Узел с наибольшей degree — центр своего кружка, но связь между сообществами держит мост B: "
                "степень у него меньше, чем у хаба, зато betweenness максимальна. Удалить B — и сеть распадётся на две части."},
    {"title": "Безмасштабная сеть против случайной", "chart": sna_chart,
     "caption": "В модели преференциального присоединения распределение степеней — степенной закон (прямая "
                "на log-log, есть хабы); в случайном графе степени сосредоточены около среднего."},
]
