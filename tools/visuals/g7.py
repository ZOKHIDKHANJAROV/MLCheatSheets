# -*- coding: utf-8 -*-
"""Рисунки группы 7: source_separation … zero_shot_learning (SVG-схемы и графики matplotlib)."""

import numpy as np
from xml.sax.saxutils import escape as _esc

P = ["#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"]


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


# =============================================================== 1. source_separation
SOURCE_SEP = svg(
    "srcsep", 340, 312, "Схема разделения источников по маскам спектрограммы",
    B(110, 8, 120, 30, "c6", "Микс x(t)"),
    A(170, 38, 170, 56),
    B(85, 58, 170, 30, "c1", "STFT → |X| и фаза ∠X"),
    A(170, 88, 170, 106),
    B(70, 108, 200, 34, "c5", "U-Net (энкодер–декодер)"),
    *[A(170, 142, cx, 166) for cx in (47, 129, 211, 293)],
    *[B(12 + 82 * i, 168, 70, 40, "c3", [f"M{s} ⊙ |X|", nm])
      for i, (s, nm) in enumerate(zip("₁₂₃₄", ["вокал", "барабаны", "бас", "другое"]))],
    *[A(cx, 208, 170, 238) for cx in (47, 129, 211, 293)],
    B(70, 240, 200, 30, "c2", "iSTFT с фазой микса ∠X"),
    A(170, 270, 170, 286),
    T(170, 302, "ŝ₁(t) … ŝ₄(t) — отдельные дорожки"),
)


def ica_chart(fig):
    from sklearn.decomposition import FastICA
    t = np.linspace(0, 8, 2000)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    S = np.c_[s1, s2]
    Amix = np.array([[1.0, 0.6], [0.5, 1.0]])
    X = S @ Amix.T
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, X[:, 0] + 3, color=P[5], lw=1.4, label="микс x₁")
    ax.plot(t, X[:, 1], color=P[0], lw=1.4, label="микс x₂")
    ax.set_title("Наблюдаем только смеси x = A·s", fontsize=10)
    ax.set_yticks([]), ax.legend(fontsize=8, loc="upper right")
    ica = FastICA(n_components=2, random_state=0, whiten="unit-variance")
    Sh = ica.fit_transform(X)
    # упорядочить по корреляции с истинными источниками и привести знак
    order = [int(np.argmax([abs(np.corrcoef(Sh[:, j], s)[0, 1]) for j in range(2)])) for s in (s1, s2)]
    ax2 = fig.add_subplot(2, 1, 2)
    for k, (j, s, c, nm) in enumerate(zip(order, (s1, s2), (P[1], P[2]), ("синус", "меандр"))):
        y = Sh[:, j] * np.sign(np.corrcoef(Sh[:, j], s)[0, 1])
        y = y / np.abs(y).max()
        ax2.plot(t, y + 3 * (1 - k), color=c, lw=1.4, label=f"ŝ: {nm}")
    ax2.set_title("FastICA восстановила источники", fontsize=10)
    ax2.set_xlabel("время t"), ax2.set_yticks([]), ax2.legend(fontsize=8, loc="upper right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# =============================================================== 2. spectral_embedding
_A_nodes = [(40, 42), (85, 30), (62, 82), (110, 70)]
_B_nodes = [(225, 72), (270, 34), (305, 78), (255, 98)]
SPECTRAL = svg(
    "spemb", 340, 290, "Шаги Spectral Embedding: граф, лапласиан, собственные векторы",
    T(10, 16, "1. Граф близости W (kNN)", a="start"),
    *[L(*_A_nodes[i], *_A_nodes[j], "ln") for i, j in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]],
    *[L(*_B_nodes[i], *_B_nodes[j], "ln") for i, j in [(0, 1), (0, 3), (1, 2), (3, 2), (1, 3)]],
    L(110, 70, 225, 72, "ln s6", dash=True),
    T(167, 62, "слабая связь", "mut", fs=11),
    *[C(x, y, 6, "c1") for x, y in _A_nodes],
    *[C(x, y, 6, "c3") for x, y in _B_nodes],
    A(170, 104, 170, 116),
    B(40, 118, 260, 30, "c5", "2. L = D − W,  D = diag(Σⱼ Wᵢⱼ)"),
    A(170, 148, 170, 164),
    B(40, 166, 260, 30, "c3", "3. Lv = λDv: малые λ₁, λ₂, …"),
    T(170, 214, "v₀ = const (λ₀ = 0) отбрасывают", "mut", fs=11),
    T(10, 238, "4. Координаты: yᵢ = (v₁(i), v₂(i), …)", a="start"),
    A(25, 260, 315, 260),
    T(322, 264, "v₁", a="middle"),
    *[C(x, 260, 5, "c1") for x in (60, 72, 84, 96)],
    *[C(x, 260, 5, "c3") for x in (236, 248, 260, 272)],
    T(78, 282, "кластер A", "t1", fs=11),
    T(254, 282, "кластер B", "t3", fs=11),
)


def spectral_chart(fig):
    from sklearn.datasets import make_moons
    from sklearn.manifold import SpectralEmbedding
    X, y = make_moons(n_samples=300, noise=0.06, random_state=0)
    ax = fig.add_subplot(2, 1, 1)
    for k in (0, 1):
        ax.scatter(*X[y == k].T, s=8, color=P[0] if k == 0 else P[2])
    ax.set_title("Исходные данные: две «луны»", fontsize=10)
    E = SpectralEmbedding(n_components=2, affinity="nearest_neighbors", n_neighbors=10,
                          random_state=0).fit_transform(X)
    ax2 = fig.add_subplot(2, 1, 2)
    for k in (0, 1):
        ax2.scatter(*E[y == k].T, s=8, color=P[0] if k == 0 else P[2])
    ax2.set_title("SpectralEmbedding: луны разведены", fontsize=10)
    ax2.set_xlabel("v₁"), ax2.set_ylabel("v₂")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# =============================================================== 3. state_space_models
SSM = svg(
    "ssm", 340, 214, "Граф линейной модели пространства состояний",
    T(10, 18, "xₜ = A·xₜ₋₁ + wₜ,  w ~ N(0, Q)", "t1", a="start"),
    A(8, 70, 38, 70), T(14, 62, "…", a="start"),
    C(60, 70, 20, "c1", "xₜ₋₁"), C(170, 70, 20, "c1", "xₜ"), C(280, 70, 20, "c1", "xₜ₊₁"),
    A(80, 70, 148, 70), A(190, 70, 258, 70),
    T(114, 62, "A"), T(224, 62, "A"),
    *[A(x, 90, x, 136) for x in (60, 170, 280)],
    *[T(x + 8, 118, "C", a="start") for x in (60, 170, 280)],
    *[B(x - 24, 138, 48, 32, "c2", s) for x, s in ((60, "yₜ₋₁"), (170, "yₜ"), (280, "yₜ₊₁"))],
    T(10, 200, "yₜ = C·xₜ + vₜ,  v ~ N(0, R)", "t2", a="start"),
    T(330, 200, "видим только y", "mut", a="end", fs=11),
)


def kalman_chart(fig):
    rng = np.random.default_rng(1)
    n = 60
    Am = np.array([[1.0, 1.0], [0.0, 1.0]])
    Cm = np.array([[1.0, 0.0]])
    Q = np.diag([0.01, 0.02])
    Rv = 9.0
    x = np.zeros(2)
    xs, ys = [], []
    for _ in range(n):
        x = Am @ x + rng.multivariate_normal([0, 0], Q)
        xs.append(x[0])
        ys.append(x[0] + rng.normal(0, 3.0))
    xh, Pm = np.zeros(2), np.eye(2) * 10
    est, sd = [], []
    for yv in ys:
        xh = Am @ xh
        Pm = Am @ Pm @ Am.T + Q
        S = Cm @ Pm @ Cm.T + Rv
        K = Pm @ Cm.T / S
        xh = xh + (K * (yv - Cm @ xh)).ravel()
        Pm = (np.eye(2) - K @ Cm) @ Pm
        est.append(xh[0]), sd.append(np.sqrt(Pm[0, 0]))
    t = np.arange(n)
    est, sd = np.array(est), np.array(sd)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(t, ys, s=10, color=P[5], label="измерения yₜ")
    ax.plot(t, xs, color="k", lw=1.2, ls="--", label="истинное xₜ")
    ax.plot(t, est, color=P[0], lw=2, label="Kalman x̂ₜ|ₜ")
    ax.fill_between(t, est - 2 * sd, est + 2 * sd, color=P[0], alpha=0.15, label="± 2σ")
    ax.set_xlabel("шаг t"), ax.set_ylabel("позиция")
    ax.legend(fontsize=8)


# =============================================================== 4. statistical_anomaly_detection
STAT_AD = svg(
    "statad", 340, 150, "Boxplot: квартили, IQR и границы выбросов",
    L(65, 42, 65, 110, "ln s4", dash=True), L(265, 42, 265, 110, "ln s4", dash=True),
    L(80, 80, 140, 80), L(190, 80, 250, 80), L(80, 70, 80, 90), L(250, 70, 250, 90),
    R(140, 60, 50, 40, "c1"), L(162, 60, 162, 100, "ln"),
    T(140, 55, "Q1"), T(190, 55, "Q3"),
    T(165, 116, "IQR = Q3 − Q1", fs=11),
    T(65, 134, "Q1 − 1.5·IQR", "t4"), T(265, 134, "Q3 + 1.5·IQR", "t4"),
    C(30, 80, 4, "t4"), C(296, 80, 4, "t4"), C(314, 80, 4, "t4"),
    T(305, 64, "выбросы", "t4", fs=11),
)


def stat_ad_chart(fig):
    rng = np.random.default_rng(0)
    data = np.r_[rng.normal(100, 15, 1000), rng.normal(172, 4, 60)]
    mu, sd = data.mean(), data.std()
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    hi_z, hi_m = mu + 3 * sd, med + 3.5 * mad / 0.6745
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=50, color=P[5], alpha=0.7)
    ax.axvline(hi_z, color=P[1], lw=2, ls="--", label=f"μ + 3σ = {hi_z:.0f}")
    ax.axvline(hi_m, color=P[3], lw=2, label=f"MAD-порог = {hi_m:.0f}")
    ax.annotate("6% выбросов\n≈ 172", xy=(172, 20), xytext=(185, 55), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.set_xlabel("значение"), ax.set_ylabel("частота")
    ax.legend(fontsize=8, loc="upper left")


# =============================================================== 5. statistical_methods
_bars = [6, 14, 26, 40, 52, 50, 38, 24, 12, 5]
STAT_METHODS = svg(
    "statm", 340, 262, "Схема bootstrap-оценки доверительного интервала",
    B(110, 8, 120, 30, "c1", "Выборка X (n)"),
    T(238, 28, "с возвращением", "mut", a="start", fs=11),
    A(150, 38, 62, 72), A(170, 38, 170, 72), A(190, 38, 278, 72),
    B(20, 74, 80, 28, "c6", "X*₁"), B(130, 74, 80, 28, "c6", "X*₂"), B(240, 74, 80, 28, "c6", "X*_B"),
    T(225, 92, "…"),
    *[A(x, 102, x, 116) for x in (60, 170, 280)],
    T(60, 130, "θ*₁"), T(170, 130, "θ*₂"), T(280, 130, "θ*_B"),
    A(60, 136, 110, 160), A(170, 136, 170, 156), A(280, 136, 230, 160),
    *[R(80 + 18 * i, 215 - b, 17, b, "c1") for i, b in enumerate(_bars)],
    L(75, 215, 265, 215),
    L(98, 150, 98, 222, "ln s4", dash=True), L(242, 150, 242, 222, "ln s4", dash=True),
    T(98, 234, "2.5%", "t4", fs=11), T(242, 234, "97.5%", "t4", fs=11),
    A(102, 248, 238, 248, "ln s4", both=True),
    T(170, 244, "95% ДИ", "t4"),
)


def bootstrap_chart(fig):
    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, 40)
    boots = np.array([rng.choice(x, len(x)).mean() for _ in range(3000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(boots, bins=40, color=P[0], alpha=0.6)
    ax.axvline(lo, color=P[3], ls="--", lw=2, label=f"95% ДИ [{lo:.2f}; {hi:.2f}]")
    ax.axvline(hi, color=P[3], ls="--", lw=2)
    ax.axvline(x.mean(), color="k", lw=1.5, label=f"среднее выборки {x.mean():.2f}")
    ax.set_xlabel("bootstrap-среднее θ*"), ax.set_ylabel("частота")
    ax.set_title("B = 3000 ресэмплов, n = 40 (экспоненц.)", fontsize=10)
    ax.legend(fontsize=8)


# =============================================================== 6. statistical_tests
STAT_TESTS = svg(
    "stt", 340, 272, "Дерево выбора статистического теста",
    B(100, 8, 140, 28, "c6", "Что сравниваем?"),
    A(140, 36, 105, 58), A(200, 36, 272, 58),
    B(10, 60, 190, 28, "c1", "Числовые выборки"),
    B(210, 60, 125, 28, "c3", "Категории"),
    A(105, 88, 105, 106),
    B(10, 108, 190, 28, "c5", "Нормальны? (Shapiro)"),
    A(80, 136, 56, 166), A(130, 136, 154, 166),
    T(54, 154, "да", "mut", a="end", fs=11), T(148, 154, "нет", "mut", a="start", fs=11),
    B(10, 168, 92, 40, "c2", ["t-test", "ANOVA (≥3)"]),
    B(108, 168, 92, 40, "c2", ["Mann-Whitney", "Wilcoxon"]),
    PA("M216,88 V186"),
    A(216, 126, 223, 126), A(216, 186, 223, 186),
    B(225, 106, 110, 40, "c3", ["χ²-тест", "(таблица)"]),
    B(225, 166, 110, 40, "c3", ["McNemar", "(пары моделей)"]),
    B(10, 232, 325, 28, "c6", "Корреляция: Pearson / Spearman"),
)


# =============================================================== 7. statistical_tests_t_test_mcnemar
MCNEMAR = svg(
    "mcn", 340, 212, "Таблица сопряжённости для теста McNemar",
    T(217, 18, "Модель 2"),
    T(165, 40, "верно", fs=11), T(270, 40, "ошибка", fs=11),
    T(60, 77, "М1 верно"), T(60, 127, "М1 ошибка"),
    B(115, 50, 100, 45, "c6", ["a", "оба верно"]),
    B(220, 50, 100, 45, "c3", ["b", "прав только М1"], fs=None),
    B(115, 100, 100, 45, "c3", ["c", "прав только М2"]),
    B(220, 100, 100, 45, "c6", ["d", "оба ошиблись"]),
    T(170, 176, "χ² = (|b − c| − 1)² / (b + c)"),
    T(170, 198, "a и d не влияют — важны только расхождения", "mut", fs=11),
)


def ttest_chart(fig):
    from scipy import stats
    rng = np.random.default_rng(3)
    s1 = 0.85 + rng.normal(0, 0.012, 10)
    s2 = s1 + 0.005 + rng.normal(0, 0.008, 10)
    tval, p = stats.ttest_rel(s2, s1)
    df = 9
    tc = stats.t.ppf(0.975, df)
    x = np.linspace(-5, 5, 400)
    y = stats.t.pdf(x, df)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, color=P[0], lw=2, label="t-распределение, df = 9")
    m = np.abs(x) >= abs(tval)
    ax.fill_between(x, 0, y, where=m & (x > 0), color=P[2], alpha=0.5, label=f"p-value = {p:.3f}")
    ax.fill_between(x, 0, y, where=m & (x < 0), color=P[2], alpha=0.5)
    for s in (-tc, tc):
        ax.axvline(s, color=P[3], ls="--", lw=1.5)
    ax.axvline(tval, color="k", lw=1.5)
    ax.text(tval + 0.1, 0.25, f"t = {tval:.2f}", fontsize=9)
    ax.text(-tc - 0.1, 0.25, "±t крит.\nα = 0.05", fontsize=8, color=P[3], ha="right")
    ax.set_xlabel("t"), ax.set_ylabel("плотность")
    ax.set_title("Paired t-test: 10 фолдов CV", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


# =============================================================== 8. stemming_lemmatization
_rows = [("running", "run", "run", ""), ("runs", "run", "run", ""), ("ran", "ran", "run", "v"),
         ("easily", "easili", "easily", ""), ("better", "better", "good", "a")]
STEM = svg(
    "stem", 340, 236, "Сравнение стемминга и лемматизации на примерах",
    B(10, 8, 90, 26, "c6", "Слово"), B(115, 8, 110, 26, "c3", "Porter stem"), B(240, 8, 90, 26, "c2", "Лемма"),
    *[L(10, 46 + 30 * i, 330, 46 + 30 * i, "ln s6") for i in range(1, 5)],
    *[T(55, 66 + 30 * i, w) + T(170, 66 + 30 * i, s, "t4" if s == "easili" else "")
      + T(285, 66 + 30 * i, lm + (f"  (pos={pos})" if pos else ""), "t2" if pos else "")
      for i, (w, s, lm, pos) in enumerate(_rows)],
    T(170, 218, "красным — не слово; зелёным — помог словарь + POS", "mut", fs=11),
)


# =============================================================== 9. stylegan_architecture
STYLEGAN = svg(
    "stg", 350, 300, "Архитектура StyleGAN: mapping network и synthesis network",
    B(15, 8, 100, 28, "c6", "z ∈ ℝ⁵¹²"),
    A(65, 36, 65, 56),
    B(15, 58, 100, 110, "c5", ["Mapping", "network", "8 × FC", "LeakyReLU"]),
    A(65, 168, 65, 196),
    B(15, 198, 100, 30, "c5", "w ∈ W"),
    T(65, 248, "A: w → (y_s, y_b)", "mut", fs=11),
    T(65, 264, "стиль для AdaIN", "mut", fs=11),
    B(170, 8, 160, 28, "c6", "Const 4×4×512"),
    A(250, 36, 250, 56),
    B(170, 58, 160, 50, "c1", ["4×4: Conv 3×3", "+ шум B, AdaIN"]),
    A(250, 108, 250, 136),
    B(170, 138, 160, 50, "c1", ["8×8: ↑2, Conv 3×3", "+ шум B, AdaIN"]),
    A(250, 188, 250, 246),
    T(262, 222, "… 16 → 1024", "mut", a="start", fs=11),
    B(170, 248, 160, 30, "c2", "toRGB → 1024×1024"),
    PA("M115,213 H140", arrow=False), L(140, 83, 140, 263, "ln s5"),
    A(140, 83, 168, 83, "ln s5"), A(140, 163, 168, 163, "ln s5"), A(140, 263, 168, 263, "ln s5"),
    T(146, 125, "A", "t5", a="start"),
)


# =============================================================== 10. subspace_clustering
def _clique_svg():
    rng = np.random.default_rng(4)
    x0, y0, cw, ch, nx, ny = 70, 20, 38, 38, 6, 5
    # кластер A: плотен только по x₁ (интервал 2), B: плотный блок в ячейке (4, 1), шум
    pa = np.c_[rng.uniform(2.15, 2.85, 30), rng.uniform(0.05, 4.95, 30)]
    pb = np.c_[rng.uniform(4.15, 4.85, 20), rng.uniform(1.15, 1.85, 20)]
    pn = np.c_[rng.uniform(0, 6, 25), rng.uniform(0, 5, 25)]
    pts = np.r_[pa, pb, pn]
    cx = np.floor(pts[:, 0]).astype(int)
    cy = np.floor(pts[:, 1]).astype(int)
    hx = np.bincount(cx, minlength=nx)
    hy = np.bincount(cy, minlength=ny)
    cell = np.zeros((nx, ny), int)
    np.add.at(cell, (cx, cy), 1)
    tau1, tau2 = 17, 10
    parts = []
    for i in range(nx):
        for j in range(ny):
            if cell[i, j] >= tau2:
                parts.append(R(x0 + i * cw, y0 + (ny - 1 - j) * ch, cw, ch, "c3"))
    for i in range(nx + 1):
        parts.append(L(x0 + i * cw, y0, x0 + i * cw, y0 + ny * ch, "ln s6"))
    for j in range(ny + 1):
        parts.append(L(x0, y0 + j * ch, x0 + nx * cw, y0 + j * ch, "ln s6"))
    for (u, v), lab in zip(pts, ["a"] * 30 + ["b"] * 20 + ["n"] * 25):
        cls = {"a": "t1", "b": "t5", "n": "t6"}[lab]
        parts.append(C(round(x0 + u * cw, 1), round(y0 + (ny - v) * ch, 1), 2.6, cls))
    base = y0 + ny * ch + 4
    k = 1.3
    for i in range(nx):
        parts.append(R(x0 + i * cw + 6, base, cw - 12, round(hx[i] * k, 1), "c3" if hx[i] >= tau1 else "c6"))
    parts.append(L(x0 - 4, base + tau1 * k, x0 + nx * cw + 4, base + tau1 * k, "ln s4", dash=True))
    parts.append(T(x0 + nx * cw + 8, base + tau1 * k + 4, "τ", "t4", a="start"))
    for j in range(ny):
        yy = y0 + (ny - 1 - j) * ch + 6
        parts.append(R(round(x0 - 4 - hy[j] * k, 1), yy, round(hy[j] * k, 1), ch - 12,
                       "c3" if hy[j] >= tau1 else "c6"))
    parts.append(L(x0 - 4 - tau1 * k, y0 - 4, x0 - 4 - tau1 * k, y0 + ny * ch + 2, "ln s4", dash=True))
    parts += [T(x0 + nx * cw + 10, base - 6, "x₁", a="start"), T(x0 - 2, 14, "x₂", a="end"),
              T(160, 284, "A — кластер в {x₁}", "t1", fs=11), T(160, 300, "B — кластер в {x₁, x₂}", "t5", fs=11)]
    return svg("subcl", 340, 308, "CLIQUE: сетка, плотные интервалы и ячейки", *parts)


SUBSPACE = _clique_svg()


# =============================================================== 11. surrogate_models
SURROGATE = svg(
    "surr", 340, 250, "Схема обучения глобальной суррогатной модели",
    B(10, 15, 70, 30, "c6", "X"), A(80, 30, 108, 30),
    B(110, 15, 120, 30, "c4", "Black-box f"), A(230, 30, 258, 30),
    B(260, 15, 70, 30, "c6", "ŷ = f(X)"),
    A(45, 45, 118, 98), A(295, 45, 222, 98),
    B(70, 100, 200, 34, "c2", "Surrogate g (дерево d ≤ 4)"),
    A(130, 134, 85, 173), A(210, 134, 255, 173),
    B(10, 175, 150, 30, "c5", "Правила if-then"),
    B(180, 175, 150, 30, "c1", "Fidelity: g(X) = f(X)?"),
    T(170, 228, "g учат на (X, ŷ), а не на истинных y", "mut", fs=11),
    T(170, 244, "fidelity > 0.9 — объяснению можно верить", "mut", fs=11),
)


def surrogate_chart(fig):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6, random_state=0)
    Xtr, Xte = X[:1400], X[1400:]
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1).fit(Xtr, y[:1400])
    ftr, fte = rf.predict(Xtr), rf.predict(Xte)
    depths = range(1, 11)
    fid = [(DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ftr).predict(Xte) == fte).mean()
           for d in depths]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(depths), fid, "o-", color=P[1], lw=2, label="fidelity (test)")
    ax.axhline(0.9, color=P[3], ls="--", lw=1.5, label="порог 0.9")
    ax.set_xlabel("max_depth суррогатного дерева"), ax.set_ylabel("доля совпадений с RF")
    ax.set_title("Суррогат-дерево для Random Forest", fontsize=10)
    ax.legend(fontsize=8)


# =============================================================== 12. swarm_intelligence
SWARM = svg(
    "pso", 340, 262, "Геометрия шага PSO: сложение трёх составляющих скорости",
    L(60, 200, 110, 70, "ln s6", dash=True), L(60, 200, 310, 170, "ln s6", dash=True),
    A(60, 200, 128, 200, "ln s6"),
    A(130, 200, 159, 124, "ln s2"),
    A(160, 122, 258, 110, "ln s3"),
    A(60, 200, 256, 112, "ln s1"),
    C(60, 200, 5, "c6"), C(110, 70, 6, "c2"), C(310, 170, 6, "c4"), C(260, 110, 5, "c1"),
    T(52, 222, "x"), T(110, 58, "pbest", "t2"), T(310, 190, "gbest", "t4"),
    T(268, 104, "x′ = x + v′", "t1", a="start"),
    T(95, 216, "w·v", fs=11), T(140, 152, "когн.", "t2", a="end", fs=11),
    T(205, 104, "соц.", "t3", fs=11), T(178, 176, "v′", "t1"),
    T(170, 246, "v′ = w·v + c₁r₁(pbest − x) + c₂r₂(gbest − x)"),
)


def pso_chart(fig):
    def rastrigin(X):
        return 10 * X.shape[1] + (X ** 2 - 10 * np.cos(2 * np.pi * X)).sum(1)

    def run(dim, iters, wsched, seed, n=30, keep=None):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-5.12, 5.12, (n, dim))
        v = np.zeros_like(x)
        pb, pf = x.copy(), rastrigin(x)
        g = pb[pf.argmin()].copy()
        hist, snaps = [], {}
        for it in range(iters):
            if keep and it in keep:
                snaps[it] = x.copy()
            w = wsched(it, iters)
            r1, r2 = rng.random(x.shape), rng.random(x.shape)
            v = w * v + 1.49618 * r1 * (pb - x) + 1.49618 * r2 * (g - x)
            x = np.clip(x + v, -5.12, 5.12)
            f = rastrigin(x)
            m = f < pf
            pb[m], pf[m] = x[m], f[m]
            g = pb[pf.argmin()].copy()
            hist.append(pf.min())
        return np.array(hist), snaps, g

    def const(it, n):
        return 0.7298

    def lin(it, n):
        return 0.9 - 0.5 * it / n

    _, snaps, g = run(2, 40, const, 0, keep={0, 39})
    ax = fig.add_subplot(2, 1, 1)
    gx = np.linspace(-5.12, 5.12, 200)
    GX, GY = np.meshgrid(gx, gx)
    Z = rastrigin(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    ax.contourf(GX, GY, Z, levels=20, cmap="Greys", alpha=0.35)
    ax.scatter(*snaps[0].T, s=14, color=P[2], label="итерация 0")
    ax.scatter(*snaps[39].T, s=14, color=P[0], label="итерация 39")
    ax.scatter([0], [0], marker="*", s=120, color=P[3], label="минимум (0, 0)")
    ax.set_title("Rastrigin 2D: рой стягивается к минимуму", fontsize=10)
    ax.legend(fontsize=7, loc="upper right"), ax.set_aspect("equal")
    ax2 = fig.add_subplot(2, 1, 2)
    for wsch, lab, c in ((const, "w = 0.7298", P[0]), (lin, "w: 0.9 → 0.4", P[1])):
        h = np.mean([run(10, 300, wsch, s)[0] for s in range(3)], axis=0)
        ax2.plot(h, color=c, lw=2, label=lab)
    ax2.set_yscale("log"), ax2.set_xlabel("итерация"), ax2.set_ylabel("f(gbest)")
    ax2.set_title("Rastrigin 10D: сходимость (3 запуска)", fontsize=10)
    ax2.legend(fontsize=8)
    fig.set_size_inches(4.2, 5.6)
    fig.tight_layout()


# =============================================================== 13. syntactic_parsing
def _arc(x1, x2, h, lab, base=125):
    return (PA(f"M{x1},{base} C{x1},{base - h} {x2},{base - h} {x2},{base}", "ln s1")
            + T((x1 + x2) / 2, round(base - 0.75 * h - 4, 1), lab, "t1", fs=11))


_words = [("The", 35, "DT"), ("cat", 90, "NN"), ("sat", 150, "VBD"), ("on", 205, "IN"),
          ("the", 255, "DT"), ("mat", 310, "NN")]
PARSE = svg(
    "dep", 345, 172, "Дерево зависимостей предложения The cat sat on the mat",
    T(150, 14, "ROOT", "t5"), A(150, 20, 150, 122, "ln s5"),
    _arc(150, 90, 35, "nsubj"), _arc(90, 35, 25, "det"), _arc(150, 205, 35, "prep"),
    _arc(205, 310, 70, "pobj"), _arc(310, 255, 25, "det"),
    *[T(x, 142, w) + T(x, 160, pos, "mut", fs=11) for w, x, pos in _words],
)


# =============================================================== 14. tabnet
TABNET = svg(
    "tabnet", 340, 318, "Decision step TabNet: маска, Feature Transformer, split",
    B(110, 8, 120, 28, "c6", "Признаки f (BN)"),
    T(90, 52, "P[d−1]: прошлые маски", "mut", fs=11),
    B(20, 60, 140, 40, "c5", ["Attentive Tr.", "M[d] = sparsemax(…)"]),
    T(108, 116, "P[d] = ∏ (γ − M[i])", "mut", fs=11),
    PA("M230,36 V48 H240 V66"),
    A(160, 80, 226, 80),
    C(240, 80, 12, "c3", "⊗"),
    A(240, 92, 240, 118),
    B(180, 120, 120, 40, "c1", ["Feature Tr.", "FC → BN → GLU"]),
    PA("M180,140 H40 V102", "ln s5"),
    T(112, 156, "d[d] → шаг d+1", "t5", fs=11),
    A(240, 160, 240, 188),
    B(190, 190, 100, 28, "c2", "ReLU(a[d])"),
    A(240, 218, 200, 248),
    T(55, 254, "от шагов 1…N", "mut", fs=11), A(60, 265, 108, 265),
    B(110, 250, 120, 30, "c2", "Σ_d → FC → ŷ"),
    T(170, 304, "M[d] разрежена → важность признаков", "mut", fs=11),
)


def tabnet_chart(fig):
    z = np.array([2.0, 1.6, 0.9, 0.3, -0.2, -1.0])
    sm = np.exp(z) / np.exp(z).sum()
    zs = np.sort(z)[::-1]
    cs = np.cumsum(zs)
    k = np.arange(1, len(z) + 1)
    kmax = k[1 + k * zs > cs].max()
    tau = (cs[kmax - 1] - 1) / kmax
    sp = np.maximum(z - tau, 0)
    ax = fig.add_subplot(1, 1, 1)
    i = np.arange(len(z))
    ax.bar(i - 0.2, sm, 0.4, color=P[5], label="softmax")
    ax.bar(i + 0.2, sp, 0.4, color=P[0], label="sparsemax")
    ax.set_xticks(i, [f"f{j + 1}\nz={v:g}" for j, v in enumerate(z)], fontsize=8)
    ax.set_ylabel("вес признака в маске")
    nz = int((sp == 0).sum())
    ax.set_title(f"sparsemax: τ = {tau:.2f}, {nz} нуля из 6", fontsize=10)
    ax.legend(fontsize=8)


# =============================================================== 15. tabu_search
TABU = svg(
    "tabu", 340, 318, "Блок-схема Tabu Search",
    B(25, 8, 210, 30, "c6", "Начальное s; s_best = s"),
    A(130, 38, 130, 56),
    B(25, 58, 210, 28, "c1", "Окрестность N(s)"),
    A(130, 86, 130, 104),
    B(25, 106, 210, 42, "c5", ["Лучший s′ ∈ N(s),", "ход не в tabu"]),
    B(245, 106, 88, 42, "c3", ["аспирация:", "f(s′) < f*"]),
    A(245, 127, 237, 127),
    A(130, 148, 130, 166),
    B(25, 168, 210, 30, "c4", "s = s′; ход → tabu-список"),
    *[R(247 + 22 * i, 173, 20, 20, "c4", 3) for i in range(4)],
    T(290, 212, "FIFO длины L", "mut", fs=11),
    A(130, 198, 130, 216),
    B(25, 218, 210, 42, "c2", ["если f(s) < f(s_best):", "s_best = s"]),
    A(130, 260, 130, 278),
    B(25, 280, 210, 30, "c6", "Стоп? да → вернуть s_best"),
    PA("M25,295 H12 V72 H23"),
    T(16, 162, "нет", "mut", a="start", fs=11),
)


def tabu_chart(fig):
    xs = np.arange(0, 120)
    f = 0.004 * (xs - 85.0) ** 2 + 1.6 * np.sin(xs / 2.2) + 0.8 * np.sin(xs / 5.1)
    nb = [-1, 1]
    g = 10
    while True:
        b = min([g + d for d in nb if 0 <= g + d < 120], key=lambda c: f[c])
        if f[b] >= f[g]:
            break
        g = b
    s, best, tabu, path = 10, 10, [], [10]
    for _ in range(120):
        cand = [s + d for d in nb if 0 <= s + d < 120]
        ok = [c for c in cand if c not in tabu or f[c] < f[best]]
        if not ok:
            ok = [min(cand, key=lambda c: tabu.index(c))]
        s = min(ok, key=lambda c: f[c])
        tabu.append(s)
        tabu = tabu[-10:]
        path.append(s)
        if f[s] < f[best]:
            best = s
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, f, color=P[5], lw=1.5, label="f(x)")
    path = np.array(path)
    ax.scatter(path, f[path], s=12, color=P[0], alpha=0.5, label="путь tabu search")
    ax.scatter([10], [f[10]], s=40, color="k", zorder=5, label="старт")
    ax.scatter([g], [f[g]], s=80, marker="X", color=P[3], zorder=5, label="жадный спуск застрял")
    ax.scatter([best], [f[best]], s=160, marker="*", color=P[1], zorder=5, label="лучшее tabu")
    ax.set_xlabel("решение x (соседи x ± 1)"), ax.set_ylabel("f(x) → min")
    ax.legend(fontsize=7, loc="upper right")


# =============================================================== 16. target_encoding
TARGET = svg(
    "tenc", 340, 256, "Сглаженный target encoding и out-of-fold схема",
    B(10, 8, 320, 46, "c1", ["enc(c) = (n_c·ȳ_c + α·ȳ) / (n_c + α)", "мало данных (n_c ≪ α) → глобальное ȳ"]),
    T(10, 80, "Out-of-fold (K = 5): без утечки таргета", a="start"),
    *[T(30, 108 + 30 * r, f"k={r + 1}", fs=11) for r in range(3)],
    *[R(55 + 55 * c, 92 + 30 * r, 52, 24, "c3" if c == r else "c2", 3) for r in range(3) for c in range(5)],
    T(190, 192, "…"),
    R(55, 206, 14, 14, "c3", 2), T(76, 217, "фолд, который кодируем", a="start", fs=11),
    R(55, 230, 14, 14, "c2", 2), T(76, 241, "по ним считаем ȳ_c", a="start", fs=11),
)


def target_chart(fig):
    n = np.logspace(0, 3, 200)
    yc, yg = 0.9, 0.3
    ax = fig.add_subplot(1, 1, 1)
    for a, c in ((1, P[0]), (10, P[1]), (50, P[2])):
        ax.plot(n, (n * yc + a * yg) / (n + a), color=c, lw=2, label=f"α = {a}")
    ax.axhline(yc, color="k", ls="--", lw=1)
    ax.axhline(yg, color=P[5], ls="--", lw=1)
    ax.text(1.1, yc + 0.02, "ȳ_c = 0.9 (среднее категории)", fontsize=8)
    ax.text(60, yg - 0.06, "ȳ = 0.3 (глобальное)", fontsize=8, color="#555")
    ax.set_xscale("log"), ax.set_ylim(0.15, 1.0)
    ax.set_xlabel("n_c — объектов в категории"), ax.set_ylabel("enc(c)")
    ax.legend(fontsize=8, loc="center right")


# =============================================================== 17. tcn
def _tcn_svg():
    xs = [60 + 17 * i for i in range(16)]
    rows = [205, 155, 105, 55]
    need = [set(range(8, 16)), {9, 11, 13, 15}, {11, 15}, {15}]
    dil = [1, 2, 4]
    parts = [T(170, 22, "RF = 1 + (k−1)·(1+2+4) = 8")]
    for lvl in range(3):
        for t in need[lvl + 1]:
            for src in (t, t - dil[lvl]):
                parts.append(L(xs[src], rows[lvl], xs[t], rows[lvl + 1], "ln s1"))
    for r, ys in enumerate(rows):
        for i, x in enumerate(xs):
            hi = i in need[r]
            parts.append(C(x, ys, 4.5 if hi else 3.5, "c1" if hi else "c6"))
    parts += [T(8, 209, "вход", a="start"), T(8, 159, "d=1", a="start"),
              T(8, 109, "d=2", a="start"), T(8, 59, "d=4", a="start"),
              T(xs[15], 42, "ŷₜ"), T(xs[15], 224, "t", fs=11), T(xs[8], 224, "t−7", fs=11),
              T(170, 242, "k = 2; будущее не используется (causal)", "mut", fs=11)]
    return svg("tcn", 340, 250, "Стек dilated causal свёрток TCN", *parts)


TCN = _tcn_svg()


def tcn_chart(fig):
    Ls = np.arange(1, 9)
    k = 3
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Ls, 1 + 2 * (k - 1) * (2 ** Ls - 1), "o-", color=P[0], lw=2, label="dilation 1, 2, 4, …")
    ax.plot(Ls, 1 + 2 * (k - 1) * Ls, "s-", color=P[5], lw=2, label="без dilation")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("число residual-блоков L"), ax.set_ylabel("рецептивное поле, шагов")
    ax.set_title("k = 3, по 2 conv в блоке: RF = 1 + 2(k−1)Σ2ⁱ", fontsize=10)
    ax.legend(fontsize=8)


# =============================================================== 18. td_learning
TD = svg(
    "td", 340, 244, "Backup-диаграммы Monte Carlo и TD(0)",
    T(85, 16, "Monte Carlo"), T(250, 16, "TD(0)"),
    C(85, 42, 10, "c6"), T(68, 46, "sₜ", a="end"),
    A(85, 52, 85, 80), T(95, 70, "Rₜ₊₁", "mut", a="start", fs=11),
    C(85, 92, 10, "c6"),
    A(85, 102, 85, 130), T(95, 120, "Rₜ₊₂", "mut", a="start", fs=11),
    C(85, 142, 10, "c6"), T(85, 172, "⋮"),
    R(75, 186, 20, 20, "c4", 2), T(105, 200, "конец", "mut", a="start", fs=11),
    T(85, 232, "цель: Gₜ = Σ γᵏRₜ₊ₖ₊₁", fs=11),
    C(250, 42, 10, "c6"), T(233, 46, "sₜ", a="end"),
    A(250, 52, 250, 98), T(260, 78, "Rₜ₊₁", "mut", a="start", fs=11),
    C(250, 110, 10, "c1"), T(265, 114, "V(sₜ₊₁)", "t1", a="start"),
    T(250, 146, "цель: Rₜ₊₁ + γV(sₜ₊₁)", fs=11),
    T(250, 178, "δ = R + γV(s′) − V(s)", "t3", fs=11),
    T(250, 200, "V(s) ← V(s) + α·δ", fs=11),
    T(250, 232, "сразу после шага", "mut", fs=11),
)


def td_chart(fig):
    true = np.arange(1, 6) / 6.0

    def run(method, alpha, rng, episodes=100):
        V = np.full(7, 0.5)
        V[0] = V[6] = 0.0
        err = []
        for _ in range(episodes):
            s, traj = 3, [3]
            while 0 < s < 6:
                s += 1 if rng.random() < 0.5 else -1
                traj.append(s)
            if method == "td":
                for a, b in zip(traj[:-1], traj[1:]):
                    r = 1.0 if b == 6 else 0.0
                    V[a] += alpha * (r + V[b] - V[a])
            else:
                G = 1.0 if traj[-1] == 6 else 0.0
                for a in traj[:-1]:
                    V[a] += alpha * (G - V[a])
            err.append(np.sqrt(np.mean((V[1:6] - true) ** 2)))
        return err

    rng = np.random.default_rng(0)
    ax = fig.add_subplot(1, 1, 1)
    for m, a, c, ls in (("td", 0.1, P[0], "-"), ("td", 0.05, P[0], "--"),
                        ("mc", 0.02, P[2], "-"), ("mc", 0.04, P[2], "--")):
        e = np.mean([run(m, a, rng) for _ in range(100)], axis=0)
        ax.plot(e, color=c, ls=ls, lw=2, label=f"{'TD(0)' if m == 'td' else 'MC'}, α = {a}")
    ax.set_xlabel("эпизод"), ax.set_ylabel("RMS ошибка V")
    ax.set_title("Random walk (5 состояний), 100 запусков", fontsize=10)
    ax.legend(fontsize=8)


# =============================================================== 19. tensorflow_keras_full_guide
_kx = [("c6", "tf.data.Dataset"), ("c1", ["Модель: Sequential /", "Functional / Subclassing"]),
       ("c5", "compile(optimizer, loss, metrics)"), ("c3", "fit(data, epochs, callbacks)"),
       ("c2", "evaluate() / predict()"), ("c6", "model.save() → SavedModel")]


def _keras_svg():
    parts, y = [], 8
    for i, (c, lab) in enumerate(_kx):
        h = 40 if isinstance(lab, list) else 28
        parts.append(B(40, y, 260, h, c, lab))
        if i < len(_kx) - 1:
            parts.append(A(170, y + h, 170, y + h + 14))
        y += h + 16
    yb = y + 14
    parts.append(A(110, y - 16, 60, yb - 2))
    parts.append(A(170, y - 16, 170, yb - 2))
    parts.append(A(230, y - 16, 280, yb - 2))
    for x, lab in ((12, "TF Serving"), (122, "TF Lite"), (232, "TF.js")):
        parts.append(B(x, yb, 96, 28, "c1", lab))
    return svg("keras", 340, yb + 36, "Рабочий цикл TensorFlow/Keras", *parts)


KERAS = _keras_svg()


# =============================================================== 20. tensorflow_serving
TFSERV = svg(
    "tfs", 340, 304, "Архитектура TensorFlow Serving",
    B(110, 8, 120, 28, "c6", "Клиент"),
    A(140, 36, 110, 84), A(200, 36, 230, 84),
    T(118, 62, "REST :8501", "t1", a="end", fs=11), T(222, 62, "gRPC :8500", "t1", a="start", fs=11),
    R(15, 86, 310, 150, "c1", 8),
    T(170, 106, "TensorFlow Serving"),
    B(30, 116, 280, 28, "c2", "Servable: модель my_model v2"),
    B(25, 180, 85, 40, "c6", ["Source", "следит"]),
    B(128, 180, 85, 40, "c6", ["Loader", "загрузка"]),
    B(231, 180, 85, 40, "c6", ["Manager", "версии"]),
    A(110, 200, 126, 200), A(213, 200, 229, 200), A(273, 180, 273, 146),
    B(40, 262, 260, 32, "c3", "/models/my_model/1/, 2/ (SavedModel)"),
    A(67, 262, 67, 222),
)


# =============================================================== 21. text_to_image_generation
T2I = svg(
    "t2i", 340, 288, "Пайплайн латентной диффузии text-to-image",
    B(10, 10, 130, 28, "c6", "Промпт c"), A(75, 38, 75, 58),
    B(10, 60, 130, 40, "c5", ["Text encoder", "CLIP / T5"]),
    B(200, 10, 130, 28, "c6", "шум z_T 4×64×64"), A(265, 38, 265, 58),
    B(200, 60, 130, 60, "c1", ["U-Net / DiT", "ε_θ(z_t, t, c)", "× T шагов"]),
    A(140, 80, 198, 80), T(170, 72, "cross-attn", "mut", fs=11), T(170, 96, "77×768", "mut", fs=11),
    A(265, 120, 265, 148),
    B(10, 150, 320, 30, "c3", "CFG: ε̂ = ε_θ(z_t,∅) + w·(ε_θ(z_t,c) − ε_θ(z_t,∅))", fs=11),
    A(265, 180, 265, 198),
    B(200, 200, 130, 28, "c6", "z₀ (латент)"),
    A(265, 228, 265, 248),
    B(200, 250, 130, 28, "c2", "VAE decoder"),
    A(200, 264, 162, 264),
    B(10, 250, 150, 28, "c2", "Изображение 512×512"),
)


# =============================================================== 22. tfidf
_tf_rows = [("document", "2/6 = 0.33", "3", "0.29", "0.10", ""), ("second", "1/6 = 0.17", "1", "1.39", "0.23", "t2"),
            ("this", "1/6 = 0.17", "4", "0", "0", "t4")]
TFIDF = svg(
    "tfidf", 340, 168, "Пример расчёта TF-IDF для трёх слов",
    T(170, 18, "Док. 2: «This document is the second document»", fs=11),
    T(170, 34, "|d| = 6 слов, N = 4 документа", "mut", fs=11),
    R(10, 42, 320, 24, "c6", 4),
    *[T(x, 58, h, fs=11) for x, h in ((50, "слово"), (130, "TF"), (190, "df"), (240, "ln(N/df)"), (300, "TF·IDF"))],
    *[T(50, 88 + 26 * i, w) + T(130, 88 + 26 * i, tf, fs=11) + T(190, 88 + 26 * i, df)
      + T(240, 88 + 26 * i, idf) + T(300, 88 + 26 * i, v, cls)
      for i, (w, tf, df, idf, v, cls) in enumerate(_tf_rows)],
    T(170, 160, "без smooth_idf и L2-нормы (как в формуле выше)", "mut", fs=11),
)


def tfidf_chart(fig):
    N = 100
    df = np.arange(1, N + 1)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df, np.log(N / df), color=P[0], lw=2, label="ln(N / df)")
    ax.plot(df, np.log((1 + N) / (1 + df)) + 1, color=P[2], lw=2, ls="--",
            label="sklearn smooth: ln((1+N)/(1+df)) + 1")
    ax.set_xlabel("df(t) — документов со словом (N = 100)"), ax.set_ylabel("IDF")
    ax.set_title("Редкие слова получают большой вес", fontsize=10)
    ax.legend(fontsize=8)


# =============================================================== 23. time_series_anomalies
def _ts_svg():
    rng = np.random.default_rng(2)
    t = np.arange(80)
    parts = []
    x0, sx = 15, 310 / 79

    def pts(y, y0, lo=-1.3, hi=1.3, h=50):
        return [(x0 + i * sx, y0 + h - (v - lo) / (hi - lo) * h) for i, v in enumerate(y)]

    base = np.sin(2 * np.pi * t / 20)
    y1 = base + rng.normal(0, 0.05, 80)
    y1[47] = 2.6
    p1 = pts(y1, 26, -1.3, 2.8)
    parts += [T(10, 16, "Point: одна точка выбивается", a="start"), poly(p1, "ln s1"), C(*p1[47], 4, "t4")]
    y2 = base + rng.normal(0, 0.05, 80)
    y2[53:57] = 0.95
    p2 = pts(y2, 112)
    lo_y, hi_y = 112 + 50 - (1 + 1.3) / 2.6 * 50, 112 + 50 - (-1 + 1.3) / 2.6 * 50
    parts += [T(10, 102, "Contextual: обычное значение не вовремя", a="start"),
              L(15, round(lo_y, 1), 325, round(lo_y, 1), "ln s6", dash=True),
              L(15, round(hi_y, 1), 325, round(hi_y, 1), "ln s6", dash=True),
              poly(p2, "ln s1"), poly(p2[52:58], "ln s4"), *[C(*p, 3, "t4") for p in p2[53:57]]]
    y3 = base + rng.normal(0, 0.05, 80)
    y3[40:58] = -0.05 + rng.normal(0, 0.03, 18)
    p3 = pts(y3, 198)
    parts += [T(10, 188, "Collective: аномален целый участок", a="start"),
              poly(p3, "ln s1"), poly(p3[39:59], "ln s4")]
    return svg("tsa", 340, 258, "Три типа аномалий во временных рядах", *parts)


TSA = _ts_svg()


def tsa_chart(fig):
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    x = 10 + 1.5 * np.sin(2 * np.pi * t / 150) + rng.normal(0, 0.4, n)
    for i, d in ((70, 3.5), (160, -3.5), (240, 4.0)):
        x[i] += d
    w = 30
    ma = np.array([x[i - w:i].mean() if i >= w else np.nan for i in range(n)])
    sd = np.array([x[i - w:i].std() if i >= w else np.nan for i in range(n)])
    up, lo = ma + 3.5 * sd, ma - 3.5 * sd
    an = (x > up) | (x < lo)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, x, color=P[5], lw=1, label="ряд")
    ax.plot(t, ma, color=P[0], lw=1.5, label="скольз. среднее")
    ax.fill_between(t, lo, up, color=P[0], alpha=0.15, label="± 3.5σ")
    ax.scatter(t[an], x[an], color=P[3], s=30, zorder=5, label="аномалии")
    ax.set_xlabel("время"), ax.set_ylabel("значение")
    ax.legend(fontsize=7, loc="lower left", ncol=2)


# =============================================================== 24. topic_modeling_lda_nmf
TOPIC = svg(
    "topic", 340, 236, "Тематическая модель как разложение матрицы документ–слово",
    R(10, 30, 110, 120, "c6", 3), T(65, 70, "X"),
    R(10, 78, 110, 12, "s4"),
    T(135, 96, "≈"),
    R(150, 30, 40, 120, "c1", 3), T(170, 70, "W"),
    R(150, 78, 40, 12, "s4"),
    T(198, 96, "·"),
    R(206, 30, 124, 40, "c3", 3), T(268, 54, "H"),
    T(65, 168, "D × V"), T(65, 184, "документ–слово", "mut", fs=11),
    T(170, 168, "D × K"), T(170, 184, "док → тема", "mut", fs=11),
    T(268, 88, "K × V"), T(268, 104, "тема → слово", "mut", fs=11),
    T(268, 134, "строка d в X ≈", "t4", fs=11), T(268, 148, "смесь тем", "t4", fs=11),
    T(170, 212, "LDA: θ_d ~ Dir(α), φ_k ~ Dir(β) — вероятности", fs=11),
    T(170, 228, "NMF: X ≈ W·H, W ≥ 0, H ≥ 0 (TF-IDF)", fs=11),
)


# =============================================================== 25. torchserve
TORCHSERVE = svg(
    "tsrv", 340, 300, "Архитектура TorchServe",
    B(110, 8, 120, 28, "c6", "Клиент"),
    A(150, 36, 62, 56), A(170, 36, 170, 56), A(190, 36, 278, 56),
    B(10, 58, 100, 40, "c1", ["Inference", ":8080"]),
    B(120, 58, 100, 40, "c5", ["Management", ":8081"]),
    B(230, 58, 100, 40, "c6", ["Metrics", ":8082"]),
    A(60, 98, 60, 118), A(170, 98, 170, 118),
    B(10, 120, 320, 34, "c3", "Frontend: очередь запросов + batching"),
    *[A(x, 154, x, 178) for x in (60, 170, 280)],
    *[B(x, 180, 100, 60, "c2", ["preprocess", "inference", "postprocess"]) for x in (10, 120, 230)],
    T(170, 254, "workers (по процессу на копию модели)", "mut", fs=11),
    B(10, 262, 320, 28, "c6", "model_store/: model.mar (веса + handler)"),
)


# =============================================================== 26. transfer_learning_cnn
def _tl_row(y, title, cls, notes):
    out = [T(10, y - 8, title, a="start")]
    labs = [["ранние", "края"], ["средние", "части"], ["поздние", "объекты"]]
    for i, (x, c) in enumerate(zip((10, 88, 166), cls)):
        out.append(B(x, y, 64, 40, c, labs[i]))
    out.append(B(248, y, 82, 40, "c3", ["голова", "N классов"]))
    out += [A(74, y + 20, 86, y + 20), A(152, y + 20, 164, y + 20), A(230, y + 20, 246, y + 20)]
    for x, s, c in notes:
        out.append(T(x, y + 56, s, c, fs=11))
    return "".join(out)


TRANSFER = svg(
    "tlcnn", 340, 228, "Feature extraction и fine-tuning предобученной CNN",
    _tl_row(26, "Feature extraction: бэкбон заморожен", ("c6", "c6", "c6"),
            [(120, "заморожено (requires_grad=False)", "mut"), (289, "LR 1e-3", "t3")]),
    _tl_row(128, "Fine-tuning: верхние слои с малым LR", ("c6", "c1", "c1"),
            [(42, "заморож.", "mut"), (159, "LR 1e-5…1e-4", "t1"), (289, "LR 1e-3", "t3")]),
    R(10, 208, 12, 12, "c6", 2), T(27, 218, "заморожен", a="start", fs=11),
    R(110, 208, 12, 12, "c1", 2), T(127, 218, "дообучается", a="start", fs=11),
    R(220, 208, 12, 12, "c3", 2), T(237, 218, "новая голова", a="start", fs=11),
)


# =============================================================== 27. transformers
TRANSF = svg(
    "trf", 340, 356, "Блок энкодера трансформера",
    T(170, 28, "выход: контекстные векторы"),
    A(170, 82, 170, 40),
    R(60, 70, 220, 206, "s6", 8),
    T(290, 176, "× N", a="start"),
    B(90, 82, 160, 24, "c6", "Add & Norm"),
    A(170, 124, 170, 108),
    B(90, 124, 160, 30, "c1", "Feed-Forward"),
    A(170, 180, 170, 156),
    B(90, 180, 160, 24, "c6", "Add & Norm"),
    A(170, 218, 170, 206),
    B(90, 218, 160, 30, "c3", "Multi-Head Attention"),
    PA("M170,282 V266", arrow=False), PA("M130,266 H210", arrow=False),
    *[A(x, 266, x, 250) for x in (130, 170, 210)],
    *[T(x + 5, 263, s, "mut", a="start", fs=11) for x, s in ((130, "Q"), (170, "K"), (210, "V"))],
    PA("M170,275 H80 V192 H88", "ln s5"),
    PA("M170,167 H80 V94 H88", "ln s5"),
    T(74, 232, "residual", "t5", a="end", fs=11),
    C(170, 292, 10, "c6", "+"),
    A(110, 292, 158, 292), T(105, 296, "pos. enc.", "mut", a="end", fs=11),
    A(170, 320, 170, 304),
    B(90, 320, 160, 26, "c6", "Embedding токенов"),
)


def pe_chart(fig):
    d = 64
    pos = np.linspace(0, 100, 800)
    ax = fig.add_subplot(1, 1, 1)
    for i, c in zip((0, 4, 8, 16), P):
        ax.plot(pos, np.sin(pos / 10000 ** (2 * i / d)), color=c, lw=1.8, label=f"PE(pos, {2 * i})")
    ax.set_xlabel("позиция pos"), ax.set_ylabel("значение")
    ax.set_title("PE(pos, 2i) = sin(pos / 10000^(2i/d)), d = 64", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2), ax.set_ylim(-1.5, 1.2)


# =============================================================== 28. tree_ensembles
ENSEMBLE = svg(
    "ens", 340, 246, "Bagging и boosting деревьев",
    T(10, 16, "Bagging / Random Forest: параллельно", a="start"),
    B(10, 52, 46, 28, "c6", "D"),
    *[A(56, 66, 108, y + 13) for y in (28, 62, 96)],
    *[B(110, y, 84, 26, "c2", f"T{s} на D*{s}") for y, s in ((28, "₁"), (62, "₂"), (96, "₃"))],
    *[A(194, y + 13, 228, 66) for y in (28, 62, 96)],
    B(230, 50, 100, 32, "c1", ["среднее /", "голосование"], fs=11),
    T(10, 150, "Boosting: последовательно", a="start"),
    B(10, 164, 40, 28, "c6", "F₀"), A(50, 178, 73, 178),
    B(75, 164, 50, 28, "c3", "h₁"), A(125, 178, 148, 178),
    B(150, 164, 50, 28, "c3", "h₂"), A(200, 178, 216, 178),
    T(228, 182, "…"), A(238, 178, 248, 178),
    B(250, 164, 50, 28, "c3", "h_M"), A(300, 178, 318, 178), T(328, 182, "ŷ"),
    T(170, 214, "hₘ учится на остатках rₘ = y − Fₘ₋₁", fs=11),
    T(170, 234, "Fₘ = Fₘ₋₁ + η·hₘ", fs=11),
)


def ensemble_chart(fig):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(n_samples=1600, n_features=15, n_informative=8, flip_y=0.05, random_state=0)
    Xtr, Xte, ytr, yte = X[:1000], X[1000:], y[:1000], y[1000:]
    n = 150
    tree_err = 1 - DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    rf = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=1).fit(Xtr, ytr)
    probs = np.cumsum([e.predict_proba(Xte)[:, 1] for e in rf.estimators_], axis=0) / np.arange(1, n + 1)[:, None]
    rf_err = ((probs > 0.5) != yte).mean(1)
    gb = GradientBoostingClassifier(n_estimators=n, max_depth=3, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    gb_err = [(p != yte).mean() for p in gb.staged_predict(Xte)]
    k = np.arange(1, n + 1)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(tree_err, color=P[5], ls="--", lw=1.5, label="одно дерево")
    ax.plot(k, rf_err, color=P[1], lw=2, label="Random Forest")
    ax.plot(k, gb_err, color=P[2], lw=2, label="Gradient Boosting")
    ax.set_xlabel("число деревьев"), ax.set_ylabel("ошибка на тесте")
    ax.legend(fontsize=8)


# =============================================================== 29. tree_visualization
TREEVIS = svg(
    "tviz", 340, 246, "Устройство узла plot_tree на примере Iris",
    B(80, 8, 170, 80, "c6", ["petal width ≤ 0.8", "gini = 0.667", "samples = 150",
                             "value = [50, 50, 50]", "class = setosa"]),
    *[T(258, y, s, "mut", a="start", fs=11) for y, s in
      ((24, "← сплит"), (38, "← примесь"), (52, "← объектов"), (66, "← по классам"), (80, "← большинство"))],
    A(130, 88, 80, 126), A(200, 88, 250, 126),
    T(98, 112, "True", "mut", a="end", fs=11), T(232, 112, "False", "mut", a="start", fs=11),
    B(10, 128, 140, 66, "c3", ["gini = 0.0", "samples = 50", "value = [50, 0, 0]", "class = setosa"]),
    B(175, 128, 155, 80, "c2", ["petal width ≤ 1.75", "gini = 0.5", "samples = 100",
                                "value = [0, 50, 50]", "class = versicolor"]),
    T(170, 236, "filled=True: цвет — класс, насыщенность — чистота", "mut", fs=11),
)


# =============================================================== 30. tts_models
TTS = svg(
    "tts", 340, 306, "Пайплайн синтеза речи: текст, акустическая модель, вокодер",
    B(15, 8, 220, 26, "c6", "Текст"), A(125, 34, 125, 48),
    B(15, 50, 220, 40, "c6", ["Text analysis", "нормализация, G2P → фонемы"]), A(125, 90, 125, 106),
    B(15, 108, 220, 40, "c1", ["Acoustic model", "Tacotron 2 / FastSpeech"]), A(125, 148, 125, 164),
    B(15, 166, 220, 28, "c5", "mel-спектрограмма"), A(125, 194, 125, 210),
    B(15, 212, 220, 40, "c3", ["Vocoder", "HiFi-GAN / WaveGlow"]), A(125, 252, 125, 268),
    B(15, 270, 220, 28, "c2", "Waveform (22 кГц)"),
    PA("M245,108 H255 V252 H245", "ln s2", arrow=False),
    T(295, 176, "VITS", "t2"), T(295, 192, "end-to-end", "t2", fs=11),
)


# =============================================================== 31. universal_approximation_theorem
_hid = [50, 95, 140, 220]
UAT = svg(
    "uat", 340, 290, "Сеть с одним скрытым слоем как универсальный аппроксиматор",
    T(170, 16, "N нейронов (скрытый слой)"),
    *[L(54, 145, 156, y) for y in _hid],
    *[A(184, y, 272, 145) for y in _hid],
    C(40, 145, 14, "c6", "x"),
    *[C(170, y, 14, "c1", "σ") for y in _hid],
    T(170, 184, "⋮"),
    C(290, 145, 16, "c2", "Σ"),
    T(290, 120, "f̂(x)"),
    T(92, 88, "w₁, b₁", "mut", a="end", fs=11),
    T(232, 84, "v₁", "mut"), T(232, 212, "v_N", "mut"),
    T(170, 262, "f̂(x) = Σᵢ vᵢ·σ(wᵢ·x + bᵢ)"),
    T(170, 280, "|f̂ − f| < ε на компакте K при большом N", "mut", fs=11),
)


def uat_chart(fig):
    def sig(z):
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    x = np.linspace(-3, 3, 400)
    ax = fig.add_subplot(2, 1, 1)
    for k, c in ((3, P[5]), (10, P[0]), (50, P[3])):
        ax.plot(x, sig(k * (x + 0.5)) - sig(k * (x - 0.5)), color=c, lw=2, label=f"w = {k}")
    ax.set_title("σ(w(x+½)) − σ(w(x−½)): 2 нейрона → «ступенька»", fontsize=10)
    ax.legend(fontsize=8)
    rng = np.random.default_rng(0)
    f = np.sin(2 * x) + 0.3 * x ** 2 - 0.5 * np.abs(x)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, f, color="k", lw=2.5, label="f(x)")
    for n, c in ((3, P[2]), (10, P[1]), (50, P[0])):
        w = rng.normal(0, 4, n)
        b = rng.uniform(-3, 3, n) * -w
        H = sig(np.outer(x, w) + b)
        H = np.c_[H, np.ones_like(x)]
        v, *_ = np.linalg.lstsq(H, f, rcond=None)
        ax2.plot(x, H @ v, color=c, lw=1.6, ls="--", label=f"N = {n}")
    ax2.set_title("Больше нейронов — точнее приближение", fontsize=10)
    ax2.set_xlabel("x"), ax2.legend(fontsize=8)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# =============================================================== 32. vae
VAE = svg(
    "vae", 340, 306, "Архитектура VAE с reparameterization trick",
    B(95, 8, 90, 26, "c6", "x"), A(140, 34, 140, 44),
    '<polygon class="c1" points="70,46 210,46 180,86 100,86"/>',
    T(140, 63, "Encoder"), T(140, 79, "q_φ(z|x)", fs=11),
    A(125, 86, 100, 98), A(155, 86, 180, 98),
    B(70, 100, 60, 26, "c1", "μ"), B(150, 100, 60, 26, "c1", "log σ²"),
    T(275, 112, "KL → N(0, I)", "t4", fs=11),
    A(100, 126, 110, 148), A(180, 126, 170, 148),
    B(60, 150, 160, 30, "c5", "z = μ + σ ⊙ ε"),
    B(240, 150, 90, 30, "c6", "ε ~ N(0, I)"), A(240, 165, 222, 165),
    A(140, 180, 140, 193),
    '<polygon class="c2" points="100,195 180,195 210,235 70,235"/>',
    T(140, 212, "Decoder"), T(140, 228, "p_θ(x|z)", fs=11),
    T(275, 206, "градиент идёт", "mut", fs=11), T(275, 220, "через μ и σ", "mut", fs=11),
    A(140, 235, 140, 246),
    B(95, 248, 90, 26, "c2", "x̂"),
    T(170, 296, "Loss = Reconstruction + KL(q(z|x) ‖ N(0, I))", fs=11),
)


# =============================================================== 33. variational_inference
VI = svg(
    "vi", 340, 232, "Разложение log p(x) на ELBO и KL",
    T(170, 16, "log p(x) = ELBO(q) + KL(q(z) ‖ p(z|x))", fs=11),
    L(60, 40, 265, 40, "ln s6", dash=True), T(268, 44, "log p(x)", a="start", fs=11),
    R(70, 40, 60, 70, "c4"), T(100, 79, "KL", fs=11),
    R(70, 110, 60, 90, "c1"), T(100, 159, "ELBO", fs=11),
    R(200, 40, 60, 16, "c4"), T(268, 62, "KL ↓", "t4", a="start", fs=11),
    R(200, 56, 60, 144, "c1"), T(230, 132, "ELBO ↑", fs=11),
    L(60, 200, 275, 200),
    A(138, 120, 192, 120), T(165, 112, "max ELBO", "mut", fs=11),
    T(100, 218, "начальная q", fs=11), T(230, 218, "после оптимизации", fs=11),
)


def vi_chart(fig):
    from scipy.optimize import minimize
    from scipy.stats import norm
    z = np.linspace(-6, 6, 1200)
    dz = z[1] - z[0]
    p = 0.5 * norm.pdf(z, -2, 0.6) + 0.5 * norm.pdf(z, 2, 0.6)

    def rev_kl(th):
        q = norm.pdf(z, th[0], np.exp(th[1])) + 1e-300
        return np.sum(q * (np.log(q) - np.log(p + 1e-300))) * dz

    th = minimize(rev_kl, [1.0, 0.0], method="Nelder-Mead").x
    m = np.sum(z * p) * dz
    s = np.sqrt(np.sum((z - m) ** 2 * p) * dz)
    ax = fig.add_subplot(1, 1, 1)
    ax.fill_between(z, p, color=P[5], alpha=0.4, label="p(z|x) — бимодальное")
    ax.plot(z, norm.pdf(z, th[0], np.exp(th[1])), color=P[0], lw=2, label="q: min KL(q‖p) — VI")
    ax.plot(z, norm.pdf(z, m, s), color=P[2], lw=2, ls="--", label="q: min KL(p‖q)")
    ax.set_xlabel("z"), ax.set_ylabel("плотность")
    ax.set_title("Reverse KL цепляется за одну моду", fontsize=10)
    ax.legend(fontsize=7, loc="upper left"), ax.set_ylim(0, 0.75)


# =============================================================== 34. video_audio_models
def _fusion_row(y0, title, kind):
    o = [T(10, y0 + 14, title, a="start"),
         B(10, y0 + 22, 60, 22, "c1", "видео", fs=11), B(10, y0 + 50, 60, 22, "c3", "аудио", fs=11)]
    if kind == "early":
        o += [A(70, y0 + 33, 91, y0 + 43), A(70, y0 + 61, 91, y0 + 51), C(100, y0 + 47, 10, "c5", "⊕"),
              A(110, y0 + 47, 123, y0 + 47), B(125, y0 + 34, 110, 26, "c5", "общая сеть"),
              A(235, y0 + 47, 263, y0 + 47), T(274, y0 + 51, "ŷ")]
    else:
        nm = ("энкодер V", "энкодер A") if kind == "mid" else ("модель V", "модель A")
        o += [A(70, y0 + 33, 93, y0 + 33), A(70, y0 + 61, 93, y0 + 61),
              B(95, y0 + 22, 90, 22, "c1", nm[0], fs=11), B(95, y0 + 50, 90, 22, "c3", nm[1], fs=11),
              A(185, y0 + 33, 213, y0 + 33), A(185, y0 + 61, 213, y0 + 61)]
        if kind == "mid":
            o += [B(215, y0 + 22, 75, 50, "c5", ["cross-", "attention"], fs=11)]
        else:
            o += [T(199, y0 + 27, "ŷ_v", "mut", fs=11), T(199, y0 + 77, "ŷ_a", "mut", fs=11),
                  B(215, y0 + 22, 75, 50, "c6", ["среднее /", "голос"], fs=11)]
        o += [A(290, y0 + 47, 312, y0 + 47), T(322, y0 + 51, "ŷ")]
    return "".join(o)


AVFUSION = svg(
    "avf", 340, 282, "Три способа слияния видео и аудио",
    _fusion_row(0, "Early fusion: объединяем на входе", "early"),
    _fusion_row(94, "Intermediate: cross-attention в середине", "mid"),
    _fusion_row(188, "Late fusion: объединяем предсказания", "late"),
)


# =============================================================== 35. video_generation
VIDGEN = svg(
    "vgen", 340, 318, "Латентная видео-диффузия на примере CogVideoX",
    B(10, 10, 150, 28, "c6", "Видео (обучение)"),
    A(85, 38, 85, 56, dash=True),
    B(10, 58, 150, 40, "c5", ["3D VAE encoder", "÷4 время, ÷8 простр."]),
    B(190, 10, 140, 28, "c6", "шум (генерация)"),
    A(85, 98, 115, 118, dash=True), A(260, 38, 215, 118),
    B(60, 120, 200, 28, "c1", "латент z: 13 × 60 × 90 × C"),
    A(160, 148, 160, 168),
    B(60, 170, 200, 44, "c3", ["DiT: 3D attention + 3D RoPE", "денойзинг × T шагов"], fs=11),
    B(280, 176, 52, 32, "c6", "текст", fs=11), A(280, 192, 262, 192),
    A(160, 214, 160, 234),
    B(60, 236, 200, 28, "c5", "3D VAE decoder"),
    A(160, 264, 160, 280),
    B(60, 282, 200, 28, "c2", "клип 49 × 480 × 720"),
)


# =============================================================== 36. vision_attention_mechanisms
SEBLOCK = svg(
    "se", 340, 286, "Блок Squeeze-and-Excitation",
    B(15, 10, 150, 30, "c1", "X: C × H × W"), A(90, 40, 90, 60),
    B(15, 62, 150, 28, "c6", "Squeeze: GAP → C"), A(90, 90, 90, 110),
    B(15, 112, 150, 28, "c5", "FC → C/r (r=16), ReLU", fs=11), A(90, 140, 90, 160),
    B(15, 162, 150, 28, "c5", "FC → C, sigmoid"), A(90, 190, 90, 210),
    B(15, 212, 150, 28, "c3", "веса каналов s"),
    T(172, 155, "Excitation", "mut", a="start", fs=11),
    A(165, 226, 245, 226),
    PA("M165,25 H260 V211"),
    T(212, 18, "X без изменений", "mut", fs=11),
    C(260, 226, 13, "c3", "⊗"),
    A(260, 239, 260, 260),
    T(260, 276, "X̃ = s ⊙ X"),
)


# =============================================================== 37. vision_transformers_vit
def _vit_svg():
    o = [R(15, 15, 96, 96, "c6", 2)]
    for i in (1, 2):
        o += [L(15 + 32 * i, 15, 15 + 32 * i, 111, "ln s6"), L(15, 15 + 32 * i, 111, 15 + 32 * i, "ln s6")]
    o += [T(130, 40, "224×224×3 → патчи 16×16", a="start", fs=11),
          T(130, 60, "N = (224/16)² = 196", a="start", fs=11),
          T(130, 80, "патч: 16·16·3 = 768", a="start", fs=11),
          A(63, 111, 63, 126),
          B(15, 128, 310, 26, "c5", "Linear projection E (768 → D)"),
          A(170, 154, 170, 168),
          B(15, 170, 50, 26, "c3", "[CLS]", fs=11)]
    for i, s in enumerate(["x₁", "x₂", "x₃", "x₄", "…", "x₁₉₆"]):
        o.append(B(75 + 36 * i, 170, 30, 26, "c1", s, fs=11))
    o += [T(190, 214, "+ position embeddings E_pos", "mut", fs=11),
          A(40, 196, 40, 224),
          B(15, 226, 310, 30, "c1", "Transformer Encoder × L (MSA + MLP)"),
          A(40, 256, 40, 272),
          B(15, 274, 130, 28, "c2", "MLP Head → класс", fs=11),
          T(155, 292, "берём выход [CLS]", "mut", a="start", fs=11)]
    return svg("vit", 340, 310, "Архитектура Vision Transformer", *o)


VIT = _vit_svg()


# =============================================================== 38. zero_shot_learning
_zs = [("cat", "0.21"), ("dog", "0.24"), ("zebra", "0.31"), ("airplane", "0.12")]
ZSL = svg(
    "zsl", 340, 250, "Zero-shot классификация с CLIP",
    B(10, 10, 110, 28, "c6", "Фото"), A(65, 38, 65, 56),
    B(10, 58, 110, 28, "c1", "Image encoder"),
    B(150, 10, 180, 28, "c6", "«a photo of a {class}»"), A(240, 38, 240, 56),
    B(150, 58, 180, 28, "c3", "Text encoder"),
    A(65, 86, 65, 98), A(240, 86, 240, 98),
    T(65, 112, "I", "t1"), T(240, 112, "T₁ … T₄", "t3"),
    T(152, 112, "cos(I, Tⱼ)", fs=11),
    *[B(20 + 78 * j, 124, 72, 40, "c2" if nm == "zebra" else "c6", [nm, v], fs=11)
      for j, (nm, v) in enumerate(_zs)],
    A(170, 164, 170, 184),
    B(60, 186, 220, 28, "c2", "softmax(100·cos) → «zebra»"),
    T(170, 236, "zebra не было в обучении — класс задан текстом", "mut", fs=11),
)


# =============================================================== словарь
VISUALS = {
    "source_separation": [
        {"title": "Разделение по маскам спектрограммы", "svg": SOURCE_SEP,
         "caption": "U-Net предсказывает для каждого источника маску Mᵢ ∈ [0, 1] на |X|; дорожку собирают "
                    "через iSTFT с фазой исходного микса. Conv-TasNet делает то же на raw waveform."},
        {"title": "ICA: слепое разделение двух сигналов", "chart": ica_chart,
         "caption": "Два источника смешаны матрицей A; FastICA восстанавливает их по независимости — "
                    "но только с точностью до порядка, знака и масштаба."},
    ],
    "spectral_embedding": [
        {"title": "Алгоритм Spectral Embedding", "svg": SPECTRAL,
         "caption": "Сильно связанные точки графа получают близкие значения собственных векторов лапласиана; "
                    "константный v₀ с λ₀ = 0 не несёт информации и отбрасывается."},
        {"title": "Луны после Spectral Embedding", "chart": spectral_chart,
         "caption": "kNN-граф почти не связывает две луны, поэтому в координатах (v₁, v₂) они разведены "
                    "и легко разделяются даже линейно."},
    ],
    "state_space_models": [
        {"title": "Граф модели пространства состояний", "svg": SSM,
         "caption": "Скрытое состояние xₜ эволюционирует через A (шум Q), а мы видим только yₜ = C·xₜ + шум R. "
                    "Управление B·uₜ (если есть) добавляется в уравнение состояния."},
        {"title": "Kalman Filter: трекинг позиции", "chart": kalman_chart,
         "caption": "Модель постоянной скорости: оценка фильтра гораздо ближе к истине, чем шумные измерения, "
                    "а коридор ± 2σ быстро сужается после первых шагов."},
    ],
    "statistical_anomaly_detection": [
        {"title": "Метод IQR на boxplot", "svg": STAT_AD,
         "caption": "Точки за пределами [Q1 − 1.5·IQR; Q3 + 1.5·IQR] считаются выбросами; "
                    "квартили почти не сдвигаются от самих выбросов."},
        {"title": "Z-score против MAD", "chart": stat_ad_chart,
         "caption": "6% выбросов раздувают σ, и порог μ + 3σ уходит правее самих выбросов — они пропущены. "
                    "Порог на медиане и MAD устойчив и их ловит."},
    ],
    "statistical_methods": [
        {"title": "Схема bootstrap", "svg": STAT_METHODS,
         "caption": "Из выборки B раз берём ресэмпл того же размера с возвращением, считаем статистику θ*; "
                    "2.5-й и 97.5-й перцентили её распределения дают 95% ДИ."},
        {"title": "Bootstrap-распределение среднего", "chart": bootstrap_chart,
         "caption": "Для скошенных данных bootstrap-распределение тоже слегка скошено — "
                    "перцентильный ДИ получается несимметричным вокруг среднего."},
    ],
    "statistical_tests": [
        {"title": "Как выбрать тест", "svg": STAT_TESTS,
         "caption": "Упрощённое дерево: для числовых данных выбор зависит от нормальности (Wilcoxon — для парных "
                    "выборок), для категорий — χ² или McNemar; связь двух величин — корреляция."},
    ],
    "statistical_tests_t_test_mcnemar": [
        {"title": "Таблица McNemar", "svg": MCNEMAR,
         "caption": "Тест смотрит только на расхождения b и c: при H₀ модели ошибаются «по очереди» одинаково часто. "
                    "При exact=True вместо χ² — биномиальный тест по b ~ Bin(b + c, ½)."},
        {"title": "Paired t-test: p-value как площадь", "chart": ttest_chart,
         "caption": "p-value — площадь хвостов за |t|; если t выходит за красные критические значения "
                    "(α = 0.05, df = n − 1 = 9), разницу считают значимой."},
    ],
    "stemming_lemmatization": [
        {"title": "Stemming против лемматизации", "svg": STEM,
         "caption": "Porter режет окончания по правилам и может выдать не-слово (easili), а нерегулярные формы "
                    "(ran, better) не трогает; лемматизатор со словарём и POS-тегом возвращает run и good."},
    ],
    "stylegan_architecture": [
        {"title": "Mapping и synthesis network", "svg": STYLEGAN,
         "caption": "z через 8-слойный MLP превращается в w; на каждом разрешении аффинное A(w) задаёт "
                    "y_s, y_b для AdaIN, а шум B добавляет стохастические детали. Вход — константа 4×4."},
    ],
    "subspace_clustering": [
        {"title": "CLIQUE: плотные единицы", "svg": SUBSPACE,
         "caption": "Гистограммы — проекции на оси: столбец A плотен только по x₁ (в 2D его ячейки ниже порога), "
                    "B плотен в обоих измерениях. Кандидаты в k-мерные единицы строятся из плотных (k−1)-мерных."},
    ],
    "surrogate_models": [
        {"title": "Глобальный суррогат", "svg": SURROGATE,
         "caption": "Суррогат учится имитировать предсказания чёрного ящика, а не истинную разметку; "
                    "fidelity показывает, насколько можно доверять полученным правилам."},
        {"title": "Fidelity против глубины дерева", "chart": surrogate_chart,
         "caption": "Fidelity быстро растёт и уже при глубине 4 выходит на плато выше 0.9; дальше дерево "
                    "только усложняется (листьев до 2ᵈ) без выигрыша — берут минимальную достаточную глубину."},
    ],
    "swarm_intelligence": [
        {"title": "Шаг частицы в PSO", "svg": SWARM,
         "caption": "Новая скорость — сумма инерции, тяги к личному рекорду и тяги к лучшему решению роя "
                    "(r₁, r₂ случайны); частица сдвигается на x + v′."},
        {"title": "PSO на функции Rastrigin", "chart": pso_chart,
         "caption": "Сверху — рой за 40 итераций собирается у глобального минимума; снизу — в 10D рой часто "
                    "застревает в локальном минимуме, линейно убывающая w меняет баланс exploration/exploitation."},
    ],
    "syntactic_parsing": [
        {"title": "Dependency-разбор", "svg": PARSE,
         "caption": "Стрелка идёт от главного слова к зависимому, над ней — тип связи; корень — глагол sat. "
                    "Каждое слово, кроме корня, имеет ровно одного родителя."},
    ],
    "tabnet": [
        {"title": "Decision step TabNet", "svg": TABNET,
         "caption": "Attentive Transformer строит разреженную маску M[d] с учётом prior P (γ ограничивает повторное "
                    "использование признаков); выход Feature Transformer делится на a[d] (в ответ) и d[d] (на следующий шаг)."},
        {"title": "Softmax против sparsemax", "chart": tabnet_chart,
         "caption": "Softmax даёт всем признакам ненулевой вес, sparsemax обнуляет слабые — "
                    "поэтому маски TabNet реально отбирают признаки."},
    ],
    "tabu_search": [
        {"title": "Цикл Tabu Search", "svg": TABU,
         "caption": "Переходим в лучший разрешённый сосед, даже если он хуже текущего; tabu-список не даёт вернуться "
                    "назад, а аспирация снимает запрет, если ход даёт новый рекорд."},
        {"title": "Tabu search против жадного спуска", "chart": tabu_chart,
         "caption": "Жадный спуск останавливается в ближайшем локальном минимуме; tabu search с памятью "
                    "на 10 решений выбирается из ямы и находит минимум глубже."},
    ],
    "target_encoding": [
        {"title": "Сглаживание и out-of-fold", "svg": TARGET,
         "caption": "Сглаживание тянет редкие категории к глобальному среднему, а out-of-fold схема кодирует "
                    "каждый фолд статистиками остальных — таргет строки не попадает в её признак."},
        {"title": "Влияние α на кодировку", "chart": target_chart,
         "caption": "При малом n_c кодировка близка к глобальному ȳ; чем больше α, тем больше объектов нужно, "
                    "чтобы «поверить» среднему категории."},
    ],
    "tcn": [
        {"title": "Dilated causal convolution", "svg": TCN,
         "caption": "Выход в момент t зависит только от t и прошлого; с dilation 1, 2, 4 три слоя с k = 2 "
                    "охватывают 8 шагов — охват растёт экспоненциально с глубиной."},
        {"title": "Рецептивное поле от числа блоков", "chart": tcn_chart,
         "caption": "С удвоением dilation рецептивное поле растёт как 2ᴸ, без dilation — линейно; "
                    "для истории в тысячи шагов хватает около 8 блоков."},
    ],
    "td_learning": [
        {"title": "MC против TD(0)", "svg": TD,
         "caption": "MC ждёт конца эпизода и обновляет V по полному возврату Gₜ; TD(0) обновляет после "
                    "одного шага по текущей оценке V(sₜ₊₁) (bootstrapping)."},
        {"title": "Random walk: TD быстрее MC", "chart": td_chart,
         "caption": "На классической задаче random walk TD(0) снижает ошибку быстрее MC при сравнимых α; "
                    "большой α ускоряет начало, но оставляет больший шумовой «пол»."},
    ],
    "tensorflow_keras_full_guide": [
        {"title": "Рабочий цикл Keras", "svg": KERAS,
         "caption": "Данные → модель любого из трёх API → compile → fit с callbacks → оценка → сохранение; "
                    "SavedModel затем разворачивают в TF Serving, TF Lite или TF.js."},
    ],
    "tensorflow_serving": [
        {"title": "Архитектура TF Serving", "svg": TFSERV,
         "caption": "Source следит за папкой модели, Loader загружает новую версию, Manager переключает "
                    "трафик между версиями; клиенты обращаются по REST (8501) или gRPC (8500)."},
    ],
    "text_to_image_generation": [
        {"title": "Латентная диффузия по промпту", "svg": T2I,
         "caption": "Эмбеддинги токенов промпта входят в денойзер через cross-attention; на каждом шаге CFG смешивает "
                    "условный и безусловный прогноз шума, а VAE-декодер превращает z₀ в картинку."},
    ],
    "tfidf": [
        {"title": "TF-IDF на примере", "svg": TFIDF,
         "caption": "«this» есть во всех документах, поэтому IDF = 0 и вес обнуляется; редкое «second» "
                    "весит больше, чем частое в документе «document»."},
        {"title": "Форма IDF", "chart": tfidf_chart,
         "caption": "IDF быстро падает с ростом df; сглаженный вариант sklearn не обнуляет слова, встречающиеся "
                    "во всех документах (минимум 1)."},
    ],
    "time_series_anomalies": [
        {"title": "Типы аномалий", "svg": TSA,
         "caption": "Point — выброс за пределы обычного диапазона; contextual — значение в пределах диапазона "
                    "(пунктир), но не в ту фазу; collective — каждая точка нормальна, аномален участок целиком."},
        {"title": "Скользящее среднее ± 3.5σ", "chart": tsa_chart,
         "caption": "Среднее и σ берутся по предыдущим 30 точкам, поэтому выброс не расширяет свой коридор "
                    "(зато расширяет коридор следующих точек); все три всплеска пойманы."},
    ],
    "topic_modeling_lda_nmf": [
        {"title": "Темы как разложение матрицы", "svg": TOPIC,
         "caption": "Строка документа в X приближается смесью тем (строка W) по словарям тем (строки H). "
                    "В LDA это вероятности с априорными Dirichlet, в NMF — просто неотрицательные веса."},
    ],
    "torchserve": [
        {"title": "Архитектура TorchServe", "svg": TORCHSERVE,
         "caption": "Inference API принимает запросы, frontend собирает их в батчи и раздаёт workers с handler; "
                    "Management API регистрирует модели из .mar и масштабирует workers."},
    ],
    "transfer_learning_cnn": [
        {"title": "Что замораживать", "svg": TRANSFER,
         "caption": "Голова всегда новая; при feature extraction обучается только она, при fine-tuning "
                    "размораживают верхние блоки с LR на 1–2 порядка меньше, чем у головы."},
    ],
    "transformers": [
        {"title": "Блок энкодера", "svg": TRANSF,
         "caption": "Вход — эмбеддинги плюс позиционное кодирование; в блоке attention и FFN, каждый обёрнут "
                    "residual-связью и LayerNorm (Add & Norm). Блоки повторяются N раз."},
        {"title": "Синусоидальное позиционное кодирование", "chart": pe_chart,
         "caption": "Каждая пара измерений — синус/косинус своей частоты: младшие меняются быстро, старшие медленно, "
                    "поэтому любая позиция получает уникальный «отпечаток»."},
    ],
    "tree_ensembles": [
        {"title": "Bagging против boosting", "svg": ENSEMBLE,
         "caption": "Bagging учит глубокие деревья независимо на bootstrap-выборках и усредняет (снижает дисперсию); "
                    "boosting добавляет мелкие деревья по очереди, каждое исправляет остатки предыдущих (снижает смещение)."},
        {"title": "Ошибка от числа деревьев", "chart": ensemble_chart,
         "caption": "Оба ансамбля заметно лучше одиночного дерева; у RF ошибка быстро выходит на плато, "
                    "boosting улучшается постепенно и требует контроля числа итераций."},
    ],
    "tree_visualization": [
        {"title": "Как читать узел plot_tree", "svg": TREEVIS,
         "caption": "Дерево глубины 2 на Iris: левая ветка — условие истинно. Лист setosa чистый (gini = 0), "
                    "правый узел делит versicolor и virginica по ширине лепестка."},
    ],
    "tts_models": [
        {"title": "Пайплайн TTS", "svg": TTS,
         "caption": "Двухэтапная схема: акустическая модель строит mel-спектрограмму, вокодер превращает её в звук; "
                    "VITS делает оба шага одной end-to-end моделью."},
    ],
    "universal_approximation_theorem": [
        {"title": "Сеть из теоремы", "svg": UAT,
         "caption": "Один скрытый слой из N нейронов σ(wᵢx + bᵢ) и линейный выход Σ vᵢ — этого достаточно, "
                    "чтобы приблизить любую непрерывную f на компакте; теорема ничего не говорит о том, как найти веса."},
        {"title": "Ступеньки и приближение", "chart": uat_chart,
         "caption": "Разность двух сигмоид даёт «ступеньку», из ступенек складывается любая функция. "
                    "Снизу — случайные нейроны с подобранными vᵢ: при N = 50 кривая почти совпадает с f."},
    ],
    "vae": [
        {"title": "Архитектура VAE", "svg": VAE,
         "caption": "Энкодер выдаёт μ и log σ², случайность вынесена в ε, поэтому градиент проходит через z к энкодеру. "
                    "KL тянет q(z|x) к N(0, I) и делает латентное пространство гладким."},
    ],
    "variational_inference": [
        {"title": "ELBO и KL", "svg": VI,
         "caption": "log p(x) не зависит от q, поэтому рост ELBO равносилен уменьшению KL(q ‖ p(z|x)): "
                    "оптимизация заменяет точный вывод."},
        {"title": "Mode-seeking поведение VI", "chart": vi_chart,
         "caption": "Гауссово q, минимизирующее KL(q‖p), садится на одну моду и недооценивает неопределённость; "
                    "прямой KL(p‖q) накрывает обе моды, но кладёт массу в провал между ними."},
    ],
    "video_audio_models": [
        {"title": "Early, intermediate и late fusion", "svg": AVFUSION,
         "caption": "Early — склеиваем признаки на входе, late — усредняем ответы отдельных моделей, "
                    "intermediate — модальности обмениваются информацией через cross-attention."},
    ],
    "video_generation": [
        {"title": "Латентная видео-диффузия", "svg": VIDGEN,
         "caption": "3D VAE сжимает 49 кадров 480×720 в латент 13×60×90; DiT денойзит весь клип сразу с полным "
                    "3D-вниманием, что и даёт временную согласованность. Энкодер нужен только при обучении."},
    ],
    "vision_attention_mechanisms": [
        {"title": "Squeeze-and-Excitation", "svg": SEBLOCK,
         "caption": "GAP сжимает каждую карту в число, два FC с бутылочным горлышком C/r дают веса s ∈ (0, 1) "
                    "для каналов, и исходный X перевзвешивается по каналам."},
    ],
    "vision_transformers_vit": [
        {"title": "Архитектура ViT", "svg": VIT,
         "caption": "Картинка режется на 196 патчей 16×16, каждый проецируется в D = 768; к последовательности "
                    "добавляют [CLS] и позиционные эмбеддинги, а класс предсказывают по выходу [CLS]."},
    ],
    "zero_shot_learning": [
        {"title": "Zero-shot с CLIP", "svg": ZSL,
         "caption": "Классы задаются текстовыми промптами: побеждает класс с наибольшим косинусным сходством "
                    "эмбеддингов картинки и текста (значения cos условные)."},
    ],
}
