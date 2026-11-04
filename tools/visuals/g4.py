# -*- coding: utf-8 -*-
"""Рисунки для шпаргалок группы g4 (latent_semantic_analysis … multi_output_regression)."""

import math

import numpy as np

BLUE, GREEN, ORANGE, RED, PURPLE, GREY = ["#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"]


# ---------------------------------------------------------------- SVG-помощники
def _svg(p, w, h, body, label):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">'
            f'<defs><marker id="{p}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            f'{body}</svg>')


def _st(fs=None, bold=False):
    parts = []
    if fs and fs != 12:
        parts.append(f"font-size:{fs}px")
    if bold:
        parts.append("font-weight:600")
    return f' style="{";".join(parts)}"' if parts else ""


def T(x, y, s, cls=None, fs=None, anchor="middle", bold=False, extra=""):
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:g}" y="{y:g}"{c}{a}{_st(fs, bold)}{extra}>{s}</text>'


def B(x, y, w, h, cls, lines=(), fs=12, sub_fs=11, rx=6, dash=False):
    """Прямоугольник с центрированным текстом; строки после первой — приглушённые."""
    d = ' stroke-dasharray="4 3"' if dash else ""
    out = [f'<rect class="{cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}"{d}/>']
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    if n:
        lh = 15
        y0 = y + h / 2 - (n - 1) * lh / 2 + 4
        for i, s in enumerate(lines):
            if i == 0:
                out.append(T(x + w / 2, y0 + i * lh, s, fs=fs))
            else:
                out.append(T(x + w / 2, y0 + i * lh, s, cls="mut", fs=sub_fs))
    return "".join(out)


def A(p, x1, y1, x2, y2, cls="ln", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return (f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"{d} '
            f'marker-end="url(#{p}-a)"/>')


def L(x1, y1, x2, y2, cls="ln", dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"{d}/>'


def PA(p, d, cls="ln", dash=True):
    da = ' stroke-dasharray="4 3"' if dash else ""
    return f'<path class="{cls}" d="{d}"{da} marker-end="url(#{p}-a)"/>'


def C(x, y, r, cls, s=None, fs=12):
    out = f'<circle class="{cls}" cx="{x:g}" cy="{y:g}" r="{r:g}"/>'
    if s:
        out += T(x, y + 4, s, fs=fs)
    return out


# ================================================================ 1. LSA
def _lsa_svg():
    p = "lsa"
    b = [T(55, 30, "TF-IDF", "mut", 11),
         B(10, 40, 90, 110, "c6", ["X"]), T(55, 166, "m × n"),
         T(113, 100, "≈", fs=18),
         B(126, 40, 34, 110, "c1", ["Uₖ"]), T(143, 166, "m × k"),
         B(170, 40, 34, 34, "c3", ["Σₖ"]), T(187, 90, "k × k"),
         B(214, 40, 110, 34, "c2", ["Vₖᵀ"]), T(269, 90, "k × n"),
         T(269, 120, "k ≪ m, n", "mut", 11),
         T(175, 196, "строки Uₖ — термы в пространстве концептов", "mut", 11),
         T(175, 213, "столбцы ΣₖVₖᵀ — документы", "mut", 11)]
    return _svg(p, 350, 225, "".join(b), "Усечённое SVD term-document матрицы")


def _lsa_chart(fig):
    rng = np.random.default_rng(0)
    n_docs, n_terms, k_true = 80, 60, 3
    topics = np.zeros((k_true, n_terms))
    for t in range(k_true):
        topics[t, t * 20:(t + 1) * 20] = rng.dirichlet(np.ones(20))
    topics = 0.9 * topics + 0.1 / n_terms
    theta = rng.dirichlet(np.ones(k_true) * 0.3, n_docs)
    counts = np.array([rng.multinomial(60, th @ topics) for th in theta]).astype(float)
    tf = counts / counts.sum(1, keepdims=True)
    idf = np.log((1 + n_docs) / (1 + (counts > 0).sum(0))) + 1
    X = tf * idf
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)[:15]
    ax = fig.add_subplot(111)
    ks = np.arange(1, len(s) + 1)
    ax.bar(ks, s, color=[BLUE if k <= k_true else GREY for k in ks])
    ax.set_xlabel("номер сингулярного значения k")
    ax.set_ylabel("σₖ")
    ax.set_title("Спектр TF-IDF матрицы (3 скрытые темы)", fontsize=10)
    ax.set_xticks(ks[::2])
    fig.tight_layout()


# ================================================================ 2. LDA (дискриминантный анализ)
def _lda_svg():
    p = "lda"
    ux, uy = math.cos(math.radians(30)), -math.sin(math.radians(30))   # длинная ось облаков
    nx, ny = -uy, ux                                                    # перпендикуляр (0.5, 0.866)
    cx, cy = 165, 110
    c0 = (cx - 24 * nx, cy - 24 * ny)
    c1 = (cx + 24 * nx, cy + 24 * ny)
    b = [f'<ellipse class="c1" cx="{c0[0]:.1f}" cy="{c0[1]:.1f}" rx="78" ry="17" transform="rotate(-30 {c0[0]:.1f} {c0[1]:.1f})"/>',
         f'<ellipse class="c4" cx="{c1[0]:.1f}" cy="{c1[1]:.1f}" rx="78" ry="17" transform="rotate(-30 {c1[0]:.1f} {c1[1]:.1f})"/>',
         T(c0[0] - 62, c0[1] - 2, "класс 0", "t1", 11), T(c1[0] + 66, c1[1] + 22, "класс 1", "t4", 11),
         A(p, cx - 125 * ux, cy - 125 * uy, cx + 125 * ux, cy + 125 * uy, "ln s3"),
         T(cx + 125 * ux + 4, cy + 125 * uy - 6, "PCA", "t3", anchor="end", bold=True),
         A(p, cx - 95 * nx, cy - 95 * ny, cx + 95 * nx, cy + 95 * ny, "ln s2"),
         T(cx + 95 * nx + 8, cy + 95 * ny - 2, "LDA", "t2", anchor="start", bold=True),
         T(180, 236, "J(w) = wᵀS_B w / wᵀS_W w → max", fs=12)]
    return _svg(p, 360, 248, "".join(b), "Направления PCA и LDA для двух классов")


def _lda_data():
    rng = np.random.default_rng(1)
    a = math.radians(30)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    cov = R @ np.diag([3.0, 0.12]) @ R.T
    mu_shift = R @ np.array([0, 0.9])
    X0 = rng.multivariate_normal(-mu_shift, cov, 300)
    X1 = rng.multivariate_normal(mu_shift, cov, 300)
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(300), np.ones(300)]
    return X, y


def _lda_chart(fig):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = _lda_data()
    zp = PCA(1).fit_transform(X)[:, 0]
    zl = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)[:, 0]
    for i, (z, name) in enumerate([(zp, "Проекция на PC1 (PCA)"), (zl, "Проекция на LD1 (LDA)")]):
        ax = fig.add_subplot(2, 1, i + 1)
        bins = np.linspace(z.min(), z.max(), 40)
        ax.hist(z[y == 0], bins, color=BLUE, alpha=0.6, label="класс 0")
        ax.hist(z[y == 1], bins, color=RED, alpha=0.6, label="класс 1")
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
        ax.legend(fontsize=8)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 3. LightGBM
def _lightgbm_svg():
    p = "lgb"
    b = [T(88, 20, "Level-wise", bold=True), T(88, 35, "(XGBoost)", "mut", 11),
         T(272, 20, "Leaf-wise", bold=True), T(272, 35, "(LightGBM)", "mut", 11)]
    # level-wise
    for (x1, y1, x2, y2) in [(88, 55, 50, 100), (88, 55, 126, 100), (50, 100, 30, 145), (50, 100, 70, 145),
                             (126, 100, 106, 145), (126, 100, 146, 145)]:
        b.append(L(x1, y1, x2, y2))
    for x, y in [(88, 55), (50, 100), (126, 100)]:
        b.append(C(x, y, 10, "c1"))
    for x in (30, 70, 106, 146):
        b.append(C(x, 145, 10, "c2"))
    # leaf-wise
    for (x1, y1, x2, y2) in [(272, 55, 232, 100), (272, 55, 312, 100), (312, 100, 287, 145), (312, 100, 337, 145),
                             (287, 145, 264, 190), (287, 145, 310, 190)]:
        b.append(L(x1, y1, x2, y2))
    b += [C(272, 55, 10, "c3", "1", 11), C(312, 100, 10, "c3", "2", 11), C(287, 145, 10, "c3", "3", 11)]
    for x, y in [(232, 100), (337, 145), (264, 190), (310, 190)]:
        b.append(C(x, y, 10, "c2"))
    b.append(L(180, 45, 180, 200, "ln s6", dash=True))
    b += [T(88, 222, "все листы уровня сразу", "mut", 11), T(272, 222, "лист с max Δloss", "mut", 11),
          T(180, 244, "рост ограничивают num_leaves и max_depth", "mut", 11)]
    return _svg(p, 360, 254, "".join(b), "Level-wise и leaf-wise рост деревьев")


# ================================================================ 4. LIME
def _lime_svg():
    p = "lime"
    steps = [("c6", "x — объясняемый пример"), ("c1", "возмущения z вокруг x"), ("c3", "f(z) — ответы black box"),
             ("c5", "веса πₓ(z) по близости к x"), ("c2", "взвешенная линейная g(z)"),
             ("c4", "коэффициенты g = важность")]
    b = []
    y = 10
    for i, (cls, s) in enumerate(steps):
        b.append(B(60, y, 240, 28, cls, [s]))
        b.append(T(45, y + 19, f"{i + 1}", "mut", 11, anchor="end"))
        if i < len(steps) - 1:
            b.append(A(p, 180, y + 28, 180, y + 40))
        y += 42
    b.append(T(180, y + 12, "ξ(x) = argmin L(f, g, πₓ) + Ω(g)", fs=12))
    return _svg(p, 360, y + 24, "".join(b), "Шаги алгоритма LIME")


def _lime_chart(fig):
    rng = np.random.default_rng(2)
    f = lambda x: 1 / (1 + np.exp(-3 * np.sin(1.3 * x) - 0.4 * x))
    x0 = 0.0
    xs = np.linspace(-3, 4, 400)
    z = x0 + rng.normal(0, 0.8, 60)
    w = np.exp(-((z - x0) ** 2) / (2 * 0.35 ** 2))
    W = np.sqrt(w)
    Amat = np.c_[np.ones_like(z), z - x0]
    coef = np.linalg.lstsq(Amat * W[:, None], f(z) * W, rcond=None)[0]
    ax = fig.add_subplot(111)
    ax.plot(xs, f(xs), color=GREY, lw=2, label="f(x) — black box")
    ax.scatter(z, f(z), s=10 + 60 * w, color=ORANGE, alpha=0.7, label="возмущения (размер = вес)")
    xl = np.linspace(x0 - 0.6, x0 + 0.6, 10)
    ax.plot(xl, coef[0] + coef[1] * (xl - x0), color=GREEN, lw=2.2, label="локальная g(x)")
    ax.scatter([x0], [f(x0)], s=70, color=RED, zorder=5, label="объясняемый x")
    ax.set_xlabel("признак")
    ax.set_ylabel("P(класс)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()


# ================================================================ 5. Линейная алгебра и оптимизация
def _linalg_svg():
    p = "lao"
    cy, r = 78, 26
    cxs = [40, 130, 225, 325]
    rx, ry = 36, 13

    def vec(x1, y1, x2, y2, cls):
        return A(p, x1, y1, x2, y2, f"ln {cls}")

    b = []
    a1, a2 = math.radians(30), math.radians(120)
    # 1: единичная окружность и v1, v2
    b.append(C(cxs[0], cy, r, "c6"))
    b.append(vec(40, cy, 40 + r * math.cos(a1), cy - r * math.sin(a1), "s3"))
    b.append(vec(40, cy, 40 + r * math.cos(a2), cy - r * math.sin(a2), "s5"))
    b.append(T(70, 62, "v₁", "t3", 11))
    b.append(T(18, 55, "v₂", "t5", 11))
    # 2: после Vᵀ — v1→e1, v2→e2
    b.append(C(cxs[1], cy, r, "c6"))
    b.append(vec(130, cy, 130 + r, cy, "s3"))
    b.append(vec(130, cy, 130, cy - r, "s5"))
    # 3: после Σ — растяжение по осям
    b.append(f'<ellipse class="c6" cx="225" cy="{cy}" rx="{rx}" ry="{ry}"/>')
    b.append(vec(225, cy, 225 + rx, cy, "s3"))
    b.append(vec(225, cy, 225, cy - ry, "s5"))
    b.append(T(240, 60, "σ₁", "t3", 11))
    # 4: после U — поворот на 45°
    phi = math.radians(45)
    b.append(f'<ellipse class="c1" cx="325" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate(-45 325 {cy})"/>')
    b.append(vec(325, cy, 325 + rx * math.cos(phi), cy - rx * math.sin(phi), "s3"))
    b.append(vec(325, cy, 325 + ry * math.cos(phi + math.pi / 2), cy - ry * math.sin(phi + math.pi / 2), "s5"))
    b.append(T(346, 38, "σ₁u₁", "t3", 11))
    # переходы
    for x1, x2, s in [(70, 100, "Vᵀ"), (162, 186, "Σ"), (265, 293, "U")]:
        b.append(A(p, x1, 120, x2, 120))
        b.append(T((x1 + x2) / 2, 114, s, bold=True))
    for x, s in zip(cxs, ["‖x‖ = 1", "поворот", "растяжение", "поворот"]):
        b.append(T(x, 142, s, "mut", 11))
    b.append(T(190, 168, "A = U Σ Vᵀ,   A v₁ = σ₁ u₁", fs=12))
    return _svg(p, 380, 180, "".join(b), "Геометрия SVD: поворот, растяжение, поворот")


def _linalg_chart(fig):
    H = np.array([1.0, 25.0])
    grad = lambda w: H * w
    f = lambda X, Y: 0.5 * (H[0] * X ** 2 + H[1] * Y ** 2)
    w0 = np.array([-4.5, 1.5])

    def run(lr, beta, n=40):
        w, v, path = w0.copy(), np.zeros(2), [w0.copy()]
        for _ in range(n):
            v = beta * v + grad(w)
            w = w - lr * v
            path.append(w.copy())
        return np.array(path)

    gd = run(0.075, 0.0)
    mom = run(0.02, 0.85)
    ax = fig.add_subplot(111)
    xs, ys = np.meshgrid(np.linspace(-5, 1.5, 200), np.linspace(-2, 2, 200))
    ax.contour(xs, ys, f(xs, ys), levels=np.geomspace(0.05, 40, 12), colors=GREY, linewidths=0.6)
    ax.plot(gd[:, 0], gd[:, 1], "o-", ms=2.5, color=RED, lw=1.2, label="GD, η = 0.075")
    ax.plot(mom[:, 0], mom[:, 1], "o-", ms=2.5, color=BLUE, lw=1.2, label="Momentum, β = 0.85")
    ax.scatter([0], [0], marker="*", s=90, color=GREEN, zorder=5, label="минимум")
    ax.set_xlabel("w₁"), ax.set_ylabel("w₂")
    ax.set_title("40 шагов на вытянутой квадратичной функции", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(False)
    fig.tight_layout()


# ================================================================ 6. Линейный автоэнкодер
def _linae_svg():
    p = "lae"
    xs_in, xz, xs_out = 50, 180, 310
    yin = [55, 85, 115, 145, 175]
    yz = [100, 130]
    b = []
    for yi in yin:
        for yj in yz:
            b.append(L(xs_in + 9, yi, xz - 9, yj, "ln s6"))
            b.append(L(xz + 9, yj, xs_out - 9, yi, "ln s6"))
    for yi in yin:
        b.append(C(xs_in, yi, 9, "c1"))
        b.append(C(xs_out, yi, 9, "c2"))
    for yj in yz:
        b.append(C(xz, yj, 9, "c3"))
    b += [T(50, 30, "X (n)", bold=True), T(180, 30, "Z (k)", bold=True), T(310, 30, "X′ (n)", bold=True),
          T(180, 160, "латентное", "mut", 11), T(180, 174, "k &lt; n", "mut", 11),
          T(112, 205, "энкодер W_e", "t1", 11), T(248, 205, "декодер W_d", "t2", 11),
          T(180, 230, "MSE = mean((X − X′)²)", fs=12),
          T(180, 248, "оптимум: подпространство PCA", "mut", 11)]
    return _svg(p, 360, 258, "".join(b), "Линейный автоэнкодер: X → Z → X′")


def _linae_chart(fig):
    rng = np.random.default_rng(3)
    N, n, k = 400, 6, 2
    Zt = rng.normal(size=(N, 3)) * np.array([2.0, 1.3, 0.6])
    X = Zt @ rng.normal(size=(3, n)) + 0.3 * rng.normal(size=(N, n))
    X = (X - X.mean(0)) / X.std(0)
    from sklearn.decomposition import PCA
    pca = PCA(k).fit(X)
    pca_mse = np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2)
    We = rng.normal(0, 0.1, (n, k))
    Wd = rng.normal(0, 0.1, (k, n))
    lr, hist = 0.05, []
    for _ in range(600):
        Hh = X @ We
        R = Hh @ Wd - X
        hist.append(np.mean(R ** 2))
        g = 2 / (N * n)
        dWd = g * Hh.T @ R
        dWe = g * X.T @ R @ Wd.T
        We -= lr * n * dWe
        Wd -= lr * n * dWd
    ax = fig.add_subplot(111)
    ax.plot(hist, color=BLUE, lw=2, label="линейный AE (градиентный спуск)")
    ax.axhline(pca_mse, color=RED, ls="--", lw=1.5, label=f"PCA, k = {k}: {pca_mse:.3f}")
    ax.set_xlabel("эпоха"), ax.set_ylabel("MSE реконструкции")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    fig.tight_layout()


# ================================================================ 7. Ансамбли линейных моделей
def _linens_svg():
    p = "len"
    b = [B(130, 10, 100, 28, "c6", ["X"]),
         B(10, 72, 100, 30, "c1", ["Ridge (L2)"]), B(130, 72, 100, 30, "c3", ["Lasso (L1)"]),
         B(250, 72, 100, 30, "c5", ["ElasticNet"]),
         A(p, 160, 38, 70, 70), A(p, 180, 38, 180, 70), A(p, 200, 38, 290, 70),
         B(65, 140, 230, 40, "c6", ["OOF-прогнозы [ŷ₁, ŷ₂, ŷ₃]", "cv = 5: без утечки"]),
         A(p, 60, 102, 120, 138), A(p, 180, 102, 180, 138), A(p, 300, 102, 240, 138),
         B(90, 210, 180, 30, "c2", ["final_estimator: Ridge"]),
         A(p, 180, 180, 180, 208), A(p, 180, 240, 180, 262),
         T(180, 278, "ŷ", bold=True)]
    return _svg(p, 360, 288, "".join(b), "Стекинг линейных моделей")


# ================================================================ 8. Логический вывод
def _logic_svg():
    p = "lil"
    b = [B(120, 10, 120, 28, "c6", ["x (данные)"]),
         A(p, 180, 38, 180, 53),
         B(70, 55, 220, 28, "c1", ["нейросеть: предикаты P(x)"]),
         A(p, 180, 83, 180, 98),
         B(90, 100, 180, 28, "c2", ["истинность p ∈ [0, 1]"]),
         A(p, 180, 128, 180, 143),
         B(60, 145, 240, 50, "c5", ["дифференцируемая логика", "AND = a·b,  OR = a+b−ab", "NOT = 1 − a"],
           sub_fs=11),
         A(p, 180, 195, 180, 213),
         B(80, 215, 200, 30, "c4", ["L = L_data + λ·L_logic"]),
         PA(p, "M80,230 H30 V69 H68", "ln s4"),
         T(16, 154, "∇L", "t4", 12)]
    return _svg(p, 360, 256, "".join(b), "Нейро-символьное обучение с дифференцируемой логикой")


def _logic_chart(fig):
    a = np.linspace(0, 1, 200)
    bv = 0.7
    ax = fig.add_subplot(111)
    ax.plot(a, a * bv, color=BLUE, lw=2, label="Product: a·b")
    ax.plot(a, np.minimum(a, bv), color=GREEN, lw=2, label="Minimum: min(a, b)")
    ax.plot(a, np.maximum(0, a + bv - 1), color=ORANGE, lw=2, label="Łukasiewicz: max(0, a+b−1)")
    ax.set_xlabel("истинность a"), ax.set_ylabel("AND(a, b)")
    ax.set_title("Нечёткий AND при b = 0.7", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()


# ================================================================ 9. LSTM
def _lstm_svg():
    p = "lstm"
    b = [T(8, 44, "Cₜ₋₁", anchor="start"), A(p, 40, 40, 342, 40), T(348, 44, "Cₜ", anchor="start"),
         T(8, 214, "hₜ₋₁, xₜ", anchor="start"), L(62, 210, 323, 210),
         B(80, 150, 56, 30, "c4", ["σ  fₜ"]), B(150, 150, 56, 30, "c1", ["σ  iₜ"]),
         B(220, 150, 60, 30, "c2", ["tanh C̃ₜ"]), B(295, 150, 56, 30, "c3", ["σ  oₜ"])]
    for x in (108, 178, 250, 323):
        b.append(A(p, x, 210, x, 182))
    b += [C(110, 40, 11, "c4", "×"), C(200, 40, 11, "c2", "+"),
          A(p, 108, 150, 110, 53),
          C(215, 100, 10, "c1", "×"),
          A(p, 178, 150, 208, 108), A(p, 250, 150, 222, 108), A(p, 213, 90, 203, 53),
          L(290, 40, 290, 70), B(272, 70, 36, 22, "c6", ["tanh"], fs=11),
          C(323, 112, 10, "c3", "×"),
          A(p, 304, 92, 316, 104), A(p, 323, 150, 323, 124),
          A(p, 333, 112, 358, 112), T(362, 116, "hₜ", anchor="start"),
          T(122, 76, "забыть", "t4", 11, anchor="start"), T(140, 120, "запомнить", "t1", 11),
          T(190, 236, "Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ,   hₜ = oₜ⊙tanh(Cₜ)", fs=12)]
    return _svg(p, 380, 248, "".join(b), "Ячейка LSTM")


# ================================================================ 10. Malware detection
def _malware_svg():
    p = "mal"
    b = [B(110, 10, 140, 28, "c6", ["файл (PE / APK)"]),
         A(p, 150, 38, 95, 63), A(p, 210, 38, 265, 63),
         B(10, 65, 165, 58, "c1", ["статический анализ", "PE, strings, opcodes", "entropy, n-grams"]),
         B(185, 65, 165, 58, "c3", ["динамический анализ", "sandbox: API calls", "сеть, файлы, реестр"]),
         A(p, 92, 123, 150, 143), A(p, 268, 123, 210, 143),
         B(90, 145, 180, 28, "c6", ["вектор признаков"]),
         A(p, 180, 173, 180, 193),
         B(70, 195, 220, 30, "c5", ["ML: GBM, RF, CNN"]),
         A(p, 140, 225, 110, 248), A(p, 220, 225, 250, 248),
         B(60, 250, 100, 28, "c4", ["malware"]), B(200, 250, 100, 28, "c2", ["benign"]),
         T(180, 300, "obfuscation и packing искажают признаки", "mut", 11)]
    return _svg(p, 360, 310, "".join(b), "Пайплайн ML-детектора malware")


# ================================================================ 11. Material design
def _matdesign_svg():
    p = "mdz"
    b = [B(120, 15, 120, 40, "c1", ["Design", "ML предлагает"]),
         B(232, 90, 120, 40, "c3", ["Make", "синтез (робот)"]),
         B(120, 165, 120, 40, "c2", ["Test", "измерение свойств"]),
         B(8, 90, 120, 40, "c5", ["Analyze", "обновить модель"]),
         A(p, 240, 42, 286, 88), A(p, 286, 132, 240, 172), A(p, 120, 178, 74, 132), A(p, 74, 88, 120, 42),
         T(180, 106, "closed loop", bold=True), T(180, 122, "BO + AL", "mut", 11)]
    return _svg(p, 360, 215, "".join(b), "Замкнутый цикл design–make–test–analyze")


def _matdesign_chart(fig):
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    f = lambda x: np.sin(3 * x) * 0.8 + 0.5 * np.cos(7 * x) * 0.5 + 0.3 * x
    Xo = np.array([0.1, 0.45, 0.9, 1.6, 1.85])
    yo = f(Xo)
    gp = GaussianProcessRegressor(ConstantKernel(1.0, "fixed") * RBF(0.3, "fixed"), alpha=1e-6).fit(Xo[:, None], yo)
    xs = np.linspace(0, 2, 400)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    best = yo.max()
    zz = (mu - best - 0.01) / np.maximum(sd, 1e-9)
    ei = (mu - best - 0.01) * norm.cdf(zz) + sd * norm.pdf(zz)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(xs, f(xs), color=GREY, ls="--", lw=1.2, label="истинное свойство")
    ax.plot(xs, mu, color=BLUE, lw=2, label="GP: среднее")
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.15, label="±2σ")
    ax.scatter(Xo, yo, color="black", s=20, zorder=5, label="измерено")
    ax.axvline(xs[ei.argmax()], color=RED, lw=1.2)
    ax.set_title("Surrogate-модель по 5 экспериментам", fontsize=10)
    ax.set_ylim(-2.6, 1.7)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    ax.set_xticklabels([])
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(xs, ei, color=GREEN, lw=2)
    ax2.fill_between(xs, 0, ei, color=GREEN, alpha=0.2)
    ax2.axvline(xs[ei.argmax()], color=RED, lw=1.2, label="следующий кандидат")
    ax2.set_title("Expected Improvement", fontsize=10)
    ax2.set_xlabel("параметр состава / структуры")
    ax2.legend(fontsize=7)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 12. Material informatics
def _matinf_svg():
    p = "mif"
    b = [B(65, 10, 230, 30, "c6", ["базы: Materials Project, OQMD"]),
         A(p, 180, 40, 180, 58),
         B(55, 60, 250, 42, "c5", ["представление", "состав, структура, граф атомов"]),
         A(p, 180, 102, 180, 120),
         B(80, 122, 200, 30, "c1", ["ML: RF, GBM, GNN"]),
         A(p, 180, 152, 180, 170),
         B(80, 172, 200, 30, "c2", ["прогноз свойства ± σ"]),
         A(p, 180, 202, 180, 220),
         B(70, 222, 220, 30, "c3", ["отбор → DFT / синтез"]),
         PA(p, "M290,237 H335 V25 H297", "ln s4"),
         T(350, 131, "новые данные", "t4", 11, extra=' transform="rotate(-90 350 131)"')]
    return _svg(p, 360, 262, "".join(b), "Рабочий процесс materials informatics")


# ================================================================ 13. Свойства материалов
def _matprop_svg():
    p = "mpr"
    b = [B(90, 10, 180, 28, "c6", ["состав + структура"]),
         A(p, 180, 38, 180, 56),
         B(40, 58, 280, 42, "c5", ["дескрипторы", "электроотрицательность, радиусы, V ячейки"], sub_fs=11),
         A(p, 180, 100, 180, 118),
         B(70, 120, 220, 30, "c1", ["модель (GP, RF-ансамбль)"]),
         A(p, 120, 150, 60, 180), A(p, 180, 150, 180, 180), A(p, 240, 150, 300, 180),
         B(10, 182, 100, 42, "c3", ["E, K", "механические"]),
         B(130, 182, 100, 42, "c2", ["band gap", "электронные"]),
         B(250, 182, 100, 42, "c4", ["κ", "термические"]),
         T(180, 250, "у каждого прогноза — неопределённость σ", "mut", 11)]
    return _svg(p, 360, 260, "".join(b), "Structure–property: от дескрипторов к свойствам")


def _matprop_chart(fig):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    rng = np.random.default_rng(4)
    f = lambda x: 1.5 + np.sin(x) + 0.25 * x
    Xtr = rng.uniform(0, 6, 35)
    ytr = f(Xtr) + rng.normal(0, 0.12, Xtr.size)
    gp = GaussianProcessRegressor(RBF(1.0) + WhiteKernel(0.01), normalize_y=True, random_state=0)
    gp.fit(Xtr[:, None], ytr)
    xs = np.linspace(-2, 10, 400)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    ax = fig.add_subplot(111)
    ax.axvspan(0, 6, color=GREEN, alpha=0.08, label="область обучения")
    ax.plot(xs, f(xs), color=GREY, ls="--", lw=1.2, label="истинная зависимость")
    ax.plot(xs, mu, color=BLUE, lw=2, label="GP-прогноз")
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.15, label="±2σ")
    ax.scatter(Xtr, ytr, s=10, color="black", zorder=5)
    ax.set_xlabel("дескриптор"), ax.set_ylabel("свойство")
    ax.set_title("Интерполяция vs экстраполяция", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()


# ================================================================ 14. Matplotlib / Seaborn
def _mpl_svg():
    p = "mpl"
    b = [B(10, 10, 340, 230, "c6", [], rx=4),
         T(20, 28, "Figure: fig", anchor="start", bold=True),
         T(170, 44, "ax.set_title()", fs=11),
         f'<rect class="c1" x="70" y="52" width="200" height="140" rx="2"/>',
         L(70, 192, 270, 192), L(70, 52, 70, 192),
         '<polyline class="ln s3" points="80,170 110,150 140,155 170,120 200,110 230,85 260,70"/>',
         '<polyline class="ln s2" points="80,180 110,175 140,165 170,160 200,150 230,140 260,135" stroke-dasharray="5 3"/>',
         B(186, 58, 78, 22, "c6", ["ax.legend()"], fs=11, rx=3)]
    for x in (100, 150, 200, 250):
        b.append(L(x, 192, x, 197))
    for y in (80, 120, 160):
        b.append(L(65, y, 70, y))
    b += [T(170, 212, "ticks", "mut", 11), T(170, 230, "ax.set_xlabel()", fs=11),
          T(44, 122, "ax.set_ylabel()", fs=11, extra=' transform="rotate(-90 44 122)"'),
          T(310, 118, "Axes: ax", "t1", 12, bold=True), T(310, 134, "subplots()", "mut", 11)]
    return _svg(p, 360, 250, "".join(b), "Анатомия Figure и Axes")


def _mpl_chart(fig):
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 10, 200)
    y = 2 + 0.8 * x + 0.12 * (x - 5) ** 2 + rng.normal(0, 0.6, x.size)
    from sklearn.linear_model import LinearRegression
    m = LinearRegression().fit(x[:, None], y)
    pred = m.predict(x[:, None])
    res = y - pred
    order = np.argsort(pred)
    ax = fig.add_subplot(111)
    ax.scatter(pred, res, s=10, color=BLUE, alpha=0.6)
    ax.axhline(0, color=RED, lw=1.5, ls="--")
    k = 25
    smooth = np.convolve(res[order], np.ones(k) / k, mode="valid")
    ax.plot(pred[order][k // 2:k // 2 + smooth.size], smooth, color=ORANGE, lw=2, label="скользящее среднее остатков")
    ax.set_xlabel("предсказание ŷ"), ax.set_ylabel("остаток y − ŷ")
    ax.set_title("Residuals plot: U-образный паттерн", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()


# ================================================================ 15. Матричная факторизация
def _mf_svg():
    p = "mfz"
    cs = 18
    vals = [["5", "3", "", "1", "", "4"],
            ["", "4", "", "?", "2", ""],
            ["1", "", "5", "", "4", ""],
            ["", "2", "4", "", "", "5"],
            ["4", "", "", "2", "1", ""]]
    hr, hc = 1, 3
    b = []
    x0, y0 = 12, 40
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            cls = "c3" if i == hr else ("c5" if j == hc else "c6")
            b.append(f'<rect class="{cls}" x="{x0 + j * cs}" y="{y0 + i * cs}" width="{cs}" height="{cs}"/>')
            if v == "?":
                b.append(T(x0 + j * cs + 9, y0 + i * cs + 13, "?", "t4", 12, bold=True))
            elif v:
                b.append(T(x0 + j * cs + 9, y0 + i * cs + 13, v, fs=11))
    b.append(T(66, 30, "R", bold=True))
    b.append(T(66, 148, "m × n, пропуски", "mut", 11))
    b.append(T(133, 88, "≈", fs=18))
    ux = 146
    for i in range(5):
        for j in range(2):
            cls = "c3" if i == hr else "c1"
            b.append(f'<rect class="{cls}" x="{ux + j * cs}" y="{y0 + i * cs}" width="{cs}" height="{cs}"/>')
    b.append(T(ux + cs, 30, "U", bold=True))
    b.append(T(ux + cs, 148, "m × k", "mut", 11))
    vx = 196
    for i in range(2):
        for j in range(6):
            cls = "c5" if j == hc else "c2"
            b.append(f'<rect class="{cls}" x="{vx + j * cs}" y="{y0 + i * cs}" width="{cs}" height="{cs}"/>')
    b.append(T(vx + 54, 30, "Vᵀ", bold=True))
    b.append(T(vx + 54, 94, "k × n", "mut", 11))
    b.append(T(170, 178, "r̂ᵤᵢ = uᵤᵀ vᵢ  (+ μ + bᵤ + bᵢ)", fs=12))
    b.append(T(170, 198, "min Σ (rᵤᵢ − uᵤᵀvᵢ)² + λ(‖uᵤ‖² + ‖vᵢ‖²)", "mut", 11))
    b.append(T(170, 213, "сумма только по известным рейтингам", "mut", 11))
    return _svg(p, 340, 224, "".join(b), "Матричная факторизация R ≈ U Vᵀ")


def _mf_chart(fig):
    rng = np.random.default_rng(6)
    m, n, k_true = 120, 90, 3
    R = rng.normal(size=(m, k_true)) @ rng.normal(size=(k_true, n)) + rng.normal(0, 0.5, (m, n))
    mask = rng.random((m, n)) < 0.25
    test = (~mask) & (rng.random((m, n)) < 0.15)

    def als(k, lam=0.1, iters=15):
        r = np.random.default_rng(0)
        U, V = r.normal(0, 0.3, (m, k)), r.normal(0, 0.3, (n, k))
        I = lam * np.eye(k)
        for _ in range(iters):
            for u in range(m):
                Vi = V[mask[u]]
                U[u] = np.linalg.solve(Vi.T @ Vi + I, Vi.T @ R[u, mask[u]])
            for i in range(n):
                Uu = U[mask[:, i]]
                V[i] = np.linalg.solve(Uu.T @ Uu + I, Uu.T @ R[mask[:, i], i])
        P = U @ V.T
        rm = lambda M: np.sqrt(np.mean((P[M] - R[M]) ** 2))
        return rm(mask), rm(test)

    ks = [1, 2, 3, 4, 6, 8, 12]
    tr, te = zip(*[als(k) for k in ks])
    ax = fig.add_subplot(111)
    ax.plot(ks, tr, "o-", color=BLUE, lw=2, label="train RMSE")
    ax.plot(ks, te, "o-", color=RED, lw=2, label="test RMSE")
    ax.axvline(k_true, color=GREY, ls="--", lw=1)
    ax.text(k_true + 0.2, max(te) * 0.95, "истинный ранг", fontsize=8, color=GREY)
    ax.set_xlabel("число факторов k"), ax.set_ylabel("RMSE")
    ax.set_title("ALS, λ = 0.1, известно 25% матрицы", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()


# ================================================================ 16. MCMC
def _mcmc_svg():
    p = "mcmc"
    b = [B(140, 10, 80, 28, "c6", ["θₜ"]),
         A(p, 180, 38, 180, 53),
         B(95, 55, 170, 28, "c1", ["θ* ~ q(θ* | θₜ)"]),
         A(p, 180, 83, 180, 98),
         B(55, 100, 250, 42, "c3", ["α = min(1, r)", "r = P(θ*)q(θₜ|θ*) / P(θₜ)q(θ*|θₜ)"]),
         A(p, 180, 142, 180, 158),
         '<polygon class="c5" points="180,160 235,182 180,204 125,182"/>',
         T(180, 186, "u &lt; α ?"), T(242, 186, "u ~ U(0, 1)", "mut", 11, anchor="start"),
         A(p, 130, 196, 85, 226), A(p, 230, 196, 275, 226),
         T(95, 205, "да", "t2", 11), T(265, 205, "нет", "t4", 11),
         B(20, 228, 130, 28, "c2", ["θₜ₊₁ = θ*"]), B(210, 228, 130, 28, "c4", ["θₜ₊₁ = θₜ"]),
         PA(p, "M20,242 H10 V24 H138"), PA(p, "M340,242 H350 V24 H222"),
         T(180, 276, "t ← t + 1;  цепь → posterior P(θ|D)", "mut", 11)]
    return _svg(p, 360, 286, "".join(b), "Шаг алгоритма Metropolis-Hastings")


def _mcmc_chart(fig):
    rng = np.random.default_rng(7)
    npdf = lambda t, m, s: np.exp(-0.5 * ((t - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    target = lambda t: 0.35 * npdf(t, -2, 0.7) + 0.65 * npdf(t, 1.5, 1.0)
    n, th, acc = 6000, -6.0, 0
    chain = np.empty(n)
    for i in range(n):
        prop = th + rng.normal(0, 1.2)
        if rng.random() < min(1, target(prop) / target(th)):
            th, acc = prop, acc + 1
        chain[i] = th
    burn = 500
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(chain[:1500], color=BLUE, lw=0.6)
    ax.axvspan(0, burn, color=RED, alpha=0.12, label="burn-in")
    ax.set_title(f"Trace plot (acceptance = {acc / n:.0%})", fontsize=10)
    ax.set_xlabel("итерация"), ax.set_ylabel("θ")
    ax.legend(fontsize=7, loc="lower right")
    ax2 = fig.add_subplot(2, 1, 2)
    ts = np.linspace(-5, 5, 300)
    ax2.hist(chain[burn:], bins=60, density=True, color=BLUE, alpha=0.45, label="сэмплы MH")
    ax2.plot(ts, target(ts), color=RED, lw=2, label="целевой posterior")
    ax2.set_xlabel("θ"), ax2.set_yticks([])
    ax2.set_title("Гистограмма цепи после burn-in", fontsize=10)
    ax2.legend(fontsize=7)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 17. Meta-features
def _meta_svg():
    p = "mft"
    b = [B(10, 12, 140, 34, "c6", ["датасеты D₁…Dₙ"]),
         A(p, 150, 29, 183, 29),
         B(185, 8, 165, 42, "c5", ["мета-признаки", "+ лучший алгоритм"]),
         A(p, 260, 50, 235, 91), T(262, 76, "обучение", "mut", 11, anchor="start"),
         B(150, 93, 140, 36, "c1", ["meta-learner (RF)"]),
         A(p, 150, 111, 132, 111),
         B(10, 90, 120, 42, "c2", ["рекомендация", "алгоритм, HP"]),
         B(10, 170, 140, 34, "c6", ["новый датасет"]),
         A(p, 150, 187, 183, 187),
         B(185, 168, 165, 38, "c5", ["мета-признаки"]),
         A(p, 260, 168, 235, 131), T(262, 152, "применение", "mut", 11, anchor="start"),
         T(180, 232, "n, d, асимметрия, энтропия, landmarkers", "mut", 11)]
    return _svg(p, 360, 244, "".join(b), "Meta-learning: выбор алгоритма по мета-признакам")


# ================================================================ 18. Siamese
def _siam_svg():
    p = "sia"
    b = [B(10, 30, 44, 28, "c6", ["x₁"]), B(10, 110, 44, 28, "c6", ["x₂"]),
         A(p, 54, 44, 72, 44), A(p, 54, 124, 72, 124),
         B(74, 27, 86, 34, "c1", ["энкодер f"]), B(74, 107, 86, 34, "c1", ["энкодер f"]),
         f'<line class="ln s5" x1="100" y1="63" x2="100" y2="105" stroke-dasharray="4 3"/>',
         T(107, 88, "общие веса", "t5", 11, anchor="start"),
         A(p, 160, 44, 182, 44), A(p, 160, 124, 182, 124),
         B(184, 30, 60, 28, "c2", ["f(x₁)"]), B(184, 110, 60, 28, "c2", ["f(x₂)"]),
         A(p, 244, 48, 268, 70), A(p, 244, 120, 268, 98),
         B(262, 64, 92, 40, "c3", ["d", "‖f(x₁) − f(x₂)‖"], sub_fs=10),
         A(p, 308, 104, 308, 168),
         B(30, 170, 310, 32, "c4", ["L = y·d² + (1 − y)·max(m − d, 0)²"]),
         T(180, 224, "y = 1 — похожая пара, y = 0 — разная", "mut", 11)]
    return _svg(p, 360, 234, "".join(b), "Siamese-сеть с contrastive loss")


def _siam_chart(fig):
    d = np.linspace(0, 2, 300)
    m = 1.0
    ax = fig.add_subplot(111)
    ax.plot(d, d ** 2, color=GREEN, lw=2, label="y = 1 (похожие): d²")
    ax.plot(d, np.maximum(m - d, 0) ** 2, color=RED, lw=2, label="y = 0 (разные): max(m − d, 0)²")
    ax.axvline(m, color=GREY, ls="--", lw=1)
    ax.text(m - 0.03, 2.2, "margin m = 1", fontsize=8, color=GREY, ha="right")
    ax.set_xlabel("расстояние d между эмбеддингами"), ax.set_ylabel("loss")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()


# ================================================================ 19. Metric learning
def _metric_svg():
    p = "mlr"
    b = [T(90, 22, "до обучения", bold=True), T(265, 22, "после", bold=True),
         L(90, 100, 150, 62, "ln s2"), L(90, 100, 118, 132, "ln s4"),
         C(90, 100, 6, "c1"), C(150, 62, 6, "c2"), C(118, 132, 6, "c4"),
         T(78, 96, "A", "t1", anchor="end", bold=True), T(150, 52, "P", "t2", bold=True),
         T(118, 152, "N", "t4", bold=True),
         A(p, 172, 100, 200, 100), T(186, 92, "", "mut", 11),
         f'<circle class="s6" cx="265" cy="100" r="56" stroke-dasharray="4 3"/>',
         L(265, 100, 290, 86, "ln s2"), L(265, 100, 334, 150, "ln s4"),
         C(265, 100, 6, "c1"), C(290, 86, 6, "c2"), C(334, 150, 6, "c4"),
         T(255, 112, "A", "t1", anchor="end", bold=True), T(292, 76, "P", "t2", bold=True),
         T(334, 170, "N", "t4", bold=True),
         T(265, 38, "d(A,P) + margin", "mut", 11),
         T(180, 196, "L = max(d(A,P) − d(A,N) + margin, 0)", fs=12)]
    return _svg(p, 360, 208, "".join(b), "Triplet loss: anchor, positive, negative")


def _metric_chart(fig):
    th = np.linspace(0, np.pi / 2, 300)
    deg = np.degrees(th)
    ax = fig.add_subplot(111)
    ax.plot(deg, np.cos(th), color=GREY, lw=2, label="Softmax: cos θ")
    ax.plot(deg, np.cos(th) - 0.35, color=ORANGE, lw=2, label="CosFace: cos θ − 0.35")
    ax.plot(deg, np.cos(th + 0.5), color=BLUE, lw=2, label="ArcFace: cos(θ + 0.5)")
    ax.set_xlabel("угол θ между эмбеддингом и центром класса, °")
    ax.set_ylabel("логит целевого класса / s")
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()


# ================================================================ 20. Визуализация метрик
def _metvis_svg():
    p = "mvz"
    x0, y0, cw, ch = 110, 50, 80, 50
    b = [T(190, 22, "Predicted label", bold=True),
         T(150, 44, "0", "mut", 11), T(230, 44, "1", "mut", 11),
         T(40, 100, "True label", bold=True, extra=' transform="rotate(-90 40 100)"'),
         T(98, 79, "0", "mut", 11, anchor="end"), T(98, 129, "1", "mut", 11, anchor="end"),
         B(x0, y0, cw, ch, "c2", ["TN"], rx=0), B(x0 + cw, y0, cw, ch, "c4", ["FP"], rx=0),
         B(x0, y0 + ch, cw, ch, "c4", ["FN"], rx=0), B(x0 + cw, y0 + ch, cw, ch, "c2", ["TP"], rx=0),
         T(180, 178, "Precision = TP / (TP + FP)"),
         T(180, 198, "Recall = TPR = TP / (TP + FN)"),
         T(180, 218, "FPR = FP / (FP + TN)"),
         T(180, 240, "ROC: TPR от FPR;  PR: Precision от Recall", "mut", 11)]
    return _svg(p, 360, 250, "".join(b), "Матрица ошибок и производные метрики")


def _metvis_chart(fig):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    rng = np.random.default_rng(8)
    n = 2000
    y = (rng.random(n) < 0.2).astype(int)
    s_good = rng.normal(1.6 * y, 1.0)
    s_weak = rng.normal(0.7 * y, 1.0)
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for s, c, name in [(s_good, BLUE, "сильная"), (s_weak, ORANGE, "слабая")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}: AUC = {auc(fpr, tpr):.2f}")
        pr, rc, _ = precision_recall_curve(y, s)
        ax2.plot(rc, pr, color=c, lw=2, label=f"{name}: AP = {average_precision_score(y, s):.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"), ax.set_ylabel("TPR"), ax.set_title("ROC-кривая", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax2.axhline(y.mean(), color="k", ls="--", lw=1, label=f"доля класса 1 = {y.mean():.2f}")
    ax2.set_xlabel("Recall"), ax2.set_ylabel("Precision"), ax2.set_title("Precision-Recall", fontsize=10)
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=7.5, loc="upper right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 21. ML pipelines
def _pipe_svg():
    p = "mlpp"
    b = [B(120, 10, 120, 28, "c6", ["X (DataFrame)"]),
         B(10, 52, 340, 112, "c6", [], rx=4, dash=True),
         T(344, 156, "ColumnTransformer", "mut", 11, anchor="end"),
         A(p, 160, 38, 100, 72), A(p, 200, 38, 260, 72),
         T(95, 86, "num: age, income", "mut", 11), T(265, 86, "cat: city, category", "mut", 11),
         B(35, 96, 120, 28, "c1", ["StandardScaler"]), B(205, 96, 120, 28, "c3", ["OneHotEncoder"]),
         T(180, 158, "hstack", "mut", 11),
         A(p, 95, 124, 150, 186), A(p, 265, 124, 210, 186),
         B(80, 188, 200, 30, "c2", ["RandomForestClassifier"]),
         A(p, 180, 218, 180, 238), T(180, 254, "ŷ", bold=True),
         T(180, 276, "fit: fit_transform только на train", "mut", 11),
         T(180, 292, "predict: transform с теми же параметрами", "mut", 11)]
    return _svg(p, 360, 302, "".join(b), "Pipeline с ColumnTransformer")


# ================================================================ 22. MLOps
def _rect_edge(cx, cy, w, h, tx, ty, pad=3):
    dx, dy = tx - cx, ty - cy
    s = min((w / 2 + pad) / abs(dx) if dx else 1e9, (h / 2 + pad) / abs(dy) if dy else 1e9)
    return cx + dx * s, cy + dy * s


def _mlops_svg():
    p = "mops"
    names = ["Data", "Preprocessing", "Features", "Training", "Evaluation", "Deployment", "Monitoring"]
    clss = ["c6", "c6", "c5", "c1", "c3", "c2", "c4"]
    cx, cy, R, RX, w, h = 180, 150, 108, 130, 96, 26
    pos = []
    for i in range(7):
        a = -math.pi / 2 + 2 * math.pi * i / 7
        pos.append((cx + RX * math.cos(a), cy + R * math.sin(a)))
    b = []
    for i in range(7):
        (x1, y1), (x2, y2) = pos[i], pos[(i + 1) % 7]
        sx, sy = _rect_edge(x1, y1, w, h, x2, y2)
        ex, ey = _rect_edge(x2, y2, w, h, x1, y1)
        b.append(A(p, sx, sy, ex, ey, "ln s4" if i == 6 else "ln"))
    for (x, y), nm, cl in zip(pos, names, clss):
        b.append(B(x - w / 2, y - h / 2, w, h, cl, [nm]))
    b += [T(cx, cy - 8, "CI/CD", bold=True), T(cx, cy + 8, "Git · DVC · MLflow", "mut", 11),
          T(cx, cy + 24, "versioning", "mut", 11)]
    lx, ly = (pos[6][0] + pos[0][0]) / 2, (pos[6][1] + pos[0][1]) / 2
    b.append(T(lx - 22, ly - 8, "retrain", "t4", 11, anchor="end"))
    return _svg(p, 360, 290, "".join(b), "Жизненный цикл ML-системы в MLOps")


# ================================================================ 23. MLP
def _mlp_svg():
    p = "mlpz"
    cols = [(50, [67, 102, 137]), (140, [50, 85, 120, 155]), (230, [50, 85, 120, 155]), (320, [85, 120])]
    b = []
    for (xa, ya), (xb, yb) in zip(cols, cols[1:]):
        for y1 in ya:
            for y2 in yb:
                b.append(L(xa + 10, y1, xb - 10, y2, "ln s6"))
    for (x, ys), cl in zip(cols, ["c6", "c1", "c1", "c2"]):
        for y in ys:
            b.append(C(x, y, 10, cl))
    for (x, _), s in zip(cols, ["Input", "Hidden 1", "Hidden 2", "Output"]):
        b.append(T(x, 26, s, bold=True))
    for (x, _), s in zip(cols, ["d признаков", "ReLU", "ReLU", "softmax"]):
        b.append(T(x, 185, s, "mut", 11))
    b.append(T(180, 212, "hₗ = ReLU(Wₗ hₗ₋₁ + bₗ)", fs=12))
    b.append(T(180, 230, "каждый нейрон связан со всеми следующего слоя", "mut", 11))
    return _svg(p, 360, 240, "".join(b), "Полносвязная сеть MLP")


def _mlp_chart(fig):
    from sklearn.datasets import make_moons
    from sklearn.neural_network import MLPClassifier
    X, y = make_moons(300, noise=0.25, random_state=0)
    clf = MLPClassifier((16, 16), max_iter=1500, random_state=0).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 200), np.linspace(-1.2, 1.7, 200))
    pr = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, pr, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    ax.contour(xx, yy, pr, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap([BLUE, RED]), s=10, edgecolors="none")
    ax.set_title(f"MLP (16, 16), ReLU: accuracy = {clf.score(X, y):.2f}", fontsize=10)
    ax.set_xlabel("x₁"), ax.set_ylabel("x₂")
    ax.grid(False)
    fig.tight_layout()


# ================================================================ 24. MLP для табличных данных
def _mlptab_svg():
    p = "mlpt"
    b = [B(15, 10, 155, 40, "c1", ["числовые", "StandardScaler"]),
         B(190, 10, 155, 40, "c3", ["категориальные", "Embedding(n_cat, d)"]),
         A(p, 92, 50, 150, 78), A(p, 268, 50, 210, 78),
         B(110, 80, 140, 28, "c6", ["concat"]),
         A(p, 180, 108, 180, 124),
         B(25, 126, 310, 76, "c6", [], rx=4, dash=True),
         T(180, 146, "блок × N,  N = 2–5", "mut", 11),
         B(38, 160, 62, 28, "c1", ["Linear"]), B(118, 160, 44, 28, "c5", ["BN"]),
         B(180, 160, 54, 28, "c2", ["ReLU"]), B(252, 160, 72, 28, "c4", ["Dropout"]),
         A(p, 100, 174, 116, 174), A(p, 162, 174, 178, 174), A(p, 234, 174, 250, 174),
         A(p, 180, 202, 180, 222),
         B(80, 224, 200, 30, "c2", ["softmax / linear"]),
         T(180, 274, "классификация / регрессия", "mut", 11)]
    return _svg(p, 360, 284, "".join(b), "Архитектура MLP для табличных данных")


# ================================================================ 25. Model deployment
def _deploy_svg():
    p = "mdep"
    b = [T(10, 20, "Real-time (REST API)", "t1", anchor="start", bold=True),
         B(10, 34, 70, 40, "c6", ["клиент"]), A(p, 80, 54, 98, 54),
         B(100, 34, 54, 40, "c6", ["LB"]), A(p, 154, 54, 172, 54),
         f'<rect class="c1" x="184" y="26" width="110" height="40" rx="6"/>',
         f'<rect class="c1" x="179" y="30" width="110" height="40" rx="6"/>',
         B(174, 34, 110, 40, "c1", ["Docker: API", "model.pkl"]),
         T(326, 50, "×N подов", "mut", 11), T(326, 64, "(k8s)", "mut", 11),
         PA(p, "M229,74 V94 H45 V76"), T(137, 108, "JSON-ответ, мс", "mut", 11),
         T(10, 140, "Batch", "t2", anchor="start", bold=True),
         B(10, 150, 95, 40, "c6", ["данные", "S3 / DB"]), A(p, 105, 170, 128, 170),
         B(130, 150, 110, 40, "c2", ["batch job", "cron / Airflow"]), A(p, 240, 170, 263, 170),
         B(265, 150, 85, 40, "c6", ["прогнозы", "в БД"]),
         T(10, 220, "Streaming: Kafka → модель → поток", "mut", 11, anchor="start"),
         T(10, 238, "Edge: TFLite / ONNX на устройстве", "mut", 11, anchor="start")]
    return _svg(p, 360, 248, "".join(b), "Типы развёртывания модели")


# ================================================================ 26. Мониторинг + A/B
def _ab_svg():
    p = "mab"
    X0, W = 85, 260
    b = [T(45, 56, "Shadow", bold=True), T(45, 121, "Canary", bold=True), T(45, 176, "A/B", bold=True),
         A(p, 45, 64, 45, 104), A(p, 45, 128, 45, 160),
         B(X0, 40, W, 24, "c6", ["старая модель: 100%"], fs=11, rx=3),
         B(X0, 68, W, 18, "s3", ["новая: копия трафика, ответ скрыт"], fs=11, rx=3, dash=True),
         B(X0, 105, W * 0.95, 24, "c6", ["старая: 95%"], fs=11, rx=3),
         f'<rect class="c3" x="{X0 + W * 0.95:g}" y="105" width="{W * 0.05:g}" height="24" rx="2"/>',
         T(X0 + W, 99, "новая: 1–5% ↓", "t3", 11, anchor="end"),
         B(X0, 160, W / 2, 24, "c6", ["A (старая): 50%"], fs=11, rx=3),
         B(X0 + W / 2, 160, W / 2, 24, "c3", ["B (новая): 50%"], fs=11, rx=3),
         A(p, 215, 186, 215, 204),
         B(X0, 206, W, 28, "c2", ["победитель → новый baseline"]),
         T(180, 256, "мониторинг ловит дрейф → переобучение → новый цикл", "mut", 11)]
    return _svg(p, 360, 266, "".join(b), "Безопасный выкат модели: shadow, canary, A/B")


def _ab_chart(fig):
    from scipy.stats import norm
    z = norm.ppf(0.975) + norm.ppf(0.8)
    rel = np.linspace(0.02, 0.3, 200)
    ax = fig.add_subplot(111)
    for p1, c in [(0.02, RED), (0.05, BLUE), (0.15, GREEN)]:
        p2 = p1 * (1 + rel)
        n = z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
        ax.plot(rel * 100, n, color=c, lw=2, label=f"базовая конверсия {p1:.0%}")
    ax.set_yscale("log")
    ax.scatter([10], [31232], color=BLUE, zorder=5, s=25)
    ax.annotate("5% → 5.5%: ≈ 31 тыс.", (10, 31232), (12, 1.2e5), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.set_xlabel("относительный MDE, %"), ax.set_ylabel("размер выборки на группу")
    ax.set_title("α = 0.05, мощность 0.8", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()


# ================================================================ 27. Model monitoring
def _mon_svg():
    p = "mmon"
    b = [B(10, 15, 80, 30, "c6", ["запросы"]), A(p, 90, 30, 118, 30),
         B(120, 15, 100, 30, "c1", ["модель"]), A(p, 220, 30, 248, 30),
         B(250, 15, 100, 30, "c6", ["лог прогнозов"]),
         L(300, 45, 300, 62), L(62, 62, 300, 62),
         A(p, 62, 62, 62, 83), A(p, 180, 62, 180, 83), A(p, 300, 62, 300, 83),
         B(10, 85, 105, 44, "c3", ["data drift", "PSI, KS"]),
         B(127, 85, 106, 44, "c5", ["performance", "AUC, MAE"]),
         B(245, 85, 105, 44, "c6", ["infra", "latency, CPU"]),
         L(62, 129, 62, 145), L(180, 129, 180, 145), L(300, 129, 300, 145), L(62, 145, 300, 145),
         A(p, 180, 145, 180, 163),
         B(60, 165, 240, 30, "c1", ["Prometheus + Grafana: пороги"]),
         A(p, 150, 195, 105, 218),
         B(40, 220, 120, 30, "c4", ["alert"]), A(p, 160, 235, 196, 235),
         B(198, 220, 130, 30, "c2", ["переобучение"]),
         T(180, 272, "метрики качества — с задержкой меток", "mut", 11)]
    return _svg(p, 360, 282, "".join(b), "Схема мониторинга модели в production")


def _mon_chart(fig):
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(9)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0.4, 1.2, 2000)
    D, pv = ks_2samp(ref, cur)
    xs = np.linspace(-4, 5, 600)
    F1 = np.searchsorted(np.sort(ref), xs, side="right") / ref.size
    F2 = np.searchsorted(np.sort(cur), xs, side="right") / cur.size
    i = np.argmax(np.abs(F1 - F2))
    ax = fig.add_subplot(111)
    ax.plot(xs, F1, color=BLUE, lw=2, label="reference (train)")
    ax.plot(xs, F2, color=ORANGE, lw=2, label="current (production)")
    ax.vlines(xs[i], min(F1[i], F2[i]), max(F1[i], F2[i]), color=RED, lw=2.5, label=f"D = {D:.2f}")
    ax.set_xlabel("значение признака"), ax.set_ylabel("ECDF")
    ax.set_title("KS-тест на дрейф: D = max |F₁ − F₂|", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()


# ================================================================ 28. Model registries
def _reg_svg():
    p = "mreg"
    stages = [("None", "c6", "только зарегистрирована", "v4"), ("Staging", "c3", "тесты перед выкатом", "v3"),
              ("Production", "c2", "обслуживает трафик", "v2"), ("Archived", "c5", "устаревшие версии", "v1")]
    b = []
    for i, (nm, cl, desc, v) in enumerate(stages):
        y = 15 + 60 * i
        b.append(B(30, y, 115, 34, cl, [nm]))
        b.append(T(158, y + 21, desc, "mut", 11, anchor="start"))
        b.append(B(306, y + 5, 44, 24, cl, [v], fs=11, rx=12))
        if i < 3:
            b.append(A(p, 87, y + 34, 87, y + 58))
    b.append(PA(p, "M30,212 H18 V152 H28", "ln s4"))
    b.append(T(12, 182, "rollback", "t4", 11, extra=' transform="rotate(-90 12 182)"'))
    b.append(T(180, 258, "transition_model_version_stage()", "mut", 11))
    return _svg(p, 360, 268, "".join(b), "Стадии версии модели в Model Registry")


# ================================================================ 29. Model serialization
def _ser_svg():
    p = "mser"
    items = [("c3", "pickle / joblib (.pkl)", "sklearn; только Python"),
             ("c1", "torch.save (.pt)", "state_dict весов"),
             ("c1", "Keras (.keras)", "SavedModel → TF Serving"),
             ("c2", "XGBoost / LightGBM", ".json / .txt — нативно"),
             ("c5", "ONNX (.onnx)", "ONNX Runtime: C++, Java, JS")]
    b = [B(10, 102, 95, 44, "c6", ["обученная", "модель"])]
    for i, (cl, a, s) in enumerate(items):
        y = 10 + 46 * i
        b.append(B(140, y, 210, 38, cl, [a, s]))
        b.append(A(p, 105, 124, 138, y + 19))
    b.append(T(180, 258, "+ метаданные: версии библиотек, метрики", "mut", 11))
    return _svg(p, 360, 268, "".join(b), "Форматы сериализации моделей")


# ================================================================ 30. Современные тренды DL (RAG)
def _rag_svg():
    p = "mtdl"
    b = [T(10, 12, "индексация (offline)", "mut", 11, anchor="start"),
         B(10, 20, 85, 30, "c6", ["документы"]), A(p, 95, 35, 118, 35),
         B(120, 20, 100, 30, "c6", ["эмбеддинги"]), A(p, 220, 35, 243, 35),
         B(245, 20, 105, 30, "c5", ["векторная БД"]),
         T(10, 92, "запрос (online)", "mut", 11, anchor="start"),
         B(10, 100, 85, 30, "c6", ["запрос"]), A(p, 95, 115, 118, 115),
         B(120, 100, 100, 30, "c6", ["эмбеддинг"]), A(p, 220, 115, 243, 115),
         B(245, 100, 105, 30, "c3", ["поиск top-k"]),
         A(p, 297, 50, 297, 98),
         A(p, 297, 130, 250, 168), A(p, 52, 130, 110, 168, dash=True),
         B(50, 170, 260, 30, "c6", ["промпт = запрос + контекст"]),
         A(p, 180, 200, 180, 220),
         B(105, 222, 150, 30, "c1", ["LLM (генерация)"]),
         A(p, 180, 252, 180, 272),
         B(80, 274, 200, 30, "c2", ["ответ + источники"])]
    return _svg(p, 360, 314, "".join(b), "Retrieval-Augmented Generation")


# ================================================================ 31. Монотонные ограничения
def _mono_svg():
    p = "mcon"
    b = [B(110, 12, 140, 30, "c1", ["income &lt; t ?"]),
         A(p, 150, 42, 98, 88), A(p, 210, 42, 262, 88),
         T(110, 66, "да", "mut", 11, anchor="end"), T(250, 66, "нет", "mut", 11, anchor="start"),
         B(50, 90, 90, 32, "c2", ["лист w_L"]), B(220, 90, 90, 32, "c2", ["лист w_R"]),
         B(70, 148, 220, 42, "c3", ["constraint +1:  w_L ≤ w_R", "границы передаются в поддеревья"]),
         T(180, 216, "+1: признак ↑ → прогноз ↑", "t2", 11),
         T(180, 234, "−1: признак ↑ → прогноз ↓", "t4", 11),
         T(180, 252, " 0: без ограничений", "mut", 11)]
    return _svg(p, 360, 262, "".join(b), "Монотонное ограничение на сплите дерева")


def _mono_chart(fig):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(10)
    x = np.sort(rng.uniform(0, 10, 250))
    y = 0.5 * x + np.sin(x) * 0.6 + rng.normal(0, 1.0, x.size)
    xs = np.linspace(0, 10, 400)[:, None]
    kw = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=5, random_state=0)
    free = HistGradientBoostingRegressor(**kw).fit(x[:, None], y)
    mono = HistGradientBoostingRegressor(monotonic_cst=[1], **kw).fit(x[:, None], y)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, color=GREY, alpha=0.5)
    ax.plot(xs[:, 0], free.predict(xs), color=RED, lw=1.5, label="без ограничений")
    ax.plot(xs[:, 0], mono.predict(xs), color=BLUE, lw=2.2, label="monotonic_cst = [1]")
    ax.set_xlabel("признак (например, доход)"), ax.set_ylabel("прогноз")
    ax.set_title("Градиентный бустинг: монотонность", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()


# ================================================================ 32. Monte Carlo (RL)
def _mc_svg():
    p = "mcrl"
    xs = [40, 130, 220, 310]
    b = []
    for i in range(3):
        b.append(A(p, xs[i] + 19, 55, xs[i + 1] - 20, 55))
        b.append(T((xs[i] + xs[i + 1]) / 2, 46, f"A{'₀₁₂'[i]} → R{'₁₂₃'[i]}", "t2", 11))
    for i in range(3):
        b.append(C(xs[i], 55, 17, "c1", f"S{'₀₁₂'[i]}"))
    b.append(B(292, 38, 36, 34, "c6", ["S₃"], rx=3))
    b.append(T(310, 88, "терминальное", "mut", 11))
    b.append(T(175, 18, "эпизод генерируется до конца", "mut", 11))
    for i in range(3):
        b.append(T(xs[i], 118, f"G{'₀₁₂'[i]}", "t3", bold=True))
    for i in (2, 1):
        b.append(A(p, xs[i] - 14, 114, xs[i - 1] + 16, 114, "ln s3", dash=True))
        b.append(T((xs[i] + xs[i - 1]) / 2, 108, "γ", "t3", 11))
    b += [T(180, 146, "G₂ = R₃"), T(180, 164, "G₁ = R₂ + γ·G₂"), T(180, 182, "G₀ = R₁ + γ·G₁"),
          B(30, 196, 300, 30, "c1", ["V(Sₜ) ← V(Sₜ) + α [Gₜ − V(Sₜ)]"]),
          T(180, 246, "returns считаются с конца эпизода назад", "mut", 11)]
    return _svg(p, 360, 256, "".join(b), "Monte Carlo prediction по эпизоду")


def _mc_chart(fig):
    rng = np.random.default_rng(11)
    n_states, n_ep = 5, 1000
    true_v = np.arange(1, 6) / 6
    V = np.zeros(n_states)
    cnt = np.zeros(n_states)
    hist = np.zeros((n_ep, n_states))
    for e in range(n_ep):
        s, visited = 2, []
        while 0 <= s < n_states:
            visited.append(s)
            s += 1 if rng.random() < 0.5 else -1
        G = 1.0 if s == n_states else 0.0
        for st in set(visited):          # first-visit; γ = 1, награда только в конце
            cnt[st] += 1
            V[st] += (G - V[st]) / cnt[st]
        hist[e] = V
    ax = fig.add_subplot(111)
    cols = [BLUE, GREEN, ORANGE, RED, PURPLE]
    for k in range(n_states):
        ax.plot(hist[:, k], color=cols[k], lw=1.5, label="ABCDE"[k])
        ax.axhline(true_v[k], color=cols[k], ls="--", lw=0.8)
    ax.set_xlabel("число эпизодов"), ax.set_ylabel("оценка V(s)")
    ax.set_title("First-visit MC, random walk из 5 состояний", fontsize=10)
    ax.legend(fontsize=7.5, ncol=5, loc="upper right")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()


# ================================================================ 33. Скользящие средние
def _ma_svg():
    p = "mav"
    n, alpha = 5, 2 / (5 + 1)
    rows = [("SMA", "1/n", [1 / n] * n), ("WMA", "∝ n, n−1, …", [(n - j) / 15 for j in range(n)]),
            ("EMA", "α(1−α)ʲ", [alpha * (1 - alpha) ** j for j in range(10)])]
    vmax, hmax = 1 / 3, 40
    slot, bw, x_left = 27, 17, 85
    b = []
    for r, (nm, sub, ws) in enumerate(rows):
        base = 62 + 62 * r
        b.append(T(10, base - 20, nm, anchor="start", bold=True))
        b.append(T(10, base - 5, sub, "mut", 11, anchor="start"))
        b.append(L(x_left - 4, base, x_left + 10 * slot, base))
        for j, wv in enumerate(ws):
            slot_i = 9 - j
            h = wv / vmax * hmax
            x = x_left + slot_i * slot + (slot - bw) / 2
            b.append(f'<rect class="{["c1", "c3", "c2"][r]}" x="{x:.1f}" y="{base - h:.1f}" '
                     f'width="{bw}" height="{h:.1f}"/>')
    yl = 62 + 62 * 2 + 16
    for slot_i, s in [(9, "t"), (5, "t−4"), (0, "t−9")]:
        b.append(T(x_left + slot_i * slot + slot / 2, yl, s, "mut", 11))
    b.append(T(180, yl + 22, "веса наблюдений: n = 5, α = 2/(n+1)", "mut", 11))
    return _svg(p, 360, yl + 32, "".join(b), "Веса SMA, WMA и EMA")


def _ma_chart(fig):
    rng = np.random.default_rng(12)
    t = np.arange(200)
    trend = np.where(t < 100, 10.0, 16.0) + 0.02 * t
    y = trend + rng.normal(0, 1.2, t.size)
    w = 20
    sma = np.convolve(y, np.ones(w) / w, mode="full")[:t.size]
    sma[:w - 1] = np.nan
    a = 2 / (w + 1)
    ema = np.empty_like(y)
    ema[0] = y[0]
    for i in range(1, y.size):
        ema[i] = a * y[i] + (1 - a) * ema[i - 1]
    ax = fig.add_subplot(111)
    ax.plot(t, y, color=GREY, lw=0.8, alpha=0.7, label="ряд")
    ax.plot(t, sma, color=BLUE, lw=2, label=f"SMA({w})")
    ax.plot(t, ema, color=ORANGE, lw=2, label=f"EMA(span={w})")
    ax.plot(t, trend, color="k", ls="--", lw=1, label="истинный уровень")
    ax.set_xlim(60, 160)
    ax.set_xlabel("t"), ax.set_ylabel("значение")
    ax.set_title("Реакция на скачок уровня", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()


# ================================================================ 34. Multi-head attention
def _mha_svg():
    p = "mha"

    def stack(x, y, w, h, cls, lines, n=3, off=5):
        out = []
        for k in range(n - 1, 0, -1):
            out.append(f'<rect class="{cls}" x="{x + k * off}" y="{y - k * off}" width="{w}" height="{h}" rx="5"/>')
        out.append(B(x, y, w, h, cls, lines, rx=5))
        return "".join(out)

    b, arrows = [], []
    for x, s in [(60, "Q"), (155, "K"), (250, "V")]:
        b.append(B(x, 10, 50, 26, "c6", [s]))
        arrows.append(A(p, x + 25, 36, x + 25, 66))
        b.append(stack(x - 10, 68, 70, 28, "c1", [f"W_{s}ⁱ"]))
        arrows.append(A(p, x + 25, 96, x + 25, 128))
    b.append(T(334, 82, "×h", "mut", 11))
    b.append(stack(50, 130, 250, 42, "c3", ["Scaled Dot-Product", "softmax(QKᵀ / √dₖ) V"]))
    b += arrows
    b.append(T(334, 150, "h голов", "mut", 11))
    b.append(A(p, 175, 172, 175, 192))
    b.append(B(95, 194, 160, 28, "c5", ["Concat(head₁…headₕ)"]))
    b.append(A(p, 175, 222, 175, 240))
    b.append(B(115, 242, 120, 28, "c2", ["Linear Wᴼ"]))
    b.append(A(p, 175, 270, 175, 288))
    b.append(T(175, 304, "MultiHead(Q, K, V)", bold=True))
    b.append(T(175, 322, "dₖ = d_model / h", "mut", 11))
    return _svg(p, 360, 332, "".join(b), "Схема Multi-head Attention")


def _mha_chart(fig):
    rng = np.random.default_rng(13)
    dks = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    n_keys, trials = 16, 300
    res = {True: [], False: []}
    for dk in dks:
        q = rng.normal(size=(trials, 1, dk))
        k = rng.normal(size=(trials, n_keys, dk))
        logits = (q * k).sum(-1)
        for scaled in (True, False):
            z = logits / np.sqrt(dk) if scaled else logits
            z = z - z.max(1, keepdims=True)
            pw = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            res[scaled].append(pw.max(1).mean())
    ax = fig.add_subplot(111)
    ax.plot(dks, res[False], "o-", color=RED, lw=2, label="softmax(QKᵀ)")
    ax.plot(dks, res[True], "o-", color=BLUE, lw=2, label="softmax(QKᵀ / √dₖ)")
    ax.axhline(1 / n_keys, color=GREY, ls="--", lw=1, label="равномерно (1/16)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dₖ"), ax.set_ylabel("средний max веса attention")
    ax.set_title("Без масштабирования softmax вырождается", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout()


# ================================================================ 35. Multi-output regression
def _mor_svg():
    p = "mor"
    b = [T(10, 20, "MultiOutputRegressor", "t1", anchor="start", bold=True),
         B(10, 32, 36, 80, "c6", ["X"])]
    for i, y in enumerate([34, 62, 90]):
        sub = "₁₂₃"[i]
        b += [A(p, 46, y + 11, 73, y + 11), B(75, y, 60, 22, "c1", [f"m{sub}"]),
              A(p, 135, y + 11, 148, y + 11), B(150, y, 34, 22, "c2", [f"y{sub}"])]
    b += [T(200, 60, "каждый yⱼ — своя", "mut", 11, anchor="start"),
          T(200, 76, "модель, связи между", "mut", 11, anchor="start"),
          T(200, 92, "выходами не учтены", "mut", 11, anchor="start"),
          T(10, 140, "RegressorChain", "t3", anchor="start", bold=True),
          B(10, 150, 36, 106, "c6", ["X"])]
    for i, y in enumerate([152, 192, 232]):
        sub = "₁₂₃"[i]
        b += [A(p, 46, y + 11, 73, y + 11), B(75, y, 60, 22, "c3", [f"m{sub}"]),
              A(p, 135, y + 11, 148, y + 11), B(150, y, 34, 22, "c2", [f"y{sub}"])]
    b += [A(p, 158, 175, 128, 190, "ln s3"), A(p, 158, 215, 128, 230, "ln s3"),
          T(200, 167, "вход m₁: X", "mut", 11, anchor="start"),
          T(200, 207, "вход m₂: [X, y₁]", "mut", 11, anchor="start"),
          T(200, 247, "вход m₃: [X, y₁, y₂]", "mut", 11, anchor="start")]
    return _svg(p, 360, 266, "".join(b), "MultiOutputRegressor и RegressorChain")


VISUALS = {
    "latent_semantic_analysis": [
        {"title": "LSA как усечённое SVD", "svg": _lsa_svg(),
         "caption": "Term-document матрица X раскладывается на k концептов: термы и документы попадают в одно "
                    "k-мерное пространство, где близость меряют косинусом."},
        {"title": "Сколько концептов оставить", "chart": _lsa_chart,
         "caption": "На синтетическом корпусе с тремя темами первые три сингулярных значения заметно выше "
                    "остальных: после «локтя» компоненты в основном описывают шум."},
    ],
    "lda_method": [
        {"title": "Геометрия LDA и PCA", "svg": _lda_svg(),
         "caption": "PCA выбирает направление наибольшей общей дисперсии (вдоль облаков), LDA — направление, где "
                    "отношение межклассового разброса S_B к внутриклассовому S_W максимально."},
        {"title": "Проекции на одну ось", "chart": _lda_chart,
         "caption": "На тех же данных проекция на PC1 смешивает классы, а одна компонента LDA разделяет их почти "
                    "полностью: здесь метки классов решают всё."},
    ],
    "lightgbm": [
        {"title": "Leaf-wise против level-wise", "svg": _lightgbm_svg(),
         "caption": "LightGBM на каждом шаге делит один лист с наибольшим снижением loss (номера — порядок "
                    "сплитов), поэтому деревья получаются глубокими и несимметричными; num_leaves и max_depth "
                    "сдерживают переобучение."},
    ],
    "lime_neural_networks": [
        {"title": "Алгоритм LIME", "svg": _lime_svg(),
         "caption": "LIME не заглядывает внутрь модели: он только опрашивает её на возмущённых примерах и "
                    "подгоняет простую модель, взвешивая точки по близости к x."},
        {"title": "Локальная линейная аппроксимация", "chart": _lime_chart,
         "caption": "Глобально f нелинейна, но около объясняемой точки её хорошо приближает прямая. Наклон этой "
                    "прямой и есть «важность» признака, и верен он только локально."},
    ],
    "linear_algebra_optimization": [
        {"title": "Геометрия SVD", "svg": _linalg_svg(),
         "caption": "Любая матрица действует как поворот Vᵀ, растяжение по осям Σ и ещё один поворот U: единичная "
                    "окружность переходит в эллипс с полуосями σ₁, σ₂."},
        {"title": "GD и Momentum на «овраге»", "chart": _linalg_chart,
         "caption": "При сильно разной кривизне по осям GD с почти предельным η скачет поперёк оврага. Momentum "
                    "гасит зигзаг и разгоняется вдоль оврага, но по инерции может проскочить минимум."},
    ],
    "linear_autoencoders": [
        {"title": "Архитектура линейного автоэнкодера", "svg": _linae_svg(),
         "caption": "Энкодер сжимает n признаков в k латентных, декодер восстанавливает n. Без нелинейностей "
                    "оптимум MSE достигается на том же подпространстве, что и у PCA."},
        {"title": "Сходимость к ошибке PCA", "chart": _linae_chart,
         "caption": "Ошибка линейного AE, обученного градиентным спуском, опускается до уровня PCA с тем же k и "
                    "не становится ниже: это и есть эквивалентность."},
    ],
    "linear_ensembles": [
        {"title": "Стекинг линейных моделей", "svg": _linens_svg(),
         "caption": "Базовые модели с разной регуляризацией дают out-of-fold прогнозы, на которых обучается "
                    "мета-модель. Кросс-валидация не даёт мета-модели увидеть прогнозы на собственном train."},
    ],
    "logical_inference_learning": [
        {"title": "Нейро-символьный цикл обучения", "svg": _logic_svg(),
         "caption": "Нейросеть выдаёт степени истинности предикатов, нечёткие логические операции превращают "
                    "правила в дифференцируемый штраф, и градиент учит сеть соблюдать знания."},
        {"title": "Разные t-нормы для AND", "chart": _logic_chart,
         "caption": "Все три варианта совпадают с булевым AND на 0 и 1, но различаются между ними: у Łukasiewicz "
                    "нулевой градиент при a &lt; 0.3, у min градиент идёт только в меньший аргумент."},
    ],
    "lstm": [
        {"title": "Схема ячейки LSTM", "svg": _lstm_svg(),
         "caption": "Forget-гейт масштабирует старую память Cₜ₋₁, input-гейт добавляет кандидата C̃ₜ, "
                    "output-гейт решает, какую часть tanh(Cₜ) выдать как hₜ. Аддитивное обновление Cₜ "
                    "сохраняет градиент на длинных последовательностях."},
    ],
    "malware_detection": [
        {"title": "Пайплайн детектора", "svg": _malware_svg(),
         "caption": "Статический анализ безопасен и быстр, динамический видит реальное поведение в sandbox; "
                    "признаки обоих видов сводятся в один вектор для классификатора."},
    ],
    "material_design": [
        {"title": "Closed-loop оптимизация", "svg": _matdesign_svg(),
         "caption": "Self-driving lab повторяет цикл design–make–test–analyze: каждое измерение обновляет "
                    "модель, и та предлагает следующего кандидата."},
        {"title": "Bayesian optimization в 1D", "chart": _matdesign_chart,
         "caption": "GP-модель даёт прогноз и неопределённость; Expected Improvement максимален там, где высок "
                    "прогноз или велика σ. Следующий эксперимент ставят в точку максимума EI (красная линия)."},
    ],
    "material_informatics": [
        {"title": "Рабочий процесс", "svg": _matinf_svg(),
         "caption": "Данные из баз превращаются в дескрипторы или граф, модель предсказывает свойства, лучшие "
                    "кандидаты проверяются DFT или синтезом, и результаты возвращаются в обучающую выборку."},
    ],
    "material_properties_ml": [
        {"title": "Structure–property", "svg": _matprop_svg(),
         "caption": "Одни и те же дескрипторы состава и структуры кормят модели разных классов свойств; оценка "
                    "неопределённости показывает, каким прогнозам можно доверять."},
        {"title": "Интерполяция и экстраполяция", "chart": _matprop_chart,
         "caption": "Внутри области обучения GP точен и уверен; за её пределами прогноз уходит к среднему, а "
                    "интервал ±2σ быстро растёт. Это сигнал не доверять экстраполяции."},
    ],
    "matplotlib_seaborn_ml": [
        {"title": "Анатомия рисунка", "svg": _mpl_svg(),
         "caption": "Figure — весь холст, Axes — отдельная система координат с графиком. Подписи, легенда и "
                    "деления настраиваются методами ax.set_*."},
        {"title": "Residuals plot", "chart": _mpl_chart,
         "caption": "Остатки линейной модели образуют дугу вместо облака вокруг нуля: в данных есть "
                    "нелинейность, которую модель не учла."},
    ],
    "matrix_factorization_svd_nmf": [
        {"title": "Факторизация матрицы рейтингов", "svg": _mf_svg(),
         "caption": "Пропущенный рейтинг (?) восстанавливается как скалярное произведение строки пользователя "
                    "в U и столбца объекта в Vᵀ; loss считается только по известным ячейкам."},
        {"title": "Выбор числа факторов", "chart": _mf_chart,
         "caption": "На матрице истинного ранга 3 ошибка на train падает с ростом k, а на test минимальна около "
                    "истинного ранга и затем растёт: лишние факторы подгоняют шум."},
    ],
    "mcmc_bayesian_inference": [
        {"title": "Шаг Metropolis-Hastings", "svg": _mcmc_svg(),
         "caption": "Evidence P(D) сокращается в отношении r, поэтому достаточно ненормированного posterior. "
                    "Отвергнутое предложение повторяет текущее состояние в цепи."},
        {"title": "Trace и гистограмма", "chart": _mcmc_chart,
         "caption": "Цепь стартует далеко (θ = −6), и первые шаги отбрасываются как burn-in. Дальше она "
                    "переходит между двумя модами, и гистограмма совпадает с целевым распределением."},
    ],
    "meta_features_learning": [
        {"title": "Meta-learning для выбора алгоритма", "svg": _meta_svg(),
         "caption": "На прошлых датасетах учится meta-learner: мета-признаки → лучший алгоритм. Для новой "
                    "задачи достаточно посчитать мета-признаки и получить рекомендацию без полного перебора."},
    ],
    "metric_based_learning_siamese": [
        {"title": "Siamese-сеть", "svg": _siam_svg(),
         "caption": "Обе ветки — одна и та же сеть с общими весами, поэтому расстояние d сравнимо для любых "
                    "пар, включая классы, которых не было в обучении."},
        {"title": "Contrastive loss", "chart": _siam_chart,
         "caption": "Похожие пары штрафуются за любое расстояние, разные — только пока ближе margin m; дальше "
                    "их не трогают."},
    ],
    "metric_learning": [
        {"title": "Triplet loss", "svg": _metric_svg(),
         "caption": "Обучение притягивает positive к anchor и отталкивает negative за пределы d(A,P) + margin; "
                    "когда негатив уже дальше, loss равен нулю."},
        {"title": "Angular margin: ArcFace и CosFace", "chart": _metric_chart,
         "caption": "Обе функции занижают логит целевого класса относительно обычного cos θ, поэтому сеть "
                    "сильнее стягивает эмбеддинги к центру класса. ArcFace сдвигает угол, CosFace — косинус."},
    ],
    "metrics_visualization": [
        {"title": "Confusion matrix и метрики", "svg": _metvis_svg(),
         "caption": "Раскладка как в sklearn: строки — истинный класс, столбцы — предсказанный. ROC и PR "
                    "получаются перебором порога и пересчётом этих четырёх ячеек."},
        {"title": "ROC и PR на дисбалансе", "chart": _metvis_chart,
         "caption": "При 20% положительных PR-кривая различает модели заметнее, чем ROC: базовый уровень "
                    "Precision равен доле класса 1, а не диагонали."},
    ],
    "ml_pipelines": [
        {"title": "Pipeline с ColumnTransformer", "svg": _pipe_svg(),
         "caption": "Числовые и категориальные столбцы проходят свои трансформеры, результаты склеиваются и "
                    "идут в модель. Весь pipeline обучается одним fit, поэтому в CV нет утечки."},
    ],
    "mlops_best_practices": [
        {"title": "Жизненный цикл ML-системы", "svg": _mlops_svg(),
         "caption": "Этапы замкнуты в цикл: мониторинг в production запускает переобучение. Версионирование "
                    "кода, данных и моделей делает каждый виток воспроизводимым."},
    ],
    "mlp": [
        {"title": "Структура MLP", "svg": _mlp_svg(),
         "caption": "Каждый скрытый слой — линейное преобразование и нелинейность; выходная активация зависит "
                    "от задачи: sigmoid, softmax или линейная."},
        {"title": "Нелинейная граница решения", "chart": _mlp_chart,
         "caption": "Два скрытых слоя по 16 ReLU-нейронов строят изогнутую границу (чёрная линия, p = 0.5) для "
                    "данных moons, которые линейно не разделяются."},
    ],
    "mlp_tabular_data": [
        {"title": "Архитектура для таблиц", "svg": _mlptab_svg(),
         "caption": "Числовые признаки нормализуют, категориальные переводят в обучаемые эмбеддинги; после "
                    "конкатенации идут 2–5 блоков Linear → BatchNorm → ReLU → Dropout."},
    ],
    "model_deployment": [
        {"title": "Типы развёртывания", "svg": _deploy_svg(),
         "caption": "Real-time — сервис в контейнере за балансировщиком, ответ за миллисекунды; batch — "
                    "периодическая задача, которая пишет прогнозы в хранилище."},
    ],
    "model_monitoring_ab_testing": [
        {"title": "Shadow → canary → A/B", "svg": _ab_svg(),
         "caption": "Риск растёт ступенями: shadow проверяет технику без влияния на пользователей, canary — "
                    "поломки на малой доле трафика, и только A/B измеряет причинный бизнес-эффект."},
        {"title": "Размер выборки для A/B", "chart": _ab_chart,
         "caption": "n ≈ (z₁₋α/₂ + z₁₋β)²·[p₁(1−p₁) + p₂(1−p₂)] / (p₂ − p₁)². Уменьшение MDE вдвое "
                    "увеличивает выборку примерно в 4 раза, а при низкой конверсии нужно ещё больше трафика."},
    ],
    "model_monitoring": [
        {"title": "Контур мониторинга", "svg": _mon_svg(),
         "caption": "Логи прогнозов питают три вида проверок: дрейф данных (сразу), качество (когда придут "
                    "метки) и инфраструктуру. Пороги в Grafana превращают их в алерты и триггеры переобучения."},
        {"title": "Data drift: KS-статистика", "chart": _mon_chart,
         "caption": "D — наибольший вертикальный разрыв между эмпирическими функциями распределения на train и "
                    "в production. На больших выборках смотрите на величину D, а не только на p-value."},
    ],
    "model_registries": [
        {"title": "Стадии версий модели", "svg": _reg_svg(),
         "caption": "Каждая версия проходит None → Staging → Production → Archived; в Production обычно одна "
                    "версия, а архив позволяет быстро откатиться."},
    ],
    "model_serialization": [
        {"title": "Форматы сериализации", "svg": _ser_svg(),
         "caption": "pickle/joblib удобны для sklearn, но привязаны к Python и небезопасны для недоверенных "
                    "файлов; ONNX переносит модель в другие языки и рантаймы."},
    ],
    "modern_trends_deep_learning": [
        {"title": "Retrieval-Augmented Generation", "svg": _rag_svg(),
         "caption": "База знаний индексируется заранее; на запрос retriever находит top-k фрагментов, и LLM "
                    "отвечает с опорой на них, что снижает галлюцинации и даёт ссылки на источники."},
    ],
    "monotonic_constraints": [
        {"title": "Ограничение на сплите", "svg": _mono_svg(),
         "caption": "При ограничении +1 бустинг принимает только сплиты, где значение левого листа не больше "
                    "правого, и передаёт границы глубже, так что монотонна вся функция."},
        {"title": "Бустинг с monotonic_cst", "chart": _mono_chart,
         "caption": "Без ограничения модель повторяет локальные провалы шума; с monotonic_cst = [1] прогноз — "
                    "неубывающая ступенчатая функция признака."},
    ],
    "monte_carlo_methods": [
        {"title": "MC prediction по эпизоду", "svg": _mc_svg(),
         "caption": "MC ждёт конца эпизода, затем считает returns с конца назад, Gₜ = Rₜ₊₁ + γGₜ₊₁, и "
                    "сдвигает V(Sₜ) к наблюдённому Gₜ. Модель среды не нужна."},
        {"title": "Сходимость оценок V(s)", "chart": _mc_chart,
         "caption": "Random walk A–E с наградой 1 на правом краю: оценки first-visit MC шумят в начале и "
                    "сходятся к истинным 1/6…5/6 (пунктир) по мере роста числа эпизодов."},
    ],
    "moving_average_methods": [
        {"title": "Веса наблюдений", "svg": _ma_svg(),
         "caption": "SMA усредняет n точек поровну, WMA линейно усиливает свежие, EMA даёт геометрически "
                    "убывающие веса всей истории (α = 2/(n+1))."},
        {"title": "Запаздывание SMA и EMA", "chart": _ma_chart,
         "caption": "После скачка уровня обе кривые отстают: SMA(20) догоняет за 20 шагов линейно, EMA "
                    "реагирует сразу, но приближается к новому уровню экспоненциально."},
    ],
    "multi_head_attention": [
        {"title": "Схема Multi-head Attention", "svg": _mha_svg(),
         "caption": "Каждая из h голов проецирует Q, K, V своими матрицами и считает attention в пространстве "
                    "размера dₖ; выходы конкатенируются и смешиваются проекцией Wᴼ."},
        {"title": "Зачем делить на √dₖ", "chart": _mha_chart,
         "caption": "Дисперсия скалярного произведения растёт как dₖ, и без масштабирования softmax "
                    "почти one-hot: градиенты затухают. Деление на √dₖ держит распределение весов стабильным."},
    ],
    "multi_output_regression": [
        {"title": "Независимые модели и цепочка", "svg": _mor_svg(),
         "caption": "MultiOutputRegressor обучает по модели на каждый выход; RegressorChain добавляет "
                    "предыдущие прогнозы в признаки следующей модели и так учитывает зависимости между выходами."},
    ],
}
