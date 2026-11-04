# -*- coding: utf-8 -*-
"""Рисунки группы g1 (capsule_networks … deep_learning_key_breakthroughs) для tools/add_visuals.py."""

import math

import numpy as np

BLUE, GREEN, ORANGE, RED, PURPLE, GREY = "#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"


# ---------------------------------------------------------------- помощники SVG
def _e(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _n(v):
    return f"{round(float(v), 1):g}"


def svg(p, w, h, label, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_e(label)}">'
            f'<defs><marker id="{p}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            + "".join(body) + "</svg>")


def T(x, y, s, cls="", a="middle"):
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{_n(x)}" y="{_n(y)}" text-anchor="{a}"{c}>{_e(s)}</text>'


def R(x, y, w, h, cls, rx=6, extra=""):
    return f'<rect class="{cls}" x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{rx}"{extra}/>'


def B(x, y, w, h, cls, *lines, tcls="", rx=6):
    out = [R(x, y, w, h, cls, rx)]
    y0 = y + h / 2 - (len(lines) - 1) * 7.5 + 4
    for i, s in enumerate(lines):
        out.append(T(x + w / 2, y0 + 15 * i, s, tcls))
    return "".join(out)


def L(x1, y1, x2, y2, cls="ln", p=None, dash=False, sw=None):
    m = f' marker-end="url(#{p}-a)"' if p else ""
    d = ' stroke-dasharray="4 3"' if dash else ""
    w = f' stroke-width="{sw}"' if sw else ""
    return f'<line class="{cls}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}"{m}{d}{w}/>'


def P(d, cls="ln", p=None, dash=False, sw=None):
    m = f' marker-end="url(#{p}-a)"' if p else ""
    da = ' stroke-dasharray="4 3"' if dash else ""
    w = f' stroke-width="{sw}"' if sw else ""
    return f'<path class="{cls}" d="{d}"{m}{da}{w}/>'


def C(cx, cy, r, cls):
    return f'<circle class="{cls}" cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}"/>'


# ================================================================ capsule_networks
def _capsule_svg():
    p = "capsule_networks"
    b = []
    for cx, i in [(70, "₁"), (180, "₂"), (290, "₃")]:
        b.append(B(cx - 30, 12, 60, 26, "c1", f"u{i}"))
        b.append(L(cx, 38, cx, 60, p=p))
        b.append(B(cx - 52, 62, 104, 28, "c5", f"ûⱼ|{i} = W{i}ⱼu{i}"))
    # к сумматору
    b.append(L(70, 90, 166, 142.3, p=p))
    b.append(L(180, 90, 180, 132, p=p))
    b.append(L(290, 90, 194, 142.3, p=p))
    b.append(T(108, 132, "c₁ⱼ", "t5"))
    b.append(T(186, 118, "c₂ⱼ", "t5", "start"))
    b.append(T(252, 132, "c₃ⱼ", "t5"))
    b.append(C(180, 150, 16, "c3"))
    b.append(T(180, 155, "Σ"))
    b.append(T(204, 154, "sⱼ = Σᵢ cᵢⱼ ûⱼ|ᵢ", a="start"))
    b.append(L(180, 166, 180, 186, p=p))
    b.append(B(130, 188, 100, 28, "c2", "squash"))
    b.append(T(240, 206, "‖vⱼ‖ ∈ [0, 1)", "mut", "start"))
    b.append(L(180, 216, 180, 238, p=p))
    b.append(B(145, 240, 70, 28, "c1", "vⱼ"))
    b.append(T(225, 258, "длина = P(entity)", "mut", "start"))
    # обратная связь routing
    b.append(P("M145,254 H26 V104 H90", "ln s5", p=p, dash=True))
    b.append(T(34, 162, "cᵢⱼ = softmax(bᵢⱼ)", "t5", "start"))
    b.append(T(34, 178, "bᵢⱼ += ûⱼ|ᵢ · vⱼ", "t5", "start"))
    b.append(T(34, 194, "r итераций", "mut", "start"))
    return svg(p, 360, 280, "Dynamic routing между капсулами", b)


def capsule_chart(fig):
    ax = fig.add_subplot(111)
    s = np.linspace(0, 5, 300)
    ax.plot(s, s ** 2 / (1 + s ** 2), color=BLUE, lw=2.2, label=r"squash: $\|v\| = \|s\|^2/(1+\|s\|^2)$")
    ax.axhline(0.9, color=GREEN, ls="--", lw=1.3, label=r"$m^+ = 0.9$ (класс есть)")
    ax.axhline(0.1, color=RED, ls="--", lw=1.3, label=r"$m^- = 0.1$ (класса нет)")
    ax.set_xlabel(r"$\|s_j\|$ до squash"), ax.set_ylabel(r"$\|v_j\|$ после squash")
    ax.set_ylim(0, 1.05)
    ax.set_title("Длина капсулы как вероятность", fontsize=10)
    ax.legend(fontsize=7.5, loc="center right")


# ================================================================ categorical_encoding
def _catenc_svg():
    p = "categorical_encoding"
    b = []
    data = [("red", 1), ("blue", 0), ("green", 1), ("red", 0)]
    b.append(B(110, 12, 70, 20, "c6", "color", rx=0))
    b.append(B(180, 12, 40, 20, "c6", "y", rx=0))
    for k, (c, y) in enumerate(data):
        b.append(B(110, 32 + 20 * k, 70, 20, "c6", c, rx=0))
        b.append(B(180, 32 + 20 * k, 40, 20, "c6", str(y), rx=0))
    for x2 in (60, 185, 305):
        b.append(L(165, 114, x2, 126, p=p))
    b.append(T(60, 142, "Label", "t1"))
    b.append(T(185, 142, "One-Hot", "t2"))
    b.append(T(305, 142, "Target", "t3"))
    # Label
    le = {"blue": 0, "green": 1, "red": 2}
    b.append(B(30, 150, 60, 20, "c1", "код", rx=0))
    for k, (c, _) in enumerate(data):
        b.append(B(30, 170 + 20 * k, 60, 20, "c1", str(le[c]), rx=0))
    # One-hot
    cats = ["blue", "green", "red"]
    for j, cat in enumerate(cats):
        b.append(B(125 + 40 * j, 150, 40, 20, "c2", cat, rx=0))
        for k, (c, _) in enumerate(data):
            b.append(B(125 + 40 * j, 170 + 20 * k, 40, 20, "c2", "1" if c == cat else "0", rx=0))
    # Target
    te = {"red": "0.5", "blue": "0", "green": "1"}
    b.append(B(275, 150, 60, 20, "c3", "mean y", rx=0))
    for k, (c, _) in enumerate(data):
        b.append(B(275, 170 + 20 * k, 60, 20, "c3", te[c], rx=0))
    b.append(T(60, 268, "ложный порядок", "mut"))
    b.append(T(185, 268, "+K колонок", "mut"))
    b.append(T(305, 268, "риск утечки", "mut"))
    return svg(p, 360, 280, "Label, One-Hot и Target encoding", b)


# ================================================================ choosing_clusters_number
def _choosek_svg():
    p = "choosing_clusters_number"
    b = [B(70, 10, 220, 28, "c1", "Перебрать K = 2…10 (KMeans)"),
         L(180, 38, 180, 52, p=p),
         R(8, 54, 344, 116, "s6", rx=8, extra=' fill="none" stroke-dasharray="4 3"'),
         B(16, 62, 160, 28, "c6", "Локоть: излом inertia"),
         B(184, 62, 160, 28, "c6", "Silhouette → max"),
         B(16, 98, 160, 28, "c6", "Davies-Bouldin → min"),
         B(184, 98, 160, 28, "c6", "Calinski-Harabasz → max"),
         B(16, 134, 328, 28, "c6", "Gap: min K, Gap(K) ≥ Gap(K+1) − sₖ₊₁"),
         L(180, 170, 180, 186, p=p),
         B(60, 188, 240, 28, "c3", "Кандидаты: где методы сходятся"),
         L(180, 216, 180, 232, p=p),
         B(40, 234, 280, 28, "c2", "Бизнес-проверка: размеры, смысл"),
         T(180, 282, "нет «правильного» K — есть разумное", "mut")]
    return svg(p, 360, 292, "Процедура выбора числа кластеров", b)


def choosek_chart(fig):
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=7)
    ks = range(1, 11)
    inert, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
        inert.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_) if k > 1 else np.nan)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(ks, inert, "o-", color=BLUE, lw=2)
    ax.plot(4, inert[3], "o", ms=11, mfc="none", mec=RED, mew=2)
    ax.annotate("локоть", (4, inert[3]), (5.5, inert[1] * 0.75), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_title("Метод локтя (истинное K = 4)", fontsize=10)
    ax.set_ylabel("inertia (WCSS)"), ax.set_xticks(list(ks))
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(ks, sil, "o-", color=GREEN, lw=2)
    ax2.plot(4, sil[3], "o", ms=11, mfc="none", mec=RED, mew=2)
    ax2.set_title("Silhouette score: максимум", fontsize=10)
    ax2.set_xlabel("K"), ax2.set_ylabel("silhouette"), ax2.set_xticks(list(ks))
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ cicd_ml
def _cicd_svg():
    p = "cicd_ml"
    b = [T(95, 16, "CI", "t1"), T(265, 16, "CD + CT", "t2"),
         B(20, 26, 150, 30, "c6", "git push / PR"),
         L(95, 56, 95, 72, p=p),
         B(20, 74, 150, 30, "c1", "Тесты кода и данных"),
         L(95, 104, 95, 120, p=p),
         B(20, 122, 150, 30, "c1", "Обучение модели"),
         L(95, 152, 95, 168, p=p),
         B(20, 170, 150, 36, "c3", "Quality gate:", "метрики ≥ порога"),
         L(95, 206, 95, 226, p=p),
         T(102, 220, "да", "mut", "start"),
         B(20, 228, 150, 30, "c5", "Model Registry"),
         L(170, 243, 188, 243, p=p),
         B(190, 228, 150, 30, "c2", "Deploy: staging"),
         L(265, 228, 265, 202, p=p),
         B(190, 170, 150, 30, "c2", "Deploy: production"),
         L(265, 170, 265, 154, p=p),
         B(190, 122, 150, 30, "c6", "Мониторинг"),
         L(265, 122, 265, 98, p=p),
         B(190, 56, 150, 40, "c4", "Триггер CT:", "drift / расписание"),
         P("M190,76 H180 V137 H172", "ln s4", p=p, dash=True)]
    return svg(p, 360, 268, "CI/CD/CT пайплайн для ML", b)


# ================================================================ clarans
def _clarans_svg():
    p = "clarans"
    b = [B(20, 14, 200, 28, "c1", "k случайных медоидов"),
         L(120, 42, 120, 60, p=p),
         B(20, 62, 200, 40, "c5", "Сосед: заменить 1 медоид", "случайной точкой"),
         L(120, 102, 120, 124, p=p),
         B(20, 126, 200, 28, "c3", "TC(сосед) < TC(тек.)?"),
         L(220, 140, 238, 140, p=p), T(229, 133, "да", "mut"),
         B(240, 120, 80, 40, "c2", "перейти,", "j = 0"),
         P("M280,120 V82 H222", "ln", p=p),
         L(120, 154, 120, 176, p=p), T(128, 170, "нет", "mut", "start"),
         B(20, 178, 200, 28, "c6", "j = j + 1"),
         L(120, 206, 120, 228, p=p),
         B(20, 230, 200, 28, "c3", "j > maxneighbor?"),
         P("M20,244 H8 V82 H18", "ln", p=p), T(14, 172, "нет", "mut", "start"),
         L(120, 258, 120, 280, p=p), T(128, 274, "да", "mut", "start"),
         B(20, 282, 200, 40, "c2", "локальный минимум:", "сохранить лучший"),
         L(220, 302, 238, 302, p=p),
         B(240, 282, 100, 40, "c1", "повторить", "numlocal раз"),
         P("M335,282 V28 H222", "ln s6", p=p, dash=True),
         T(180, 346, "TC = Σ dist(x, ближайший медоид)", "mut")]
    return svg(p, 360, 356, "Алгоритм CLARANS", b)


# ================================================================ classical_rl_tabular
def _rl_svg():
    p = "classical_rl_tabular"
    b = [R(90, 14, 180, 70, "c1"),
         T(180, 34, "Агент"),
         T(180, 54, "Q-таблица: S × A"),
         T(180, 72, "π(s) = argmaxₐ Q(s,a)", "mut"),
         B(90, 150, 180, 44, "c2", "Среда (MDP)", "P(s′|s,a), R(s,a)"),
         P("M270,49 H320 V172 H272", "ln", p=p),
         T(312, 120, "действие aₜ", "", "end"),
         P("M90,172 H40 V49 H88", "ln", p=p),
         T(48, 112, "состояние sₜ₊₁", "", "start"),
         T(48, 128, "награда rₜ₊₁", "", "start"),
         T(180, 224, "Q-learning (off-policy):", "t1"),
         T(180, 244, "Q(s,a) ← Q(s,a) + α[r + γ·maxₐ′Q(s′,a′) − Q(s,a)]"),
         T(180, 268, "SARSA (on-policy): Q(s′,a′) вместо max", "mut")]
    return svg(p, 360, 280, "Цикл агент–среда и обновление Q-learning", b)


def rl_chart(fig):
    n, gamma = 5, 0.9
    goal, pit = (4, 4), (2, 2)
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    V = np.zeros((n, n))

    def step(r, c, m):
        return min(max(r + m[0], 0), n - 1), min(max(c + m[1], 0), n - 1)

    def q(r, c, m, V):
        r2, c2 = step(r, c, m)
        rew = 1.0 if (r2, c2) == goal else (-1.0 if (r2, c2) == pit else 0.0)
        return rew + (0 if (r2, c2) in (goal, pit) else gamma * V[r2, c2])

    for _ in range(60):
        Vn = np.zeros_like(V)
        for r in range(n):
            for c in range(n):
                if (r, c) not in (goal, pit):
                    Vn[r, c] = max(q(r, c, m, V) for m in moves)
        V = Vn
    ax = fig.add_subplot(111)
    im = ax.imshow(V, cmap="Blues", vmin=0, vmax=1)
    glyph = ["↑", "→", "↓", "←"]
    for r in range(n):
        for c in range(n):
            if (r, c) == goal:
                ax.text(c, r, "G", ha="center", va="center", fontsize=13, color=GREEN, weight="bold")
            elif (r, c) == pit:
                ax.text(c, r, "яма", ha="center", va="center", fontsize=10, color=RED, weight="bold")
            else:
                a = int(np.argmax([q(r, c, m, V) for m in moves]))
                ax.text(c, r, glyph[a], ha="center", va="center", fontsize=14,
                        color="white" if V[r, c] > 0.6 else "#2b3440")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    ax.set_title("Value iteration, γ = 0.9: V(s) и жадная π", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("V(s)")


# ================================================================ clip_multimodal_models
def _clip_svg():
    p = "clip_multimodal_models"
    b = [B(140, 10, 160, 26, "c2", "Text Encoder"),
         T(132, 28, "подписи →", "mut", "end"),
         L(220, 36, 220, 46, p=p),
         T(45, 104, "фото", "mut"),
         B(8, 110, 76, 48, "c1", "Image", "Encoder"),
         L(84, 134, 90, 134, p=p)]
    for j in range(4):
        b.append(B(140 + 40 * j, 48, 40, 24, "c2", f"T{'₁₂₃₄'[j]}", rx=0))
    for i in range(4):
        b.append(B(92, 74 + 30 * i, 40, 30, "c1", f"I{'₁₂₃₄'[i]}", rx=0))
        for j in range(4):
            b.append(B(140 + 40 * j, 74 + 30 * i, 40, 30, "c3" if i == j else "c6",
                       f"I{'₁₂₃₄'[i]}·T{'₁₂₃₄'[j]}", rx=0))
    b += [T(180, 216, "softmax по строкам и столбцам, τ", ""),
          T(180, 236, "L = ½ (CE строк + CE столбцов)", "t1"),
          T(180, 256, "диагональ — верные пары (фото, текст)", "mut")]
    return svg(p, 340, 266, "Contrastive обучение CLIP", b)


# ================================================================ cluster_visualization
def _clvis_svg():
    p = "cluster_visualization"
    rng = np.random.default_rng(3)
    b = [B(14, 12, 150, 30, "c6", "X: n × d, d ≫ 2"),
         L(164, 27, 194, 27, p=p),
         B(196, 12, 150, 30, "c6", "StandardScaler"),
         L(240, 42, 112, 68, p=p),
         L(280, 42, 278, 68, p=p),
         B(30, 70, 150, 40, "c1", "PCA", "линейная, быстро"),
         B(196, 70, 150, 40, "c5", "t-SNE / UMAP", "локальная структура"),
         L(105, 110, 140, 134, p=p),
         L(271, 110, 236, 134, p=p),
         R(60, 136, 240, 124, "c6")]
    for (cx, cy), cls in [((115, 180), "t1"), ((215, 168), "t2"), ((180, 228), "t3")]:
        for dx, dy in rng.normal(0, 1, (14, 2)) * [14, 10]:
            b.append(C(cx + dx, cy + dy, 3, cls))
        b.append(T(cx, cy + 5, "×"))
    b += [T(180, 276, "цвет = метка кластера, × = центроид", "mut")]
    return svg(p, 360, 286, "Пайплайн визуализации кластеров", b)


def clvis_chart(fig):
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    X, _ = make_blobs(n_samples=300, n_features=10, centers=4, cluster_std=2.5, random_state=1)
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, n_init=4, random_state=0).fit(X)
    cols = np.array([BLUE, GREEN, ORANGE, PURPLE])[km.labels_]
    pca = PCA(2).fit(X)
    Z = pca.transform(X)
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(Z[:, 0], Z[:, 1], s=8, c=cols, alpha=0.7)
    Cz = pca.transform(km.cluster_centers_)
    ax.scatter(Cz[:, 0], Cz[:, 1], marker="X", s=90, c="black", edgecolors="white")
    ev = pca.explained_variance_ratio_
    ax.set_title(f"PCA: PC1+PC2 объясняют {100 * ev.sum():.0f}% дисперсии", fontsize=10)
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2")
    Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(Y[:, 0], Y[:, 1], s=8, c=cols, alpha=0.7)
    ax2.set_title("t-SNE: кластеры разделены, но расстояния\nмежду ними не интерпретируются", fontsize=9.5)
    ax2.set_xticks([]), ax2.set_yticks([])
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ clustering_metrics
def _clmetrics_svg():
    p = "clustering_metrics"
    rng = np.random.default_rng(5)
    b = ['<ellipse class="c2" cx="100" cy="95" rx="72" ry="55"/>',
         '<ellipse class="c3" cx="270" cy="95" rx="64" ry="55"/>',
         T(100, 30, "свой кластер A", "t2"), T(270, 30, "ближайший B", "t3")]
    ia = (128, 92)
    A = [(62, 70), (80, 120), (98, 64), (60, 104), (108, 132), (140, 120)]
    Bp = [(240, 70), (262, 120), (296, 80), (300, 118), (250, 96)]
    for x, y in A:
        b.append(L(ia[0], ia[1], x, y, "ln s2"))
    for x, y in Bp:
        b.append(L(ia[0], ia[1], x, y, "ln s3", dash=True))
    for x, y in A:
        b.append(C(x, y, 4, "t2"))
    for x, y in Bp:
        b.append(C(x, y, 4, "t3"))
    b.append(C(ia[0], ia[1], 6, "c4"))
    b.append(T(ia[0] - 4, ia[1] - 10, "i"))
    b.append(T(84, 92, "a(i)", "t2"))
    b.append(T(190, 62, "b(i)", "t3"))
    b += [T(180, 178, "a(i) — среднее расстояние внутри кластера"),
          T(180, 196, "b(i) — до ближайшего чужого кластера"),
          T(180, 222, "s(i) = (b − a) / max(a, b) ∈ [−1, 1]", "t1"),
          T(180, 242, "s ≈ 1 хорошо, s < 0 — точка не в том кластере", "mut")]
    _ = rng
    return svg(p, 360, 254, "Геометрия silhouette", b)


def clmetrics_chart(fig):
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_samples
    X, _ = make_blobs(n_samples=300, centers=[[0, 0], [4, 0], [2, 3.2]], cluster_std=[0.6, 1.1, 0.8],
                      random_state=2)
    lab = KMeans(n_clusters=3, n_init=4, random_state=0).fit_predict(X)
    s = silhouette_samples(X, lab)
    ax = fig.add_subplot(111)
    y0 = 0
    for k, col in zip(range(3), [BLUE, GREEN, ORANGE]):
        v = np.sort(s[lab == k])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=col, alpha=0.8)
        ax.text(-0.08, y0 + len(v) / 2, f"C{k}", va="center", fontsize=9)
        y0 += len(v) + 10
    ax.axvline(s.mean(), color=RED, ls="--", lw=1.5, label=f"среднее = {s.mean():.2f}")
    ax.set_xlabel("s(i)"), ax.set_yticks([])
    ax.set_xlim(-0.15, 1)
    ax.set_title("Silhouette plot, K = 3", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


# ================================================================ cnn_architectures
def _row(b, x, y, items, gap):
    for lab, w, cls in items:
        if lab is None:
            b.append(R(x, y + 3, 8, 20, "c6", rx=1))
            w = 8
        else:
            b.append(B(x, y, w, 26, cls, lab, rx=3))
        x += w + gap


def _cnnarch_svg():
    p = "cnn_architectures"
    b = [T(12, 16, "LeNet-5 (1998) · tanh · 60K", "t1", "start")]
    _row(b, 12, 24, [("Conv6", 44, "c1"), (None, 0, ""), ("Conv16", 50, "c1"), (None, 0, ""),
                     ("FC120", 46, "c3"), ("FC84", 40, "c3"), ("10", 28, "c2")], 4)
    b.append(T(12, 76, "AlexNet (2012) · ReLU, Dropout · 60M", "t1", "start"))
    _row(b, 12, 84, [("C11×11", 48, "c1"), (None, 0, ""), ("C5×5", 38, "c1"), (None, 0, ""),
                     ("3×C3×3", 50, "c1"), (None, 0, ""), ("2×FC4096", 62, "c3"), ("1000", 36, "c2")], 4)
    b.append(T(12, 136, "VGG-16 (2014) · только 3×3 · 138M", "t1", "start"))
    items = []
    for lab in ["2×C3", "2×C3", "3×C3", "3×C3", "3×C3"]:
        items += [(lab, 36, "c1"), (None, 0, "")]
    items += [("3×FC", 36, "c3"), ("1000", 36, "c2")]
    _row(b, 12, 144, items, 3)
    for k, ch in enumerate(["64", "128", "256", "512", "512"]):
        b.append(T(12 + 18 + k * 50, 186, ch, "mut"))
    b.append(R(12, 203, 8, 14, "c6", rx=1))
    b.append(T(26, 214, "pooling", "mut", "start"))
    b.append(R(92, 203, 14, 14, "c1", rx=2))
    b.append(T(110, 214, "свёртка", "mut", "start"))
    b.append(R(178, 203, 14, 14, "c3", rx=2))
    b.append(T(196, 214, "полносвязный", "mut", "start"))
    return svg(p, 360, 226, "Схемы LeNet-5, AlexNet и VGG-16", b)


def cnnarch_chart(fig):
    ax = fig.add_subplot(111)
    nets = [("AlexNet", 60, 16.4, BLUE, (8, 4)), ("VGG-16/19", 141, 7.3, PURPLE, (-40, 10)),
            ("GoogLeNet", 6.8, 6.7, GREEN, (6, 6)), ("ResNet-50", 25.5, 3.6, RED, (6, -12))]
    for name, prm, err, col, off in nets:
        ax.scatter(prm, err, s=70, color=col, zorder=3)
        ax.annotate(name, (prm, err), xytext=off, textcoords="offset points", fontsize=9, color=col)
    ax.set_xscale("log")
    ax.set_xlim(3, 400), ax.set_ylim(0, 19)
    ax.set_xlabel("параметры, млн (log)"), ax.set_ylabel("top-5 error ImageNet, %")
    ax.set_title("Больше параметров ≠ точнее", fontsize=10)


# ================================================================ cnn_time_series_tcn
def _tcn_svg():
    p = "cnn_time_series_tcn"
    xs = [60 + 18 * t for t in range(16)]
    ys = [200, 150, 100, 50]
    labels = ["вход x", "d = 1", "d = 2", "d = 4"]
    act = [set(range(8, 16)), {15, 14, 13, 12, 11, 10, 9, 8}, {15, 13, 11, 9}, {15, 11}, {15}]
    need = [set(range(8, 16)), {9, 11, 13, 15}, {11, 15}, {15}]
    b = []
    edges = []
    for lvl, d in [(1, 1), (2, 2), (3, 4)]:
        for t in need[lvl]:
            for src in (t, t - d):
                edges.append(L(xs[src], ys[lvl - 1], xs[t], ys[lvl], "ln s1"))
    b += edges
    for lvl, y in enumerate(ys):
        b.append(T(8, y + 4, labels[lvl], "mut", "start"))
        for t, x in enumerate(xs):
            on = t in need[lvl]
            cls = ("c3" if lvl == 3 else "c1") if on else "c6"
            b.append(C(x, y, 5.5 if on else 4, cls))
    _ = act
    b += [L(xs[0], 222, xs[-1], 222, p=p), T(xs[-1], 238, "время t", "mut", "end"),
          T(180, 260, "RF = 1 + (k − 1)·Σd = 1 + 1·(1+2+4) = 8", "t1"),
          T(180, 278, "k = 2; связи только в прошлое (causal)", "mut")]
    return svg(p, 360, 288, "Dilated causal convolutions в TCN", b)


def tcn_chart(fig):
    ax = fig.add_subplot(111)
    Ls = np.arange(1, 9)
    k = 3
    ax.semilogy(Ls, 1 + (k - 1) * (2 ** Ls - 1), "o-", color=BLUE, lw=2, label="dilation 1, 2, 4, …")
    ax.semilogy(Ls, 1 + (k - 1) * Ls, "s-", color=GREY, lw=2, label="без dilation")
    ax.set_xlabel("число слоёв L"), ax.set_ylabel("receptive field (шагов)")
    ax.set_title("Receptive field при k = 3", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ collaborative_filtering_user_item_based
def _cf_svg():
    p = "collaborative_filtering_user_item_based"
    M = ["54·?1", "1·2·4", "5514·", "·1525", "2·415"]
    sub = "₁₂₃₄₅"
    b = []
    for j in range(5):
        b.append(T(70 + 44 * j + 22, 34, f"i{sub[j]}"))
    for k, row in enumerate(M):
        b.append(T(58, 40 + 26 * k + 17, f"u{sub[k]}", "", "end"))
        for j, ch in enumerate(row):
            cls = "c3" if ch == "?" else "c6"
            b.append(B(70 + 44 * j, 40 + 26 * k, 44, 26, cls, ch, tcls="mut" if ch == "·" else "", rx=0))
    b.append(R(68, 92, 224, 26, "s1", rx=3, extra=' stroke-width="2.5"'))
    b.append(R(112, 38, 48, 134, "s2", rx=3, extra=' stroke-width="2.5"'))
    b += [T(20, 194, "user-based: похожий пользователь u₃", "t1", "start"),
          T(20, 212, "item-based: похожий объект i₂", "t2", "start"),
          T(180, 240, "r̂(u,i) = Σᵥ sim(u,v)·r(v,i) / Σᵥ |sim(u,v)|"),
          T(180, 260, "· — нет оценки, ? — прогнозируем r(u₁, i₄)", "mut")]
    return svg(p, 360, 270, "User-based и item-based CF на матрице рейтингов", b)


# ================================================================ community_detection
def _community_svg():
    p = "community_detection"
    A = [(60, 60), (130, 62), (150, 120), (95, 150), (45, 120)]
    Bn = [(220, 70), (300, 58), (320, 120), (270, 155), (210, 128)]
    eA = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    eB = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    m = len(eA) + len(eB) + 1
    # модулярность: Q = Σ_c [L_c/m − (d_c/2m)²]
    dA = 2 * len(eA) + 1
    dB = 2 * len(eB) + 1
    Q = len(eA) / m - (dA / (2 * m)) ** 2 + len(eB) / m - (dB / (2 * m)) ** 2
    b = []
    for i, j in eA:
        b.append(L(*A[i], *A[j], "ln s1"))
    for i, j in eB:
        b.append(L(*Bn[i], *Bn[j], "ln s3"))
    b.append(L(*A[2], *Bn[4], "ln s4", dash=True, sw=2.5))
    for x, y in A:
        b.append(C(x, y, 9, "c1"))
    for x, y in Bn:
        b.append(C(x, y, 9, "c3"))
    b += [T(180, 112, "мост", "t4"),
          T(180, 184, "max betweenness → удалить (Girvan-Newman)", "t4"),
          T(180, 212, "Q = (1/2m) Σᵢⱼ [Aᵢⱼ − kᵢkⱼ/2m] δ(cᵢ, cⱼ)"),
          T(180, 234, f"здесь m = {m}, Q ≈ {Q:.2f} (> 0.3 — структура есть)", "mut")]
    return svg(p, 360, 246, "Два сообщества и мост между ними", b)


# ================================================================ comparison_neural_classical
def _cmp_svg():
    p = "comparison_neural_classical"
    rows = [(("Таблицы",), "c2", ("Бустинг: CatBoost,", "LightGBM, XGBoost")),
            (("Изображения,", "текст, звук"), "c1", ("Нейросети:", "CNN, ViT, трансформеры")),
            (("Таблица + текст,", "огромные данные"), "c5", ("Гибрид / NN", "+ эмбеддинги категорий"))]
    b = [T(180, 22, "Тип данных → семейство моделей", "t1")]
    for k, (left, cls, right) in enumerate(rows):
        y = 36 + 54 * k
        b.append(B(10, y, 150, 40, "c6", *left))
        b.append(L(160, y + 20, 186, y + 20, p=p))
        b.append(B(188, y, 162, 40, cls, *right))
    b.append(B(10, 206, 340, 40, "c3", "Сначала сильный простой бейзлайн;", "сложная модель — если выигрыш на CV"))
    return svg(p, 360, 256, "Выбор между нейросетями и классикой", b)


def cmp_chart(fig):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 200))
    f = lambda t: 1.0 * (t > 3) + 1.5 * (t > 6.5) - 0.8 * (t > 8.5)
    y = f(x) + rng.normal(0, 0.12, x.size)
    g = np.linspace(0, 10, 500)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, random_state=0).fit(x[:, None], y)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=0.01, max_iter=3000,
                       random_state=0).fit(x[:, None] / 5 - 1, y)
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=6, color=GREY, alpha=0.5)
    ax.plot(g, f(g), color="black", lw=1, ls=":", label="истина")
    ax.plot(g, gb.predict(g[:, None]), color=GREEN, lw=2, label="бустинг")
    ax.plot(g, mlp.predict(g[:, None] / 5 - 1), color=BLUE, lw=2, label="MLP")
    ax.set_title("Пороговая зависимость: деревья точнее\nловят ступеньки, MLP сглаживает", fontsize=9.5)
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper left")


# ================================================================ computational_chemistry_ml
def _chem_svg():
    p = "computational_chemistry_ml"
    atoms = {"C1": (110, 88, "C", "c6"), "C2": (190, 88, "C", "c6"),
             "O3": (245, 42, "O", "c4"), "O4": (245, 134, "O", "c4")}
    b = [T(180, 18, "SMILES: CC(=O)O → граф", "t1")]
    x1, y1, *_ = atoms["C1"]
    x2, y2, *_ = atoms["C2"]
    b.append(L(x1, y1, x2, y2))
    xo, yo, *_ = atoms["O3"]
    dx, dy = xo - x2, yo - y2
    ln_ = math.hypot(dx, dy)
    px, py = -dy / ln_ * 3, dx / ln_ * 3
    b.append(L(x2 + px, y2 + py, xo + px, yo + py))
    b.append(L(x2 - px, y2 - py, xo - px, yo - py))
    xo4, yo4, *_ = atoms["O4"]
    b.append(L(x2, y2, xo4, yo4))
    # сообщения от соседей к C2
    for k in ("C1", "O3", "O4"):
        nx, ny, *_ = atoms[k]
        ux, uy = x2 - nx, y2 - ny
        d = math.hypot(ux, uy)
        ux, uy = ux / d, uy / d
        qx, qy = -uy * 11, ux * 11
        b.append(L(nx + 19 * ux + qx, ny + 19 * uy + qy, x2 - 19 * ux + qx, y2 - 19 * uy + qy, "ln s1", p=p))
    for x, y, s, cls in atoms.values():
        b.append(C(x, y, 13, cls))
        b.append(T(x, y + 4, s))
    b.append(T(262, 139, "H", "mut", "start"))
    b += [T(180, 166, "атом = узел, связь = ребро", "mut"),
          B(40, 176, 280, 44, "c1", "Message passing (T шагов)", "hᵥ ← U(hᵥ, Σᵤ M(hᵥ, hᵤ, eᵥᵤ))"),
          L(180, 220, 180, 236, p=p),
          B(40, 238, 280, 28, "c5", "Readout: ŷ = R({hᵥ | v ∈ G})"),
          L(180, 266, 180, 282, p=p),
          B(40, 284, 280, 28, "c2", "logP, растворимость, токсичность")]
    return svg(p, 360, 322, "Молекула как граф и MPNN", b)


# ================================================================ computational_social_science
def _css_svg():
    p = "computational_social_science"
    b = []
    for k, s in enumerate(["Соцсети", "Мобильные", "Опросы", "Тексты, web"]):
        x = 11 + 86 * k
        b.append(B(x, 12, 80, 28, "c6", s))
        b.append(L(x + 40, 40, 180 + (x + 40 - 180) * 0.4, 62, p=p))
    b += [B(30, 64, 300, 28, "c4", "Этика: согласие, анонимизация"),
          L(70, 92, 63, 114, p=p), L(180, 92, 180, 114, p=p), L(290, 92, 297, 114, p=p)]
    for k, (t, s, cls) in enumerate([("Сети", "центральность", "c1"), ("NLP", "тональность, LDA", "c2"),
                                     ("ABM, прогноз", "симуляции", "c5")]):
        x = 10 + 117 * k
        b.append(R(x, 116, 106, 42, cls))
        b.append(T(x + 53, 133, t))
        b.append(T(x + 53, 150, s, "mut"))
    b += [L(63, 158, 150, 180, p=p), L(180, 158, 180, 180, p=p), L(297, 158, 210, 180, p=p),
          B(40, 182, 280, 28, "c3", "Валидация: bias выборки, ground truth"),
          L(180, 210, 180, 228, p=p),
          B(40, 230, 280, 28, "c2", "Выводы: поведение, тренды, политики")]
    return svg(p, 360, 270, "Пайплайн исследования в CSS", b)


# ================================================================ concept_activation_vectors
def _cav_svg():
    p = "concept_activation_vectors"
    rng = np.random.default_rng(11)
    p0 = np.array([180.0, 102.0])
    nv = np.array([0.8, -0.6])
    dv = np.array([0.6, 0.8])
    b = [R(20, 12, 320, 180, "c6"), T(332, 186, "активации слоя l", "mut", "end")]
    e1, e2 = p0 - 100 * dv, p0 + 100 * dv
    b.append(L(*e1, *e2, "ln s6", dash=True, sw=2))
    for t, s in zip(rng.uniform(35, 90, 12), rng.uniform(-40, 40, 12)):
        q = p0 + t * nv + s * dv
        b.append(C(*q, 4, "t3"))
    for t, s in zip(rng.uniform(-90, -35, 12), rng.uniform(-60, 40, 12)):
        q = p0 + t * nv + s * dv
        b.append(C(*q, 4, "t6"))
    a0 = p0 - 50 * dv
    b.append(L(*a0, *(a0 + 50 * nv), "ln s4", p=p, sw=2.5))
    b.append(T(*(a0 + 50 * nv + [6, 2]), "v_C", "t4", "start"))
    x = p0 - 20 * nv + 50 * dv
    g = np.array([0.96, -0.28]) * 48
    b.append(L(*x, *(x + g), "ln s5", p=p, sw=2))
    b.append(C(*x, 5, "c5"))
    b.append(T(x[0] - 8, x[1] + 16, "x", "t5"))
    b.append(T(*(x + g + [4, 4]), "∇ₕfₖ(x)", "t5", "start"))
    b.append(T(282, 44, "концепт C", "t3"))
    b.append(T(72, 168, "random", "t6"))
    b += [T(180, 214, "CAV v_C = w / ‖w‖ — нормаль к границе"),
          T(180, 234, "S(x) = ∇ₕfₖ(x) · v_C"),
          T(180, 256, "TCAV = доля x ∈ Xₖ, у которых S(x) > 0", "t1")]
    return svg(p, 360, 266, "CAV и производная по направлению", b)


# ================================================================ concept_drift
def _gauss_path(x0, x1, mu, sig, base, hgt, n=40):
    xs = np.linspace(x0, x1, n)
    ys = base - hgt * np.exp(-0.5 * ((xs - mu) / sig) ** 2)
    return "M" + " L".join(f"{_n(a)},{_n(c)}" for a, c in zip(xs, ys))


def _drift_svg():
    p = "concept_drift"
    b = []
    titles = [("Data drift", "P(X) меняется"), ("Concept drift", "P(Y|X) меняется"), ("Label drift", "P(Y) меняется")]
    for k, (t, s) in enumerate(titles):
        x = 8 + 117 * k
        b.append(R(x, 4, 110, 128, "c6"))
        b.append(T(x + 55, 22, t, "t1"))
        b.append(T(x + 55, 38, s, "mut"))
    # data drift: две колоколообразные кривые
    b.append(L(16, 116, 110, 116))
    b.append(P(_gauss_path(16, 110, 48, 11, 116, 58), "ln s6", dash=True, sw=2))
    b.append(P(_gauss_path(16, 110, 78, 11, 116, 58), "ln s1", sw=2))
    # concept drift: те же точки, новая граница
    pts = [(140, 60), (152, 84), (146, 108), (166, 70), (172, 100), (190, 58), (200, 86),
           (194, 112), (214, 68), (220, 100)]
    for x, y in pts:
        b.append(C(x, y, 3.5, "t1" if 60 * (y - 122) + 70 * (x - 162) < 0 else "t4"))
    b.append(L(180, 50, 180, 122, "ln s6", dash=True, sw=2))
    b.append(L(162, 122, 222, 52, "ln s4", sw=2))
    # label drift: доли классов до/после
    for j, (h0, h1) in enumerate([(56, 32), (24, 48)]):
        x = 262 + 42 * j
        b.append(R(x, 116 - h0, 14, h0, "c6", rx=1))
        b.append(R(x + 16, 116 - h1, 14, h1, "c3", rx=1))
        b.append(T(x + 15, 128, f"y={j}", "mut"))
    b += [T(63, 152, "KS-test, PSI"), T(180, 152, "DDM, ADWIN"), T(297, 152, "χ², доли y"),
          T(180, 176, "пунктир — было, сплошное — стало", "mut")]
    return svg(p, 360, 186, "Три вида дрейфа и детекторы", b)


def drift_chart(fig):
    rng = np.random.default_rng(4)
    n, t0 = 1400, 700
    err = np.concatenate([rng.random(t0) < 0.10, rng.random(n - t0) < 0.28]).astype(float)
    pmin, smin = np.inf, np.inf
    P_, W_, D_ = [], [], []
    det = None
    for i in range(1, n + 1):
        pi = err[:i].mean()
        si = math.sqrt(pi * (1 - pi) / i)
        if i >= 100 and pi + si < pmin + smin:
            pmin, smin = pi, si
        P_.append(pi + si)
        W_.append(pmin + 2 * smin)
        D_.append(pmin + 3 * smin)
        if det is None and i >= 100 and pi + si > pmin + 3 * smin:
            det = i
    t = np.arange(1, n + 1)
    roll = np.convolve(err, np.ones(50) / 50, mode="same")
    ax = fig.add_subplot(111)
    ax.plot(t, roll, color=GREY, lw=0.8, alpha=0.6, label="ошибка, окно 50")
    ax.plot(t, P_, color=BLUE, lw=2, label=r"$p_i + s_i$")
    ax.plot(t, W_, color=ORANGE, lw=1.3, ls="--", label="warning: min + 2s")
    ax.plot(t, D_, color=RED, lw=1.3, ls="--", label="drift: min + 3s")
    ax.axvline(t0, color="black", lw=1, ls=":")
    ax.text(t0 + 10, 0.42, "реальный\nдрейф", fontsize=8)
    if det:
        ax.axvline(det, color=RED, lw=1.5)
        ax.text(det + 10, 0.33, f"DDM: t = {det}", fontsize=8, color=RED)
    ax.set_ylim(0, 0.5), ax.set_xlim(100, n)
    ax.set_xlabel("номер примера в потоке"), ax.set_ylabel("доля ошибок")
    ax.set_title("DDM: ошибка 10% → 28%", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


# ================================================================ constrained_optimization_ml
def _constr_svg():
    p = "constrained_optimization_ml"
    O = np.array([110.0, 165.0])
    r = 60
    W = np.array([240.0, 95.0])
    u = (W - O) / np.linalg.norm(W - O)
    xh = O + r * u
    dist = np.linalg.norm(W - xh)
    b = [C(*O, r, "c2")]
    for rr, cls in [(30, "ln s6"), (60, "ln s6"), (dist, "ln s1")]:
        b.append(f'<circle class="{cls}" cx="{_n(W[0])}" cy="{_n(W[1])}" r="{_n(rr)}"/>')
    b.append(T(92, 186, "g(w) ≤ 0", "t2"))
    b.append(C(*W, 4, "t4"))
    b.append(T(W[0] + 8, W[1] + 4, "w*", "t4", "start"))
    b.append(T(W[0] + 8, W[1] + 20, "без огранич.", "mut", "start"))
    b.append(L(*xh, *(xh - 42 * u), "ln s1", p=p, sw=2))
    b.append(L(*xh, *(xh + 40 * u), "ln s2", p=p, sw=2))
    b.append(C(*xh, 5, "c3"))
    b.append(T(xh[0] - 5, xh[1] + 22, "x̂", "t3"))
    gf = xh - 42 * u
    b.append(T(gf[0] - 4, gf[1] - 6, "∇f", "t1", "end"))
    gg = xh + 40 * u
    b.append(T(gg[0] + 2, gg[1] + 16, "∇g", "t2", "start"))
    b += [T(180, 248, "KKT: ∇f(x̂) + μ∇g(x̂) = 0,  μ ≥ 0,  μ·g(x̂) = 0"),
          T(180, 268, "L(x, μ) = f(x) + μ·g(x)", "t1")]
    return svg(p, 360, 278, "Условие KKT: касание линии уровня и границы", b)


def constr_chart(fig):
    A = np.array([[2.0, 0.8], [0.8, 1.0]])
    c = np.array([1.6, 0.3])
    f = lambda w: np.einsum("...i,ij,...j->...", w - c, A, w - c)
    g1, g2 = np.meshgrid(np.linspace(-0.3, 2.0, 200), np.linspace(-0.3, 2.0, 200))
    Z = f(np.stack([g1, g2], -1))
    w = np.array([0.05, 0.1])
    path = [w.copy()]
    for _ in range(25):
        w = np.clip(w - 0.15 * 2 * A @ (w - c), 0, 1)
        path.append(w.copy())
    path = np.array(path)
    ax = fig.add_subplot(111)
    ax.contour(g1, g2, Z, levels=[0.05, 0.2, 0.5, 1, 2, 3.5, 5.5], colors=GREY, linewidths=0.8)
    ax.add_patch(__import__("matplotlib").patches.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.18, lw=0))
    ax.plot(path[:, 0], path[:, 1], "o-", color=BLUE, ms=3.5, lw=1.4, label="projected GD")
    ax.plot(*c, "*", color=RED, ms=13, label="min без ограничений")
    ax.plot(*path[-1], "o", color=ORANGE, ms=9, label="решение на границе")
    ax.set_aspect("equal"), ax.set_xlim(-0.3, 2.0), ax.set_ylim(-0.3, 2.0)
    ax.set_xlabel("w₁"), ax.set_ylabel("w₂")
    ax.set_title("Шаг градиента → проекция на [0, 1]²", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


# ================================================================ context_aware_recommendations
def _ctx_svg():
    p = "context_aware_recommendations"

    def row(y, title, items):
        out = [T(10, y - 8, title, "t1", "start")]
        x = 10
        for k, (lab, w, cls) in enumerate(items):
            out.append(B(x, y, w, 28, cls, lab))
            if k < len(items) - 1:
                out.append(L(x + w, y + 14, x + w + 12, y + 14, p=p))
            x += w + 14
        return out

    b = []
    b += row(24, "Pre-filtering", [("U×I×C", 70, "c6"), ("фильтр по c", 84, "c3"),
                                   ("CF: U×I", 72, "c1"), ("top-N", 70, "c2")])
    b += row(84, "Post-filtering", [("U×I×C", 70, "c6"), ("CF: U×I", 72, "c1"),
                                    ("top-N", 70, "c2"), ("фильтр по c", 84, "c3")])
    b += row(144, "Contextual modeling", [("U×I×C", 70, "c6"), ("f(u, i, c): TF / FM / NN", 170, "c5"),
                                          ("top-N", 70, "c2")])
    b.append(T(180, 196, "c — контекст: время, место, устройство, компания", "mut"))
    return svg(p, 360, 206, "Три способа учесть контекст", b)


# ================================================================ cost_sensitive_learning
def _cost_svg():
    p = "cost_sensitive_learning"
    b = [T(175, 22, "предсказано 0"), T(265, 22, "предсказано 1"),
         T(124, 51, "истинный 0", "", "end"), T(124, 81, "истинный 1", "", "end"),
         B(130, 30, 90, 30, "c2", "0 (TN)", rx=0), B(220, 30, 90, 30, "c3", "C_FP (FP)", rx=0),
         B(130, 60, 90, 30, "c4", "C_FN (FN)", rx=0), B(220, 60, 90, 30, "c2", "0 (TP)", rx=0),
         T(180, 118, "p = P(y = 1 | x). Ожидаемая стоимость:", "mut"),
         T(180, 140, "ответ 1: (1 − p)·C_FP;  ответ 0: p·C_FN"),
         T(180, 166, "→ класс 1, если p > p* = C_FP / (C_FP + C_FN)", "t1"),
         T(180, 188, "C_FP = 1, C_FN = 10 ⇒ p* ≈ 0.09, а не 0.5", "mut")]
    return svg(p, 360, 198, "Матрица стоимости и оптимальный порог", b)


def cost_chart(fig):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=6000, n_features=8, n_informative=4, weights=[0.9],
                               class_sep=0.8, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    pr = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    th = np.linspace(0.01, 0.95, 95)
    cost = [((pr >= t) & (yte == 0)).sum() * 1 + ((pr < t) & (yte == 1)).sum() * 10 for t in th]
    cost = np.array(cost) / len(yte)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(th, cost, color=BLUE, lw=2)
    tb = th[np.argmin(cost)]
    ax.axvline(0.5, color=GREY, ls="--", lw=1.2)
    ax.text(0.52, 0.2 * cost.max() + 0.8 * cost.min(), "порог 0.5", fontsize=8, color=GREY)
    ax.axvline(1 / 11, color=RED, ls="--", lw=1.2)
    ax.text(1 / 11 + 0.02, cost.max() * 0.75, "p* = 1/11", fontsize=8, color=RED)
    ax.plot(tb, cost.min(), "o", color=RED)
    ax.set_xlabel("порог"), ax.set_ylabel("стоимость / объект")
    ax.set_title("C_FN = 10, C_FP = 1: минимум далеко от 0.5", fontsize=9.5)
    ax2 = fig.add_subplot(2, 1, 2)
    pt = np.linspace(0.01, 1, 300)
    for gm, col in [(0, GREY), (1, GREEN), (2, BLUE), (5, PURPLE)]:
        ax2.plot(pt, -(1 - pt) ** gm * np.log(pt), color=col, lw=2, label=f"γ = {gm}")
    ax2.set_ylim(0, 3)
    ax2.set_xlabel(r"$p_t$ (вероятность верного класса)"), ax2.set_ylabel("loss")
    ax2.set_title("Focal loss гасит лёгкие примеры", fontsize=9.5)
    ax2.legend(fontsize=8)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ cotraining
def _cotrain_svg():
    p = "cotraining"
    b = [B(20, 12, 320, 28, "c6", "L — мало меток,  U — много без меток"),
         L(90, 40, 90, 78, p=p), L(270, 40, 270, 78, p=p),
         B(20, 80, 140, 40, "c1", "h₁ на view x¹", "(текст страницы)"),
         B(200, 80, 140, 40, "c5", "h₂ на view x²", "(текст ссылок)"),
         L(90, 120, 90, 158, p=p), L(270, 120, 270, 158, p=p),
         T(84, 144, "размечает U", "mut", "end"), T(276, 144, "размечает U", "mut", "start"),
         B(20, 160, 140, 40, "c1", "L₁: top-k", "уверенных h₁"),
         B(200, 160, 140, 40, "c5", "L₂: top-k", "уверенных h₂"),
         L(160, 166, 206, 122, "ln s1", p=p, sw=2),
         L(200, 166, 154, 122, "ln s5", p=p, sw=2),
         T(180, 224, "каждая модель учится на метках другой", "t1"),
         T(180, 246, "прогноз: среднее P(y|x¹) и P(y|x²)", "mut")]
    return svg(p, 360, 256, "Схема co-training", b)


# ================================================================ credit_scoring
def _credit_svg():
    p = "credit_scoring"
    steps = [("c6", "Заявка + кредитная история (БКИ)"), ("c1", "Бининг признаков"),
             ("c1", "WoE = ln(% good / % bad)"), ("c5", "Логистическая регрессия → PD"),
             ("c3", "Score = Offset + Factor·ln(odds)"), ("c2", "Score ≥ cut-off → одобрить")]
    b = []
    for k, (cls, s) in enumerate(steps):
        y = 12 + 42 * k
        b.append(B(40, y, 280, 28, cls, s))
        if k < len(steps) - 1:
            b.append(L(180, y + 28, 180, y + 40, p=p))
    b.append(T(180, 272, "odds = good : bad; Factor = PDO / ln 2", "mut"))
    return svg(p, 360, 282, "Пайплайн скоркарты", b)


def credit_chart(fig):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    good = rng.normal(640, 55, 3000)
    bad = rng.normal(565, 55, 400)
    auc = roc_auc_score(np.r_[np.zeros(good.size), np.ones(bad.size)], -np.r_[good, bad])
    ax = fig.add_subplot(2, 1, 1)
    bins = np.linspace(380, 820, 45)
    ax.hist(good, bins, density=True, color=GREEN, alpha=0.55, label="good")
    ax.hist(bad, bins, density=True, color=RED, alpha=0.55, label="bad (дефолт)")
    ax.set_title(f"Распределение баллов: AUC = {auc:.2f}, Gini = {2 * auc - 1:.2f}", fontsize=9.5)
    ax.set_yticks([]), ax.legend(fontsize=8)
    ax2 = fig.add_subplot(2, 1, 2)
    g = np.linspace(380, 820, 500)
    Fg = np.searchsorted(np.sort(good), g) / good.size
    Fb = np.searchsorted(np.sort(bad), g) / bad.size
    ax2.plot(g, Fg, color=GREEN, lw=2, label="CDF good")
    ax2.plot(g, Fb, color=RED, lw=2, label="CDF bad")
    i = np.argmax(Fb - Fg)
    ax2.vlines(g[i], Fg[i], Fb[i], color=BLUE, lw=2.5)
    ax2.text(g[i] + 8, (Fg[i] + Fb[i]) / 2, f"KS = {Fb[i] - Fg[i]:.2f}", color=BLUE, fontsize=9)
    ax2.set_xlabel("скоринговый балл"), ax2.set_ylabel("доля ≤ балла")
    ax2.legend(fontsize=8, loc="lower right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ cross_modal_retrieval
def _xmodal_svg():
    p = "cross_modal_retrieval"
    a, pos, neg, neg2 = (232, 100), (262, 120), (318, 44), (322, 162)
    dap = math.hypot(pos[0] - a[0], pos[1] - a[1])
    b = [B(8, 40, 74, 28, "c2", "Текст"), L(82, 54, 96, 54, p=p),
         B(98, 40, 74, 28, "c2", "Text enc."),
         B(8, 120, 74, 28, "c1", "Фото"), L(82, 134, 96, 134, p=p),
         B(98, 120, 74, 28, "c1", "Image enc."),
         R(188, 12, 164, 180, "c6"),
         L(172, 54, 225, 95, p=p), L(172, 134, 254, 122, p=p),
         f'<circle class="ln s6" cx="{a[0]}" cy="{a[1]}" r="{_n(dap + 22)}" stroke-dasharray="4 3"/>',
         L(*a, *pos, "ln s2", sw=2), L(*a, *neg, "ln s4", dash=True, sw=1.5),
         C(*a, 5, "t2"), R(pos[0] - 5, pos[1] - 5, 10, 10, "t1", rx=1),
         R(neg[0] - 5, neg[1] - 5, 10, 10, "t4", rx=1), R(neg2[0] - 5, neg2[1] - 5, 10, 10, "t4", rx=1),
         T(226, 88, "a", "t2", "end"), T(270, 136, "p", "t1", "start"), T(310, 40, "n", "t4", "end"),
         T(296, 72, "d(a,n)", "t4", "start"),
         T(270, 184, "общее пространство", "mut"),
         T(180, 216, "Triplet: L = max(0, d(a,p) − d(a,n) + m)", "t1"),
         T(180, 236, "пунктир: d(a,p) + m — негатив должен быть дальше", "mut")]
    return svg(p, 360, 246, "Общее пространство эмбеддингов и triplet loss", b)


# ================================================================ cure
def _cure_svg():
    p = "cure"
    rng = np.random.default_rng(8)
    alpha = 0.3
    b = ['<ellipse class="c1" cx="115" cy="100" rx="92" ry="44"/>',
         '<circle class="c3" cx="290" cy="100" r="50"/>']
    for _ in range(40):
        t, rr = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 1)) * 0.85
        b.append(C(115 + 92 * rr * np.cos(t), 100 + 44 * rr * np.sin(t), 2, "t6"))
        b.append(C(290 + 50 * rr * np.cos(t), 100 + 50 * rr * np.sin(t), 2, "t6"))
    shrunk = {}
    for name, (cx, cy, ax_, ay_), angs in [("A", (115, 100, 92, 44), [0, 60, 120, 180, 240, 300]),
                                           ("B", (290, 100, 50, 50), [0, 90, 180, 270])]:
        shrunk[name] = []
        for ang in angs:
            t = math.radians(ang)
            px, py = cx + 0.92 * ax_ * math.cos(t), cy + 0.92 * ay_ * math.sin(t)
            qx, qy = px + alpha * (cx - px), py + alpha * (cy - py)
            shrunk[name].append((qx, qy))
            b.append(C(px, py, 4, "s6"))
            if math.hypot(qx - px, qy - py) > 10:
                b.append(L(px, py, qx + (px - qx) * 0.25, qy + (py - qy) * 0.25, "ln", p=p))
            b.append(C(qx, qy, 4, "t1" if name == "A" else "t3"))
        b.append(T(cx, cy + 5, "+"))
    pa = min(shrunk["A"], key=lambda q: q[0])
    pa = max(shrunk["A"], key=lambda q: q[0])
    pb = min(shrunk["B"], key=lambda q: q[0])
    b.append(L(*pa, *pb, "ln s4", dash=True, sw=2))
    b.append(T((pa[0] + pb[0]) / 2, 90, "min", "t4"))
    b += [T(180, 176, "○ — представители на краю, ● — после усадки"),
          T(180, 200, "p′ = p + α·(centroid − p), здесь α = 0.3", "t1"),
          T(180, 220, "dist(C₁, C₂) = min по парам представителей", "mut")]
    return svg(p, 360, 230, "Представители и усадка в CURE", b)


# ================================================================ curriculum_learning
def _curr_svg():
    p = "curriculum_learning"
    seg = ["c2"] * 3 + ["c3"] * 4 + ["c4"] * 3
    b = [T(180, 18, "Примеры по сложности (scorer)")]
    for k, c in enumerate(seg):
        b.append(R(90 + 24 * k, 26, 24, 18, c, rx=0))
    b += [T(90, 58, "лёгкие", "mut", "start"), T(330, 58, "сложные", "mut", "end")]
    for k, (name, nseg) in enumerate([("Этап 1", 3), ("Этап 2", 7), ("Этап 3", 10)]):
        y = 70 + 30 * k
        b.append(T(12, y + 15, name, "", "start"))
        for j in range(nseg):
            b.append(R(90 + 24 * j, y, 24, 20, seg[j], rx=0))
    b += [L(70, 72, 70, 128, p=p),
          T(180, 180, "Self-paced: vᵢ = 1, если lossᵢ < λ", "t1"),
          T(180, 200, "λ растёт → входят всё более сложные", "mut")]
    return svg(p, 360, 210, "Curriculum: от лёгких примеров к сложным", b)


def curr_chart(fig):
    ax = fig.add_subplot(111)
    e = np.arange(0, 101)
    T_, c0 = 60, 0.1
    baby = np.minimum((e // 10 + 1) / 10, 1)
    lin = np.minimum(1, c0 + (1 - c0) * e / T_)
    root = np.minimum(1, np.sqrt(c0 ** 2 + (1 - c0 ** 2) * e / T_))
    ax.step(e, baby, where="post", color=ORANGE, lw=2, label="baby steps (10 этапов)")
    ax.plot(e, lin, color=BLUE, lw=2, label="linear")
    ax.plot(e, root, color=GREEN, lw=2, label="root: быстрее в начале")
    ax.set_xlabel("эпоха"), ax.set_ylabel("доля данных в обучении")
    ax.set_ylim(0, 1.05)
    ax.set_title("Pacing function: сколько примеров доступно", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


# ================================================================ data_preprocessing_missing_encoding
def _prep_svg():
    p = "data_preprocessing_missing_encoding"
    b = [B(110, 10, 140, 28, "c6", "DataFrame X"),
         L(150, 38, 92, 62, p=p), L(210, 38, 268, 62, p=p),
         T(92, 76, "числовые", "t1"), T(268, 76, "категориальные", "t3"),
         B(12, 84, 160, 40, "c1", "SimpleImputer(median)", "+ индикатор пропуска"),
         B(188, 84, 160, 40, "c3", "SimpleImputer", "(most_frequent)"),
         L(92, 124, 92, 146, p=p), L(268, 124, 268, 146, p=p),
         B(12, 148, 160, 40, "c1", "StandardScaler"),
         B(188, 148, 160, 40, "c3", "OneHotEncoder", "(handle_unknown=ignore)"),
         L(92, 188, 150, 212, p=p), L(268, 188, 210, 212, p=p),
         B(70, 214, 220, 28, "c5", "ColumnTransformer → модель"),
         T(180, 264, "fit только на train — без утечки", "mut")]
    return svg(p, 360, 274, "Pipeline: заполнение пропусков и кодирование", b)


def prep_chart(fig):
    from sklearn.impute import KNNImputer
    rng = np.random.default_rng(2)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = 0.8 * x1 + rng.normal(0, 0.5, n)
    miss = rng.random(n) < 1 / (1 + np.exp(-(1.5 * x1 - 1)))  # MAR: зависит от x1
    X = np.c_[x1, np.where(miss, np.nan, x2)]
    mean_fill = np.nanmean(X[:, 1])
    knn = KNNImputer(n_neighbors=5).fit_transform(X)[:, 1]
    ax = fig.add_subplot(111)
    ax.scatter(x1[~miss], x2[~miss], s=8, color=GREY, alpha=0.6, label="наблюдаемые")
    ax.scatter(x1[miss], np.full(miss.sum(), mean_fill), s=12, color=RED, label="mean imputation")
    ax.scatter(x1[miss], knn[miss], s=12, color=BLUE, marker="^", label="KNNImputer")
    ax.set_xlabel("x₁ (без пропусков)"), ax.set_ylabel("x₂ (пропуски MAR)")
    ax.set_title(f"Пропущено {miss.mean():.0%} x₂, чаще при большом x₁", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")


# ================================================================ data_visualization_ml
def _dvis_svg():
    p = "data_visualization_ml"
    rows = [("Распределение признака", "hist, KDE, boxplot"), ("Связь признаков", "scatter, heatmap corr"),
            ("Ошибки классификатора", "confusion, ROC, PR"), ("Переобучение?", "learning curves"),
            ("Качество регрессии", "residual plot"), ("Структура, кластеры", "PCA / t-SNE")]
    cls = ["c1", "c1", "c3", "c3", "c2", "c5"]
    b = [T(95, 16, "вопрос", "mut"), T(270, 16, "график", "mut")]
    for k, (q, g) in enumerate(rows):
        y = 24 + 38 * k
        b.append(B(12, y, 166, 28, "c6", q))
        b.append(L(178, y + 14, 192, y + 14, p=p))
        b.append(B(194, y, 154, 28, cls[k], g))
    return svg(p, 360, 252, "Какой график для какого вопроса", b)


def dvis_chart(fig):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=3000, n_features=10, n_informative=4, weights=[0.9],
                               class_sep=0.9, random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    s = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, s)
    pr, rc, _ = precision_recall_curve(yte, s)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="случайная")
    ax.set_xlabel("FPR"), ax.set_ylabel("TPR"), ax.set_title("ROC-кривая", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(rc, pr, color=ORANGE, lw=2, label=f"AP = {average_precision_score(yte, s):.2f}")
    ax2.axhline(yte.mean(), color=GREY, ls="--", lw=1, label=f"доля класса 1 = {yte.mean():.2f}")
    ax2.set_xlabel("Recall"), ax2.set_ylabel("Precision"), ax2.set_ylim(0, 1.02)
    ax2.set_title("PR-кривая: честнее при дисбалансе", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ date_time_features
def _dt_svg():
    p = "date_time_features"
    cx, cy, r = 115, 120, 82
    b = [C(cx, cy, r, "s6"), L(cx - r - 6, cy, cx + r + 6, cy, "ln s6", dash=True),
         L(cx, cy - r - 6, cx, cy + r + 6, "ln s6", dash=True)]
    pos = {}
    for m in range(1, 13):
        t = 2 * math.pi * m / 12
        x, y = cx + r * math.cos(t), cy - r * math.sin(t)
        pos[m] = (x, y)
        lx, ly = cx + (r + 15) * math.cos(t), cy - (r + 15) * math.sin(t)
        b.append(T(lx, ly + 4, str(m), "t3" if m in (12, 1) else "mut"))
    b.append(L(*pos[12], *pos[1], "ln s3", sw=3))
    for m, (x, y) in pos.items():
        b.append(C(x, y, 5 if m in (12, 1) else 4, "c3" if m in (12, 1) else "c1"))
    b += [T(226, 60, "month_sin:", "", "start"), T(226, 78, "sin(2π·m/12)", "t1", "start"),
          T(226, 106, "month_cos:", "", "start"), T(226, 124, "cos(2π·m/12)", "t1", "start"),
          T(226, 160, "12 и 1 — соседи", "t3", "start")]
    b.append(L(30, 240, 330, 240))
    for m in range(1, 13):
        x = 30 + (m - 1) * 300 / 11
        b.append(L(x, 236, x, 244))
        if m in (1, 12):
            b.append(T(x, 258, str(m), "t3"))
    b.append(T(180, 258, "на прямой: |12 − 1| = 11", "mut"))
    return svg(p, 360, 268, "Циклическое кодирование месяца", b)


def dt_chart(fig):
    ax = fig.add_subplot(111)
    h = np.arange(24)
    raw = np.abs(h - 0)
    cyc = np.hypot(np.sin(2 * np.pi * h / 24) - 0, np.cos(2 * np.pi * h / 24) - 1)
    ax.plot(h, raw / raw.max(), "o-", color=GREY, ms=4, lw=1.5, label="|h − 0| (сырое)")
    ax.plot(h, cyc / cyc.max(), "o-", color=BLUE, ms=4, lw=2, label="евклидово в (sin, cos)")
    ax.annotate("23:00 близко к 0:00", (23, cyc[23] / cyc.max()), (7.5, 0.04), fontsize=8.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_xlabel("час h"), ax.set_ylabel("расстояние до 0:00 (норм.)")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_title("Сырой час vs циклическое кодирование", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


# ================================================================ dcgan_conditional_gan
def _gan_svg():
    p = "dcgan_conditional_gan"
    b = [B(20, 14, 120, 28, "c6", "z ~ N(0, I)"),
         B(220, 14, 120, 28, "c3", "y — класс"),
         L(80, 42, 120, 68, p=p), L(270, 42, 240, 68, p=p),
         B(60, 70, 240, 44, "c1", "Generator G", "ConvT ↑ + BN + ReLU → Tanh"),
         L(150, 114, 100, 134, p=p),
         B(40, 136, 110, 28, "c4", "G(z|y)"),
         B(210, 136, 110, 28, "c2", "x ~ данные"),
         L(95, 164, 140, 194, p=p), L(265, 164, 220, 194, p=p),
         B(60, 196, 240, 44, "c5", "Discriminator D(x|y)", "Conv stride 2 + LeakyReLU(0.2)"),
         P("M340,28 H352 V218 H302", "ln s3", p=p, dash=True),
         L(180, 240, 180, 258, p=p),
         B(110, 260, 140, 28, "c6", "P(реальное) ∈ [0, 1]"),
         T(180, 310, "min_G max_D V(D, G) =", "t1"),
         T(180, 328, "E log D(x|y) + E log(1 − D(G(z|y)|y))")]
    return svg(p, 360, 338, "Conditional DCGAN", b)


# ================================================================ decision_theory
def _dec_svg():
    p = "decision_theory"
    cols = [(100, 65, ("дождь", "P = 0.3")), (165, 65, ("сухо", "P = 0.7")),
            (230, 60, ("R(a)",)), (290, 60, ("max L",))]
    b = [B(10, 14, 90, 36, "c6", "L(θ, a)", rx=0)]
    for x, w, lab in cols:
        b.append(B(x, 14, w, 36, "c6", *lab, rx=0))
    rows = [("взять зонт", ["2", "1", "1.3", "2"]), ("без зонта", ["10", "0", "3.0", "10"])]
    for k, (a, vals) in enumerate(rows):
        y = 50 + 30 * k
        b.append(B(10, y, 90, 30, "c6", a, rx=0))
        for (x, w, _), v in zip(cols, vals):
            best = k == 0 and x >= 230
            b.append(B(x, y, w, 30, "c2" if best else "c6", v, rx=0))
    b += [T(180, 134, "R(a) = Σθ P(θ)·L(θ, a) → min по a", "t1"),
          T(180, 156, "Байес: 0.3·2 + 0.7·1 = 1.3 < 3.0 → зонт"),
          T(180, 176, "Минимакс: min(max L) = 2 → зонт", "mut"),
          T(180, 194, "Лаплас (P равные): 1.5 < 5 → зонт", "mut")]
    return svg(p, 360, 204, "Таблица потерь и критерии выбора", b)


def dec_chart(fig):
    ax = fig.add_subplot(111)
    pp = np.linspace(0, 0.5, 200)
    r1, r2 = 2 * pp + 1 * (1 - pp), 10 * pp
    ax.plot(pp, r1, color=GREEN, lw=2, label="R(зонт) = 1 + p")
    ax.plot(pp, r2, color=RED, lw=2, label="R(без зонта) = 10p")
    ax.fill_between(pp, 0, np.minimum(r1, r2), color=BLUE, alpha=0.12, label="байесовский риск")
    ax.axvline(1 / 9, color=GREY, ls="--", lw=1.2)
    ax.text(1 / 9 + 0.01, 3.6, "p = 1/9", fontsize=9)
    ax.axvline(0.3, color=BLUE, ls=":", lw=1.2)
    ax.text(0.31, 0.25, "p = 0.3", fontsize=9, color=BLUE)
    ax.set_xlabel("P(дождь) = p"), ax.set_ylabel("ожидаемые потери R(a)")
    ax.set_ylim(0, 4)
    ax.set_title("Оптимальное действие меняется при p = 1/9", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


# ================================================================ deep_learning_key_breakthroughs
def _dl_svg():
    p = "deep_learning_key_breakthroughs"
    ev = [("1943", "нейрон McCulloch–Pitts", "c6"), ("1958", "перцептрон Розенблатта", "c6"),
          ("1969", "«Перцептроны» → AI-зима", "c4"), ("1986", "backpropagation", "c1"),
          ("1997", "LSTM", "c1"), ("1998", "LeNet-5", "c1"), ("2012", "AlexNet + GPU, ImageNet", "c3"),
          ("2014", "GAN, VGG, Inception", "c1"), ("2015", "ResNet, BatchNorm", "c3"),
          ("2017", "Transformer: attention", "c3"), ("2018–20", "BERT, GPT-3, ViT", "c1"),
          ("2020+", "LLM, diffusion, CLIP", "c5")]
    b = [L(80, 14, 80, 22 * len(ev) + 10, "ln s6", sw=2)]
    for k, (yr, s, cls) in enumerate(ev):
        y = 22 + 22 * k
        b.append(C(80, y, 6, cls))
        b.append(T(68, y + 4, yr, "mut", "end"))
        b.append(T(94, y + 4, s, "t4" if cls == "c4" else "", "start"))
    b.append(T(94, 22 * len(ev) + 30, "● оранжевые — переломные моменты", "t3", "start"))
    return svg(p, 360, 22 * len(ev) + 40, "Хронология прорывов DL", b)


def dl_chart(fig):
    ax = fig.add_subplot(111)
    yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    err = [28.2, 25.8, 16.4, 11.7, 6.7, 3.57, 2.99, 2.25]
    cols = [GREY, GREY, ORANGE, BLUE, BLUE, ORANGE, BLUE, BLUE]
    ax.bar(yrs, err, color=cols, width=0.7)
    for x, e in zip(yrs, err):
        ax.text(x, e + 0.6, f"{e:g}", ha="center", fontsize=8)
    ax.axhline(5.1, color=RED, ls="--", lw=1.3)
    ax.text(2015.6, 6.0, "человек ≈ 5.1%", fontsize=8.5, color=RED)
    ax.text(2012, 19.5, "AlexNet", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(2015, 7.6, "ResNet", ha="center", fontsize=8.5, color=ORANGE)
    ax.set_xticks(yrs), ax.set_xticklabels([str(y)[2:] for y in yrs])
    ax.set_xlabel("год ILSVRC (20xx)"), ax.set_ylabel("top-5 error, %")
    ax.set_ylim(0, 32)
    ax.set_title("ImageNet: ошибка победителя по годам", fontsize=10)


# ================================================================ VISUALS
VISUALS = {
    "capsule_networks": [
        {"title": "Dynamic routing by agreement", "svg": _capsule_svg(),
         "caption": "Каждая капсула uᵢ делает прогноз ûⱼ|ᵢ для верхней капсулы; сумма sⱼ взвешена cᵢⱼ = softmax(bᵢⱼ). "
                    "Если прогноз согласуется с выходом vⱼ (большое скалярное произведение), bᵢⱼ растёт — так r итераций."},
        {"title": "Squash и пороги margin loss", "chart": capsule_chart,
         "caption": "Squash сжимает длину вектора в [0, 1), сохраняя направление. Margin loss требует ‖vₖ‖ > 0.9 "
                    "для присутствующего класса и ‖vₖ‖ < 0.1 для отсутствующих."},
    ],
    "categorical_encoding": [
        {"title": "Один столбец — три кодировки", "svg": _catenc_svg(),
         "caption": "LabelEncoder нумерует по алфавиту и вводит ложный порядок (red > green > blue). One-Hot даёт "
                    "K бинарных колонок; Target encoding заменяет категорию средним y — нужна регуляризация и out-of-fold."},
    ],
    "choosing_clusters_number": [
        {"title": "Как выбирать K", "svg": _choosek_svg(),
         "caption": "Ни один метод не даёт истину: перебирают K, смотрят несколько критериев (у DB лучше меньше, у "
                    "остальных — больше) и выбирают K, где они согласны и кластеры имеют смысл для задачи."},
        {"title": "Локоть и silhouette на данных с 4 кластерами", "chart": choosek_chart,
         "caption": "Inertia падает всегда, но после K = 4 — медленно: это «локоть». Silhouette здесь же достигает "
                    "максимума — два метода сходятся."},
    ],
    "cicd_ml": [
        {"title": "Пайплайн CI/CD/CT", "svg": _cicd_svg(),
         "caption": "CI проверяет код и данные, обучает и пропускает модель через quality gate; не прошла порог — "
                    "пайплайн падает. CD выкатывает модель из registry, а мониторинг и drift запускают Continuous Training."},
    ],
    "clarans": [
        {"title": "Рандомизированный поиск CLARANS", "svg": _clarans_svg(),
         "caption": "Узел поиска — набор k медоидов, сосед отличается заменой одного медоида. Проверив maxneighbor "
                    "соседей без улучшения TC, алгоритм фиксирует локальный минимум; из numlocal запусков берётся лучший."},
    ],
    "classical_rl_tabular": [
        {"title": "Цикл агент–среда и Q-learning", "svg": _rl_svg(),
         "caption": "Агент выбирает aₜ по Q-таблице, среда возвращает sₜ₊₁ и rₜ₊₁. Q-learning обновляет Q к цели "
                    "r + γ·max Q(s′,·) независимо от реально выбранного действия (off-policy), SARSA — по выбранному a′."},
        {"title": "Value iteration в GridWorld 5×5", "chart": rl_chart,
         "caption": "Награда +1 за вход в G и −1 за вход в яму, γ = 0.9: V(s) убывает как 0.9^(d−1) с расстоянием до цели. "
                    "Жадная стратегия обходит яму стороной."},
    ],
    "clip_multimodal_models": [
        {"title": "Contrastive pre-training в CLIP", "svg": _clip_svg(),
         "caption": "В батче из N пар считается матрица N×N косинусных сходств нормированных эмбеддингов. "
                    "Loss тянет вверх диагональ (верные пары) и вниз остальные N² − N ячеек — в обе стороны."},
    ],
    "cluster_visualization": [
        {"title": "Путь от многомерных данных к картинке", "svg": _clvis_svg(),
         "caption": "Признаки масштабируют, затем понижают размерность: PCA сохраняет глобальную геометрию, "
                    "t-SNE/UMAP — соседство. Цвет точек — метки кластеризации, обученной в исходном пространстве."},
        {"title": "Одни и те же кластеры: PCA и t-SNE", "chart": clvis_chart,
         "caption": "4 кластера в 10D, метки KMeans. PCA — линейная проекция, центроиды можно проецировать тем же "
                    "преобразованием; t-SNE разделяет группы чётче, но размеры и расстояния на нём условны."},
    ],
    "clustering_metrics": [
        {"title": "Silhouette для одной точки", "svg": _clmetrics_svg(),
         "caption": "a(i) — средняя длина зелёных отрезков (свой кластер), b(i) — оранжевых (ближайший чужой). "
                    "Если точка ближе к чужому кластеру, b < a и s(i) < 0."},
        {"title": "Silhouette plot", "chart": clmetrics_chart,
         "caption": "Каждая полоса — s(i) одной точки, внутри кластера по убыванию; красная линия — среднее. "
                    "У широкого кластера значения ниже, а точки у границы с соседом близки к нулю."},
    ],
    "cnn_architectures": [
        {"title": "LeNet → AlexNet → VGG", "svg": _cnnarch_svg(),
         "caption": "Схема одна — свёртки + pooling, затем полносвязные слои, — но растут глубина и ширина. "
                    "VGG заменяет большие ядра стопками 3×3 и удваивает число каналов после каждого pooling."},
        {"title": "Параметры против ошибки на ImageNet", "chart": cnnarch_chart,
         "caption": "Данные из таблицы сравнения: GoogLeNet с 6.8M параметров точнее VGG со 138M. "
                    "Основные параметры VGG и AlexNet — в полносвязных слоях."},
    ],
    "cnn_time_series_tcn": [
        {"title": "Dilated causal convolutions", "svg": _tcn_svg(),
         "caption": "Выход в момент t зависит только от прошлого. С kernel_size = 2 и dilation 1, 2, 4 три слоя "
                    "видят 8 шагов: receptive field растёт экспоненциально по числу слоёв."},
        {"title": "Receptive field от числа слоёв", "chart": tcn_chart,
         "caption": "При удвоении dilation в каждом слое RF = 1 + (k−1)(2ᴸ − 1), без dilation — лишь 1 + (k−1)L. "
                    "8 слоёв с k = 3: 511 шагов против 17."},
    ],
    "collaborative_filtering_user_item_based": [
        {"title": "Матрица рейтингов: строки и столбцы", "svg": _cf_svg(),
         "caption": "Чтобы предсказать r(u₁, i₄), user-based CF ищет похожих пользователей (строка u₃ похожа на u₁) "
                    "и берёт их оценки i₄; item-based — похожие на i₄ объекты (i₂) и оценки самого u₁."},
    ],
    "community_detection": [
        {"title": "Сообщества и модулярность", "svg": _community_svg(),
         "caption": "Внутри групп рёбра плотные, между ними — одно. Через мост проходят все кратчайшие пути между "
                    "группами, поэтому Girvan-Newman удаляет его первым; Q сравнивает долю внутренних рёбер со случайным графом."},
    ],
    "comparison_neural_classical": [
        {"title": "Эмпирическое правило выбора", "svg": _cmp_svg(),
         "caption": "Таблицы → бустинг, неструктурированные данные → нейросети, смешанные и очень большие данные — "
                    "гибриды. Любая сложная модель сравнивается с сильным простым бейзлайном."},
        {"title": "Ступенчатая зависимость: бустинг vs MLP", "chart": cmp_chart,
         "caption": "Деревья естественно моделируют пороги, MLP смещён к гладким функциям и размывает ступеньки — "
                    "одна из причин, почему бустинг сильнее на таблицах."},
    ],
    "computational_chemistry_ml": [
        {"title": "Молекула → граф → свойство", "svg": _chem_svg(),
         "caption": "Уксусная кислота CC(=O)O как граф: атомы — узлы, связи — рёбра (двойная C=O). MPNN T раз "
                    "собирает сообщения соседей (синие стрелки для центрального C), readout агрегирует узлы в прогноз свойства."},
    ],
    "computational_social_science": [
        {"title": "Пайплайн CSS-исследования", "svg": _css_svg(),
         "caption": "Цифровые следы проходят этический фильтр до анализа. Сети, NLP и агентное моделирование "
                    "дополняют друг друга, а выводы проверяются на смещение выборки и внешние данные."},
    ],
    "concept_activation_vectors": [
        {"title": "CAV в пространстве активаций", "svg": _cav_svg(),
         "caption": "Линейный классификатор отделяет активации примеров концепта от случайных; CAV — нормаль к "
                    "границе. Если градиент логита класса k направлен «по» CAV (S > 0), концепт повышает прогноз."},
    ],
    "concept_drift": [
        {"title": "Data, concept и label drift", "svg": _drift_svg(),
         "caption": "Data drift двигает распределение признаков, concept drift меняет правило y(x) при тех же X, "
                    "label drift — доли классов. Каждый ловят своим инструментом."},
        {"title": "DDM на потоке ошибок", "chart": drift_chart,
         "caption": "DDM запоминает минимум p + s и поднимает тревогу, когда p + s превышает min + 3s. "
                    "Детекция наступает с задержкой: нужно накопить ошибки после дрейфа."},
    ],
    "constrained_optimization_ml": [
        {"title": "Геометрия KKT", "svg": _constr_svg(),
         "caption": "Минимум f без ограничений w* вне допустимой области, поэтому решение x̂ на границе: там линия "
                    "уровня f касается g(w) = 0, а ∇f и ∇g противоположны — ∇f + μ∇g = 0 при μ > 0."},
        {"title": "Projected gradient descent", "chart": constr_chart,
         "caption": "Каждый шаг — обычный градиентный шаг и обрезка (проекция) на [0, 1]². Траектория упирается "
                    "в границу и скользит по ней к точке касания с линией уровня."},
    ],
    "context_aware_recommendations": [
        {"title": "Pre-, post-filtering и contextual modeling", "svg": _ctx_svg(),
         "caption": "Контекст можно учесть до модели (отфильтровать данные), после (переранжировать top-N) или "
                    "внутри — моделируя тензор U×I×C через TF, FM или нейросеть."},
    ],
    "cost_sensitive_learning": [
        {"title": "Cost matrix → порог решения", "svg": _cost_svg(),
         "caption": "Выбираем ответ с меньшей ожидаемой стоимостью. Если пропуск позитива в 10 раз дороже ложной "
                    "тревоги, оптимальный порог p* = 1/11, а не 0.5."},
        {"title": "Threshold moving и focal loss", "chart": cost_chart,
         "caption": "Сверху — реальная стоимость логрегрессии на данных 90/10: минимум рядом с теоретическим p*. "
                    "Снизу — focal loss: с ростом γ уверенно классифицированные примеры почти не дают вклада."},
    ],
    "cotraining": [
        {"title": "Две модели учат друг друга", "svg": _cotrain_svg(),
         "caption": "h₁ и h₂ обучены на разных views. Уверенные метки h₁ (L₁) пополняют обучающую выборку h₂, "
                    "и наоборот — крест-накрест; цикл повторяется, пока U не опустеет."},
    ],
    "credit_scoring": [
        {"title": "Классическая скоркарта", "svg": _credit_svg(),
         "caption": "Признаки бинуются и заменяются WoE, логрегрессия даёт вероятность дефолта PD, а log-odds "
                    "линейно переводятся в баллы (PDO — сколько баллов удваивает шансы)."},
        {"title": "Разделение good/bad: Gini и KS", "chart": credit_chart,
         "caption": "Сверху — распределения баллов хороших и дефолтных клиентов. KS — максимальный разрыв "
                    "кумулятивных распределений; синтетические данные."},
    ],
    "cross_modal_retrieval": [
        {"title": "Общее embedding-пространство", "svg": _xmodal_svg(),
         "caption": "Два энкодера переводят текст и фото в одно пространство. Triplet loss тянет позитив p к запросу a "
                    "и отталкивает негатив n, пока тот не окажется дальше d(a,p) на margin m."},
    ],
    "cure": [
        {"title": "Несколько представителей и усадка", "svg": _cure_svg(),
         "caption": "Кластер описывается c удалёнными точками (○), сдвинутыми к центроиду на долю α (●). "
                    "Так CURE держит вытянутую форму, а расстояние между кластерами — минимум по парам представителей."},
    ],
    "curriculum_learning": [
        {"title": "Расписание от лёгкого к сложному", "svg": _curr_svg(),
         "caption": "Scorer упорядочивает примеры по сложности, и на каждом этапе в обучение добавляется "
                    "следующая порция. В self-paced модель сама берёт примеры с loss < λ, а λ постепенно растёт."},
        {"title": "Pacing functions", "chart": curr_chart,
         "caption": "Baby steps добавляет данные ступенями, linear и root — плавно; root быстрее расширяет "
                    "выборку в начале, когда модель учится быстрее всего."},
    ],
    "data_preprocessing_missing_encoding": [
        {"title": "ColumnTransformer: пропуски и кодирование", "svg": _prep_svg(),
         "caption": "Числовые и категориальные столбцы обрабатываются разными ветками, индикатор сохраняет сам факт "
                    "пропуска. Все шаги обучаются только на train внутри Pipeline."},
        {"title": "Mean imputation vs KNNImputer", "chart": prep_chart,
         "caption": "Пропуски x₂ чаще при больших x₁ (MAR). Заполнение средним кладёт все точки на одну линию "
                    "и разрушает связь x₁–x₂; KNNImputer восстанавливает значения по похожим строкам."},
    ],
    "data_visualization_ml": [
        {"title": "Какой график для какого вопроса", "svg": _dvis_svg(),
         "caption": "Тип графика выбирают по вопросу к данным или модели, а не наоборот."},
        {"title": "ROC и PR на несбалансированных данных", "chart": dvis_chart,
         "caption": "При 10% позитивов ROC-AUC выглядит высоким, а PR-кривая показывает реальную цену высокого "
                    "recall: базовая линия PR — доля класса 1, а не диагональ."},
    ],
    "date_time_features": [
        {"title": "Месяц на окружности", "svg": _dt_svg(),
         "caption": "Пара (sin, cos) кладёт месяцы на окружность: декабрь и январь оказываются соседями, "
                    "тогда как в числовом виде между ними 11."},
        {"title": "Расстояние до полуночи", "chart": dt_chart,
         "caption": "Для сырого часа 23:00 — самый далёкий от 0:00, в (sin, cos) — один из ближайших; "
                    "максимум расстояния приходится на полдень."},
    ],
    "dcgan_conditional_gan": [
        {"title": "Conditional DCGAN", "svg": _gan_svg(),
         "caption": "Метка y подаётся и генератору, и дискриминатору (оранжевый пунктир), поэтому D проверяет "
                    "не просто «реально ли», а «реально ли для класса y». Внутри — правила DCGAN: stride вместо pooling, BN, Tanh на выходе G."},
    ],
    "decision_theory": [
        {"title": "Таблица потерь: брать ли зонт", "svg": _dec_svg(),
         "caption": "Строки — действия, столбцы — состояния природы. Байесовский критерий усредняет потери по P(θ), "
                    "минимакс смотрит на худший случай, Лаплас считает состояния равновероятными."},
        {"title": "Риск действий от вероятности дождя", "chart": dec_chart,
         "caption": "Риск каждого действия линеен по p; нижняя огибающая — байесовский риск. Зонт выгоден уже при "
                    "p > 1/9: потери асимметричны, как и пороги в cost-sensitive классификации."},
    ],
    "deep_learning_key_breakthroughs": [
        {"title": "Хронология", "svg": _dl_svg(),
         "caption": "От модели нейрона до LLM: после «Перцептронов» Минского — зима, затем backprop, CNN и LSTM; "
                    "взлёт начался с AlexNet на GPU и ускорился с ResNet и Transformer."},
        {"title": "ImageNet top-5 error по годам", "chart": dl_chart,
         "caption": "До 2012 года — признаки, собранные вручную; AlexNet снизила ошибку до 16.4%, к 2015 ResNet "
                    "(3.57%) превзошла оценку человека ≈ 5.1%."},
    ],
}
