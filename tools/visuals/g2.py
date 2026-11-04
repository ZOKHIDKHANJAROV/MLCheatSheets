# -*- coding: utf-8 -*-
"""Рисунки для шпаргалок группы g2 (deep_learning_timeline … geoinformation_systems_ml)."""

import numpy as np

B, G, O, R, V, S = "#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"


# ---------------------------------------------------------------- helpers
def _svg(p, w, h, label, body):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">'
            f'<defs><marker id="{p}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
            + "".join(body) + "</svg>")


def T(x, y, s, cls="", a="middle", fs=None, bold=False):
    c = f' class="{cls}"' if cls else ""
    f = f' font-size="{fs}"' if fs else ""
    b = ' font-weight="bold"' if bold else ""
    return f'<text x="{x:g}" y="{y:g}" text-anchor="{a}"{c}{f}{b}>{s}</text>'


def Bx(x, y, w, h, cls, lines=(), fs=None, rx=6, extra=""):
    """Прямоугольник с центрированными строками. Строка — str или (str, cls, fs)."""
    out = [f'<rect class="{cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}"{extra}/>']
    lines = [ln if isinstance(ln, tuple) else (ln, "", fs) for ln in lines]
    n = len(lines)
    lh = (fs or 12) + 3
    y0 = y + h / 2 - (n - 1) * lh / 2 + 4
    for i, (s, c, f) in enumerate(lines):
        out.append(T(x + w / 2, y0 + i * lh, s, c, fs=f or fs))
    return "".join(out)


def A(p, x1, y1, x2, y2, cls="ln", dash=False, head=True):
    d = ' stroke-dasharray="4 3"' if dash else ""
    m = f' marker-end="url(#{p}-a)"' if head else ""
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"{d}{m}/>'


def Pth(p, d, cls="ln", dash=False, head=True, extra=""):
    ds = ' stroke-dasharray="4 3"' if dash else ""
    m = f' marker-end="url(#{p}-a)"' if head else ""
    return f'<path class="{cls}" d="{d}"{ds}{m}{extra}/>'


def poly(xs, ys):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


# ================================================================ 1. deep_learning_timeline
def svg_timeline():
    p = "dlt"
    ev = [("1943", "Нейрон McCulloch–Pitts", "c6"), ("1958", "Перцептрон Розенблатта", "c6"),
          ("1969", "Минский и Пейперт → первая зима AI", "c4"),
          ("1986", "Backprop: Rumelhart, Hinton, Williams", "c1"),
          ("1997", "LSTM (Hochreiter, Schmidhuber)", "c1"),
          ("1998", "LeNet-5 → вторая зима (SVM, RF)", "c4"),
          ("2006", "DBN, послойный pretraining", "c2"),
          ("2012", "AlexNet: ImageNet 26% → 15%", "c3"),
          ("2015", "ResNet (152 слоя), BatchNorm", "c3"),
          ("2017", "Transformer: «Attention is All You Need»", "c5"),
          ("2020", "GPT-3 (175B), few-shot", "c5"),
          ("2022", "ChatGPT, Stable Diffusion", "c5")]
    body = [f'<line class="ln" x1="62" y1="12" x2="62" y2="364"/>']
    for i, (yr, s, c) in enumerate(ev):
        y = 24 + 30 * i
        body += [T(52, y + 4, yr, "mut", "end", 11), f'<circle class="{c}" cx="62" cy="{y}" r="5"/>',
                 T(74, y + 4, s, a="start")]
    return _svg(p, 360, 376, "Хронология глубокого обучения", body)


def chart_timeline(fig):
    ax = fig.add_subplot(111)
    d = [("LeNet-5", 1998, 6e4, (6, -4)), ("AlexNet", 2012, 6e7, (-6, -14)), ("VGG-16", 2014, 1.38e8, (-24, 8)),
         ("ResNet-152", 2015, 6e7, (8, -12)), ("BERT-large", 2018, 3.4e8, (6, -6)),
         ("GPT-2", 2019, 1.5e9, (-32, 4)), ("GPT-3", 2020, 1.75e11, (-34, -3))]
    yrs = [x[1] for x in d]
    ps = [x[2] for x in d]
    ax.semilogy(yrs, ps, "o-", color=B, lw=1.5, ms=5)
    for name, yr, pp, off in d:
        ax.annotate(name, (yr, pp), xytext=off, textcoords="offset points", fontsize=8, ha="left")
    ax.set_xlim(1996, 2025)
    ax.set_ylim(1e4, 2e12)
    ax.set_xlabel("год")
    ax.set_ylabel("число параметров")
    ax.set_title("Рост размера моделей (лог. шкала)", fontsize=10)


# ================================================================ 2. defensive_distillation
def svg_distill():
    p = "ddist"
    body = [
        Bx(50, 10, 260, 28, "c6", ["Обучающие данные (x, y)"]),
        A(p, 180, 38, 180, 58), T(190, 52, "жёсткие метки y", "mut", "start", 10),
        Bx(50, 58, 260, 40, "c1", ["Teacher F", ("softmax(z / T), T = 20", "", 11)]),
        A(p, 180, 98, 180, 118), T(190, 112, "F(x) при той же T", "mut", "start", 10),
        Bx(50, 118, 260, 28, "c3", ["Soft labels: [0.62, 0.25, 0.13]"]),
        A(p, 180, 146, 180, 166), T(190, 160, "loss = KL(soft ‖ student)", "mut", "start", 10),
        Bx(50, 166, 260, 40, "c2", ["Student F_d (та же архитектура)", ("softmax(z / T), T = 20", "", 11)]),
        A(p, 180, 206, 180, 226), T(190, 220, "развёртывание", "mut", "start", 10),
        Bx(50, 226, 260, 40, "c5", ["Inference: T = 1", ("логиты велики → softmax насыщен", "", 11)]),
        T(180, 290, "∇ₓ F_d ≈ 0 → gradient masking", "t4", fs=12),
    ]
    return _svg(p, 360, 302, "Схема defensive distillation", body)


def chart_distill(fig):
    ax = fig.add_subplot(111)
    z = np.array([4.0, 2.0, 1.0, 0.0, -1.0])
    k = np.arange(len(z))
    for i, (Tt, c) in enumerate([(1, R), (5, O), (20, B)]):
        e = np.exp(z / Tt - (z / Tt).max())
        ax.bar(k + (i - 1) * 0.27, e / e.sum(), width=0.27, color=c, label=f"T = {Tt}")
    ax.set_xticks(k, [f"класс {i + 1}" for i in k], fontsize=8)
    ax.set_ylabel("softmax(z / T)")
    ax.set_title("Логиты z = [4, 2, 1, 0, −1]", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ 3. demand_traffic_forecasting
def svg_demand():
    p = "dtf"
    ins = [("История", "спроса", "c1"), ("Календарь,", "праздники", "c3"), ("Промо,", "цены", "c3"),
           ("Погода,", "экономика", "c6")]
    body = []
    tgt = [110, 150, 210, 250]
    for i, (a, b, c) in enumerate(ins):
        x = 12 + 86 * i
        body.append(Bx(x, 12, 78, 40, c, [a, b], fs=11))
        body.append(A(p, x + 39, 52, tgt[i], 86))
    body += [
        Bx(30, 86, 300, 46, "c1", ["Признаки: лаги 1 / 7 / 365,", ("rolling mean/std, день недели, праздник", "", 11)]),
        A(p, 180, 132, 180, 152),
        Bx(30, 152, 300, 46, "c2", ["Модель", ("SARIMA · Prophet · XGBoost · LSTM", "", 11)]),
        A(p, 180, 198, 180, 218),
        Bx(80, 218, 200, 30, "c5", ["Прогноз ŷ + интервал"]),
        A(p, 180, 248, 180, 268),
        Bx(55, 268, 250, 30, "c3", ["Запасы, мощности, логистика"]),
        T(180, 318, "факт продаж → переобучение модели", "mut", fs=10),
    ]
    return _svg(p, 360, 328, "Пайплайн прогноза спроса", body)


def chart_demand(fig):
    rng = np.random.default_rng(1)
    n, h = 140, 28
    t = np.arange(n + h)
    dow = t % 7
    week = np.array([0, -4, -3, -2, 3, 18, 14])[dow]
    y = 100 + 0.15 * t + week + rng.normal(0, 3.5, t.size)
    y[[103, 104]] += [40, 25]  # праздник
    X = np.column_stack([np.ones_like(t), t] + [(dow == d).astype(float) for d in range(1, 7)])
    w, *_ = np.linalg.lstsq(X[:n], y[:n], rcond=None)
    yh = X @ w
    sd = np.std(y[:n] - yh[:n])
    ax = fig.add_subplot(111)
    s = 70
    ax.plot(t[s:n], y[s:n], color=S, lw=1.2, label="история")
    ax.plot(t[n:], y[n:], color=S, lw=1.2, ls=":", label="факт")
    ax.plot(t[n:], yh[n:], color=B, lw=2, label="прогноз")
    ax.fill_between(t[n:], yh[n:] - 1.96 * sd, yh[n:] + 1.96 * sd, color=B, alpha=0.15, label="95% интервал")
    ax.annotate("праздник", (103, y[103]), xytext=(110, y[103] - 4), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=R), color=R)
    ax.axvline(n - 0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("день")
    ax.set_ylabel("спрос")
    ax.set_title("Тренд + недельная сезонность", fontsize=10)
    ax.set_ylim(80, None)
    ax.legend(fontsize=7, loc="lower left", ncol=2)


# ================================================================ 4. differentiable_programming
def svg_diffprog():
    p = "dfp"
    body = [
        A(p, 30, 26, 330, 26, "ln s1"), T(180, 16, "forward pass: вычисляем значения", "t1", fs=11),
        f'<circle class="c1" cx="40" cy="80" r="18"/>', T(40, 85, "x₁"),
        f'<circle class="c1" cx="40" cy="170" r="18"/>', T(40, 175, "x₂"),
        f'<circle class="c3" cx="150" cy="72" r="20"/>', T(150, 76, "sin", fs=11),
        f'<circle class="c3" cx="150" cy="170" r="20"/>', T(150, 175, "×"),
        f'<circle class="c3" cx="255" cy="120" r="20"/>', T(255, 125, "+"),
        T(334, 125, "f", bold=True),
        A(p, 58, 78, 128, 73), A(p, 55, 92, 134, 157), A(p, 58, 170, 128, 170),
        A(p, 168, 80, 237, 111), A(p, 168, 162, 237, 129), A(p, 275, 120, 322, 120),
        T(8, 52, "∂f/∂x₁ = x₂ + cos x₁", "t4", "start", 11),
        T(8, 208, "∂f/∂x₂ = x₁", "t4", "start", 11),
        T(176, 58, "∂f/∂sin = 1", "t4", "start", 11),
        T(176, 196, "∂f/∂(×) = 1", "t4", "start", 11),
        A(p, 330, 228, 30, 228, "ln s4", dash=True),
        T(180, 248, "reverse mode: градиенты от f к входам", "t4", fs=11),
    ]
    return _svg(p, 360, 258, "Граф вычислений f = x1·x2 + sin x1", body)


def chart_diffprog(fig):
    ax = fig.add_subplot(111)
    h = np.logspace(-16, -1, 200)
    x = 1.0
    fwd = np.abs((np.sin(x + h) - np.sin(x)) / h - np.cos(x))
    cen = np.abs((np.sin(x + h) - np.sin(x - h)) / (2 * h) - np.cos(x))
    ax.loglog(h, np.maximum(fwd, 1e-17), color=O, lw=1.5, label="forward difference")
    ax.loglog(h, np.maximum(cen, 1e-17), color=V, lw=1.5, label="central difference")
    ax.axhline(1.1e-16, color=G, lw=2, label="AD (точно до округления)")
    ax.set_xlabel("шаг h")
    ax.set_ylabel("|ошибка f′(1)|")
    ax.set_ylim(1e-17, 1)
    ax.set_title("f(x) = sin x: численная производная vs AD", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")


# ================================================================ 5. differential_privacy
def svg_dp():
    p = "dpv"
    body = [T(52, 14, "D", bold=True), T(52, 140, "D′", bold=True),
            '<rect class="ln" x="12" y="20" width="80" height="86" rx="4"/>',
            '<rect class="ln" x="12" y="146" width="80" height="86" rx="4"/>']
    for i in range(4):
        body.append(f'<rect class="{"c3" if i == 3 else "c6"}" x="20" y="{28 + 18 * i}" width="64" height="14" rx="2"/>')
        if i < 3:
            body.append(f'<rect class="c6" x="20" y="{154 + 18 * i}" width="64" height="14" rx="2"/>')
    body.append('<rect class="s4" x="20" y="208" width="64" height="14" rx="2" stroke-dasharray="3 2"/>')
    body += [A(p, 92, 63, 120, 110), A(p, 92, 189, 120, 138),
             Bx(120, 96, 110, 54, "c1", ["M = f(D)", "+ Lap(Δf/ε)"]),
             A(p, 230, 123, 246, 123)]
    o = np.linspace(-4, 5, 120)
    xs = 250 + (o + 4) / 9 * 100
    for mu, c in [(0, "s1"), (1, "s3")]:
        pdf = 0.5 * np.exp(-np.abs(o - mu))
        body.append(Pth(p, poly(xs, 170 - pdf / 0.5 * 90), f"ln {c}", head=False))
    body += ['<line class="ln" x1="248" y1="170" x2="352" y2="170"/>',
             T(266, 72, "M(D)", "t1", fs=11), T(336, 72, "M(D′)", "t3", fs=11),
             T(300, 186, "выход M", "mut", fs=10),
             T(180, 262, "P[M(D) ∈ S] ≤ e^ε · P[M(D′) ∈ S]"),
             T(180, 282, "D и D′ отличаются одной записью; Δf — чувствительность", "mut", fs=10)]
    return _svg(p, 360, 292, "Механизм Лапласа и определение ε-DP", body)


def chart_dp(fig):
    o = np.linspace(95, 106, 600)
    ax = fig.add_subplot(2, 1, 1)
    for eps, c in [(0.2, O), (1.0, B), (3.0, G)]:
        b = 1 / eps
        ax.plot(o, np.exp(-np.abs(o - 100) / b) / (2 * b), color=c, lw=2, label=f"ε = {eps}")
    ax.axvline(100, color="k", lw=0.8, ls=":")
    ax.set_title("Шум Лапласа для count = 100 (Δf = 1)", fontsize=9)
    ax.set_xlabel("ответ механизма")
    ax.legend(fontsize=7)
    ax2 = fig.add_subplot(2, 1, 2)
    eps = 1.0
    loss = (np.abs(o - 101) - np.abs(o - 100)) * eps
    ax2.plot(o, loss, color=V, lw=2, label="ln p_D(o) / p_D′(o)")
    ax2.axhline(eps, color=R, ls="--", lw=1)
    ax2.axhline(-eps, color=R, ls="--", lw=1, label="±ε")
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title("Privacy loss при ε = 1: всегда в [−ε, ε]", fontsize=9)
    ax2.set_xlabel("ответ o (D: 100, D′: 101)")
    ax2.legend(fontsize=7, loc="center right")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 6. diffusion_models
def svg_diffusion():
    p = "dfm"
    rng = np.random.default_rng(3)
    xs = [15, 95, 205, 290]
    names = ["x₀", "x₁", "xₜ", "x_T"]
    op = [1, 0.75, 0.3, 0]
    nd = [0, 25, 60, 110]
    body = [A(p, 20, 42, 340, 42, "ln s4"), T(180, 32, "прямой процесс q: + шум βₜ", "t4", fs=11)]
    for x, nm, a, k in zip(xs, names, op, nd):
        body.append(f'<rect class="c6" x="{x}" y="60" width="50" height="50" rx="3"/>')
        if a > 0:
            body.append(f'<circle class="c1" cx="{x + 20}" cy="80" r="11" opacity="{a}"/>')
            body.append(f'<path class="c2" d="M{x + 22},104 L{x + 36},80 L{x + 46},104 z" opacity="{a}"/>')
        for px, py in zip(rng.uniform(x + 3, x + 47, k), rng.uniform(63, 107, k)):
            body.append(f'<circle class="t6" cx="{px:.1f}" cy="{py:.1f}" r="1.3"/>')
        body.append(T(x + 25, 128, nm))
    body += [A(p, 65, 85, 93, 85), A(p, 145, 85, 162, 85), T(175, 89, "…"), A(p, 186, 85, 203, 85),
             A(p, 255, 85, 288, 85),
             A(p, 340, 146, 20, 146, "ln s1"), T(180, 164, "обратный процесс p_θ: сеть ε_θ убирает шум", "t1", fs=11),
             T(180, 192, "xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε"),
             T(180, 214, "L = ‖ε − ε_θ(xₜ, t)‖²")]
    return _svg(p, 360, 226, "Прямой и обратный процесс диффузии", body)


def chart_diffusion(fig):
    Tn = 1000
    t = np.arange(1, Tn + 1)
    beta = np.linspace(1e-4, 0.02, Tn)
    ab_lin = np.cumprod(1 - beta)
    s = 0.008
    f = lambda tt: np.cos((tt / Tn + s) / (1 + s) * np.pi / 2) ** 2
    ab_cos = f(t) / f(0)
    ax = fig.add_subplot(111)
    ax.plot(t, ab_lin, color=O, lw=2, label="linear β")
    ax.plot(t, ab_cos, color=B, lw=2, label="cosine")
    ax.set_xlabel("шаг t")
    ax.set_ylabel("ᾱₜ — доля сигнала")
    ax.set_title("Noise schedule: сколько сигнала осталось", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ 7. digital_humanities
def svg_dh():
    p = "dhm"
    body = [
        Bx(40, 10, 280, 28, "c6", ["Архив: рукописи, книги, картины"]), A(p, 180, 38, 180, 54),
        Bx(40, 54, 280, 28, "c3", ["OCR (PyTesseract), оцифровка"]), A(p, 180, 82, 180, 98),
        Bx(40, 98, 280, 28, "c1", ["Корпус: тексты + метаданные"]), A(p, 180, 126, 180, 144),
        '<rect class="s6" x="6" y="144" width="348" height="78" rx="6" stroke-dasharray="4 3"/>',
        Bx(14, 151, 162, 28, "c2", ["Topic modeling: LDA"]),
        Bx(184, 151, 162, 28, "c2", ["Stylometry: PCA, SVM"]),
        Bx(14, 187, 162, 28, "c2", ["NER: люди, места"]),
        Bx(184, 187, 162, 28, "c2", ["Сети: NetworkX"]),
        A(p, 180, 222, 180, 242),
        Bx(40, 242, 280, 28, "c5", ["Интерпретация исследователем"]),
    ]
    return _svg(p, 360, 280, "Пайплайн цифровой гуманитаристики", body)


def chart_dh(fig):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(7)
    words = 6
    pa = rng.dirichlet(np.ones(words) * 3)
    pb = np.clip(pa + rng.normal(0, 0.06, words), 0.01, None)
    pb /= pb.sum()

    def chunks(pr, k):
        return np.array([rng.multinomial(1500, pr) / 1500 for _ in range(k)])

    Xa, Xb, Xq = chunks(pa, 15), chunks(pb, 15), chunks(pa, 3)
    pca = PCA(2).fit(np.vstack([Xa, Xb]))
    ax = fig.add_subplot(111)
    for X, c, lab, m in [(Xa, B, "автор A", "o"), (Xb, O, "автор B", "s")]:
        Z = pca.transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], color=c, s=22, marker=m, label=lab, alpha=0.8)
    Z = pca.transform(Xq)
    ax.scatter(Z[:, 0], Z[:, 1], color=R, s=90, marker="*", label="аноним (3 фрагмента)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Stylometry: частоты 6 служебных слов → PCA", fontsize=9)
    ax.legend(fontsize=7)


# ================================================================ 8. distributed_training
def svg_dist():
    p = "dtr"
    body = [Bx(60, 10, 240, 28, "c6", ["Глобальный batch → 4 шарда"])]
    starts = [100, 150, 210, 260]
    for i in range(4):
        x = 12 + 86 * i
        cx = x + 37
        body += [A(p, starts[i], 38, cx, 66), Bx(x, 66, 74, 46, "c1", [f"GPU {i}", ("копия w", "mut", 10)], fs=12),
                 A(p, cx, 112, cx, 150, "ln s3"), T(cx + 6, 136, f"g{'₀₁₂₃'[i]}", "t3", "start", 12),
                 A(p, cx, 182, cx, 208)]
    body += [Bx(12, 150, 336, 32, "c3", ["all_reduce: ḡ = (g₀ + g₁ + g₂ + g₃) / 4"]),
             Bx(12, 208, 336, 32, "c2", ["w ← w − η·ḡ на каждом GPU"]),
             T(180, 262, "веса остаются одинаковыми; обмен — NCCL", "mut", fs=10)]
    return _svg(p, 360, 272, "Data parallelism с all_reduce", body)


def chart_dist(fig):
    ax = fig.add_subplot(111)
    m = np.arange(1, 65)
    for pp, c in [(2, G), (4, B), (8, O), (16, R)]:
        ax.plot(m, (pp - 1) / (m + pp - 1), color=c, lw=2, label=f"p = {pp} GPU")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("число micro-batches m")
    ax.set_ylabel("доля простоя (bubble)")
    ax.set_title("Pipeline: bubble = (p − 1) / (m + p − 1)", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ 9. dynamic_programming_rl
def svg_dprl():
    p = "dprl"
    body = [
        Bx(20, 30, 90, 40, "c1", ["π"], fs=14), Bx(250, 30, 90, 40, "c2", ["V"], fs=14),
        A(p, 110, 42, 248, 42, "ln s2"), T(180, 24, "evaluation: V → V^π", "t2", fs=11),
        A(p, 250, 60, 112, 60, "ln s1"), T(180, 82, "improvement: π ← greedy(V)", "t1", fs=11),
        T(12, 112, "Policy Evaluation:", "t2", "start", 11),
        T(12, 130, "V(s) ← Σₐ π(a|s) Σₛ′ P(s′|s,a)[r + γV(s′)]", a="start", fs=11),
        T(12, 158, "Policy Improvement:", "t1", "start", 11),
        T(12, 176, "π(s) ← argmaxₐ Σₛ′ P(s′|s,a)[r + γV(s′)]", a="start", fs=11),
        T(12, 204, "Value Iteration (оба шага сразу):", "t5", "start", 11),
        T(12, 222, "V(s) ← maxₐ Σₛ′ P(s′|s,a)[r + γV(s′)]", a="start", fs=11),
        T(180, 252, "повтор до сходимости → π*, V*", "mut", fs=11),
    ]
    return _svg(p, 360, 262, "Generalized Policy Iteration", body)


def chart_dprl(fig):
    rng = np.random.default_rng(0)
    nS, nA = 20, 3
    P = rng.dirichlet(np.ones(nS), size=(nS, nA))
    Rw = rng.normal(0, 1, (nS, nA))
    ax = fig.add_subplot(111)
    for g, c in [(0.5, G), (0.9, B), (0.99, O)]:
        Vv = np.zeros(nS)
        d = []
        for _ in range(150):
            Vn = (Rw + g * P @ Vv).max(1)
            d.append(max(np.abs(Vn - Vv).max(), 1e-16))
            Vv = Vn
        ax.semilogy(d, color=c, lw=2, label=f"γ = {g}")
    ax.set_xlabel("итерация k")
    ax.set_ylabel("‖Vₖ₊₁ − Vₖ‖∞")
    ax.set_title("Value Iteration: сходимость ~ γᵏ", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ 10. early_late_fusion
def svg_fusion():
    p = "elf"
    body = [
        T(12, 18, "Early fusion", "t1", "start", bold=True),
        Bx(12, 28, 64, 26, "c6", ["Видео"]), Bx(12, 64, 64, 26, "c6", ["Аудио"]),
        A(p, 76, 41, 104, 54), A(p, 76, 77, 104, 66),
        Bx(104, 42, 76, 36, "c3", ["concat", "[xᵥ; xₐ]"], fs=11),
        A(p, 180, 60, 206, 60), Bx(206, 42, 82, 36, "c1", ["Одна модель"]),
        A(p, 288, 60, 318, 60), T(334, 65, "ŷ", bold=True),
        T(180, 104, "учит взаимодействия модальностей", "mut", fs=10),
        '<line class="s6" x1="10" y1="114" x2="350" y2="114" stroke-dasharray="4 3"/>',
        T(12, 134, "Late fusion", "t2", "start", bold=True),
        Bx(12, 144, 64, 26, "c6", ["Видео"]), Bx(12, 200, 64, 26, "c6", ["Аудио"]),
        A(p, 76, 157, 96, 157), A(p, 76, 213, 96, 213),
        Bx(96, 144, 80, 26, "c1", ["Модель 1"]), Bx(96, 200, 80, 26, "c1", ["Модель 2"]),
        A(p, 176, 157, 210, 174), A(p, 176, 213, 210, 196),
        T(192, 156, "p₁", "mut", fs=11), T(192, 222, "p₂", "mut", fs=11),
        Bx(210, 164, 96, 42, "c2", ["Объединение", ("avg · vote · stack", "", 10)]),
        A(p, 306, 185, 326, 185), T(340, 190, "ŷ", bold=True),
        T(180, 248, "легко убрать модальность; нет взаимодействий", "mut", fs=10),
    ]
    return _svg(p, 360, 258, "Early и late fusion", body)


# ================================================================ 11. efficient_transformers
def svg_efftr():
    p = "eft"
    n, cs = 8, 9
    rnd = {(2, 6), (5, 1), (7, 3), (4, 7), (6, 2), (1, 4)}

    def mask(kind):
        cells = []
        for i in range(n):
            for j in range(n):
                c = "c6"
                if kind == "full":
                    c = "c1"
                else:
                    if abs(i - j) <= 1:
                        c = "c1"
                    if kind in ("lf", "bb") and (i == 0 or j == 0):
                        c = "c3"
                    if kind == "bb" and (i, j) in rnd:
                        c = "c4"
                cells.append((i, j, c))
        return cells

    body = [T(180, 16, "строки — query i, столбцы — key j", "mut", fs=10)]
    spec = [("full", 54, 30, "Full attention", "O(n²)"), ("loc", 234, 30, "Local (окно w)", "O(n·w)"),
            ("lf", 54, 164, "Longformer", "local + global"), ("bb", 234, 164, "BigBird", "+ random связи")]
    for kind, gx, gy, name, cx in spec:
        for i, j, c in mask(kind):
            body.append(f'<rect class="{c}" x="{gx + j * cs}" y="{gy + i * cs}" width="{cs}" height="{cs}"/>')
        body += [T(gx + 36, gy + 72 + 16, name, fs=12), T(gx + 36, gy + 72 + 30, cx, "mut", fs=11)]
    return _svg(p, 360, 272, "Паттерны разреженного внимания", body)


def chart_efftr(fig):
    ax = fig.add_subplot(111)
    nn = np.logspace(np.log10(512), np.log10(65536), 50)
    ax.loglog(nn, nn ** 2, color=R, lw=2, label="full: n²")
    ax.loglog(nn, nn * 512, color=B, lw=2, label="local / Longformer: n·w, w = 512")
    ax.loglog(nn, nn * 256, color=G, lw=2, ls="--", label="Linformer: n·k, k = 256")
    ax.set_xlabel("длина последовательности n")
    ax.set_ylabel("элементов в матрице внимания")
    ax.set_title("Квадратичный vs линейный рост", fontsize=10)
    ax.legend(fontsize=7)


# ================================================================ 12. em_algorithm
def svg_em():
    p = "emv"
    X = lambda th: 30 + th * 35
    Y = lambda v: 30 + (0.5 - v) * 30
    L = lambda th: -((th - 6) ** 2) / 8
    tt, c = 2.0, 0.35
    E = lambda th: L(tt) + (-(tt - 6) / 4) * (th - tt) - c * (th - tt) ** 2
    t1 = tt + 1 / (2 * c)
    th = np.linspace(0, 9, 120)
    the = np.linspace(0, 7, 100)
    body = ['<line class="ln" x1="28" y1="235" x2="345" y2="235"/>', T(350, 239, "θ", a="end", fs=11),
            Pth(p, poly(X(th), Y(L(th))), "ln s1", head=False),
            Pth(p, poly(X(the), Y(E(the))), "ln s3", head=False),
            A(p, X(tt), Y(L(tt)), X(tt), 235, "ln s6", dash=True, head=False),
            A(p, X(t1), Y(L(t1)), X(t1), 235, "ln s6", dash=True, head=False),
            f'<circle class="c1" cx="{X(tt):.1f}" cy="{Y(L(tt)):.1f}" r="4"/>',
            f'<circle class="c3" cx="{X(t1):.1f}" cy="{Y(E(t1)):.1f}" r="4"/>',
            f'<circle class="c1" cx="{X(t1):.1f}" cy="{Y(L(t1)):.1f}" r="4"/>',
            T(X(tt), 250, "θᵗ", fs=12), T(X(t1) + 6, 250, "θᵗ⁺¹", fs=12),
            T(300, 45, "log p(X|θ)", "t1", fs=12), T(160, 178, "ELBO(q, θ)", "t3", "start", 12),
            T(92, 92, "E-шаг: KL = 0", "t3", "end", 11),
            T(X(t1), 40, "M-шаг: max ELBO", "t3", fs=11),
            T(X(t1) + 8, 81, "KL", "t4", "start", 10),
            T(180, 270, "L(θᵗ⁺¹) ≥ ELBO(θᵗ⁺¹) ≥ ELBO(θᵗ) = L(θᵗ)", fs=11)]
    return _svg(p, 360, 280, "EM как подъём по нижней оценке ELBO", body)


def chart_em(fig):
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-2, 0.8, 200), rng.normal(2, 1.2, 300)])
    pi, mu, sd = np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([1.5, 1.5])
    npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    ll = []
    for _ in range(40):
        w = pi * npdf(x[:, None], mu, sd)
        ll.append(np.log(w.sum(1)).sum())
        g = w / w.sum(1, keepdims=True)
        Nk = g.sum(0)
        pi = Nk / len(x)
        mu = (g * x[:, None]).sum(0) / Nk
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / Nk)
    ax = fig.add_subplot(2, 1, 1)
    ax.hist(x, bins=40, density=True, color=S, alpha=0.5)
    xs = np.linspace(-5, 6, 300)
    for k, c in enumerate([B, O]):
        ax.plot(xs, pi[k] * npdf(xs, mu[k], sd[k]), color=c, lw=2, label=f"компонента {k + 1}")
    ax.plot(xs, (pi * npdf(xs[:, None], mu, sd)).sum(1), color="k", lw=1, ls="--", label="смесь")
    ax.set_title("GMM после 40 итераций EM", fontsize=9)
    ax.legend(fontsize=7)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.arange(1, 41), ll, "o-", color=G, ms=3)
    ax2.set_xlabel("итерация")
    ax2.set_ylabel("log-likelihood")
    ax2.set_title("Правдоподобие не убывает", fontsize=9)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 13. encoder_decoder
def svg_encdec():
    p = "edc"
    body = []
    enc = [15, 70, 125]
    dec = [226, 276, 326]
    ins = ["Я", "люблю", "ML"]
    dins = ["&lt;s&gt;", "I", "love"]
    outs = ["I", "love", "ML"]
    for i, x in enumerate(enc):
        body += [Bx(x, 95, 42, 32, "c1", [f"h{'₁₂₃'[i]}"]), A(p, x + 21, 158, x + 21, 129), T(x + 21, 172, ins[i], fs=11)]
        if i < 2:
            body.append(A(p, x + 42, 111, enc[i + 1], 111))
    for i, x in enumerate(dec):
        body += [Bx(x, 95, 40, 32, "c2", [f"s{'₁₂₃'[i]}"]), A(p, x + 20, 158, x + 20, 129), T(x + 20, 172, dins[i], fs=11),
                 A(p, x + 20, 95, x + 20, 62), T(x + 20, 54, outs[i], "t2", fs=12)]
        if i < 2:
            body.append(A(p, x + 40, 111, dec[i + 1], 111))
    body += [A(p, 167, 111, 176, 111), f'<circle class="c3" cx="195" cy="111" r="18"/>', T(195, 116, "c"),
             A(p, 213, 111, 225, 111), T(195, 146, "context", "mut", fs=10),
             T(91, 200, "Encoder", "t1", bold=True), T(296, 200, "Decoder", "t2", bold=True),
             T(188, 222, "вход decoder: предыдущий токен (teacher forcing)", "mut", fs=10)]
    return _svg(p, 375, 232, "Seq2seq encoder-decoder", body)


# ================================================================ 14. ensemble_diversification
def svg_ensdiv():
    p = "edv"
    body = []

    def block(y0, title, cls, errs):
        out = [T(12, y0 - 8, title, cls, "start", 12, bold=True)]
        rows = [f"M{'₁₂₃'[i]}" for i in range(3)]
        for r in range(4):
            y = y0 + r * 22 + (6 if r == 3 else 0)
            if r < 3:
                e = errs[r]
            else:
                e = {j for j in range(10) if sum(j in es for es in errs) >= 2}
            out.append(T(60, y + 15, rows[r] if r < 3 else "Голос.", a="end", fs=11, bold=(r == 3)))
            for j in range(10):
                bad = j in e
                out.append(f'<rect class="{"c4" if bad else "c2"}" x="{70 + 20 * j}" y="{y}" width="18" height="18" rx="2"/>')
                if bad:
                    out.append(T(79 + 20 * j, y + 14, "✗", fs=11))
            out.append(T(278, y + 14, f"{100 - 10 * len(e)}%", a="start", fs=11, bold=(r == 3)))
        return out

    body += block(30, "Ошибки в разных местах", "t2", [{0, 3, 7}, {1, 5, 8}, {2, 6, 9}])
    body += block(158, "Ошибки совпадают", "t4", [{0, 3, 7}, {0, 3, 8}, {0, 3, 9}])
    return _svg(p, 330, 262, "Голосование при разных и совпадающих ошибках", body)


def chart_ensdiv(fig):
    ax = fig.add_subplot(111)
    M = np.arange(1, 51)
    for rho, c in [(0.0, G), (0.3, B), (0.6, O), (0.9, R)]:
        ax.plot(M, rho + (1 - rho) / M, color=c, lw=2, label=f"ρ = {rho}")
    ax.set_xlabel("число моделей M")
    ax.set_ylabel("Var(среднего) / σ²")
    ax.set_ylim(0, 1.05)
    ax.set_title("Var = ρσ² + (1 − ρ)σ²/M", fontsize=10)
    ax.legend(fontsize=8, loc="center right")


# ================================================================ 15. ensemble_methods_streaming
def svg_stream():
    p = "ems"
    body = [Bx(80, 10, 200, 28, "c6", ["Поток (xₜ, yₜ)"])]
    tx = [20, 135, 250]
    st = [130, 180, 230]
    ks = [("k = 1", 92, 58, "end"), ("k = 0", 186, 64, "start"), ("k = 2", 268, 58, "start")]
    for i, x in enumerate(tx):
        body += [A(p, st[i], 38, x + 45, 80, dash=(i == 1)), T(ks[i][1], ks[i][2], ks[i][0], "t3", ks[i][3], 11),
                 Bx(x, 80, 90, 40, "c1", [f"Дерево {i + 1}", ("+ ADWIN", "mut", 10)], fs=12),
                 A(p, x + 45, 120, [130, 180, 230][i], 160)]
    body += [Bx(90, 160, 180, 32, "c2", ["Взвешенное голосование"]), A(p, 180, 192, 180, 212),
             T(180, 228, "ŷₜ", bold=True),
             T(180, 252, "kᵢ ~ Poisson(λ): λ = 1 Online Bagging, λ = 6 Leveraging", "mut", fs=10),
             T(180, 268, "drift (ADWIN) → фоновое дерево заменяет основное", "mut", fs=10)]
    return _svg(p, 360, 278, "Online bagging и ADWIN", body)


def chart_stream(fig):
    from scipy.stats import poisson
    ax = fig.add_subplot(111)
    k = np.arange(0, 15)
    ax.bar(k - 0.2, poisson.pmf(k, 1), width=0.4, color=B, label="λ = 1 (Online Bagging)")
    ax.bar(k + 0.2, poisson.pmf(k, 6), width=0.4, color=O, label="λ = 6 (Leveraging Bagging)")
    ax.set_xlabel("k — сколько раз модель учится на примере")
    ax.set_ylabel("P(k)")
    ax.set_title("Вес примера k ~ Poisson(λ)", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ 16. ensemble_methods_voting_stacking
def svg_stack():
    p = "evs"
    body = [Bx(60, 10, 240, 28, "c6", ["Train (X, y), K фолдов"])]
    names = ["LogReg", "RandomForest", "kNN"]
    for i, nm in enumerate(names):
        x = 12 + 116 * i
        body += [A(p, [130, 180, 230][i], 38, x + 52, 64), Bx(x, 64, 104, 36, "c1", [nm]),
                 A(p, x + 52, 100, [110, 180, 250][i], 136)]
    body += [Bx(40, 136, 280, 30, "c3", ["OOF-предсказания: [p₁, p₂, p₃]"]), A(p, 180, 166, 180, 186),
             Bx(80, 186, 200, 30, "c5", ["Meta-model (LogReg)"]), A(p, 180, 216, 180, 234),
             T(180, 250, "ŷ", bold=True),
             T(180, 274, "fit на K−1 фолдах, predict на отложенном → нет утечки", "mut", fs=10),
             T(180, 290, "Voting: вместо meta-model — среднее pᵢ или голоса", "mut", fs=10)]
    return _svg(p, 360, 300, "Stacking на out-of-fold предсказаниях", body)


def chart_voting(fig):
    ax = fig.add_subplot(111)
    pr = [0.9, 0.45, 0.45]
    names = ["модель 1", "модель 2", "модель 3", "soft: среднее"]
    vals = pr + [np.mean(pr)]
    cols = [B, B, B, G]
    ax.bar(range(4), vals, color=cols, width=0.6)
    ax.axhline(0.5, color=R, ls="--", lw=1.2, label="порог 0.5")
    for i, v in enumerate(vals):
        ax.text(i, 0.15, f"{v:.2f}", ha="center", fontsize=9, color="white")
    ax.set_xticks(range(4), names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(класс 1)")
    ax.set_title("Hard: 1 голос из 3 → класс 0\nSoft: 0.60 → класс 1", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")


# ================================================================ 17. ensemble_tuning
def svg_enstune():
    p = "etn"
    steps = [("1. Разнообразные базовые модели", "c1"), ("2. OOF-предсказания (K-fold)", "c6"),
             ("3. Корреляция ошибок → убрать похожие", "c3"), ("4. Веса / meta-model на OOF", "c5"),
             ("5. Калибровка вероятностей", "c2"), ("6. Проверка на holdout", "c4")]
    body = []
    for i, (s, c) in enumerate(steps):
        y = 10 + 44 * i
        body.append(Bx(30, y, 300, 30, c, [s]))
        if i < 5:
            body.append(A(p, 180, y + 30, 180, y + 44))
    body.append(Pth(p, "M330,245 L348,245 L348,113 L332,113", "ln s4", dash=True))
    return _svg(p, 360, 272, "Порядок настройки ансамбля", body)


def chart_enstune(fig):
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import GradientBoostingRegressor
    X, y = make_friedman1(n_samples=700, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = X[:400], X[400:], y[:400], y[400:]
    ax = fig.add_subplot(111)
    for lr, c in [(0.5, R), (0.1, B), (0.02, G)]:
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=lr, max_depth=3, random_state=0).fit(Xtr, ytr)
        err = [np.mean((yte - p) ** 2) for p in m.staged_predict(Xte)]
        ax.plot(np.arange(1, 301), err, color=c, lw=2, label=f"learning_rate = {lr}")
        k = int(np.argmin(err))
        ax.plot(k + 1, err[k], "o", color=c, ms=5)
    ax.set_ylim(0, 12)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("MSE на test")
    ax.set_title("Boosting: learning_rate ↔ n_estimators", fontsize=10)
    ax.legend(fontsize=8)


# ================================================================ 18. evolutionary_strategies
def svg_es():
    p = "evs2"
    body = [
        Bx(15, 24, 150, 44, "c1", ["θ — центр поиска", ("σ — радиус шума", "mut", 10)]),
        Bx(195, 24, 150, 44, "c3", ["εᵢ ~ N(0, I)", ("i = 1 … n", "mut", 10)]),
        Bx(195, 150, 150, 44, "c2", ["Fᵢ = F(θ + σεᵢ)", ("параллельно, без ∇F", "mut", 10)]),
        Bx(15, 150, 150, 44, "c5", [("θ ← θ + α/(nσ)·Σ Fᵢεᵢ", "", 11), ("Fᵢ нормализованы", "mut", 10)]),
        A(p, 165, 46, 193, 46), A(p, 270, 68, 270, 148), A(p, 195, 172, 167, 172), A(p, 90, 150, 90, 70),
        T(180, 113, "одно поколение", "mut", fs=10),
    ]
    return _svg(p, 360, 210, "Цикл эволюционной стратегии", body)


def chart_es(fig):
    rng = np.random.default_rng(4)
    f = lambda P: (P[..., 0] - 1) ** 2 + 5 * (P[..., 1] - 0.5) ** 2
    th = np.array([-1.6, -1.4])
    sig, alpha, n = 0.3, 0.05, 30
    path, pops = [th.copy()], {}
    for g in range(41):
        eps = rng.normal(size=(n, 2))
        pop = th + sig * eps
        F = -f(pop)
        if g in (0, 10, 40):
            pops[g] = pop
        Fz = (F - F.mean()) / (F.std() + 1e-8)
        th = th + alpha / (n * sig) * (Fz @ eps)
        path.append(th.copy())
    path = np.array(path)
    ax = fig.add_subplot(111)
    gx, gy = np.meshgrid(np.linspace(-2.2, 2, 120), np.linspace(-2, 1.8, 120))
    ax.contour(gx, gy, f(np.stack([gx, gy], -1)), levels=12, colors=S, linewidths=0.6)
    for g, c in [(0, R), (10, O), (40, G)]:
        ax.scatter(pops[g][:, 0], pops[g][:, 1], s=8, color=c, alpha=0.7, label=f"поколение {g}")
    ax.plot(path[:, 0], path[:, 1], color=B, lw=2, label="путь θ")
    ax.plot(1, 0.5, "k*", ms=10)
    ax.set_title("ES на f = (x−1)² + 5(y−0.5)²", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")


# ================================================================ 19. explainability_accountability
def svg_xai():
    p = "xac"
    body = [T(140, 24, "Model-agnostic", "t1", fs=12, bold=True), T(276, 24, "Model-specific", "t2", fs=12, bold=True),
            T(38, 66, "Global", fs=12, bold=True), T(38, 128, "Local", fs=12, bold=True),
            Bx(76, 34, 128, 56, "c1", ["Permutation", "importance, PDP"], fs=11),
            Bx(212, 34, 128, 56, "c2", ["Коэффициенты,", "feature importance"], fs=11),
            Bx(76, 96, 128, 56, "c3", ["LIME,", "Kernel SHAP"], fs=11),
            Bx(212, 96, 128, 56, "c5", ["TreeSHAP,", "Grad-CAM"], fs=11),
            T(10, 180, "Accountability:", "t4", "start", 12, bold=True)]
    for i, s in enumerate(["Логи", "Аудит", "Оспаривание", "Исправление"]):
        x = 10 + 88 * i
        body.append(Bx(x, 190, 76, 28, "c6", [s], fs=10))
        if i < 3:
            body.append(A(p, x + 76, 204, x + 87, 204))
    return _svg(p, 352, 230, "Типы объяснений и цепочка подотчётности", body)


# ================================================================ 20. fairness_constraints
def svg_fcon():
    p = "fco"
    body = []
    mains = [("Данные", 10), ("Обучение", 80), ("Модель", 150), ("Решения", 220)]
    for i, (s, y) in enumerate(mains):
        body.append(Bx(20, y, 120, 28, "c6", [s]))
        if i < 3:
            body.append(A(p, 80, y + 28, 80, y + 70))
    stages = [(36, "c1", "Pre-processing", "reweighting, resampling"),
              (106, "c3", "In-processing", "constraint, adversarial"),
              (176, "c2", "Post-processing", "пороги по группам")]
    for y, c, a, b in stages:
        body += [Bx(165, y, 185, 40, c, [a, (b, "", 11)]),
                 A(p, 165, y + 20, 84, y + 20, "ln s6", dash=True)]
    body += [T(180, 272, "DP: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)", "mut", fs=11),
             T(180, 290, "Equal Opportunity: TPR равны по группам", "mut", fs=11)]
    return _svg(p, 360, 300, "Где добавить fairness-ограничения", body)


def chart_fcon(fig):
    from scipy.stats import norm
    ax = fig.add_subplot(111)
    t = np.linspace(0, 1, 300)
    target = 0.8
    for mu, c, lab in [(0.65, B, "A = 0"), (0.52, O, "A = 1")]:
        tpr = 1 - norm.cdf((t - mu) / 0.12)
        ax.plot(t, tpr, color=c, lw=2, label=f"TPR, группа {lab}")
        thr = mu + 0.12 * norm.ppf(1 - target)
        ax.plot([thr, thr], [0, target], color=c, ls=":", lw=1.2)
        ax.plot(thr, target, "o", color=c)
    ax.axvline(0.5, color=S, lw=1, ls="--", label="общий порог 0.5")
    ax.axhline(target, color=R, lw=1, ls="--")
    ax.text(0.6, target + 0.04, "TPR = 0.8 у обеих", fontsize=8, color=R)
    ax.set_xlabel("порог t")
    ax.set_ylabel("TPR")
    ax.set_title("Post-processing под Equal Opportunity", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")


# ================================================================ 21. fairness_ml
def svg_fml():
    p = "fml"
    body = [
        Bx(15, 20, 150, 44, "c6", ["Исторические", "данные с bias"], fs=11),
        Bx(195, 20, 150, 44, "c1", ["Модель учится", "на перекосе"], fs=11),
        Bx(195, 140, 150, 44, "c3", ["Решения:", "больше патрулей"], fs=11),
        Bx(15, 140, 150, 44, "c4", ["Новые данные:", "больше задержаний"], fs=11),
        A(p, 165, 42, 193, 42), A(p, 270, 64, 270, 138), A(p, 195, 162, 167, 162), A(p, 90, 140, 90, 66),
        T(180, 106, "feedback loop", "mut", fs=11),
        T(180, 220, "Удалить A недостаточно:", "t4", fs=12),
        T(180, 238, "прокси (индекс, имя) восстанавливают A", "mut", fs=11),
    ]
    return _svg(p, 360, 250, "Петля обратной связи bias", body)


# ================================================================ 22. fastai
def svg_fastai():
    p = "fai"
    body = [
        Bx(40, 10, 280, 30, "c6", ["DataBlock → DataLoaders"]), A(p, 180, 40, 180, 56),
        Bx(40, 56, 280, 30, "c1", ["vision_learner(dls, resnet34)"]), A(p, 180, 86, 180, 102),
        Bx(40, 102, 280, 30, "c3", ["lr_find() → выбрать LR"]), A(p, 180, 132, 180, 148),
        '<rect class="s2" x="20" y="148" width="320" height="96" rx="6" stroke-dasharray="4 3"/>',
        T(180, 164, "fine_tune(epochs)", "t2", fs=12, bold=True),
        Bx(34, 174, 140, 58, "c2", ["freeze", "учим голову", "1 эпоха"], fs=11),
        Bx(186, 174, 140, 58, "c2", ["unfreeze", "все слои,", "discriminative LR"], fs=11),
        A(p, 174, 203, 185, 203), A(p, 180, 244, 180, 262),
        Bx(40, 262, 280, 30, "c5", ["export() → load_learner()"]),
    ]
    return _svg(p, 360, 302, "Рабочий процесс FastAI", body)


def chart_fastai(fig):
    n = 1000
    it = np.arange(n)
    pct, lr_max, div, div_f = 0.25, 1e-3, 25, 1e5
    n1 = int(n * pct)
    cos = lambda a, b, x: b + (a - b) / 2 * (1 + np.cos(np.pi * x))
    x1, x2 = it[:n1] / n1, (it[n1:] - n1) / (n - n1)
    lr = np.concatenate([cos(lr_max / div, lr_max, x1), cos(lr_max, lr_max / div_f, x2)])
    mom = np.concatenate([cos(0.95, 0.85, x1), cos(0.85, 0.95, x2)])
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(it, lr, color=B, lw=2)
    ax.set_ylabel("learning rate")
    ax.set_title("fit_one_cycle: warm-up 25%, затем annealing", fontsize=9)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(it, mom, color=O, lw=2)
    ax2.set_ylabel("momentum")
    ax2.set_xlabel("итерация")
    ax2.set_title("Momentum движется обратно LR: 0.95 → 0.85 → 0.95", fontsize=9)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 23. feature_interaction
def svg_finter():
    p = "fin"
    body = [Bx(140, 14, 40, 28, "c1", ["x₁"], rx=3), Bx(180, 14, 40, 28, "c1", ["x₂"], rx=3),
            A(p, 180, 44, 180, 72), T(190, 62, "PolynomialFeatures(degree=2)", "mut", "start", 10)]
    cells = [("1", "c6"), ("x₁", "c1"), ("x₂", "c1"), ("x₁²", "c3"), ("x₁x₂", "c5"), ("x₂²", "c3")]
    for i, (s, c) in enumerate(cells):
        body.append(Bx(24 + 52 * i, 74, 52, 30, c, [s], rx=3))
    leg = [(20, "c6", "bias"), (80, "c1", "исходные"), (160, "c3", "степени"), (240, "c5", "взаимодействие")]
    for x, c, s in leg:
        body += [f'<rect class="{c}" x="{x}" y="120" width="11" height="11" rx="2"/>', T(x + 15, 130, s, "mut", "start", 10)]
    body += [T(180, 164, "[2, 3] → [1, 2, 3, 4, 6, 9]"),
             T(180, 190, "interaction_only: [1, x₁, x₂, x₁x₂]", fs=11),
             T(180, 216, "число признаков: C(n + d, d)"),
             T(180, 234, "n = 10, d = 3 → 286", "mut", fs=11)]
    return _svg(p, 360, 246, "PolynomialFeatures для двух признаков", body)


def chart_finter(fig):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, (300, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    gx, gy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
    grid = np.c_[gx.ravel(), gy.ravel()]
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#ffecd6", "#dbe8ff"])
    for k, (inter, title) in enumerate([(False, "Без взаимодействия: x₁, x₂"), (True, "С признаком x₁·x₂")]):
        f = (lambda Z: np.c_[Z, Z[:, 0] * Z[:, 1]]) if inter else (lambda Z: Z)
        m = LogisticRegression(C=100).fit(f(X), y)
        acc = m.score(f(X), y)
        ax = fig.add_subplot(2, 1, k + 1)
        ax.contourf(gx, gy, m.predict(f(grid)).reshape(gx.shape), cmap=cm, alpha=0.9)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=7, color=O)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=7, color=B)
        ax.set_title(f"{title} — accuracy {acc:.2f}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(False)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 24. feature_stores
def svg_fstore():
    p = "fst"
    body = [
        Bx(12, 10, 160, 30, "c6", ["Batch: DWH, озеро"]), Bx(188, 10, 160, 30, "c6", ["Stream: Kafka"]),
        A(p, 92, 40, 140, 62), A(p, 268, 40, 220, 62),
        Bx(40, 62, 280, 30, "c3", ["Feature transformations"]),
        A(p, 140, 92, 92, 118), A(p, 220, 92, 268, 118),
        Bx(12, 118, 160, 44, "c1", ["Offline store", ("история, time travel", "", 11)]),
        Bx(188, 118, 160, 44, "c2", ["Online store", ("последние значения, мс", "", 11)]),
        A(p, 172, 140, 186, 140, "ln s6", dash=True),
        A(p, 92, 162, 92, 190), A(p, 268, 162, 268, 190),
        Bx(12, 190, 160, 40, "c5", ["Обучение", ("point-in-time join", "", 11)]),
        Bx(188, 190, 160, 40, "c5", ["Inference", ("real-time запрос", "", 11)]),
        Bx(12, 248, 336, 28, "c6", [("Feature registry: метаданные, версии, владельцы", "", 11)]),
    ]
    return _svg(p, 360, 288, "Архитектура Feature Store", body)


# ================================================================ 25. federated_learning
def svg_fl():
    p = "fdl"
    body = [Bx(90, 10, 180, 44, "c1", ["Сервер: глобальная wₜ", ("wₜ₊₁ = Σₖ (nₖ/n)·wₜᵏ", "", 11)])]
    downs = [(120, 54, 56, 150), (172, 54, 172, 150), (240, 54, 304, 150)]
    ups = [(72, 150, 134, 54), (188, 150, 188, 54), (288, 150, 226, 54)]
    lab = [(70, 100, "end", 120, 106, "start"), (166, 104, "end", 194, 104, "start"),
           (290, 100, "start", 242, 106, "end")]
    for i in range(3):
        x = 12 + 116 * i
        body += [A(p, *downs[i], "ln s1"), A(p, *ups[i], "ln s2", dash=True),
                 T(lab[i][0], lab[i][1], "wₜ", "t1", lab[i][2], 11), T(lab[i][3], lab[i][4], "wₜᵏ", "t2", lab[i][5], 11),
                 Bx(x, 150, 104, 44, "c2", [f"Клиент {i + 1}", ("E эпох SGD", "", 11)]),
                 f'<line class="ln" x1="{x + 52}" y1="194" x2="{x + 52}" y2="212"/>',
                 Bx(x + 12, 212, 80, 24, "c6", [f"данные {i + 1}"], fs=11)]
    body.append(T(180, 262, "сырые данные не покидают клиентов", "mut", fs=11))
    return _svg(p, 360, 272, "Раунд FedAvg", body)


# ================================================================ 26. few_shot_learning_traditional
def svg_fewshot():
    p = "fsl"
    rng = np.random.default_rng(11)
    centers = [(80, 105), (275, 95), (185, 205)]
    cls = ["c1", "c2", "c3"]
    q = np.array([160.0, 140.0])
    body = [T(180, 16, "3-way 5-shot: пространство эмбеддингов f(x)", "mut", fs=10)]
    protos = []
    for (cx, cy), c in zip(centers, cls):
        pts = np.array([cx, cy]) + rng.normal(0, 24, (5, 2))
        protos.append(pts.mean(0))
        for x, y in pts:
            body.append(f'<circle class="{c}" cx="{x:.1f}" cy="{y:.1f}" r="5"/>')
    d = [np.linalg.norm(q - pr) for pr in protos]
    best = int(np.argmin(d))
    for k, pr in enumerate(protos):
        if k == best:
            body.append(A(p, q[0], q[1], pr[0], pr[1], "ln s4", head=False))
        else:
            body.append(A(p, q[0], q[1], pr[0], pr[1], "ln s6", dash=True, head=False))
    for k, (pr, c) in enumerate(zip(protos, cls)):
        x, y = pr
        body.append(f'<rect class="{c}" x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" '
                    f'transform="rotate(45 {x:.1f} {y:.1f})" stroke-width="2.5"/>')
        body.append(T(x + 14, y - 10, f"c{'₁₂₃'[k]}", a="start", fs=12, bold=True))
    body += [f'<circle class="c4" cx="{q[0]}" cy="{q[1]}" r="7" stroke-width="2"/>', T(q[0] - 12, q[1] + 4, "q", a="end", bold=True),
             T(180, 262, "cₖ = (1/K)·Σ f(xᵢ),  xᵢ ∈ Sₖ (support)"),
             T(180, 282, "p(y = k | q) ∝ exp(−d(f(q), cₖ))"),
             T(180, 300, "кружки — support, ромбы — прототипы", "mut", fs=10)]
    return _svg(p, 360, 310, "Prototypical Networks", body)


# ================================================================ 27. fraud_detection
def svg_fraud():
    p = "frd"
    body = [
        Bx(90, 10, 180, 28, "c6", ["Транзакция"]), A(p, 180, 38, 180, 56),
        Bx(40, 56, 280, 40, "c1", ["Признаки из Feature Store", ("агрегаты 1ч / 24ч, отклонения", "", 11)]),
        A(p, 130, 96, 90, 118), A(p, 230, 96, 270, 118),
        Bx(20, 118, 140, 32, "c3", ["Rules"]), Bx(200, 118, 140, 32, "c5", ["ML: XGBoost / LGB"]),
        A(p, 90, 150, 140, 172), A(p, 270, 150, 220, 172),
        Bx(90, 172, 180, 28, "c2", ["Итоговый score s"]),
        A(p, 130, 200, 64, 230), A(p, 180, 200, 180, 230), A(p, 230, 200, 296, 230),
        Bx(12, 230, 104, 40, "c2", ["s &lt; t₁", ("одобрить", "", 11)]),
        Bx(128, 230, 104, 40, "c3", ["t₁ ≤ s &lt; t₂", ("ручная проверка", "", 11)]),
        Bx(244, 230, 104, 40, "c4", ["s ≥ t₂", ("блок", "", 11)]),
        T(180, 292, "метки (chargeback) → переобучение, мониторинг drift", "mut", fs=10),
    ]
    return _svg(p, 360, 302, "Real-time скоринг транзакций", body)


def chart_fraud(fig):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    rng = np.random.default_rng(8)
    n, prev = 20000, 0.01
    y = (rng.random(n) < prev).astype(int)
    z = rng.normal(0, 1, n) + 2.6 * y
    s = 1 / (1 + np.exp(-(z - 3)))
    pr, rc, _ = precision_recall_curve(y, s)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(rc, pr, color=B, lw=2, label=f"модель, AP = {average_precision_score(y, s):.2f}")
    ax.axhline(y.mean(), color=S, ls="--", label=f"случайная: {y.mean():.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("PR-кривая при 1% мошенничества", fontsize=9)
    ax.legend(fontsize=7)
    th = np.linspace(0.01, 0.99, 99)
    cfp, cfn = 5, 100
    cost = [cfp * np.sum((s >= t) & (y == 0)) + cfn * np.sum((s < t) & (y == 1)) for t in th]
    cost = np.array(cost) / 1000
    k = int(np.argmin(cost))
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(th, cost, color=V, lw=2)
    ax2.plot(th[k], cost[k], "o", color=R)
    ax2.annotate(f"минимум при t ≈ {th[k]:.2f}", (th[k], cost[k]), xytext=(10, 15), textcoords="offset points",
                 fontsize=8, color=R)
    ax2.set_xlabel("порог t")
    ax2.set_ylabel("потери, тыс.")
    ax2.set_title("Стоимость: FP = 5, FN = 100", fontsize=9)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ================================================================ 28. function_approximation_rl
def svg_tiles():
    p = "fap"
    body = ['<rect class="c1" x="100" y="70" width="40" height="40" opacity="0.7"/>',
            '<rect class="c3" x="113" y="83" width="40" height="40" opacity="0.7"/>',
            '<rect class="ln" x="20" y="30" width="160" height="160"/>']
    for k in range(1, 4):
        body.append(f'<line class="ln s1" x1="{20 + 40 * k}" y1="30" x2="{20 + 40 * k}" y2="190"/>')
        body.append(f'<line class="ln s1" x1="20" y1="{30 + 40 * k}" x2="180" y2="{30 + 40 * k}"/>')
    for k in range(4):
        body.append(f'<line class="ln s3" x1="{33 + 40 * k}" y1="30" x2="{33 + 40 * k}" y2="190" stroke-dasharray="4 3"/>')
        body.append(f'<line class="ln s3" x1="20" y1="{43 + 40 * k}" x2="180" y2="{43 + 40 * k}" stroke-dasharray="4 3"/>')
    body += ['<circle class="ah" cx="125" cy="100" r="4"/>', T(129, 97, "s", a="start", bold=True),
             T(100, 20, "пространство состояний", "mut", fs=10),
             T(196, 50, "— Тайлинг 1", "t1", "start", 11), T(196, 68, "- - Тайлинг 2 (сдвиг)", "t3", "start", 11),
             T(196, 100, "Активны 2 тайла:", a="start", fs=11),
             T(196, 118, "x(s) = [0…1…0 | 0…1…0]", a="start", fs=11),
             T(196, 146, "V̂(s, w) = wᵀx(s)", a="start", fs=12),
             T(196, 164, "сумма весов 2 тайлов", "mut", "start", 10),
             T(180, 214, "w ← w + α[R + γV̂(s′, w) − V̂(s, w)]∇V̂(s, w)", fs=11),
             T(180, 232, "для линейной V̂: ∇V̂(s, w) = x(s)", "mut", fs=10)]
    return _svg(p, 360, 244, "Tile coding", body)


def chart_fapprox(fig):
    s = np.linspace(0, 1, 200)
    Vt = np.sin(3 * np.pi * s) * np.exp(-s) + s
    rng = np.random.default_rng(9)
    xs = rng.uniform(0, 1, 60)
    ys = np.sin(3 * np.pi * xs) * np.exp(-xs) + xs + rng.normal(0, 0.08, 60)
    c = np.linspace(0, 1, 9)
    phi = lambda x: np.exp(-((x[:, None] - c) ** 2) / (2 * 0.07 ** 2))
    w, *_ = np.linalg.lstsq(phi(xs), ys, rcond=None)
    wl, *_ = np.linalg.lstsq(np.c_[np.ones_like(xs), xs], ys, rcond=None)
    ax = fig.add_subplot(111)
    Ph = phi(s)
    for j in range(len(c)):
        ax.plot(s, 0.25 * Ph[:, j] - 0.95, color=S, lw=0.8)
    ax.plot(s, Vt, color="k", lw=1.5, ls="--", label="истинная V(s)")
    ax.scatter(xs, ys, s=8, color=S, alpha=0.7)
    ax.plot(s, np.c_[np.ones_like(s), s] @ wl, color=O, lw=2, label="линейная: [1, s]")
    ax.plot(s, Ph @ w, color=B, lw=2, label="9 RBF-признаков")
    ax.set_xlabel("состояние s")
    ax.set_ylabel("V̂(s)")
    ax.set_title("Признаки решают, что может выучить V̂ = wᵀx(s)", fontsize=9)
    ax.set_ylim(-1.05, 2.1)
    ax.legend(fontsize=7, loc="upper center", ncol=2)


# ================================================================ 29. future_deep_learning
def svg_future():
    p = "fdl2"
    body = [
        Pth(p, "M86,56 L140,124", head=False), Pth(p, "M274,56 L220,124", head=False),
        Pth(p, "M110,140 L118,140", head=False), Pth(p, "M250,140 L242,140", head=False),
        Pth(p, "M86,224 L140,156", head=False), Pth(p, "M274,224 L220,156", head=False),
        '<ellipse class="c5" cx="180" cy="140" rx="62" ry="24"/>', T(180, 145, "Будущее DL", bold=True),
        Bx(16, 16, 140, 40, "c1", ["Scaling", ("больше данных и FLOPs", "", 10)]),
        Bx(204, 16, 140, 40, "c2", ["Эффективность", ("MoE, distillation", "", 10)]),
        Bx(6, 120, 104, 40, "c3", ["Multimodal", ("any-to-any", "", 10)]),
        Bx(250, 120, 104, 40, "c3", ["Embodied AI", ("роботы, сенсоры", "", 10)]),
        Bx(16, 224, 140, 40, "c4", ["World models", ("и reasoning", "", 10)]),
        Bx(204, 224, 140, 40, "c6", ["Continual learning", ("без забывания", "", 10)]),
        T(180, 290, "+ нейро-символьный AI, alignment, AI for Science", "mut", fs=10),
    ]
    return _svg(p, 360, 300, "Направления развития глубокого обучения", body)


# ================================================================ 30. gan
def svg_gan():
    p = "gan"
    body = [
        Bx(8, 36, 76, 32, "c6", ["z ~ N(0, I)"], fs=11), A(p, 84, 52, 102, 52),
        Bx(102, 32, 54, 40, "c3", ["G"], fs=14), A(p, 156, 52, 176, 52),
        Bx(178, 36, 74, 32, "c3", ["G(z)"]), T(215, 82, "fake", "mut", fs=10),
        Bx(178, 120, 74, 32, "c2", ["x ~ p_data"], fs=11), T(215, 166, "real", "mut", fs=10),
        A(p, 252, 52, 274, 78), A(p, 252, 136, 274, 106),
        Bx(276, 66, 64, 52, "c1", ["D"], fs=14),
        A(p, 308, 118, 308, 150), T(308, 166, "D(·) ∈ [0, 1]", fs=11),
        Pth(p, "M308,66 L308,22 L129,22 L129,30", "ln s4", dash=True),
        T(218, 15, "∇ через D → обновляем G", "t4", fs=10),
        T(180, 204, "min_G max_D 𝔼[log D(x)] + 𝔼[log(1 − D(G(z)))]", fs=11),
        T(180, 224, "D: real → 1, fake → 0;  G: обмануть D", "mut", fs=10),
    ]
    return _svg(p, 360, 234, "Схема GAN", body)


def chart_gan(fig):
    d = np.linspace(0.005, 0.995, 300)
    ax = fig.add_subplot(111)
    ax.plot(d, np.log(1 - d), color=R, lw=2, label="saturating: min log(1 − D(G(z)))")
    ax.plot(d, -np.log(d), color=B, lw=2, label="non-saturating: min −log D(G(z))")
    ax.axvspan(0, 0.1, color=S, alpha=0.15)
    ax.text(0.015, -3.6, "начало\nобучения", fontsize=8)
    ax.set_xlabel("D(G(z)) — насколько D верит подделке")
    ax.set_ylabel("loss генератора")
    ax.set_title("При D(G(z)) ≈ 0 saturating-loss почти плоский", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


# ================================================================ 31. generalization_theory_neural_networks
def svg_gen():
    p = "gtn"
    th = np.linspace(-3, 3, 240)
    L = lambda t: 1 - 0.85 * np.exp(-(t + 1.5) ** 2 / (2 * 0.15 ** 2)) - 0.85 * np.exp(-(t - 1.3) ** 2 / (2 * 0.6 ** 2))
    X = lambda t: 20 + (t + 3) * 320 / 6
    Y = lambda v: 188 - v * 150
    body = ['<line class="ln" x1="18" y1="190" x2="342" y2="190"/>', T(346, 194, "θ", a="start", fs=11),
            Pth(p, poly(X(th), Y(L(th))), "ln s1", head=False),
            Pth(p, poly(X(th), Y(L(th - 0.3))), "ln s3", dash=True, head=False),
            T(20, 16, "— train loss", "t1", "start", 11), T(190, 16, "- - test loss (сдвиг)", "t3", "start", 11)]
    for t0 in (-1.5, 1.3):
        body.append(A(p, X(t0), Y(L(t0)), X(t0), Y(L(t0 - 0.3)), "ln s4", head=False))
        body.append(f'<circle class="c1" cx="{X(t0):.1f}" cy="{Y(L(t0)):.1f}" r="3.5"/>')
    body += [T(X(-1.5), 208, "sharp-минимум", fs=11), T(X(-1.5), 222, "test ≫ train", "t4", fs=11),
             T(X(1.3), 208, "flat-минимум", fs=11), T(X(1.3), 222, "test ≈ train", "t2", fs=11)]
    return _svg(p, 360, 232, "Острые и плоские минимумы", body)


def chart_gen(fig):
    n, d = 40, 10
    ps = np.unique(np.round(np.logspace(np.log10(4), np.log10(400), 36)).astype(int))
    errs = np.zeros(len(ps))
    for seed in range(8):
        rng = np.random.default_rng(seed)
        wt = rng.normal(size=d) / np.sqrt(d)
        Xtr, Xte = rng.normal(size=(n, d)), rng.normal(size=(500, d))
        ytr = Xtr @ wt + 0.3 * rng.normal(size=n)
        yte = Xte @ wt
        W = rng.normal(size=(d, ps.max())) / np.sqrt(d)
        for i, pp in enumerate(ps):
            Ftr, Fte = np.maximum(Xtr @ W[:, :pp], 0), np.maximum(Xte @ W[:, :pp], 0)
            beta = np.linalg.pinv(Ftr) @ ytr
            errs[i] += np.mean((Fte @ beta - yte) ** 2) / 8
    ax = fig.add_subplot(111)
    ax.loglog(ps, errs, "o-", color=B, ms=3, lw=1.5)
    ax.axvline(n, color=R, ls="--", lw=1)
    ax.text(n * 1.5, errs.max() * 0.6, "p = n\n(interpolation\nthreshold)", fontsize=8, color=R)
    ax.set_xlabel("число параметров p (random ReLU features)")
    ax.set_ylabel("test MSE")
    ax.set_title(f"Double descent, n = {n} объектов", fontsize=10)


# ================================================================ 32. generative_models_classical
def svg_hmm():
    p = "gmc"
    xs = [50, 130, 210, 290]
    body = [T(180, 22, "A: P(sₜ | sₜ₋₁) — переходы", "t1", fs=11),
            A(p, 8, 70, 30, 70), T(12, 62, "π", "t5", fs=12)]
    for i, x in enumerate(xs):
        body += [f'<circle class="c1" cx="{x}" cy="70" r="18"/>', T(x, 75, f"s{'₁₂₃₄'[i]}"),
                 A(p, x, 88, x, 128), Bx(x - 17, 130, 34, 30, "c2", [f"o{'₁₂₃₄'[i]}"], rx=3)]
        if i < 3:
            body.append(A(p, x + 18, 70, xs[i + 1] - 20, 70))
    body += [A(p, 308, 70, 326, 70), T(338, 74, "…"),
             T(180, 182, "B: P(oₜ | sₜ) — эмиссии", "t2", fs=11),
             T(180, 214, "Генеративная: P(X, Y) = P(Y)·P(X | Y)", "t5", fs=12),
             T(180, 234, "Дискриминативная: только P(Y | X)", "mut", fs=11)]
    return _svg(p, 360, 244, "Hidden Markov Model", body)


def chart_kde(fig):
    from scipy.stats import norm
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(-1.5, 0.5, 60), rng.normal(1.5, 0.9, 90)])
    g = np.linspace(-4, 5, 400)
    true = 0.4 * norm.pdf(g, -1.5, 0.5) + 0.6 * norm.pdf(g, 1.5, 0.9)
    ax = fig.add_subplot(111)
    ax.fill_between(g, true, color=S, alpha=0.25, label="истинная плотность")
    for h, c in [(0.08, R), (0.4, B), (1.5, O)]:
        kde = norm.pdf((g[:, None] - x) / h).sum(1) / (len(x) * h)
        ax.plot(g, kde, color=c, lw=1.6, label=f"KDE, h = {h}")
    ax.plot(x, np.full_like(x, -0.015), "|", color="k", ms=6, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("плотность")
    ax.set_title("KDE: bandwidth h решает всё", fontsize=10)
    ax.legend(fontsize=7)


# ================================================================ 33. genetic_algorithms
def svg_ga():
    p = "gal"
    a = [1, 1, 0, 1, 0, 0, 1, 0]
    b = [0, 1, 1, 0, 1, 1, 0, 1]
    cut = 3
    c1 = a[:cut] + b[cut:]
    c2 = b[:cut] + a[cut:]
    m1 = c1.copy()
    m1[6] = 1 - m1[6]
    body = []

    def row(y, label, bits, src):
        out = [T(10, y + 16, label, a="start", fs=11)]
        for i, bit in enumerate(bits):
            out.append(Bx(110 + 26 * i, y, 26, 24, src[i], [str(bit)], rx=2))
        return out

    body += row(20, "Родитель A", a, ["c1"] * 8)
    body += row(50, "Родитель B", b, ["c3"] * 8)
    body += row(100, "Потомок 1", c1, ["c1"] * cut + ["c3"] * 5)
    body += row(130, "Потомок 2", c2, ["c3"] * cut + ["c1"] * 5)
    body += row(184, "Мутация", m1, ["c1"] * cut + ["c3"] * 3 + ["c4"] + ["c3"])
    body += [A(p, 188, 14, 188, 160, "ln s4", dash=True, head=False), T(188, 11, "точка разреза", "t4", fs=10),
             T(200, 90, "кроссовер ↓", "mut", "start", 10), T(200, 176, "мутация: бит 7, 0 → 1", "mut", "start", 10),
             T(170, 236, "селекция → кроссовер → мутация → замещение", fs=11),
             T(170, 254, "повтор, пока растёт fitness", "mut", fs=10)]
    return _svg(p, 340, 264, "Кроссовер и мутация", body)


def chart_ga(fig):
    rng = np.random.default_rng(10)
    L, N, gens = 60, 60, 80
    pop = rng.integers(0, 2, (N, L))
    best, mean = [], []
    for _ in range(gens):
        fit = pop.sum(1)
        best.append(fit.max())
        mean.append(fit.mean())
        elite = pop[np.argsort(fit)[-2:]]
        i, j = rng.integers(0, N, (2, N))
        par = np.where((fit[i] >= fit[j])[:, None], pop[i], pop[j])
        mates = par[rng.permutation(N)]
        cut = rng.integers(1, L, N)
        do = rng.random(N) < 0.8
        mask = (np.arange(L) < cut[:, None]) | ~do[:, None]
        child = np.where(mask, par, mates)
        flip = rng.random((N, L)) < 1 / L
        child = child ^ flip
        child[:2] = elite
        pop = child
    ax = fig.add_subplot(111)
    ax.plot(best, color=G, lw=2, label="лучшая особь")
    ax.plot(mean, color=B, lw=2, label="среднее по популяции")
    ax.axhline(L, color=S, ls="--", lw=1, label=f"оптимум = {L}")
    ax.set_xlabel("поколение")
    ax.set_ylabel("fitness (число единиц)")
    ax.set_title(f"OneMax, {L} бит: турнир + кроссовер + мутация 1/L", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")


# ================================================================ 34. geoinformation_systems_ml
def svg_geo():
    p = "gis"
    bands = [("B2", "Blue", "c1"), ("B3", "Green", "c2"), ("B4", "Red", "c4"), ("B8", "NIR", "c5")]
    body = []
    for i, (a, b, c) in enumerate(bands):
        body.append(Bx(12 + 86 * i, 12, 78, 36, c, [a, (b, "", 10)], fs=11))
    body += [A(p, 223, 48, 210, 76), A(p, 309, 48, 290, 76),
             Bx(140, 76, 208, 40, "c3", [("NDVI = (NIR − Red)/(NIR + Red)", "", 11), ("≈0.8 растения, &lt;0 вода", "mut", 10)]),
             A(p, 51, 48, 56, 138), A(p, 137, 48, 96, 138), A(p, 170, 116, 124, 140),
             Bx(12, 140, 120, 44, "c1", ["U-Net", ("сегментация", "", 11)]),
             A(p, 132, 166, 166, 178)]
    grid = ["FFFAAA", "FFWAAA", "FWWAUU", "WWAAUU", "WAAUUU", "AAAUUU"]
    cmap = {"F": "c2", "W": "c1", "U": "c6", "A": "c3"}
    for r, rowv in enumerate(grid):
        for c, ch in enumerate(rowv):
            body.append(f'<rect class="{cmap[ch]}" x="{170 + 16 * c}" y="{134 + 16 * r}" width="16" height="16"/>')
    for i, (ch, s) in enumerate([("F", "лес"), ("W", "вода"), ("U", "город"), ("A", "поле")]):
        y = 140 + 20 * i
        body += [f'<rect class="{cmap[ch]}" x="284" y="{y}" width="12" height="12" rx="2"/>', T(300, y + 10, s, a="start", fs=11)]
    body.append(T(218, 248, "карта land cover", "mut", fs=10))
    return _svg(p, 360, 258, "Мультиспектральные каналы, NDVI и сегментация", body)


def chart_geo(fig):
    wl = np.linspace(450, 900, 300)
    veg = np.interp(wl, [450, 500, 550, 600, 670, 700, 740, 800, 900], [0.04, 0.05, 0.11, 0.07, 0.04, 0.12, 0.42, 0.48, 0.50])
    soil = np.interp(wl, [450, 900], [0.10, 0.30])
    water = np.interp(wl, [450, 550, 700, 800, 900], [0.07, 0.05, 0.02, 0.01, 0.005])
    ax = fig.add_subplot(111)
    for y, c, lab in [(veg, G, "растительность"), (soil, O, "почва"), (water, B, "вода")]:
        r4, r8 = np.interp(665, wl, y), np.interp(842, wl, y)
        ax.plot(wl, y, color=c, lw=2, label=f"{lab}: NDVI ≈ {(r8 - r4) / (r8 + r4):.2f}")
    for x, lab in [(665, "B4 Red"), (842, "B8 NIR")]:
        ax.axvline(x, color=S, ls="--", lw=1)
        ax.text(x + 4, 0.70, lab, fontsize=8)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("длина волны, нм")
    ax.set_ylabel("отражение")
    ax.set_title("Спектры (схематично) и NDVI", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


# ================================================================ VISUALS
VISUALS = {
    "deep_learning_timeline": [
        {"title": "Ключевые вехи", "svg": svg_timeline(),
         "caption": "Красные точки — «зимы AI», когда нейросети теряли популярность; после 2012 года прорывы идут каждые 2–3 года."},
        {"title": "Рост числа параметров", "chart": chart_timeline,
         "caption": "Шкала логарифмическая: от LeNet-5 (~60 тыс.) до GPT-3 (175 млрд) — рост примерно в 3 млн раз за 22 года."},
    ],
    "defensive_distillation": [
        {"title": "Три этапа защиты", "svg": svg_distill(),
         "caption": "Teacher и student учатся при высокой T, а работают при T = 1: логиты оказываются большими, softmax "
                    "насыщается, и градиент по входу почти исчезает — атакам вроде FGSM не за что зацепиться."},
        {"title": "Softmax с температурой", "chart": chart_distill,
         "caption": "При T = 1 почти вся масса у одного класса; при T = 20 распределение мягкое и несёт «тёмное знание» "
                    "о сходстве классов."},
    ],
    "demand_traffic_forecasting": [
        {"title": "От данных к решениям", "svg": svg_demand(),
         "caption": "Внешние факторы входят в модель через признаки; результат прогноза — не число, а интервал для "
                    "планирования запасов."},
        {"title": "Прогноз с недельной сезонностью", "chart": chart_demand,
         "caption": "Регрессия на тренд и день недели повторяет недельный цикл, но праздничный всплеск она не "
                    "предсказала бы — праздники нужно добавлять отдельным признаком."},
    ],
    "differentiable_programming": [
        {"title": "Граф вычислений и reverse mode", "svg": svg_diffprog(),
         "caption": "Forward pass считает значения слева направо; reverse mode идёт обратно и по правилу цепочки "
                    "суммирует вклады: x₁ получает градиент и через ×, и через sin."},
        {"title": "AD точнее конечных разностей", "chart": chart_diffprog,
         "caption": "У численной производной есть оптимальный шаг h: при большом h растёт ошибка усечения, при малом — "
                    "ошибка округления. AD точна до машинной точности при любом h."},
    ],
    "differential_privacy": [
        {"title": "Механизм Лапласа", "svg": svg_dp(),
         "caption": "Соседние наборы данных D и D′ дают почти одинаковые распределения ответа: по выходу нельзя "
                    "уверенно сказать, была ли в данных конкретная запись."},
        {"title": "Шум и privacy loss", "chart": chart_dp,
         "caption": "Меньше ε — шире шум (scale = Δf/ε) и сильнее защита. Логарифм отношения плотностей для D и D′ "
                    "никогда не выходит за ±ε — это и есть ε-DP."},
    ],
    "diffusion_models": [
        {"title": "Прямой и обратный процесс", "svg": svg_diffusion(),
         "caption": "Прямой процесс q за T шагов превращает изображение в чистый шум; сеть ε_θ учится предсказывать "
                    "добавленный шум и при генерации проходит цепочку обратно от x_T к x₀."},
        {"title": "Linear vs cosine schedule", "chart": chart_diffusion,
         "caption": "При linear-расписании сигнал исчезает уже к середине цепочки, и последние шаги почти "
                    "бесполезны; cosine убывает плавнее."},
    ],
    "digital_humanities": [
        {"title": "Пайплайн исследования", "svg": svg_dh(),
         "caption": "Качество OCR определяет всё дальнейшее; ML-методы находят паттерны, но выводы делает гуманитарий."},
        {"title": "Атрибуция авторства", "chart": chart_dh,
         "caption": "Синтетический пример: фрагменты двух авторов разделяются по частотам служебных слов, а "
                    "анонимные фрагменты ложатся в облако автора A."},
    ],
    "distributed_training": [
        {"title": "Data Parallelism", "svg": svg_dist(),
         "caption": "Каждый GPU считает градиент на своём шарде; all_reduce усредняет градиенты, поэтому после шага "
                    "копии весов совпадают."},
        {"title": "Pipeline bubble", "chart": chart_dist,
         "caption": "В pipeline parallelism GPU простаивают при заполнении и опустошении конвейера; чем больше "
                    "micro-batches m на шаг, тем меньше доля простоя."},
    ],
    "dynamic_programming_rl": [
        {"title": "Generalized Policy Iteration", "svg": svg_dprl(),
         "caption": "Evaluation подгоняет V под текущую π, improvement делает π жадной относительно V. Policy "
                    "Iteration доводит оценку до сходимости, Value Iteration делает один шаг."},
        {"title": "Сходимость Value Iteration", "chart": chart_dprl,
         "caption": "Случайный MDP с 20 состояниями: оператор Беллмана — сжатие с коэффициентом γ, поэтому при "
                    "γ = 0.99 нужно на порядки больше итераций, чем при γ = 0.5."},
    ],
    "early_late_fusion": [
        {"title": "Early vs Late fusion", "svg": svg_fusion(),
         "caption": "Early fusion склеивает признаки до модели и видит связи модальностей; late fusion объединяет "
                    "только выходы отдельных моделей, зато переживает пропуск модальности."},
    ],
    "efficient_transformers": [
        {"title": "Паттерны внимания", "svg": svg_efftr(),
         "caption": "Цветные клетки — пары (query, key), для которых считается attention. Local оставляет диагональ, "
                    "Longformer добавляет глобальный токен (оранжевый), BigBird — ещё и случайные связи (красные)."},
        {"title": "Память под attention", "chart": chart_efftr,
         "caption": "При n = 64K полная матрица в 128 раз больше, чем у окна w = 512; линейные методы растут "
                    "пропорционально n."},
    ],
    "em_algorithm": [
        {"title": "EM как подъём по ELBO", "svg": svg_em(),
         "caption": "E-шаг строит нижнюю оценку, касающуюся log p(X|θ) в θᵗ; M-шаг находит её максимум θᵗ⁺¹. "
                    "Правдоподобие в новой точке не меньше старого."},
        {"title": "EM для смеси двух гауссиан", "chart": chart_em,
         "caption": "Из плохой стартовой точки компоненты разъезжаются к истинным модам; log-likelihood растёт "
                    "монотонно и выходит на плато."},
    ],
    "encoder_decoder": [
        {"title": "Seq2seq: encoder → context → decoder", "svg": svg_encdec(),
         "caption": "Encoder сжимает вход в вектор c, decoder разворачивает его в выход по одному токену. При "
                    "обучении на вход decoder подаётся истинный предыдущий токен (teacher forcing)."},
    ],
    "ensemble_diversification": [
        {"title": "Почему важны разные ошибки", "svg": svg_ensdiv(),
         "caption": "Каждая модель верна на 70%. Если ошибки в разных местах, голосование даёт 100%; если ошибки "
                    "совпадают, ансамбль ничего не исправляет."},
        {"title": "Дисперсия среднего vs корреляция", "chart": chart_ensdiv,
         "caption": "При корреляции ρ дисперсия ансамбля не опускается ниже ρσ² — добавлять модели бесполезно, "
                    "нужно снижать корреляцию."},
    ],
    "ensemble_methods_streaming": [
        {"title": "Online Bagging с детекцией drift", "svg": svg_stream(),
         "caption": "Каждое дерево учится на примере k раз, где k ~ Poisson(λ), — это имитирует bootstrap в потоке. "
                    "ADWIN следит за ошибкой и при drift заменяет дерево фоновым."},
        {"title": "Распределение Poisson(λ)", "chart": chart_stream,
         "caption": "При λ = 1 около 37% моделей пропускают пример (k = 0); Leveraging Bagging с λ = 6 даёт "
                    "примерам больший вес и больше разнообразия."},
    ],
    "ensemble_methods_voting_stacking": [
        {"title": "Stacking на OOF-предсказаниях", "svg": svg_stack(),
         "caption": "Meta-model учится только на out-of-fold предсказаниях: базовые модели не видели эти объекты, "
                    "поэтому нет утечки."},
        {"title": "Hard vs soft voting", "chart": chart_voting,
         "caption": "Две неуверенные модели (0.45) перевешивают уверенную в hard voting, а в soft voting "
                    "решает средняя вероятность 0.60."},
    ],
    "ensemble_tuning": [
        {"title": "Порядок настройки", "svg": svg_enstune(),
         "caption": "Веса и мета-модель подбираются на OOF-предсказаниях, а holdout используется один раз в конце; "
                    "если результат хуже, возвращаемся к составу ансамбля."},
        {"title": "Learning rate и n_estimators", "chart": chart_enstune,
         "caption": "Большой learning_rate быстро выходит на плато с высокой ошибкой; маленький сходится медленно и за "
                    "300 деревьев ещё не дошёл до минимума — эти параметры подбирают вместе. Точки — лучшая итерация."},
    ],
    "evolutionary_strategies": [
        {"title": "Одно поколение ES", "svg": svg_es(),
         "caption": "Градиент не нужен: θ сдвигается в сторону шумовых направлений εᵢ, которые дали высокую "
                    "fitness. Оценки Fᵢ независимы и легко параллелятся."},
        {"title": "ES на вытянутой квадратичной функции", "chart": chart_es,
         "caption": "Популяция (точки) смещается к минимуму (звезда) по оценке градиента из случайных возмущений; "
                    "размер облака задаёт σ."},
    ],
    "explainability_accountability": [
        {"title": "Типы объяснений и подотчётность", "svg": svg_xai(),
         "caption": "Методы делятся по двум осям: всё поведение модели или одно решение, любая модель или "
                    "конкретный класс моделей. Accountability добавляет к объяснениям процесс исправления."},
    ],
    "fairness_constraints": [
        {"title": "Три точки вмешательства", "svg": svg_fcon(),
         "caption": "Pre-processing меняет данные, in-processing — функцию потерь, post-processing — только пороги "
                    "готовой модели."},
        {"title": "Пороги для Equal Opportunity", "chart": chart_fcon,
         "caption": "При общем пороге 0.5 TPR групп различаются; разные пороги по группам выравнивают TPR на "
                    "уровне 0.8."},
    ],
    "fairness_ml": [
        {"title": "Feedback loop", "svg": svg_fml(),
         "caption": "Решения модели порождают её будущие обучающие данные, и исходный перекос подтверждает сам себя."},
    ],
    "fastai": [
        {"title": "От данных до модели", "svg": svg_fastai(),
         "caption": "fine_tune() сначала обучает новую голову на замороженном backbone, затем размораживает все слои "
                    "и учит их с discriminative learning rates."},
        {"title": "One-cycle policy", "chart": chart_fastai,
         "caption": "LR растёт от lr_max/25 до lr_max за первые 25% итераций, затем убывает косинусом почти до нуля; "
                    "momentum в это время меняется в противофазе."},
    ],
    "feature_interaction": [
        {"title": "PolynomialFeatures(degree=2)", "svg": svg_finter(),
         "caption": "Из двух признаков получается шесть: константа, исходные, квадраты и произведение x₁x₂ — "
                    "единственное настоящее взаимодействие."},
        {"title": "XOR: линейной модели нужен x₁·x₂", "chart": chart_finter,
         "caption": "Класс зависит от знака x₁·x₂. Без взаимодействия логистическая регрессия проводит одну прямую и "
                    "даёт лишь около 60% accuracy; с признаком x₁·x₂ задача решается почти идеально."},
    ],
    "feature_stores": [
        {"title": "Offline и online store", "svg": svg_fstore(),
         "caption": "Одни и те же трансформации наполняют offline store для обучения и online store для inference — "
                    "так устраняется training-serving skew."},
    ],
    "federated_learning": [
        {"title": "Раунд FedAvg", "svg": svg_fl(),
         "caption": "Сервер рассылает wₜ, клиенты обучают модель на своих данных и возвращают только веса; сервер "
                    "усредняет их с весами nₖ/n."},
    ],
    "few_shot_learning_traditional": [
        {"title": "Prototypical Networks", "svg": svg_fewshot(),
         "caption": "Прототип класса — среднее эмбеддингов его support-примеров; query относится к ближайшему "
                    "прототипу (сплошная линия)."},
    ],
    "fraud_detection": [
        {"title": "Real-time скоринг", "svg": svg_fraud(),
         "caption": "Правила и ML-модель работают вместе, а два порога делят транзакции на три исхода; спорные "
                    "случаи уходят на ручную проверку."},
        {"title": "PR-кривая и выбор порога", "chart": chart_fraud,
         "caption": "При 1% мошенничества точность случайной модели — 0.01, поэтому PR AUC информативнее ROC AUC. "
                    "Порог выбирают по бизнес-стоимости FP и FN, а не по умолчанию 0.5."},
    ],
    "function_approximation_rl": [
        {"title": "Tile coding", "svg": svg_tiles(),
         "caption": "Каждый тайлинг активирует ровно один тайл; сдвинутые сетки дают точное разрешение, а размер "
                    "тайла задаёт, насколько широко обобщается обновление."},
        {"title": "Линейная аппроксимация V(s)", "chart": chart_fapprox,
         "caption": "Модель линейна по w в обоих случаях, но признаки [1, s] дают только прямую, а 9 RBF-признаков "
                    "(серые колокола внизу) повторяют форму V."},
    ],
    "future_deep_learning": [
        {"title": "Основные направления", "svg": svg_future(),
         "caption": "Масштабирование соседствует с поиском эффективности; мультимодальность, embodied AI и world "
                    "models ведут к пониманию физического мира."},
    ],
    "gan": [
        {"title": "Генератор против дискриминатора", "svg": svg_gan(),
         "caption": "D учится отличать real от fake, а G получает градиент только через D (красный пунктир) и учится "
                    "его обманывать."},
        {"title": "Saturating vs non-saturating loss", "chart": chart_gan,
         "caption": "В начале обучения D легко распознаёт подделки (D(G(z)) ≈ 0), и у log(1 − D) почти нет "
                    "градиента; −log D там круто падает, поэтому на практике G учат именно так."},
    ],
    "generalization_theory_neural_networks": [
        {"title": "Flat vs sharp minima", "svg": svg_gen(),
         "caption": "Test loss — тот же ландшафт, немного сдвинутый. В остром минимуме сдвиг резко поднимает ошибку, "
                    "в плоском почти не меняет — поэтому SGD, предпочитающий плоские минимумы, обобщает лучше."},
        {"title": "Double descent", "chart": chart_gen,
         "caption": "Регрессия с минимальной нормой на случайных признаках: ошибка достигает пика у порога "
                    "интерполяции p = n и снова падает в сильно перепараметризованном режиме."},
    ],
    "generative_models_classical": [
        {"title": "HMM как генеративная модель", "svg": svg_hmm(),
         "caption": "Скрытые состояния образуют марковскую цепь (A), каждое порождает наблюдение (B). Генеративная "
                    "модель описывает, как возникают данные, и поэтому умеет их сэмплировать."},
        {"title": "Kernel Density Estimation", "chart": chart_kde,
         "caption": "Малый h даёт шумные пики на каждой точке, большой h сливает две моды в одну; хороший h близок к "
                    "истинной плотности."},
    ],
    "genetic_algorithms": [
        {"title": "Кроссовер и мутация", "svg": svg_ga(),
         "caption": "Одноточечный кроссовер обменивает хвосты хромосом после точки разреза, мутация инвертирует "
                    "случайный бит и возвращает утраченное разнообразие."},
        {"title": "Сходимость на OneMax", "chart": chart_ga,
         "caption": "Благодаря элитизму лучшая fitness не убывает; среднее по популяции тянется за ней, а "
                    "последние биты находятся медленнее всего."},
    ],
    "geoinformation_systems_ml": [
        {"title": "От каналов к карте", "svg": svg_geo(),
         "caption": "Каналы Sentinel-2 и индексы вроде NDVI подаются в U-Net, которая выдаёт класс поверхности для "
                    "каждого пикселя."},
        {"title": "Почему NDVI работает", "chart": chart_geo,
         "caption": "Растительность слабо отражает красный и сильно — ближний ИК, поэтому её NDVI высокий; у воды "
                    "NDVI отрицательный. Кривые схематичные."},
    ],
}
