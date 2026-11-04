# -*- coding: utf-8 -*-
"""Рисунки для шпаргалок группы g0 (3d_computer_vision … canonical_correlation_analysis_method)."""

import numpy as np

P = ["#1a5fb4", "#2e8b57", "#e07b00", "#d32f2f", "#7048b8", "#8a96a3"]
BLUE, GREEN, ORANGE, RED, PURPLE, GRAY = P


# ---------- помощники для SVG ----------

def svg(topic, w, h, body, label):
    defs = (f'<defs><marker id="{topic}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>')
    body = body.replace("MK", f"url(#{topic}-a)")
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{label}">{defs}{body}</svg>')


def box(x, y, w, h, cls, t, sub=None, fs=None, rx=6):
    f = f' font-size="{fs}"' if fs else ""
    s = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'
    cx = x + w / 2
    if sub:
        s += (f'<text x="{cx}" y="{y + h / 2 - 2}" text-anchor="middle"{f}>{t}</text>'
              f'<text class="mut" x="{cx}" y="{y + h / 2 + 12}" text-anchor="middle" font-size="10">{sub}</text>')
    else:
        s += f'<text x="{cx}" y="{y + h / 2 + 4}" text-anchor="middle"{f}>{t}</text>'
    return s


def ar(x1, y1, x2, y2, cls="ln", dash=False, both=False, head=True):
    d = ' stroke-dasharray="4 3"' if dash else ""
    m = ' marker-end="MK"' if head else ""
    if both:
        m += ' marker-start="MK"'
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{d}{m}/>'


def pa(d, cls="ln", dash=False, head=True):
    ds = ' stroke-dasharray="4 3"' if dash else ""
    m = ' marker-end="MK"' if head else ""
    return f'<path class="{cls}" d="{d}"{ds}{m}/>'


def tx(x, y, t, cls=None, anchor="middle", fs=None, extra=""):
    c = f' class="{cls}"' if cls else ""
    f = f' font-size="{fs}"' if fs else ""
    return f'<text{c} x="{x}" y="{y}" text-anchor="{anchor}"{f}{extra}>{t}</text>'


def dot(x, y, cls="t1", r=3.5):
    return f'<circle class="{cls}" cx="{x}" cy="{y}" r="{r}"/>'


def cycle4(texts, subs, classes, back_label=None):
    """Четыре блока по кругу: верх-лево → верх-право → низ-право → низ-лево → верх-лево."""
    (x1, x2), w, h, y1, y2 = (8, 182), 150, 40, 15, 120
    pos = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    s = "".join(box(x, y, w, h, c, t, sb) for (x, y), t, sb, c in zip(pos, texts, subs, classes))
    s += ar(158, 35, 180, 35) + ar(257, 55, 257, 118) + ar(182, 140, 160, 140) + ar(83, 120, 83, 57)
    return s


# ---------- 3D Computer Vision ----------

S_3D = svg("3d_computer_vision", 340, 232, "".join([
    ar(40, 130, 318, 130), tx(326, 134, "Z", anchor="middle"),
    '<line class="s1" x1="110" y1="56" x2="110" y2="190" stroke-width="2.5"/>',
    tx(110, 48, "плоскость изображения", "mut", fs=10),
    '<line class="ln s3" x1="290" y1="60" x2="40" y2="130"/>',
    ar(290, 60, 290, 128, "ln s6", dash=True, head=False), tx(296, 100, "Y", anchor="start"),
    ar(110, 110.4, 110, 130, "ln s6", dash=True, head=False), tx(104, 124, "y", anchor="end"),
    dot(290, 60, "t4", 4.5), tx(330, 48, "P = (X, Y, Z)", anchor="end"),
    dot(110, 110.4, "t1", 4), tx(118, 125, "p = (x, y)", anchor="start", fs=11),
    '<circle class="c6" cx="40" cy="130" r="6"/>', tx(40, 116, "C"),
    ar(42, 152, 288, 152, both=True), tx(165, 147, "Z (глубина)", "mut", fs=10),
    ar(42, 172, 108, 172, both=True), tx(75, 186, "f", "mut", fs=11),
    tx(222, 182, "x = f·X / Z,  y = f·Y / Z"),
    tx(222, 202, "p ≃ K [R | t] P", fs=12),
    tx(222, 220, "K — внутренние, [R|t] — внешние", "mut", fs=10),
]), "Модель камеры-обскуры")


# ---------- Activation functions ----------

def _nb(cx, cy, x, y, r=24):
    v = np.array([cx - x, cy - y], float)
    v /= np.linalg.norm(v)
    return round(cx - r * v[0], 1), round(cy - r * v[1], 1)


S_ACT = svg("activation_functions", 340, 165, "".join(
    [tx(20, y + 4, f"x{s}") for y, s in [(40, "₁"), (80, "₂"), (120, "₃")]]
    + [ar(32, y, *_nb(130, 80, 32, y)) for y in (40, 80, 120)]
    + [tx(66, 48, "w₁", "mut", fs=10), tx(66, 75, "w₂", "mut", fs=10), tx(66, 122, "w₃", "mut", fs=10),
       '<circle class="c1" cx="130" cy="80" r="22"/>', tx(130, 85, "Σ", fs=14),
       tx(130, 124, "z = wᵀx + b", "mut", fs=11),
       ar(152, 80, 178, 80), box(180, 62, 70, 36, "c3", "f(z)"), tx(215, 114, "нелинейность", "mut", fs=10),
       ar(250, 80, 288, 80), tx(296, 84, "a = f(z)", anchor="start"),
       tx(170, 152, "без f: W₂(W₁x) = (W₂W₁)x — снова линейно", "mut", fs=11)]),
    "Место функции активации в нейроне")


def act_chart(fig):
    from scipy.special import erf
    x = np.linspace(-4, 4, 400)
    sig = 1 / (1 + np.exp(-x))
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    fs = [("Sigmoid", sig, sig * (1 - sig), BLUE), ("Tanh", np.tanh(x), 1 - np.tanh(x) ** 2, GREEN),
          ("ReLU", np.maximum(0, x), (x > 0).astype(float), RED),
          ("Leaky ReLU (α=0.1)", np.where(x > 0, x, 0.1 * x), np.where(x > 0, 1, 0.1), ORANGE),
          ("GELU", x * Phi, Phi + x * phi, PURPLE)]
    ax = fig.add_subplot(2, 1, 1)
    for n, f, _, c in fs:
        ax.plot(x, f, color=c, lw=1.8, label=n)
    ax.set_ylim(-1.5, 3), ax.set_title("Функции активации", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax2 = fig.add_subplot(2, 1, 2)
    for n, _, d, c in fs:
        ax2.plot(x, d, color=c, lw=1.8, label=n)
    ax2.set_ylim(-0.2, 1.25), ax2.set_xlabel("x")
    ax2.set_title("Производные: у sigmoid max = 0.25", fontsize=10)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- Active learning ----------

S_AL = svg("active_learning", 340, 205, cycle4(
    ["Размеченные L", "Модель", "Query strategy", "Оракул"],
    ["малый стартовый набор", "fit(L)", "x* = argmax инф-сти в U", "человек даёт y*"],
    ["c2", "c1", "c3", "c5"])
    + tx(170, 85, "L ← L ∪ {(x*, y*)}", "mut", fs=11) + tx(170, 102, "U ← U \\ {x*}", "mut", fs=11)
    + tx(170, 192, "стоп: исчерпан бюджет разметки или цель", "mut", fs=11),
    "Цикл активного обучения")


def al_chart(fig):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=1200, n_features=10, n_informative=5, n_redundant=0,
                               class_sep=1.5, flip_y=0.0, random_state=1)
    Xp, yp, Xt, yt = X[:1000], y[:1000], X[1000:], y[1000:]
    ax = fig.add_subplot(1, 1, 1)
    sizes = np.arange(10, 91, 10)
    for name, c in [("Random", GRAY), ("Uncertainty", BLUE)]:
        accs = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            lab = list(np.r_[rng.choice(np.where(yp == 0)[0], 5, False), rng.choice(np.where(yp == 1)[0], 5, False)])
            a = []
            while True:
                m = LogisticRegression(solver="liblinear").fit(Xp[lab], yp[lab])
                a.append(m.score(Xt, yt))
                if len(lab) >= sizes[-1]:
                    break
                pool = np.setdiff1d(np.arange(1000), lab)
                if name == "Random":
                    new = rng.choice(pool, 10, False)
                else:
                    pr = m.predict_proba(Xp[pool])[:, 1]
                    new = pool[np.argsort(np.abs(pr - 0.5))[:10]]
                lab += list(new)
            accs.append(a)
        ax.plot(sizes, np.mean(accs, 0), "o-", ms=3, color=c, lw=2, label=name)
    ax.set_xlabel("число размеченных примеров"), ax.set_ylabel("accuracy на test")
    ax.set_title("Uncertainty sampling быстрее\nвыходит на плато", fontsize=10)
    ax.legend(fontsize=8)


# ---------- Adaptive windowing ----------

S_ADW = svg("adaptive_windowing", 340, 232, "".join(
    [f'<rect class="{"c6" if i < 12 else "c3"}" x="{20 + 15 * i}" y="36" width="14" height="18" rx="2"/>'
     for i in range(20)]
    + [tx(110, 28, "W₀ (старые)"), tx(260, 28, "W₁ (новые)"),
       ar(199.5, 30, 199.5, 60, "ln s4", dash=True, head=False),
       ar(20, 68, 318, 68), tx(20, 84, "время →", "mut", "start", fs=10), tx(320, 84, "xₜ", "mut", "end", fs=10),
       box(60, 96, 220, 30, "c1", "|μ̂₀ − μ̂₁| ≥ ε_cut ?"),
       ar(120, 126, 85, 150), ar(220, 126, 255, 150),
       box(10, 152, 150, 34, "c4", "да: drift, удалить W₀", fs=11),
       box(180, 152, 150, 34, "c2", "нет: окно растёт", fs=11),
       tx(170, 208, "ε_cut = √( ln(4/δ′) / 2m ),  δ′ = δ / |W|", "mut", fs=11),
       tx(170, 224, "m — гармоническое среднее |W₀| и |W₁|", "mut", fs=11)]),
    "Схема ADWIN")


def _adwin(x, delta=0.002, minn=10, every=4):
    W, sizes, means, drifts = [], [], [], []
    for t, v in enumerate(x):
        W.append(v)
        if t % every == 0 and len(W) >= 2 * minn:
            a = np.asarray(W)
            n = len(a)
            cs = np.cumsum(a)
            n0 = np.arange(minn, n - minn + 1)
            n1 = n - n0
            d = np.abs(cs[n0 - 1] / n0 - (cs[-1] - cs[n0 - 1]) / n1)
            m = 1 / (1 / n0 + 1 / n1)
            eps = np.sqrt(np.log(4 * n / delta) / (2 * m))
            gap = d - eps
            if gap.max() > 0:
                W = W[n0[np.argmax(gap)]:]
                drifts.append(t)
        sizes.append(len(W))
        means.append(np.mean(W))
    return np.array(sizes), np.array(means), drifts


def adw_chart(fig):
    rng = np.random.default_rng(3)
    mu = np.r_[np.full(300, 0.25), np.full(300, 0.65), np.linspace(0.65, 0.35, 100), np.full(200, 0.35)]
    x = np.clip(mu + rng.normal(0, 0.12, mu.size), 0, 1)
    sizes, means, drifts = _adwin(x)
    t = np.arange(x.size)
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(t, x, s=2, color=GRAY, alpha=0.5)
    ax.plot(t, mu, "--", color="k", lw=1, label="истинное среднее")
    ax.plot(t, means, color=BLUE, lw=2, label="среднее по окну W")
    for d in drifts:
        ax.axvline(d, color=RED, lw=1, alpha=0.7)
    ax.set_title("Резкий drift (t=300) и плавный (t=600–700)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    ax2.plot(t, sizes, color=ORANGE, lw=2)
    for d in drifts:
        ax2.axvline(d, color=RED, lw=1, alpha=0.7)
    ax2.set_ylabel("|W|"), ax2.set_xlabel("время t")
    ax2.set_title("Размер окна: сброс при drift (красные линии)", fontsize=10)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- Advanced ensembling (stacking) ----------

S_ENS = svg("advanced_ensembling", 340, 240, "".join([
    box(120, 8, 100, 30, "c6", "X_train"), tx(228, 28, "cv = 5", "mut", "start", fs=11),
    ar(150, 38, 62, 58), ar(170, 38, 170, 58), ar(190, 38, 278, 58),
    box(15, 60, 90, 30, "c1", "RF"), box(125, 60, 90, 30, "c1", "SVM"), box(235, 60, 90, 30, "c1", "LR"),
    ar(60, 90, 60, 113), ar(170, 90, 170, 113), ar(280, 90, 280, 113),
    box(30, 115, 280, 38, "c3", "OOF-предсказания [p_rf, p_svm, p_lr]", "мета-признаки, без утечки"),
    ar(170, 153, 170, 173),
    box(95, 175, 150, 30, "c2", "мета-модель (LR)"),
    ar(170, 205, 170, 226), tx(178, 234, "ŷ", anchor="start"),
]), "Схема стекинга")


# ---------- Adversarial attacks ----------

S_ADV = svg("adversarial_attacks", 340, 172, "".join([
    box(6, 20, 86, 64, "c2", "x", "«панда» 57.7%", fs=14),
    tx(108, 57, "+ ε·"),
    box(124, 20, 86, 64, "c6", "sign(∇ₓL)", "шум, ε = 0.007", fs=11),
    tx(226, 57, "="),
    box(242, 20, 86, 64, "c4", "x_adv", "«гиббон» 99.3%"),
    tx(170, 118, "x_adv = clip(x + ε·sign(∇ₓL(θ, x, y)), 0, 1)", fs=11),
    tx(170, 140, "‖x_adv − x‖∞ ≤ ε: глазу не видно", "mut", fs=11),
    tx(170, 158, "один шаг в сторону роста loss (FGSM)", "mut", fs=11),
]), "FGSM: панда превращается в гиббона")


def _lin_data(seed=0, n=1000, d=20):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    X, y = make_classification(n_samples=n, n_features=d, n_informative=10, n_redundant=0, class_sep=1.5, random_state=seed)
    return StandardScaler().fit_transform(X), y


def adv_chart(fig):
    from sklearn.linear_model import LogisticRegression
    X, y = _lin_data()
    Xtr, ytr, Xte, yte = X[:700], y[:700], X[700:], y[700:]
    m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    w = m.coef_[0]
    p = m.predict_proba(Xte)[:, 1]
    g = np.sign((p - yte)[:, None] * w[None, :])     # знак ∇ₓ лосса
    rng = np.random.default_rng(0)
    rs = rng.choice([-1, 1], size=Xte.shape)
    eps = np.linspace(0, 0.5, 26)
    fg = [m.score(Xte + e * g, yte) for e in eps]
    rd = [m.score(Xte + e * rs, yte) for e in eps]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(eps, fg, "o-", ms=3, color=RED, lw=2, label="FGSM: ε·sign(∇ₓL)")
    ax.plot(eps, rd, "o-", ms=3, color=GRAY, lw=2, label="случайный ±ε шум")
    ax.set_xlabel("ε (в единицах std признака)"), ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1), ax.legend(fontsize=8)
    ax.set_title("Логистическая регрессия, 20 признаков:\nтот же бюджет ε, разный эффект", fontsize=10)


# ---------- Adversarial attacks (deep learning): PGD ----------

S_PGD = svg("adversarial_attacks_deep_learning", 340, 250, "".join([
    '<rect class="c6" x="70" y="40" width="160" height="160" rx="2" fill-opacity="0.6"/>',
    tx(76, 56, "‖δ‖∞ ≤ ε", "mut", "start", fs=10),
    pa("M 225 20 Q 175 120 225 220", "ln s4", dash=True, head=False),
    tx(290, 134, "другой класс", "t4", fs=11), tx(120, 182, "верный класс", "t2", fs=11),
    dot(150, 120, "t2", 4.5), tx(150, 138, "x⁰ = x", fs=11),
    ar(150, 120, 187, 94, "ln s1"), dot(190, 92, "t1", 3.5), tx(180, 82, "x¹", fs=11),
    ar(190, 92, 258, 54, "ln s1", dash=True), dot(262, 52, "t6", 3.5), tx(270, 44, "x̃²", anchor="start", fs=11),
    ar(262, 52, 233, 52, "ln s3"), tx(248, 46, "Π", "t3", fs=12),
    dot(230, 52, "t4", 4), tx(236, 68, "x²", anchor="start", fs=11),
    tx(150, 216, "проекция Π: clip в [x − ε, x + ε]", "mut", fs=10),
    tx(170, 240, "xᵗ⁺¹ = Π( xᵗ + α·sign(∇ₓL(θ, xᵗ, y)) )", fs=11),
]), "Шаги PGD с проекцией на ε-шар")


# ---------- Adversarial robustness ----------

def _side(dx, boundary, ball_cls):
    A = [(40, 60), (60, 120), (35, 155), (80, 80)]
    B = [(125, 50), (135, 110), (115, 160), (148, 82)]
    if dx:
        A = [(205, 60), (225, 120), (210, 155), (230, 80)]
        B = [(295, 50), (305, 110), (290, 160), (318, 82)]
    s = "".join(dot(x, y, "t1", 4) for x, y in A) + "".join(f'<rect class="c4" x="{x - 4}" y="{y - 4}" width="8" height="8"/>' for x, y in B)
    s += boundary
    near = A[3] if not dx else A[3]
    s += f'<circle class="{ball_cls}" cx="{near[0]}" cy="{near[1]}" r="14" stroke-dasharray="3 2" fill="none"/>'
    return s


S_ROB = svg("adversarial_robustness", 340, 222, "".join([
    tx(85, 18, "обычная модель", fs=11), tx(262, 18, "робастная модель", fs=11),
    _side(0, pa("M 100 28 Q 70 60 88 82 T 96 140 T 92 190", "ln", head=False), "s4"),
    _side(1, '<line class="ln" x1="262" y1="28" x2="262" y2="190"/>', "s2"),
    '<circle class="s2" cx="295" cy="50" r="14" stroke-dasharray="3 2" fill="none"/>',
    ar(170, 28, 170, 200, "ln s6", dash=True, head=False),
    tx(85, 210, "ε-шар задел границу", "t4", fs=11), tx(262, 210, "запас (margin) ≥ ε", "t2", fs=11),
]), "Граница решений обычной и робастной модели")


# ---------- Adversarial training ----------

S_AT = svg("adversarial_training", 340, 222, "".join([
    tx(170, 20, "min_θ  E [ max_‖δ‖≤ε  L(f_θ(x + δ), y) ]", fs=12),
    box(10, 38, 140, 40, "c6", "батч (x, y)"),
    box(190, 38, 140, 40, "c4", "атака PGD", "внутренний max по δ"),
    ar(150, 58, 188, 58),
    ar(260, 78, 260, 118),
    box(190, 120, 140, 40, "c3", "x_adv = x + δ*"),
    ar(190, 140, 152, 140),
    box(10, 120, 140, 40, "c1", "L(x) + L(x_adv)", "внешний min по θ"),
    ar(80, 120, 80, 80), tx(86, 103, "шаг optimizer", "mut", "start", fs=10),
    tx(170, 190, "PGD из K шагов → обучение ≈ в K+1 раз дольше", "mut", fs=11),
    tx(170, 208, "типично ε = 8/255 для изображений", "mut", fs=11),
]), "Цикл adversarial training")


def at_chart(fig):
    rng = np.random.default_rng(0)
    n, d = 400, 50
    y = rng.choice([-1, 1], n)
    X = rng.normal(0, 1, (n, d))
    X[:, :2] += 1.2 * y[:, None]           # 2 информативных признака, 48 шумовых
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]

    def fit(eps_tr, iters=800, lr=0.1, l2=1e-3):
        w, b = np.zeros(d), 0.0
        for _ in range(iters):
            m = ytr * (Xtr @ w + b) - eps_tr * np.abs(w).sum()   # худший случай в L∞-шаре
            s = -1 / (1 + np.exp(m))
            gw = (s * ytr) @ Xtr / len(ytr) - eps_tr * np.sign(w) * s.mean() + l2 * w
            gb = (s * ytr).mean()
            w -= lr * gw
            b -= lr * gb
        return w, b

    eps = np.linspace(0, 0.6, 25)
    ax = fig.add_subplot(1, 1, 1)
    for et, name, c in [(0.0, "обычное обучение", GRAY), (0.15, "adversarial training, ε=0.15", BLUE)]:
        w, b = fit(et)
        acc = [np.mean(yte * (Xte @ w + b) - e * np.abs(w).sum() > 0) for e in eps]
        ax.plot(eps, acc, "o-", ms=3, lw=2, color=c, label=name)
    ax.axvline(0.15, color=BLUE, ls=":", lw=1)
    ax.set_xlabel("ε атаки (L∞)"), ax.set_ylabel("robust accuracy"), ax.set_ylim(0, 1)
    ax.set_title("Линейная модель: точность под\nхудшей атакой в ε-шаре", fontsize=10)
    ax.legend(fontsize=8)


# ---------- Algorithmic trading ----------

S_TRD = svg("algorithmic_trading", 340, 232, "".join([
    box(20, 8, 220, 32, "c6", "Данные: OHLCV, стакан, новости", fs=11),
    ar(130, 40, 130, 52),
    box(20, 54, 220, 32, "c1", "Признаки: returns, RSI, MACD", fs=11),
    ar(130, 86, 130, 98),
    box(20, 100, 220, 32, "c5", "Модель → сигнал buy / sell", fs=11),
    ar(130, 132, 130, 144),
    box(20, 146, 220, 32, "c3", "Риск: размер позиции, stop-loss", fs=11),
    ar(130, 178, 130, 190),
    box(20, 192, 220, 32, "c2", "Исполнение ордеров", fs=11),
    pa("M 240 208 L 278 208 L 278 116 L 243 116", "ln", dash=True),
    tx(296, 162, "бэктест, метрики", "mut", fs=10, extra=' transform="rotate(-90 296 162)"'),
]), "Конвейер алгоритмической торговли")


def trd_chart(fig):
    rng = np.random.default_rng(7)
    n = 500
    drift = 0.0025 * np.sin(np.linspace(0, 3 * np.pi, n))
    r = drift + rng.normal(0, 0.012, n)
    price = 100 * np.exp(np.cumsum(r))

    def sma(a, k):
        out = np.full(a.size, np.nan)
        c = np.cumsum(np.r_[0, a])
        out[k - 1:] = (c[k:] - c[:-k]) / k
        return out

    f, s = sma(price, 20), sma(price, 50)
    sig = np.where(np.isnan(s), 0, (f > s).astype(float))
    pos = np.r_[0, sig[:-1]]                      # входим на следующем баре — без lookahead
    t = np.arange(n)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, price, color=GRAY, lw=1, label="цена")
    ax.plot(t, f, color=BLUE, lw=1.5, label="SMA 20")
    ax.plot(t, s, color=ORANGE, lw=1.5, label="SMA 50")
    ch = np.diff(sig)
    up, dn = np.where(ch > 0)[0] + 1, np.where(ch < 0)[0] + 1
    ax.scatter(up, price[up], marker="^", color=GREEN, s=40, zorder=3, label="buy")
    ax.scatter(dn, price[dn], marker="v", color=RED, s=40, zorder=3, label="sell")
    ax.set_title("Trend following: пересечение SMA 20 / 50", fontsize=10)
    ax.legend(fontsize=6.5, ncol=3, loc="lower left")
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    ax2.plot(t, np.exp(np.cumsum(r)), color=GRAY, lw=1.5, label="buy & hold")
    ax2.plot(t, np.exp(np.cumsum(pos * r)), color=BLUE, lw=2, label="стратегия")
    ax2.set_title("Кривая капитала (без комиссий)", fontsize=10)
    ax2.set_xlabel("день"), ax2.legend(fontsize=8)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- ALS ----------

def _grid(x0, y0, rows, cols, known, cls_known="c1", q=True):
    s = ""
    for i in range(rows):
        for j in range(cols):
            k = (i, j) in known if known is not None else True
            s += f'<rect class="{cls_known if k else "c6"}" x="{x0 + 16 * j}" y="{y0 + 16 * i}" width="16" height="16"/>'
            if not k and q:
                s += tx(x0 + 16 * j + 8, y0 + 16 * i + 12, "?", "mut", fs=10)
    return s


_KN = {(0, 0), (0, 3), (1, 1), (1, 4), (1, 5), (2, 0), (2, 2), (3, 3), (3, 5), (4, 1), (4, 2), (4, 4)}
S_ALS = svg("als_recommender", 340, 222, "".join([
    _grid(10, 30, 5, 6, _KN), tx(58, 124, "R  (m×n)"), tx(58, 20, "известные рейтинги", "mut", fs=10),
    tx(120, 76, "≈", fs=16),
    _grid(134, 30, 5, 2, None, "c2"), tx(150, 124, "U  (m×k)"),
    tx(178, 76, "×", fs=14),
    _grid(192, 30, 2, 6, None, "c3"), tx(240, 82, "Vᵀ  (k×n)"),
    tx(170, 150, "шаг U: uᵢ = (VᵢᵀVᵢ + λI)⁻¹ Vᵢᵀ rᵢ", fs=11),
    tx(170, 170, "шаг V: vⱼ = (UⱼᵀUⱼ + λI)⁻¹ Uⱼᵀ rⱼ", fs=11),
    tx(170, 194, "каждый шаг — ridge-регрессия, решается точно", "mut", fs=10),
    tx(170, 212, "L = Σ (rᵢⱼ − uᵢᵀvⱼ)² + λ(‖U‖² + ‖V‖²)", "mut", fs=10),
]), "Матричная факторизация ALS")


def als_chart(fig):
    rng = np.random.default_rng(0)
    m, n, k = 120, 90, 5
    R = rng.normal(0, 1, (m, k)) @ rng.normal(0, 1, (k, n)) / np.sqrt(k) * 1.2 + 3
    R += rng.normal(0, 0.3, R.shape)
    mask = rng.random(R.shape) < 0.3
    test = (~mask) & (rng.random(R.shape) < 0.2)
    lam = 0.5
    U, V = rng.normal(0, 0.1, (m, k)), rng.normal(0, 0.1, (n, k))
    mu = R[mask].mean()
    tr, te = [], []
    for it in range(12):
        Rc = R - mu
        for i in range(m):
            j = mask[i]
            Vi = V[j]
            U[i] = np.linalg.solve(Vi.T @ Vi + lam * np.eye(k), Vi.T @ Rc[i, j])
        for j in range(n):
            i = mask[:, j]
            Uj = U[i]
            V[j] = np.linalg.solve(Uj.T @ Uj + lam * np.eye(k), Uj.T @ Rc[i, j])
        P_ = U @ V.T + mu
        tr.append(np.sqrt(np.mean((P_ - R)[mask] ** 2)))
        te.append(np.sqrt(np.mean((P_ - R)[test] ** 2)))
    ax = fig.add_subplot(1, 1, 1)
    it = np.arange(1, 13)
    ax.plot(it, tr, "o-", ms=3, lw=2, color=BLUE, label="train RMSE")
    ax.plot(it, te, "o-", ms=3, lw=2, color=ORANGE, label="test RMSE")
    ax.axhline(0.3, color=GRAY, ls="--", lw=1, label="уровень шума")
    ax.set_xlabel("итерация (U-шаг + V-шаг)"), ax.set_ylabel("RMSE")
    ax.set_title("ALS сходится за несколько итераций", fontsize=10)
    ax.legend(fontsize=8)


# ---------- ANOVA ----------

_G = [(70, [125, 135, 150, 160, 130], "t1", "s1", "A"), (150, [90, 100, 110, 120, 105], "t2", "s2", "B"),
      (230, [70, 80, 95, 100, 80], "t3", "s3", "C")]
_J = [-8, 6, -4, 8, 0]
S_ANOVA = svg("anova", 340, 222, "".join(
    [ar(40, 110, 262, 110, "ln s6", dash=True, head=False), tx(268, 114, "ȳ общее", "mut", "start", fs=11)]
    + ["".join(dot(cx + j, v, tc, 4) for j, v in zip(_J, vals))
       + f'<line class="{sc}" x1="{cx - 22}" y1="{np.mean(vals)}" x2="{cx + 22}" y2="{np.mean(vals)}" stroke-width="2.5"/>'
       + tx(cx, 178, f"группа {g}", fs=11)
       for cx, vals, tc, sc, g in _G]
    + [ar(100, 110, 100, 138, "ln s4"), tx(104, 128, "SSB", "t4", "start", fs=11),
       ar(78, 155, 78, 143, "ln s5"), tx(86, 158, "SSW", "t5", "start", fs=11),
       tx(170, 200, "F = MSB / MSW = [SSB/(k−1)] / [SSW/(N−k)]", fs=11),
       tx(170, 216, "F ≫ 1 → между группами разброс больше случайного", "mut", fs=10)]),
    "Межгрупповая и внутригрупповая вариация")


def anova_chart(fig):
    from scipy import stats
    d1, d2 = 2, 12
    x = np.linspace(0, 8, 400)
    pdf = stats.f.pdf(x, d1, d2)
    crit = stats.f.ppf(0.95, d1, d2)
    fobs = 5.0
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, pdf, color=BLUE, lw=2, label=f"F({d1}, {d2}) при H₀")
    ax.fill_between(x, pdf, where=x >= crit, color=RED, alpha=0.35, label=f"α = 0.05, F_крит = {crit:.2f}")
    ax.fill_between(x, pdf, where=x >= fobs, color=ORANGE, alpha=0.6)
    ax.axvline(fobs, color=ORANGE, lw=2, label=f"F = {fobs}: p = {stats.f.sf(fobs, d1, d2):.3f}")
    ax.set_xlabel("F"), ax.set_ylabel("плотность"), ax.set_ylim(0, 0.3)
    ax.set_title("3 группы по 5 наблюдений: k−1 = 2, N−k = 12", fontsize=10)
    ax.legend(fontsize=8)


# ---------- Astrophysics & cosmology: SBI ----------

S_ASTRO = svg("astrophysics_cosmology_ml", 340, 222, "".join([
    box(8, 18, 104, 42, "c6", "prior p(θ)", "Ω_m, σ₈, H₀"),
    ar(112, 39, 128, 39),
    box(130, 18, 104, 42, "c3", "симулятор", "N-body, линзинг"),
    ar(234, 39, 250, 39),
    box(252, 18, 80, 42, "c6", "x_sim", "карты, P(k)"),
    ar(292, 60, 292, 98),
    pa("M 60 60 L 60 121 L 128 121", "ln", dash=True), tx(66, 90, "θ", "mut", "start", fs=11),
    box(130, 100, 202, 42, "c5", "нейросеть q_φ(θ | x)", "обучение на парах (θ, x_sim)"),
    ar(231, 142, 231, 168),
    box(8, 170, 104, 42, "c1", "x_obs", "CMB, обзоры"),
    ar(112, 191, 128, 191),
    box(130, 170, 202, 42, "c2", "posterior p(θ | x_obs)", "без MCMC: мгновенный вывод"),
]), "Simulation-based inference")


# ---------- Audio generation: WaveNet ----------

def _wavenet():
    xs = [44 + 18 * i for i in range(16)]
    rows = [(24, "d=8"), (66, "d=4"), (108, "d=2"), (150, "d=1"), (192, "вход")]
    dil = [8, 4, 2, 1]
    act = [{15}]
    for d in dil:
        act.append({t for a in act[-1] for t in (a, a - d) if t >= 0})
    s = ""
    for li, d in enumerate(dil):
        y0, y1 = rows[li][0], rows[li + 1][0]
        for a in act[li]:
            for t in (a, a - d):
                s += f'<line class="ln s1" x1="{xs[a]}" y1="{y0 + 4}" x2="{xs[t]}" y2="{y1 - 4}"/>'
    for li, (y, lab) in enumerate(rows):
        s += tx(4, y + 4, lab, "mut", "start", fs=10)
        for t in range(16):
            s += f'<circle class="{"c1" if t in act[li] else "c6"}" cx="{xs[t]}" cy="{y}" r="4"/>'
    s += tx(178, 218, "receptive field = 16 = 2⁴ отсчётов, только прошлое", "mut", fs=10)
    s += tx(178, 238, "z = tanh(W_f ∗ x) ⊙ σ(W_g ∗ x)", fs=11)
    return s


S_AUDIO = svg("audio_generation", 340, 248, _wavenet(), "Dilated causal convolutions WaveNet")


def audio_chart(fig):
    from scipy.signal import spectrogram, chirp
    sr = 8000
    t = np.arange(sr) / sr
    x = chirp(t, 200, 1, 2000) * np.exp(-((t - 0.5) / 0.35) ** 2) + 0.4 * chirp(t, 400, 1, 4000 * 0.999)
    x += 0.02 * np.random.default_rng(0).normal(size=t.size)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t[:400] * 1000, x[:400], color=BLUE, lw=1)
    ax.set_title("Raw waveform: 400 отсчётов = 50 мс при 8 кГц", fontsize=10)
    ax.set_xlabel("мс")
    ax2 = fig.add_subplot(2, 1, 2)
    f, tt, S = spectrogram(x, sr, nperseg=256, noverlap=192)
    ax2.pcolormesh(tt, f, 10 * np.log10(S + 1e-10), shading="auto", cmap="magma", vmin=-80)
    ax2.set_title("Спектрограмма: время × частота", fontsize=10)
    ax2.set_xlabel("с"), ax2.set_ylabel("Гц"), ax2.grid(False)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- Autoencoders ----------

S_AE = svg("autoencoders", 340, 206, "".join([
    '<rect class="c6" x="10" y="30" width="22" height="110" rx="3"/>', tx(21, 158, "x"), tx(21, 22, "784", "mut", fs=10),
    '<polygon class="c1" points="40,30 120,65 120,105 40,140"/>', tx(80, 89, "encoder", fs=11),
    '<rect class="c3" x="128" y="62" width="22" height="46" rx="3"/>', tx(139, 89, "z"), tx(139, 56, "32", "mut", fs=10),
    tx(139, 124, "bottleneck", "mut", fs=10),
    '<polygon class="c2" points="158,65 238,30 238,140 158,105"/>', tx(198, 89, "decoder", fs=11),
    '<rect class="c6" x="246" y="30" width="22" height="110" rx="3"/>', tx(257, 158, "x̂"), tx(257, 22, "784", "mut", fs=10),
    pa("M 21 164 Q 139 196 257 164", "ln s4", dash=True, head=False),
    tx(304, 80, "L =", anchor="middle", fs=11), tx(304, 96, "‖x − x̂‖²", anchor="middle", fs=11),
    tx(139, 198, "ошибка восстановления", "t4", fs=10),
]), "Архитектура автоэнкодера")


def ae_chart(fig):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    X = load_digits().data / 16.0
    ks = [1, 2, 4, 8, 16, 32, 48]
    err = []
    for k in ks:
        p = PCA(k, random_state=0).fit(X)
        err.append(np.mean((p.inverse_transform(p.transform(X)) - X) ** 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ks, err, "o-", color=BLUE, lw=2)
    ax.set_xscale("log", base=2), ax.set_xticks(ks), ax.set_xticklabels(ks)
    ax.set_xlabel("размер латентного z"), ax.set_ylabel("MSE восстановления")
    ax.set_title("Digits 8×8 (64 пикселя), линейный AE ≈ PCA:\nузкий bottleneck теряет детали", fontsize=10)


# ---------- AutoML ----------

S_AUTOML = svg("automl", 340, 292, "".join([
    box(20, 8, 210, 34, "c6", "Данные (X, y)"),
    ar(125, 42, 125, 54),
    box(20, 56, 210, 34, "c1", "Предобработка", "пропуски, кодирование категорий"),
    ar(125, 90, 125, 102),
    box(20, 104, 210, 34, "c1", "Feature engineering", "новые признаки, отбор"),
    ar(125, 138, 125, 150),
    box(20, 152, 210, 34, "c5", "Модель + гиперпараметры", "CASH: Bayesian opt., meta-learning"),
    box(250, 152, 80, 34, "c3", "CV-оценка", fs=11),
    ar(230, 162, 248, 162), ar(250, 177, 232, 177),
    tx(290, 202, "бюджет времени", "mut", fs=10),
    ar(125, 186, 125, 198),
    box(20, 200, 210, 34, "c2", "Ансамбль лучших", "ensemble_size = 50"),
    ar(125, 234, 125, 246),
    box(20, 248, 210, 34, "c3", "Лучший pipeline"),
]), "Что автоматизирует AutoML")


# ---------- AutoML for NN: NAS ----------

S_NAS = svg("automl_neural_networks", 340, 200, "".join([
    box(10, 20, 140, 44, "c6", "Search space", "слои, связи, операции"),
    ar(150, 42, 188, 42), tx(169, 36, "A", "mut", fs=10),
    box(190, 20, 140, 44, "c1", "Search strategy", "RL, эволюция, BO, DARTS"),
    ar(235, 64, 235, 118), tx(229, 96, "a ∈ A", "mut", "end", fs=10),
    ar(285, 118, 285, 66), tx(291, 96, "score", "mut", "start", fs=10),
    box(190, 120, 140, 44, "c3", "Оценка качества", "обучение / weight sharing"),
    ar(190, 142, 152, 142),
    box(10, 120, 140, 44, "c2", "Лучшая архитектура", fs=11),
    tx(170, 188, "полный NAS ≈ 1000s GPU-часов; one-shot — дешевле", "mut", fs=10),
]), "Три компонента Neural Architecture Search")


# ---------- Bayesian approach ----------

def _curve(x0, y0, w, h, f):
    t = np.linspace(0.001, 0.999, 60)
    v = f(t)
    v = v / v.max()
    pts = " ".join(f"{x0 + w * a:.1f},{y0 + h - h * b:.1f}" for a, b in zip(t, v))
    return f'<polyline class="{{c}}" points="{pts}" fill="none" stroke-width="2"/>'


def _mini(x0, cls, f, lab, sub):
    s = f'<rect class="c6" x="{x0}" y="20" width="90" height="70" rx="4" fill-opacity="0.5"/>'
    s += f'<line class="ln" x1="{x0 + 6}" y1="82" x2="{x0 + 84}" y2="82"/>'
    s += _curve(x0 + 6, 28, 78, 54, f).replace("{c}", cls)
    s += tx(x0 + 45, 108, lab, fs=11) + tx(x0 + 45, 124, sub, "mut", fs=10)
    return s


S_BAYES = svg("bayesian_approach", 340, 168, "".join([
    _mini(8, "s1", lambda t: t * (1 - t), "P(θ)", "prior Beta(2,2)"),
    tx(110, 60, "×", fs=16),
    _mini(125, "s3", lambda t: t ** 7 * (1 - t) ** 3, "P(D|θ)", "7 орлов из 10"),
    tx(228, 60, "∝", fs=16),
    _mini(242, "s2", lambda t: t ** 8 * (1 - t) ** 4, "P(θ|D)", "posterior Beta(9,5)"),
    tx(170, 154, "P(θ|D) = P(D|θ) · P(θ) / P(D)", fs=12),
]), "Prior, likelihood и posterior")


def bayes_chart(fig):
    from scipy import stats
    rng = np.random.default_rng(1)
    flips = rng.random(200) < 0.7
    t = np.linspace(0, 1, 400)
    ax = fig.add_subplot(1, 1, 1)
    for n, c in [(0, GRAY), (5, ORANGE), (20, GREEN), (200, BLUE)]:
        h = flips[:n].sum()
        a, b = 2 + h, 2 + n - h
        ax.plot(t, stats.beta.pdf(t, a, b), color=c, lw=2, label=f"n = {n}: Beta({a}, {b})")
    ax.axvline(0.7, color=RED, ls="--", lw=1, label="истинное θ = 0.7")
    ax.set_xlabel("θ (вероятность орла)"), ax.set_ylabel("плотность posterior")
    ax.set_title("С ростом данных posterior сужается,\nвлияние prior Beta(2,2) исчезает", fontsize=10)
    ax.legend(fontsize=7)


# ---------- Bayesian hyperparameter optimization ----------

S_BHO = svg("bayesian_hyperparameter_optimization", 340, 206, "".join([
    box(8, 15, 150, 40, "c6", "Стартовые точки", "несколько случайных x"),
    ar(158, 35, 180, 35),
    box(182, 15, 150, 40, "c1", "GP-суррогат", "μ(x), σ(x) по истории"),
    ar(257, 55, 257, 118),
    box(182, 120, 150, 40, "c3", "Acquisition", "x* = argmax EI(x)"),
    ar(182, 140, 160, 140),
    box(8, 120, 150, 40, "c2", "Обучить модель", "y* = score(x*)"),
    ar(130, 120, 205, 57), tx(150, 92, "обновить GP", "mut", "end", fs=10),
    tx(170, 182, "EI: баланс exploration (σ) и exploitation (μ)", "mut", fs=11),
    tx(170, 198, "стоп: исчерпан бюджет n_iter", "mut", fs=11),
]), "Цикл байесовской оптимизации")


def bho_chart(fig):
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    f = lambda x: np.sin(3 * x) * np.exp(-0.2 * x) + 0.3 * x / 5
    x = np.linspace(0, 5, 300)
    X = np.array([0.0, 1.0, 2.6, 3.6, 4.6])
    gp = GaussianProcessRegressor(Matern(length_scale=0.7, nu=2.5), alpha=1e-6, normalize_y=True,
                                  optimizer=None).fit(X[:, None], f(X))
    mu, sd = gp.predict(x[:, None], return_std=True)
    best = f(X).max()
    z = (mu - best - 0.01) / np.maximum(sd, 1e-9)
    ei = (mu - best - 0.01) * norm.cdf(z) + sd * norm.pdf(z)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(x, f(x), "--", color="k", lw=1, label="истинная f(x)")
    ax.plot(x, mu, color=BLUE, lw=2, label="μ(x) GP")
    ax.fill_between(x, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.18, label="μ ± 2σ")
    ax.scatter(X, f(X), color=RED, zorder=3, s=25, label="проверенные")
    ax.set_title("Суррогат: неопределённость между точками", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    ax2.plot(x, ei, color=GREEN, lw=2)
    ax2.fill_between(x, ei, color=GREEN, alpha=0.2)
    xm = x[np.argmax(ei)]
    ax2.axvline(xm, color=ORANGE, lw=2, label=f"следующая точка x* = {xm:.2f}")
    ax2.set_title("Expected Improvement", fontsize=10), ax2.set_xlabel("гиперпараметр x")
    ax2.legend(fontsize=7, loc="upper left")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- BERT fine-tuning ----------

_TOK = ["[CLS]", "This", "is", "great", "[SEP]"]
S_BFT = svg("bert_finetuning", 340, 222, "".join(
    [box(22 + 60 * i, 190, 54, 24, "c6", t, fs=11) for i, t in enumerate(_TOK)]
    + [ar(49 + 60 * i, 190, 49 + 60 * i, 162) for i in range(5)]
    + [box(20, 120, 296, 40, "c1", "BERT encoder (bert-base, 12 слоёв)", "предобучен; дообучаются все веса, lr = 2e-5"),
       ar(49, 120, 49, 94), tx(55, 110, "h_[CLS]", "mut", "start", fs=10),
       tx(250, 110, "hᵢ токенов → NER", "mut", fs=10),
       box(10, 62, 110, 30, "c3", "Linear (новый)", fs=11),
       ar(120, 77, 148, 77),
       box(150, 62, 120, 30, "c2", "softmax → класс", fs=11),
       tx(210, 52, "num_labels = 2", "mut", fs=10),
       tx(170, 24, "голова классификации поверх [CLS]", "mut", fs=11)]),
    "Fine-tuning BERT для классификации")


def bft_chart(fig):
    steps = np.arange(0, 3001)
    warm, total, peak = 500, 3000, 2e-5
    lr = np.where(steps < warm, peak * steps / warm, peak * (total - steps) / (total - warm))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, lr * 1e5, color=BLUE, lw=2)
    ax.axvline(warm, color=ORANGE, ls="--", lw=1.5, label="warmup_steps = 500")
    ax.set_xlabel("шаг обучения"), ax.set_ylabel("learning rate, ×1e-5")
    ax.set_title("Linear warmup + linear decay (Trainer)", fontsize=10)
    ax.legend(fontsize=8)


# ---------- BERT language models: MLM ----------

_MT = ["[CLS]", "кот", "[MASK]", "на", "ковре", "[SEP]"]
S_BLM = svg("bert_language_models", 340, 244, "".join(
    [box(8 + 54 * i, 196, 50, 24, "c3" if t == "[MASK]" else "c6", t, fs=10) for i, t in enumerate(_MT)]
    + [ar(33 + 54 * i, 196, 33 + 54 * i, 156) for i in range(6)]
    + [tx(170, 236, "вход = E_tok + E_seg + E_pos", "mut", fs=10)]
    + [
       box(8, 110, 320, 44, "c1", "Transformer Encoder × 12", "self-attention: контекст слева и справа"),
       ar(141, 110, 141, 80), box(96, 50, 90, 28, "c3", "«сидит»", fs=11),
       tx(250, 62, "MLM: 15% токенов", "mut", fs=10),
       ar(33, 110, 33, 80), box(4, 50, 58, 28, "c5", "NSP", fs=11),
       tx(170, 28, "предсказать скрытый токен по обеим сторонам", "mut", fs=11)]),
    "Masked Language Model в BERT")


# ---------- Bias in neural networks ----------

S_BNN = svg("bias_neural_networks", 340, 205, cycle4(
    ["Данные", "Обучение модели", "Решения", "Новые данные"],
    ["historical, selection bias", "aggregation, evaluation", "deployment bias", "собраны по решениям"],
    ["c6", "c1", "c3", "c4"])
    + tx(170, 86, "feedback loop:", "t4", fs=11) + tx(170, 102, "bias усиливается", "t4", fs=11)
    + tx(170, 188, "аудит fairness — на каждом звене цикла", "mut", fs=11),
    "Петля усиления bias")


def bnn_chart(fig):
    from scipy.stats import norm
    s = np.linspace(0, 1, 400)
    thr = 0.5
    ax = fig.add_subplot(1, 1, 1)
    for mu, c, g in [(0.66, BLUE, "A"), (0.53, ORANGE, "B")]:
        pdf = norm.pdf(s, mu, 0.12)
        tpr = norm.sf(thr, mu, 0.12)
        ax.plot(s, pdf, color=c, lw=2, label=f"группа {g}: TPR = {tpr:.2f}")
        ax.fill_between(s, pdf, where=s >= thr, color=c, alpha=0.2)
    ax.axvline(thr, color=RED, lw=2, ls="--", label="общий порог 0.5")
    ax.set_xlabel("score модели (только y = 1)"), ax.set_ylabel("плотность")
    ax.set_title("Measurement bias: у группы B занижены\nскоры → equal opportunity нарушен", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")


# ---------- Bias–variance decomposition ----------

def _target(cx, cy, pts):
    s = "".join(f'<circle class="{c}" cx="{cx}" cy="{cy}" r="{r}"/>' for r, c in [(42, "c6"), (28, "c1"), (13, "c4")])
    s += "".join(f'<circle class="t5" cx="{cx + dx}" cy="{cy + dy}" r="3.2"/>' for dx, dy in pts)
    return s


_TIGHT = [(0, 0), (4, -3), (-3, 4), (3, 4), (-4, -2), (1, -5)]
_WIDE = [(20, -14), (-22, 8), (8, 24), (-14, -24), (26, 12), (-6, 16)]
_OFF = (18, -18)
S_BV = svg("bias_variance_decomposition", 340, 262, "".join([
    tx(118, 16, "низкая variance", fs=11), tx(258, 16, "высокая variance", fs=11),
    tx(22, 85, "низкий bias", fs=11, extra=' transform="rotate(-90 22 85)"'),
    tx(22, 200, "высокий bias", fs=11, extra=' transform="rotate(-90 22 200)"'),
    _target(118, 80, _TIGHT), _target(258, 80, _WIDE),
    _target(118, 198, [(x + _OFF[0], y + _OFF[1]) for x, y in _TIGHT]),
    _target(258, 198, [(x + _OFF[0], y + _OFF[1]) for x, y in _WIDE]),
    tx(188, 256, "центр — истинное f(x); точки — f̂(x) на разных выборках", "mut", fs=10),
]), "Bias и variance на мишенях")


def bv_chart(fig):
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * np.pi * x)
    sig, n, R = 0.3, 30, 200
    xg = np.linspace(0.05, 0.95, 50)
    degs = np.arange(1, 9)
    b2, var = [], []
    for d in degs:
        preds = np.empty((R, xg.size))
        for r in range(R):
            x = rng.uniform(0, 1, n)
            y = f(x) + rng.normal(0, sig, n)
            preds[r] = np.polynomial.Polynomial.fit(x, y, d)(xg)
        b2.append(np.mean((preds.mean(0) - f(xg)) ** 2))
        var.append(np.mean(preds.var(0)))
    b2, var = np.array(b2), np.array(var)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(degs, b2, "o-", ms=3, lw=2, color=BLUE, label="Bias²")
    ax.plot(degs, var, "o-", ms=3, lw=2, color=ORANGE, label="Variance")
    ax.axhline(sig ** 2, color=GRAY, ls="--", lw=1, label="σ² (шум)")
    ax.plot(degs, b2 + var + sig ** 2, "o-", ms=3, lw=2.5, color=RED, label="ожидаемая ошибка")
    ax.set_yscale("log"), ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("степень полинома (сложность)"), ax.set_xticks(degs)
    ax.set_title("sin(2πx) + шум, n = 30, 200 выборок", fontsize=10)
    ax.legend(fontsize=7.5)


# ---------- Bidirectional RNN ----------

def _birnn():
    cols = [50, 130, 210, 290]
    s = ""
    for i, c in enumerate(cols):
        s += tx(c, 234, f"x{'₁₂₃₄'[i]}")
        s += ar(c - 16, 222, c - 16, 128) + ar(c + 16, 222, c + 16, 178)
        s += box(c - 36, 100, 40, 26, "c1", f"→h{'₁₂₃₄'[i]}", fs=11)
        s += box(c - 4, 150, 40, 26, "c5", f"←h{'₁₂₃₄'[i]}", fs=11)
        s += ar(c - 16, 100, c - 10, 68, "ln s1") + ar(c + 16, 150, c + 10, 68, "ln s5")
        s += box(c - 30, 42, 60, 24, "c3", "[→h; ←h]", fs=10)
        s += ar(c, 42, c, 26) + tx(c, 18, f"y{'₁₂₃₄'[i]}", fs=11)
        if i < 3:
            s += ar(c + 4, 113, c + 42, 113, "ln s1")
            s += ar(c + 76, 163, c + 38, 163, "ln s5")
    return s


S_BIRNN = svg("bidirectional_rnn", 340, 244, _birnn(), "Развёрнутая двунаправленная RNN")


# ---------- Binning ----------

_PTS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 24, 29, 35, 42, 52, 65, 80, 100]


def _binning():
    X = lambda v: 20 + 3 * v
    s = tx(20, 22, "Equal width (pd.cut): одинаковая ширина", "t1", "start", fs=11)
    ew = [2, 26.5, 51, 75.5, 100]
    for e in ew:
        s += f'<line class="s1" x1="{X(e)}" y1="34" x2="{X(e)}" y2="92" stroke-width="1.5"/>'
    for a, b in zip(ew[:-1], ew[1:]):
        cnt = sum(a <= v < b or (b == 100 and v == 100) for v in _PTS)
        s += tx((X(a) + X(b)) / 2, 72, str(cnt), "t1", fs=12)
    q = np.quantile(_PTS, [0, .25, .5, .75, 1])
    for e in q:
        s += f'<line class="s3" x1="{X(e):.1f}" y1="98" x2="{X(e):.1f}" y2="156" stroke-width="1.5"/>'
    for a, b in zip(q[:-1], q[1:]):
        s += tx(f"{(X(a) + X(b)) / 2:.1f}", 130, "5", "t3", fs=12)
    s += '<line class="ln" x1="20" y1="95" x2="320" y2="95"/>'
    s += "".join(dot(X(v), 95, "t5", 2.6) for v in _PTS)
    s += tx(20, 176, "Equal frequency (pd.qcut): поровну точек", "t3", "start", fs=11)
    return s


S_BIN = svg("binning_discretization", 340, 186, _binning(), "Равная ширина и равная частота бинов")


def bin_chart(fig):
    rng = np.random.default_rng(0)
    x = rng.lognormal(3, 0.6, 1000)
    k = 5
    ew = np.linspace(x.min(), x.max(), k + 1)
    q = np.quantile(x, np.linspace(0, 1, k + 1))
    for i, (edges, c, name) in enumerate([(ew, BLUE, "Equal width"), (q, ORANGE, "Equal frequency")]):
        ax = fig.add_subplot(2, 1, i + 1)
        ax.hist(x, bins=60, color=GRAY, alpha=0.6)
        for e in edges:
            ax.axvline(e, color=c, lw=1.8)
        cnt = np.histogram(x, edges)[0]
        ax.set_title(f"{name}, 5 бинов: {' / '.join(map(str, cnt))} точек", fontsize=10)
        ax.set_xlim(0, x.max() * 1.02)
    ax.set_xlabel("значение признака (lognormal)")
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- Bioinformatics ----------

_BIO = [("Образец ДНК / РНК", "c6", ""), ("Секвенирование NGS", "c1", "FASTQ"),
        ("QC и фильтрация ридов", "c1", "FastQC"), ("Выравнивание на референс", "c3", "BAM"),
        ("Variant calling (DeepVariant)", "c5", "VCF"), ("Аннотация, патогенность", "c2", "отчёт")]
S_BIO = svg("bioinformatics_computational_biology", 340, 250, "".join(
    box(20, 8 + 40 * i, 210, 30, c, t, fs=11) + (tx(244, 27 + 40 * i, f, "mut", "start", fs=11) if f else "")
    + (ar(125, 38 + 40 * i, 125, 46 + 40 * i) if i < 5 else "")
    for i, (t, c, f) in enumerate(_BIO)), "Конвейер анализа NGS-данных")


# ---------- Biologically inspired: LIF ----------

def _train(x0, x1, y, ticks):
    s = f'<line class="ln" x1="{x0}" y1="{y}" x2="{x1}" y2="{y}"/>'
    return s + "".join(f'<line class="ln" x1="{t}" y1="{y}" x2="{t}" y2="{y - 10}"/>' for t in ticks)


S_LIF = svg("biologically_inspired_architectures", 340, 180, "".join([
    _train(10, 90, 40, [22, 50, 75]), _train(10, 90, 70, [35, 62]), _train(10, 90, 100, [18, 44, 58, 84]),
    tx(50, 124, "входные спайки sᵢ", "mut", fs=10),
    ar(94, 40, *_nb(160, 70, 94, 40, 30)), ar(94, 70, 128, 70), ar(94, 100, *_nb(160, 70, 94, 100, 30)),
    '<circle class="c5" cx="160" cy="70" r="28"/>', tx(160, 68, "v(t)"), tx(160, 82, "leak", "mut", fs=10),
    ar(188, 70, 208, 70), box(210, 55, 54, 30, "c3", "v &gt; θ ?", fs=11),
    ar(264, 70, 276, 70), _train(278, 334, 76, [292, 318]), tx(306, 52, "спайк", "mut", fs=10),
    pa("M 237 85 C 237 122, 178 124, 170 98", "ln s4", dash=True), tx(210, 134, "сброс v = 0", "t4", fs=10),
    tx(170, 164, "v[t] = β·v[t−1] + Σ wᵢ sᵢ[t],  β = 0.9", fs=11),
]), "LIF-нейрон")


def lif_chart(fig):
    rng = np.random.default_rng(2)
    T = 120
    inp = (rng.random((T, 3)) < 0.12) @ np.array([0.35, 0.3, 0.4])
    v, vs, spikes = 0.0, [], []
    for t in range(T):
        v = 0.9 * v + inp[t]
        if v > 1.0:
            vs.append(v)
            spikes.append(t)
            v = 0.0
        else:
            vs.append(v)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(T), vs, color=PURPLE, lw=1.8, label="мембранный потенциал v")
    ax.axhline(1.0, color=RED, ls="--", lw=1.2, label="порог θ = 1")
    ax.vlines(spikes, 1.15, 1.35, color="k", lw=2, label="выходные спайки")
    ax.vlines(np.where(inp > 0)[0], -0.2, -0.08, color=GRAY, lw=1.2, label="входные спайки")
    ax.set_ylim(-0.25, 1.45), ax.set_xlabel("шаг t")
    ax.set_title("LIF: накопление, утечка (β = 0.9), сброс", fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)


# ---------- BIRCH ----------

def _birch():
    s = ""
    for j in range(3):
        s += f'<rect class="c1" x="{110 + 40 * j}" y="12" width="40" height="26"/>' + tx(130 + 40 * j, 30, f"CF{'₁₂₃'[j]}", fs=11)
    tx_ = tx(260, 30, "root", "mut", "start", fs=10)
    s += tx_
    for i in range(3):
        x0 = 10 + 115 * i
        s += ar(130 + 40 * i, 38, x0 + 45, 76)
        for j in range(3):
            s += f'<rect class="c3" x="{x0 + 30 * j}" y="78" width="30" height="24"/>' + tx(x0 + 15 + 30 * j, 94, "CF", fs=10)
            cx, cy = x0 + 15 + 30 * j, 132
            s += f'<circle class="s2" cx="{cx}" cy="{cy}" r="11" stroke-dasharray="3 2" fill="none"/>'
            for dx, dy in [(-4, -3), (3, -4), (0, 4), (5, 2)]:
                s += dot(cx + dx, cy + dy, "t2", 1.8)
            s += f'<line class="ln s6" x1="{cx}" y1="102" x2="{cx}" y2="120"/>'
    s += tx(170, 162, "листья → subclusters радиуса ≤ threshold", "mut", fs=10)
    s += tx(170, 184, "CF = (N, LS, SS)", fs=12)
    s += tx(170, 202, "LS = Σ xᵢ,  SS = Σ ‖xᵢ‖²,  центр = LS / N", "mut", fs=10)
    s += tx(170, 220, "CF₁ + CF₂ = (N₁+N₂, LS₁+LS₂, SS₁+SS₂)", "mut", fs=10)
    return s


S_BIRCH = svg("birch", 340, 230, _birch(), "CF-дерево BIRCH")


def birch_chart(fig):
    from sklearn.cluster import Birch
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=800, centers=4, cluster_std=0.7, random_state=2)
    m = Birch(threshold=0.6, n_clusters=4).fit(X)
    ax = fig.add_subplot(1, 1, 1)
    for k in range(4):
        ax.scatter(*X[m.labels_ == k].T, s=6, alpha=0.45, color=P[k])
    sc = m.subcluster_centers_
    ax.scatter(*sc.T, marker="x", s=22, color="k", lw=1.2, label=f"{len(sc)} subclusters (CF-листья)")
    ax.set_title("threshold = 0.6 → subclusters,\nзатем Agglomerative до 4 кластеров", fontsize=10)
    ax.legend(fontsize=8, loc="best")


# ---------- Blending ----------

S_BLEND = svg("blending", 340, 232, "".join([
    '<rect class="c1" x="10" y="18" width="160" height="28"/>', tx(90, 36, "train 50%", fs=11),
    '<rect class="c3" x="170" y="18" width="80" height="28"/>', tx(210, 36, "holdout 25%", fs=11),
    '<rect class="c6" x="250" y="18" width="80" height="28"/>', tx(290, 36, "test 25%", fs=11),
    ar(80, 46, 80, 73), tx(86, 63, "fit", "mut", "start", fs=10),
    box(10, 75, 140, 30, "c1", "модели: RF, GB", fs=11),
    ar(150, 90, 163, 90), ar(210, 46, 210, 73),
    box(165, 75, 135, 30, "c3", "прогнозы на holdout", fs=11),
    ar(232, 105, 232, 133), tx(226, 124, "fit на y_holdout", "mut", "end", fs=10),
    box(165, 135, 135, 30, "c2", "мета-модель (LR)", fs=11),
    ar(232, 165, 210, 188),
    pa("M 318 46 L 318 205 L 248 205", "ln", dash=True), tx(312, 120, "y_test", "mut", "end", fs=10),
    box(95, 190, 150, 30, "c5", "оценка на test", fs=11),
    pa("M 80 105 L 80 150 L 163 150", "ln", dash=True), tx(86, 126, "прогнозы", "mut", "start", fs=10),
    tx(86, 140, "на X_test", "mut", "start", fs=10),
]), "Схема блендинга")


# ---------- Bootstrap ----------

def _boot():
    s = box(95, 8, 150, 32, "c6", "X = {x₁, …, x₅}", fs=12)
    s += tx(252, 29, "с возвратом", "mut", "start", fs=10)
    rs = [("x₂ x₂ x₅ x₁ x₄", "θ*₁"), ("x₃ x₁ x₁ x₅ x₃", "θ*₂"), ("x₄ x₅ x₂ x₂ x₂", "θ*_B")]
    for (t, th), x0 in zip(rs, (8, 116, 232)):
        s += box(x0, 68, 100, 40, "c1", t, "→ " + th, fs=11)
        s += ar(170, 40, x0 + 50, 66)
    s += tx(224, 92, "…", fs=12)
    for i in range(11):
        h = 55 * np.exp(-((i - 5) / 2.3) ** 2 / 2)
        s += f'<rect class="c3" x="{60 + 20 * i}" y="{190 - h:.1f}" width="19" height="{h:.1f}"/>'
    for x0, x1 in [(58, 112), (166, 170), (282, 228)]:
        s += ar(x0, 108, x1, 128)
    s += '<line class="ln" x1="50" y1="190" x2="290" y2="190"/>'
    for x in (80, 260):
        s += f'<line class="s4" x1="{x}" y1="124" x2="{x}" y2="194" stroke-dasharray="4 3" stroke-width="2"/>'
    s += tx(80, 206, "2.5%", "t4", fs=10) + tx(260, 206, "97.5%", "t4", fs=10)
    s += tx(170, 206, "распределение θ*", "mut", fs=10)
    s += tx(170, 224, "95% CI: перцентили θ* (B ≈ 10 000)", "mut", fs=11)
    return s


S_BOOT = svg("bootstrap_methods", 340, 232, _boot(), "Алгоритм bootstrap")


def boot_chart(fig):
    data = np.array([23, 25, 27, 29, 31, 33, 35, 37, 39, 41])
    rng = np.random.default_rng(42)
    bm = rng.choice(data, (10000, data.size), replace=True).mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(bm, bins=np.arange(25.9, 38.3, 0.4), color=ORANGE, alpha=0.7)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.axvline(data.mean(), color=BLUE, lw=2, label=f"среднее выборки = {data.mean():.0f}")
    ax.axvline(lo, color=RED, ls="--", lw=2, label=f"95% CI: [{lo:.1f}, {hi:.1f}]")
    ax.axvline(hi, color=RED, ls="--", lw=2)
    ax.set_xlabel("bootstrap-среднее θ*"), ax.set_ylabel("частота")
    ax.set_title("B = 10 000 выборок с возвратом из n = 10", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")


# ---------- Box–Jenkins ----------

S_BJ = svg("box_jenkins", 340, 278, "".join([
    box(40, 8, 200, 36, "c6", "Ряд yₜ", "стационарность: ADF / KPSS → d"),
    ar(140, 44, 140, 58),
    box(40, 60, 200, 36, "c1", "1. Идентификация", "ACF / PACF → p, q"),
    ar(140, 96, 140, 110),
    box(40, 112, 200, 36, "c3", "2. Оценка", "MLE коэффициентов φ, θ"),
    ar(140, 148, 140, 162),
    box(40, 164, 200, 36, "c5", "3. Диагностика", "Ljung-Box: остатки — белый шум?"),
    ar(140, 200, 140, 232), tx(148, 220, "да", "t2", "start", fs=11),
    pa("M 240 182 L 290 182 L 290 78 L 243 78", "ln s4"), tx(296, 134, "нет", "t4", "start", fs=11),
    box(40, 234, 200, 36, "c2", "4. Прогноз", "+ доверительные интервалы"),
]), "Итерационный цикл Box-Jenkins")


def _acf(x, L):
    x = x - x.mean()
    c0 = x @ x
    return np.array([1.0] + [(x[:-k] @ x[k:]) / c0 for k in range(1, L + 1)])


def _pacf(r, L):
    out, phi = [1.0], np.zeros(0)
    for k in range(1, L + 1):
        if k == 1:
            phi = np.array([r[1]])
        else:
            pk = (r[k] - phi @ r[k - 1:0:-1]) / (1 - phi @ r[1:k])
            phi = np.r_[phi - pk * phi[::-1], pk]
        out.append(phi[-1])
    return np.array(out)


def bj_chart(fig):
    rng = np.random.default_rng(0)
    n, L = 400, 15
    e = rng.normal(size=n + 100)
    y = np.zeros(n + 100)
    for t in range(2, n + 100):
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + e[t]
    y = y[100:]
    r = _acf(y, L)
    pc = _pacf(r, L)
    band = 1.96 / np.sqrt(n)
    lags = np.arange(L + 1)
    for i, (v, name, c) in enumerate([(r, "ACF: затухает (с колебаниями)", BLUE),
                                       (pc, "PACF: обрыв после лага 2 → p = 2", ORANGE)]):
        ax = fig.add_subplot(2, 1, i + 1)
        ax.vlines(lags[1:], 0, v[1:], color=c, lw=2)
        ax.scatter(lags[1:], v[1:], color=c, s=14, zorder=3)
        ax.axhspan(-band, band, color=GRAY, alpha=0.25)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(name, fontsize=10), ax.set_ylim(-0.6, 0.7)
    ax.set_xlabel("лаг")
    fig.suptitle("AR(2): yₜ = 0.6yₜ₋₁ − 0.3yₜ₋₂ + εₜ", fontsize=10)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


# ---------- Brain vs NN ----------

S_BRAIN = svg("brain_neural_networks_connection", 340, 222, "".join([
    tx(170, 18, "биологический нейрон", "mut", fs=11),
    box(8, 30, 68, 30, "c2", "синапсы", fs=11), ar(76, 45, 90, 45),
    box(92, 30, 68, 30, "c2", "дендриты", fs=11), ar(160, 45, 174, 45),
    box(176, 30, 84, 30, "c2", "тело клетки", fs=11), ar(260, 45, 274, 45),
    box(276, 30, 58, 30, "c2", "аксон", fs=11),
    ar(42, 62, 42, 126, "ln s6", dash=True, both=True), ar(126, 62, 126, 126, "ln s6", dash=True, both=True),
    ar(218, 62, 218, 126, "ln s6", dash=True, both=True), ar(305, 62, 305, 126, "ln s6", dash=True, both=True),
    tx(46, 98, "сила", "mut", "start", fs=10),
    tx(222, 98, "сумма", "mut", "start", fs=10),
    tx(301, 98, "спайк", "mut", "end", fs=10),
    box(8, 128, 68, 30, "c1", "веса wᵢ", fs=11), tx(84, 147, "×", fs=12),
    box(92, 128, 68, 30, "c1", "входы xᵢ", fs=11), ar(160, 143, 174, 143),
    box(176, 128, 84, 30, "c1", "Σ wᵢxᵢ + b", fs=11), ar(260, 143, 274, 143),
    box(276, 128, 58, 30, "c1", "f(z) → y", fs=11),
    tx(170, 176, "искусственный нейрон", "mut", fs=11),
    tx(170, 198, "y = f(Σ wᵢxᵢ + b)", fs=12),
    tx(170, 216, "мозг: спайки 0/1 ~100 Гц; ИНС: непрерывные числа", "mut", fs=10),
]), "Соответствие биологического и искусственного нейрона")


# ---------- CCA ----------

S_CCA = svg("canonical_correlation_analysis", 340, 196, "".join([
    box(10, 20, 70, 60, "c1", "X", "n × p", fs=14),
    box(10, 110, 70, 60, "c3", "Y", "n × q", fs=14),
    ar(80, 50, 158, 50), tx(119, 44, "веса a", "mut", fs=10),
    ar(80, 140, 158, 140), tx(119, 134, "веса b", "mut", fs=10),
    box(160, 32, 74, 36, "c1", "u = Xa"),
    box(160, 122, 74, 36, "c3", "v = Yb"),
    ar(197, 70, 197, 120, "ln s4", both=True),
    tx(244, 90, "ρ₁ = max", "t4", "start", fs=11), tx(244, 106, "corr(u, v)", "t4", "start", fs=11),
    tx(170, 186, "следующие пары: max corr при corr(uₖ, u₁) = 0", "mut", fs=10),
]), "Идея канонического корреляционного анализа")


def _cca_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    X = np.c_[z1 + 0.5 * rng.normal(size=n), z1 - z2 + 0.8 * rng.normal(size=n),
              z2 + 1.0 * rng.normal(size=n), rng.normal(size=n)]
    Y = np.c_[z1 + 0.6 * rng.normal(size=n), z2 + 0.9 * rng.normal(size=n), rng.normal(size=n)]
    return X, Y


def cca_chart(fig):
    from sklearn.cross_decomposition import CCA
    X, Y = _cca_data()
    Xc, Yc = CCA(n_components=2).fit(X, Y).transform(X, Y)
    ax = fig.add_subplot(1, 1, 1)
    for k, c in [(0, BLUE), (1, ORANGE)]:
        r = np.corrcoef(Xc[:, k], Yc[:, k])[0, 1]
        ax.scatter(Xc[:, k] / Xc[:, k].std(), Yc[:, k] / Yc[:, k].std(), s=6, alpha=0.5, color=c,
                   label=f"пара {k + 1}: ρ = {r:.2f}")
    ax.set_xlabel("uₖ = X aₖ (стандартизовано)"), ax.set_ylabel("vₖ = Y bₖ")
    ax.set_title("Канонические переменные: первая пара\nкоррелирует сильнее второй", fontsize=10)
    ax.legend(fontsize=8, markerscale=2)


S_CCAM = svg("canonical_correlation_analysis_method", 340, 178, "".join(
    [tx(41, 14, "метод", "mut", fs=10), tx(113, 14, "вход", "mut", fs=10), tx(172, 14, "цель", "mut", "start", fs=10)]
    + [box(10, y, 62, 34, c, m, fs=12) + box(80, y, 66, 34, "c6", inp, fs=11) + ar(146, y + 17, 166, y + 17)
       + tx(172, y + 21, goal, anchor="start", fs=11)
       for y, m, c, inp, goal in [(22, "PCA", "c1", "X", "max дисперсии Xw"),
                                  (68, "LDA", "c2", "X + метки", "max разделения классов"),
                                  (114, "CCA", "c3", "X и Y", "max corr(Xa, Yb)")]]
    + [tx(170, 168, "CCA ищет общую структуру двух наборов признаков", "mut", fs=10)]),
    "CCA, PCA и LDA: входы и цели")


def ccam_chart(fig):
    from sklearn.cross_decomposition import CCA
    X, Y = _cca_data(1)
    Xc, Yc = CCA(n_components=3).fit(X, Y).transform(X, Y)
    rho = [np.corrcoef(Xc[:, k], Yc[:, k])[0, 1] for k in range(3)]
    ax = fig.add_subplot(1, 1, 1)
    b = ax.bar([1, 2, 3], rho, color=[BLUE, BLUE, GRAY], width=0.6)
    for rect, v in zip(b, rho):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks([1, 2, 3]), ax.set_ylim(0, 1)
    ax.set_xlabel("номер канонической пары k"), ax.set_ylabel("ρₖ = corr(uₖ, vₖ)")
    ax.set_title("2 общих скрытых фактора → 2 значимые\nканонические корреляции, третья ≈ 0", fontsize=10)


VISUALS = {
    "3d_computer_vision": [
        {"title": "Проекция камеры-обскуры", "svg": S_3D,
         "caption": "Точка P проецируется на плоскость изображения по лучу через центр камеры C: координаты делятся на "
                    "глубину Z, поэтому далёкие объекты меньше. В общем виде p ≃ K[R|t]P в однородных координатах."},
    ],
    "activation_functions": [
        {"title": "Где стоит функция активации", "svg": S_ACT,
         "caption": "Нейрон считает линейную комбинацию z = wᵀx + b и пропускает её через нелинейность f. "
                    "Без f любая глубина сводится к одному линейному слою."},
        {"title": "Формы функций и их производные", "chart": act_chart,
         "caption": "Sigmoid и tanh насыщаются — производная у краёв ≈ 0 (vanishing gradient). У ReLU производная 1 "
                    "при x > 0 и 0 при x < 0 (dying ReLU); Leaky ReLU и GELU оставляют небольшой градиент слева."},
    ],
    "active_learning": [
        {"title": "Цикл active learning", "svg": S_AL,
         "caption": "Модель сама выбирает из пула U самые информативные примеры, человек их размечает, "
                    "и цикл повторяется, пока не кончится бюджет разметки."},
        {"title": "Uncertainty sampling против случайного выбора", "chart": al_chart,
         "caption": "Логистическая регрессия, среднее по 3 запускам: запрос самых неуверенных примеров (p ≈ 0.5) "
                    "даёт ту же точность при меньшем числе меток, чем случайная разметка."},
    ],
    "adaptive_windowing": [
        {"title": "Как ADWIN режет окно", "svg": S_ADW,
         "caption": "ADWIN перебирает разбиения окна на старую W₀ и новую W₁ части; если средние различаются "
                    "больше порога ε_cut (граница Хёфдинга), старая часть отбрасывается."},
        {"title": "ADWIN на потоке с drift", "chart": adw_chart,
         "caption": "На стабильных участках окно растёт и оценка уточняется; после drift старая часть отрезается. "
                    "Граница Хёфдинга консервативна, поэтому срабатывание запаздывает на несколько десятков точек."},
    ],
    "advanced_ensembling": [
        {"title": "Stacking с out-of-fold предсказаниями", "svg": S_ENS,
         "caption": "Базовые модели обучаются по фолдам (cv=5), и их предсказания на отложенных фолдах становятся "
                    "признаками для мета-модели: так она не видит переобученных прогнозов на train."},
    ],
    "adversarial_attacks": [
        {"title": "FGSM: классический пример с пандой", "svg": S_ADV,
         "caption": "Добавка ε·sign(∇ₓL) незаметна глазу (ε = 0.007), но сдвигает каждый пиксель в сторону роста "
                    "loss — и сеть уверенно видит гиббона."},
        {"title": "Accuracy при росте ε", "chart": adv_chart,
         "caption": "При одинаковой норме ‖δ‖∞ = ε направленное возмущение FGSM рушит точность, а случайный шум "
                    "почти не влияет: опасен именно знак градиента."},
    ],
    "adversarial_attacks_deep_learning": [
        {"title": "PGD: шаги и проекция на ε-шар", "svg": S_PGD,
         "caption": "PGD делает несколько шагов α·sign(∇ₓL); если шаг выходит за L∞-шар вокруг x, точка "
                    "проецируется обратно (clip). Так находится adversarial пример за границей решения внутри шара."},
    ],
    "adversarial_robustness": [
        {"title": "Почему модель уязвима", "svg": S_ROB,
         "caption": "Слева граница проходит ближе ε к точке — внутри ε-шара есть adversarial пример. Робастная "
                    "модель держит запас (margin) не меньше ε от данных."},
    ],
    "adversarial_training": [
        {"title": "Min-max цикл adversarial training", "svg": S_AT,
         "caption": "Внутренний max: атака (обычно PGD) ищет худшее δ в ε-шаре для текущих весов. "
                    "Внешний min: шаг оптимизатора по loss на чистых и adversarial примерах."},
        {"title": "Устойчивость после adversarial training", "chart": at_chart,
         "caption": "Логистическая регрессия на 2 информативных и 48 шумовых признаках: adversarial training "
                    "обнуляет веса шумовых признаков, чуть теряет на ε = 0, но держит точность под атакой."},
    ],
    "algorithmic_trading": [
        {"title": "Конвейер торговой системы", "svg": S_TRD,
         "caption": "Данные превращаются в признаки, модель выдаёт сигнал, риск-менеджмент задаёт размер позиции, "
                    "затем ордера исполняются. Бэктест замыкает цикл и проверяет стратегию на истории."},
        {"title": "Trend following на пересечении скользящих", "chart": trd_chart,
         "caption": "Синтетическая цена: позиция открыта, пока SMA 20 выше SMA 50 (вход на следующем баре, без "
                    "lookahead). Стратегия пропускает часть падений, но опаздывает на разворотах."},
    ],
    "als_recommender": [
        {"title": "R ≈ U·Vᵀ и чередующиеся шаги", "svg": S_ALS,
         "caption": "При фиксированной V задача для каждого uᵢ — ridge-регрессия по известным рейтингам "
                    "пользователя, и наоборот. Каждый шаг решается в замкнутой форме и легко параллелится."},
        {"title": "Сходимость ALS", "chart": als_chart,
         "caption": "Синтетическая матрица ранга 5, известно 30% рейтингов: ошибка падает за 3–5 итераций, "
                    "а test RMSE выходит на плато чуть выше уровня шума."},
    ],
    "anova": [
        {"title": "Межгрупповая и внутригрупповая вариация", "svg": S_ANOVA,
         "caption": "SSB — отклонения средних групп от общего среднего, SSW — разброс точек вокруг своего среднего "
                    "группы. F сравнивает их дисперсии с учётом степеней свободы."},
        {"title": "F-распределение и p-value", "chart": anova_chart,
         "caption": "При H₀ статистика имеет распределение F(k−1, N−k). Если наблюдаемое F попало в красную "
                    "область (правее F_крит), H₀ о равенстве средних отвергается."},
    ],
    "astrophysics_cosmology_ml": [
        {"title": "Simulation-based inference", "svg": S_ASTRO,
         "caption": "Параметры θ из prior прогоняются через симулятор; нейросеть учится по парам (θ, x) "
                    "приближать posterior q_φ(θ|x). Затем к реальным наблюдениям x_obs она применяется мгновенно."},
    ],
    "audio_generation": [
        {"title": "Dilated causal convolutions (WaveNet)", "svg": S_AUDIO,
         "caption": "Каждый слой удваивает dilation (1, 2, 4, 8), поэтому receptive field растёт экспоненциально: "
                    "выход зависит от 16 прошлых отсчётов при 4 слоях и не видит будущего."},
        {"title": "Waveform и спектрограмма", "chart": audio_chart,
         "caption": "Один и тот же сигнал (chirp с гармоникой): raw waveform — тысячи отсчётов в секунду, "
                    "спектрограмма компактнее и показывает рост частоты во времени."},
    ],
    "autoencoders": [
        {"title": "Encoder → bottleneck → decoder", "svg": S_AE,
         "caption": "Encoder сжимает 784 входа до z размерности 32, decoder восстанавливает x̂. Обучение "
                    "минимизирует ошибку восстановления, поэтому z вынужден хранить главное."},
        {"title": "Ошибка восстановления и размер z", "chart": ae_chart,
         "caption": "Для линейного автоэнкодера оптимум совпадает с PCA: ошибка быстро падает на первых "
                    "компонентах, дальше выигрыш от расширения bottleneck убывает."},
    ],
    "automl": [
        {"title": "Что автоматизирует AutoML", "svg": S_AUTOML,
         "caption": "AutoML перебирает пайплайны: предобработку, признаки, модель и гиперпараметры (задача CASH), "
                    "оценивая каждую попытку по CV в пределах бюджета времени, а лучшие объединяет в ансамбль."},
    ],
    "automl_neural_networks": [
        {"title": "Компоненты NAS", "svg": S_NAS,
         "caption": "Стратегия поиска выбирает архитектуру a из пространства A, оценка качества возвращает "
                    "score, и цикл повторяется. Weight sharing (one-shot NAS) удешевляет самый дорогой шаг — оценку."},
    ],
    "bayesian_approach": [
        {"title": "Prior × likelihood ∝ posterior", "svg": S_BAYES,
         "caption": "Пример с монетой: prior Beta(2,2) умножается на правдоподобие 7 орлов из 10, и получается "
                    "posterior Beta(9,5) — сопряжённость даёт ответ в замкнутой форме."},
        {"title": "Posterior по мере накопления данных", "chart": bayes_chart,
         "caption": "Чем больше бросков, тем уже posterior и тем ближе его пик к истинному θ = 0.7; "
                    "при малых n prior заметно тянет оценку к 0.5."},
    ],
    "bayesian_hyperparameter_optimization": [
        {"title": "Цикл байесовской оптимизации", "svg": S_BHO,
         "caption": "GP-суррогат по уже проверенным точкам даёт μ(x) и σ(x), acquisition function выбирает "
                    "следующую точку, модель обучается, и результат обновляет GP."},
        {"title": "GP-суррогат и Expected Improvement", "chart": bho_chart,
         "caption": "EI велик там, где μ(x) высокое или неопределённость σ(x) большая: следующая проба — компромисс "
                    "между exploitation и exploration."},
    ],
    "bert_finetuning": [
        {"title": "Голова классификации поверх BERT", "svg": S_BFT,
         "caption": "Для классификации берут выход токена [CLS] и добавляют новый линейный слой; обучаются и голова, "
                    "и все веса BERT, но с маленьким learning rate (2e-5)."},
        {"title": "Расписание learning rate", "chart": bft_chart,
         "caption": "warmup_steps = 500: lr плавно растёт от 0 до 2e-5, затем линейно убывает до 0. "
                    "Warmup защищает предобученные веса от крупных шагов в начале."},
    ],
    "bert_language_models": [
        {"title": "Masked Language Model", "svg": S_BLM,
         "caption": "Вход — сумма токенного, сегментного и позиционного эмбеддингов. BERT предсказывает "
                    "замаскированный токен по контексту с обеих сторон; [CLS] используется для NSP."},
    ],
    "bias_neural_networks": [
        {"title": "Петля усиления bias", "svg": S_BNN,
         "caption": "Смещённые данные дают смещённую модель, её решения порождают новые данные, и при "
                    "переобучении bias усиливается. Аудит нужен на каждом шаге цикла."},
        {"title": "Один порог — разный TPR", "chart": bnn_chart,
         "caption": "Если признаки для группы B измеряются хуже и скоры занижены, общий порог 0.5 пропускает "
                    "заметно больше её положительных примеров: нарушается equal opportunity."},
    ],
    "bias_variance_decomposition": [
        {"title": "Bias и variance на мишенях", "svg": S_BV,
         "caption": "Bias — насколько центр облака прогнозов смещён от истины, variance — насколько прогнозы "
                    "разбросаны между обучающими выборками."},
        {"title": "Разложение ошибки по сложности модели", "chart": bv_chart,
         "caption": "Посчитано по 200 выборкам: bias² падает с ростом степени полинома, variance растёт, "
                    "а их сумма плюс σ² образует U-образную кривую с минимумом на средней сложности."},
    ],
    "bidirectional_rnn": [
        {"title": "Развёрнутая BiRNN", "svg": S_BIRNN,
         "caption": "Прямая RNN идёт слева направо, обратная — справа налево, у каждой свои веса. "
                    "В каждом шаге их состояния склеиваются [→hₜ; ←hₜ], поэтому yₜ видит весь вход."},
    ],
    "binning_discretization": [
        {"title": "Равная ширина и равная частота", "svg": S_BIN,
         "caption": "На скошенных данных equal width кладёт почти все точки в первый бин, а equal frequency "
                    "(квантили) делает бины узкими там, где данных много."},
        {"title": "Границы бинов на lognormal-признаке", "chart": bin_chart,
         "caption": "Equal width оставляет правые бины почти пустыми; квантильный биннинг даёт по 200 точек "
                    "в каждом бине, но разной ширины."},
    ],
    "bioinformatics_computational_biology": [
        {"title": "Конвейер NGS → варианты", "svg": S_BIO,
         "caption": "Типичный путь от образца до клинической интерпретации; справа — форматы данных на выходе "
                    "каждого шага. ML (например, DeepVariant) заменяет эвристики на этапе variant calling."},
    ],
    "biologically_inspired_architectures": [
        {"title": "LIF-нейрон", "svg": S_LIF,
         "caption": "Потенциал v накапливает входные спайки и постепенно «утекает» (β < 1). При превышении "
                    "порога θ нейрон выдаёт спайк и сбрасывает v."},
        {"title": "Динамика мембранного потенциала", "chart": lif_chart,
         "caption": "Одиночные входы затухают, не дойдя до порога; спайк возникает только когда несколько входов "
                    "приходят близко по времени — в SNN важен тайминг."},
    ],
    "birch": [
        {"title": "CF-дерево", "svg": S_BIRCH,
         "caption": "Каждый узел хранит только CF-тройки (N, LS, SS), а не сами точки; они складываются, "
                    "поэтому точку можно добавить за один проход сверху вниз."},
        {"title": "Subclusters и финальные кластеры", "chart": birch_chart,
         "caption": "Листья CF-дерева дают десятки компактных subclusters (крестики), а финальный шаг "
                    "объединяет их в n_clusters = 4 кластера."},
    ],
    "blending": [
        {"title": "Схема блендинга", "svg": S_BLEND,
         "caption": "Базовые модели учатся на train, их прогнозы на holdout становятся признаками для мета-модели. "
                    "На test тот же путь: прогнозы базовых моделей → мета-модель → оценка."},
    ],
    "bootstrap_methods": [
        {"title": "Алгоритм bootstrap", "svg": S_BOOT,
         "caption": "Из исходной выборки B раз тянем n элементов с возвратом (элементы повторяются), на каждой "
                    "считаем статистику θ*, и по распределению θ* строим доверительный интервал."},
        {"title": "Bootstrap-распределение среднего", "chart": boot_chart,
         "caption": "Данные из примера шпаргалки: перцентильный 95% CI — это 2.5% и 97.5% квантили "
                    "bootstrap-средних, без предположений о нормальности."},
    ],
    "box_jenkins": [
        {"title": "Итерационный цикл Box-Jenkins", "svg": S_BJ,
         "caption": "Сначала добиваемся стационарности (d), по ACF/PACF выбираем p и q, оцениваем модель и "
                    "проверяем остатки. Если они не белый шум — возвращаемся к идентификации."},
        {"title": "ACF и PACF процесса AR(2)", "chart": bj_chart,
         "caption": "Для AR(p) PACF обрывается после лага p, а ACF затухает постепенно; для MA(q) — наоборот. "
                    "Серая полоса — ±1.96/√n."},
    ],
    "brain_neural_networks_connection": [
        {"title": "Биологический и искусственный нейрон", "svg": S_BRAIN,
         "caption": "Синапсы соответствуют весам, дендриты — входам, тело клетки — суммированию, "
                    "а потенциал действия в аксоне — функции активации."},
    ],
    "canonical_correlation_analysis": [
        {"title": "Идея CCA", "svg": S_CCA,
         "caption": "CCA ищет веса a и b, при которых проекции u = Xa и v = Yb максимально коррелируют. "
                    "Следующие пары ищутся некоррелированными с предыдущими."},
        {"title": "Канонические пары на синтетике", "chart": cca_chart,
         "caption": "X и Y связаны двумя скрытыми факторами: первая каноническая пара ловит более сильную "
                    "связь, вторая — более слабую."},
    ],
    "canonical_correlation_analysis_method": [
        {"title": "CCA, PCA и LDA", "svg": S_CCAM,
         "caption": "PCA работает с одним набором X, LDA — с X и метками, CCA — с двумя наборами признаков "
                    "и максимизирует корреляцию их линейных комбинаций."},
        {"title": "Канонические корреляции по компонентам", "chart": ccam_chart,
         "caption": "Число заметных ρₖ показывает, сколько независимых связей между X и Y: здесь две, "
                    "третья компонента ловит только шум."},
    ],
}
