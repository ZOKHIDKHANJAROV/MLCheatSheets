# -*- coding: utf-8 -*-
"""Образцы рисунков для tools/add_visuals.py: SVG-схема (GRU) и график matplotlib (квантильная регрессия)."""

import numpy as np

GRU_CELL = """
<svg viewBox="0 0 420 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Схема ячейки GRU">
  <defs><marker id="gru-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>
  <!-- линия состояния h(t-1) → h(t) -->
  <text x="8" y="44">hₜ₋₁</text>
  <line class="ln" x1="40" y1="40" x2="380" y2="40" marker-end="url(#gru-a)"/>
  <text x="386" y="44">hₜ</text>
  <!-- вход x(t) -->
  <text x="20" y="214">xₜ</text>
  <line class="ln" x1="38" y1="208" x2="300" y2="208"/>
  <!-- гейты -->
  <rect class="c4" x="70" y="140" width="70" height="34" rx="6"/>
  <text x="105" y="161" text-anchor="middle">σ  reset rₜ</text>
  <rect class="c1" x="170" y="140" width="76" height="34" rx="6"/>
  <text x="208" y="161" text-anchor="middle">σ  update zₜ</text>
  <rect class="c2" x="270" y="140" width="70" height="34" rx="6"/>
  <text x="305" y="161" text-anchor="middle">tanh  h̃ₜ</text>
  <line class="ln" x1="105" y1="208" x2="105" y2="176" marker-end="url(#gru-a)"/>
  <line class="ln" x1="208" y1="208" x2="208" y2="176" marker-end="url(#gru-a)"/>
  <line class="ln" x1="300" y1="208" x2="300" y2="176" marker-end="url(#gru-a)"/>
  <!-- h(t-1) подаётся в гейты -->
  <line class="ln s6" x1="60" y1="40" x2="60" y2="128" stroke-dasharray="4 3"/>
  <line class="ln s6" x1="60" y1="128" x2="105" y2="128" stroke-dasharray="4 3" marker-end="url(#gru-a)"/>
  <line class="ln s6" x1="60" y1="128" x2="208" y2="128" stroke-dasharray="4 3"/>
  <line class="ln s6" x1="208" y1="128" x2="208" y2="138" stroke-dasharray="4 3" marker-end="url(#gru-a)"/>
  <!-- reset влияет на кандидата -->
  <circle class="c4" cx="250" cy="112" r="10"/><text x="250" y="116" text-anchor="middle">⊙</text>
  <line class="ln s4" x1="140" y1="150" x2="241" y2="117" marker-end="url(#gru-a)"/>
  <line class="ln s4" x1="258" y1="120" x2="290" y2="138" marker-end="url(#gru-a)"/>
  <text class="mut" x="152" y="108" font-size="10">rₜ ⊙ hₜ₋₁</text>
  <!-- смешивание: (1 − z)·h + z·h̃ -->
  <circle class="c1" cx="230" cy="40" r="13"/><text x="230" y="44" text-anchor="middle">⊙</text>
  <circle class="c2" cx="330" cy="40" r="13"/><text x="330" y="44" text-anchor="middle">+</text>
  <line class="ln s1" x1="208" y1="138" x2="226" y2="54" marker-end="url(#gru-a)"/>
  <line class="ln s2" x1="305" y1="138" x2="326" y2="55" marker-end="url(#gru-a)"/>
  <text class="mut" x="160" y="76" font-size="10">1 − zₜ: сколько оставить</text>
  <text class="mut" x="318" y="98" font-size="10">zₜ ⊙ h̃ₜ</text>
</svg>
"""


def quantile_chart(fig):
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 160))
    y = 1 + 0.5 * x + rng.normal(0, 0.25 + 0.15 * x)
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(x, y, s=6, alpha=0.35, color="#8a96a3")
    from sklearn.linear_model import QuantileRegressor
    for q, c in [(0.1, "#2e8b57"), (0.5, "#1a5fb4"), (0.9, "#d32f2f")]:
        m = QuantileRegressor(quantile=q, alpha=0, solver="highs").fit(x[:, None], y)
        ax.plot(x, m.predict(x[:, None]), color=c, lw=2, label=f"τ = {q}")
    ax.set_title("Линии квантилей расходятся\nс ростом разброса", fontsize=9)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.legend(fontsize=7)
    ax2 = fig.add_subplot(2, 1, 2)
    e = np.linspace(-2, 2, 200)
    for q, c in [(0.1, "#2e8b57"), (0.5, "#1a5fb4"), (0.9, "#d32f2f")]:
        ax2.plot(e, np.where(e >= 0, q * e, (q - 1) * e), color=c, lw=2, label=f"τ = {q}")
    ax2.set_title("Pinball loss: штраф\nнесимметричен", fontsize=9)
    ax2.set_xlabel("ошибка y − ŷ"), ax2.legend(fontsize=7)
    fig.set_size_inches(4.2, 5.2)
    fig.tight_layout()


VISUALS = {
    "gru": [
        {"title": "Схема ячейки GRU", "svg": GRU_CELL,
         "caption": "Reset-гейт решает, сколько прошлого брать в кандидата h̃ₜ; update-гейт zₜ смешивает "
                    "старое состояние и кандидата: hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ."},
    ],
    "quantile_regression": [
        {"title": "Квантили и pinball loss", "chart": quantile_chart,
         "caption": "Слева — регрессия на 10-й, 50-й и 90-й квантили при растущем шуме: интервал расширяется. "
                    "Справа — функция потерь: при τ = 0.9 недопрогноз штрафуется в 9 раз сильнее перепрогноза."},
    ],
}
