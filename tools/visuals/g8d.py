# -*- coding: utf-8 -*-
"""Рисунки группы 8d (ИИ-агенты): оценка агентов и безопасность агентов."""

import numpy as np

EVAL_PIPELINE = """
<svg viewBox="0 0 340 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Конвейер оценки агента">
  <defs><marker id="agent_evaluation-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>
  <rect class="c1" x="10" y="10" width="140" height="40" rx="6"/>
  <text x="80" y="27" text-anchor="middle" style="font-size:12px;font-weight:600">Набор задач</text>
  <text x="80" y="42" text-anchor="middle" class="mut" style="font-size:10px">цель + проверка</text>
  <rect class="c5" x="190" y="10" width="140" height="40" rx="6"/>
  <text x="260" y="27" text-anchor="middle" style="font-size:12px;font-weight:600">Агент</text>
  <text x="260" y="42" text-anchor="middle" class="mut" style="font-size:10px">n попыток на задачу</text>
  <line class="ln" x1="150" y1="30" x2="188" y2="30" marker-end="url(#agent_evaluation-a)"/>
  <rect class="c6" x="190" y="86" width="140" height="40" rx="6"/>
  <text x="260" y="103" text-anchor="middle" style="font-size:12px;font-weight:600">Трасса</text>
  <text x="260" y="118" text-anchor="middle" class="mut" style="font-size:10px">шаги, токены, время</text>
  <line class="ln" x1="260" y1="50" x2="260" y2="84" marker-end="url(#agent_evaluation-a)"/>
  <rect class="c3" x="10" y="86" width="140" height="40" rx="6"/>
  <text x="80" y="103" text-anchor="middle" style="font-size:12px;font-weight:600">Оценщики</text>
  <text x="80" y="118" text-anchor="middle" class="mut" style="font-size:10px">код / эталон / LLM-судья</text>
  <line class="ln" x1="188" y1="106" x2="152" y2="106" marker-end="url(#agent_evaluation-a)"/>
  <text x="170" y="98" text-anchor="middle" class="mut" style="font-size:9px">проверка</text>
  <rect class="c2" x="10" y="170" width="320" height="50" rx="6"/>
  <text x="170" y="190" text-anchor="middle" style="font-size:12px;font-weight:600">Метрики по задачам (c из n)</text>
  <text x="170" y="208" text-anchor="middle" class="mut" style="font-size:11px">pass@k · pass^k · шаги · стоимость · латентность</text>
  <line class="ln" x1="80" y1="126" x2="80" y2="168" marker-end="url(#agent_evaluation-a)"/>
  <line class="ln" x1="260" y1="126" x2="260" y2="168" marker-end="url(#agent_evaluation-a)"/>
  <rect class="c4" x="10" y="248" width="320" height="40" rx="6"/>
  <text x="170" y="265" text-anchor="middle" style="font-size:12px;font-weight:600">Регрессионный набор</text>
  <text x="170" y="280" text-anchor="middle" class="mut" style="font-size:10px">сбои из трасс становятся новыми задачами</text>
  <line class="ln" x1="170" y1="220" x2="170" y2="246" marker-end="url(#agent_evaluation-a)"/>
  <path class="ln s6" stroke-dasharray="4 3" d="M334,268 h4 v-250 h-4" marker-end="url(#agent_evaluation-a)"/>
</svg>
"""


def pass_chart(fig):
    from math import comb
    n, c_values = 20, [16]
    ks = np.arange(1, 11)
    ax = fig.add_subplot(1, 1, 1)
    c = c_values[0]
    at = [1 - comb(n - c, int(k)) / comb(n, int(k)) for k in ks]
    hat = [comb(c, int(k)) / comb(n, int(k)) for k in ks]
    ax.plot(ks, at, marker="o", lw=2, label="pass@k")
    ax.plot(ks, hat, marker="s", lw=2, label="pass^k")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k (число попыток)")
    ax.set_ylabel("вероятность")
    ax.set_title("Одна задача: n = 20, c = 16 (доля успеха 0.8)\nвозможность растёт, надёжность падает", fontsize=9)
    ax.legend(fontsize=8)
    fig.set_size_inches(4.2, 3.2)
    fig.tight_layout()


SAFETY_TRIAD = """
<svg viewBox="0 0 340 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Смертельная триада агента">
  <defs><marker id="agent_safety-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>
  <rect class="c1" x="10" y="10" width="150" height="46" rx="6"/>
  <text x="85" y="29" text-anchor="middle" style="font-size:12px;font-weight:600">1. Приватные данные</text>
  <text x="85" y="46" text-anchor="middle" class="mut" style="font-size:10px">почта, файлы, БД</text>
  <rect class="c3" x="180" y="10" width="150" height="46" rx="6"/>
  <text x="255" y="29" text-anchor="middle" style="font-size:12px;font-weight:600">2. Недоверенный контент</text>
  <text x="255" y="46" text-anchor="middle" class="mut" style="font-size:10px">веб, входящие письма</text>
  <rect class="c5" x="95" y="84" width="150" height="46" rx="6"/>
  <text x="170" y="103" text-anchor="middle" style="font-size:12px;font-weight:600">3. Внешний канал</text>
  <text x="170" y="120" text-anchor="middle" class="mut" style="font-size:10px">email, HTTP, публикация</text>
  <line class="ln s4" x1="85" y1="58" x2="150" y2="150" marker-end="url(#agent_safety-a)"/>
  <line class="ln s4" x1="255" y1="58" x2="190" y2="150" marker-end="url(#agent_safety-a)"/>
  <line class="ln s4" x1="170" y1="132" x2="170" y2="150" marker-end="url(#agent_safety-a)"/>
  <rect class="c4" x="85" y="152" width="170" height="40" rx="8"/>
  <text x="170" y="169" text-anchor="middle" style="font-size:12px;font-weight:600">Утечка данных</text>
  <text x="170" y="184" text-anchor="middle" class="mut" style="font-size:10px">все три вместе в одной сессии</text>
  <text x="170" y="216" text-anchor="middle" style="font-size:11px;font-weight:600">Защита: убрать любую одну ногу</text>
  <rect class="c2" x="10" y="228" width="100" height="58" rx="6"/>
  <text x="60" y="246" text-anchor="middle" style="font-size:10px;font-weight:600">Нет секретов</text>
  <text x="60" y="260" text-anchor="middle" class="mut" style="font-size:9px">минимум данных,</text>
  <text x="60" y="272" text-anchor="middle" class="mut" style="font-size:9px">least privilege</text>
  <rect class="c2" x="120" y="228" width="100" height="58" rx="6"/>
  <text x="170" y="246" text-anchor="middle" style="font-size:10px;font-weight:600">Нет недоверия</text>
  <text x="170" y="260" text-anchor="middle" class="mut" style="font-size:9px">источники только</text>
  <text x="170" y="272" text-anchor="middle" class="mut" style="font-size:9px">из белого списка</text>
  <rect class="c2" x="230" y="228" width="100" height="58" rx="6"/>
  <text x="280" y="246" text-anchor="middle" style="font-size:10px;font-weight:600">Нет канала</text>
  <text x="280" y="260" text-anchor="middle" class="mut" style="font-size:9px">закрытая сеть,</text>
  <text x="280" y="272" text-anchor="middle" class="mut" style="font-size:9px">подтверждение человеком</text>
</svg>
"""

VISUALS = {
    "agent_evaluation": [
        {"title": "Конвейер оценки агента", "svg": EVAL_PIPELINE,
         "caption": "Каждая задача прогоняется n раз; трассы и проверки дают метрики по задачам. "
                    "Найденные сбои возвращаются в регрессионный набор."},
        {"title": "pass@k и pass^k от k", "chart": pass_chart,
         "caption": "При доле успеха 0.8 вероятность «хотя бы один успех» pass@k быстро приближается к 1, "
                    "а вероятность «все k успешны» pass^k убывает примерно как 0.8ᵏ: надёжность и способность — разные метрики."},
    ],
    "agent_safety": [
        {"title": "Смертельная триада", "svg": SAFETY_TRIAD,
         "caption": "Утечка возможна, когда в одной сессии сочетаются приватные данные, недоверенный контент "
                    "и внешний канал. Достаточно надёжно убрать любую одну из трёх возможностей."},
    ],
}
