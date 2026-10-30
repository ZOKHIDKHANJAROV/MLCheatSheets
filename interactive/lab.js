// Общие помощники для интерактивных демонстраций (interactive/*.html): холст, цвета темы, ползунки, случайные числа.
// Без внешних библиотек — страницы работают офлайн, в том числе через file://.
(function () {
  "use strict";

  // Холст, который подстраивается под ширину контейнера и плотность пикселей экрана.
  // draw(ctx, w, h) вызывается при каждом изменении размера; ratio — высота / ширина.
  function canvas(el, ratio, draw) {
    const ctx = el.getContext("2d");
    const view = { ctx, w: 0, h: 0, draw };
    function resize() {
      const w = el.clientWidth;
      const h = Math.round(w * ratio);
      const dpr = window.devicePixelRatio || 1;
      el.style.height = h + "px";
      el.width = Math.round(w * dpr);
      el.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      view.w = w; view.h = h;
      view.draw(ctx, w, h);
    }
    view.redraw = () => { if (view.w > 0) view.draw(ctx, view.w, view.h); };   // до первой разметки размер ещё 0
    new ResizeObserver(resize).observe(el);
    onTheme(view.redraw);
    return view;
  }

  // Цвета берутся из CSS-переменных, поэтому графики сразу перекрашиваются под тёмную тему.
  function color(name) {
    return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
  }
  function onTheme(fn) {
    new MutationObserver(fn).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  // Линейное отображение координат данных в пиксели.
  function scale(d0, d1, r0, r1) {
    const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
    f.invert = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
    return f;
  }

  // Ползунок <input type=range id=...> с выводом значения в <output for=...>.
  function slider(id, onInput, format) {
    const input = document.getElementById(id);
    const out = document.querySelector(`output[for="${id}"]`);
    const fmt = format || (v => v);
    const update = () => { if (out) out.textContent = fmt(+input.value); onInput(+input.value); };
    input.addEventListener("input", update);
    if (out) out.textContent = fmt(+input.value);
    return { get value() { return +input.value; }, set value(v) { input.value = v; update(); }, input };
  }

  // Детерминированный генератор (mulberry32), чтобы «новые данные» можно было воспроизвести по seed.
  function rng(seed) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.normal = () => {
      const u = 1 - next(), v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return next;
  }

  // Сетка и оси для графика в координатах данных.
  function axes(ctx, sx, sy, opts) {
    const o = Object.assign({ xTicks: [], yTicks: [], xLabel: "", yLabel: "" }, opts);
    ctx.save();
    ctx.strokeStyle = color("grid");
    ctx.lineWidth = 1;
    ctx.fillStyle = color("muted");
    ctx.font = "11px system-ui, sans-serif";
    o.xTicks.forEach(t => {
      const x = Math.round(sx(t)) + .5;
      ctx.beginPath(); ctx.moveTo(x, sy.range0); ctx.lineTo(x, sy.range1); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(fmtTick(t), x, sy.range0 + 14);
    });
    o.yTicks.forEach(t => {
      const y = Math.round(sy(t)) + .5;
      ctx.beginPath(); ctx.moveTo(sx.range0, y); ctx.lineTo(sx.range1, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(fmtTick(t), sx.range0 - 5, y + 4);
    });
    if (o.xLabel) { ctx.textAlign = "right"; ctx.fillText(o.xLabel, sx.range1, sy.range0 + 28); }
    if (o.yLabel) { ctx.textAlign = "left"; ctx.fillText(o.yLabel, sx.range0 + 4, sy.range1 - 6); }
    ctx.restore();
  }
  function fmtTick(t) { return Math.abs(t) >= 100 || Number.isInteger(t) ? String(t) : t.toFixed(1); }

  // Шкала с запомненными границами области рисования (нужно для axes).
  function frame(d0, d1, r0, r1) {
    const s = scale(d0, d1, r0, r1);
    s.range0 = r0; s.range1 = r1;
    return s;
  }

  function dot(ctx, x, y, r, fill, stroke) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
  }

  // Координаты указателя мыши или пальца внутри холста.
  function pointer(el, ev) {
    const r = el.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }

  // Двумерные наборы для классификации: точки [x, y, класс 0/1] в области x ∈ [−1.2, 1.2], y ∈ [−1, 1].
  function dataset(shape, n, noise, r) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      let c = i % 2, x, y;
      if (shape === "moons") {
        const t = Math.PI * r();
        x = (c ? 1 - Math.cos(t) : Math.cos(t)) * 0.5 - 0.25;
        y = (c ? 0.5 - Math.sin(t) : Math.sin(t)) * 0.7 - 0.1;
      } else if (shape === "circles") {
        const t = 2 * Math.PI * r(), rad = c ? 0.3 : 0.75;
        x = Math.cos(t) * rad; y = Math.sin(t) * rad * 0.95;
      } else if (shape === "xor") {
        x = r() * 1.9 - 0.95; y = r() * 1.7 - 0.85;
        c = (x > 0) !== (y > 0) ? 1 : 0;
      } else {   // blobs
        x = (c ? 0.35 : -0.35) + r.normal() * 0.28; y = (c ? -0.2 : 0.2) + r.normal() * 0.28;
      }
      pts.push([x + r.normal() * noise * 0.5, y + r.normal() * noise * 0.5, c]);
    }
    return pts;
  }

  window.Lab = { canvas, color, onTheme, scale, frame, slider, rng, axes, dot, pointer, dataset };
})();
