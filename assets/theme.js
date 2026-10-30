// Тема оформления для всех страниц проекта: "auto" (как в системе), "light" или "dark".
// Выбор хранится в localStorage и переключается кнопкой в портале (index.html).
// Скрипт подключается в <head> без defer, чтобы тема применилась до отрисовки — без «вспышки» белого.
(function () {
  var KEY = "mlcs-theme";
  var MODES = ["auto", "light", "dark"];
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function stored() {
    try { var m = localStorage.getItem(KEY); return MODES.indexOf(m) >= 0 ? m : "auto"; }
    catch (e) { return "auto"; }
  }

  var mode = stored();

  function apply(m) {
    mode = MODES.indexOf(m) >= 0 ? m : "auto";
    var dark = mode === "dark" || (mode === "auto" && !!media && media.matches);
    root.setAttribute("data-theme", dark ? "dark" : "light");
    root.setAttribute("data-theme-mode", mode);
  }

  // Передаём тему вложенным страницам (шпаргалка в портале открыта во iframe).
  // При открытии через file:// у iframe может быть своё хранилище, поэтому сообщаем явно.
  function notifyFrames() {
    var frames = document.querySelectorAll("iframe");
    for (var i = 0; i < frames.length; i++) {
      try { frames[i].contentWindow.postMessage({ mlcsTheme: mode }, "*"); } catch (e) { /* фрейм ещё не загружен */ }
    }
  }

  function set(m) {
    apply(m);
    try { localStorage.setItem(KEY, mode); } catch (e) { /* хранилище недоступно — тема действует до перезагрузки */ }
    notifyFrames();
  }

  apply(mode);

  if (media) {
    var onSystemChange = function () { if (mode === "auto") { apply("auto"); notifyFrames(); } };
    if (media.addEventListener) media.addEventListener("change", onSystemChange);
    else if (media.addListener) media.addListener(onSystemChange);
  }
  window.addEventListener("storage", function (e) { if (e.key === KEY) apply(e.newValue); });
  window.addEventListener("message", function (e) {
    if (e.source === window.parent && e.data && MODES.indexOf(e.data.mlcsTheme) >= 0) apply(e.data.mlcsTheme);
  });

  window.mlcsTheme = { get: function () { return mode; }, set: set, notifyFrames: notifyFrames };
})();
