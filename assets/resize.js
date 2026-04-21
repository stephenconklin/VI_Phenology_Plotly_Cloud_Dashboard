/* Drag-to-resize dividers — pure DOM, no Dash callbacks involved.
   1. #sidebar-resize  — horizontal, adjusts sidebar width
   2. #resize-divider  — vertical, adjusts map / charts split             */

(function () {
  "use strict";

  function init() {
    var divider    = document.getElementById("resize-divider");
    var mapWrapper = document.getElementById("map-wrapper");
    var mainPanel  = document.getElementById("main-panel-col");
    var sideHandle = document.getElementById("sidebar-resize");
    var sidebarCol = document.getElementById("sidebar-col");

    if (!divider || !mapWrapper || !mainPanel) return false;

    // ── Sidebar horizontal resize ───────────────────────────────────────────

    if (sideHandle && sidebarCol) {
      var hDragging  = false;
      var hStartX    = 0;
      var hStartW    = 0;

      sideHandle.addEventListener("mousedown", function (e) {
        e.preventDefault();
        hDragging = true;
        hStartX   = e.clientX;
        hStartW   = sidebarCol.getBoundingClientRect().width;
        sideHandle.classList.add("dragging");
        document.body.style.cursor     = "col-resize";
        document.body.style.userSelect = "none";
      });

      document.addEventListener("mousemove", function (e) {
        if (!hDragging) return;
        var delta = e.clientX - hStartX;
        var minW  = 160;
        var maxW  = Math.round(window.innerWidth * 0.55);
        var newW  = Math.min(Math.max(hStartW + delta, minW), maxW);
        document.documentElement.style.setProperty("--sidebar-w", newW + "px");
      });

      document.addEventListener("mouseup", function () {
        if (!hDragging) return;
        hDragging = false;
        sideHandle.classList.remove("dragging");
        document.body.style.cursor     = "";
        document.body.style.userSelect = "";
        setTimeout(fireResize, 50);
      });

      sideHandle.addEventListener("touchstart", function (e) {
        var t     = e.touches[0];
        hDragging = true;
        hStartX   = t.clientX;
        hStartW   = sidebarCol.getBoundingClientRect().width;
        sideHandle.classList.add("dragging");
      }, { passive: true });

      document.addEventListener("touchmove", function (e) {
        if (!hDragging) return;
        var t    = e.touches[0];
        var delta = t.clientX - hStartX;
        var minW  = 160;
        var maxW  = Math.round(window.innerWidth * 0.55);
        var newW  = Math.min(Math.max(hStartW + delta, minW), maxW);
        document.documentElement.style.setProperty("--sidebar-w", newW + "px");
      }, { passive: true });

      document.addEventListener("touchend", function () {
        if (!hDragging) return;
        hDragging = false;
        sideHandle.classList.remove("dragging");
        setTimeout(fireResize, 50);
      });
    }

    var dragging  = false;
    var startY    = 0;
    var startMapH = 0;

    // ── Mouse drag ──────────────────────────────────────────────────────────

    divider.addEventListener("mousedown", function (e) {
      e.preventDefault();
      dragging  = true;
      startY    = e.clientY;
      startMapH = mapWrapper.getBoundingClientRect().height;
      divider.classList.add("dragging");
      document.body.style.cursor     = "row-resize";
      document.body.style.userSelect = "none";
    });

    document.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      var delta  = e.clientY - startY;
      var panelH = mainPanel.getBoundingClientRect().height;
      var minH   = 80;
      var maxH   = panelH - 6 - 80;   // 6px divider + 80px min chart area
      var newH   = Math.min(Math.max(startMapH + delta, minH), maxH);
      mainPanel.style.setProperty("--map-h", newH + "px");
      mainPanel.style.setProperty("--charts-h", (panelH - newH - 6) + "px");
    });

    document.addEventListener("mouseup", function () {
      if (!dragging) return;
      dragging = false;
      divider.classList.remove("dragging");
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
      afterDrag();
    });

    // ── Touch drag ──────────────────────────────────────────────────────────

    divider.addEventListener("touchstart", function (e) {
      var t = e.touches[0];
      dragging  = true;
      startY    = t.clientY;
      startMapH = mapWrapper.getBoundingClientRect().height;
      divider.classList.add("dragging");
    }, { passive: true });

    document.addEventListener("touchmove", function (e) {
      if (!dragging) return;
      var t      = e.touches[0];
      var delta  = t.clientY - startY;
      var panelH = mainPanel.getBoundingClientRect().height;
      var minH   = 80;
      var maxH   = panelH - 6 - 80;
      var newH   = Math.min(Math.max(startMapH + delta, minH), maxH);
      mainPanel.style.setProperty("--map-h", newH + "px");
      mainPanel.style.setProperty("--charts-h", (panelH - newH - 6) + "px");
    }, { passive: true });

    document.addEventListener("touchend", function () {
      if (!dragging) return;
      dragging = false;
      divider.classList.remove("dragging");
      afterDrag();
    });

    // ── Post-drag cleanup ───────────────────────────────────────────────────

    function fireResize() {
      // Leaflet listens to window 'resize' and calls invalidateSize automatically.
      // Plotly (responsive:true) does the same. One event handles both.
      window.dispatchEvent(new Event("resize"));
    }

    function afterDrag() {
      // Small delay so the CSS flex layout finishes reflowing before resize fires.
      setTimeout(fireResize, 50);
    }

    // Set --charts-h on init so the CSS calc is valid before any drag.
    (function () {
      var panelH = mainPanel.getBoundingClientRect().height;
      var mapH   = mapWrapper.getBoundingClientRect().height;
      mainPanel.style.setProperty("--charts-h", (panelH - mapH - 6) + "px");
    })();

    // Fire once on init so Leaflet recalculates its size after React hydration.
    setTimeout(fireResize, 200);

    return true;
  }

  // Retry until Dash has rendered the component tree (up to 20 × 200 ms = 4 s)
  function tryInit(attempts) {
    if (init()) return;
    if (attempts > 0) setTimeout(function () { tryInit(attempts - 1); }, 200);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { tryInit(20); });
  } else {
    tryInit(20);
  }
})();
