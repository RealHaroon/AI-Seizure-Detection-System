// Event page: chart of movement in the bed area over time, so a reviewer can see why the alert fired.
// Single series (movement) + alert-level reference line + event span + "alert fired" marker.
// Plain SVG, redrawn on resize; hover shows a crosshair and the exact value.
(function () {
  const root = document.getElementById("trace");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const niceStep = (range, target) => {
    const raw = range / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  };
  const fmt = (v, d = 3) => (v == null ? "–" : v.toFixed(d));

  fetch(root.dataset.src, { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (!data || !data.samples.length) {
        root.querySelector(".chart-empty").hidden = false;
        return;
      }
      const pts = data.samples.filter((s) => s[1] != null);
      if (!pts.length) {
        root.querySelector(".chart-empty").hidden = false;
        return;
      }
      root.querySelector(".chart-wrap").hidden = false;
      fillTable(pts);
      const draw = () => render(root.querySelector(".chart-svg"), data, pts);
      draw();
      new ResizeObserver(draw).observe(root.querySelector(".chart-wrap"));
    })
    .catch(() => { root.querySelector(".chart-empty").hidden = false; });

  function fillTable(pts) {
    // One row per second (mean movement), for screen readers and exact values.
    const body = root.querySelector("tbody"), bins = new Map();
    for (const [t, m] of pts) {
      const k = Math.floor(t);
      if (!bins.has(k)) bins.set(k, []);
      bins.get(k).push(m);
    }
    for (const [k, ms] of [...bins.entries()].sort((a, b) => a[0] - b[0])) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${k}</td><td>${fmt(ms.reduce((a, b) => a + b, 0) / ms.length)}</td>`;
      body.appendChild(tr);
    }
  }

  function render(svg, data, pts) {
    svg.textContent = "";
    const W = svg.clientWidth || 600, H = 260;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    const m = { l: 48, r: 16, t: 22, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const t0 = pts[0][0], t1 = pts[pts.length - 1][0];
    const thr = data.threshold;
    // Log scale: the alert level is tiny next to the peaks of a real seizure; on a linear axis the moment
    // movement crosses it would be invisible. Values below the floor (near-stillness) sit on the baseline.
    const yMin = thr ? thr / 10 : 0.001;
    const yMax = Math.max(...pts.map((p) => p[1]), thr ? thr * 10 : yMin * 100) * 1.3;
    const L = Math.log10, ly0 = L(yMin), ly1 = L(yMax);
    const x = (t) => m.l + ((t - t0) / (t1 - t0 || 1)) * iw;
    const y = (v) => m.t + ih - ((L(Math.max(v, yMin)) - ly0) / (ly1 - ly0)) * ih;

    // Gridlines + y ticks at powers of ten (hairline, recessive)
    for (let e = Math.ceil(ly0); e <= Math.floor(ly1); e++) {
      const v = Math.pow(10, e);
      el("line", { x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v), class: "c-grid" }, svg);
      el("text", { x: m.l - 6, y: y(v) + 4, class: "c-tick", "text-anchor": "end" }, svg).textContent =
        v >= 1 ? v.toFixed(0) : v.toString();
    }
    // x ticks: seconds from when the movement began
    const xs = niceStep(t1 - t0, 6);
    for (let t = Math.ceil(t0 / xs) * xs; t <= t1 + 1e-9; t += xs) {
      el("text", { x: x(t), y: m.t + ih + 18, class: "c-tick", "text-anchor": "middle" }, svg).textContent =
        `${Math.round(t)} s`;
    }
    el("text", { x: m.l, y: 12, class: "c-axis-label" }, svg).textContent = "Movement (bed widths / s, log scale)";

    // Event span (movement start -> end), a neutral wash behind the data
    const evEnd = data.end != null ? data.end : t1;
    el("rect", { x: x(0), y: m.t, width: Math.max(0, x(evEnd) - x(0)), height: ih, class: "c-span" }, svg);

    // Series: 10% area wash + 2px line
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    el("path", { d: `${line}L${x(t1)},${y(yMin)}L${x(t0)},${y(yMin)}Z`, class: "c-area" }, svg);
    el("path", { d: line, class: "c-line" }, svg);

    // Alert level (reference line) with its label
    if (thr) {
      el("line", { x1: m.l, x2: m.l + iw, y1: y(thr), y2: y(thr), class: "c-ref" }, svg);
      // Label at the left: before the movement starts the trace is calm, so the label never sits on the line.
      el("text", { x: m.l + 4, y: y(thr) - 6, class: "c-ref-label", "text-anchor": "start" }, svg).textContent =
        `Alert level ${thr}`;
    }
    // Alert fired marker (status: critical, always with a text label)
    const ax = x(data.detected_at);
    el("line", { x1: ax, x2: ax, y1: m.t, y2: m.t + ih, class: "c-alert" }, svg);
    el("text", { x: ax + 6, y: m.t + 12, class: "c-alert-label" }, svg).textContent =
      `Alert fired (+${data.detected_at.toFixed(1)} s)`;

    // Hover: crosshair + tooltip with the nearest sample
    const hair = el("line", { y1: m.t, y2: m.t + ih, class: "c-hair", visibility: "hidden" }, svg);
    const dot = el("circle", { r: 4, class: "c-dot", visibility: "hidden" }, svg);
    const tip = root.querySelector(".chart-tip");
    const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const t = t0 + ((ev.clientX - r.left) * (W / r.width) - m.l) / iw * (t1 - t0);
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
      const px = x(best[0]), py = y(best[1]);
      hair.setAttribute("x1", px); hair.setAttribute("x2", px); hair.setAttribute("visibility", "visible");
      dot.setAttribute("cx", px); dot.setAttribute("cy", py); dot.setAttribute("visibility", "visible");
      const ratio = thr ? ` · ${(best[1] / thr).toFixed(1)}× alert level` : "";
      tip.textContent = `${best[0].toFixed(1)} s · movement ${fmt(best[1])}${ratio}`;
      tip.hidden = false;
      const left = Math.min(Math.max(px * (r.width / W), 60), r.width - 60);
      tip.style.left = `${left}px`;
    });
    hit.addEventListener("pointerleave", () => {
      hair.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true;
    });
  }
})();
