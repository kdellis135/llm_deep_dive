/* LLM Deep Dive — interactive widgets (vanilla JS, no dependencies).
 * Usage in Markdown:  <div class="widget" data-widget="softmax"></div>
 */
(function () {
  "use strict";

  // ---------- helpers ----------
  const PAL = ["#4f7cff", "#ff6d3a", "#21a366", "#b04fd6", "#e0a100", "#00a3b4", "#e0457b"];
  const redrawers = [];

  function el(tag, attrs, parent, text) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function fg(node) { return getComputedStyle(node).color || "#333"; }
  function alpha(color, a) {
    const m = color.match(/[\d.]+/g);
    if (!m || m.length < 3) return color;
    return "rgba(" + m[0] + "," + m[1] + "," + m[2] + "," + a + ")";
  }
  function title(root, t) { el("div", { class: "w-title" }, root, t); }
  function controls(root) { return el("div", { class: "w-controls" }, root); }
  function slider(parent, label, min, max, step, value, fmt, onInput) {
    const lab = el("label", null, parent);
    el("span", null, lab, label);
    const inp = el("input", { type: "range", min: min, max: max, step: step, value: value }, lab);
    const out = el("span", null, lab);
    const f = fmt || ((v) => v);
    const update = () => { out.textContent = f(parseFloat(inp.value)); };
    inp.addEventListener("input", () => { update(); onInput(); });
    update();
    return { get: () => parseFloat(inp.value), set: (v) => { inp.value = v; update(); }, input: inp };
  }
  function checkbox(parent, label, checked, onChange) {
    const lab = el("label", null, parent);
    const inp = el("input", { type: "checkbox" }, lab);
    inp.checked = !!checked;
    el("span", null, lab, label);
    inp.addEventListener("change", onChange);
    return inp;
  }
  function select(parent, label, options, onChange) {
    const lab = el("label", null, parent);
    el("span", null, lab, label);
    const s = el("select", null, lab);
    options.forEach((o) => { const op = el("option", { value: o[0] }, s, o[1]); });
    s.addEventListener("change", onChange);
    return s;
  }
  function button(parent, label, onClick) {
    const b = el("button", { type: "button" }, parent, label);
    b.addEventListener("click", onClick);
    return b;
  }
  function makeCanvas(parent, w, h) {
    const c = el("canvas", null, parent);
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.aspectRatio = w + " / " + h;
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c: c, ctx: ctx, w: w, h: h };
  }
  function clear(cv) { cv.ctx.clearRect(0, 0, cv.w, cv.h); }
  function softmax(z, T) {
    const t = T === undefined ? 1 : T;
    const m = Math.max.apply(null, z.map((v) => v / t));
    const e = z.map((v) => Math.exp(v / t - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  function entropy(p) { return -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0); }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function erf(x) { // Abramowitz & Stegun 7.1.26
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function fmtNum(x) {
    if (!isFinite(x)) return "∞";
    const a = Math.abs(x);
    if (a >= 1e12) return (x / 1e12).toFixed(2) + "T";
    if (a >= 1e9) return (x / 1e9).toFixed(2) + "B";
    if (a >= 1e6) return (x / 1e6).toFixed(2) + "M";
    if (a >= 1e3) return (x / 1e3).toFixed(2) + "K";
    return x.toFixed(2);
  }
  function fmtBytes(b) {
    const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return b.toFixed(2) + " " + u[i];
  }
  function bars(cv, values, labels, opts) {
    const o = opts || {};
    const ctx = cv.ctx, W = cv.w, H = cv.h, n = values.length;
    const col = o.color || PAL[0];
    const textCol = o.fg;
    const pad = 28, top = 14, bottom = 34;
    const maxV = o.max !== undefined ? o.max : Math.max.apply(null, values.concat([1e-9]));
    const bw = (W - 2 * pad) / n;
    ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    for (let i = 0; i < n; i++) {
      const h = (H - top - bottom) * (values[i] / maxV);
      const x = pad + i * bw + bw * 0.12, y = H - bottom - h;
      ctx.fillStyle = Array.isArray(col) ? col[i] : col;
      ctx.fillRect(x, y, bw * 0.76, Math.max(h, 0));
      ctx.fillStyle = textCol;
      if (o.showValues !== false) ctx.fillText(values[i].toFixed(3), x + bw * 0.38, y - 3);
      if (labels) ctx.fillText(labels[i], x + bw * 0.38, H - bottom + 14);
    }
    ctx.strokeStyle = alpha(textCol, 0.4);
    ctx.beginPath(); ctx.moveTo(pad, H - bottom); ctx.lineTo(W - pad, H - bottom); ctx.stroke();
  }
  function axes(cv, x0, x1, y0, y1, fgc, xlabel, ylabel) {
    const ctx = cv.ctx;
    const L = 44, R = 12, T = 12, B = 30;
    const X = (x) => L + (x - x0) / (x1 - x0) * (cv.w - L - R);
    const Y = (y) => cv.h - B - (y - y0) / (y1 - y0) * (cv.h - T - B);
    ctx.strokeStyle = alpha(fgc, 0.35); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, cv.h - B); ctx.lineTo(cv.w - R, cv.h - B); ctx.stroke();
    ctx.fillStyle = fgc; ctx.font = "11px sans-serif";
    if (xlabel) { ctx.textAlign = "center"; ctx.fillText(xlabel, (L + cv.w - R) / 2, cv.h - 6); }
    if (ylabel) { ctx.save(); ctx.translate(11, (T + cv.h - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(ylabel, 0, 0); ctx.restore(); }
    return { X: X, Y: Y, L: L, R: R, T: T, B: B };
  }

  const W = {};

  // ---------- 1. Linear transformation ----------
  W["linear-transform"] = function (root) {
    title(root, "A 2×2 matrix as a transformation of the plane");
    const cc = controls(root);
    const redraw = () => draw();
    const a = slider(cc, "a", -2, 2, 0.05, 1, (v) => v.toFixed(2), redraw);
    const b = slider(cc, "b", -2, 2, 0.05, 0.5, (v) => v.toFixed(2), redraw);
    const c = slider(cc, "c", -2, 2, 0.05, 0, (v) => v.toFixed(2), redraw);
    const d = slider(cc, "d", -2, 2, 0.05, 1, (v) => v.toFixed(2), redraw);
    const pc = controls(root);
    const presets = { Identity: [1, 0, 0, 1], "Rotate 45°": [0.71, -0.71, 0.71, 0.71], Shear: [1, 1, 0, 1], Scale: [1.5, 0, 0, 0.5], Reflect: [1, 0, 0, -1], "Project (det 0)": [1, 0.5, 0.5, 0.25] };
    Object.keys(presets).forEach((k) => button(pc, k, () => { const p = presets[k]; a.set(p[0]); b.set(p[1]); c.set(p[2]); d.set(p[3]); draw(); }));
    const cv = makeCanvas(root, 520, 400);
    const out = el("div", { class: "w-out" }, root);
    function draw() {
      const A = a.get(), B = b.get(), C = c.get(), D = d.get();
      const ctx = cv.ctx, s = 40, cx = cv.w / 2, cy = cv.h / 2, f = fg(root);
      clear(cv);
      const P = (x, y) => [cx + (A * x + B * y) * s, cy - (C * x + D * y) * s];
      // original grid
      ctx.strokeStyle = alpha(f, 0.08); ctx.lineWidth = 1;
      for (let i = -8; i <= 8; i++) {
        ctx.beginPath(); ctx.moveTo(cx + i * s, 0); ctx.lineTo(cx + i * s, cv.h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, cy - i * s); ctx.lineTo(cv.w, cy - i * s); ctx.stroke();
      }
      // transformed grid
      ctx.strokeStyle = alpha(PAL[0], 0.35);
      for (let i = -8; i <= 8; i++) {
        let p = P(i, -8), q = P(i, 8); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
        p = P(-8, i); q = P(8, i); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      }
      // unit square image
      const det = A * D - B * C;
      ctx.fillStyle = det >= 0 ? alpha(PAL[4], 0.28) : alpha(PAL[6], 0.28);
      ctx.beginPath();
      [[0, 0], [1, 0], [1, 1], [0, 1]].forEach((v, i) => { const p = P(v[0], v[1]); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
      ctx.closePath(); ctx.fill();
      // eigenvectors
      const tr = A + D, disc = tr * tr / 4 - det;
      let eigTxt;
      if (disc >= 0) {
        const l1 = tr / 2 + Math.sqrt(disc), l2 = tr / 2 - Math.sqrt(disc);
        eigTxt = "eigenvalues λ₁ = " + l1.toFixed(3) + ", λ₂ = " + l2.toFixed(3) + "  (dashed lines = eigenvector directions: vectors there only get scaled)";
        [l1, l2].forEach((l) => {
          let vx, vy;
          if (Math.abs(B) > 1e-9) { vx = B; vy = l - A; } else if (Math.abs(C) > 1e-9) { vx = l - D; vy = C; } else { vx = (Math.abs(l - A) < 1e-9) ? 1 : 0; vy = 1 - vx; }
          const n = Math.hypot(vx, vy) || 1; vx /= n; vy /= n;
          ctx.setLineDash([6, 5]); ctx.strokeStyle = alpha(f, 0.6);
          ctx.beginPath(); ctx.moveTo(cx - vx * 600, cy + vy * 600); ctx.lineTo(cx + vx * 600, cy - vy * 600); ctx.stroke();
          ctx.setLineDash([]);
        });
      } else {
        const im = Math.sqrt(-disc);
        eigTxt = "eigenvalues λ = " + (tr / 2).toFixed(3) + " ± " + im.toFixed(3) + "i  (complex: the map rotates, so no real direction is preserved)";
      }
      function arrow(x, y, col, lab) {
        const p = P(x, y);
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p[0], p[1]); ctx.stroke();
        const ang = Math.atan2(p[1] - cy, p[0] - cx);
        ctx.beginPath(); ctx.moveTo(p[0], p[1]);
        ctx.lineTo(p[0] - 10 * Math.cos(ang - 0.4), p[1] - 10 * Math.sin(ang - 0.4));
        ctx.lineTo(p[0] - 10 * Math.cos(ang + 0.4), p[1] - 10 * Math.sin(ang + 0.4)); ctx.fill();
        ctx.font = "bold 13px sans-serif"; ctx.fillText(lab, p[0] + 6, p[1] - 6); ctx.lineWidth = 1;
      }
      arrow(1, 0, PAL[1], "W·e₁ = (a, c)");
      arrow(0, 1, PAL[2], "W·e₂ = (b, d)");
      out.textContent = "W = [[" + A.toFixed(2) + ", " + B.toFixed(2) + "], [" + C.toFixed(2) + ", " + D.toFixed(2) + "]]\n" +
        "det(W) = ad − bc = " + det.toFixed(3) + "  → the unit square's area is scaled by |det| = " + Math.abs(det).toFixed(3) + (det < 0 ? " and orientation flips" : "") + (Math.abs(det) < 1e-6 ? " (the plane collapses to a line: W is not invertible)" : "") + "\n" + eigTxt;
    }
    redrawers.push(draw); draw();
  };

  // ---------- 2. Gradient descent / momentum / Adam ----------
  W["gradient-descent"] = function (root) {
    title(root, "SGD vs Momentum vs Adam on a loss surface");
    const cc = controls(root);
    const fsel = select(cc, "surface", [["bowl", "Ill-conditioned bowl  ½(x² + κy²)"], ["rosen", "Rosenbrock banana"]], () => { reset(); });
    const kap = slider(cc, "κ", 1, 50, 1, 10, (v) => v.toFixed(0), () => reset());
    const lr = slider(cc, "η (SGD, momentum)", -4, 0, 0.05, -1.05, (v) => Math.pow(10, v).toExponential(1), () => reset());
    const lra = slider(cc, "η (Adam)", -3, 0, 0.05, -1, (v) => Math.pow(10, v).toExponential(1), () => reset());
    const steps = slider(cc, "steps", 10, 500, 10, 100, (v) => v.toFixed(0), () => reset());
    const bc = controls(root);
    button(bc, "▶ Run", () => run());
    const cv = makeCanvas(root, 640, 380);
    const out = el("div", { class: "w-out" }, root);
    let paths = [], frame = 0, anim = null;
    function F() {
      const k = kap.get();
      if (fsel.value === "bowl") return {
        f: (x, y) => 0.5 * (x * x + k * y * y), g: (x, y) => [x, k * y],
        x0: -3.5, x1: 3.5, y0: -2.1, y1: 2.1, start: [-3.2, 1.6], opt: [0, 0]
      };
      return {
        f: (x, y) => (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
        g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
        x0: -2, x1: 2, y0: -1, y1: 3, start: [-1.5, 2.5], opt: [1, 1]
      };
    }
    function simulate() {
      const P = F(), n = steps.get(), eta = Math.pow(10, lr.get()), etaA = Math.pow(10, lra.get());
      const res = [];
      // SGD
      let x = P.start[0], y = P.start[1], pts = [[x, y]];
      for (let i = 0; i < n; i++) { const g = P.g(x, y); x -= eta * g[0]; y -= eta * g[1]; pts.push([x, y]); if (!isFinite(x + y) || Math.abs(x) + Math.abs(y) > 1e4) break; }
      res.push({ name: "SGD", pts: pts, col: PAL[1] });
      // Momentum (heavy ball)
      x = P.start[0]; y = P.start[1]; pts = [[x, y]]; let vx = 0, vy = 0; const beta = 0.9;
      for (let i = 0; i < n; i++) { const g = P.g(x, y); vx = beta * vx + g[0]; vy = beta * vy + g[1]; x -= eta * vx; y -= eta * vy; pts.push([x, y]); if (!isFinite(x + y) || Math.abs(x) + Math.abs(y) > 1e4) break; }
      res.push({ name: "Momentum β=0.9", pts: pts, col: PAL[2] });
      // Adam
      x = P.start[0]; y = P.start[1]; pts = [[x, y]];
      let mx = 0, my = 0, sx = 0, sy = 0; const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      for (let i = 1; i <= n; i++) {
        const g = P.g(x, y);
        mx = b1 * mx + (1 - b1) * g[0]; my = b1 * my + (1 - b1) * g[1];
        sx = b2 * sx + (1 - b2) * g[0] * g[0]; sy = b2 * sy + (1 - b2) * g[1] * g[1];
        const mhx = mx / (1 - Math.pow(b1, i)), mhy = my / (1 - Math.pow(b1, i));
        const shx = sx / (1 - Math.pow(b2, i)), shy = sy / (1 - Math.pow(b2, i));
        x -= etaA * mhx / (Math.sqrt(shx) + eps); y -= etaA * mhy / (Math.sqrt(shy) + eps);
        pts.push([x, y]);
      }
      res.push({ name: "Adam", pts: pts, col: PAL[3] });
      return res;
    }
    let bg = null;
    function background() {
      const P = F(), ctx = cv.ctx, nx = 160, ny = 95;
      const cw = cv.w / nx, ch = cv.h / ny;
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
        const x = P.x0 + (i + 0.5) / nx * (P.x1 - P.x0), y = P.y1 - (j + 0.5) / ny * (P.y1 - P.y0);
        const v = Math.log(1e-3 + P.f(x, y));
        const band = Math.floor(v * 2.2);
        const shade = 0.06 + 0.10 * ((band % 2 + 2) % 2) + 0.02 * Math.max(0, Math.min(8, v));
        ctx.fillStyle = alpha(PAL[0], Math.min(shade, 0.45));
        ctx.fillRect(i * cw, j * ch, cw + 0.5, ch + 0.5);
      }
    }
    function drawPaths(k) {
      const P = F(), ctx = cv.ctx;
      const X = (x) => (x - P.x0) / (P.x1 - P.x0) * cv.w, Y = (y) => (P.y1 - y) / (P.y1 - P.y0) * cv.h;
      clear(cv); background();
      ctx.fillStyle = fg(root); ctx.beginPath(); ctx.arc(X(P.opt[0]), Y(P.opt[1]), 5, 0, 7); ctx.fill();
      ctx.font = "11px sans-serif"; ctx.fillText("minimum", X(P.opt[0]) + 7, Y(P.opt[1]) - 6);
      paths.forEach((p, idx) => {
        ctx.strokeStyle = p.col; ctx.lineWidth = 2; ctx.beginPath();
        const m = Math.min(k, p.pts.length - 1);
        for (let i = 0; i <= m; i++) { const q = p.pts[i]; i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1])); }
        ctx.stroke();
        const q = p.pts[m]; ctx.fillStyle = p.col; ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 4, 0, 7); ctx.fill();
        ctx.fillRect(10, 10 + idx * 16, 12, 10); ctx.fillStyle = fg(root); ctx.fillText(p.name, 28, 19 + idx * 16);
      });
      const fx = F().f;
      out.textContent = "step " + Math.min(k, steps.get()) + "   " + paths.map((p) => {
        const q = p.pts[Math.min(k, p.pts.length - 1)]; const v = fx(q[0], q[1]);
        return p.name + ": loss = " + (isFinite(v) ? v.toExponential(2) : "diverged");
      }).join("   |   ");
    }
    function reset() { if (anim) cancelAnimationFrame(anim); paths = simulate(); frame = 0; drawPaths(0); }
    function run() {
      if (anim) cancelAnimationFrame(anim);
      paths = simulate(); frame = 0;
      const n = steps.get(), per = Math.max(1, Math.round(n / 120));
      const tick = () => { frame += per; drawPaths(frame); if (frame < n) anim = requestAnimationFrame(tick); };
      tick();
    }
    redrawers.push(() => drawPaths(frame)); reset();
  };

  // ---------- 3. Softmax with temperature ----------
  W["softmax"] = function (root) {
    title(root, "Softmax and temperature");
    const cc = controls(root);
    const lab = el("label", null, cc); el("span", null, lab, "logits z =");
    const inp = el("input", { type: "text", value: "3.0, 2.5, 1.8, 1.5, 0.9, 0.4, 0.1, -1.0" }, lab);
    const cc2 = controls(root);
    const T = slider(cc2, "temperature T", 0.05, 4, 0.05, 1, (v) => v.toFixed(2), () => draw());
    inp.addEventListener("input", () => draw());
    const cv = makeCanvas(root, 640, 230);
    const out = el("div", { class: "w-out" }, root);
    function draw() {
      const z = inp.value.split(",").map((s) => parseFloat(s)).filter((v) => isFinite(v));
      if (!z.length) return;
      const p = softmax(z, T.get());
      clear(cv);
      bars(cv, p, z.map((v, i) => "z" + (i + 1) + "=" + v), { fg: fg(root), max: 1, color: PAL[0] });
      const H = entropy(p);
      out.textContent = "p_i = exp(z_i/T) / Σ_j exp(z_j/T)     Σp = " + p.reduce((a, b) => a + b, 0).toFixed(3) +
        "\nentropy H(p) = " + H.toFixed(3) + " nats = " + (H / Math.LN2).toFixed(3) + " bits   (max possible = log " + z.length + " = " + Math.log(z.length).toFixed(3) + " nats)" +
        "\nT → 0: one-hot on the argmax (greedy).  T = 1: the model's distribution.  T → ∞: uniform.";
    }
    redrawers.push(draw); draw();
  };

  // ---------- 4. Activation functions ----------
  W["activations"] = function (root) {
    title(root, "Activation functions and their derivatives");
    const cc = controls(root);
    const fns = {
      ReLU: [(x) => Math.max(0, x), (x) => (x > 0 ? 1 : 0)],
      "GELU": [(x) => 0.5 * x * (1 + erf(x / Math.SQRT2)), (x) => 0.5 * (1 + erf(x / Math.SQRT2)) + x * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI)],
      "SiLU / Swish": [(x) => x / (1 + Math.exp(-x)), (x) => { const s = 1 / (1 + Math.exp(-x)); return s + x * s * (1 - s); }],
      sigmoid: [(x) => 1 / (1 + Math.exp(-x)), (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }],
      tanh: [(x) => Math.tanh(x), (x) => 1 - Math.tanh(x) * Math.tanh(x)]
    };
    const boxes = {};
    Object.keys(fns).forEach((k, i) => { boxes[k] = checkbox(cc, k, i < 3, () => draw()); });
    const der = checkbox(cc, "show derivative (dashed)", false, () => draw());
    const cv = makeCanvas(root, 640, 300);
    function draw() {
      clear(cv);
      const f = fg(root), ax = axes(cv, -4, 4, -1.5, 4, f, "x", "f(x)"), ctx = cv.ctx;
      ctx.strokeStyle = alpha(f, 0.2); ctx.beginPath(); ctx.moveTo(ax.X(-4), ax.Y(0)); ctx.lineTo(ax.X(4), ax.Y(0)); ctx.moveTo(ax.X(0), ax.Y(-1.5)); ctx.lineTo(ax.X(0), ax.Y(4)); ctx.stroke();
      let li = 0;
      Object.keys(fns).forEach((k, i) => {
        if (!boxes[k].checked) return;
        const col = PAL[i];
        [0, 1].forEach((which) => {
          if (which === 1 && !der.checked) return;
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(which ? [5, 4] : []);
          ctx.beginPath();
          for (let s = 0; s <= 400; s++) { const x = -4 + 8 * s / 400, y = fns[k][which](x); s ? ctx.lineTo(ax.X(x), ax.Y(y)) : ctx.moveTo(ax.X(x), ax.Y(y)); }
          ctx.stroke(); ctx.setLineDash([]);
        });
        ctx.fillStyle = col; ctx.fillRect(ax.L + 10, 16 + li * 16, 12, 10); ctx.fillStyle = f; ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.fillText(k, ax.L + 28, 25 + li * 16); li++;
      });
    }
    redrawers.push(draw); draw();
  };

  // ---------- 5. Attention patterns ----------
  W["attention"] = function (root) {
    title(root, "Attention weights: patterns, causal mask, and sharpness");
    const toks = ["<bos>", "The", "cat", "sat", "on", "the", "mat", ".", "The", "cat", "slept"];
    const n = toks.length;
    const cc = controls(root);
    const head = select(cc, "head type", [["prev", "previous-token head"], ["same", "same-token (copy) head"], ["induction", "induction head"], ["sink", "attention-sink head"], ["random", "random scores"]], () => draw());
    const causal = checkbox(cc, "causal mask", true, () => draw());
    const sharp = slider(cc, "score scale", 0.2, 5, 0.1, 2.5, (v) => v.toFixed(1), () => draw());
    const cv = makeCanvas(root, 600, 470);
    const out = el("div", { class: "w-out" }, root, "Rows = queries (the token doing the looking). Columns = keys (tokens being looked at). Each row sums to 1. Hover a row.");
    let hoverRow = -1;
    function scores() {
      const rng = mulberry32(7), S = [];
      for (let i = 0; i < n; i++) {
        S.push([]);
        for (let j = 0; j < n; j++) {
          let s = 0.3 * gaussian(rng);
          const ty = head.value;
          if (ty === "prev") s += (j === i - 1) ? 3 : 0;
          if (ty === "same") s += (j !== i && toks[j].toLowerCase() === toks[i].toLowerCase()) ? 3 : 0;
          if (ty === "induction") s += (j >= 1 && j < i && toks[j - 1].toLowerCase() === toks[i].toLowerCase()) ? 3.2 : 0;
          if (ty === "sink") s += (j === 0) ? 3 : 0;
          if (ty === "random") s = 1.2 * gaussian(rng);
          S[i].push(s);
        }
      }
      return S;
    }
    function draw() {
      const S = scores(), ctx = cv.ctx, f = fg(root), k = sharp.get();
      clear(cv);
      const L = 70, Tp = 60, cell = Math.min((cv.w - L - 10) / n, (cv.h - Tp - 10) / n);
      ctx.font = "11px sans-serif"; ctx.fillStyle = f;
      const A = S.map((row, i) => {
        const z = row.map((v, j) => (causal.checked && j > i) ? -Infinity : v * k / 2.5);
        const m = Math.max.apply(null, z.filter((v) => v > -Infinity));
        const e = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - m)));
        const s = e.reduce((a, b) => a + b, 0);
        return e.map((v) => v / s);
      });
      for (let j = 0; j < n; j++) { ctx.save(); ctx.translate(L + j * cell + cell / 2, Tp - 6); ctx.rotate(-Math.PI / 3); ctx.textAlign = "left"; ctx.fillText(toks[j], 0, 0); ctx.restore(); }
      for (let i = 0; i < n; i++) {
        ctx.textAlign = "right"; ctx.fillStyle = i === hoverRow ? PAL[1] : f; ctx.fillText(toks[i], L - 6, Tp + i * cell + cell * 0.65);
        for (let j = 0; j < n; j++) {
          const v = A[i][j];
          ctx.fillStyle = (causal.checked && j > i) ? alpha(f, 0.05) : alpha(PAL[0], 0.05 + 0.95 * v);
          ctx.fillRect(L + j * cell + 1, Tp + i * cell + 1, cell - 2, cell - 2);
          if (i === hoverRow) { ctx.fillStyle = v > 0.5 ? "#fff" : f; ctx.textAlign = "center"; ctx.fillText(v.toFixed(2), L + j * cell + cell / 2, Tp + i * cell + cell * 0.62); }
        }
      }
      if (hoverRow >= 0) {
        ctx.strokeStyle = PAL[1]; ctx.lineWidth = 2; ctx.strokeRect(L, Tp + hoverRow * cell, n * cell, cell); ctx.lineWidth = 1;
        const best = A[hoverRow].indexOf(Math.max.apply(null, A[hoverRow]));
        out.textContent = "Query \"" + toks[hoverRow] + "\" (position " + hoverRow + ") puts most weight on \"" + toks[best] + "\" (position " + best + "), weight " + A[hoverRow][best].toFixed(2) + ". Output = Σ_j a_ij · v_j.";
      }
      cv.geom = { L: L, Tp: Tp, cell: cell };
    }
    cv.c.addEventListener("mousemove", (e) => {
      const r = cv.c.getBoundingClientRect(), y = (e.clientY - r.top) / r.height * cv.h;
      const g = cv.geom; if (!g) return;
      const i = Math.floor((y - g.Tp) / g.cell);
      const nr = (i >= 0 && i < n) ? i : -1;
      if (nr !== hoverRow) { hoverRow = nr; draw(); }
    });
    redrawers.push(draw); draw();
  };

  // ---------- 6. Why scale by 1/sqrt(d) ----------
  W["attention-scaling"] = function (root) {
    title(root, "Why attention divides by √d: random q·k grows with dimension");
    const cc = controls(root);
    const d = slider(cc, "head dim d", 1, 10, 1, 6, (v) => String(Math.pow(2, v)), () => draw());
    const sc = checkbox(cc, "scale scores by 1/√d", false, () => draw());
    button(cc, "resample", () => { seed++; draw(); });
    let seed = 1;
    const cv = makeCanvas(root, 640, 220);
    const out = el("div", { class: "w-out" }, root);
    function draw() {
      const dim = Math.pow(2, d.get()), rng = mulberry32(seed * 101), nk = 16;
      const q = []; for (let i = 0; i < dim; i++) q.push(gaussian(rng));
      const s = [];
      for (let j = 0; j < nk; j++) { let dot = 0; for (let i = 0; i < dim; i++) dot += q[i] * gaussian(rng); s.push(sc.checked ? dot / Math.sqrt(dim) : dot); }
      const p = softmax(s);
      clear(cv);
      bars(cv, p, s.map((_, j) => "k" + (j + 1)), { fg: fg(root), max: 1, color: PAL[3], showValues: false });
      const mean = s.reduce((a, b) => a + b, 0) / nk, sd = Math.sqrt(s.reduce((a, b) => a + (b - mean) * (b - mean), 0) / nk);
      out.textContent = "q, k ~ N(0, I_d).  Var(q·k) = d = " + dim + ", so std ≈ √d = " + Math.sqrt(dim).toFixed(1) + ".  Observed std of scores: " + sd.toFixed(2) +
        "\nmax attention weight = " + Math.max.apply(null, p).toFixed(3) + ",  entropy = " + entropy(p).toFixed(3) + " nats (uniform = " + Math.log(nk).toFixed(3) + ")" +
        (sc.checked ? "\nWith 1/√d scaling the score variance stays ≈ 1 for any d, so softmax stays soft and gradients flow." : "\nWithout scaling, large d → huge scores → softmax saturates to one-hot → vanishing gradients.");
    }
    redrawers.push(draw); draw();
  };

  // ---------- 7. RoPE ----------
  W["rope"] = function (root) {
    title(root, "Rotary position embedding (RoPE): rotate each 2-D pair by m·θᵢ");
    const cc = controls(root);
    const m = slider(cc, "query position m", 0, 64, 1, 5, (v) => v.toFixed(0), () => draw());
    const nn = slider(cc, "key position n", 0, 64, 1, 2, (v) => v.toFixed(0), () => draw());
    const sh = slider(cc, "shift both by s", 0, 200, 1, 0, (v) => v.toFixed(0), () => draw());
    const cv = makeCanvas(root, 680, 380);
    const out = el("div", { class: "w-out" }, root);
    const dim = 64, base = 10000;
    const rng = mulberry32(42), q = [], k = [];
    for (let i = 0; i < dim; i++) { q.push(gaussian(rng)); k.push(gaussian(rng)); }
    const theta = []; for (let i = 0; i < dim / 2; i++) theta.push(Math.pow(base, -2 * i / dim));
    function rot(v, pos) {
      const r = v.slice();
      for (let i = 0; i < dim / 2; i++) {
        const c = Math.cos(pos * theta[i]), s = Math.sin(pos * theta[i]);
        r[2 * i] = v[2 * i] * c - v[2 * i + 1] * s; r[2 * i + 1] = v[2 * i] * s + v[2 * i + 1] * c;
      }
      return r;
    }
    function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
    function draw() {
      const ctx = cv.ctx, f = fg(root), M = m.get() + sh.get(), N = nn.get() + sh.get();
      clear(cv);
      const pairs = [0, 4, 10, 20], R = 52;
      ctx.font = "11px sans-serif";
      pairs.forEach((pi, idx) => {
        const cx = 85 + idx * 170, cy = 80;
        ctx.strokeStyle = alpha(f, 0.25); ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
        const draws = [[q, M, PAL[1], "q@m"], [k, N, PAL[2], "k@n"]];
        draws.forEach((dd) => {
          const v = rot(dd[0], dd[1]), x = v[2 * pi], y = v[2 * pi + 1], nrm = Math.hypot(x, y) || 1;
          ctx.strokeStyle = dd[2]; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * x / nrm, cy - R * y / nrm); ctx.stroke(); ctx.lineWidth = 1;
        });
        ctx.fillStyle = f; ctx.textAlign = "center";
        ctx.fillText("pair i=" + pi + "  θᵢ=" + theta[pi].toExponential(1), cx, cy + R + 16);
        ctx.fillText("angle gap (m−n)θᵢ = " + (((M - N) * theta[pi]) % (2 * Math.PI)).toFixed(2) + " rad", cx, cy + R + 30);
      });
      // relative-distance curve
      const top = 190, H = 170, L = 50, Wd = cv.w - 70;
      const xs = []; let mx = 0;
      for (let r = -64; r <= 64; r++) { const v = dot(rot(q, r), k); xs.push(v); mx = Math.max(mx, Math.abs(v)); }
      ctx.strokeStyle = alpha(f, 0.3); ctx.beginPath(); ctx.moveTo(L, top + H / 2); ctx.lineTo(L + Wd, top + H / 2); ctx.moveTo(L + Wd / 2, top); ctx.lineTo(L + Wd / 2, top + H); ctx.stroke();
      ctx.strokeStyle = PAL[0]; ctx.lineWidth = 2; ctx.beginPath();
      xs.forEach((v, i) => { const X = L + i / 128 * Wd, Y = top + H / 2 - v / mx * H / 2 * 0.95; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke(); ctx.lineWidth = 1;
      const rel = Math.max(-64, Math.min(64, M - N));
      const X = L + (rel + 64) / 128 * Wd, Y = top + H / 2 - dot(rot(q, rel), k) / mx * H / 2 * 0.95;
      ctx.fillStyle = PAL[1]; ctx.beginPath(); ctx.arc(X, Y, 5, 0, 7); ctx.fill();
      ctx.fillStyle = f; ctx.textAlign = "center"; ctx.fillText("score q_mᵀ k_n as a function of relative offset m − n  (−64 … 64)", L + Wd / 2, top + H + 14);
      const s1 = dot(rot(q, M), rot(k, N)), s0 = dot(rot(q, m.get()), rot(k, nn.get()));
      out.textContent = "score(m=" + M + ", n=" + N + ") = " + s1.toFixed(4) + "    score without shift (m=" + m.get() + ", n=" + nn.get() + ") = " + s0.toFixed(4) +
        "\nThe two are equal: RoPE scores depend only on m − n, never on absolute position. Low pairs (i=0) rotate fast; high pairs rotate slowly.";
    }
    redrawers.push(draw); draw();
  };

  // ---------- 8. Sampling: temperature, top-k, top-p ----------
  W["sampling"] = function (root) {
    title(root, "Decoding: temperature → top-k → top-p (nucleus)");
    const toks = ["Paris", "the", "a", "London", "France", "located", "known", "one", "home", "not", "Rome", "big"];
    const logits = [5.0, 3.2, 2.9, 2.2, 2.0, 1.6, 1.2, 0.9, 0.6, 0.3, 0.1, -0.5];
    const cc = controls(root);
    const T = slider(cc, "temperature", 0.1, 3, 0.05, 1, (v) => v.toFixed(2), () => draw());
    const K = slider(cc, "top-k", 1, 12, 1, 12, (v) => v.toFixed(0), () => draw());
    const Pp = slider(cc, "top-p", 0.05, 1, 0.01, 1, (v) => v.toFixed(2), () => draw());
    const bc = controls(root);
    let rng = mulberry32(3), lastPick = "";
    button(bc, "sample 1 token", () => { sample(1); });
    button(bc, "sample 1000", () => { sample(1000); });
    const cv = makeCanvas(root, 680, 240);
    const out = el("div", { class: "w-out" }, root);
    let cur = null;
    function compute() {
      const p = softmax(logits, T.get());
      const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
      const keep = new Array(p.length).fill(false);
      let cum = 0;
      for (let r = 0; r < order.length; r++) {
        const i = order[r];
        if (r >= K.get()) break;
        keep[i] = true; cum += p[i];
        if (cum >= Pp.get()) break;
      }
      const z = keep.reduce((a, kk, i) => a + (kk ? p[i] : 0), 0);
      const q = p.map((v, i) => (keep[i] ? v / z : 0));
      return { p: p, q: q, keep: keep };
    }
    function sample(times) {
      const c = compute(), counts = new Array(toks.length).fill(0);
      for (let t = 0; t < times; t++) { let r = rng(), i = 0; while (i < c.q.length - 1 && r >= c.q[i]) { r -= c.q[i]; i++; } counts[i]++; }
      lastPick = times === 1 ? "sampled: \"" + toks[counts.indexOf(1)] + "\"" : "1000 samples: " + toks.map((t, i) => [t, counts[i]]).filter((x) => x[1] > 0).map((x) => x[0] + "×" + x[1]).join(", ");
      draw();
    }
    function draw() {
      cur = compute(); clear(cv);
      bars(cv, cur.q, toks, { fg: fg(root), max: 1, color: cur.keep.map((k) => (k ? PAL[0] : alpha(fg(root), 0.15))) });
      const kept = cur.keep.filter((x) => x).length;
      out.textContent = "Prompt: \"The capital of France is …\"   kept " + kept + " of " + toks.length + " tokens (grey = filtered, bars show renormalised probabilities)\n" + lastPick;
    }
    redrawers.push(draw); draw();
  };

  // ---------- 9. BPE ----------
  W["bpe"] = function (root) {
    title(root, "Byte-Pair Encoding, one merge at a time");
    const cc = controls(root);
    const lab = el("label", null, cc); el("span", null, lab, "training text");
    const inp = el("input", { type: "text", value: "low lower lowest newer newest wider widest low low newest" }, lab);
    const bc = controls(root);
    button(bc, "merge once", () => { step(1); });
    button(bc, "merge ×10", () => { step(10); });
    button(bc, "reset", () => { reset(); });
    const view = el("div", { class: "w-tokens" }, root);
    const out = el("div", { class: "w-out" }, root);
    let words = [], merges = [];
    function reset() {
      const pieces = inp.value.match(/\s*\S+/g) || [];
      words = pieces.map((w) => Array.from(w.replace(/ /g, "␣")));
      merges = []; render();
    }
    function step(times) {
      for (let t = 0; t < times; t++) {
        const counts = new Map();
        words.forEach((w) => { for (let i = 0; i < w.length - 1; i++) { const key = w[i] + "\u0000" + w[i + 1]; counts.set(key, (counts.get(key) || 0) + 1); } });
        if (!counts.size) break;
        let best = null, bc2 = 0;
        counts.forEach((v, k) => { if (v > bc2) { bc2 = v; best = k; } });
        if (bc2 < 2) break;
        const pr = best.split("\u0000"), merged = pr[0] + pr[1];
        merges.push(pr[0] + " + " + pr[1] + " → " + merged + "  (count " + bc2 + ")");
        words = words.map((w) => { const r = []; let i = 0; while (i < w.length) { if (i < w.length - 1 && w[i] === pr[0] && w[i + 1] === pr[1]) { r.push(merged); i += 2; } else { r.push(w[i]); i++; } } return r; });
      }
      render();
    }
    function render() {
      view.innerHTML = "";
      const vocab = new Set(); let ntok = 0;
      words.forEach((w) => {
        w.forEach((t) => { vocab.add(t); ntok++; const s = el("span", null, view, t); let h = 0; for (const ch of t) h = (h * 31 + ch.charCodeAt(0)) % 360; s.style.background = "hsla(" + h + ",70%,60%,0.35)"; });
        el("span", { style: "background:transparent" }, view, " ");
      });
      out.textContent = "tokens: " + ntok + "   distinct symbols: " + vocab.size + "   merges learned: " + merges.length + "\n" + merges.map((m, i) => (i + 1) + ". " + m).join("\n");
    }
    inp.addEventListener("change", reset);
    reset();
  };

  // ---------- 10. Learning-rate schedule ----------
  W["lr-schedule"] = function (root) {
    title(root, "Learning-rate schedules: warmup + cosine / linear / WSD");
    const cc = controls(root);
    const typ = select(cc, "decay", [["cos", "cosine"], ["lin", "linear"], ["wsd", "warmup-stable-decay"]], () => draw());
    const wu = slider(cc, "warmup %", 0, 30, 1, 5, (v) => v + "%", () => draw());
    const mn = slider(cc, "min lr / peak", 0, 0.5, 0.01, 0.1, (v) => v.toFixed(2), () => draw());
    const dec = slider(cc, "WSD decay %", 5, 50, 1, 20, (v) => v + "%", () => draw());
    const cv = makeCanvas(root, 640, 240);
    function lrAt(t) { // t in [0,1]
      const w = wu.get() / 100, r = mn.get();
      if (t < w) return t / w;
      if (typ.value === "cos") { const u = (t - w) / (1 - w); return r + (1 - r) * 0.5 * (1 + Math.cos(Math.PI * u)); }
      if (typ.value === "lin") { const u = (t - w) / (1 - w); return 1 - (1 - r) * u; }
      const ds = 1 - dec.get() / 100;
      if (t < ds) return 1;
      return 1 - (1 - r) * (t - ds) / (1 - ds);
    }
    function draw() {
      clear(cv); const f = fg(root), ax = axes(cv, 0, 1, 0, 1.05, f, "training progress (fraction of total steps)", "lr / peak lr"), ctx = cv.ctx;
      ctx.strokeStyle = PAL[1]; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 500; i++) { const t = i / 500; i ? ctx.lineTo(ax.X(t), ax.Y(lrAt(t))) : ctx.moveTo(ax.X(t), ax.Y(lrAt(t))); }
      ctx.stroke(); ctx.lineWidth = 1;
    }
    redrawers.push(draw); draw();
  };

  // ---------- 11. Chinchilla scaling ----------
  W["chinchilla"] = function (root) {
    title(root, "Compute-optimal model size (Chinchilla): L(N, D) = E + A/N^α + B/D^β, C ≈ 6ND");
    const cc = controls(root);
    const fit = select(cc, "fit", [["epoch", "Besiroglu et al. 2024 (replication)"], ["hoff", "Hoffmann et al. 2022 (Approach 3)"]], () => draw());
    const C = slider(cc, "compute C (FLOPs) = 10^", 18, 27, 0.1, 23.5, (v) => v.toFixed(1), () => draw());
    const cv = makeCanvas(root, 640, 260);
    const out = el("div", { class: "w-out" }, root);
    function params() {
      if (fit.value === "hoff") return { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 };
      return { E: 1.8172, A: 482.01, B: 2085.43, a: 0.3478, b: 0.3658 };
    }
    function draw() {
      const p = params(), c = Math.pow(10, C.get());
      const Lf = (N) => p.E + p.A / Math.pow(N, p.a) + p.B / Math.pow(c / (6 * N), p.b);
      const G = Math.pow((p.a * p.A) / (p.b * p.B), 1 / (p.a + p.b));
      const Nopt = G * Math.pow(c / 6, p.b / (p.a + p.b)), Dopt = c / (6 * Nopt);
      const lo = Math.log10(Nopt) - 2, hi = Math.log10(Nopt) + 2;
      let ymin = Infinity, ymax = -Infinity; const pts = [];
      for (let i = 0; i <= 200; i++) { const ln = lo + (hi - lo) * i / 200, v = Lf(Math.pow(10, ln)); pts.push([ln, v]); ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
      ymax = Math.min(ymax, ymin + 1.5);
      clear(cv); const f = fg(root), ax = axes(cv, lo, hi, ymin - 0.02, ymax, f, "log₁₀ N (parameters) at fixed compute", "predicted loss"), ctx = cv.ctx;
      ctx.strokeStyle = PAL[0]; ctx.lineWidth = 2.5; ctx.beginPath();
      pts.forEach((q, i) => { const y = Math.min(q[1], ymax); i ? ctx.lineTo(ax.X(q[0]), ax.Y(y)) : ctx.moveTo(ax.X(q[0]), ax.Y(y)); });
      ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = PAL[1]; ctx.beginPath(); ctx.arc(ax.X(Math.log10(Nopt)), ax.Y(Lf(Nopt)), 5, 0, 7); ctx.fill();
      out.textContent = "IsoFLOP curve at C = 10^" + C.get().toFixed(1) + " FLOPs.  Optimum: N* = " + fmtNum(Nopt) + " params, D* = " + fmtNum(Dopt) + " tokens  (D*/N* = " + (Dopt / Nopt).toFixed(1) + " tokens/param)\n" +
        "predicted loss L(N*, D*) = " + Lf(Nopt).toFixed(3) + "   exponents: N* ∝ C^" + (p.b / (p.a + p.b)).toFixed(3) + ",  D* ∝ C^" + (p.a / (p.a + p.b)).toFixed(3) +
        "\nLeft of the dot: model too small for the budget (undertrained capacity). Right: model too big (too few tokens).";
    }
    redrawers.push(draw); draw();
  };

  // ---------- 12. KV-cache memory ----------
  W["kv-cache"] = function (root) {
    title(root, "Inference memory calculator: weights + KV cache");
    const cc = controls(root);
    const presets = {
      "Llama-2-7B (MHA)": [7, 32, 32, 128], "Llama-3-8B (GQA 8)": [8, 32, 8, 128], "Llama-3-70B (GQA 8)": [70, 80, 8, 128], "Mistral-7B (GQA 8)": [7.2, 32, 8, 128]
    };
    const ps = select(cc, "preset", Object.keys(presets).map((k) => [k, k]), () => { const p = presets[ps.value]; N.set(p[0]); Ly.set(p[1]); Hk.set(p[2]); Dh.set(p[3]); draw(); });
    const cc2 = controls(root);
    const N = slider(cc2, "params (B)", 0.5, 400, 0.5, 7, (v) => v + "B", () => draw());
    const Ly = slider(cc2, "layers L", 1, 128, 1, 32, (v) => v, () => draw());
    const Hk = slider(cc2, "KV heads", 1, 128, 1, 32, (v) => v, () => draw());
    const Dh = slider(cc2, "head dim", 32, 256, 32, 128, (v) => v, () => draw());
    const cc3 = controls(root);
    const T = slider(cc3, "context T", 9, 20, 1, 12, (v) => String(Math.pow(2, v)), () => draw());
    const Bt = slider(cc3, "batch B", 1, 256, 1, 1, (v) => v, () => draw());
    const wb = select(cc3, "weights dtype", [["2", "bf16 (2 B)"], ["1", "int8/fp8 (1 B)"], ["0.5", "int4 (0.5 B)"]], () => draw());
    const kb = select(cc3, "KV dtype", [["2", "bf16 (2 B)"], ["1", "fp8 (1 B)"]], () => draw());
    const out = el("div", { class: "w-out" }, root);
    function draw() {
      const t = Math.pow(2, T.get());
      const kv = 2 * Ly.get() * Hk.get() * Dh.get() * t * Bt.get() * parseFloat(kb.value);
      const perTok = 2 * Ly.get() * Hk.get() * Dh.get() * parseFloat(kb.value);
      const w = N.get() * 1e9 * parseFloat(wb.value);
      out.textContent = "KV cache = 2 (K and V) × L × H_kv × d_h × T × B × bytes\n" +
        "         = 2 × " + Ly.get() + " × " + Hk.get() + " × " + Dh.get() + " × " + t + " × " + Bt.get() + " × " + kb.value + " = " + fmtBytes(kv) +
        "\nper token: " + fmtBytes(perTok) + "     weights: " + fmtBytes(w) + "     total ≈ " + fmtBytes(kv + w) + "  (+ activations & framework overhead)";
    }
    draw();
  };

  // ---------- 13. Entropy, cross-entropy, KL ----------
  W["kl"] = function (root) {
    title(root, "Entropy, cross-entropy and KL divergence (asymmetric!)");
    const cc = controls(root);
    const pl = [], ql = [];
    const lab = ["A", "B", "C", "D"];
    const init = [[1.5, 0.5, 0, -1], [0, 0, 0, 0]];
    el("span", null, cc, "P logits:");
    for (let i = 0; i < 4; i++) pl.push(slider(cc, lab[i], -3, 3, 0.1, init[0][i], (v) => v.toFixed(1), () => draw()));
    const cc2 = controls(root);
    el("span", null, cc2, "Q logits:");
    for (let i = 0; i < 4; i++) ql.push(slider(cc2, lab[i], -3, 3, 0.1, init[1][i], (v) => v.toFixed(1), () => draw()));
    const cv = makeCanvas(root, 640, 220);
    const out = el("div", { class: "w-out" }, root);
    function draw() {
      const P = softmax(pl.map((s) => s.get())), Q = softmax(ql.map((s) => s.get()));
      clear(cv);
      const vals = [], labels = [], cols = [];
      for (let i = 0; i < 4; i++) { vals.push(P[i], Q[i]); labels.push("P(" + lab[i] + ")", "Q(" + lab[i] + ")"); cols.push(PAL[0], PAL[1]); }
      bars(cv, vals, labels, { fg: fg(root), max: 1, color: cols });
      const HP = entropy(P), HPQ = -P.reduce((a, p, i) => a + p * Math.log(Q[i]), 0);
      const KLpq = P.reduce((a, p, i) => a + p * Math.log(p / Q[i]), 0), KLqp = Q.reduce((a, q, i) => a + q * Math.log(q / P[i]), 0);
      out.textContent = "H(P) = " + HP.toFixed(4) + "   H(P,Q) = " + HPQ.toFixed(4) + "   KL(P‖Q) = " + KLpq.toFixed(4) + "   KL(Q‖P) = " + KLqp.toFixed(4) + "   (nats)\n" +
        "Check: H(P,Q) = H(P) + KL(P‖Q) → " + HP.toFixed(4) + " + " + KLpq.toFixed(4) + " = " + (HP + KLpq).toFixed(4) + ".   KL ≥ 0, = 0 iff P = Q.";
    }
    redrawers.push(draw); draw();
  };

  // ---------- 14. LoRA parameter count ----------
  W["lora"] = function (root) {
    title(root, "LoRA: trainable parameters for W + BA");
    const cc = controls(root);
    const d = slider(cc, "d_out", 256, 16384, 256, 4096, (v) => v, () => draw());
    const k = slider(cc, "d_in", 256, 16384, 256, 4096, (v) => v, () => draw());
    const r = slider(cc, "rank r", 1, 256, 1, 16, (v) => v, () => draw());
    const out = el("div", { class: "w-out" }, root);
    function draw() {
      const full = d.get() * k.get(), lo = r.get() * (d.get() + k.get());
      out.textContent = "full ΔW: d_out × d_in = " + fmtNum(full) + " params\nLoRA B (d_out×r) + A (r×d_in): r(d_out + d_in) = " + fmtNum(lo) + " params  →  " + (100 * lo / full).toFixed(3) + "% of full";
    }
    draw();
  };

  // ---------- boot ----------
  function boot() {
    document.querySelectorAll(".widget[data-widget]").forEach((node) => {
      if (node.dataset.ready) return;
      const fn = W[node.dataset.widget];
      if (!fn) { node.textContent = "Unknown widget: " + node.dataset.widget; return; }
      node.dataset.ready = "1";
      try { fn(node); } catch (e) { node.textContent = "Widget error: " + e.message; }
    });
  }
  if (typeof document$ !== "undefined" && document$.subscribe) document$.subscribe(boot);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
  new MutationObserver(() => redrawers.forEach((f) => { try { f(); } catch (e) { } }))
    .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
})();
