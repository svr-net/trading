// Small dependency-free canvas chart library: line (with bands and sample paths),
// bar, scatter, histogram and heatmap charts with crosshair/hover tooltips, legends and a
// table view. Colours come from CSS custom properties so light/dark themes apply.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const seriesColor = (i) => css(`--series-${(i % 8) + 1}`);

export const fmt = {
  num(v, digits = 0) {
    if (v === null || v === undefined || Number.isNaN(v)) return '–';
    return Number(v).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
  },
  compact(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '–';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'bn';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'm';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    return v.toPrecision(3);
  },
  pct(v, digits = 2) { return (100 * v).toFixed(digits) + '%'; },
  bp(v, digits = 1) { return (1e4 * v).toFixed(digits) + 'bp'; },
  years(v) { return Number(v).toFixed(2) + 'y'; },
  ratio(v, digits = 2) { return v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(digits); },
};

/** x-axis formatter for charts drawn against day indices: shows the date of the day. */
export const dateAxis = (dates) => (v) => {
  const d = dates[Math.max(0, Math.min(dates.length - 1, Math.round(v)))];
  return d === undefined ? '' : String(d);
};

function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) { const d = Math.abs(min) || 1; min -= d * 0.5; max += d * 0.5; }
  const span = max - min;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}

function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

class Base {
  constructor(container, cfg) {
    this.root = document.createElement('div');
    this.root.className = 'chart';
    this.canvas = document.createElement('canvas');
    this.legend = document.createElement('div');
    this.legend.className = 'legend';
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'tooltip';
    this.tableBtn = document.createElement('button');
    this.tableBtn.className = 'table-toggle';
    this.tableBtn.textContent = 'Table';
    this.tableBox = document.createElement('div');
    this.tableBox.className = 'table-wrap';
    this.tableBox.hidden = true;
    this.root.append(this.tableBtn, this.canvas, this.legend, this.tooltip, this.tableBox);
    container.append(this.root);
    this.tableBtn.addEventListener('click', () => {
      this.tableBox.hidden = !this.tableBox.hidden;
      this.tableBtn.textContent = this.tableBox.hidden ? 'Table' : 'Chart';
      this.canvas.hidden = !this.tableBox.hidden;
      if (!this.tableBox.hidden) this.renderTable();
    });
    this.canvas.addEventListener('mousemove', (e) => this.onHover(e));
    this.canvas.addEventListener('mouseleave', () => { this.hoverIndex = null; this.tooltip.style.display = 'none'; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(this.canvas);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.draw());
    this.pad = { l: 64, r: 16, t: 12, b: 36 };
    this.update(cfg);
  }

  update(cfg) {
    this.cfg = cfg;
    if (cfg.height) this.canvas.style.height = cfg.height + 'px';
    this.prepare();
    this.renderLegend();
    if (!this.tableBox.hidden) this.renderTable();
    this.draw();
    return this;
  }

  setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return null;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  drawAxes(ctx, w, h, xTicks, yTicks, xs, ys, xFmt, yFmt) {
    const { l, r, t, b } = this.pad;
    ctx.font = '11px ' + css('--font');
    ctx.strokeStyle = css('--grid');
    ctx.fillStyle = css('--text-muted');
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yTicks) {
      const y = Math.round(ys(v)) + 0.5;
      if (y < t - 1 || y > h - b + 1) continue;
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(w - r, y); ctx.stroke();
      ctx.fillText(yFmt(v), l - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xTicks) {
      const x = xs(v);
      if (x < l - 1 || x > w - r + 1) continue;
      ctx.fillText(xFmt(v), x, h - b + 6);
    }
    if (this.cfg.xLabel) { ctx.fillText(this.cfg.xLabel, (l + w - r) / 2, h - 14); }
    if (this.cfg.yLabel) {
      ctx.save(); ctx.translate(12, (t + h - b) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle'; ctx.fillText(this.cfg.yLabel, 0, 0); ctx.restore();
    }
  }

  renderLegend() {
    this.legend.innerHTML = '';
    const items = this.legendItems();
    if (items.length < 2) return;
    for (const it of items) {
      const s = document.createElement('span');
      const i = document.createElement('i');
      if (it.box) i.className = 'box';
      i.style.background = it.color;
      if (it.dash) i.style.background = `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)`;
      s.append(i, document.createTextNode(it.name));
      this.legend.append(s);
    }
  }

  showTooltip(px, py, title, rows) {
    this.tooltip.innerHTML = '';
    const t = document.createElement('div');
    t.className = 't';
    t.textContent = title;
    this.tooltip.append(t);
    for (const r of rows) {
      const row = document.createElement('div');
      row.className = 'row';
      const i = document.createElement('i');
      i.style.background = r.color;
      row.append(i, document.createTextNode(`${r.name}: ${r.value}`));
      this.tooltip.append(row);
    }
    this.tooltip.style.display = 'block';
    const w = this.canvas.clientWidth;
    const tw = this.tooltip.offsetWidth;
    this.tooltip.style.left = (px + 14 + tw > w ? px - tw - 14 : px + 14) + 'px';
    this.tooltip.style.top = Math.max(0, py - 10) + 'px';
  }

  renderTable() {
    const { headers, rows } = this.tableData();
    const table = document.createElement('table');
    table.className = 'data';
    const tr = document.createElement('tr');
    for (const h of headers) { const th = document.createElement('th'); th.textContent = h; tr.append(th); }
    table.append(tr);
    for (const r of rows) {
      const row = document.createElement('tr');
      for (const c of r) { const td = document.createElement('td'); td.textContent = c; row.append(td); }
      table.append(row);
    }
    this.tableBox.innerHTML = '';
    this.tableBox.append(table);
  }
}

/**
 * Line chart. cfg: { x, series: [{ name, y, x?, colorIndex?, color?, dash?, band: {lower, upper}? }],
 *   samples?: [[...]] thin background paths, xLabel, yLabel, xFormat, yFormat, zero?: bool, height }
 */
class LineChart extends Base {
  prepare() {
    const c = this.cfg;
    const xs = [], ys = [];
    for (const s of c.series) {
      const sx = s.x || c.x;
      xs.push(...sx);
      ys.push(...s.y);
      if (s.band) ys.push(...s.band.lower, ...s.band.upper);
    }
    if (c.samples) for (const p of c.samples) ys.push(...p);
    if (c.zero) ys.push(0);
    this.xr = extent(xs);
    this.yr = extent(ys);
  }

  legendItems() {
    return this.cfg.series.filter((s) => !s.hideLegend).map((s, i) => ({ name: s.name, color: this.colorOf(s, i), dash: s.dash }));
  }

  colorOf(s, i) { return s.color || seriesColor(s.colorIndex ?? i); }

  scales(w, h) {
    const { l, r, t, b } = this.pad;
    const [x0, x1] = this.xr;
    const yTicks = niceTicks(this.yr[0], this.yr[1]);
    const y0 = Math.min(yTicks[0], this.yr[0]), y1 = Math.max(yTicks[yTicks.length - 1], this.yr[1]);
    const xs = (v) => l + ((v - x0) / (x1 - x0 || 1)) * (w - l - r);
    const ys = (v) => h - b - ((v - y0) / (y1 - y0 || 1)) * (h - t - b);
    return { xs, ys, yTicks, xTicks: niceTicks(x0, x1, 6) };
  }

  draw() {
    const s = this.setupCanvas();
    if (!s) return;
    const { ctx, w, h } = s;
    const c = this.cfg;
    const { xs, ys, xTicks, yTicks } = this.scales(w, h);
    this.xs = xs;
    const xFmt = c.xFormat || ((v) => fmt.compact(v));
    const yFmt = c.yFormat || ((v) => fmt.compact(v));
    this.drawAxes(ctx, w, h, xTicks, yTicks, xs, ys, xFmt, yFmt);
    if (this.yr[0] < 0 && this.yr[1] > 0) {
      ctx.strokeStyle = css('--text-muted'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.pad.l, ys(0)); ctx.lineTo(w - this.pad.r, ys(0)); ctx.stroke();
    }
    if (c.samples) {
      ctx.strokeStyle = css('--text-muted');
      ctx.globalAlpha = 0.25;
      ctx.lineWidth = 1;
      for (const p of c.samples) this.path(ctx, c.x, p, xs, ys, c.stepSamples);
      ctx.globalAlpha = 1;
    }
    c.series.forEach((ser, i) => {
      const sx = ser.x || c.x;
      const color = this.colorOf(ser, i);
      if (ser.band) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.14;
        ctx.beginPath();
        sx.forEach((x, k) => (k ? ctx.lineTo(xs(x), ys(ser.band.upper[k])) : ctx.moveTo(xs(x), ys(ser.band.upper[k]))));
        for (let k = sx.length - 1; k >= 0; k--) ctx.lineTo(xs(sx[k]), ys(ser.band.lower[k]));
        ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (ser.pointsOnly) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.5;
        sx.forEach((x, k) => { ctx.beginPath(); ctx.arc(xs(x), ys(ser.y[k]), 2.2, 0, 2 * Math.PI); ctx.fill(); });
        ctx.globalAlpha = 1;
        return;
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = ser.width || 2;
      ctx.setLineDash(ser.dash ? [5, 4] : []);
      this.path(ctx, sx, ser.y, xs, ys, ser.step);
      ctx.setLineDash([]);
      if (ser.markers) {
        ctx.fillStyle = color;
        sx.forEach((x, k) => { ctx.beginPath(); ctx.arc(xs(x), ys(ser.y[k]), 4, 0, 2 * Math.PI); ctx.fill(); });
      }
    });
    if (this.hoverIndex != null) {
      const x = xs(c.x[this.hoverIndex]);
      ctx.strokeStyle = css('--text-muted'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, this.pad.t); ctx.lineTo(x, h - this.pad.b); ctx.stroke();
      c.series.forEach((ser, i) => {
        if (ser.x) return;
        const v = ser.y[this.hoverIndex];
        if (!Number.isFinite(v)) return;
        ctx.fillStyle = this.colorOf(ser, i);
        ctx.strokeStyle = css('--surface-1'); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, ys(v), 4.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      });
    }
  }

  path(ctx, x, y, xs, ys, step) {
    ctx.beginPath();
    let started = false;
    for (let k = 0; k < x.length; k++) {
      if (!Number.isFinite(y[k])) { started = false; continue; }
      const px = xs(x[k]), py = ys(y[k]);
      if (!started) { ctx.moveTo(px, py); started = true; } else if (step) { ctx.lineTo(px, ys(y[k - 1])); ctx.lineTo(px, py); } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  onHover(e) {
    const c = this.cfg;
    if (!c.x || !this.xs) return;
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    let best = 0, bd = Infinity;
    c.x.forEach((x, k) => { const d = Math.abs(this.xs(x) - mx); if (d < bd) { bd = d; best = k; } });
    this.hoverIndex = best;
    this.draw();
    const xFmt = c.xFormat || ((v) => fmt.compact(v));
    const yFmt = c.tooltipFormat || c.yFormat || ((v) => fmt.compact(v));
    const rows = c.series.filter((s) => !s.x).map((s, i) => ({ name: s.name, color: this.colorOf(s, i), value: yFmt(s.y[best]) }));
    this.showTooltip(mx, e.clientY - rect.top, `${c.xLabel || 'x'} = ${xFmt(c.x[best])}`, rows);
  }

  tableData() {
    const c = this.cfg;
    const series = c.series.filter((s) => !s.x);
    const yFmt = c.tooltipFormat || c.yFormat || ((v) => fmt.compact(v));
    return {
      headers: [c.xLabel || 'x', ...series.map((s) => s.name)],
      rows: c.x.map((x, k) => [(c.xFormat || ((v) => fmt.compact(v)))(x), ...series.map((s) => yFmt(s.y[k]))]),
    };
  }
}

/** Bar chart. cfg: { labels, series: [{ name, values }], yFormat, height } — grouped bars, 0 baseline. */
class BarChart extends Base {
  prepare() {
    const vals = this.cfg.series.flatMap((s) => s.values);
    vals.push(0);
    this.yr = extent(vals);
  }

  legendItems() { return this.cfg.series.map((s, i) => ({ name: s.name, color: s.color || seriesColor(s.colorIndex ?? i), box: true })); }

  draw() {
    const s = this.setupCanvas();
    if (!s) return;
    const { ctx, w, h } = s;
    const c = this.cfg;
    const { l, r, t, b } = this.pad;
    const yTicks = niceTicks(this.yr[0], this.yr[1]);
    const y0 = Math.min(yTicks[0], this.yr[0]), y1 = Math.max(yTicks[yTicks.length - 1], this.yr[1]);
    const ys = (v) => h - b - ((v - y0) / (y1 - y0 || 1)) * (h - t - b);
    const n = c.labels.length, m = c.series.length;
    const slot = (w - l - r) / n;
    const barW = Math.max(2, Math.min(36, (slot * 0.8 - 2 * (m - 1)) / m));
    const yFmt = c.yFormat || ((v) => fmt.compact(v));
    // axes (category labels drawn below)
    this.drawAxes(ctx, w, h, [], yTicks, () => 0, ys, () => '', yFmt);
    ctx.fillStyle = css('--text-muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const every = Math.ceil(n / Math.max(1, Math.floor((w - l - r) / 70)));
    c.labels.forEach((lab, i) => { if (i % every === 0) ctx.fillText(String(lab), l + slot * (i + 0.5), h - b + 6); });
    this.hits = [];
    c.series.forEach((ser, k) => {
      ctx.fillStyle = ser.color || seriesColor(ser.colorIndex ?? k);
      ser.values.forEach((v, i) => {
        const x = l + slot * (i + 0.5) - (m * barW + 2 * (m - 1)) / 2 + k * (barW + 2);
        const yTop = ys(Math.max(v, 0)), yBot = ys(Math.min(v, 0));
        const hgt = Math.max(1, yBot - yTop);
        const rad = Math.min(4, barW / 2, hgt);
        ctx.beginPath();
        if (v >= 0) ctx.roundRect(x, yTop, barW, hgt, [rad, rad, 0, 0]);
        else ctx.roundRect(x, yTop, barW, hgt, [0, 0, rad, rad]);
        ctx.globalAlpha = this.hover && (this.hover.i !== i) ? 0.45 : 1;
        ctx.fill();
        this.hits.push({ x: x - 1, w: barW + 2, i, k });
      });
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = css('--text-muted');
    ctx.beginPath(); ctx.moveTo(l, ys(0)); ctx.lineTo(w - r, ys(0)); ctx.stroke();
  }

  onHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const { l, r } = this.pad;
    const n = this.cfg.labels.length;
    const i = Math.floor(((mx - l) / (this.canvas.clientWidth - l - r)) * n);
    if (i < 0 || i >= n) { this.hover = null; this.tooltip.style.display = 'none'; this.draw(); return; }
    this.hover = { i };
    this.draw();
    const yFmt = this.cfg.tooltipFormat || this.cfg.yFormat || ((v) => fmt.compact(v));
    this.showTooltip(mx, e.clientY - rect.top, String(this.cfg.labels[i]),
      this.cfg.series.map((s, k) => ({ name: s.name, color: s.color || seriesColor(s.colorIndex ?? k), value: yFmt(s.values[i]) })));
  }

  tableData() {
    const yFmt = this.cfg.tooltipFormat || this.cfg.yFormat || ((v) => fmt.compact(v));
    return {
      headers: [this.cfg.xLabel || '', ...this.cfg.series.map((s) => s.name)],
      rows: this.cfg.labels.map((lab, i) => [String(lab), ...this.cfg.series.map((s) => yFmt(s.values[i]))]),
    };
  }
}

/** Scatter chart. cfg: { series: [{ name, x, y }], diagonal?: bool, xLabel, yLabel } (max 3 series). */
class ScatterChart extends Base {
  prepare() {
    const xs = this.cfg.series.flatMap((s) => s.x), ys = this.cfg.series.flatMap((s) => s.y);
    this.xr = extent(xs);
    this.yr = extent(ys);
    if (this.cfg.square) {
      const lo = Math.min(this.xr[0], this.yr[0]), hi = Math.max(this.xr[1], this.yr[1]);
      this.xr = [lo, hi]; this.yr = [lo, hi];
    }
  }

  legendItems() { return this.cfg.series.map((s, i) => ({ name: s.name, color: seriesColor(s.colorIndex ?? i), box: true })); }

  draw() {
    const s = this.setupCanvas();
    if (!s) return;
    const { ctx, w, h } = s;
    const { l, r, t, b } = this.pad;
    const xT = niceTicks(...this.xr, 6), yT = niceTicks(...this.yr);
    const x0 = Math.min(xT[0], this.xr[0]), x1 = Math.max(xT[xT.length - 1], this.xr[1]);
    const y0 = Math.min(yT[0], this.yr[0]), y1 = Math.max(yT[yT.length - 1], this.yr[1]);
    const xs = (v) => l + ((v - x0) / (x1 - x0 || 1)) * (w - l - r);
    const ys = (v) => h - b - ((v - y0) / (y1 - y0 || 1)) * (h - t - b);
    this.scale = { xs, ys };
    const f = (v) => fmt.compact(v);
    this.drawAxes(ctx, w, h, xT, yT, xs, ys, this.cfg.xFormat || f, this.cfg.yFormat || f);
    if (this.cfg.diagonal) {
      ctx.strokeStyle = css('--text-muted'); ctx.setLineDash([4, 4]);
      const lo = Math.max(x0, y0), hi = Math.min(x1, y1);
      ctx.beginPath(); ctx.moveTo(xs(lo), ys(lo)); ctx.lineTo(xs(hi), ys(hi)); ctx.stroke(); ctx.setLineDash([]);
    }
    this.cfg.series.forEach((ser, k) => {
      ctx.fillStyle = seriesColor(ser.colorIndex ?? k);
      ctx.globalAlpha = ser.x.length > 300 ? 0.45 : 0.85;
      const rad = ser.x.length > 300 ? 2 : 4;
      for (let i = 0; i < ser.x.length; i++) { ctx.beginPath(); ctx.arc(xs(ser.x[i]), ys(ser.y[i]), rad, 0, 2 * Math.PI); ctx.fill(); }
    });
    ctx.globalAlpha = 1;
  }

  onHover(e) {
    if (!this.scale) return;
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bd = 144;
    this.cfg.series.forEach((ser, k) => ser.x.forEach((x, i) => {
      const d = (this.scale.xs(x) - mx) ** 2 + (this.scale.ys(ser.y[i]) - my) ** 2;
      if (d < bd) { bd = d; best = { k, i }; }
    }));
    if (!best) { this.tooltip.style.display = 'none'; return; }
    const ser = this.cfg.series[best.k];
    const f = this.cfg.tooltipFormat || ((v) => fmt.compact(v));
    this.showTooltip(mx, my, ser.name, [
      { name: this.cfg.xLabel || 'x', color: seriesColor(ser.colorIndex ?? best.k), value: f(ser.x[best.i]) },
      { name: this.cfg.yLabel || 'y', color: seriesColor(ser.colorIndex ?? best.k), value: f(ser.y[best.i]) },
    ]);
  }

  tableData() {
    const rows = [];
    this.cfg.series.forEach((s) => s.x.slice(0, 500).forEach((x, i) => rows.push([s.name, fmt.compact(x), fmt.compact(s.y[i])])));
    return { headers: ['series', this.cfg.xLabel || 'x', this.cfg.yLabel || 'y'], rows };
  }
}

/**
 * Heatmap of a matrix. cfg: { rows, cols (labels), values [row][col], format, xLabel, yLabel,
 * center (value drawn neutral; default 0) }. Cells above the centre shade towards the
 * accent colour, cells below towards the critical colour.
 */
class Heatmap extends Base {
  prepare() {
    const vals = this.cfg.values.flat().filter(Number.isFinite);
    const c = this.cfg.center ?? 0;
    this.span = Math.max(1e-12, ...vals.map((v) => Math.abs(v - c)));
    this.pad = { l: this.cfg.rowWidth || 72, r: 12, t: 8, b: this.cfg.xLabel ? 50 : 36 };
  }

  legendItems() { return []; }

  geometry(w, h) {
    const { l, r, t, b } = this.pad;
    const nr = this.cfg.rows.length, nc = this.cfg.cols.length;
    return { l, t, cw: (w - l - r) / nc, ch: (h - t - b) / nr, nr, nc };
  }

  draw() {
    const s = this.setupCanvas();
    if (!s) return;
    const { ctx, w, h } = s;
    const c = this.cfg, g = this.geometry(w, h), center = c.center ?? 0;
    const f = c.format || ((v) => fmt.compact(v));
    ctx.font = '11px ' + css('--font');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < g.nr; i++)
      for (let j = 0; j < g.nc; j++) {
        const v = c.values[i][j];
        const x = g.l + j * g.cw, y = g.t + i * g.ch;
        ctx.fillStyle = css('--surface-2');
        ctx.fillRect(x + 1, y + 1, g.cw - 2, g.ch - 2);
        if (Number.isFinite(v)) {
          ctx.globalAlpha = Math.min(1, 0.12 + 0.88 * Math.abs(v - center) / this.span);
          ctx.fillStyle = v >= center ? css('--accent') : css('--critical');
          ctx.fillRect(x + 1, y + 1, g.cw - 2, g.ch - 2);
          ctx.globalAlpha = 1;
        }
        const strong = Number.isFinite(v) && Math.abs(v - center) / this.span > 0.55;
        ctx.fillStyle = strong ? '#ffffff' : css('--text-primary');
        if (g.cw > 26 && g.ch > 14) ctx.fillText(Number.isFinite(v) ? f(v) : '–', x + g.cw / 2, y + g.ch / 2);
        if (this.hover && this.hover[0] === i && this.hover[1] === j) {
          ctx.strokeStyle = css('--text-primary'); ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, g.cw - 2, g.ch - 2);
        }
      }
    ctx.fillStyle = css('--text-muted');
    ctx.textAlign = 'right';
    const rowStep = Math.ceil(12 / g.ch);
    c.rows.forEach((lab, i) => { if (i % rowStep === 0) ctx.fillText(String(lab), g.l - 6, g.t + (i + 0.5) * g.ch); });
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const colStep = Math.ceil(40 / g.cw);
    c.cols.forEach((lab, j) => { if (j % colStep === 0) ctx.fillText(String(lab), g.l + (j + 0.5) * g.cw, g.t + g.nr * g.ch + 6); });
    if (c.xLabel) ctx.fillText(c.xLabel, g.l + (g.nc * g.cw) / 2, h - 16);
    if (c.yLabel) {
      ctx.save(); ctx.translate(10, g.t + (g.nr * g.ch) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle'; ctx.fillText(c.yLabel, 0, 0); ctx.restore();
    }
  }

  onHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const g = this.geometry(this.canvas.clientWidth, this.canvas.clientHeight);
    const j = Math.floor((mx - g.l) / g.cw), i = Math.floor((my - g.t) / g.ch);
    if (i < 0 || j < 0 || i >= g.nr || j >= g.nc) { this.hover = null; this.tooltip.style.display = 'none'; this.draw(); return; }
    this.hover = [i, j];
    this.draw();
    const f = this.cfg.tooltipFormat || this.cfg.format || ((v) => fmt.compact(v));
    this.showTooltip(mx, my, `${this.cfg.yLabel || 'row'} ${this.cfg.rows[i]} · ${this.cfg.xLabel || 'column'} ${this.cfg.cols[j]}`,
      [{ name: this.cfg.name || 'value', color: css('--accent'), value: f(this.cfg.values[i][j]) }]);
  }

  tableData() {
    const f = this.cfg.format || ((v) => fmt.compact(v));
    return { headers: ['', ...this.cfg.cols.map(String)], rows: this.cfg.rows.map((r, i) => [String(r), ...this.cfg.values[i].map(f)]) };
  }
}

export const heatmap = (el, cfg) => new Heatmap(el, cfg);
export const lineChart = (el, cfg) => new LineChart(el, cfg);
export const barChart = (el, cfg) => new BarChart(el, cfg);
export const scatterChart = (el, cfg) => new ScatterChart(el, cfg);

/** Histogram of raw values as a bar chart. */
export function histogram(el, values, { bins = 40, name = 'paths', xFormat = (v) => fmt.compact(v), height } = {}) {
  const [lo, hi] = extent(values);
  const width = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v - lo) / width))]++;
  const labels = counts.map((_, i) => xFormat(lo + (i + 0.5) * width));
  return barChart(el, { labels, series: [{ name, values: counts.map((c) => c / values.length) }], yFormat: (v) => fmt.pct(v, 0), tooltipFormat: (v) => fmt.pct(v, 2), height });
}
