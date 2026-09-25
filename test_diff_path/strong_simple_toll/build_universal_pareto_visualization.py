from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT = HERE / "final_universal_pareto.json"


def compact_data(payload):
    result = {}
    for load in ("1", "2", "3"):
        result[load] = {}
        rows = payload["loads"][load]["evaluation"]["rows"]
        for class_name in ("High", "Medium", "Low"):
            result[load][class_name] = {}
            for source in ("baseline", "candidate"):
                result[load][class_name][source] = [
                    round(float(row["norm_fulfill"]), 8)
                    for row in rows[source]
                    if row["class"] == class_name
                ]
    return result


FRAGMENT = r'''
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<section id="upareto-cdf-v1" class="upv-root" aria-labelledby="upv-title">
  <style>
    #upareto-cdf-v1 {
      --upv-blue: var(--viz-series-1, #2f78d4);
      --upv-orange: var(--viz-series-2, #f28e2b);
      --upv-fg: var(--foreground, #17212b);
      --upv-muted: var(--muted-foreground, #667383);
      --upv-border: var(--border, #dce3ea);
      --upv-panel: var(--card, transparent);
      --upv-popover: var(--popover, #ffffff);
      color: var(--upv-fg);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      padding: 10px 12px 18px;
      width: 100%;
      box-sizing: border-box;
    }
    #upareto-cdf-v1 * { box-sizing: border-box; }
    #upareto-cdf-v1 .upv-header { margin: 0 0 12px; }
    #upareto-cdf-v1 h2 {
      font-size: clamp(22px, 3.1vw, 38px);
      letter-spacing: -0.025em;
      line-height: 1.08;
      margin: 0 0 4px;
      font-weight: 760;
    }
    #upareto-cdf-v1 .upv-subtitle {
      color: var(--upv-muted);
      font-size: clamp(12px, 1.5vw, 16px);
      margin: 0;
    }
    #upareto-cdf-v1 .upv-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      align-items: center;
      margin: 14px 0 10px;
    }
    #upareto-cdf-v1 .upv-legend button {
      appearance: none;
      border: 0;
      background: transparent;
      color: inherit;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 3px 0;
      font: inherit;
      font-size: 13px;
    }
    #upareto-cdf-v1 .upv-legend button[aria-pressed="false"] { opacity: .38; }
    #upareto-cdf-v1 .upv-key { width: 32px; height: 0; border-top: 3px solid; border-radius: 999px; }
    #upareto-cdf-v1 .upv-key.base { border-color: var(--upv-blue); }
    #upareto-cdf-v1 .upv-key.neo { border-color: var(--upv-orange); border-top-style: dashed; }
    #upareto-cdf-v1 .upv-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px 16px;
      width: 100%;
    }
    #upareto-cdf-v1 .upv-panel {
      min-width: 0;
      background: var(--upv-panel);
      border-radius: 8px;
      position: relative;
    }
    #upareto-cdf-v1 .upv-panel h3 {
      font-size: clamp(14px, 1.55vw, 18px);
      font-weight: 650;
      line-height: 1.2;
      text-align: center;
      margin: 0 0 2px;
    }
    #upareto-cdf-v1 svg { display: block; width: 100%; height: auto; overflow: visible; }
    #upareto-cdf-v1 .domain { stroke: color-mix(in srgb, var(--upv-fg) 72%, transparent); }
    #upareto-cdf-v1 .tick text { fill: var(--upv-muted); font-size: 10.5px; }
    #upareto-cdf-v1 .tick line { stroke: var(--upv-border); }
    #upareto-cdf-v1 .upv-axis-label { fill: var(--upv-fg); font-size: 11px; }
    #upareto-cdf-v1 .upv-guide { stroke: var(--upv-muted); stroke-dasharray: 3 3; opacity: .65; }
    #upareto-cdf-v1 .upv-tooltip {
      position: fixed;
      z-index: 9999;
      pointer-events: none;
      opacity: 0;
      color: var(--upv-fg);
      background: var(--upv-popover);
      border: 1px solid var(--upv-border);
      border-radius: 7px;
      box-shadow: 0 5px 18px color-mix(in srgb, var(--upv-fg) 14%, transparent);
      padding: 7px 9px;
      font-size: 11px;
      line-height: 1.45;
      white-space: nowrap;
    }
    #upareto-cdf-v1 .upv-tooltip b { font-weight: 680; }
    @media (max-width: 760px) {
      #upareto-cdf-v1 .upv-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 480px) {
      #upareto-cdf-v1 { padding-inline: 4px; }
      #upareto-cdf-v1 .upv-grid { grid-template-columns: 1fr; gap: 16px; }
    }
  </style>
  <header class="upv-header">
    <h2 id="upv-title">GEANT 8sp — strict ESM CDF</h2>
    <p class="upv-subtitle">A single class-symmetric U-Pareto objective and Pareto guard at 1×, 2×, and 3× load · Level-4 snapshots 400–499</p>
  </header>
  <nav class="upv-legend" aria-label="Series visibility">
    <button type="button" data-series-toggle="baseline" aria-pressed="true"><span class="upv-key base"></span>Hattrick</button>
    <button type="button" data-series-toggle="candidate" aria-pressed="true"><span class="upv-key neo"></span>U-Pareto · strict ESM</button>
  </nav>
  <div class="upv-grid"></div>
  <div class="upv-tooltip" role="status" aria-live="polite"></div>
  <script>
  (() => {
    const root = document.getElementById("upareto-cdf-v1");
    const raw = __DATA__;
    const loads = ["1", "2", "3"];
    const classes = ["High", "Medium", "Low"];
    const domains = { High: [0.88, 1.008], Medium: [0.60, 1.30], Low: [0.60, 1.60] };
    const colors = { baseline: "var(--upv-blue)", candidate: "var(--upv-orange)" };
    const labels = { baseline: "Hattrick", candidate: "U-Pareto" };
    const visible = { baseline: true, candidate: true };
    const tooltip = d3.select(root).select(".upv-tooltip");
    const charts = [];

    function quantile(sorted, p) {
      const i = (sorted.length - 1) * p;
      const lo = Math.floor(i), hi = Math.ceil(i), t = i - lo;
      return sorted[lo] * (1 - t) + sorted[hi] * t;
    }
    function erf(x) {
      const sign = x < 0 ? -1 : 1;
      const a = Math.abs(x);
      const t = 1 / (1 + 0.3275911 * a);
      const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
      return sign * y;
    }
    function normalCdf(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
    function sharedBandwidth(a, b, left, right) {
      const pooled = a.concat(b).slice().sort(d3.ascending);
      const mean = d3.mean(pooled);
      const deviation = Math.sqrt(d3.sum(pooled, d => (d - mean) ** 2) / Math.max(1, pooled.length - 1));
      let robust = (quantile(pooled, .75) - quantile(pooled, .25)) / 1.349;
      if (robust <= 1e-12) robust = deviation;
      const scale = deviation > 0 ? Math.min(deviation, robust) : robust;
      const silverman = 0.9 * scale * pooled.length ** (-0.2);
      return Math.max(1.5 * silverman, 0.045 * (pooled[pooled.length - 1] - pooled[0]), 2e-5 * (right - left), 1e-10);
    }
    function smoothCdf(samples, left, right, bandwidth) {
      const span = right - left;
      const start = Math.max(left + .025 * span, d3.min(samples) - 3.5 * bandwidth);
      const end = Math.min(right - .025 * span, d3.max(samples) + 3.5 * bandwidth);
      const dense = d3.range(701).map(i => {
        const x = start + (end - start) * i / 700;
        return { x, y: d3.mean(samples, s => normalCdf((x - s) / bandwidth)) };
      });
      const y0 = dense[0].y, den = Math.max(dense[dense.length - 1].y - y0, 1e-12);
      let prior = 0;
      dense.forEach(d => { d.y = prior = Math.max(prior, Math.max(0, Math.min(1, (d.y - y0) / den))); });
      return [{x: left, y: 0}, {x: start, y: 0}, ...dense, {x: end, y: 1}, {x: right, y: 1}];
    }
    function interp(points, x) {
      const bisect = d3.bisector(d => d.x).left;
      const i = Math.max(1, Math.min(points.length - 1, bisect(points, x)));
      const a = points[i - 1], b = points[i];
      const t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
      return a.y + Math.max(0, Math.min(1, t)) * (b.y - a.y);
    }
    function tooltipPosition(event) {
      const node = tooltip.node();
      const box = node.getBoundingClientRect();
      const pad = 10;
      let left = event.clientX + 14, top = event.clientY + 14;
      if (left + box.width > innerWidth - pad) left = event.clientX - box.width - 14;
      if (top + box.height > innerHeight - pad) top = event.clientY - box.height - 14;
      tooltip.style("left", `${Math.max(pad, left)}px`).style("top", `${Math.max(pad, top)}px`);
    }
    function build(load, className) {
      const panel = d3.select(root).select(".upv-grid").append("article").attr("class", "upv-panel");
      panel.append("h3").text(`${load}× load · ${className}`);
      const svg = panel.append("svg").attr("role", "img");
      svg.append("title").text(`${load}× load ${className} normalized fulfillment CDF`);
      svg.append("desc").text("Smoothed CDF comparison of Hattrick and U-Pareto under strict ESM inference. Curves farther right indicate higher normalized fulfillment.");
      const chart = { load, className, panel, svg, points: {} };
      charts.push(chart);
      return chart;
    }
    function draw(chart) {
      const width = Math.max(270, chart.panel.node().clientWidth);
      const compact = width < 360;
      const height = compact ? 235 : 250;
      const margin = {top: 8, right: 10, bottom: 48, left: 48};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const [left, right] = domains[chart.className];
      const x = d3.scaleLinear().domain([left, right]).range([0, innerW]);
      const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
      const a = raw[chart.load][chart.className].baseline;
      const b = raw[chart.load][chart.className].candidate;
      const bandwidth = sharedBandwidth(a, b, left, right);
      chart.points.baseline = smoothCdf(a, left, right, bandwidth);
      chart.points.candidate = smoothCdf(b, left, right, bandwidth);
      chart.svg.attr("viewBox", `0 0 ${width} ${height}`).selectAll("g.upv-canvas").remove();
      const g = chart.svg.append("g").attr("class", "upv-canvas").attr("transform", `translate(${margin.left},${margin.top})`);
      const xTicks = compact ? 4 : 5;
      g.append("g").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(xTicks).tickSize(-innerH).tickPadding(7));
      g.append("g").call(d3.axisLeft(y).ticks(5).tickSize(-innerW).tickPadding(7).tickFormat(d3.format(".1f")));
      g.append("text").attr("class", "upv-axis-label").attr("x", innerW / 2).attr("y", innerH + 41).attr("text-anchor", "middle").text("Normalized FulfillRatio");
      g.append("text").attr("class", "upv-axis-label").attr("transform", "rotate(-90)").attr("x", -innerH / 2).attr("y", -37).attr("text-anchor", "middle").text("CDF");
      const line = d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveMonotoneX);
      ["baseline", "candidate"].forEach(source => {
        g.append("path")
          .datum(chart.points[source])
          .attr("data-upv-series", source)
          .attr("fill", "none")
          .attr("stroke", colors[source])
          .attr("stroke-width", 2.8)
          .attr("stroke-linecap", "round")
          .attr("stroke-linejoin", "round")
          .attr("stroke-dasharray", source === "candidate" ? "9 6" : null)
          .attr("opacity", visible[source] ? 1 : 0)
          .attr("d", line);
      });
      const hover = g.append("g").attr("display", "none");
      hover.append("line").attr("class", "upv-guide").attr("y1", 0).attr("y2", innerH);
      ["baseline", "candidate"].forEach(source => hover.append("circle").attr("data-hover-series", source).attr("r", 3.4).attr("fill", colors[source]).attr("stroke", "var(--upv-popover)").attr("stroke-width", 1.4));
      g.append("rect")
        .attr("data-chart-hit", "true")
        .attr("data-chart-hover-overlay", "cross-series")
        .attr("width", innerW).attr("height", innerH).attr("fill", "transparent")
        .on("pointerenter", () => { hover.attr("display", null); tooltip.style("opacity", 1); })
        .on("pointerleave", () => { hover.attr("display", "none"); tooltip.style("opacity", 0); })
        .on("pointermove", function(event) {
          const px = Math.max(0, Math.min(innerW, d3.pointer(event, this)[0]));
          const xv = x.invert(px);
          hover.select("line").attr("x1", px).attr("x2", px);
          const rows = [];
          ["baseline", "candidate"].forEach(source => {
            const yv = interp(chart.points[source], xv);
            hover.select(`circle[data-hover-series='${source}']`).attr("display", visible[source] ? null : "none").attr("cx", px).attr("cy", y(yv));
            if (visible[source]) rows.push(`<span style="color:${colors[source]}">●</span> ${labels[source]}: ${d3.format(".3f")(yv)}`);
          });
          tooltip.html(`<b>${chart.load}× · ${chart.className}</b><br>x = ${d3.format(chart.className === "High" ? ".4f" : ".3f")(xv)}<br>${rows.join("<br>")}`);
          tooltipPosition(event);
        });
    }
    loads.forEach(load => classes.forEach(className => build(load, className)));
    let resizeTimer;
    const redraw = () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => charts.forEach(draw), 60); };
    new ResizeObserver(redraw).observe(root);
    charts.forEach(draw);
    d3.select(root).selectAll("button[data-series-toggle]").on("click", function() {
      const source = this.dataset.seriesToggle;
      visible[source] = !visible[source];
      this.setAttribute("aria-pressed", visible[source] ? "true" : "false");
      d3.select(root).selectAll(`[data-upv-series='${source}']`).attr("opacity", visible[source] ? 1 : 0);
    });
  })();
  </script>
</section>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    embedded = json.dumps(compact_data(payload), ensure_ascii=False, separators=(",", ":"))
    output = FRAGMENT.replace("__DATA__", embedded)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(output.strip() + "\n", encoding="utf-8")
    print(args.destination)


if __name__ == "__main__":
    main()
