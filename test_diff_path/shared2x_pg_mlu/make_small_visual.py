from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "artifacts" / "small"
CLASSES = ("High", "Medium", "Low")
SERIES = (
    ("control", "Original MLU"),
    ("pg_l0p1", "PG-MLU λ=0.1"),
    ("pg_l0p5", "PG-MLU λ=0.5"),
)


def load() -> dict[str, dict[str, list[float]]]:
    result = {}
    for key, label in SERIES:
        values = np.loadtxt(
            DATA_DIR / f"{key}_norm_fulfill.csv",
            delimiter=",",
            skiprows=1,
            dtype=np.float64,
        ).reshape(-1, 3)
        result[label] = {
            class_name: [round(float(value), 8) for value in values[:, index]]
            for index, class_name in enumerate(CLASSES)
        }
    return result


def fragment(data: dict[str, dict[str, list[float]]]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f'''<section id="pg-mlu-small-cdf">
  <h2>2× PG-MLU small-range validation</h2>
  <div class="viz-row pg-legend" aria-label="Series visibility"></div>
  <div class="pg-panels"></div>
  <div class="tooltip" role="tooltip" hidden></div>
</section>
<style>
  #pg-mlu-small-cdf {{ width: 100%; color: var(--foreground); }}
  #pg-mlu-small-cdf .pg-panels {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
  #pg-mlu-small-cdf .pg-panel {{ min-width: 0; }}
  #pg-mlu-small-cdf .pg-panel h3 {{ margin-bottom: 4px; font-weight: 500; }}
  #pg-mlu-small-cdf .pg-chart {{ display: block; width: 100%; min-height: 260px; }}
  #pg-mlu-small-cdf .pg-frame {{ fill: none; stroke: var(--border); stroke-width: 1; }}
  #pg-mlu-small-cdf .pg-grid line {{ stroke: var(--border); stroke-opacity: 0.5; }}
  #pg-mlu-small-cdf .pg-grid path {{ display: none; }}
  #pg-mlu-small-cdf .pg-axis text,
  #pg-mlu-small-cdf .axis-title {{ fill: var(--foreground); font-size: 12px; }}
  #pg-mlu-small-cdf .pg-axis path,
  #pg-mlu-small-cdf .pg-axis line {{ stroke: var(--border); }}
  #pg-mlu-small-cdf .pg-line {{ fill: none; stroke-width: 2.2; }}
  #pg-mlu-small-cdf .pg-hover-guide {{ stroke: var(--foreground); stroke-opacity: 0.5; stroke-width: 1; pointer-events: none; }}
  #pg-mlu-small-cdf .pg-hover-marker {{ stroke: var(--background); stroke-width: 1.5; pointer-events: none; }}
  #pg-mlu-small-cdf .pg-hit {{ fill: transparent; pointer-events: all; }}
  #pg-mlu-small-cdf .pg-legend button {{ color: var(--foreground); background: transparent; border: 0; padding: 4px 6px; }}
  #pg-mlu-small-cdf .pg-swatch {{ display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }}
  #pg-mlu-small-cdf .tooltip {{ position: absolute; pointer-events: none; background: var(--popover); color: var(--popover-foreground); padding: 8px 10px; border: 1px solid var(--border); }}
  @media (max-width: 720px) {{
    #pg-mlu-small-cdf .pg-panels {{ grid-template-columns: 1fr; }}
  }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("pg-mlu-small-cdf");
  const raw = {payload};
  const classes = ["High", "Medium", "Low"];
  const names = Object.keys(raw);
  const colors = ["var(--viz-series-1)", "var(--viz-series-2)", "var(--viz-series-3)"];
  const visible = new Map(names.map(name => [name, true]));
  const legend = root.querySelector(".pg-legend");
  const panels = root.querySelector(".pg-panels");
  const tooltip = root.querySelector(".tooltip");

  names.forEach((name, index) => {{
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", "true");
    button.innerHTML = `<span class="pg-swatch" style="background:${{colors[index]}}"></span>${{name}}`;
    button.addEventListener("click", () => {{
      const next = !visible.get(name);
      visible.set(name, next);
      button.setAttribute("aria-pressed", String(next));
      button.style.opacity = next ? "1" : "0.45";
      drawAll();
    }});
    legend.appendChild(button);
  }});

  const panelState = classes.map(className => {{
    const panel = document.createElement("section");
    panel.className = "pg-panel";
    const heading = document.createElement("h3");
    heading.textContent = className;
    const svg = d3.select(panel).append("svg")
      .attr("class", "pg-chart")
      .attr("role", "img")
      .attr("aria-label", `${{className}} NormFulFill empirical cumulative distribution`);
    svg.append("title").text(`${{className}} NormFulFill CDF`);
    svg.append("desc").text("Original MLU compared with two prediction-gap MLU strengths under strict ESM inference.");
    panel.prepend(heading);
    panels.appendChild(panel);
    return {{className, panel, svg}};
  }});

  function ecdf(values, domain) {{
    const sorted = values.slice().sort(d3.ascending);
    const points = [{{x: domain[0], y: 0}}, {{x: sorted[0], y: 0}}];
    sorted.forEach((x, index) => points.push({{x, y: (index + 1) / sorted.length}}));
    points.push({{x: domain[1], y: 1}});
    return points;
  }}

  function draw(state) {{
    const width = Math.max(320, state.panel.getBoundingClientRect().width || 320);
    const height = 300;
    const margin = {{top: 10, right: 14, bottom: 54, left: 64}};
    const all = names.flatMap(name => raw[name][state.className]);
    const extent = d3.extent(all);
    const span = Math.max(extent[1] - extent[0], 0.01);
    const domain = [extent[0] - span * 0.08, extent[1] + span * 0.08];
    const x = d3.scaleLinear().domain(domain).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);
    const svg = state.svg.attr("viewBox", `0 0 ${{width}} ${{height}}`);
    svg.selectAll("g, path, rect, line, circle, text.axis-title").remove();
    svg.append("g").attr("class", "pg-grid")
      .attr("transform", `translate(${{margin.left}},0)`)
      .call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
    svg.append("rect").attr("class", "pg-frame").attr("data-chart-frame", "")
      .attr("x", margin.left).attr("y", margin.top)
      .attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom);
    svg.append("g").attr("class", "pg-axis")
      .attr("transform", `translate(0,${{height - margin.bottom}})`)
      .call(d3.axisBottom(x).ticks(width < 420 ? 4 : 5));
    svg.append("g").attr("class", "pg-axis")
      .attr("transform", `translate(${{margin.left}},0)`)
      .call(d3.axisLeft(y).ticks(5));
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x")
      .attr("x", (margin.left + width - margin.right) / 2).attr("y", height - 10)
      .attr("text-anchor", "middle").text("NormFulFill");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y")
      .attr("transform", "rotate(-90)").attr("x", -(margin.top + height - margin.bottom) / 2)
      .attr("y", 16).attr("text-anchor", "middle").text("CDF");
    const line = d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveLinear);
    names.forEach((name, index) => {{
      if (!visible.get(name)) return;
      svg.append("path").datum(ecdf(raw[name][state.className], domain))
        .attr("class", "pg-line").attr("data-series", name)
        .attr("stroke", colors[index]).attr("d", line);
    }});
    const guide = svg.append("line").attr("class", "pg-hover-guide").attr("data-chart-hover-guide", "").style("display", "none");
    const markers = names.map((name, index) => svg.append("circle")
      .attr("class", "pg-hover-marker").attr("data-chart-hover-marker", "")
      .attr("r", 4).attr("fill", colors[index]).style("display", "none"));
    svg.append("rect").attr("class", "pg-hit").attr("data-chart-hit", "")
      .attr("data-chart-hover-overlay", "cross-series")
      .attr("x", margin.left).attr("y", margin.top)
      .attr("width", width - margin.left - margin.right)
      .attr("height", height - margin.top - margin.bottom)
      .on("pointermove", event => {{
        const [px] = d3.pointer(event);
        const xv = x.invert(px);
        guide.style("display", null).attr("x1", px).attr("x2", px).attr("y1", margin.top).attr("y2", height - margin.bottom);
        const rows = [];
        names.forEach((name, index) => {{
          if (!visible.get(name)) {{ markers[index].style("display", "none"); return; }}
          const sorted = raw[name][state.className].slice().sort(d3.ascending);
          const yv = d3.bisectRight(sorted, xv) / sorted.length;
          markers[index].style("display", null).attr("cx", px).attr("cy", y(yv));
          rows.push(`${{name}}: ${{yv.toFixed(3)}}`);
        }});
        tooltip.hidden = false;
        tooltip.textContent = `NormFulFill ${{xv.toFixed(4)}} · ` + rows.join(" · ");
        const rootRect = root.getBoundingClientRect();
        tooltip.style.left = `${{event.clientX - rootRect.left + 12}}px`;
        tooltip.style.top = `${{event.clientY - rootRect.top + 12}}px`;
      }})
      .on("pointerleave", () => {{
        guide.style("display", "none");
        markers.forEach(marker => marker.style("display", "none"));
        tooltip.hidden = true;
      }});
  }}

  function drawAll() {{ panelState.forEach(draw); }}
  drawAll();
  new ResizeObserver(drawAll).observe(root);
}})();
</script>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment(load()), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
