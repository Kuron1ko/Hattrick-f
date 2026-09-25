from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
OUTPUT = Path(
    r"C:\Users\nnqab\.codex\visualizations\2026\08\20"
    r"\01a01e4d-cc90-7e20-b168-31a37bdb83df"
    r"\asymmetric-knn-cdf-1x-2x-3x.html"
)
SOURCES = {
    "1×": HERE / "artifacts" / "1x_level4" / "report.json",
    "2×": TEST_DIR
    / "shared2x_teacher_toll_distill"
    / "artifacts"
    / "asymmetric_knn_level4"
    / "final_rows.json",
    "3×": HERE / "artifacts" / "3x_level4" / "report.json",
}
CLASSES = ("High", "Medium", "Low")


def series_data() -> dict:
    result = {}
    for load, source_path in SOURCES.items():
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        result[load] = {}
        for source in ("baseline", "candidate"):
            result[load][source] = {
                class_name: sorted(
                    round(float(row["norm_fulfill"]), 8)
                    for row in payload["rows"][source]
                    if row["class"] == class_name
                )
                for class_name in CLASSES
            }
    return result


def main() -> None:
    embedded = json.dumps(series_data(), separators=(",", ":"), ensure_ascii=False)
    fragment = r'''<div id="lcr-cdf-grid-20260823" class="lcr-viz-root">
  <style>
    #lcr-cdf-grid-20260823 {
      --lcr-text: var(--color-text-primary, #1f2937);
      --lcr-muted: var(--color-text-secondary, #667085);
      --lcr-panel: var(--color-background-primary, #ffffff);
      --lcr-grid: color-mix(in srgb, var(--lcr-muted) 22%, transparent);
      --lcr-border: color-mix(in srgb, var(--lcr-muted) 30%, transparent);
      color: var(--lcr-text);
      background: var(--lcr-panel);
      font-family: var(--font-sans, Inter, ui-sans-serif, system-ui, sans-serif);
      width: 100%;
      box-sizing: border-box;
      padding: 16px 18px 22px;
    }
    #lcr-cdf-grid-20260823 * { box-sizing: border-box; }
    #lcr-cdf-grid-20260823 .lcr-header { display:flex; align-items:flex-start; justify-content:space-between; gap:18px; margin-bottom:14px; }
    #lcr-cdf-grid-20260823 h2 { margin:0 0 5px; font-size:clamp(20px,2vw,30px); line-height:1.15; letter-spacing:-0.02em; }
    #lcr-cdf-grid-20260823 .lcr-subtitle { margin:0; color:var(--lcr-muted); font-size:13px; line-height:1.45; }
    #lcr-cdf-grid-20260823 .lcr-legend { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
    #lcr-cdf-grid-20260823 .lcr-toggle { display:flex; align-items:center; gap:7px; border:1px solid var(--lcr-border); border-radius:999px; background:var(--lcr-panel); color:var(--lcr-text); padding:7px 11px; font:inherit; font-size:12px; cursor:pointer; }
    #lcr-cdf-grid-20260823 .lcr-toggle[aria-pressed="false"] { opacity:.45; }
    #lcr-cdf-grid-20260823 .lcr-swatch { width:19px; height:0; border-top:3px solid; }
    #lcr-cdf-grid-20260823 .lcr-swatch.baseline { border-color:var(--viz-series-1); }
    #lcr-cdf-grid-20260823 .lcr-swatch.candidate { border-color:var(--viz-series-2); border-top-style:dashed; }
    #lcr-cdf-grid-20260823 .lcr-grid { display:grid; grid-template-columns:repeat(3,minmax(280px,1fr)); gap:12px; }
    #lcr-cdf-grid-20260823 .lcr-panel { position:relative; min-width:0; border:1px solid var(--lcr-border); border-radius:12px; background:var(--lcr-panel); padding:10px 10px 5px; overflow:hidden; }
    #lcr-cdf-grid-20260823 .lcr-panel-title { display:flex; align-items:baseline; justify-content:space-between; gap:8px; margin:0 2px -3px; }
    #lcr-cdf-grid-20260823 .lcr-panel-title strong { font-size:14px; }
    #lcr-cdf-grid-20260823 .lcr-panel-title span { color:var(--lcr-muted); font-size:11px; font-weight:600; }
    #lcr-cdf-grid-20260823 svg { display:block; width:100%; height:auto; min-height:210px; }
    #lcr-cdf-grid-20260823 .axis text { fill:var(--lcr-muted); font-size:10px; }
    #lcr-cdf-grid-20260823 .axis path, #lcr-cdf-grid-20260823 .axis line { stroke:var(--lcr-border); }
    #lcr-cdf-grid-20260823 .gridline line { stroke:var(--lcr-grid); }
    #lcr-cdf-grid-20260823 .gridline path { stroke:none; }
    #lcr-cdf-grid-20260823 .axis-label { fill:var(--lcr-muted); font-size:10.5px; font-weight:600; }
    #lcr-cdf-grid-20260823 .cdf-line { fill:none; stroke-width:2.2; vector-effect:non-scaling-stroke; }
    #lcr-cdf-grid-20260823 .cdf-line.baseline { stroke:var(--viz-series-1); }
    #lcr-cdf-grid-20260823 .cdf-line.candidate { stroke:var(--viz-series-2); stroke-dasharray:7 4; }
    #lcr-cdf-grid-20260823 .crosshair { stroke:var(--lcr-muted); stroke-width:1; stroke-dasharray:2 3; pointer-events:none; }
    #lcr-cdf-grid-20260823 .lcr-tooltip { position:absolute; pointer-events:none; display:none; min-width:142px; padding:8px 9px; border:1px solid var(--lcr-border); border-radius:8px; background:var(--lcr-panel); color:var(--lcr-text); box-shadow:0 6px 20px color-mix(in srgb, var(--lcr-text) 14%, transparent); font-size:11px; line-height:1.5; z-index:5; }
    #lcr-cdf-grid-20260823 .lcr-tip-x { font-weight:700; margin-bottom:3px; }
    #lcr-cdf-grid-20260823 .lcr-tip-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    #lcr-cdf-grid-20260823 .lcr-tip-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; }
    #lcr-cdf-grid-20260823 .lcr-tip-dot.baseline { background:var(--viz-series-1); }
    #lcr-cdf-grid-20260823 .lcr-tip-dot.candidate { background:var(--viz-series-2); }
    #lcr-cdf-grid-20260823 .lcr-note { margin:12px 2px 0; color:var(--lcr-muted); font-size:11px; line-height:1.5; }
    @media (max-width: 920px) {
      #lcr-cdf-grid-20260823 .lcr-header { flex-direction:column; }
      #lcr-cdf-grid-20260823 .lcr-legend { justify-content:flex-start; }
      #lcr-cdf-grid-20260823 .lcr-grid { grid-template-columns:repeat(2,minmax(260px,1fr)); }
    }
    @media (max-width: 610px) {
      #lcr-cdf-grid-20260823 { padding:12px; }
      #lcr-cdf-grid-20260823 .lcr-grid { grid-template-columns:1fr; }
    }
  </style>
  <div class="lcr-header">
    <div>
      <h2>GEANT 8sp · strict-ESM Level-4 CDF</h2>
      <p class="lcr-subtitle">Load-specific Hattrick vs. priority-asymmetric local case retrieval (Hattrick-LCR), snapshots 400–499</p>
    </div>
    <div class="lcr-legend" aria-label="Series visibility">
      <button type="button" class="lcr-toggle" data-series="baseline" aria-pressed="true"><span class="lcr-swatch baseline"></span>Hattrick</button>
      <button type="button" class="lcr-toggle" data-series="candidate" aria-pressed="true"><span class="lcr-swatch candidate"></span>Hattrick-LCR</button>
    </div>
  </div>
  <div class="lcr-grid" aria-label="3 by 3 CDF chart grid"></div>
  <p class="lcr-note">Exact empirical CDF with flat 0/1 tails. For NormFulfill, a curve farther right (hence lower at the same x) is better. Hyperparameters k<sub>M</sub>=2 and k<sub>L</sub>=32 were frozen from the 2× small experiment and reused unchanged at 1× and 3×.</p>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {
    const root = document.getElementById('lcr-cdf-grid-20260823');
    const data = __DATA__;
    const loads = ['1×','2×','3×'];
    const classes = ['High','Medium','Low'];
    const domains = {High:[0.88,1.005], Medium:[0.85,1.20], Low:[0.88,1.48]};
    const enabled = {baseline:true, candidate:true};
    const width = 420, height = 286, margin = {top:14,right:12,bottom:42,left:47};
    const grid = d3.select(root).select('.lcr-grid');

    function ecdf(values, domain) {
      const n = values.length;
      return [[domain[0],0],[values[0],0],...values.map((v,i)=>[v,(i+1)/n]),[domain[1],1]];
    }

    for (const load of loads) {
      for (const cls of classes) {
        const panel = grid.append('section').attr('class','lcr-panel');
        const heading = panel.append('div').attr('class','lcr-panel-title');
        heading.append('strong').text(cls);
        heading.append('span').text(load);
        const svg = panel.append('svg').attr('viewBox',`0 0 ${width} ${height}`).attr('role','img').attr('aria-label',`${load} ${cls} NormFulfill CDF`);
        const x = d3.scaleLinear().domain(domains[cls]).range([margin.left,width-margin.right]);
        const y = d3.scaleLinear().domain([0,1]).range([height-margin.bottom,margin.top]);
        svg.append('g').attr('class','gridline').attr('transform',`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(cls==='High'?6:7).tickSize(-(height-margin.top-margin.bottom)).tickFormat(''));
        svg.append('g').attr('class','gridline').attr('transform',`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(width-margin.left-margin.right)).tickFormat(''));
        svg.append('g').attr('class','axis').attr('transform',`translate(0,${height-margin.bottom})`).call(d3.axisBottom(x).ticks(cls==='High'?6:7).tickFormat(cls==='High'?d3.format('.2f'):d3.format('.1f')));
        svg.append('g').attr('class','axis').attr('transform',`translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.1f')));
        svg.append('text').attr('class','axis-label').attr('x',(margin.left+width-margin.right)/2).attr('y',height-5).attr('text-anchor','middle').text('NormFulfill');
        svg.append('text').attr('class','axis-label').attr('transform','rotate(-90)').attr('x',-(margin.top+height-margin.bottom)/2).attr('y',12).attr('text-anchor','middle').text('CDF');
        const line = d3.line().x(d=>x(d[0])).y(d=>y(d[1])).curve(d3.curveStepAfter);
        for (const source of ['baseline','candidate']) {
          svg.append('path').datum(ecdf(data[load][source][cls],domains[cls])).attr('class',`cdf-line ${source}`).attr('d',line);
        }
        const crosshair = svg.append('line').attr('class','crosshair').attr('y1',margin.top).attr('y2',height-margin.bottom).style('display','none');
        const tip = panel.append('div').attr('class','lcr-tooltip');
        svg.append('rect').attr('x',margin.left).attr('y',margin.top).attr('width',width-margin.left-margin.right).attr('height',height-margin.top-margin.bottom).attr('fill','transparent').style('cursor','crosshair')
          .on('mousemove', function(event) {
            const [px] = d3.pointer(event,this); const xv = x.invert(px+margin.left); crosshair.attr('x1',x(xv)).attr('x2',x(xv)).style('display',null);
            const rows = ['baseline','candidate'].filter(s=>enabled[s]).map(source=>{ const vals=data[load][source][cls]; const cdf=d3.bisectRight(vals,xv)/vals.length; const label=source==='baseline'?'Hattrick':'Hattrick-LCR'; return `<div class="lcr-tip-row"><span><i class="lcr-tip-dot ${source}"></i>${label}</span><b>${d3.format('.2f')(cdf)}</b></div>`; }).join('');
            tip.html(`<div class="lcr-tip-x">x = ${d3.format(cls==='High'?'.4f':'.3f')(xv)}</div>${rows}`).style('display','block');
            const bounds=panel.node().getBoundingClientRect(); const mouse=d3.pointer(event,panel.node()); tip.style('left',`${Math.min(mouse[0]+12,bounds.width-158)}px`).style('top',`${Math.max(35,mouse[1]-18)}px`);
          }).on('mouseleave',()=>{crosshair.style('display','none');tip.style('display','none');});
      }
    }
    d3.select(root).selectAll('.lcr-toggle').on('click',function(){ const key=this.dataset.series; enabled[key]=!enabled[key]; this.setAttribute('aria-pressed',String(enabled[key])); d3.select(root).selectAll(`.cdf-line.${key}`).style('display',enabled[key]?null:'none'); });
  })();
  </script>
</div>'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fragment.replace("__DATA__", embedded), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
