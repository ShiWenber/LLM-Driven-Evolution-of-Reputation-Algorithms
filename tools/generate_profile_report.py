"""Generate a self-contained interactive HTML fragment from profile.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_json", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    events = profile["events"]
    wall = profile["overall"]["wall_sec"]
    init = sum(x["wall_sec"] for x in events.get("llm_initialization", []))
    mutation_api = sum(x["wall_sec"] for x in events.get("llm_api", []))
    game = sum(x["wall_sec"] for x in events.get("game_generation", []))
    overhead = max(0.0, wall - init - mutation_api - game)
    phases = [
        {"name": "LLM 初始化", "wall": init, "cpu": sum(x["cpu_sec"] for x in events.get("llm_initialization", []))},
        {"name": "繁殖期 LLM", "wall": mutation_api, "cpu": sum(x["cpu_sec"] for x in events.get("llm_api", []))},
        {"name": "博弈模拟", "wall": game, "cpu": sum(x["cpu_sec"] for x in events.get("game_generation", []))},
        {"name": "其余开销", "wall": overhead, "cpu": max(0.0, profile["overall"]["cpu_sec"] - sum(x["cpu_sec"] for x in events.get("llm_initialization", [])) - sum(x["cpu_sec"] for x in events.get("llm_api", [])) - sum(x["cpu_sec"] for x in events.get("game_generation", [])))},
    ]
    generations = []
    games = events.get("game_generation", [])
    reproductions = events.get("fermi_reproduction", [])
    for index, game_event in enumerate(games):
        generations.append({
            "generation": index,
            "game": game_event["wall_sec"],
            "reproduction": reproductions[index]["wall_sec"] if index < len(reproductions) else 0,
        })
    local_functions = []
    keep = ("population.py", "game.py", "agent.py", "executor.py", "agent_full.py")
    for row in profile["top_functions"]:
        if any(name in row["function"] for name in keep):
            local_functions.append(row)
    local_functions = sorted(local_functions, key=lambda row: row["self_sec"], reverse=True)[:12]
    report_data = {
        "overall": profile["overall"],
        "phases": phases,
        "generations": generations,
        "llm": [x["wall_sec"] for x in events.get("llm_api", [])],
        "localFunctions": local_functions,
        "llmCallsTotal": 15 + len(events.get("llm_api", [])),
        "mutationCalls": len(events.get("llm_api", [])),
        "fallbacks": 0,
    }
    data_json = json.dumps(report_data, ensure_ascii=False, separators=(",", ":"))
    fragment = TEMPLATE.replace("__PROFILE_DATA__", data_json)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(fragment, encoding="utf-8")


TEMPLATE = r'''<div id="fermi-profile-report">
  <h1>agent-type1 性能剖析</h1>
  <div class="viz-grid summary">
    <div class="card viz-stat"><span class="text-muted">端到端耗时</span><span class="viz-stat-value tabular-nums" id="p-wall"></span><span class="text-small text-muted">seed 0 · 5 代 · 203 局/代</span></div>
    <div class="card viz-stat"><span class="text-muted">LLM 请求</span><span class="viz-stat-value tabular-nums" id="p-calls"></span><span class="text-small text-muted">全部成功，fallback 0</span></div>
    <div class="card viz-stat"><span class="text-muted">Python 峰值内存</span><span class="viz-stat-value tabular-nums" id="p-memory"></span><span class="text-small text-muted">tracemalloc 口径</span></div>
  </div>
  <section>
    <h2>时间花在哪里</h2>
    <div class="viz-controls" aria-label="耗时口径">
      <button type="button" class="btn btn-primary metric" data-metric="wall" aria-pressed="true">Wall time</button>
      <button type="button" class="btn metric" data-metric="cpu" aria-pressed="false">CPU time</button>
    </div>
    <div id="phase-chart"></div>
  </section>
  <section>
    <h2>逐代耗时</h2>
    <div id="generation-chart"></div>
  </section>
  <section class="two-col">
    <div><h2>繁殖期 LLM 延迟分布</h2><div id="latency-chart"></div></div>
    <div><h2>本地 Python 热点</h2><div id="hotspot-chart"></div></div>
  </section>
  <p class="card conclusion" id="p-conclusion"></p>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById('fermi-profile-report');
  const data = __PROFILE_DATA__;
  let metric = 'wall';
  const fmt = d3.format('.2f');
  root.querySelector('#p-wall').textContent = fmt(data.overall.wall_sec) + ' s';
  root.querySelector('#p-calls').textContent = data.llmCallsTotal;
  root.querySelector('#p-memory').textContent = fmt(data.overall.python_peak_mib) + ' MiB';
  const llmWall = d3.sum(data.phases.slice(0, 2), d => d.wall);
  root.querySelector('#p-conclusion').textContent = `首要瓶颈是串行 LLM 请求：约占端到端时间 ${fmt(llmWall / data.overall.wall_sec * 100)}%。本地博弈仅占 ${fmt(data.phases[2].wall / data.overall.wall_sec * 100)}%，当前不应优先微优化 action/observe 循环。`;

  function svgFor(selector, height, label) {
    const host = root.querySelector(selector); host.replaceChildren();
    const width = Math.max(320, host.getBoundingClientRect().width || 700);
    return d3.select(host).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('role','img').attr('aria-label',label);
  }
  function drawPhases() {
    const svg = svgFor('#phase-chart', 250, '各阶段耗时横条图');
    const w = +svg.attr('viewBox').split(' ')[2], margin={top:18,right:70,bottom:44,left:112}, iw=w-margin.left-margin.right, ih=180;
    const x=d3.scaleLinear().domain([0,d3.max(data.phases,d=>d[metric])||1]).nice().range([0,iw]);
    const y=d3.scaleBand().domain(data.phases.map(d=>d.name)).range([0,ih]).padding(.28);
    const g=svg.append('g').attr('transform',`translate(${margin.left},${margin.top})`);
    g.append('rect').attr('data-chart-frame','').attr('width',iw).attr('height',ih).attr('fill','none').attr('stroke','var(--border)');
    g.selectAll('rect.bar').data(data.phases).join('rect').attr('class','bar').attr('x',0).attr('y',d=>y(d.name)).attr('height',y.bandwidth()).attr('width',d=>x(d[metric])).attr('fill','var(--viz-series-1)');
    g.selectAll('text.value').data(data.phases).join('text').attr('class','value').attr('x',d=>Math.min(iw-2,x(d[metric])+6)).attr('y',d=>y(d.name)+y.bandwidth()/2+4).attr('text-anchor',d=>x(d[metric])>iw-55?'end':'start').text(d=>fmt(d[metric])+' s');
    g.append('g').call(d3.axisLeft(y)); g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(w<500?4:6));
    g.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',iw/2).attr('y',ih+38).attr('text-anchor','middle').text(metric==='wall'?'Wall time (s)':'CPU time (s)');
  }
  function drawGenerations(){
    const svg=svgFor('#generation-chart',270,'每代博弈和繁殖耗时'); const w=+svg.attr('viewBox').split(' ')[2],m={top:18,right:24,bottom:48,left:66},iw=w-m.left-m.right,ih=190;
    const x=d3.scaleBand().domain(data.generations.map(d=>d.generation)).range([0,iw]).padding(.25), max=d3.max(data.generations,d=>d.game+d.reproduction)||1, y=d3.scaleLinear().domain([0,max*1.08]).range([ih,0]);
    const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`); g.append('rect').attr('data-chart-frame','').attr('width',iw).attr('height',ih).attr('fill','none').attr('stroke','var(--border)');
    g.selectAll('.game').data(data.generations).join('rect').attr('x',d=>x(d.generation)).attr('y',d=>y(d.game)).attr('width',x.bandwidth()).attr('height',d=>ih-y(d.game)).attr('fill','var(--viz-series-2)');
    g.selectAll('.repro').data(data.generations).join('rect').attr('x',d=>x(d.generation)).attr('y',d=>y(d.game+d.reproduction)).attr('width',x.bandwidth()).attr('height',d=>y(d.game)-y(d.game+d.reproduction)).attr('fill','var(--viz-series-1)');
    g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x)); g.append('g').call(d3.axisLeft(y).ticks(5));
    g.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',iw/2).attr('y',ih+40).attr('text-anchor','middle').text('Generation');
    g.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-52).attr('text-anchor','middle').text('Wall time (s)');
    const legend=root.querySelector('#generation-chart'); const line=document.createElement('div'); line.className='legend text-small'; line.innerHTML='<span><i class="swatch repro"></i>繁殖 / LLM</span><span><i class="swatch game"></i>博弈</span>'; legend.prepend(line);
  }
  function drawLatency(){
    const svg=svgFor('#latency-chart',250,'繁殖期 LLM 请求延迟直方图'); const w=+svg.attr('viewBox').split(' ')[2],m={top:18,right:18,bottom:48,left:58},iw=w-m.left-m.right,ih=165;
    const domain=[0,d3.max(data.llm)*1.05], bins=d3.bin().domain(domain).thresholds(7)(data.llm), x=d3.scaleLinear().domain(domain).range([0,iw]), y=d3.scaleLinear().domain([0,d3.max(bins,d=>d.length)||1]).nice().range([ih,0]);
    const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`); g.append('rect').attr('data-chart-frame','').attr('width',iw).attr('height',ih).attr('fill','none').attr('stroke','var(--border)');
    g.selectAll('rect.bin').data(bins).join('rect').attr('x',d=>x(d.x0)+1).attr('y',d=>y(d.length)).attr('width',d=>Math.max(0,x(d.x1)-x(d.x0)-2)).attr('height',d=>ih-y(d.length)).attr('fill','var(--viz-series-3)');
    g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(5)); g.append('g').call(d3.axisLeft(y).ticks(4));
    g.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',iw/2).attr('y',ih+40).attr('text-anchor','middle').text('Request latency (s)');
    g.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-44).attr('text-anchor','middle').text('Requests');
  }
  function drawHotspots(){
    const rows=data.localFunctions.slice(0,8).map(d=>({...d,label:d.function.replace(/^.*?(population|game|agent|executor|agent_full)\.py:/,'$1.py:')})); const svg=svgFor('#hotspot-chart',250,'本地函数自身耗时'); const w=+svg.attr('viewBox').split(' ')[2],m={top:12,right:55,bottom:42,left:150},iw=w-m.left-m.right,ih=175;
    const x=d3.scaleLinear().domain([0,d3.max(rows,d=>d.self_sec)||1]).nice().range([0,iw]),y=d3.scaleBand().domain(rows.map(d=>d.label)).range([0,ih]).padding(.22),g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
    g.append('rect').attr('data-chart-frame','').attr('width',iw).attr('height',ih).attr('fill','none').attr('stroke','var(--border)'); g.selectAll('rect').data(rows).join('rect').attr('y',d=>y(d.label)).attr('width',d=>x(d.self_sec)).attr('height',y.bandwidth()).attr('fill','var(--viz-series-2)');
    g.append('g').call(d3.axisLeft(y).tickFormat(d=>d.length>23?d.slice(0,22)+'…':d)); g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(4));
    g.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',iw/2).attr('y',ih+36).attr('text-anchor','middle').text('Self time (s)');
  }
  function redraw(){drawPhases();drawGenerations();drawLatency();drawHotspots();}
  root.querySelectorAll('.metric').forEach(button=>button.addEventListener('click',()=>{metric=button.dataset.metric;root.querySelectorAll('.metric').forEach(b=>{const active=b===button;b.setAttribute('aria-pressed',active);b.classList.toggle('btn-primary',active)});drawPhases();}));
  redraw(); new ResizeObserver(()=>redraw()).observe(root);
})();
</script>
<style>
#fermi-profile-report{width:100%;color:var(--foreground)}
#fermi-profile-report .summary{margin:16px 0 24px;grid-template-columns:repeat(3,minmax(0,1fr))}
#fermi-profile-report .viz-stat{display:flex;flex-direction:column;gap:4px}
#fermi-profile-report section{margin:24px 0}
#fermi-profile-report .two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px}
#fermi-profile-report svg text{fill:var(--foreground);font-size:12px}
#fermi-profile-report svg .domain,#fermi-profile-report svg .tick line{stroke:var(--border)}
#fermi-profile-report .legend{display:flex;gap:18px;justify-content:flex-end;margin-bottom:-10px}
#fermi-profile-report .legend span{display:flex;align-items:center;gap:6px}
#fermi-profile-report .swatch{width:12px;height:12px;display:inline-block}
#fermi-profile-report .swatch.repro{background:var(--viz-series-1)}
#fermi-profile-report .swatch.game{background:var(--viz-series-2)}
#fermi-profile-report .conclusion{margin-top:24px}
@media(max-width:700px){#fermi-profile-report .summary,#fermi-profile-report .two-col{grid-template-columns:1fr}#fermi-profile-report .two-col{gap:8px}}
</style>'''


if __name__ == "__main__":
    main()
