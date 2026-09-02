"""Generate a readable standalone interactive HTML report from profile.json."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_evolution(profile_path: Path) -> dict:
    candidates = list((profile_path.parent / "experiment-output").rglob("evolutionary.json"))
    if not candidates:
        return {}
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_json", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
    evolution = _load_evolution(args.profile_json)
    events, config = profile["events"], profile.get("config", {})
    result_config = evolution.get("config", {})
    wall = profile["overall"]["wall_sec"]

    init_events = events.get("llm_initialization", [])
    reproduction_events = events.get("fermi_reproduction", [])
    game_events = events.get("game_generation", [])
    llm_events = events.get("llm_api", [])
    init = sum(x["wall_sec"] for x in init_events)
    reproduction = sum(x["wall_sec"] for x in reproduction_events)
    game = sum(x["wall_sec"] for x in game_events)
    init_cpu = sum(x["cpu_sec"] for x in init_events)
    reproduction_cpu = sum(x["cpu_sec"] for x in reproduction_events)
    game_cpu = sum(x["cpu_sec"] for x in game_events)
    phases = [
        {"name": "LLM 初始化批次", "wall": init, "cpu": init_cpu},
        {"name": "Fermi / LLM 批次", "wall": reproduction, "cpu": reproduction_cpu},
        {"name": "博弈模拟", "wall": game, "cpu": game_cpu},
        {
            "name": "其余开销",
            "wall": max(0.0, wall - init - reproduction - game),
            "cpu": max(
                0.0,
                profile["overall"]["cpu_sec"]
                - init_cpu - reproduction_cpu - game_cpu,
            ),
        },
    ]
    generations = [
        {
            "generation": index,
            "game": event["wall_sec"],
            "reproduction": (
                reproduction_events[index]["wall_sec"]
                if index < len(reproduction_events) else 0
            ),
        }
        for index, event in enumerate(game_events)
    ]

    keep = ("population.py", "game.py", "agent.py", "executor.py", "agent_full.py")
    local_functions = [
        row for row in profile["top_functions"]
        if any(name in row["function"] for name in keep)
    ]
    local_functions = sorted(
        local_functions, key=lambda row: row["self_sec"], reverse=True
    )[:10]

    population_size = int(config.get("population_size", 0) or 0)
    target_interactions = int(config.get("target_interactions", 0) or 0)
    pairs_per_round = max(1, population_size // 2)
    games_per_generation = (
        math.ceil(target_interactions / pairs_per_round) * pairs_per_round
        if target_interactions > 0 else 0
    )
    lineage_events = evolution.get("lineage_events", [])
    fermi_jobs = sum(event.get("origin") != "initial" for event in lineage_events)
    expected_jobs = population_size + fermi_jobs if lineage_events else len(llm_events)
    retries = max(0, len(llm_events) - expected_jobs)
    fallbacks = int(result_config.get("fallback_init_count", 0) or 0) + int(
        result_config.get("fallback_mutation_count", 0) or 0
    )
    report_data = {
        "overall": profile["overall"],
        "phases": phases,
        "generations": generations,
        "llm": [x["wall_sec"] for x in llm_events],
        "localFunctions": local_functions,
        "llmCallsTotal": len(llm_events),
        "retries": retries,
        "fallbacks": fallbacks,
        "config": {
            "agentType": config.get("agent_type", "unknown"),
            "seed": config.get("seed", "?"),
            "generations": config.get("gens", len(game_events)),
            "populationSize": population_size,
            "updatesPerGeneration": config.get("updates_per_gen", "?"),
            "concurrency": result_config.get(
                "llm_concurrency", config.get("llm_concurrency") or population_size
            ),
            "gamesPerGeneration": games_per_generation,
        },
    }
    data_json = json.dumps(report_data, ensure_ascii=False, separators=(",", ":"))
    document = TEMPLATE.replace("__PROFILE_DATA__", data_json)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(document, encoding="utf-8")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fermi 实验性能剖析</title>
<style>
:root{color-scheme:light;--bg:#f4f7fb;--surface:#fff;--text:#172033;--muted:#64748b;--border:#dbe3ef;--blue:#2563eb;--cyan:#0891b2;--amber:#d97706;--green:#059669;--shadow:0 8px 28px rgba(31,41,55,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.5}
#fermi-profile-report{width:min(1240px,calc(100% - 40px));margin:0 auto;padding:36px 0 56px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}h2{font-size:20px;margin:0 0 16px}.subtitle{margin:0 0 24px;color:var(--muted)}
.grid{display:grid}.summary{grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card,.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow)}.stat{padding:18px;display:flex;flex-direction:column;min-height:118px}.stat-label,.muted{color:var(--muted)}.stat-value{font-size:27px;font-weight:750;margin:7px 0 1px;font-variant-numeric:tabular-nums}.small{font-size:13px}
.panel{padding:22px;margin-top:18px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}.two-col .panel{min-width:0}.chart{width:100%;min-height:220px;overflow:hidden}.controls{display:flex;gap:8px;margin:-4px 0 10px}.metric{border:1px solid var(--border);background:#fff;color:var(--text);border-radius:8px;padding:7px 12px;cursor:pointer}.metric.active{background:var(--blue);border-color:var(--blue);color:#fff}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{fill:var(--text);font-size:12px}svg .domain,svg .tick line{stroke:#aab7ca}.legend{display:flex;gap:16px;justify-content:flex-end;color:var(--muted);font-size:13px}.legend span{display:flex;align-items:center;gap:6px}.swatch{width:11px;height:11px;border-radius:3px;display:inline-block}.conclusion{padding:18px 20px;border-left:4px solid var(--blue);margin-top:18px;background:#eff6ff;border-radius:10px}.empty{color:var(--muted);display:grid;place-items:center;height:190px}
@media(max-width:900px){.summary{grid-template-columns:repeat(2,1fr)}.two-col{grid-template-columns:1fr}}@media(max-width:560px){#fermi-profile-report{width:min(100% - 20px,1240px);padding-top:22px}.summary{grid-template-columns:1fr}h1{font-size:25px}.panel{padding:14px}}
</style>
</head>
<body>
<main id="fermi-profile-report">
  <h1 id="report-title">性能剖析</h1><p class="subtitle" id="report-subtitle"></p>
  <section class="grid summary">
    <div class="card stat"><span class="stat-label">端到端耗时</span><span class="stat-value" id="p-wall"></span><span class="small muted">完整实验 wall time</span></div>
    <div class="card stat"><span class="stat-label">LLM API 调用</span><span class="stat-value" id="p-calls"></span><span class="small muted" id="p-retries"></span></div>
    <div class="card stat"><span class="stat-label">实际并发上限</span><span class="stat-value" id="p-concurrency"></span><span class="small muted">配置值</span></div>
    <div class="card stat"><span class="stat-label">Python 峰值内存</span><span class="stat-value" id="p-memory"></span><span class="small muted">tracemalloc 口径</span></div>
  </section>
  <section class="panel"><h2>端到端时间构成</h2><div class="controls"><button class="metric active" data-metric="wall">Wall time</button><button class="metric" data-metric="cpu">CPU time</button></div><div class="chart" id="phase-chart"></div></section>
  <section class="panel"><h2>逐代耗时</h2><div class="legend"><span><i class="swatch" style="background:var(--blue)"></i>Fermi / LLM</span><span><i class="swatch" style="background:var(--cyan)"></i>博弈</span></div><div class="chart" id="generation-chart"></div></section>
  <div class="two-col"><section class="panel"><h2>全部 LLM 请求延迟</h2><div class="chart" id="latency-chart"></div></section><section class="panel"><h2>本地 Python 热点</h2><div class="chart" id="hotspot-chart"></div></section></div>
  <p class="conclusion" id="p-conclusion"></p>
</main>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const data=__PROFILE_DATA__,root=document.getElementById('fermi-profile-report'),fmt=d3.format('.2f');let metric='wall';const c=data.config;
  root.querySelector('#report-title').textContent=`${c.agentType} 性能剖析`;root.querySelector('#report-subtitle').textContent=`seed ${c.seed} · ${c.generations} 代 · ${c.populationSize} agents · ${c.gamesPerGeneration} 局/代 · ${c.updatesPerGeneration} learners/代（无放回）`;
  root.querySelector('#p-wall').textContent=fmt(data.overall.wall_sec)+' s';root.querySelector('#p-calls').textContent=data.llmCallsTotal;root.querySelector('#p-concurrency').textContent=c.concurrency;root.querySelector('#p-memory').textContent=fmt(data.overall.python_peak_mib)+' MiB';root.querySelector('#p-retries').textContent=`重试 ${data.retries} · fallback ${data.fallbacks}`;
  const batchWall=data.phases[0].wall+data.phases[1].wall,llmWork=d3.sum(data.llm);root.querySelector('#p-conclusion').textContent=`LLM 批次占端到端时间 ${fmt(batchWall/data.overall.wall_sec*100)}%。API 请求累计耗时 ${fmt(llmWork)} 秒，在并发 ${c.concurrency} 下被压缩进 ${fmt(batchWall)} 秒的批次时间；本地博弈仅 ${fmt(data.phases[2].wall)} 秒。`;
  const color={blue:'#2563eb',cyan:'#0891b2',amber:'#d97706',green:'#059669',border:'#dbe3ef'};
  function hostSvg(selector,height,label){const host=root.querySelector(selector);host.replaceChildren();const width=Math.max(360,host.clientWidth||700);return d3.select(host).append('svg').attr('viewBox',`0 0 ${width} ${height}`).attr('role','img').attr('aria-label',label)}function frame(g,w,h){g.append('rect').attr('width',w).attr('height',h).attr('rx',5).attr('fill','#fbfdff').attr('stroke',color.border)}
  function phases(){const svg=hostSvg('#phase-chart',260,'各阶段耗时'),w=+svg.attr('viewBox').split(' ')[2],m={top:14,right:82,bottom:46,left:140},iw=w-m.left-m.right,ih=184,x=d3.scaleLinear().domain([0,d3.max(data.phases,d=>d[metric])||1]).nice().range([0,iw]),y=d3.scaleBand().domain(data.phases.map(d=>d.name)).range([0,ih]).padding(.3),g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);frame(g,iw,ih);g.selectAll('.bar').data(data.phases).join('rect').attr('x',0).attr('y',d=>y(d.name)).attr('height',y.bandwidth()).attr('width',d=>x(d[metric])).attr('rx',4).attr('fill',(d,i)=>[color.blue,color.amber,color.cyan,color.green][i]);g.selectAll('.value').data(data.phases).join('text').attr('x',d=>Math.min(iw-5,x(d[metric])+7)).attr('y',d=>y(d.name)+y.bandwidth()/2+4).attr('text-anchor',d=>x(d[metric])>iw-60?'end':'start').text(d=>fmt(d[metric])+' s');g.append('g').call(d3.axisLeft(y).tickSize(0)).call(g=>g.select('.domain').remove());g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(6));g.append('text').attr('x',iw/2).attr('y',ih+40).attr('text-anchor','middle').text(metric==='wall'?'Wall time (s)':'CPU time (s)')}
  function generations(){const svg=hostSvg('#generation-chart',270,'逐代耗时'),w=+svg.attr('viewBox').split(' ')[2],m={top:12,right:26,bottom:48,left:62},iw=w-m.left-m.right,ih=195,x=d3.scaleBand().domain(data.generations.map(d=>d.generation)).range([0,iw]).padding(.32),max=d3.max(data.generations,d=>d.game+d.reproduction)||1,y=d3.scaleLinear().domain([0,max*1.1]).range([ih,0]),g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);frame(g,iw,ih);g.selectAll('.game').data(data.generations).join('rect').attr('x',d=>x(d.generation)).attr('y',d=>y(d.game)).attr('width',x.bandwidth()).attr('height',d=>ih-y(d.game)).attr('fill',color.cyan);g.selectAll('.repro').data(data.generations).join('rect').attr('x',d=>x(d.generation)).attr('y',d=>y(d.game+d.reproduction)).attr('width',x.bandwidth()).attr('height',d=>y(d.game)-y(d.game+d.reproduction)).attr('fill',color.blue);g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x));g.append('g').call(d3.axisLeft(y).ticks(5));g.append('text').attr('x',iw/2).attr('y',ih+40).attr('text-anchor','middle').text('Generation');g.append('text').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-46).attr('text-anchor','middle').text('Wall time (s)')}
  function latency(){if(!data.llm.length){root.querySelector('#latency-chart').innerHTML='<div class="empty">无 LLM 请求</div>';return}const svg=hostSvg('#latency-chart',260,'LLM 延迟分布'),w=+svg.attr('viewBox').split(' ')[2],m={top:12,right:18,bottom:48,left:52},iw=w-m.left-m.right,ih=180,domain=[0,d3.max(data.llm)*1.05],bins=d3.bin().domain(domain).thresholds(Math.min(8,data.llm.length))(data.llm),x=d3.scaleLinear().domain(domain).range([0,iw]),y=d3.scaleLinear().domain([0,d3.max(bins,d=>d.length)||1]).nice().range([ih,0]),g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);frame(g,iw,ih);g.selectAll('.bin').data(bins).join('rect').attr('x',d=>x(d.x0)+1).attr('y',d=>y(d.length)).attr('width',d=>Math.max(1,x(d.x1)-x(d.x0)-2)).attr('height',d=>ih-y(d.length)).attr('fill',color.amber);g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(5));g.append('g').call(d3.axisLeft(y).ticks(4));g.append('text').attr('x',iw/2).attr('y',ih+40).attr('text-anchor','middle').text('Request latency (s)')}
  function hotspots(){const rows=data.localFunctions.slice(0,8).map(d=>({...d,label:d.function.replace(/^.*?(population|game|agent|executor|agent_full)\.py:/,'$1.py:')}));if(!rows.length){root.querySelector('#hotspot-chart').innerHTML='<div class="empty">无本地热点数据</div>';return}const svg=hostSvg('#hotspot-chart',260,'本地热点'),w=+svg.attr('viewBox').split(' ')[2],m={top:10,right:55,bottom:48,left:154},iw=w-m.left-m.right,ih=182,x=d3.scaleLinear().domain([0,d3.max(rows,d=>d.self_sec)||1]).nice().range([0,iw]),y=d3.scaleBand().domain(rows.map(d=>d.label)).range([0,ih]).padding(.25),g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);frame(g,iw,ih);g.selectAll('.bar').data(rows).join('rect').attr('y',d=>y(d.label)).attr('width',d=>x(d.self_sec)).attr('height',y.bandwidth()).attr('fill',color.cyan);g.append('g').call(d3.axisLeft(y).tickFormat(d=>d.length>24?d.slice(0,23)+'…':d).tickSize(0)).call(g=>g.select('.domain').remove());g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).ticks(4));g.append('text').attr('x',iw/2).attr('y',ih+40).attr('text-anchor','middle').text('Self time (s)')}
  function redraw(){phases();generations();latency();hotspots()}root.querySelectorAll('.metric').forEach(b=>b.addEventListener('click',()=>{metric=b.dataset.metric;root.querySelectorAll('.metric').forEach(x=>x.classList.toggle('active',x===b));phases()}));redraw();let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(redraw,120)})
})();
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()
