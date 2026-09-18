"""Audit the standalone injection-placement experiment and draw paper figures."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,statistics,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.prepare_manuscript_v2_evidence import signature
from experiments.run_worst_fitness_injection import CONDITIONS,stream_seed
OUT=ROOT/'results/worst_fitness_injection_sili_c4_20260918'
FIG=ROOT/'paper_zh/figures/manuscript_v2'
COLORS=['#247A91','#C96B35']
LABELS=['Fermi-gated','Worst-fitness']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(data,events):
    tr=data['trajectory'];values=[r['cooperation_rate_mean'] for r in tr];cfg=data['config']
    assert len(tr)==100 and [r['generation'] for r in tr]==list(range(100))
    assert all(r['n_interactions']==10000 for r in tr)
    assert data['final_population']==tr[-1]['population']
    assert cfg['fallback_init_count']==cfg['fallback_mutation_count']==0
    condition=cfg['experiment']['condition'];seed=cfg['seed']
    lineages={r['lineage_id']:r for r in data['lineage_events']}
    def root(lid):
        seen=set()
        while lineages[lid]['parent_lineage_id'] is not None:
            assert lid not in seen;seen.add(lid);lid=lineages[lid]['parent_lineage_id']
        return lid
    roots=Counter(root(a['lineage_id']) for a in data['final_population'])
    probes=[('|'.join(signature(a['code']))) for a in data['final_population']]
    # Threshold summaries are descriptive and were specified before inspecting outcomes.
    drops=[i for i in range(1,100) if values[i-1]>=.8 and values[i]<=.3]
    low=[i for i,v in enumerate(values) if v<=.3]
    episodes=[];i=0
    while i<100:
        if values[i]>.3:i+=1;continue
        start=i
        while i+1<100 and values[i+1]<=.3:i+=1
        episodes.append({'start':start,'end':i,'length':i-start+1});i+=1
    injected=[e for e in lineages.values() if e['origin']=='independent_init']
    rewrites=[e for e in lineages.values() if e['origin']=='imitate']
    assert len(injected)==sum(len(e['injections']) for e in events)
    event_rows=[]
    for event in events:
        g=event['generation'];old=tr[g-1]['population']
        for target in event['injections']:
            i=target['index'];a=old[i]
            assert a['fitness']==target['fitness'] and a['cooperation_rate']==target['cooperation']
            assert a['lineage_id']==target['old_lineage_id']
            if condition=='worst_per_generation':assert a['fitness']==min(p['fitness'] for p in old)
            event_rows.append({'condition':condition,'seed':seed,'generation':g,'target_index':i,
                'target_fitness':a['fitness'],'target_cooperation':a['cooperation_rate'],
                'population_cooperation':values[g-1],
                'target_minus_population_cooperation':a['cooperation_rate']-values[g-1],
                'next_generation_cooperation_change':values[g]-values[g-1],
                'prior_population_low':values[g-1]<=.3,'target_high_cooperation':a['cooperation_rate']>=.9})
    initial_desc=sum(n for lid,n in roots.items() if lineages[lid]['origin']=='initial')
    return {'condition':condition,'seed':seed,'initial':values[0],'late20':statistics.mean(values[-20:]),'final':values[-1],
        'whole_run_mean':statistics.mean(values),'minimum':min(values),'low_generations':len(low),
        'abrupt_drops':len(drops),'longest_low_episode':max([e['length'] for e in episodes],default=0),
        'abrupt_drops_without_injection':sum(not events[g-1]['injections'] for g in drops),
        'independent_births':len(injected),'parental_rewrites':len(rewrites),
        'initial_root_descendants':initial_desc,'injected_root_descendants':16-initial_desc,
        'surviving_roots':len(roots),'surviving_injected_roots':sum(lineages[lid]['origin']=='independent_init' for lid in roots),
        'largest_root_family':max(roots.values()),'scoring_like_joint':probes.count('GGGGBBBB|CDCD'),
        'conditional_action':sum(p.endswith('|CDCD') for p in probes),'distinct_signatures':len(set(probes)),
        'fermi_jobs_overridden':sum(e['overrode_fermi_job'] for e in events),
        'target_cooperation_mean':statistics.mean(r['target_cooperation'] for r in event_rows),
        'target_minus_population_mean':statistics.mean(r['target_minus_population_cooperation'] for r in event_rows),
        'target_high_cooperation_count':sum(r['target_high_cooperation'] for r in event_rows)},event_rows,probes,episodes

def analyze(plot=True):
    plan=read(OUT/'plan.json');rows=[];event_rows=[];sources=[];curves={c:[] for c in CONDITIONS};norms={c:[] for c in CONDITIONS};raws={};api=Counter();returned=set();episodes=[]
    for snapshot in plan['code_manifest']:
        assert sha(OUT/'code_snapshot'/snapshot['path'])==snapshot['sha256']
    for job in plan['jobs']:
        folder=OUT/job['condition']/f"seed{job['seed']}"
        assert read(folder/'status.json')['state']=='completed'
        path=folder/'evolutionary.json';data=read(path);cfg=data['config'];raws[job['condition'],job['seed']]=data
        assert cfg['llm_model']==job['model']==plan['model']
        assert cfg['llm_concurrency']==job['concurrency']==plan['llm_concurrency']==4
        assert cfg['llm_provider']==job['provider']==plan['provider']=='sili'
        assert cfg['api_base_url']==job['base_url']==plan['base_url']=='https://api.siliconflow.cn/v1'
        assert cfg['llm_extra_body']==plan['extra_body']=={'enable_thinking':False}
        assert cfg['observation_schedule']=='asynchronous' and cfg['benefit']==3 and cfg['cost']==1
        assert sha(ROOT/job['source'])==job['source_sha256']
        assert data['trajectory'][0]==read(ROOT/job['source'])['trajectory'][0]
        events=lines(folder/'selection_events.jsonl');jobs=lines(folder/'offspring_jobs.jsonl');calls=lines(folder/'api_audit.jsonl')
        assert calls
        for call in calls:
            assert call['llm_concurrency']==4
            assert call['provider']=='sili' and call['base_url']==plan['base_url']
            assert call['requested_model']==plan['model'] and call['extra_body']==plan['extra_body']
            if call['ok']:assert call['response_model']==plan['model']
        assert len(events)==99
        for event in events:
            if job['condition']=='worst_per_generation':
                import random
                expected=random.Random(stream_seed(job['seed'],event['generation'],'injection')).random()
                assert event['injection_draw']==expected
                assert len(event['injections'])==int(expected<.1)
        for birth in jobs:
            a=data['trajectory'][birth['generation']]['population'][birth['output_index']]
            assert a['birth_gen']==birth['generation'] and a['origin']==birth['origin']
            assert a['parent_lineage_id']==birth['parent_lineage_id']
            assert birth['code_generated'] and hashlib.sha256(a['code'].encode()).hexdigest()==birth['code_sha256']
            assert birth['operator'] in ('llm_init','llm_mutate')
            if birth['operator']=='llm_init':assert birth['parent_id'] is birth['parent_lineage_id'] is None
        row,er,pr,ep=summarize(data,events);rows.append(row);event_rows.extend(er);norms[job['condition']].extend(pr)
        assert len(jobs)==row['independent_births']+row['parental_rewrites']
        episodes.append({'condition':job['condition'],'seed':job['seed'],'low_episodes':ep})
        curves[job['condition']].append([g['cooperation_rate_mean'] for g in data['trajectory']])
        api['requests']+=len(calls);api['errors']+=sum(not c['ok'] for c in calls)
        api['tokens']+=sum((c.get('usage') or {}).get('total_tokens',0) for c in calls)
        returned.update(c['response_model'] for c in calls if c['ok'])
        for name in ('evolutionary.json','selection_events.jsonl','offspring_jobs.jsonl','api_audit.jsonl'):
            p=folder/name;sources.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    for seed in range(5):
        a,b=[raws[c,seed] for c in CONDITIONS]
        assert a['trajectory'][0]==b['trajectory'][0]
        assert a['config']['evaluation_seeds']==b['config']['evaluation_seeds']
    aggregate={}
    numeric=[k for k in rows[0] if k not in ('condition','seed')]
    for c in CONDITIONS:
        selected=[r for r in rows if r['condition']==c]
        aggregate[c]={k:{'mean':statistics.mean(r[k] for r in selected),'sample_sd':statistics.stdev(r[k] for r in selected),
                        'sum':sum(r[k] for r in selected)} for k in numeric}
        aggregate[c]['pooled_signatures']=dict(Counter(norms[c]))
    paired=[]
    for seed in range(5):
        a,b=[next(r for r in rows if r['condition']==c and r['seed']==seed) for c in CONDITIONS]
        paired.append({'seed':seed,'late_difference':b['late20']-a['late20'],'whole_run_difference':b['whole_run_mean']-a['whole_run_mean'],
                       'low_generation_difference':b['low_generations']-a['low_generations']})
    result={'rows':rows,'aggregate':aggregate,'paired':paired,'sources':sources,'curves':curves,'api':dict(api),
        'provider':plan['provider'],'base_url':plan['base_url'],'requested_model':plan['model'],'llm_concurrency':plan['llm_concurrency'],
        'returned_models':sorted(returned),'episodes':episodes,'definitions':{'low':'cooperation <=0.3','abrupt_drop':'previous >=0.8 and current <=0.3',
        'late':'mean generations 80–99','target_cooperation':'prior-generation full-window executed individual cooperation; fitness uses last20%'}}
    write_json(OUT/'analysis.json',result);write_csv(OUT/'summary.csv',rows);write_csv(OUT/'injection_events.csv',event_rows);write_csv(OUT/'paired.csv',paired)
    if plot:draw(result,event_rows)
    return result

def draw(data,events):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.7),layout='constrained')
    for j,c in enumerate(CONDITIONS):
        arr=np.array(data['curves'][c]);ax=axs[0,j]
        for i,v in enumerate(arr):ax.plot(range(100),v,lw=.65,alpha=.45,color=COLORS[j])
        ax.plot(arr.mean(0),lw=1.6,color=COLORS[j]);ax.axvspan(80,99,color='gray',alpha=.1)
        ax.set(title=f'({chr(97+j)}) {LABELS[j]}',xlabel='Generation',ylabel='Cooperation',ylim=(0,1.03),xlim=(0,99))
    for row in data['paired']:
        x,y=row['late_difference']*100,row['whole_run_difference']*100
        axs[0,2].scatter(x,y,color='#555555',s=22)
        axs[0,2].annotate(str(row['seed']),(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    axs[0,2].axhline(0,color='#aaa',lw=.7);axs[0,2].axvline(0,color='#aaa',lw=.7)
    bounds=[max(2,max(abs(r[key])*100 for r in data['paired'])*1.25) for key in ('late_difference','whole_run_difference')]
    axs[0,2].set(title='(c) Paired cooperation changes',xlabel='Late mean change (pp)',ylabel='Whole-run change (pp)',xlim=(-bounds[0],bounds[0]),ylim=(-bounds[1],bounds[1]))
    for seed in range(5):
        vals=[next(r['surviving_roots'] for r in data['rows'] if r['condition']==c and r['seed']==seed) for c in CONDITIONS]
        xs=np.array([0,1])+(seed-2)*.045
        axs[1,1].plot(xs,vals,color='#bbb',lw=.8)
        axs[1,1].scatter(xs,vals,c=COLORS,s=19,marker=['o','s','^','D','v'][seed])
    for j,c in enumerate(CONDITIONS):
        selected=[r for r in data['rows'] if r['condition']==c]
        axs[1,0].scatter([j+(s-2)*.045 for s in range(5)],[r['independent_births'] for r in selected],color=COLORS[j],s=19)
        e=[r for r in events if r['condition']==c]
        axs[1,2].scatter([r['population_cooperation'] for r in e],[r['target_cooperation'] for r in e],color=COLORS[j],
            s=10,alpha=.5,marker='o' if j==0 else 'x',label=LABELS[j])
    axs[1,0].set(title='(d) Independent births',xticks=[0,1],xticklabels=['Gated','Worst'],ylabel='Injections per run',xlim=(-.4,1.4),ylim=(0,None))
    axs[1,1].set(title='(e) Surviving root families',xticks=[0,1],xticklabels=['Gated','Worst'],ylabel='Roots at generation 99',xlim=(-.4,1.4),ylim=(0,max(r['surviving_roots'] for r in data['rows'])+1))
    axs[1,2].plot([0,1],[0,1],ls='--',color='#888',lw=.7)
    axs[1,2].set(title='(f) Who is replaced?',xlabel='Population cooperation',ylabel='Target cooperation',xlim=(-.02,1.03),ylim=(-.02,1.03))
    axs[1,2].legend(frameon=False,loc='lower right',fontsize=8)
    for ext in ('pdf','png'):fig.savefig(FIG/f'worst_fitness_injection.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    write_json(FIG/'worst_fitness_provenance.json',{'sources':data['sources'],'provider':data['provider'],'model':data['requested_model'],'note':'Ten corrected sili-provider matched-initial asynchronous runs, five per mechanism. Each seed has common predetermined evaluation seeds. Injection frequency differs by design; target associations are not isolated causal effects.'})

if __name__=='__main__':
    result=analyze();print(json.dumps({'rows':result['rows'],'paired':result['paired'],'api':result['api']},indent=2))
