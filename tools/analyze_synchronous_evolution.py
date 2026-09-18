"""Describe one current synchronous evolutionary run against archived trajectories."""
from pathlib import Path
import csv, hashlib, json, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/manuscript_v2/synchronous_evolution_20260918'
SYNC=OUT/'official_sync_g100_i10000_N16_baseline_seed0/evolutionary.json'
ARCHIVE=ROOT/'results/manuscript_v2/v4_flash/runs'
FIG=ROOT/'paper_zh/figures/manuscript_v2'
MATCHED=('agent_type','population_size','num_generations','target_interactions_per_gen',
         'benefit','cost','observability','observability_p','fitness_window_fraction',
         'action_error_probability','observation_error_probability','llm_thinking',
         'learning_method','fermi_beta','mutation_rate_on_adoption','fermi_init_source',
         'imitation_learning_mode','updates_per_gen','mutation_temperature','initial_reputation')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def metrics(data,condition):
    trajectory=data['trajectory']
    assert [r['generation'] for r in trajectory]==list(range(100))
    assert all(r['n_interactions']==10000 for r in trajectory)
    values=[r['cooperation_rate_mean'] for r in trajectory]
    assert all(0<=v<=1 for v in values)
    events=[i for i in range(1,100) if values[i-1]>=.8 and values[i]<=.3]
    return {'condition':condition,'seed':data['config']['seed'],'initial':values[0],
            'late20':statistics.mean(values[80:100]),'final':values[-1],
            'whole_run_mean':statistics.mean(values),'minimum':min(values),
            'low_generations':sum(v<=.3 for v in values),'sharp_drops':len(events),
            'sharp_drop_generations':events,
            'fallback_init':data['config']['fallback_init_count'],
            'fallback_mutation':data['config']['fallback_mutation_count']}

def analyze():
    assert read(OUT/'completion.json')['exit_code']==0
    sync=read(SYNC)
    assert sync['config']['observation_schedule']=='synchronous'
    assert sync['config']['num_rounds_per_gen']==1250
    asynchronous=[read(ARCHIVE/f'seed{i}/evolutionary.json') for i in range(5)]
    for old in asynchronous:
        assert old['config']['observation_schedule']=='asynchronous'
        assert old['config']['num_rounds_per_gen']==10000
        for key in MATCHED:
            assert old['config'][key]==sync['config'][key],(key,old['config'][key],sync['config'][key])
    rows=[metrics(old,'Archived asynchronous') for old in asynchronous]+[metrics(sync,'Current synchronous')]
    numeric=('initial','late20','final','whole_run_mean','minimum','low_generations','sharp_drops')
    group={key:{'mean':statistics.mean(r[key] for r in rows[:5]),
                'sample_sd':statistics.stdev(r[key] for r in rows[:5])} for key in numeric}
    audits=[json.loads(line) for line in (OUT/'api_audit.jsonl').read_text().splitlines()]
    sources=[ARCHIVE/f'seed{i}/evolutionary.json' for i in range(5)]+[SYNC]
    source_records=[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
    old_codes=[r['code'] for r in asynchronous[0]['trajectory'][0]['population']]
    new_codes=[r['code'] for r in sync['trajectory'][0]['population']]
    assert old_codes!=new_codes
    result={'rows':rows,'asynchronous_aggregate':group,'matched_parameters':list(MATCHED),
            'sources':source_records,'initial_programs_matched':False,
            'sync_model':sync['config']['llm_model'],'archive_model':asynchronous[0]['config']['llm_model'],
            'api':{'requests':len(audits),'errors':sum(not r['success'] for r in audits),
                   'returned_models':sorted({r['returned_model'] for r in audits if r['success']}),
                   'total_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in audits)},
            'comparison_unit':'one synchronous trajectory versus five archived asynchronous trajectories; no paired inference',
            'sharp_drop_definition':'Adjacent generations from cooperation >=0.8 to <=0.3; not permanent extinction.',
            'low_generation_definition':'Generation cooperation <=0.3',
            'limits':['Different model endpoints and independently generated initial programs.',
                      'Scheduling also changes matching and the LLM rules prompt.',
                      'One synchronous run cannot estimate cross-seed variability or a causal scheduling effect.']}
    (OUT/'comparison_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(7,2.65),gridspec_kw={'width_ratios':[1.75,1]},layout='constrained')
    ax=axes[0]
    for seed,old in enumerate(asynchronous):
        values=[r['cooperation_rate_mean'] for r in old['trajectory']]
        ax.plot(range(100),values,color='#B8B8B8',lw=.7,alpha=.65,
                label='Archived async.: other seeds' if seed==1 else None) if seed else None
    ax.plot(range(100),[r['cooperation_rate_mean'] for r in asynchronous[0]['trajectory']],
            color='#2878B5',lw=1.1,ls='--',label='Archived async.: seed 0')
    ax.plot(range(100),[r['cooperation_rate_mean'] for r in sync['trajectory']],
            color='#C46B31',lw=1.1,label='Current sync.: seed 0')
    ax.axvspan(79.5,99.5,color='#777777',alpha=.07,zorder=-1)
    ax.set(xlabel='Generation',ylabel='Cooperation',xlim=(0,99),ylim=(-.025,1.035),title='(a) Evolutionary trajectories')
    ax.legend(fontsize=7.8,frameon=False,loc='lower right')
    ax=axes[1]
    positions=np.arange(3)
    for seed,row in enumerate(rows[:5]):
        vals=[row[k] for k in ('initial','late20','final')]
        ax.scatter(positions-.17+(seed-2)*.028,vals,marker='o',s=18,
                   color='#2878B5',alpha=.7,label='Archived async. (n=5)' if seed==0 else None)
    ax.scatter(positions+.18,[rows[-1][k] for k in ('initial','late20','final')],
               marker='D',s=27,color='#C46B31',label='Current sync. (n=1)')
    ax.set(xticks=positions,xticklabels=['Initial','Late 20','Final'],ylabel='Cooperation',
           ylim=(-.025,1.035),xlim=(-.5,2.5),title='(b) Individual-run summaries')
    ax.legend(fontsize=7.8,frameon=False,loc='center right')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#DDDDDD',lw=.5)
        ax.set_axisbelow(True)
    for suffix in ('pdf','png'):
        fig.savefig(FIG/f'evolution_protocol_comparison.{suffix}',dpi=220,bbox_inches='tight')
        fig.savefig(OUT/f'comparison.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (FIG/'evolution_protocol_provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,
                        'pdf.fonttype':42,'ps.fonttype':42}):
        r=analyze()
    print(json.dumps({'rows':r['rows'],'api':r['api'],'asynchronous_aggregate':r['asynchronous_aggregate']},indent=2))
