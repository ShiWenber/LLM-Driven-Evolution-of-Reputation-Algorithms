"""Audit matched injection cohorts and replay their programs for reputation agreement.

No LLM calls. Matrix measurements are new frozen-population replays, not stored
historical states. The evolutionary seed is the independent replication unit.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, csv, hashlib, json, random, shutil, statistics, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.game import DonorGame
from experiments.analysis.consensus.core import disagreement
from tools.prepare_manuscript_v2_evidence import signature
from tools.analyze_llm_reinitialization_control import summarize, validate_jobs

OUT=ROOT/'results/manuscript_v2/injection_source_comparison_20260918'
SOURCES={'baseline':ROOT/'results/manuscript_v2/v4_1_flash/runs',
         'llm':ROOT/'results/llm_reinitialization_20260917/v4_1_flash'}
GENERATIONS=[*range(0,100,10),99]
REPLAY_SEEDS=[0,1,2,3,4]
INTERACTIONS=10000
MATCHED=('population_size','num_generations','benefit','cost','seed','llm_model',
         'target_interactions_per_gen','fitness_window_fraction','fermi_beta',
         'updates_per_gen','agent_type','imitation_learning_mode','mutation_temperature',
         'llm_thinking','llm_max_tokens','action_error_probability',
         'observation_error_probability','observability','observability_p',
         'observation_schedule','mutation_rate_on_adoption')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];norms=[];roots=[];sources=[];curves={c:[] for c in SOURCES};operators=Counter()
    for seed in range(5):
        baseline=read(SOURCES['baseline']/f'seed{seed}/evolutionary.json')
        for condition,folder in SOURCES.items():
            path=folder/f'seed{seed}/evolutionary.json';data=read(path)
            assert data['trajectory'][0]==baseline['trajectory'][0]
            for key in MATCHED:
                assert data['config'][key]==baseline['config'][key],(condition,seed,key)
            assert data['config']['observation_schedule']=='asynchronous'
            assert data['config']['agent_type']=='agent-type1'
            assert all(r['n_interactions']==INTERACTIONS for r in data['trajectory'])
            row,ns,rs=summarize(data,condition,seed)
            rows.append(row);norms.extend(ns);roots.extend(rs)
            curves[condition].append([r['cooperation_rate_mean'] for r in data['trajectory']])
            source={'condition':condition,'seed':seed,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}
            if condition=='llm':
                jobs_path=path.parent/'offspring_jobs.jsonl'
                jobs=[json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
                operators.update(validate_jobs(data,jobs))
                assert data['config']['control']['source_sha256']==digest(SOURCES['baseline']/f'seed{seed}/evolutionary.json')
                source.update(jobs_sha256=digest(jobs_path),jobs_path=jobs_path.relative_to(ROOT).as_posix())
            sources.append(source)
    pooled={}
    for condition in SOURCES:
        selected=[r for r in rows if r['condition']==condition]
        pooled[condition]={k:{'mean':statistics.mean(r[k] for r in selected),'sample_sd':statistics.stdev(r[k] for r in selected)}
                           for k in ('gen0','last20','final')}
        pooled[condition].update({k:sum(r[k] for r in selected) for k in ('initial_root_descendants','injected_root_descendants',
                                                    'independent_births','parent_conditioned_births','fallback_init','fallback_mutation')})
        pooled[condition]['norms']={}
        for generation in (0,99):
            selected_norms=[n for n in norms if n['condition']==condition and n['generation']==generation]
            counts=Counter(n['signature'] for n in selected_norms)
            pooled[condition]['norms'][str(generation)]={'n':len(selected_norms),'distinct':len(counts),
                'image_scoring_joint':counts['GGGGBBBB|CDCD'],
                'conditional_action':sum(n['signature'].endswith('|CDCD') for n in selected_norms),
                'reward_C_penalize_D':sum(n['signature'].startswith('GGGGBBBB|') for n in selected_norms),
                'counts':dict(counts)}
    paired=[{'seed':s,'baseline_late':next(r['last20'] for r in rows if r['seed']==s and r['condition']=='baseline'),
             'llm_late':next(r['last20'] for r in rows if r['seed']==s and r['condition']=='llm')} for s in range(5)]
    for row in paired: row['difference']=row['llm_late']-row['baseline_late']
    summary={'sources':sources,'rows':rows,'pooled':pooled,'paired':paired,'curves':curves,
             'operators':dict(operators),'matched_parameters':list(MATCHED),'generation0_exact_match':True,
             'cohort':'Official-model cohort, historically classified as v4.1-flash; separate from low-initial-cooperation ARK v4 cohort.',
             'replay_protocol':{'generations':GENERATIONS,'seeds':REPLAY_SEEDS,'interactions':INTERACTIONS,
                 'schedule':'asynchronous','matrix_samples':list(range(8200,10001,200)),
                 'state':'Reset to zero for each frozen population and replay; no LLM, no evolution within replay.'}}
    write_json(OUT/'evolution_summary.json',summary)
    write_csv(OUT/'evolution_summary.csv',rows);write_csv(OUT/'norm_probes.csv',norms)
    write_csv(OUT/'surviving_roots.csv',roots);write_csv(OUT/'paired_late.csv',paired)
    manifest=[]
    for rel in ('experiments/v2_quantitative/game.py','experiments/v2_quantitative/agent.py',
                'experiments/v2_quantitative/executor.py','experiments/analysis/consensus/core.py',
                'tools/prepare_manuscript_v2_evidence.py','tools/analyze_injection_sources.py'):
        src=ROOT/rel;dst=OUT/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        manifest.append({'path':rel,'sha256':digest(src)})
    write_json(OUT/'analysis_manifest.json',manifest)
    print(json.dumps({'pooled':pooled,'operators':dict(operators),'paired':paired},indent=2))
    return summary

def matrix_of(agents):
    matrix=np.array([[a.get_reputation(b.agent_id) for b in agents] for a in agents],dtype=float)
    observed=np.array([[b.agent_id in a.reputations for b in agents] for a in agents],dtype=bool)
    np.fill_diagonal(observed,False)
    return matrix,observed

def replay_one(task):
    condition,seed,generation,replay_seed=task
    destination=OUT/'replays'/condition/f'seed{seed}_g{generation}_r{replay_seed}.json'
    data=read(SOURCES[condition]/f'seed{seed}/evolutionary.json')
    records=data['trajectory'][generation]['population'];cfg=data['config']
    random.seed(replay_seed);np.random.seed(replay_seed)
    agents=[QuantitativeAgent(a['agent_id'],a['code'],executor=V2StrategyExecutor(a['code'])) for a in records]
    game=DonorGame(population_size=len(agents),benefit=cfg['benefit'],cost=cfg['cost'],
                   observability=cfg['observability'],observability_p=cfg['observability_p'],
                   seed=replay_seed,fitness_window_fraction=cfg['fitness_window_fraction'],
                   action_error_probability=cfg['action_error_probability'],
                   observation_error_probability=cfg['observation_error_probability'],
                   observation_schedule='asynchronous')
    game.setup_population(agents)
    values=[]
    for completed in range(1,INTERACTIONS+1):
        interactions=game.play_step()['interactions']
        assert len(interactions)==1
        game.distribute_observations_and_self_judgments(interactions)
        if completed>=8200 and completed%200==0:
            matrix,observed=matrix_of(agents)
            assert observed.sum()==len(agents)*(len(agents)-1)
            values.append({'interactions':completed,'disagreement':disagreement(matrix,observed)})
    d=statistics.mean(r['disagreement'] for r in values)
    coops=sum(row[key]=='cooperate' for row in game._global_log for key in ('donor_action','recipient_action'))
    result={'condition':condition,'seed':seed,'generation':generation,'replay_seed':replay_seed,
            'interactions':len(game._global_log),'cooperation':coops/(2*INTERACTIONS),
            'disagreement_mean':d,'consensus_mean':1-d,'samples':values,
            'agent_ids':[a.agent_id for a in agents],
            'matrix_final':matrix.tolist(),'observed_final':observed.tolist()}
    write_json(destination,result)
    return condition,seed,generation,replay_seed

def run_replays(workers):
    summary=read(OUT/'evolution_summary.json')
    for source in summary['sources']:assert digest(ROOT/source['path'])==source['sha256']
    tasks=[(c,s,g,r) for c in SOURCES for s in range(5) for g in GENERATIONS for r in REPLAY_SEEDS]
    assert len(tasks)==550
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(replay_one,t) for t in tasks]
        for completed,future in enumerate(as_completed(futures),1):
            future.result()
            if completed%25==0:print(f'{completed}/{len(tasks)} frozen-population replays completed',flush=True)
    write_json(OUT/'replay_completion.json',{'runs':len(tasks),'elapsed_seconds':time.perf_counter()-start,'workers':workers})

def summarize_replays():
    rows=[];trajectory=[]
    for c in SOURCES:
        for s in range(5):
            for g in GENERATIONS:
                reps=[read(OUT/'replays'/c/f'seed{s}_g{g}_r{r}.json') for r in REPLAY_SEEDS]
                for rep in reps:
                    assert rep['interactions']==INTERACTIONS and len(rep['samples'])==10
                    matrix=np.array(rep['matrix_final']);mask=np.array(rep['observed_final'])
                    assert abs(disagreement(matrix,mask)-rep['samples'][-1]['disagreement'])<1e-12
                    assert abs(rep['consensus_mean']+rep['disagreement_mean']-1)<1e-12
                row={'condition':c,'seed':s,'generation':g,
                     'consensus':statistics.mean(r['consensus_mean'] for r in reps),
                     'disagreement':statistics.mean(r['disagreement_mean'] for r in reps),
                     'replay_cooperation':statistics.mean(r['cooperation'] for r in reps)}
                rows.append(row)
        for g in GENERATIONS:
            selected=[r for r in rows if r['condition']==c and r['generation']==g]
            trajectory.append({'condition':c,'generation':g,'mean':statistics.mean(r['consensus'] for r in selected),
                               'sample_sd':statistics.stdev(r['consensus'] for r in selected)})
    for seed in range(5):
        left=next(r for r in rows if r['condition']=='baseline' and r['seed']==seed and r['generation']==0)
        right=next(r for r in rows if r['condition']=='llm' and r['seed']==seed and r['generation']==0)
        assert left['consensus']==right['consensus'] and left['replay_cooperation']==right['replay_cooperation']
    summary={'per_seed':rows,'trajectory':trajectory,'definition':'C = 1 - D; D is within-target RMS reputation disagreement, excluding self scores.',
             'aggregation':'Average 10 late-window samples within replay, then 5 replay seeds within evolutionary seed; group SD across 5 evolutionary seeds.',
             'source':'New frozen-population replays with independent pair draws and immediate observations; not recovered historical matrices.'}
    write_json(OUT/'consensus_summary.json',summary);write_csv(OUT/'consensus_per_seed.csv',rows)
    print(json.dumps([r for r in trajectory if r['generation'] in (0,99)],indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','replay','summarize'))
    parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='replay':run_replays(args.workers)
    else:summarize_replays()
