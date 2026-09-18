"""Standalone matched comparison of Fermi-gated and worst-fitness LLM injection.

No resume mode, no changes to the ordinary evolutionary CLI. prepare audits
initial populations without API calls; run executes both arms from those codes.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, shutil, threading, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from experiments.config.load_env import get_api_key, get_base_url, get_model
from experiments.evolution_log import build_evolution_results, trajectory_entry, write_evolution_json
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.evolution_architecture import (
    FermiEvolutionRule, GenerationPlan, OffspringJob, RetainedAgent)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'results/worst_fitness_injection_sili_c4_20260918'
DEFAULT_LLM_CONCURRENCY=4
DEFAULT_SEED_WORKERS=5
CONDITIONS=('fermi_gated','worst_per_generation')

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(6):
        try:
            tmp.replace(path);break
        except PermissionError:
            if attempt==5:raise
            time.sleep(.1*(attempt+1))
def stream_seed(seed,generation,stream):
    return int.from_bytes(hashlib.sha256(f'{seed}/{generation}/{stream}'.encode()).digest()[:8],'big')

class InjectionRule:
    """Plan from pre-update fitness; injection wins any slot collision."""
    def __init__(self,condition,*,seed,probability=.1,beta=5,updates_per_gen=16):
        if condition not in CONDITIONS:raise ValueError(condition)
        if not 0<=probability<=1:raise ValueError('Probability outside [0,1]')
        self.condition=condition;self.seed=seed;self.probability=probability
        self.name=condition;self.beta=beta;self.updates_per_gen=updates_per_gen;self.audit=[]

    def plan(self,population,*,rng,next_gen,lineage_by_agent_id):
        rate=self.probability if self.condition=='fermi_gated' else 0.
        rule=FermiEvolutionRule(beta=self.beta,mutation_rate=rate,updates_per_gen=self.updates_per_gen,init_source='llm')
        plan=rule.plan(population,rng=rng,next_gen=next_gen,lineage_by_agent_id=lineage_by_agent_id)
        jobs=list(plan.jobs);accepted=len(jobs);overridden=False;draw=None;target=None
        if self.condition=='worst_per_generation':
            independent=random.Random(stream_seed(self.seed,next_gen,'injection'))
            draw=independent.random()
            if draw<self.probability:
                minimum=min(a.fitness for a in population)
                tied=[i for i,a in enumerate(population) if a.fitness==minimum]
                target=independent.choice(tied);agent=population[target]
                overridden=any(job.output_index==target for job in jobs)
                jobs=[job for job in jobs if job.output_index!=target]
                jobs.append(OffspringJob(ordinal=len(jobs),output_index=target,operator='llm_init',
                    mutation_kind=None,preserve_agent_id=agent.agent_id,parent_id=None,
                    parent_lineage_id=None,parent_code=None,parent_fitness=None,
                    origin='independent_init',birth_gen=next_gen))
        jobs=tuple(replace(job,ordinal=i) for i,job in enumerate(jobs))
        changed={j.output_index for j in jobs}
        retained=tuple(RetainedAgent(i,a) for i,a in enumerate(population) if i not in changed)
        self.audit.append({'generation':next_gen,'condition':self.condition,'fermi_accepted':accepted,
            'injection_draw':draw,'injection_target_index':target,'overrode_fermi_job':overridden,
            'minimum_fitness':min(a.fitness for a in population),
            'injections':[{'index':j.output_index,'agent_id':population[j.output_index].agent_id,
                'old_lineage_id':population[j.output_index].lineage_id,
                'fitness':population[j.output_index].fitness,
                'cooperation':None} for j in jobs if j.operator=='llm_init'],
            'rewrites':sum(j.operator=='llm_mutate' for j in jobs)})
        return GenerationPlan(len(population),retained,jobs)

class MeasuredPopulation(V2EvolutionaryPopulation):
    def __init__(self,*,folder=None,**kwargs):
        super().__init__(**kwargs);self.folder=folder;self.audit_lock=threading.Lock()
        self.context=threading.local();self.proxy=None;self.evaluation_seeds=[]
    def _run_one_generation(self):
        g=self.round_num_offset
        seed=random.Random(self.seed).randrange(10**9) if g==0 else stream_seed(self.seed,g,'evaluation')%10**9
        self.evaluation_seeds.append({'generation':g,'seed':seed})
        return self._get_game_scenario().evaluate(self.agents,generation_seed=seed,num_rounds=self.num_rounds_per_gen).as_dict()
    def append(self,name,record):
        if self.folder is not None:
            with self.audit_lock:
                with (self.folder/name).open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    def _get_llm_client(self):
        client=super()._get_llm_client()
        if self.folder is None:return client
        with self.audit_lock:
            if self.proxy is None:
                def create(**kwargs):
                    start=time.monotonic();r={'time_utc':now(),'generation':getattr(self.context,'generation',None),
                        'operator':getattr(self.context,'operator',None),'requested_model':kwargs['model'],
                        'provider':self.llm_provider,'base_url':self.api_base_url,'extra_body':kwargs.get('extra_body'),
                        'llm_concurrency':self.llm_concurrency}
                    try:
                        response=client.chat.completions.create(**kwargs)
                        r.update(ok=True,response_model=response.model,finish_reason=response.choices[0].finish_reason,
                                 usage=response.usage.model_dump() if response.usage else None)
                        return response
                    except Exception as exc:
                        r.update(ok=False,error_type=type(exc).__name__,status_code=getattr(exc,'status_code',None));raise
                    finally:
                        r['elapsed_seconds']=time.monotonic()-start;self.append('api_audit.jsonl',r)
                self.proxy=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return self.proxy
    def _run_offspring_job(self,job):
        self.context.generation=job.birth_gen;self.context.operator=job.operator
        result=super()._run_offspring_job(job)
        self.append('offspring_jobs.jsonl',{'generation':job.birth_gen,'operator':job.operator,
            'output_index':job.output_index,'parent_id':job.parent_id,'parent_lineage_id':job.parent_lineage_id,
            'origin':job.origin,'code_generated':result.code is not None,
            'code_sha256':hashlib.sha256(result.code.encode()).hexdigest() if result.code else None})
        if result.code is None:raise RuntimeError('No validated LLM code; stop without a fallback population.')
        return result

def make_population(config,condition,*,model,concurrency,provider='sili',folder=None,credentials=False):
    endpoint=provider_endpoint(provider)
    keys=('population_size','target_interactions_per_gen','fitness_window_fraction','benefit','cost',
          'action_error_probability','observation_error_probability','observability','observability_p',
          'seed','agent_type','llm_thinking','learning_method','fermi_beta','imitation_learning_mode','updates_per_gen')
    kw={key:config[key] for key in keys}
    kw.update(observation_schedule='asynchronous',num_generations=100,mutation_temperature=.8,
        llm_model=model,llm_provider=provider,api_key=get_api_key(provider) if credentials else '',
        api_base_url=endpoint,llm_concurrency=concurrency,llm_max_tokens_base=4000,
        mutation_rate_on_adoption=.1 if condition=='fermi_gated' else 0.,fermi_init_source='llm',
        evolution_rule=InjectionRule(condition,seed=config['seed'],beta=config['fermi_beta'],updates_per_gen=config['updates_per_gen']),
        folder=folder)
    pop=MeasuredPopulation(**kw)
    # SiliconFlow uses enable_thinking, not the official provider's thinking object.
    pop._llm_extra_body={'enable_thinking':bool(pop.llm_thinking)}
    return pop

def initialize(pop,source):
    initial=source['trajectory'][0]['population']
    pop.agents=[pop._make_agent(a['code'],a['agent_id']) for a in initial]
    pop._next_agent_id=max(a.agent_id for a in pop.agents)+1;pop._init_lineage()

def evaluate(pop,generation):
    pop._round_offset=generation;stats=pop._run_one_generation()
    for a,f in zip(pop.agents,stats['payoffs'],strict=True):a.fitness=f
    return trajectory_entry(generation=generation,cooperation_rate_mean=stats['cooperation_rate_mean'],
        n_interactions=stats['n_interactions'],fitness_mean=sum(stats['payoffs'])/len(stats['payoffs']),
        fitness_max=max(stats['payoffs']),population=[pop._agent_record(a) for a in pop.agents])

def provider_endpoint(provider):
    if provider!='sili':raise ValueError('This corrected experiment requires provider sili')
    endpoint=get_base_url(provider).rstrip('/')
    url=urlsplit(endpoint)
    if url.scheme!='https' or url.netloc!='api.siliconflow.cn' or url.path!='/v1':
        raise ValueError('sili must route to https://api.siliconflow.cn/v1')
    return endpoint

def validate_job_provider(job):
    endpoint=provider_endpoint(job.get('provider'))
    if job.get('base_url')!=endpoint:raise ValueError('Prepared endpoint differs from current provider')
    if not job.get('model'):raise ValueError('Missing prepared model')

def prepare(out,model,concurrency,provider='sili'):
    if out.exists():raise FileExistsError('Use a fresh output directory; no overwriting or resume mode.')
    endpoint=provider_endpoint(provider);jobs=[]
    if not model:raise ValueError('SILI_MODEL is missing')
    for seed in range(5):
        source=ROOT/f'results/manuscript_v2/v4_1_flash/runs/seed{seed}/evolutionary.json';data=read(source)
        initials=[]
        for condition in CONDITIONS:
            pop=make_population(data['config'],condition,model=model,concurrency=concurrency,provider=provider)
            initialize(pop,data);entry=evaluate(pop,0)
            assert entry==data['trajectory'][0],(condition,seed,'initial replay differs')
            folder=out/condition/f'seed{seed}';save(folder/'initial.json',entry)
            save(folder/'status.json',{'state':'prepared','generation':0})
            jobs.append({'condition':condition,'seed':seed,'source':source.relative_to(ROOT).as_posix(),
                         'source_sha256':sha(source),'folder':str(folder),'model':model,'concurrency':concurrency,
                         'provider':provider,'base_url':endpoint})
            initials.append(entry)
        assert initials[0]==initials[1]
        print(f'Exact initial replay verified for both arms: seed {seed}',flush=True)
    paths=list((ROOT/'experiments/v2_quantitative').glob('*.py'))+[Path(__file__).resolve(),ROOT/'experiments/evolution_log.py',ROOT/'experiments/config/load_env.py']
    manifest=[]
    for src in paths:
        rel=src.relative_to(ROOT);dst=out/'code_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        manifest.append({'path':rel.as_posix(),'sha256':sha(src)})
    save(out/'plan.json',{'created_utc':now(),'jobs':jobs,'code_manifest':manifest,'generations':100,
        'provider':provider,'base_url':endpoint,'model':model,'llm_concurrency':concurrency,'extra_body':{'enable_thinking':False},
        'injection_probability':.1,'frequency':'Once per generation for worst; conditional on each Fermi acceptance for gated.',
        'primary_endpoint':'Within-seed mean cooperation over evaluated generations 80–99.',
        'secondary_endpoints':['Injected birth counts and target fitness/cooperation','Lineage survival','Cooperation drops and recovery','Functional norm probes'],
        'randomness':'Shared deterministic evaluation seed per generation; population RNG for Fermi; independent keyed RNG for worst injection and random tie breaking.',
        'scope':'Joint change of gating, target and injection frequency; not an isolated target-only intervention.'})

def run_job(job):
    folder=Path(job['folder']);start=time.monotonic()
    with (folder/'run.log').open('w',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
        try:
            validate_job_provider(job)
            source=ROOT/job['source'];assert sha(source)==job['source_sha256'];data=read(source)
            pop=make_population(data['config'],job['condition'],model=job['model'],concurrency=job['concurrency'],provider=job['provider'],folder=folder,credentials=True)
            initialize(pop,data);trajectory=[evaluate(pop,0)];assert trajectory[0]==read(folder/'initial.json')
            for g in range(1,100):
                save(folder/'status.json',{'state':'generating','generation':g-1,'next_generation':g,'time_utc':now()})
                old=trajectory[-1]['population']
                pop._select_and_reproduce_by_method(next_gen=g)
                event=pop.evolution_rule.audit[-1]
                for injected in event['injections']:
                    injected['cooperation']=old[injected['index']]['cooperation_rate']
                pop.append('selection_events.jsonl',event)
                trajectory.append(evaluate(pop,g))
                result=build_evolution_results(trajectory=trajectory,final_population=trajectory[-1]['population'],
                    lineage_events=pop._lineage_events,config=pop._result_config(len(trajectory),
                        experiment={'condition':job['condition'],'injection_probability':.1,
                                    'source':job['source'],'source_sha256':job['source_sha256'],'planned_generations':100},
                        evaluation_seeds=pop.evaluation_seeds,llm_provider=job['provider'],api_base_url=job['base_url'],llm_extra_body=pop._llm_extra_body))
                write_evolution_json(folder/'trajectory_partial.json',result)
                save(folder/'status.json',{'state':'running','generation':g,'cooperation':trajectory[-1]['cooperation_rate_mean'],'time_utc':now()})
                print(f"Generation {g}: cooperation={trajectory[-1]['cooperation_rate_mean']:.5f}",flush=True)
            assert pop._fallback_init_count==pop._fallback_mutation_count==0
            write_evolution_json(folder/'evolutionary.json',result)
            status={'state':'completed','generation':99,'time_utc':now(),'elapsed_seconds':time.monotonic()-start}
        except Exception as exc:
            import traceback;traceback.print_exc()
            status={'state':'failed','error_type':type(exc).__name__,'time_utc':now()}
        save(folder/'status.json',status);return {**job,**status}

def group_jobs_by_seed(jobs):
    """A worker owns one seed and executes its two conditions sequentially."""
    grouped={}
    for job in jobs:grouped.setdefault(job['seed'],[]).append(job)
    if set(grouped)!=set(range(5)):raise ValueError('Expected exactly seeds 0 through 4')
    for seed_jobs in grouped.values():
        if sorted(j['condition'] for j in seed_jobs)!=sorted(CONDITIONS):
            raise ValueError('Each seed must have exactly one job per condition')
    return [sorted(grouped[seed],key=lambda j:CONDITIONS.index(j['condition'])) for seed in sorted(grouped)]

def run_seed_jobs(jobs):
    rows=[]
    for job in jobs:
        row=run_job(job);rows.append(row)
        print(row['condition'],row['seed'],row['state'],flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['prepare','run','status'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--model',default=None)
    parser.add_argument('--provider',choices=['sili'],default='sili')
    parser.add_argument('--llm-concurrency',type=int,default=DEFAULT_LLM_CONCURRENCY)
    parser.add_argument('--workers',type=int,default=DEFAULT_SEED_WORKERS,help='Concurrent seed workers; each executes both conditions sequentially')
    args=parser.parse_args();out=args.output.resolve()
    if args.mode=='prepare':return prepare(out,get_model(args.provider,args.model),args.llm_concurrency,args.provider)
    plan=read(out/'plan.json')
    if args.mode=='status':
        for job in plan['jobs']:print(job['condition'],job['seed'],read(Path(job['folder'])/'status.json'))
        return
    for job in plan['jobs']:validate_job_provider(job)
    if any(job['concurrency']!=args.llm_concurrency for job in plan['jobs']):
        raise ValueError('Requested LLM concurrency differs from the prepared experiment')
    seed_jobs=group_jobs_by_seed(plan['jobs'])
    if not get_api_key(args.provider):raise RuntimeError('sili credentials unavailable')
    for record in plan['code_manifest']:assert sha(ROOT/record['path'])==record['sha256'],record['path']
    if any(read(Path(j['folder'])/'status.json')['state']!='prepared' for j in plan['jobs']):
        raise RuntimeError('Only fresh prepared jobs may run; no resume or overwrite.')
    with (out/'RUN_STARTED').open('x',encoding='utf-8') as f:f.write(now())
    save(out/'execution.json',{'started_utc':now(),'seed_workers':min(args.workers,len(seed_jobs)),
        'llm_concurrency':args.llm_concurrency,'maximum_concurrent_llm_requests':min(args.workers,len(seed_jobs))*args.llm_concurrency,
        'schedule':'One worker per seed; conditions execute sequentially in CONDITIONS order.'})
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_seed_jobs,jobs) for jobs in seed_jobs]):
            results.extend(future.result());save(out/'batch_summary.json',results)
    if any(r['state']!='completed' for r in results):raise RuntimeError('Some runs failed; inspect status and logs.')

if __name__=='__main__':main()
