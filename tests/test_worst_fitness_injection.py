import random
import pytest
from experiments.run_worst_fitness_injection import InjectionRule
from experiments.v2_quantitative.evolution_architecture import AgentSnapshot,FermiEvolutionRule

def population(fitness=(1.,-2.,3.,0.)):
    return tuple(AgentSnapshot(i,'program',f,100+i) for i,f in enumerate(fitness))

def plan(rule,pop=None,rng=None):
    pop=population() if pop is None else pop
    return rule.plan(pop,rng=random.Random(9) if rng is None else rng,next_gen=1,lineage_by_agent_id={a.agent_id:a.lineage_id for a in pop})

def test_original_arm_preserves_fermi_planner():
    pop=population();mapping={a.agent_id:a.lineage_id for a in pop}
    expected=FermiEvolutionRule(beta=5,mutation_rate=.1,updates_per_gen=4,init_source='llm').plan(pop,rng=random.Random(9),next_gen=1,lineage_by_agent_id=mapping)
    assert plan(InjectionRule('fermi_gated',seed=2,updates_per_gen=4))==expected

def test_probability_zero_has_only_fermi_rewrites():
    p=plan(InjectionRule('worst_per_generation',seed=2,probability=0,updates_per_gen=4))
    assert all(j.operator=='llm_mutate' for j in p.jobs)

def test_minimum_is_replaced_without_parent_and_no_slot_collision():
    rule=InjectionRule('worst_per_generation',seed=2,probability=1,updates_per_gen=4)
    p=plan(rule);fresh=[j for j in p.jobs if j.operator=='llm_init']
    assert len(fresh)==1 and fresh[0].output_index==1
    assert fresh[0].parent_id is fresh[0].parent_code is fresh[0].parent_fitness is None
    assert fresh[0].parent_lineage_id is None
    indices=[j.output_index for j in p.jobs]+[r.output_index for r in p.retained]
    assert sorted(indices)==[0,1,2,3]
    assert [j.ordinal for j in p.jobs]==list(range(len(p.jobs)))
    assert rule.audit[-1]['injections'][0]['fitness']==-2

class RejectAll:
    def sample(self,values,k):return list(values)[:k]
    def randrange(self,n):return 0
    def random(self):return 1.

def test_injection_does_not_require_fermi_acceptance():
    rule=InjectionRule('worst_per_generation',seed=1,probability=1,updates_per_gen=4)
    p=plan(rule,rng=RejectAll())
    assert rule.audit[-1]['fermi_accepted']==0
    assert len(p.jobs)==1 and p.jobs[0].operator=='llm_init'

def test_injection_overrides_a_previously_accepted_rewrite():
    class AcceptAll(RejectAll):
        def random(self):return 0.
    rule=InjectionRule('worst_per_generation',seed=1,probability=1,updates_per_gen=4)
    p=plan(rule,rng=AcceptAll())
    assert rule.audit[-1]['fermi_accepted']==4
    assert rule.audit[-1]['overrode_fermi_job'] is True
    assert len(p.jobs)==4 and len({j.output_index for j in p.jobs})==4
    assert sum(j.operator=='llm_mutate' for j in p.jobs)==3

def test_random_tie_breaking_stays_within_minimum_set():
    chosen=set()
    for seed in range(30):
        p=plan(InjectionRule('worst_per_generation',seed=seed,probability=1,updates_per_gen=4),population((-1.,4.,-1.,5.)))
        chosen.add(next(j.output_index for j in p.jobs if j.operator=='llm_init'))
    assert chosen=={0,2}

def test_trigger_and_ties_independent_of_fermi_rng():
    for seed in range(20):
        rules=[InjectionRule('worst_per_generation',seed=seed,updates_per_gen=4) for _ in range(2)]
        plan(rules[0],rng=random.Random(3));plan(rules[1],rng=random.Random(99))
        assert rules[0].audit[-1]['injection_draw']==rules[1].audit[-1]['injection_draw']
        assert rules[0].audit[-1]['injection_target_index']==rules[1].audit[-1]['injection_target_index']

@pytest.mark.parametrize('probability',[-.1,1.1])
def test_bad_probability_rejected(probability):
    with pytest.raises(ValueError):InjectionRule('worst_per_generation',seed=0,probability=probability)

def test_provider_routing_and_payload(monkeypatch):
    from experiments import run_worst_fitness_injection as runner
    requested=[]
    def endpoint(provider):
        requested.append(provider)
        return 'https://api.siliconflow.cn/v1'
    monkeypatch.setattr(runner,'get_base_url',endpoint)
    monkeypatch.setattr(runner,'get_api_key',lambda provider: requested.append(provider) or 'test-key')
    source=runner.read(runner.ROOT/'results/manuscript_v2/v4_1_flash/runs/seed0/evolutionary.json')
    assert runner.DEFAULT_LLM_CONCURRENCY==4
    pop=runner.make_population(source['config'],'fermi_gated',model='configured-sili-model',concurrency=runner.DEFAULT_LLM_CONCURRENCY,credentials=True)
    assert pop.llm_concurrency==4
    assert requested and set(requested)=={'sili'}
    assert pop.llm_provider=='sili' and pop.llm_model=='configured-sili-model'
    assert pop.api_base_url=='https://api.siliconflow.cn/v1'
    assert pop._llm_extra_body=={'enable_thinking':False}

@pytest.mark.parametrize('job',[
    {'provider':'deepseek','base_url':'https://api.deepseek.com','model':'deepseek-flash'},
    {'model':'deepseek-flash'},
    {'provider':'sili','base_url':'https://api.deepseek.com','model':'some-model'},
])
def test_wrong_or_unrecorded_provider_is_rejected(monkeypatch,job):
    from experiments import run_worst_fitness_injection as runner
    monkeypatch.setattr(runner,'get_base_url',lambda _: 'https://api.siliconflow.cn/v1')
    with pytest.raises(ValueError):runner.validate_job_provider(job)


def test_five_distinct_seed_workers_with_sequential_conditions(monkeypatch):
    from experiments import run_worst_fitness_injection as runner
    assert runner.DEFAULT_SEED_WORKERS==5
    jobs=[{'seed':seed,'condition':c} for seed in range(5) for c in runner.CONDITIONS]
    groups=runner.group_jobs_by_seed(list(reversed(jobs)))
    assert len(groups)==5
    assert [g[0]['seed'] for g in groups]==list(range(5))
    for seed,group in enumerate(groups):
        assert all(j['seed']==seed for j in group)
        assert [j['condition'] for j in group]==list(runner.CONDITIONS)
    calls=[]
    def fake_run(job):
        calls.append((job['seed'],job['condition']))
        return {**job,'state':'completed'}
    monkeypatch.setattr(runner,'run_job',fake_run)
    assert len(runner.run_seed_jobs(groups[3]))==2
    assert calls==[(3,c) for c in runner.CONDITIONS]


def test_seed_groups_reject_missing_or_duplicate_conditions():
    from experiments import run_worst_fitness_injection as runner
    jobs=[{'seed':seed,'condition':c} for seed in range(5) for c in runner.CONDITIONS]
    with pytest.raises(ValueError):runner.group_jobs_by_seed(jobs[:-1])
    with pytest.raises(ValueError):runner.group_jobs_by_seed(jobs+[jobs[0]])
