"""Scientific regression checks against Hilbe 2018 Table 1 and R=3 dynamics."""
import pytest
from experiments.v2_quantitative.baselines import get_baseline, LEADING_EIGHT, make_leading_eight
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.game import DonorGame

# Independent transcription by ROW from the published table (columns L1--L8).
PUBLISHED_ROWS = [
    (1, 'cooperate', 1, 'GGGGGGGG'),
    (1, 'cooperate', -1, 'GBGGBBGB'),
    (-1, 'cooperate', 1, 'GGGGGGGG'),
    (-1, 'cooperate', -1, 'GGGBGBBB'),
    (1, 'defect', 1, 'BBBBBBBB'),
    (1, 'defect', -1, 'GGGGGGGG'),
    (-1, 'defect', 1, 'BBBBBBBB'),
    (-1, 'defect', -1, 'BBGGGGBB'),
]

@pytest.mark.parametrize('actor,action,recipient,expected', PUBLISHED_ROWS)
def test_all_assessments_against_published_table(actor, action, recipient, expected):
    for i, name in enumerate(LEADING_EIGHT):
        ex = V2StrategyExecutor(get_baseline(name))
        # Interior scores distinguish increments from clamping.
        a, b = actor/3, recipient/3
        for observer in (-1., 0., 1.):
            for partner_action in ('cooperate', 'defect'):
                out = ex.observe(a, action, b, partner_action, observer)
                assert out == pytest.approx(a + (1/3 if expected[i]=='G' else -1/3))

@pytest.mark.parametrize('self_rep,recipient,expected', [
    (1,1,'CCCCCCCC'), (1,-1,'DDDDDDDD'),
    (-1,1,'CCCCCCCC'), (-1,-1,'CCDDDDDD'), (0,0,'CCCCCCCC')])
def test_published_action_table(self_rep, recipient, expected):
    for i,name in enumerate(LEADING_EIGHT):
        assert V2StrategyExecutor(get_baseline(name)).decide(self_rep,recipient) == (expected[i]=='C')

@pytest.mark.parametrize('name', LEADING_EIGHT)
@pytest.mark.parametrize('p', (1.,.5,.1))
def test_neutral_homogeneous_population_cooperates(name,p):
    code=get_baseline(name)
    agents=[QuantitativeAgent(i,code,V2StrategyExecutor(code)) for i in range(12)]
    game=DonorGame(12,observability='partial',observability_p=p,seed=87)
    game.setup_population(agents)
    for _ in range(20):
        game.play_round(); game.distribute_observations_and_self_judgments()
    assert all(a.cooperation_rate==1. for a in agents)

def test_discrimination_justified_defection_and_forgiveness():
    ss=V2StrategyExecutor(get_baseline('SS'))
    sj=V2StrategyExecutor(get_baseline('SJ'))
    assert ss.decide(-1,1) and sj.decide(-1,1)
    assert not ss.decide(1,-1) and not sj.decide(1,-1)
    assert ss.observe(0,'defect',-1,'cooperate',1)>0
    assert sj.observe(0,'defect',-1,'cooperate',1)>0
    assert ss.observe(0,'cooperate',-1,'defect',1)>0
    assert sj.observe(0,'cooperate',-1,'defect',1)<0
    rep=-1.
    for _ in range(3): rep=ss.observe(rep,'cooperate',1,'defect',-1)
    assert rep==0. and ss.decide(1,rep)

def test_actor_rating_not_observer_self_rating():
    ex=V2StrategyExecutor(get_baseline('L7'))
    assert ex.observe(1/3,'defect',-1,'cooperate',-1)==pytest.approx(2/3)
    assert ex.observe(-1/3,'defect',-1,'cooperate',1)==pytest.approx(-2/3)

def test_image_scoring_uses_recipient():
    ex=V2StrategyExecutor(get_baseline('IS'))
    assert ex.decide(-1,1) and not ex.decide(1,-1)

@pytest.mark.parametrize('seed', range(8))
def test_symmetric_cooperation_counts_both_players(seed):
    from experiments.v2_quantitative.evolution_architecture import ReputationPrisonersDilemmaScenario
    agents=[QuantitativeAgent(i,get_baseline(name),V2StrategyExecutor(get_baseline(name)))
            for i,name in enumerate(('ALLC','ALLD'))]
    scenario=ReputationPrisonersDilemmaScenario(population_size=2,benefit=2,cost=1,
        observability='full',observability_p=1,fitness_window_interactions=1,num_rounds_per_gen=1)
    result=scenario.evaluate(agents,generation_seed=seed,num_rounds=1)
    assert result.cooperation_rate_mean==0.5
    assert result.payoffs==(-1.,2.)

@pytest.mark.parametrize('name', ('IS+','SS+','SJ+'))
def test_unsupported_legacy_names_fail_explicitly(name):
    with pytest.raises(ValueError,match='mislabelled'): get_baseline(name)

@pytest.mark.parametrize('radius',(0,-1,True,1.5))
def test_invalid_resolution(radius):
    with pytest.raises(ValueError): make_leading_eight('L1',radius)
