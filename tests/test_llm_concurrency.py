"""LLM parallel-call concurrency tests (pytest).

Verifies that ``_parallel_llm_map`` caps the number of simultaneously
active jobs at ``llm_concurrency`` while preserving input order, and
that ``llm_concurrency == 1`` degrades to strictly serial execution.
"""
import threading
import time
from types import SimpleNamespace

import pytest

from experiments.v2_quantitative.population import V2EvolutionaryPopulation


@pytest.fixture
def population():
    pop = V2EvolutionaryPopulation.__new__(V2EvolutionaryPopulation)
    pop._llm_client = object()
    pop._llm_client_lock = threading.Lock()
    return pop


def test_parallel_map_caps_active_jobs_and_preserves_order(population):
    population.llm_concurrency = 3
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def job(index):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
        return index

    assert population._parallel_llm_map(job, range(8)) == list(range(8))
    assert state["maximum"] <= 3
    assert state["maximum"] > 1


def test_concurrency_one_is_serial(population):
    population.llm_concurrency = 1
    active = 0
    maximum = 0

    def job(index):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        active -= 1
        return index

    assert population._parallel_llm_map(job, [1, 2, 3]) == [1, 2, 3]
    assert maximum == 1


def test_valid_code_request_is_limited_to_three_api_calls(population, monkeypatch):
    calls = []
    population._call_llm = (
        lambda system, user, **kwargs: calls.append((system, user, kwargs))
        or "invalid code"
    )
    population._validate_code = lambda code: (_ for _ in ()).throw(
        ValueError("invalid")
    )
    monkeypatch.setattr(time, "sleep", lambda _delay: None)

    assert population._request_valid_code("prompt", "test") is None
    assert len(calls) == 3
    assert all(call[2]["max_retries"] == 1 for call in calls)


def test_call_llm_does_not_sleep_after_final_failure(population, monkeypatch):
    class Completions:
        @staticmethod
        def create(**_kwargs):
            raise RuntimeError("network failure")

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    population._llm_client = Client()
    population.llm_model = "test-model"
    population.mutation_temperature = 0.0
    population._llm_max_tokens = 10
    population._llm_extra_body = {}
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    assert population._call_llm("system", "user", max_retries=3) is None
    assert sleeps == [1, 2]


def test_fermi_fallback_state_is_committed_on_main_thread(population):
    class FixedRng:
        def __init__(self):
            self.random_values = iter([0.0, 0.5, 0.0, 0.5])

        @staticmethod
        def randrange(_limit):
            return 0

        @staticmethod
        def sample(population, count):
            return list(population)[:count]

        def random(self):
            return next(self.random_values)

    population.llm_concurrency = 2
    population.agents = [
        SimpleNamespace(agent_id=0, code="code-0", fitness=1.0),
        SimpleNamespace(agent_id=1, code="code-1", fitness=1.0),
    ]
    population.rng = FixedRng()
    population.fermi_beta = 5.0
    population.mutation_rate_on_adoption = 0.0
    population.updates_per_gen = 2
    population.agent_type = "agent-type1"
    population._fallback_mutation_count = 0
    population._slot_lineage = {0: 10, 1: 11}
    population._llm_small_mutate_code = lambda *_args: None
    population._make_agent = lambda code, agent_id: SimpleNamespace(
        agent_id=agent_id, code=code, fitness=0.0
    )
    lineage_events = []
    population._new_lineage = lambda *args: lineage_events.append(args)

    population._select_and_reproduce_fermi(next_gen=1)

    assert population._fallback_mutation_count == 2
    assert population.agents[0].code == "code-1"
    assert population.agents[1].code == "code-0"
    assert len(lineage_events) == 2


def test_updates_per_generation_cannot_exceed_population_size():
    with pytest.raises(ValueError, match="without-replacement"):
        V2EvolutionaryPopulation(population_size=3, updates_per_gen=4)
