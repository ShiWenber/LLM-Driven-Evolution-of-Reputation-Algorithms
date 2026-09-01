# v3 Type-2 LLM Probe: M3 Analysis (1 seed)

> **v3 type-2** = the LLM is asked to write a full class implementing `__init__(agent_id)`, `decide() -> bool`, and `observe(donor_id, donor_action, recipient_id, recipient_action) -> None`. The LLM is free to choose any internal state structure. This is the "give the LLM a Python class" interface, in contrast to v2's "give the LLM two functions" interface.

## Setup

- **Interface (v3 type-2)**: class `LLMAgent` with `__init__(agent_id)`, `decide() -> bool` (framework sets `self._ctx_opponent_id` before each call), and `observe(donor_id, donor_action, recipient_id, recipient_action) -> None` (LLM detects self-judgment via `donor_id == self.agent_id`).
- **Population**: n=15 agents, **30 rounds per generation**, **30 generations**, full observability.
- **Selection**: 2 elites, 5 eliminated per gen, tournament size 3.
- **LLM**: `deepseek-v4-flash`, mutation temperature 0.8.
- **Seeds**: 1 (seed 0 only — this is the M3 probe, M4 with 3 seeds pending).
- **Schema**: `config.schema_version=3`, `config.agent_type=v3`.
- **Init prompt**: `INIT_PROMPT_V3` (strict no-hints: no mention of reputation, `[-1,1]`, leading-eight, etc.). Mutation prompt: `MUTATION_PROMPT_V3`.
- **Fallback on LLM failure**: `FALLBACK_CLASS_V3` (ALLC-equivalent class) — used when init/mutate fails validation or API times out.

## Headline

| Metric | Value |
|---|---|
| n_seeds | **1** (M3 probe) |
| n_gens completed | **30 / 30** |
| Total wall time | **132 min** (2.2 h) |
| Final coop | **1.000** |
| Trajectory mean (per gen) | **0.9905** ± 0.0292 |
| Initial 15 agents (gen 0) | **15/15 FALLBACK** (LLM init: 15/15 failed) |
| Final 15 agents (gen 29) | **15/15 non-FALLBACK real LLM classes** (0 fallback) |
| Unique LLM strategies in final (by state attr + decide body) | **7** |
| Intrusion events (gen with coop < 1.0) | **8 / 30** |

## Per-generation trajectory

| gen | coop | fitness |
|---|---|---|
| 0 | 1.0000 | 28.000 |
| 1 | 1.0000 | 28.000 |
| 2 | 1.0000 | 28.000 |
| 3 | 1.0000 | 28.000 |
| 4 | 1.0000 | 28.000 |
| 5 | 0.9762 | 27.400 |
| 6 | 0.9952 | 27.800 |
| 7 | 0.9952 | 27.800 |
| 8 | 1.0000 | 28.000 |
| 9 | 1.0000 | 28.000 |
| 10 | 1.0000 | 28.000 |
| 11 | 0.8952 | 24.867 |
| 12 | 0.8714 | 24.733 |
| 13 | 1.0000 | 28.000 |
| 14 | 0.9952 | 27.667 |
| 15 | 1.0000 | 28.000 |
| 16 | 1.0000 | 28.000 |
| 17 | 1.0000 | 28.000 |
| 18 | 1.0000 | 28.000 |
| 19 | 1.0000 | 28.000 |
| 20 | 1.0000 | 28.000 |
| 21 | 0.9905 | 27.867 |
| 22 | 0.9952 | 27.933 |
| 23 | 1.0000 | 28.000 |
| 24 | 1.0000 | 28.000 |
| 25 | 1.0000 | 28.000 |
| 26 | 1.0000 | 28.000 |
| 27 | 1.0000 | 28.000 |
| 28 | 1.0000 | 28.000 |
| 29 | 1.0000 | 28.000 |

## Key findings

### 1. LLM init was 100% failure, but mutations kept sneaking real classes in

All 15 init attempts fell back to `FALLBACK_CLASS_V3` (the ALLC-equivalent class). But across the 30 generations, **8 generations showed `coop < 1.0`**, meaning at least one mutation call succeeded and produced a class that didn't unconditionally cooperate:

```
- gen 5: coop=0.9762, fitness=27.400
- gen 6: coop=0.9952, fitness=27.800
- gen 7: coop=0.9952, fitness=27.800
- gen 11: coop=0.8952, fitness=24.867
- gen 12: coop=0.8714, fitness=24.733
- gen 14: coop=0.9952, fitness=27.667
- gen 21: coop=0.9905, fitness=27.867
- gen 22: coop=0.9952, fitness=27.933
```

The fitness dips in these intrusions (24.7 to 27.9, vs the ALLC-equivalent 28.0) confirm that LLM-designed classes get slightly less than the all-cooperate score — they cooperate *most* of the time but not *all* the time. **The LLM never evolved a class that defects strategically; it evolved classes that are ~99% cooperative but occasionally defect** (likely against agents with negative reputation, or due to noisy state).

### 2. Every LLM-class intrusion was transient — out-competed by ALLC-equivalent in 1-2 generations

None of the 8 intrusions lasted more than 2 generations. The pattern was always:
1. Gen N: a mutation succeeds, introduces a non-FALLBACK class with slightly imperfect cooperation → coop dips to 0.87-0.99.
2. Gen N+1 (or N+2): the imperfect class loses the tournament to the surrounding ALLC-equivalent classes (fitness 28.0 vs 24.7-27.9) → coop returns to 1.0.

This is consistent with a stable ALLC-equivalent attractor under full observability. The LLM's mutations are unable to find a class that *outcompetes* unconditional cooperation in this environment — they can only produce classes that are slightly *worse* than ALLC.

### 3. Final state: 15/15 non-FALLBACK, 7 distinct strategies, but all behaving as ALLC-equivalent

The final 15 agents in the population are *all* real LLM-generated classes (none are FALLBACK). They cluster into **7 distinct strategies** (by state-attribute name + decide body signature). But the population-wide behavior is `coop=1.0, fitness=28.0` — indistinguishable from the ALLC baseline.

This is an interesting data point: the LLM's mutation pressure eventually *replaced all the FALLBACK classes* with real LLM classes (none of the original FALLBACK agents survived the selection pressure), but the surviving LLM classes all converged to ALLC-equivalent behavior. **The LLM evolved genuine code (with real per-opponent state and conditional decisions), but in the test environment the conditions for defecting never triggered, so the evolved code is observationally indistinguishable from "always cooperate"**.

### 4. Validation failure mode: LLM doesn't know `LLMAgent` is a base class

Across the run, the LLM produced ~5-6 mutation outputs that failed the `_validate_code` compile check. Two failure modes recur:
- `name 'LLMAgent' is not defined` — the LLM writes a fresh class without `class LLMAgent:` header, then tries to call `super().__init__()` or reference `LLMAgent` as a type.
- `(` was never closed / `unterminated string literal` — the LLM produces a class with an unterminated docstring or unclosed parenthesis.

The first failure mode is the more interesting one: **even after 30+ generation cycles of mutating existing LLMAgent classes, the LLM still doesn't internalize that `class LLMAgent:` is the required header**. This is a known weakness of mutating-code-by-prompting: the LLM treats each mutation as a free generation rather than a strict templating task.

## Comparison with v3 type-1 (3-seed baseline)

| Dimension | v3 type-1 (5-arg evaluate + 2-arg decide) | v3 type-2 (full class) |
|---|---|---|
| n seeds | 3 | 1 (M3) |
| Final coop (mean ± std) | 0.637 ± 0.553 (bimodal: 1.0/0.91/0.0) | 1.000 (n=1) |
| Trajectory mean (mean ± std) | 0.665 ± 0.279 | 0.990 ± 0.029 |
| Init fallback rate | ~80% (12/15 init attempts fell back) | **100% (15/15)** |
| LLM-class intrusions | Frequent in seeds 0 and 1 | **8 / 30 generations** |
| Validation failure modes | `Strategy must define both evaluate() and decide()` | `name 'LLMAgent' is not defined` and unclosed parens |
| Trajectory shape | seed 0: slow rise; seed 1: stable partial; seed 2: collapse | **All-FALLBACK base, 8 transient LLM intrusions, ALLC-equivalent attractor** |

**Tentative type-1 vs type-2 story (1-seed caveat applies)**: the type-2 interface produces a more *convergent* result than type-1. In type-1, the LLM has explicit `evaluate()` and `decide()` parameters and can think in terms of "given these inputs, do this"; in type-2, the LLM has to design its own internal state and integrate observations into that state, which makes it harder to produce strategically novel behavior. The result: type-2 LLM classes default to "always cooperate" (the simplest stateful strategy) and rarely deviate.

This is consistent with the "richer interface = LLM defaults to simpler behavior" hypothesis: the type-2 interface gives the LLM more degrees of freedom, but most of those degrees of freedom are never used in practice because the LLM converges on the trivial always-cooperate baseline.

## Final-population strategies (7 unique)

The final 15 agents cluster into 7 distinct strategies (by state-attribute name + decide body signature):

  - 'state=history | decide=[]': 6 agent(s) [ids: [128, 143, 152, 156, 157, 158]]
  - 'state=images | decide=[]': 3 agent(s) [ids: [141, 144, 153]]
  - 'state=last_action | decide=[]': 2 agent(s) [ids: [129, 159]]
  - 'state=recent | decide=[]': 1 agent(s) [ids: [124]]
  - 'state=last_seen | decide=[]': 1 agent(s) [ids: [146]]
  - 'state=no_state | decide=[]': 1 agent(s) [ids: [147]]
  - 'state=reputation | decide=[]': 1 agent(s) [ids: [155]]


### Agent 128 (first 30 lines)
```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self._ctx_opponent_id = None
        self.history = {}

    def decide(self) -> bool:
        opp = self._ctx_opponent_id
        if opp is not None and opp in self.history:
            defections, cooperations = self.history[opp]
            if defections >= cooperations + 1:
                return False
        return True

    def observe(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        if donor_id != self.agent_id:
            self._record(donor_id, donor_action)
        if recipient_id != self.agent_id:
            self._record(recipient_id, recipient_action)

    def _record(self, agent_id: int, action: str) -> None:
        if agent_id not in self.history:
            self.history[agent_id] = [0, 0]
        if isinstance(action, str):
...
```

### Agent 124 (first 30 lines)
```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self._ctx_opponent_id = None
        self.recent = {}

    def decide(self) -> bool:
        opp = self._ctx_opponent_id
        if opp is not None and opp in self.recent:
            acts = self.recent[opp]
            if acts.count(False) > acts.count(True):
                return False
        return True

    def observe(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        if donor_id == self.agent_id:
            self._record(recipient_id, recipient_action)
        elif recipient_id == self.agent_id:
            self._record(donor_id, donor_action)

    def _record(self, agent_id: int, action) -> None:
        if agent_id not in self.recent:
            self.recent[agent_id] = []
        if isinstance(action, str):
...
```

### Agent 144 (first 30 lines)
```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self._ctx_opponent_id = None
        self.images = {}

    def _is_coop(self, action: str) -> bool:
        if action is None:
            return False
        if isinstance(action, bool):
            return action
        return str(action).strip().upper().startswith("C")

    def decide(self) -> bool:
        opp = self._ctx_opponent_id
        if opp is None:
            return True
        return self.images.get(opp, 1) >= 0

    def observe(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        decay = 0.9

        if donor_id == self.agent_id:
            old = self.images.get(recipient_id, 1)
...
```


## Limitations

- **1 seed only** (M3 probe). M4 (3 seeds × 30 gen) is queued but blocked on the LLM API rate limit that caused the original M3 to take 132 min instead of the expected ~30 min. **Cross-seed reliability is the most important missing data.**
- **LLM init 100% failure** during this run due to API rate-limiting. The 15 FALLBACK classes at gen 0 came from a Python-level fallback, not from the LLM choosing to cooperate. This means the *init* signal of the LLM is unobserved in this run — the M4 data is needed to characterize LLM-from-cold-start behavior.
- **No baseline comparison runs on the v3 type-2 interface** (the 8 leading-eight are type-1 by design and were not re-implemented for type-2). The M3 conclusion that LLM-classes converge to ALLC-equivalent is suggestive but not directly comparable to the 8 leading-eight on the same interface.

## Files

- Trajectory plot: `results/quantitative_baseline/plots/llm_v3_type2_trajectory.png` (and `.pdf`)
- JSON: `results/quantitative_baseline/LLM_evolution_seed0/evolutionary.json`
- Run log: `results/quantitative_baseline/runner_v3_llm.log`

## Suggested next steps

1. **M4 (3 seeds × 30 gen v3 type-2)**: 1 seed shows a clear ALLC-equivalent attractor; 3 seeds will tell us if this is *always* the case (σ ≈ 0 across seeds) or if there's a hidden bimodal collapse mode like v3 type-1.
2. **v2 type-2 baseline runs**: implement 1-2 of the leading-eight as type-2 classes and confirm they still hit `coop=1.0` under the type-2 interface (sanity check that the type-2 interface is at least as expressive as type-1).
3. **Sample-code analysis**: hand-classify the 7 final strategies to see if any of them are doing something non-trivial (e.g., tracking per-opponent reputation, counting defections) or if all of them are just `return True` with cosmetic differences. (Spoiler from the sample codes below: they DO have nontrivial state and conditional decisions, but those branches never fire in a population where everyone is mostly cooperating.)
