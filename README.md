# LLM-Driven Evolution of Reputation Algorithms

![LLM-evolved Evolution of Reputation Algorithms](README.assets/project_overall_architecture.png)

![Visualization of LLM-evolved strategies](README.assets/visual_clustering_analysis_workflow-v4.png)

## Demo

### PCA strategy-space evolution

| `agent-type1` | `agent-type2` |
| --- | --- |
| ![agent-type1 seed4 PCA strategy evolution in the joint six-seed space](README.assets/agent-type1_joint6_seed4_pca_evolution.gif) | ![agent-type2 seed0 PCA strategy evolution](README.assets/agent-type2_seed0_pca_evolution.gif) |

For `agent-type1`, a single embedding, K-means model, and PCA projection are fit
jointly on every strategy from seeds 0–5. Only seed 4 is rendered because it has
the highest final cooperation rate (`1.000`) among the six runs. Its animation
therefore moves inside the common six-seed coordinate system, and its colors
use the common `K=19` cluster labels. The `agent-type2` panel continues to show
its seed-0 analysis.

A compact research showcase of how LLM-evolved strategies behave in reputation-driven cooperation games.

The core result is that cooperation can emerge from LLM-generated strategies, but the outcome is strongly seed-dependent and sensitive to the implementation details of the reputation logic. Classical indirect-reciprocity norms remain more stable, while learned strategies are promising but less robust.

### 1) Cooperation evolution: agent-type1 vs. agent-type2

![Cooperation evolution comparison between agent-type1 and agent-type2](README.assets/g100_3seed_1000inter.png)

The figure uses all six updated `agent-type1` runs (seeds 0–5) and the three
available `agent-type2` runs (seeds 0–2). Both panels use 100 generations,
1,000 target interactions per generation, a population of 16, and
generation-level state reset. Thin lines show individual seeds, thick lines
show the mean for each agent type, and shaded bands show one standard
deviation. The updated `agent-type1` prompt and interface can produce high
cooperation, but its runs remain substantially more variable than the three
`agent-type2` runs.

### 2) agent-type1 vs. agent-type2: different strategy interfaces

The two agent types solve the same reputation game but expose different strategy
interfaces to the LLM:

| Agent type | Generated strategy | State and memory | Search space |
| --- | --- | --- | --- |
| `agent-type1` | Two functions: `observe(...)` and `decide(...)` | Updates donor and recipient reputations through a fixed functional interface | Narrower and more explicitly guided |
| `agent-type2` | A complete `LLMAgent` class with `__init__`, `decide()`, and `observe(...)` | May maintain internal state and update it after interactions | Richer and less constrained |

This distinction matters when reading the results: differences between the two
agent types reflect a change in the representation available to evolution, not
just a new experiment revision. The corrected `agent-type2` dynamics are not
uniformly better across seeds. Some seeds improve sharply, while others enter a
low-cooperation basin, producing a more bimodal evolutionary landscape.

### 3) Compared against the leading-eight norms

The canonical leading-eight strategies stay near cooperation rate 1.0 across the full horizon. The LLM trajectories are less reliable and more variable, which makes the empirical message precise: learned systems can discover effective cooperation, but they do not yet match the robustness of classical indirect-reciprocity norms.

### 4) Population structure in embedding space

| `agent-type1` | `agent-type2` |
| --- | --- |
| ![agent-type1 seed4 strategy-cluster composition from six-seed joint clustering](README.assets/agent-type1_joint6_seed4_cluster_composition.png) | ![agent-type2 seed0 strategy-cluster composition](README.assets/agent-type2_seed0_cluster_composition.png) |

The `agent-type1` panel shows seed 4 using labels learned jointly from all six
agent-type1 runs; it is not a seed-4-only clustering. The `agent-type2` panel
shows seed 0 in its own shared analysis. The agent-type1 population moves among
a larger set of strategy families, whereas agent-type2 rapidly becomes
dominated by one of two broad families.

This is the core story of the project: the environment supports cooperation, the LLM can discover cooperative policies, the corrected implementation changes the attractor structure, and the classical norms remain the reliability benchmark.

### 5) Final-survivor ancestry: agent-type1 vs. agent-type2

| `agent-type1` | `agent-type2` |
| --- | --- |
| ![agent-type1 seed4 final-survivor ancestry tree using six-seed joint clusters](README.assets/agent-type1_joint6_seed4_survivor_tree.png) | ![agent-type2 seed0 final-survivor ancestry tree](README.assets/agent-type2_seed0_survivor_tree.png) |

The agent-type1 tree is the highest-final-cooperation run (seed 4), colored by
the six-seed joint clusters; the agent-type2 tree remains seed 0. Both show only
ancestry paths leading to final survivors. Squares mark roots and triangles
mark final survivors.

### 6) Lineage survival intervals

| `agent-type1` | `agent-type2` |
| --- | --- |
| ![agent-type1 seed4 lineage survival intervals using six-seed joint clusters](README.assets/agent-type1_joint6_seed4_lineage_survival.png) | ![agent-type2 seed0 lineage survival intervals](README.assets/agent-type2_seed0_lineage_survival.png) |

Each horizontal interval runs from a collapsed lineage's birth to its last
appearance. Agent-type1 uses the common six-seed analysis (`K=19`) and displays
seed 4; agent-type2 uses its seed-0 analysis (`K=2`).

### 7) Representative final survivor from each dominant lineage family

This section documents the representatives used by the bidirectional invasion
experiment: seed 4 for `agent-type1` and seed 0 for `agent-type2`, matching the
representative runs used by the visual analyses above.

The selection rule is identical for both runs: group final agents by root
lineage, choose the root family with the most final members, then choose the
highest-fitness member of that family (breaking ties by agent ID).

| Agent type | Dominant root lineage | Final members | Representative | Fitness | Behavioral summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `agent-type1` | 14 | 15/16 | agent 0 | 25.0 | Uses context-sensitive bounded reputation updates and cooperates with neutral-to-good opponents |
| `agent-type2` | 11 | 13/16 | agent 1 | 26.0 | Maintains opponent histories, estimates conditional cooperation, and adapts trust, generosity, exploration, and risk tolerance |

#### agent-type1 representative (complete strategy)

Source: agent 0 in the
[`agent-type1` seed-4 evolution record](results/quantitative_baseline/LLM_agent-type1_fermi_z_v3_g100_1000inter_N16_genreset_seed4/evolutionary.json).

```python
def observe(
    donor_reputation: float,
    donor_action: str,
    recipient_reputation: float,
    recipient_action: str,
    my_reputation: float,
) -> tuple[float, float]:
    if donor_action == "cooperate":
        weight = 0.4 if recipient_reputation >= 0 else 0.2
        donor_new = donor_reputation + weight * (1 - abs(donor_reputation))
        if recipient_action == "defect":
            donor_new -= 0.3
    else:
        donor_new = donor_reputation - 0.5 if recipient_reputation >= 0 else donor_reputation + 0.3 * (1 - abs(donor_reputation))
        if recipient_action == "cooperate":
            donor_new -= 0.4

    if recipient_action == "cooperate":
        weight = 0.4 if donor_reputation >= 0 else 0.2
        recipient_new = recipient_reputation + weight * (1 - abs(recipient_reputation))
        if donor_action == "defect":
            recipient_new -= 0.3
    else:
        recipient_new = recipient_reputation - 0.5 if donor_reputation >= 0 else recipient_reputation + 0.3 * (1 - abs(recipient_reputation))
        if donor_action == "cooperate":
            recipient_new -= 0.4

    return (
        max(-1.0, min(1.0, donor_new)),
        max(-1.0, min(1.0, recipient_new)),
    )

def decide(my_reputation: float, opponent_reputation: float) -> bool:
    if opponent_reputation >= 0.3 and my_reputation > -0.35:
        return True
    if opponent_reputation < -0.2:
        return False
    return opponent_reputation > -0.1
```

#### agent-type2 representative (decision core)

The complete evolved class is stored as agent 1 in the
[`agent-type2` seed-0 evolution record](results/quantitative_baseline/LLM_v3_fermi_z_v3_g100_1000inter_N16_genreset_seed0/evolutionary.json).
Its `observe(...)` method maintains the histories and adaptive statistics
consumed by the following abridged decision core:

```python
def decide(self) -> bool:
    # Abridged for readability; see the linked evolution record for exact code.
    opponent = self._ctx_opponent_id
    if opponent is None:
        return random.random() < 0.5

    history = self.opponent_history.get(opponent, [])
    if history:
        observed_cooperation = sum(a == "cooperate" for a in history) / len(history)
        self.trust = 0.7 * observed_cooperation + 0.3 * self.reputation
    else:
        self.trust = self.reputation

    threshold = (
        0.5 - self.generosity
        + self.round_num * 0.001
        + self.strategy_adjustment
    )
    if random.random() < self.exploration_rate:
        return random.random() < 0.5
    if self.trust > threshold:
        return True
    return random.random() < 0.2 * self.risk_tolerance
```

The contrast is structural as well as behavioral: the updated `agent-type1`
expresses observation and action as two stateless functions, while
`agent-type2` combines opponent modeling with persistent state and online
adaptation.

### 8) Best evolved strategies vs. the Leading Eight: bidirectional invasion

We tested the two representative strategies selected above against all eight
canonical norms (`IS`, `SS`, `SJ`, `SC`, `SH`, `IS+`, `SS+`, and `SJ+`). The
experiment uses the project's existing agent executors, private-reputation game
logic, and fixed-strategy Fermi imitation module rather than a separate game
implementation.

For each agent type and norm, the experiment starts with one invader in a
population of 15 and tests both directions. Bar length is the number of
fixations across seeds `0`, `1`, and `2`; leftward bars show a Leading Eight
norm invading the evolved strategy, and rightward bars show the evolved
strategy invading the norm.

![Bidirectional invasion results for the best agent-type1 and agent-type2 strategies against the Leading Eight](README.assets/best_vs_leading_eight_invasion.png)

The formal batch contains 96 runs: two agent types, eight norms, two invasion
directions, and three seeds. Every run uses 50 generations, 1,000 interactions
per generation, an 800-interaction burn-in, a 200-interaction fitness window,
`b=2`, `c=1`, full observation, observer-private reputations, `beta=5`, and 15
Fermi updates per generation. As in the population module, every slot is
re-instantiated after selection: its stable ID and selected strategy type are
preserved, while private reputations and strategy-internal state reset.

The updated result is strongly asymmetric for `agent-type2`: it fixed in two
of three runs against `IS`, `SS`, `SC`, `SH`, `IS+`, and `SS+`, and none of
those six norms fixed in the reverse direction. `SJ` and `SJ+` show the opposite
pattern, fixing in two of three runs against `agent-type2` while resisting all
three outward invasion attempts. The seed-4 `agent-type1` representative fixed
in one of three runs against every Leading Eight norm, and every norm likewise
fixed in one of three reverse-direction runs; it shows no directional advantage
in this three-seed batch.
Because each cell contains only three seeds, these fractions are descriptive
results rather than precise estimates of fixation probability.

Run or resume this single-invader batch with:

```powershell
uv run run-best-leading-eight-invasion --invader-counts 1 --workers 8
```

The runner also supports an initial-frequency sweep with `n=1..14`. Formal
outputs remain available for reproducibility. Regenerate the README figure with
`uv run plot-best-leading-eight-invasion`; the experiment runner is
`experiments.analysis.invasion.run_best_leading_eight_invasion`.

---

## Project overview

This repository contains the experimental code and paper artifacts for
studying cooperation in LLM-coded populations with private reputation stores,
quantitative assessment, and controlled observability.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A DeepSeek API key for LLM-generating experiments

The API configuration is loaded from a root-level `.env` file. Start from
`.env.example` and keep `.env` local; it is ignored by Git.

## Install

From the repository root:

```powershell
uv sync
```

The project dependencies and executable entry points are defined in
`pyproject.toml`, with versions locked in `uv.lock`.

## Project layout

```text
.
├── experiments/
│   ├── agents/              # Agent interfaces and LLM prompts
│   ├── analysis/            # Analysis and visualization modules
│   ├── config/              # YAML settings and environment loading
│   ├── evolution/           # Evolutionary population code
│   ├── game/                # Quantitative reputation games
│   ├── sandbox/             # Strategy validation and execution
│   ├── tools/               # Re-run orchestration
│   ├── run_fermi_v3.py      # Legacy-named CLI for agent-type2 evolution
│   └── v2_quantitative/     # Quantitative-assessment engine
├── results/                 # Experiment outputs and figures
├── PAPER_DRAFT.md
├── pyproject.toml
└── uv.lock
```

The Agent 2 invasion runner and dashboard are maintained under
`experiments.analysis.invasion` and `experiments.analysis`, respectively.
Analysis commands resolve project-relative result directories at runtime;
paths can be overridden with explicit CLI options.

The matched `agent-type1` versus `agent-type2` evolution plot is provided by
`experiments.analysis.plot_evolution_curves` and writes PNG/PDF figures under
`results/quantitative_baseline/plots/` by default. To refresh the figure used
at the top of this README, run:

```powershell
uv run plot-evolution-curves --output README.assets/g100_3seed_1000inter.png --no-pdf
```

## Common commands

The configured project scripts can be run with `uv run`:

```powershell
# Main legacy donor-game CLI
uv run llm-reputation --help

# Audit or re-run the documented experiment batches
uv run rerun-experiments --audit
uv run rerun-experiments --experiments 2 --seeds 0 1 2

# Agent 2 invasion experiment (336 fixed-strategy runs by default)
uv run run-agent2-invasion --help

# Best agent-type1 and agent-type2 representatives vs. all Leading Eight norms
uv run run-best-leading-eight-invasion --invader-counts 1 --workers 8
uv run plot-best-leading-eight-invasion

# Generate the invasion dashboard
uv run plot-agent2-invasion

# Plot cooperation evolution curves
uv run plot-evolution-curves

# Generate legacy paper figures and summary tables
uv run make-figures --help

# Plot lineage survival and final-survivor trees
uv run plot-lineage --help

# agent-type2 Fermi-style LLM evolution experiment (legacy module name; see below)
uv run python -m experiments.run_fermi_v3 --help
```

The same commands are available through module execution, for example:

```powershell
uv run python -m experiments.analysis.plot_agent2_schmid_invasion
uv run python -m experiments.analysis.plot_evolution_curves
uv run python -m experiments.analysis.make_figures --help
uv run python -m experiments.analysis.plot_lineage --help
uv run python -m experiments.run_fermi_v3 --dry-run --seed 0
```

`uv run` requires a command or module; there is no single implicit default
task for this project.

## Historical Agent 2 / Schmid L1-L2-L7-L8 invasion experiment

The current invasion study uses the evolved `agent_id=2` strategy from
generation 99 of the seed-2 production run against the four norms identified
by Schmid et al. (2023) as robust under quantitative assessment with private
and noisy information: `L1`, `L2`, `L7`, and `L8`.

The runner uses:

- `N=15`, 50 generations, and seeds `0, 1, 2`;
- exactly 1,000 pair interactions per generation;
- the first 800 interactions as burn-in;
- the final 200 interactions for selection fitness;
- benefit `b=2`, cost `c=1`, full observation, and observer-private reputations;
- synchronous fixed-strategy Fermi imitation with `beta=5` and 15 updates per generation;
- both directions: Agent 2 invading a norm and a norm invading Agent 2;
- initial invader counts `n=1..14`.

Run or resume the experiment with:

```powershell
uv run run-agent2-invasion
```

Completed JSON files are cached, so re-running resumes without repeating
finished cells. Use `--smoke` for a small validation run, or `--help` for
selection of norms, directions, seeds, and output paths.

Results are written under:

```text
results/quantitative_baseline/invasion/
└── agent2_schmid_L1_L2_L7_L8_mainmatched/
    ├── summary.json
    ├── agent2_schmid_bidirectional_invasion_dashboard.png
    ├── agent2_invades_norm/<norm>/n<k>_seed<s>/invasion.json
    └── norm_invades_agent2/<norm>/n<k>_seed<s>/invasion.json
```

Generate the dashboard again with:

```powershell
uv run plot-agent2-invasion

# Equivalent module form:
uv run python -m experiments.analysis.invasion.run_agent2_schmid_invasion --help
```

The plotting module validates that all 336 formal result files exist, that
every trajectory uses the `1000/800/200` interaction split, and that the four
norms have matching population-frequency trajectories before producing the
figure.

## agent-type2 Fermi-style LLM evolution experiment

The entry point `experiments/run_fermi_v3.py` uses a legacy filename and is a
command-line launcher for the `agent-type2` (full `LLMAgent` class) population evolution with the Fermi imitation
selection scheme. It is a thin CLI wrapper over
`experiments.v2_quantitative.population.V2EvolutionaryPopulation` and mirrors
the production-run script `_run_fermi_3seed_100gen_v3.py` at the repo root.

Run a single seed:

```powershell
uv run python -m experiments.run_fermi_v3 --seed 0 --gens 100 --target-interactions 1000
```

Run multiple seeds sequentially (default is seeds `0 1 2`):

```powershell
uv run python -m experiments.run_fermi_v3 --seeds 0 1 2
```

Preview the run plan without touching the API:

```powershell
uv run python -m experiments.run_fermi_v3 --dry-run --seeds 0 1 2
```

Common options:

| Option | Default | Description |
| --- | --- | --- |
| `--seed N` / `--seeds N...` | `[0, 1, 2]` | Seed(s) to run; `--seed` overrides `--seeds` |
| `--gens N` | `100` | Number of generations |
| `--target-interactions N` | `1000` | Target PD interactions per generation |
| `--population-size N` | `15` | Population size |
| `--updates-per-gen N` | `15` | Fermi imitation updates per generation |
| `--fermi-beta F` | `5.0` | Fermi selection strength |
| `--mutation-rate F` | `0.1` | Mutation probability on adoption (`mu`) |
| `--mutation-temperature F` | `0.8` | LLM mutation temperature |
| `--benefit F` / `--cost F` | `2.0` / `1.0` | PD payoffs |
| `--observability S` | `full` | Observability mode |
| `--provider S` | `deepseek` | API provider for key/base-url lookup |
| `--model S` | provider default | LLM model name |
| `--llm-thinking` | off | Enable LLM thinking mode |
| `--agent-type {v2,v3}` | `v3` | Legacy CLI values: `v2` selects `agent-type1`; `v3` selects `agent-type2` |
| `--label S` | `LLM_v3_fermi_z_v3_g100_1000inter` | Output directory / summary label |
| `--output-root PATH` | `results/quantitative_baseline` | Root for per-seed result folders |
| `--dry-run` | off | Validate and print the seed plan without running |

Per seed, results are written to
`<output-root>/<label>_seed<s>/evolutionary.json`, and a combined
`<label>_summary.json` is written after each seed completes. The launcher reads
API keys from `.env` via `experiments.config.load_env` and exits non-zero if a
seed fails.

## Re-running the original experiment batches

The orchestration tool documents the earlier donor-game and IPD comparison
experiments:

```powershell
uv run rerun-experiments --audit
uv run rerun-experiments --experiments 1 --seeds 0 1 2
uv run rerun-experiments --experiments 2 --seeds 0 1 2
uv run rerun-experiments --experiments 3 --seeds 0 1 2
uv run rerun-experiments --experiments 4 --seeds 0 1 2
uv run rerun-experiments --experiments 5 --seeds 0 1 2
```

See [`experiments/tools/README.md`](experiments/tools/README.md) for the
full batch plan and trial-specific options.

## Key references

- Schmid, L., Ekbatani, F., Hilbe, C. & Chatterjee, K. (2023).
  *Quantitative assessment can stabilize indirect reciprocity under imperfect
  information.* Nature Communications 14, 2086.
- Ohtsuki, H. & Iwasa, Y. (2006).
  *The leading eight: Social norms that can maintain cooperation by indirect
  reciprocity.* Journal of Theoretical Biology 239, 435–444.
- Willis, R., Du, Y., Leibo, J. Z. & Luck, M. (2025).
  *Will Systems of LLM Agents Cooperate: An Investigation into a Social
  Dilemma.* arXiv:2501.16173.

## Repository status

- Core experiment and re-run tooling are implemented.
- The Agent 2 / Schmid L1–L2–L7–L8 bidirectional invasion runner is implemented
  and has completed 336 runs.
- The package exposes `uv run` entry points for the main CLI, re-run tool,
  invasion runner, and invasion dashboard.
- Paper drafts and supplementary artifacts remain under active development.

---

