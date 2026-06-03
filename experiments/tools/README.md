# Re-run Experiments

This directory contains the experimental code for the LLM-reputation
paper. Below is the recipe to re-run all experiments after the
mutation-prompt fix (see `CHANGELOG.md` and `ISSUES.md`).

## Prerequisites

1. **DeepSeek API key**: get one at https://platform.deepseek.com/api_keys
2. **Configure `.env`** at the project root with:
   ```
   DEEPSEEK_API_KEY=sk-your-new-key-here
   DEEPSEEK_API_BASE=https://api.deepseek.com
   DEEPSEEK_MODEL=deepseek-v4-flash
   ```
   (`.env` is git-ignored; `.env.example` is the template.)

3. **Install dependencies** (Python 3.12+):
   ```bash
   pip install -e .
   # or
   uv sync
   ```

## Run a single experiment

Audit existing results first:
```bash
python -m experiments.tools.rerun --audit
```

Re-run donor-game experiments (after mutation-prompt fix):
```bash
# Experiment 1: Observability contrast (PRIVATE, FULL), 3 seeds
python -m experiments.tools.rerun --experiments 1 --seeds 0 1 2

# Experiment 2: Threshold scan, 9 obs levels × 3 seeds
python -m experiments.tools.rerun --experiments 2 --seeds 0 1 2

# Experiment 3: Static control
python -m experiments.tools.rerun --experiments 3 --seeds 0 1 2

# Experiment 4: Random mutation control
python -m experiments.tools.rerun --experiments 4 --seeds 0 1 2
```

Run the IPD baseline (Willis et al. 2025 comparison):
```bash
python -m experiments.tools.rerun --experiments 5 --seeds 0 1 2
```

Run the full plan (≈88 trials, several hours wall-clock):
```bash
python -m experiments.tools.rerun --experiments 1 2 3 4 5 --seeds 0 1 2 3 4
```

Dry-run (print commands without executing):
```bash
python -m experiments.tools.rerun --experiments 1 --dry-run
```

## Run a single trial interactively

```bash
# Single evolutionary trial
python -m experiments.main --run evolutionary \
    --observability partial_0.3 --seeds 1 --output results/test

# Single threshold trial
python -m experiments.main --run threshold \
    --p-values 0,0.3,1.0 --seeds 1 --output results/test

# Single IPD trial
python -m experiments.evolution.ipd_evolution \
    --seed 0 --output results/test
```

## Results layout

```
results/
├── exp1_obs_contrast/         # PRIVATE, FULL × seeds
├── exp2_threshold_scan/       # 9 p-values × seeds
├── exp3_static_control/        # 3 obs × seeds (no evolution)
├── exp4_random_mutation/       # 3 obs × seeds (random mutation)
├── ipd_baseline/               # IPD Moran-style × seeds
└── figures/                    # generated plots
```

Each trial produces a single JSON file like
`results/exp2_threshold_scan/exp2_threshold_scan_20260604_150000.json`.

## Notes

- The mutation-prompt fix (Issue 2 in `ISSUES.md`) is **required** before
  re-running: results from before the fix may have conflated direct and
  indirect reciprocity signals in the LLM's strategy generation.
- Each trial is 3-10 minutes wall-clock; full plan is 4-12 hours.
- API costs: ≈$20-50 for the full plan with DeepSeek-V4-Flash.
