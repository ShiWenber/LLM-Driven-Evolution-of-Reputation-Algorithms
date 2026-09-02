# Tools

Development helper scripts for profiling, reporting, and documentation.
These are **not** experiment entry points (those live in
`experiments/` and are registered in `pyproject.toml`); they are
one-off / diagnostic utilities.

| Script | Purpose | Input → Output |
|---|---|---|
| `profile_fermi_experiment.py` | Profile the Fermi evolution run; emit machine-readable timing data (JSON + cProfile pstats) without touching the experiment code (phase-boundary monkey-patch). | `--output-dir <dir>` → `<dir>/profile.json`, `.pstats`, experiment output |
| `generate_profile_report.py` | Turn `profile.json` into a self-contained interactive HTML fragment (D3 charts: phase breakdown, per-generation cost, LLM latency, local hotspots). | `<profile.json> <output.html>` → standalone HTML |

## Profiling pipeline

```powershell
# 1. Profile a short Fermi run (e.g. 5 gens, 200 interactions/gen)
python tools/profile_fermi_experiment.py --output-dir results/profiling/run1 ^
    --seed 0 --gens 5 --target-interactions 200 --agent-type agent-type1

# 2. Render the interactive HTML report from the profile
python tools/generate_profile_report.py results/profiling/run1/profile.json ^
    results/profiling/run1/report.html
```

Notes:
- `profile_fermi_experiment.py` imports the **current** experiment
  implementation (`experiments.run_fermi_v3` +
  `experiments.v2_quantitative`) and drives it through the normal CLI
  entry point, so the profiled behavior matches a real run.
- The HTML report embeds all data inline (no external JSON needed),
  but requires network access for the D3 CDN.
