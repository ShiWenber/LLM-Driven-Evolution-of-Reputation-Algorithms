# LLM-Driven Evolution of Reputation Algorithms under Private Observation

This repository contains the experimental code and draft paper for a
TCSS-targeting submission studying the evolution of cooperation in
LLM-coded multi-agent populations with **private reputation stores**
and a controlled observability parameter.

## Project layout

```
.
├── PAPER_DRAFT.md          # TCSS submission draft (v2, 4 June 2026)
├── CHANGELOG.md            # Version history
├── ISSUES.md               # Known issues + fixes (mutation-prompt fix applied)
├── .env.example            # Template for API key configuration
├── experiments/
│   ├── agents/             # CodeAgent + LLM prompts
│   ├── analysis/           # Result analysis helpers
│   ├── config/             # settings.yaml + load_env.py
│   ├── evolution/          # Population, selection, mutation
│   │   ├── ipd_evolution.py    # NEW: IPD baseline (Willis comparison)
│   │   ├── mutation.py         # mutation-prompt fix applied
│   │   └── population.py
│   ├── game/               # DonorGame + IPDGame
│   ├── main.py             # CLI for donor-game experiments
│   ├── results/            # JSON results (legacy + new)
│   ├── sandbox/            # Code execution + validation
│   └── tools/              # Rerun orchestration
│       ├── rerun.py            # Seeded trial runner
│       └── README.md           # How to re-run experiments
├── pyproject.toml          # Dependencies (managed by uv)
└── uv.lock
```

## Quickstart

1. **Configure API key** (one-time):
   ```bash
   cp .env.example .env
   # Edit .env and add your DEEPSEEK_API_KEY
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Re-run experiments**:
   ```bash
   python -m experiments.tools.rerun --audit
   python -m experiments.tools.rerun --experiments 2 --seeds 0 1 2
   ```

See `experiments/tools/README.md` for the full re-run recipe.

## Key references

- Willis, R., Du, Y., Leibo, J. Z. & Luck, M. (2025). *Will Systems of
  LLM Agents Cooperate: An Investigation into a Social Dilemma.*
  arXiv:2501.16173. — Direct baseline for IPD LLM-driven evolution.
- Schmid, L., Ekbatani, F., Hilbe, C. & Chatterjee, K. (2023).
  *Quantitative assessment can stabilize indirect reciprocity under
  imperfect information.* Nature Communications 14, 2086. —
  Theoretical anchor for the private-observation phase transition.
- Ohtsuki, H. & Iwasa, Y. (2006). *The leading eight: social norms
  that can maintain cooperation by indirect reciprocity.* J. Theor.
  Biol. 239, 435–444. — Classical indirect-reciprocity baseline.

## Status

- [x] Git initialised
- [x] `CHANGELOG.md` written
- [x] `.env.example` provided; `.env` git-ignored
- [x] `mutation.py` fixed for direct-reciprocity leakage
- [x] `PAPER_DRAFT.md` re-aimed at TCSS with Willis-comparison framing
- [x] IPD baseline (Willis comparison) implemented
- [x] Rerun orchestration tool with seed management
- [ ] DeepSeek API key rotated (user action)
- [ ] All experiments re-run with mutation-prompt fix
- [ ] IPD baseline experiments completed
- [ ] Results section of paper populated with new data
- [ ] Final proofreading + submission
