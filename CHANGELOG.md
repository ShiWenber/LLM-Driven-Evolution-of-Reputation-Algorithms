# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — TCSS submission preparation

### Added
- TCSS-targeted writing style profile (Willis et al. 2025 as reference)
- `changelog.md` (this file) for explicit version history

### Changed
- `mutation.py`: removed `partner_action` system-message guidance to prevent
  direct-reciprocity leakage (fix for Issue 2 in ISSUES.md)
- Re-aimed paper from Interface Focus (Royal Society) to
  **IEEE Transactions on Computational Social Systems (TCSS)**

### Pending
- Re-run experiments after mutation prompt fix
- Add IPD-comparison experiment block (Willis-baseline)

## [0.1.0] — 2026-05-09 — Initial draft (Interface Focus)

### Added
- Donor-game environment with private reputation stores
- `evaluate()` and `decide()` agent function pair architecture
- LLM-driven initialization of 20 diverse strategy pairs
- Tournament selection with elitism (2 elite, binary tournament, bottom 4 eliminated)
- LLM-driven mutation with random fallback
- Four experiments: Evolutionary / Threshold scan / Static control /
  Random-mutation control
- Phase transition at observability p* ≈ 0.2-0.3
- Emergence of pure Image Scoring (p=0.3), hybrid reciprocity (p=0.2),
  adaptive self-modelling strategies (p=1.0)
- 3 seeds × 9 observability levels × 2 LLM mutation operators
