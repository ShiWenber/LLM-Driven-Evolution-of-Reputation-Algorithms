# Fixation benchmark: assessing a candidate strategy the Schmid way

`run-fixation-benchmark` measures **fixation probability** following Schmid et
al. (2023), which is the quantity a stability claim actually needs.

The invasion sweep measures a single frequency axis — the candidate's initial
share — and nothing is inferred about the opposite ordering. Whether the
candidate is itself invadable is a separate experiment, measured here directly
rather than derived from the sweep.

---

## The benchmark

### What the imitation sweep does and does not show

`run-invasion --norms` runs a deterministic payoff-imitation process
for 50 generations and reports the invader's final frequency. That is a
finite-time transient. It cannot distinguish "grows but stalls" from "fixes",
and the manuscript already states the consequence: it "cannot measure fixation
probability or establish evolutionary stability".

A concrete failure mode appears below: `readme_best` grows from rare against
every leading-eight norm in the imitation sweep, yet its fixation probability
against them is essentially zero.

### The method reproduced

From Schmid, Ekbatani, Hilbe & Chatterjee (2023), *Quantitative assessment can
stabilize indirect reciprocity under imperfect information*, Nat. Commun. 14:2086
(doi:10.1038/s41467-023-37817-x):

1. **Two-type mixtures.** For every composition of `k` mutants and `N−k`
   residents, simulate the reputation dynamics. The paper restricts this
   analysis to two types on purpose — it "does not consider the coexistence of
   more than two strategies" — which avoids the three-way reputation ambiguity
   of the norm set.

2. **Stationary payoffs, not per-generation resets.** Reputations accumulate.
   After a burn-in the payoffs are measured:

   $$\pi_i = \frac{1}{N-1}\sum_{j \neq i}\left(b\,\hat{x}_{ji} - c\,\hat{x}_{ij}\right)$$

   with $\hat{x}_{ij}$ the stationary cooperation rate of `i` towards `j`. For
   this repository's simultaneous Prisoner's Dilemma with $b=2$, $c=1$ this is
   exactly the mean payoff per participation, which is what the implementation
   accumulates. **No `reset_for_generation` call happens in the measured
   phase**, unlike the imitation sweeps.

3. **Fixation probability, not a fate.** With
   $d_k = \pi_M(k) - \pi_R(k)$ the Traulsen–Hauert formula gives

   $$\rho_{MR} = \frac{1}{1 + \sum_{i=1}^{N-1}\prod_{k=1}^{i}\exp(-\beta\, d_k)}$$

   Neutrality is $\rho = 1/N$. The mutant is always the candidate, so the
   ordering is fixed and there is no second random stream to confound the
   comparison.

### Usage

If `--burn-in` and `--measure` are omitted, burn-in is automatically set to
`population_size × 10^4` and measurement to `population_size × 3 × 10^4`.
For example, N=20 uses 200,000 interactions for burn-in and 600,000 for
measurement. This follows the reference implementation's 1:3 burn-in-to-
measurement ratio. Explicit arguments can still override this scaling for
smoke tests or controlled reruns.

```powershell
uv run run-fixation-benchmark `
  --candidate "readme_best=experiments/analysis/invasion/custom_strategies/readme_best.py" `
  --probes ALLC ALLD L1 L2 L3 L4 L5 L6 L7 L8 `
  --population-size 50 --burn-in 12000 --measure 12000 --beta 1.0 `
  --action-error 0.01 --observation-error 0.01 --workers 28 `
  --output results/quantitative_baseline/fixation/readme_best_N50

uv run plot-fixation-benchmark `
  --summary results/quantitative_baseline/fixation/readme_best_N50/fixation_benchmark.json
```

### Reading the output

The payoff-difference curve `d(k)` is the informative object, plotted on the
left of the figure:

* `d(k) > 0` — the candidate is the fitter type at that mix.
* A curve that **crosses zero** means an interior equilibrium. Crossing from
  positive at small `k` to negative at large `k` is *positive frequency
  dependence* → coexistence. The opposite is bistability.
* `ρ` integrates the whole curve, so a candidate that wins only while rare can
  still have `ρ ≈ 0`.

### Stationarity is the load-bearing assumption

Every result stores a block-wise payoff series (`block_payoffs`) so the burn-in
can be checked rather than trusted. A `stationarity` block summarises, per probe,
the largest disagreement between the first and second half of the measurement
window, the composition responsible, and the list of compositions above a 0.10
threshold. `run-fixation-benchmark` prints a warning when any probe trips it, and
`plot-fixation-benchmark` annotates the figure.

An example from this repository, at `N=50` and a 12000-interaction burn-in:

```
!! STATIONARITY WARNING --------------------------------------------
These probes still drifted during the measurement window, so their
payoff differences (and therefore rho) are not converged:
      L3  max gap 1.153 at k=48  (drifting k = [47, 48, 49])
Raise --burn-in and/or --measure and re-run with --force.
-------------------------------------------------------------------
```

The drift is not a slow tail: the system is still far from its plateau when the
measurement window *opens*. Splitting the measured blocks shows it plainly for
`k = 48` against L1:

```
blocks: -0.161  +0.005  +1.192  +1.487  +1.496  +1.514
```

### Why the extreme compositions are slow: they are bimodal

At `k = 47…49` the candidate occupies 94–98 % of the population, so the probe is a
handful of agents. The reputation dynamics then admit more than one basin — for
instance "everyone cooperates" (payoff difference ≈ 0) versus "the minority is
ostracised because it is judged bad" (difference ≈ +1.5) — and the transition
between them is slow. A single run reports whichever basin it fell into.

This is not a defect of the implementation, but it does mean a single run per
composition is not an estimate. Three responses are implemented:

1. **`--replicates N` (default 5).** Each composition is run with `N`
   independent seeds and averaged; the per-composition standard deviation is
   stored as `payoff_difference_std` and drawn as a shaded band on the figure.
2. **`rho_at_minus_1sd` / `rho_at_plus_1sd`.** How far `rho` would move if every
   composition sat one standard deviation to either side. Plotted as whiskers.
   These are sensitivity bounds, not confidence intervals — the spread is
   bimodality, not sampling error around a single stationary value.
3. **Concatenated block series.** Stationarity is judged over all replicates
   together, so a lucky single draw cannot mask drift.

Schmid et al. sidestep this by running 5×10⁶ steps per composition — roughly 400×
the 12000 used here. Matching that budget is not practical in pure Python, so the
defaults are smaller and the diagnostics carry the weight instead. **Do not quote
a `rho` whose stationarity block is tripped.**

### Validation

A leading-eight norm used as the candidate must reproduce the pattern Schmid et
al. report. `L1` at `N=10`, burn-in 5000, measure 5000, `β=1`, zero noise:

| invader | resident | ρ | verdict | expected |
| --- | --- | ---: | --- | --- |
| L1 | ALLD | 0.2081 | invades | ρ > 1/N ✓ |
| ALLD | L1 | 0.0099 | blocked | ρ < 1/N ✓ |
| L1 | ALLC | **0.1000** | neutral | ρ ≈ 1/N ✓ |
| ALLC | L1 | **0.1000** | neutral | ρ ≈ 1/N ✓ |

The ALLC pair is an analytic check, not a coincidence: L1 and ALLC behave
identically when every reputation is good, so $d_k = 0$ for all `k`, giving
$\rho = 1/(1 + (N-1)) = 1/N$ exactly. The implementation returns `0.1000` to
four decimals.

This matches the paper's own summary that "ALLC and each leading eight norm are
approximately neutral with respect to each other (their fixation probability
into one another is ≈ 1/N each)", with the leading-eight norm's ability to
invade ALLD being "the deciding factor".

### Measured results

`N=50`, 12000 burn-in + 12000 measure interactions, **5 replicates per
composition**, `β=1`, 1 % action and 1 % observation error. Neutral
`ρ = 0.0200`. Every cell is a full 49-composition sweep.

Brackets give `[ρ at −1 s.d., ρ at +1 s.d.]` of the replicate payoff spread, i.e.
how far the value would move if every composition sat one standard deviation to
either side.

| probe | `readme_best` invades |
| --- | ---: |
| ALLC | 0.0000 [0.0000, 0.0000] |
| ALLD | **0.0803** [0.065, 0.095] |
| L1 | 0.0000 [0.0000, 0.0000] |
| L2 | 0.0000 [0.0000, 0.0000] |
| L3 | 0.0001 [0.000, 0.000] |
| L4 | 0.0001 [0.000, 0.000] |
| L5 | 0.0001 [0.000, 0.000] |
| L6 | 0.0001 [0.000, 0.000] |
| L7 | 0.0000 [0.0000, 0.0000] |
| L8 | 0.0000 [0.0000, 0.0000] |

`readme_best` can invade **ALLD** (`ρ = 0.0803`, 4.0 × neutral) and **nothing
else**: against every leading-eight norm and against ALLC its fixation
probability is at or below neutral, so it cannot take any of them over.

The ALLD result is the interesting one for the paper's argument. `readme_best`
fixates against ALLD from rare, which is the behaviour expected of a norm that
punishes unconditional defection.

**`n50_seed2`** (same sweep, same seeds):

| probe | `n50_seed2` invades |
| --- | ---: |
| ALLC | 0.0228 [0.019, 0.026] |
| ALLD | 0.0054 [0.003, 0.009] |
| L1 | 0.0230 [0.019, 0.026] |
| L2 | 0.0230 [0.019, 0.026] |
| L3 | 0.0230 [0.019, 0.026] |
| L4 | 0.0230 [0.019, 0.026] |
| L5 | 0.0230 [0.019, 0.026] |
| L6 | 0.0230 [0.019, 0.026] |
| L7 | 0.0230 [0.019, 0.026] |
| L8 | 0.0229 [0.019, 0.026] |

`n50_seed2` cannot fixate against anything: `ρ` sits at neutral (≈0.023) against
every leading-eight norm and ALLC, and is *suppressed* against ALLD
(`ρ = 0.0054 < 0.0200`).

So the two candidates differ sharply from each other. `readme_best` gains an edge
against ALLD (4.0 × neutral) while failing against every norm; `n50_seed2` gains
an edge nowhere and is actively suppressed by ALLD. `n50_seed2` is therefore
**payoff-neutral against the whole norm set** — close to the ALLC/L1 boundary of
Schmid's classification.

### Caveat: stationarity

Both candidates still drift at extreme compositions, but to very different
degrees. The `stationarity` block records, per probe, the largest disagreement
between the first and second halves of the measurement window (threshold 0.10):

| candidate | probes above threshold | worst |
| --- | --- | ---: |
| `readme_best` | ALLC, ALLD, L8 | 0.446 (ALLC, `k=46`) |
| `n50_seed2` | ALLC and all of L1–L8 | 0.874 (ALLC, `k=49`) |

For `readme_best` the drift is concentrated at `k = 32…46`, i.e. compositions
where `readme_best` is already the large majority and ALLC/ALLD is a shrinking
minority. `n50_seed2` drifts in the *same* regime but harder and earlier:
almost every probe still shows a half-to-half gap above 0.7 at `k = 46…49`,
where `n50_seed2` holds 92–98 % of the population.

**This matters for how much weight each value can carry.** The ordered product
accumulates from the small-`k` end, which is the converged one; the drifting
`k = 46…49` compositions sit at the *back* of the product and therefore carry
the least weight. The reported `ρ` is consequently more robust than the worst
stationarity gap alone would suggest, but the amplitudes are still soft and
should be re-measured with a larger `--burn-in` before being quoted.

**Read any single-run value with caution, and re-run with a larger `--burn-in`
before quoting a `ρ` whose stationarity block is tripped.** The detector is part
of the output, so this is a check rather than a promise:

```
!! STATIONARITY WARNING --------------------------------------------
These probes still drifted during the measurement window, so their
payoff differences (and therefore rho) are not converged:
    ALLC  max gap 0.446 at k=46  (drifting k = [32, 33, 40, 41, 43, 45, 46])
    ALLD  max gap 0.138 at k=22  (drifting k = [22])
      L8  max gap 0.130 at k=36  (drifting k = [36])
Raise --burn-in and/or --measure and re-run with --force.
-------------------------------------------------------------------
```

This is exactly the distinction the imitation sweep cannot make.
`readme_best` grows from a 10 % minority against L1 in the N=100 imitation sweep
(mean final share 0.82), which reads as an invasion — yet its fixation
probability against L1 is `0.0000`. The growth was a transient *inside a
coexisting mixture*; the payoff-difference curve shows why it never completes.
Reporting only the imitation endpoint would have been misleading.

The benchmark reports one ordering only — the candidate as mutant, each probe as
resident. The opposite ordering is a different experiment and is not inferred
from this sweep.

### Regression tests

`tests/test_fixation_benchmark.py` pins:

* `d_k = 0` for all `k` ⇒ `ρ = 1/N` exactly, for `N ∈ {5, 10, 50}`.
* monotonicity of `ρ` in the payoff advantage, and `β` scaling.
* `ρ` remains a probability and stays finite for extreme inputs.
* `ALLC` vs `ALLC` mixture has `d = 0` exactly.
* the sweep defines a single frequency axis; there is no reverse label.
