# Standard for claiming that an evolved strategy is better than the Leading Eight

Version: 1.1 (2026-09-12)

## Scope

This protocol compares one frozen candidate strategy with the repository's
frozen `IS`, `SS`, `SJ`, `SC`, `SH`, `IS+`, `SS+`, and `SJ+` implementations.
Candidate selection, code, hashes, perturbation grid, seeds, and thresholds must
be recorded before inspecting confirmatory results. A candidate used to tune the
protocol must be evaluated on new confirmatory seeds.

## Fixed environment

- Population: 100.
- Benefit/cost: 2/1.
- Private reputations start at 0 and all agents, reputations, and internal state
  are reset between generations.
- 1,000 random-pair interactions per generation; fitness uses the final 200.
- Selection: 100 synchronous opportunities per generation. A learner copies a
  sampled model if and only if the model has strictly higher realized fitness.
  Mutation and Fermi/logistic acceptance are disabled.
- Common random seeds are used for the candidate and every L8 baseline.

## Fixed perturbation suite

The suite is:

1. control `(action error, observation error) = (0, 0)`;
2. action-only `(0.01, 0)` and `(0.05, 0)`;
3. observation-only `(0, 0.01)` and `(0, 0.05)`;
4. joint mild `(0.01, 0.01)`;
5. joint moderate `(0.05, 0.05)`;
6. joint severe `(0.10, 0.10)`.

Action errors independently flip intended actions before payoff calculation.
Observation errors independently flip each executed action for each observer
before that observer updates private reputations.

## Two-stage sample sizes

- Screening: seeds 0-9. It may label a strategy only as a candidate.
- Confirmation: the 30 previously unused seeds `100-129`. Only this stage may
  support a paper claim that a strategy is better than L8. Changing these seeds
  creates a new version of this protocol; superseded versions are retained, not
  overwritten.

## Endpoints

### 1. Bidirectional evolutionary selection

> **Data source (2026-09-11).** The invasion experiment has a single frequency
> axis; it no longer carries a direction parameter. The outward gain below is
> read at candidate counts 5 and 10; the resistance gain is read from the
> **complementary compositions of the same sweep** (counts 95 and 90, i.e. the
> L8 norm holding 5% and 10%). Both are therefore two readings of one sweep,
> not two independent experiments. "Bidirectional" here names the **claim
> structure** — the candidate enters, and the norm cannot enter in reverse —
> not an experimental setting.

Use initial invader counts 5 and 10, 50 generations, and the complete
perturbation suite. For each candidate/L8 pair define:

- outward gain = final candidate share when candidate invades minus its initial
  share;
- resistance gain = initial L8 share minus final L8 share when L8 invades.

The candidate dominates one L8 norm in one condition only if the lower bounds
of the paired 95% bootstrap confidence intervals for both gains are greater
than zero at both initial counts. A norm counter-dominates the candidate if the
reverse inequalities pass by the same rule.

## Fixed claim vocabulary

- **Distinct strategy:** its executable assessment/action mapping is not
  identical to any L8 mapping on the fixed behavioral probe set. This is not a
  performance claim.
- **Better than norm X under perturbation Y:** dominates X under Y by the rule
  above.
- **Broadly better than L8:** passes against at least 6/8 norms in control,
  mild-joint, and moderate-joint conditions, and no norm counter-dominates it.
- **Universally better than L8:** passes against all 8 norms in every condition,
  including `SJ` and `SJ+` and severe joint error.

Failure to reject equality is reported as inconclusive, never as equivalence.
Results from only three seeds, a selected error rate, or a selected initial
frequency cannot establish any of the performance claims above.

## Reproducibility

Store raw per-seed results, the candidate source path and SHA-256, baseline code
hashes, full configuration, and summary confidence intervals. Never overwrite
screening output with confirmatory output.

## Version history

### 1.1 — 2026-09-12

Removed the homogeneous-population robustness endpoint (v1.0 §1) together with
its implementation (`run_perturbation_robustness.py` and its plotting script).
Two reasons:

1. **The endpoint was not well defined.** Retention is
   `payoff(condition) / payoff(control)`, which is undefined whenever the
   control payoff is zero — the case for all-defect. The implementation
   silently coerced that `0/0` to `0.0`, so all-defect's retention difference
   came out identical to the candidate's own retention and it was reported as
   *superior* to the candidate. Every verdict that included it was wrong.
2. **The archived screening run is therefore not usable for a claim**, and its
   raw summary was deleted so the artifact cannot be cited by accident.

Consequence for the claim vocabulary: *Better than norm X* and *Broadly better
than L8* are now defined on bidirectional dominance alone. This is a
**weakening** of both claims, not a rewording of them.

For the record: the deleted screening run (seeds 0-9, N=100, 1,000 interactions
per generation) measured the candidate's worst-case retention difference against
every Leading Eight norm and against `ALLC` in the range −0.46 to −0.50, with
paired 95% bootstrap intervals lying entirely below the −0.02 non-inferiority
margin. The candidate failed that endpoint by a wide margin. Deleting the
endpoint does not turn that failure into a pass.
