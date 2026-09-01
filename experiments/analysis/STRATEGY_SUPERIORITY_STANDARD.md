# Standard for claiming that an evolved strategy is better than the Leading Eight

Version: 1.0 (fixed before evaluating future candidates)

## Scope

This protocol compares one frozen candidate strategy with the repository's
frozen `IS`, `SS`, `SJ`, `SC`, `SH`, `IS+`, `SS+`, and `SJ+` implementations.
Candidate selection, code, hashes, perturbation grid, seeds, and thresholds must
be recorded before inspecting confirmatory results. A candidate used to tune the
protocol must be evaluated on new confirmatory seeds.

## Fixed environment

- Population: 100 for invasion tests; 100 for homogeneous robustness tests.
- Benefit/cost: 2/1.
- Private reputations start at 0 and all agents, reputations, and internal state
  are reset between generations.
- 1,000 random-pair interactions per generation; fitness uses the final 200.
- Selection: 100 synchronous opportunities per generation. A learner copies a
  sampled model if and only if the model has strictly higher realized fitness.
  Mutation and Fermi/logistic acceptance are disabled.
- Common random seeds are used for the candidate and every L8 baseline.

## Fixed perturbation suite

The robustness suite is:

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
  creates a new version of this protocol and must not replace version 1.0.

## Endpoints

### 1. Homogeneous-population robustness

For each strategy and perturbation condition, record executed cooperation rate
and mean payoff per agent. The primary robustness endpoint is payoff retention:

`payoff(condition) / payoff(control)`.

A candidate is robustly non-inferior to one L8 norm only if the lower bound of
the paired 95% bootstrap confidence interval for the candidate-minus-norm payoff
retention is at least `-0.02`. It is robustly superior only if that lower bound
is greater than zero. Both moderate joint error `(0.05, 0.05)` and the minimum
retention across the complete suite must pass.

### 2. Bidirectional evolutionary selection

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
- **Better than norm X under perturbation Y:** passes homogeneous robustness
  non-inferiority and bidirectional dominance against X under Y.
- **Broadly better than L8:** passes against at least 6/8 norms in control,
  mild-joint, and moderate-joint conditions; no norm counter-dominates it; and
  homogeneous worst-case retention is non-inferior to the median L8 retention.
- **Universally better than L8:** passes against all 8 norms in every condition,
  including `SJ` and `SJ+` and severe joint error.

Failure to reject equality is reported as inconclusive, never as equivalence.
Results from only three seeds, a selected error rate, or a selected initial
frequency cannot establish any of the performance claims above.

## Reproducibility

Store raw per-seed results, the candidate source path and SHA-256, baseline code
hashes, full configuration, and summary confidence intervals. Never overwrite
screening output with confirmatory output.
