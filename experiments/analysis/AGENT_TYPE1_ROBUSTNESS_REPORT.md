# agent-type1 seed4 perturbation-robustness report

Protocol: [`STRATEGY_SUPERIORITY_STANDARD.md`](STRATEGY_SUPERIORITY_STANDARD.md),
version 1.0.

Candidate: the frozen seed4 agent-type1 representative selected by the dominant
root-family rule. Confirmation used the pre-fixed seeds 100-129. The raw and
summary output is stored in
`results/quantitative_baseline/robustness/agent-type1_seed4_confirmation/summary.json`.

## Confirmatory homogeneous-population result

The table reports mean executed cooperation rate over 30 common seeds.

| Strategy | 1% action + 1% observation | 5% + 5% | 10% + 10% |
| --- | ---: | ---: | ---: |
| agent-type1 seed4 | 0.916 | 0.514 | 0.418 |
| IS / SC | 0.973 | 0.874 | 0.766 |
| IS+ | 0.973 | 0.867 | 0.737 |
| SS / SH / SS+ | 0.946 | 0.677 | 0.367 |
| SJ / SJ+ | 0.654 | 0.240 | 0.206 |

At moderate joint error, candidate-minus-baseline payoff-retention confidence
intervals are negative for `IS`, `SC`, `IS+`, `SS`, `SH`, and `SS+`. They are
positive for `SJ` and `SJ+`: 95% paired bootstrap CI `[0.2646, 0.2839]` for
each. The candidate also exceeds `SJ`/`SJ+` on the pre-specified worst-case
retention endpoint: `[0.2068, 0.2181]`.

The candidate's worst-case retention exceeds `SS`/`SH`/`SS+`, but its moderate
joint-error result is worse. Version 1.0 requires both endpoints, so these three
comparisons do not pass. The candidate does not pass against `IS`, `SC`, or
`IS+`.

## Allowed conclusion

The homogeneous confirmation establishes that agent-type1 seed4 is more robust
than the repository's `SJ` and `SJ+` baselines under the fixed robustness
criteria. It does not establish broad superiority over the Leading Eight.

This is only the homogeneous robustness component. The protocol additionally
requires 30-seed, bidirectional invasion confirmation at initial counts 5 and
10 before the phrase "better than SJ/SJ+ under perturbation" may be used as a
complete evolutionary-performance claim.
