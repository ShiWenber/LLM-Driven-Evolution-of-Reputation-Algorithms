# Code-policy literature verification notes

Verified: 2026-09-16. Scope: prompts, search, evaluation, and transfer boundaries for direct reciprocity. Read-only literature investigation; no experiments performed.

## CSRO: verified primary source

Hennes, Li, Schultz, and Lanctot (2026), *Code-Space Response Oracles: Generating Interpretable Multi-Agent Policies with Large Language Models*. [Full manuscript](https://arxiv.org/html/2603.10098v1), [metadata](https://arxiv.org/abs/2603.10098).

- **§2.2–2.4:** Game rules, executable API, opponent code/descriptions, and best-response objective condition generation. Opponents may be filtered by support or Top-5. Outer iterations recompute a meta-equilibrium; inner variants are ZeroShot, LinearRefinement, and AlphaEvolve. LinearRefinement triggers for negative utility, accepts improvements, and stops at nonnegative utility or budget. AlphaEvolve evolves diverse program subpopulations using expected utility against the mixture.
- **§3:** Gemini 2.5 Pro; 20 outer iterations; linear budget 10. Environments: 1,000-round RRPS and 100-hand Leduc. External evaluators: 43 RRPS bots; Leduc CFR+ (10,000 iterations), AlwaysCall, AlwaysFold. Metrics: mean population return, negative worst population payoff, their difference. These measure restricted-population performance, not exact exploitability.
- **§3.3–4:** Baselines include PSRO-IMPALA, direct-action LLM, Q-learning, contextual regret minimization, and CFR+. RRPS uses five seeds; Leduc three. CSRO does not dominate every baseline/metric.
- **Appendix A.2:** Listing 3 supplies RRPS rules/API; Listing 4 supplies Leduc rules, current code, opponent summaries, and SEARCH/REPLACE instructions.
- **§6:** Prompt/model dependence, API costs, code errors, context scalability, and pretrained strategy knowledge limit interpretation.

## Closest predecessor: LLM-PSRO

Bachrach et al. (2025), *Combining Code Generating Large Language Models and Self-Play to Iteratively Refine Strategies in Games*, IJCAI Demonstrations, pp. 10999–11003. [Official record](https://www.ijcai.org/proceedings/2025/1249), [official PDF](https://www.ijcai.org/proceedings/2025/1249.pdf).

**§2, p. 11000:** Checkers bots undergo pairwise evaluation (1,000 games), fictitious-play equilibrium approximation, and best-of-N candidate generation. The prompt includes the complete mixture bot and component source. The best candidate against that mixture enters the population; there is no candidate-level feedback refinement.

**§3, pp. 11000–11001:** CodeLlama-7b-Instruct-hf; five iterations; twenty runs; three initial bots. Evidence centers on declining equilibrium mass of initial bots relative to their population share. **§5, p. 11001:** Extension beyond deterministic, fully observable, zero-sum Checkers remains future work.

## Adjacent IPD evidence: open-source games

Sistla and Kleiman-Weiner (2025), *Evaluating LLMs in Open-Source Games*, NeurIPS 2025. [Official PDF](https://papers.nips.cc/paper_files/paper/2025/file/9649863f655babbe8102b84726eb2829-Paper-Conference.pdf).

**§4, pp. 3–5:** SPARC contains 239 Axelrod strategies. The classification task asks whether a program always cooperates against AllC for ten rounds. Comment removal, class masking, and identifier obfuscation test reliance on semantic cues. This is code-understanding evaluation, not proof of strategy robustness.

**§5, p. 6:** Kimi-K2 generates natural-language strategy specifications followed by Python; ten seeds, ten meta-rounds, ten base-game rounds. Objectives distinguish payoff maximization, cooperative payoff maximization, and deceptive payoff maximization. **§2, p. 3; Figure 1, p. 2:** Programs inspect opponent source code. Therefore its game includes transparency-based commitments unavailable to ordinary history-only IPD.

## Transfer judgments (our synthesis, not paper claims)

1. Borrow the separation of candidate generation, measured feedback, population selection, and independent evaluation. Do not equate a PSRO equilibrium search with finite-population evolutionary fixation.
2. In ordinary direct reciprocity, constrain executable policies to allowed bilateral histories. Giving the generator training-opponent code and giving a deployed policy access to its current opponent's source are distinct interventions.
3. A low worst-case loss against a finite test set does not establish ESS, invasion resistance, or a global best response. These require separately specified evolutionary and adversarial tests.
4. A fair comparison should separate model prior, feedback search, opponent information, and evaluation leakage. Match generation budgets and reserve untouched opponent families and random seeds before optimization.
5. Treat interpretability as inspectability unless behavioral tests establish that comments and summaries faithfully describe executable decisions.

## Unverified lead: excluded from substantive conclusions

[OpenReview paper VBpIRGHktc](https://openreview.net/pdf?id=VBpIRGHktc) appeared in search-index excerpts as an ICLR 2026 workshop contribution on IPD code evolution with a Hall of Fame. Direct access hit a browser challenge and title/authors were not verified. Do not cite as a confirmed paper or import its mechanism/results until independently retrieved. Alternate indexed PDF: https://openreview.net/pdf/aa441be384d7a8a0c4c90ca4bed6ceefd815b458.pdf
