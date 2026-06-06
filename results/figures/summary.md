# Experimental Summary (Standard plan, real LLM-driven runs)

| Run | Observability | n | Mean | Std | Min | Max |
|---|---|---|---|---|---|---|
| evolutionary | full | 3 | 0.371 | 0.258 | 0.018 | 0.627 |
| evolutionary | partial_0.3 | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| evolutionary | partial_0.7 | 3 | 0.228 | 0.228 | 0.013 | 0.544 |
| evolutionary | private | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| random-mutation | full | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| random-mutation | partial_0.3 | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| random-mutation | private | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| static | full | 2 | 0.478 | 0.018 | 0.460 | 0.496 |
| static | partial_0.3 | 2 | 0.452 | 0.039 | 0.413 | 0.491 |
| static | private | 2 | 0.619 | 0.019 | 0.600 | 0.638 |
| threshold | full | 2 | 0.002 | 0.002 | 0.000 | 0.004 |
| threshold | partial_0.1 | 2 | 0.002 | 0.002 | 0.000 | 0.004 |
| threshold | partial_0.3 | 2 | 0.509 | 0.151 | 0.358 | 0.660 |
| threshold | partial_0.5 | 2 | 0.104 | 0.073 | 0.031 | 0.178 |
| threshold | partial_0.7 | 2 | 0.390 | 0.226 | 0.164 | 0.616 |
| threshold | private | 2 | 0.002 | 0.002 | 0.000 | 0.004 |