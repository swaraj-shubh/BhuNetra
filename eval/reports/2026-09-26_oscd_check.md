# M3 sanity check: change rules on OSCD test pairs (pixel level)

Bitemporal pairs, so no harmonic/persistence stage: this checks the index rules, thresholds and MIN_EVENT_PX sieve only.

| variant | precision | recall | F1 |
|---|---|---|---|
| construction_rule | 0.069 | 0.040 | 0.051 |
| construction_or_clearance | 0.095 | 0.058 | 0.072 |
| (all-change baseline) | 0.044 | 1.000 | - |
