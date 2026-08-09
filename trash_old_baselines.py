"""Trash old (wrong-baseline) trial directories.

Old names that are NOT in the new baseline set:
  ALLC, ALLD, IS_Plus, Judging, QuantIS_Schmid2023, SimpleStanding,
  StrictIS, StrictJudgment, LLM_evolution_seed0
"""
import os
import send2trash

OUT = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline"
old = [
    "IS_Plus_seed0", "IS_Plus_seed1", "IS_Plus_seed2",
    "Judging_seed0", "Judging_seed1", "Judging_seed2",
    "QuantIS_Schmid2023_seed0", "QuantIS_Schmid2023_seed1", "QuantIS_Schmid2023_seed2",
    "SimpleStanding_seed0", "SimpleStanding_seed1", "SimpleStanding_seed2",
    "StrictIS_seed0", "StrictIS_seed1", "StrictIS_seed2",
    "StrictJudgment_seed0", "StrictJudgment_seed1", "StrictJudgment_seed2",
    "LLM_evolution_seed0",
    "ALLC_seed0", "ALLC_seed1", "ALLC_seed2",
    "ALLD_seed0", "ALLD_seed1", "ALLD_seed2",
]
for d in old:
    p = os.path.join(OUT, d)
    if os.path.exists(p):
        send2trash.send2trash(p)
        print(f"Trashed: {d}")
    else:
        print(f"Skip (not found): {d}")
