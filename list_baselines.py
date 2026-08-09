"""List new baseline set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")))
from experiments.v2_quantitative.baselines import BASELINES
print(f"{len(BASELINES)} baselines:")
for k in BASELINES:
    print(f"  {k}")
