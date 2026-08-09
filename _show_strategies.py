"""Pick one representative code per strategy and print it."""
import json
import re
from pathlib import Path

p = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline\LLM_evolution_seed0\evolutionary.json")
r = json.loads(p.read_text(encoding="utf-8"))
final = r["final_population"]

def is_fallback(code):
    s = code.strip()
    return ("self._ctx_opponent_id = None" in s
            and "def decide(self) -> bool:" in s
            and "return True" in s
            and "return None" in s
            and s.count("def ") <= 4)

state_attr_re = re.compile(r"self\.(\w+)\s*=\s*(\{\}|\[\]|\(\))")
groups = {}
for a in final:
    c = a.get("code", "")
    if is_fallback(c):
        continue
    m = state_attr_re.search(c)
    attr = m.group(1) if m else "no_state"
    groups.setdefault(attr, []).append((a["agent_id"], c))

# Print one full code per group, in order of count
for attr, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    agent_id, code = items[0]
    print("=" * 70)
    print(f"STRATEGY: state=.{attr}   ({len(items)} agent(s), sample agent_id={agent_id})")
    print("=" * 70)
    print(code)
    print()
