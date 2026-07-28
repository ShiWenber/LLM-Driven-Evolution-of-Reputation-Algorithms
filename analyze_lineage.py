"""Per-generation strategy analysis for LLM_evolution seeds.

Loads the new schema (v2) JSON that has per-generation population snapshots
and produces:
  - prints each seed's strategy-family classification per generation
  - detects when key strategy families first appear and when they collapse
  - writes results/quantitative_baseline/lineage_analysis.json

Strategy families (string-match heuristics on the LLM-generated code):
  ALLD          : always return False in decide()
  ALLC          : always return True in decide()
  IS_permissive : decide(rop >= -0.2)  (very permissive)
  IS_strict     : decide(rop >= 0.0+)
  image_scoring : 4-quadrant evaluate with step_coop/step_defect (no self_rep factor)
  context_aware : 4-quadrant + factor = 1.0 + k*my_reputation
  random        : any else
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
OUT = ROOT / "results" / "quantitative_baseline"


def classify(code: str) -> str:
    if "return False" in code and "decide" in code:
        return "ALLD"
    if "return True" in code and "decide" in code and "return False" not in code:
        return "ALLC"
    m_dec = re.search(r"return\s+opponent_reputation\s*([><=!]+)\s*(-?\d+\.?\d*)", code)
    if m_dec:
        op, thresh = m_dec.group(1), float(m_dec.group(2))
        if thresh <= -0.1:
            return "IS_permissive"
        elif thresh == 0.0:
            return "IS_mid"
        else:
            return "IS_strict"
    return "random"


def main():
    report = {}
    for seed in [0, 1, 2]:
        path = OUT / f"LLM_evolution_seed{seed}" / "evolutionary.json"
        if not path.exists():
            print(f"[seed{seed}] missing, skip")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        trajectory = data["trajectory"]
        # Per-gen family counts
        per_gen = []
        for entry in trajectory:
            gen = entry["generation"]
            pop = entry.get("population", [])
            families = [classify(a["code"]) for a in pop]
            cnt = Counter(families)
            coop_rate = entry["cooperation_rate_mean"]
            per_gen.append({
                "generation": gen,
                "coop_rate": coop_rate,
                "family_counts": dict(cnt),
                "n_agents": len(families),
            })
        # First-appearance of each family
        first_appearance = {}
        for pg in per_gen:
            for fam in pg["family_counts"]:
                if fam not in first_appearance:
                    first_appearance[fam] = pg["generation"]
        # Detect collapse: coop_rate drops below 0.5 after being > 0.5
        collapse = None
        for i in range(1, len(per_gen)):
            if per_gen[i - 1]["coop_rate"] >= 0.5 and per_gen[i]["coop_rate"] < 0.5:
                collapse = per_gen[i]["generation"]
                break
        # Recovery after collapse
        recovery = None
        if collapse is not None:
            for i in range(collapse + 1, len(per_gen)):
                if per_gen[i]["coop_rate"] >= 0.9:
                    recovery = per_gen[i]["generation"]
                    break
        report[f"seed{seed}"] = {
            "per_gen": per_gen,
            "first_appearance": first_appearance,
            "collapse_gen": collapse,
            "recovery_gen": recovery,
            "final_coop": trajectory[-1]["cooperation_rate_mean"],
        }
        # Pretty print
        print(f"\n=== seed{seed} ===")
        print(f"final coop = {report[f'seed{seed}']['final_coop']:.3f}")
        print(f"first_appearance: {first_appearance}")
        print(f"collapse_gen: {collapse}, recovery_gen: {recovery}")
        print("gen | coop   | family_counts")
        for pg in per_gen:
            fc = " ".join(f"{k}={v}" for k, v in sorted(pg["family_counts"].items()))
            print(f" {pg['generation']:2d} | {pg['coop_rate']:.3f} | {fc}")
    out_path = OUT / "lineage_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved lineage analysis to {out_path}")


if __name__ == "__main__":
    main()
