"""
Framework diagram v5 — final cleanup.

Fixes from v4:
- decide() → action label moved DOWN to clear Legend entirely
- Legend at (lx=13.8, ly=8.7) and decide() arrow ends at y=7.45 (1.25 gap) ✓
- "Game env" legend no longer overlaps CodeAgent.observe box
- "(fitness, code)" label moved further LEFT and OFF the down arrow
- validator's "error" feedback arrow shortened and label moved
- Re-added the executor → decide/observe "returns float / bool" arrow
  (it was missing in v4); routed through a small notch in the cell
- Tightened exec(code) and reads/writes state labels so they don't crowd
  the StrategyExecutor box edges
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colors
COL_LLM       = "#7E57C2"
COL_LLM_LIGHT = "#E1D5F2"
COL_GAME      = "#2E7D32"
COL_GAME_LIGHT= "#D6EBD8"
COL_AGENT     = "#1565C0"
COL_AGENT_LIGHT="#D6E5F2"
COL_SANDBOX   = "#E65100"
COL_SANDBOX_LIGHT="#FCE4D2"
COL_SELECTION = "#C62828"
COL_SELECTION_LIGHT="#F4D2D2"
COL_OBS       = "#5D4037"
COL_OBS_LIGHT = "#E5D9D4"
COL_NEUTRAL   = "#424242"
COL_HEADER_BG = "#37474F"

def box(ax, x, y, w, h, text, fc, ec=None, fontsize=8.5, fontweight="normal",
        textcolor="black", boxstyle="round,pad=0.04,rounding_size=0.10"):
    if ec is None: ec = fc
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                       linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=3)

def arrow(ax, x1, y1, x2, y2, color=COL_NEUTRAL, lw=1.8, style="-|>",
          connectionstyle="arc3,rad=0", label=None, labeloffset=(0, 0.10),
          labelsize=7.5, label_va="center"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, color=color, lw=lw, mutation_scale=12,
                        connectionstyle=connectionstyle, zorder=4)
    ax.add_patch(a)
    if label is not None:
        xm, ym = (x1+x2)/2 + labeloffset[0], (y1+y2)/2 + labeloffset[1]
        ax.text(xm, ym, label, fontsize=labelsize, color=color, ha="center", va=label_va,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=color, alpha=0.95, lw=0.4), zorder=5)

# Figure
fig, ax = plt.subplots(figsize=(15, 12))
ax.set_xlim(0, 15)
ax.set_ylim(0, 12)
ax.set_aspect("equal"); ax.axis("off")

# Title
ax.text(7.5, 11.65, "LLM-Driven Code Evolution: Framework and Code Pipeline",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text(7.5, 11.30,
        "Each agent runs LLM-authored evaluate()/decide() code; classical selection acts on it; LLM re-enters as a directed mutation operator",
        ha="center", va="center", fontsize=9, color=COL_NEUTRAL, style="italic")

# Top: Generation loop header
box(ax, 0.3, 9.95, 14.4, 0.50,
    "GENERATION LOOP   (G=10 by default)   →   DonorGame.run_simulation(T=20)   →   "
    "select_survivors   →   mutate_batch   →   repeat",
    "#ECEFF1", ec=COL_HEADER_BG, fontsize=9.5, fontweight="bold",
    textcolor=COL_HEADER_BG,
    boxstyle="round,pad=0.04,rounding_size=0.08")

# Layer A: DonorGame + Observability + CodeAgent.observe
box(ax, 0.3, 8.0, 5.0, 1.7,
    "DonorGame.play_round()\n"
    "experiments/game/donor_game.py\n"
    "• Shuffle donor order\n"
    "• Each agent donates once / round\n"
    "• Payoff:  cooperate → donor -c, recipient +b\n"
    "           defect     → ±0\n"
    "• log to _global_log",
    COL_GAME_LIGHT, ec=COL_GAME, fontsize=8.5)

box(ax, 5.6, 8.0, 4.3, 1.7,
    "_distribute_observations()\n"
    "experiments/game/donor_game.py\n"
    "• private    →  no observations\n"
    "• partial_X  →  each agent sees X% of\n"
    "                  other interactions\n"
    "• full       →  agent sees everything",
    COL_OBS_LIGHT, ec=COL_OBS, fontsize=8.5)

box(ax, 10.2, 8.0, 4.5, 1.7,
    "CodeAgent.observe()\n"
    "experiments/agents/code_agent.py\n"
    "• For each observed interaction:\n"
    "      run evaluate()  →  update private\n"
    "      reputation of donor\n"
    "• Reputation store:  dict[agent_id, float]\n"
    "   initial = 0.01 (cold-start safety)",
    COL_AGENT_LIGHT, ec=COL_AGENT, fontsize=8.5)

# Layer A arrows
arrow(ax, 5.3, 8.85, 5.6, 8.85, color=COL_GAME, lw=2)
arrow(ax, 9.9, 8.85, 10.2, 8.85, color=COL_OBS, lw=2, label="interaction log")
# decide() back-edge: from CodeAgent.observe DOWN to AGENT CELL decide/observe box
# Label moved to y=7.65 (just above the arrow) so it doesn't intrude on Legend
arrow(ax, 13.4, 8.0, 13.4, 7.45, color=COL_AGENT, lw=2, style="-|>")
ax.text(12.55, 7.65, "decide()  →  action", fontsize=7, color=COL_AGENT, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=COL_AGENT, alpha=0.95, lw=0.4),
        zorder=5)
# fitness signal
arrow(ax, 2.5, 8.0, 2.5, 7.45, color=COL_GAME, lw=2, style="-|>")
ax.text(3.20, 7.65, "fitness signal", fontsize=7, color=COL_GAME, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=COL_GAME, alpha=0.95, lw=0.4),
        zorder=5)

# Layer B: AGENT CELL
container = FancyBboxPatch((0.3, 3.95), 14.4, 3.0,
                            boxstyle="round,pad=0.04,rounding_size=0.15",
                            linewidth=1.2, edgecolor=COL_AGENT,
                            facecolor=COL_AGENT_LIGHT, alpha=0.35, zorder=0)
ax.add_patch(container)
ax.text(0.55, 6.80, "AGENT CELL  —  one per population member (N=15 by default)",
        ha="left", va="center", fontsize=10, fontweight="bold", color=COL_AGENT)

# 4 sub-boxes
box(ax, 0.6, 4.15, 4.0, 2.50,
    "STRATEGY CODE   (LLM-authored)\n\n"
    "def evaluate(\n"
    "    current_reputation,\n"
    "    observation, my_history,\n"
    "    round_num) -> float [-1, 1]\n\n"
    "def decide(\n"
    "    recipient_reputation,\n"
    "    round_num, my_history) -> bool",
    "white", ec=COL_AGENT, fontsize=8.5)

box(ax, 4.85, 4.15, 3.3, 2.50,
    "StrategyExecutor\n"
    "experiments/sandbox/executor.py\n\n"
    "• exec(code) into restricted\n"
    "    namespace\n"
    "• Wraps evaluate() / decide()\n"
    "    calls\n"
    "• Catches runtime exceptions\n"
    "    → safe False / no rep change\n"
    "• Loads once at construction",
    COL_SANDBOX_LIGHT, ec=COL_SANDBOX, fontsize=8.5)

box(ax, 8.4, 4.15, 3.5, 2.50,
    "PRIVATE STATE\n"
    "(per-agent, per-generation)\n\n"
    "reputations  : dict[agent_id, float]\n"
    "my_history   : list of dicts\n"
    "   {round, role, partner,\n"
    "    action, partner_action}\n"
    "fitness, total_donations,\n"
    "   total_decisions, generation",
    COL_AGENT_LIGHT, ec=COL_AGENT, fontsize=8.5)

box(ax, 12.15, 4.15, 2.4, 2.50,
    "decide()  /  observe()\n"
    "agents/code_agent.py\n\n"
    "Caller-side wrappers\n"
    "that feed the\n"
    "strategy code:\n"
    "  decide(rep, round, hist)\n"
    "  evaluate(rep, obs, hist,\n"
    "            round)",
    COL_AGENT_LIGHT, ec=COL_AGENT, fontsize=8)

# Internal arrows in agent cell
arrow(ax, 4.6, 5.40, 4.85, 5.40, color=COL_AGENT, lw=1.8, label="exec(code)")
arrow(ax, 8.15, 5.40, 8.4, 5.40, color=COL_SANDBOX, lw=1.8, label="reads/writes state")
# private state → decide/observe caller (right, top)
arrow(ax, 11.9, 5.10, 12.15, 5.10, color=COL_AGENT, lw=1.6, style="-|>",
      label="read rep / hist", labeloffset=(0, 0.15))
# decide/observe caller → private state (left, bottom)
arrow(ax, 12.15, 4.55, 11.9, 4.55, color=COL_AGENT, lw=1.6, style="-|>",
      label="rep update", labeloffset=(0, -0.15))
# executor → decide/observe (returns float / bool) — routed through
# the bottom of the cell to avoid colliding with private state.
# We draw a short L-shaped path: down from executor, right to decide/observe.
arrow(ax, 6.5, 4.15, 6.5, 3.85, color=COL_SANDBOX, lw=1.4, style="-|>")
arrow(ax, 6.5, 3.85, 13.3, 3.85, color=COL_SANDBOX, lw=1.4, style="-|>")
arrow(ax, 13.3, 3.85, 13.3, 4.15, color=COL_SANDBOX, lw=1.4, style="-|>")
# label below the horizontal segment, INSIDE the cell area (so it's between the cell bottom and selection box top)
ax.text(9.5, 3.75, "returns float / bool (evaluate) or bool (decide)", fontsize=7,
        color=COL_SANDBOX, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=COL_SANDBOX, alpha=0.95, lw=0.4),
        zorder=5)

# Layer C: Selection
box(ax, 0.3, 2.1, 5.0, 1.5,
    "select_survivors()\n"
    "experiments/evolution/selection.py\n\n"
    "• rank_by_fitness(desc)\n"
    "• elite_count=2:  top 2 unchanged\n"
    "• num_to_eliminate=5:  bottom 5 removed\n"
    "• tournament_select(K=3, n=5)\n"
    "     →  5 parents for next generation",
    COL_SELECTION_LIGHT, ec=COL_SELECTION, fontsize=8.5)

# (fitness, code) flow: from AGENT CELL bottom to selection top
# Place the label to the LEFT of the down arrow (no overlap)
arrow(ax, 2.5, 3.95, 2.5, 3.65, color=COL_SELECTION, lw=2.0, style="-|>")
ax.text(1.55, 3.78, "(fitness, code)", fontsize=7.5, color=COL_SELECTION, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=COL_SELECTION, alpha=0.95, lw=0.4),
        zorder=5)

# Layer D: Mutation pipeline
box(ax, 5.6, 2.1, 4.4, 1.5,
    "build_mutation_prompt()\n"
    "experiments/agents/prompts.py\n\n"
    "• inject  parent_code  (the strategy)\n"
    "• inject  parent_fitness  (a number)\n"
    "• instruct: vary scoring rules,\n"
    "   thresholds, tracking variables\n"
    "• NO mention of reputation /\n"
    "   IS / Standing",
    COL_LLM_LIGHT, ec=COL_LLM, fontsize=8.5)

box(ax, 10.3, 2.1, 4.4, 1.5,
    "DeepSeek / Intern LLM\n"
    "experiments/evolution/mutation.py\n"
    "  MutationOperator._call_llm()\n\n"
    "• temperature       = 0.8\n"
    "• max_tokens        = 3000\n"
    "• per_call_timeout  = 60s\n"
    "• max_retries       = 3  (exp. backoff)",
    COL_LLM_LIGHT, ec=COL_LLM, fontsize=8.5)

# parents → mutation prompt
arrow(ax, 5.3, 2.85, 5.6, 2.85, color=COL_SELECTION, lw=2.0,
      label="parents (code, fitness)", labeloffset=(0, 0.15))
# prompt → LLM
arrow(ax, 10.0, 2.85, 10.3, 2.85, color=COL_LLM, lw=2.0,
      label="prompt", labeloffset=(0, 0.15))
# LLM → validator
arrow(ax, 12.5, 2.1, 12.5, 1.7, color=COL_LLM, lw=1.8, style="-|>")

# Layer E: Validator
box(ax, 5.6, 0.35, 6.9, 1.3,
    "validate_strategy_code()       ↺   retry up to 3×\n"
    "experiments/sandbox/validator.py\n"
    "•  AST parse              •  must define  evaluate  AND  decide\n"
    "•  exact signature match   •  forbid  os / subprocess / __import__\n"
    "on failure:  error message fed back into prompt\n"
    "after 3 fails:  fallback to  RandomMutationOperator  (control arm)",
    "white", ec=COL_SANDBOX, fontsize=8.5)

# validator → mutation prompt (error feedback) — label to the right
arrow(ax, 6.5, 1.65, 6.5, 2.05, color=COL_SANDBOX, lw=1.5, style="-|>")
ax.text(6.85, 1.85, "error", fontsize=7, color=COL_SANDBOX, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COL_SANDBOX, alpha=0.95, lw=0.4),
        zorder=5)

# child code back to population: route along LEFT margin
arrow(ax, 5.6, 0.7, 0.6, 0.7, color=COL_LLM, lw=1.5, style="-|>")
arrow(ax, 0.6, 0.7, 0.6, 2.0, color=COL_LLM, lw=1.5, style="-|>",
      label="child code\n(replaces 5 eliminated)",
      labeloffset=(0.5, 0.5), labelsize=7)

# Right-side legend (kept at the upper right where it doesn't fight anything)
lx, ly = 13.8, 9.0
ax.text(lx, ly + 0.30, "Legend", fontsize=9, fontweight="bold", color=COL_NEUTRAL)
def legend_item(y, color, text):
    rect = plt.Rectangle((lx, y), 0.30, 0.22, facecolor=color, edgecolor=color, zorder=5)
    ax.add_patch(rect)
    ax.text(lx + 0.40, y + 0.11, text, ha="left", va="center",
            fontsize=8, color=COL_NEUTRAL)
legend_item(ly - 0.05, COL_LLM,       "LLM")
legend_item(ly - 0.35, COL_GAME,      "Game env")
legend_item(ly - 0.65, COL_AGENT,     "Agent code")
legend_item(ly - 0.95, COL_SANDBOX,   "Sandbox / validator")
legend_item(ly - 1.25, COL_SELECTION, "Selection")
legend_item(ly - 1.55, COL_OBS,       "Observability")

# Bottom note
ax.text(7.5, 0.05,
        "Purple = LLM  •  Green = game env  •  Blue = agent code  "
        "•  Red = selection  •  Orange = sandbox/validator  •  Brown = observability",
        ha="center", va="bottom", fontsize=8, color=COL_NEUTRAL, style="italic")

plt.tight_layout()
import os
out_dir = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\figures\framework"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "framework.pdf"), bbox_inches="tight", dpi=200)
plt.savefig(os.path.join(out_dir, "framework.png"),  bbox_inches="tight", dpi=160)
print("Wrote framework.pdf and framework.png (v5)")
