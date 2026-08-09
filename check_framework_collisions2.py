"""
Pass 2b: visual debug — overlay all text bbox rectangles on the figure
to see where labels actually obscure box content.
"""
import sys, importlib.util, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "pf",
    r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\plot_framework.py",
)
mod = importlib.util.module_from_spec(spec)

# Re-render framework
fig, ax = plt.subplots(figsize=(15, 12))
ax.set_xlim(0, 15); ax.set_ylim(0, 12); ax.set_aspect("equal"); ax.axis("off")
spec.loader.exec_module(mod)

# Now overlay debug boxes
import matplotlib.patches as mp
for i, t in enumerate(ax.texts):
    txt = t.get_text().strip()
    if not txt:
        continue
    bbox = t.get_window_extent()
    # to figure coords
    inv = fig.transFigure.inverted()
    p = bbox.transformed(inv)
    w, h = p.x1 - p.x0, p.y1 - p.y0
    # Skip legend internal labels
    if "LLM" == txt or "Game env" == txt or "Agent code" == txt or "Sandbox" == txt \
       or "Selection" == txt or "Observability" == txt or "Legend" == txt:
        continue
    rect = mp.Rectangle(
        (p.x0, p.y0), w, h,
        linewidth=0.5, edgecolor="red", facecolor="yellow", alpha=0.25, zorder=10,
    )
    fig.patches.append(rect)

out = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\figures\framework\framework_debug.png"
plt.savefig(out, bbox_inches="tight", dpi=140)
print("Wrote", out)
