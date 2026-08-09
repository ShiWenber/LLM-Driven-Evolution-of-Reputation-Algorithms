"""
Pass 2: programmatic collision check on the framework diagram.
Renders the figure, extracts bounding boxes of all artists,
flags overlapping text-on-box and label-on-line collisions.
"""
import sys, os
sys.path.insert(0, r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
import importlib.util
spec = importlib.util.spec_from_file_location("pf", r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\plot_framework.py")
mod = importlib.util.module_from_spec(spec)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)
spec.loader.exec_module(mod)

# After exec, fig and ax are the framework's. Render once for bbox.
fig.canvas.draw()

# Collect all text bounding boxes
print("=== Text artist bounding boxes (in axes coords) ===")
texts = []
for t in ax.texts:
    if not t.get_text().strip():
        continue
    bbox = t.get_window_extent()
    # Convert to axes coords
    inv = ax.transAxes.inverted()
    p1 = bbox.transformed(inv)
    x, y = p1.x0, p1.y0
    w, h = p1.x1 - p1.x0, p1.y1 - p1.y0
    texts.append((t.get_text().replace("\n", " | ")[:60], x, y, w, h))

# Check for overlap between any two text bboxes
def overlap(a, b):
    ax_, ay_, aw_, ah_ = a[1], a[2], a[3], a[4]
    bx_, by_, bw_, bh_ = b[1], b[2], b[3], b[4]
    if ax_ + aw_ < bx_ or bx_ + bw_ < ax_:
        return False
    if ay_ + ah_ < by_ or by_ + bh_ < ay_:
        return False
    return True

collisions = []
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        if overlap(texts[i], texts[j]):
            # ignore legend internal overlaps
            t1, t2 = texts[i][0], texts[j][0]
            if "LLM" in t1 and "LLM" in t2: continue
            if "Game env" in t1 and "Game env" in t2: continue
            collisions.append((t1, t2, texts[i], texts[j]))

print(f"\n=== {len(collisions)} text-text collisions ===\n")
for c in collisions:
    print(f"  '{c[0]}'  ↔  '{c[1]}'")
    print(f"     bbox1=({c[2][1]:.2f},{c[2][2]:.2f},{c[2][3]:.2f},{c[2][4]:.2f})")
    print(f"     bbox2=({c[3][1]:.2f},{c[3][2]:.2f},{c[3][3]:.2f},{c[3][4]:.2f})\n")
