"""List top-level dirs by size, exclude .git and __pycache__."""
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")

# Two levels deep
items = []
for p in ROOT.rglob("*"):
    if p.is_dir():
        if p.name in (".git", "__pycache__"):
            continue
        rel = p.relative_to(ROOT)
        if len(rel.parts) == 2:
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".git" not in f.parts and "__pycache__" not in f.parts)
            items.append((size, str(p)))

items.sort(reverse=True)
for size, path in items[:30]:
    print(f"{size/1e6:6.2f} MB  {path[len(str(ROOT))+1:]}")
