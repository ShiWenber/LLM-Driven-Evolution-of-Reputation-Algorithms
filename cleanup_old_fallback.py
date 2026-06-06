"""Clean up old fallback-only experiment data, then run a fresh Standard plan."""
import shutil
import sys
from pathlib import Path
from datetime import datetime

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
RESULTS = REPO / 'results'

# Backup the existing manifest and log
backup_dir = RESULTS / 'fallback_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir.mkdir(parents=True, exist_ok=True)
for f in ('_manifest.json', 'standard_run.log'):
    src = RESULTS / f
    if src.exists():
        dst = backup_dir / Path(f).name
        shutil.copy2(src, dst)
        print(f"Backed up {f} -> {dst}")

# Move (not delete) the old exp* directories
for sub in ('exp1_method', 'exp2_threshold', 'exp3_static', 'exp4_random_mut'):
    src = RESULTS / sub
    if src.exists():
        dst = backup_dir / sub
        shutil.move(str(src), str(dst))
        print(f"Moved {sub} -> {dst}")

print(f"\nOld fallback data preserved in: {backup_dir}")
print("(gitignored, won't be committed unless you explicitly add it)")
