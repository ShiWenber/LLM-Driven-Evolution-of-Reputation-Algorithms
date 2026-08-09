"""Zip the entire project, excluding .git, __pycache__, and debug scripts."""
import os
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
OUT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation.zip")

# Exclude these names at any level
EXCLUDE_DIRS = {".git", "__pycache__"}
EXCLUDE_FILES_ANYWHERE = {".env", ".env.*", "*.key", "*.pem"}
# Exclude these top-level files (debug scripts cluttering the root)
EXCLUDE_ROOT_FILES = {
    "_tmp_check.py", "smoke_test.py", "smoke_test_mutation.py",
    "analyze_asymmetric_delta.py",
}
# Files matching these patterns at any level
EXCLUDE_PATTERNS = [
    "analyze_asymmetric_delta_v",  # v1, v2, ... v11 (debug history)
    "analyze_seed0_strategies.py",
    "analyze_seed0_evolution.py",
    "analyze_seed0_full.py",
    "analyze_seed2_final.py",
    "analyze_exp6.py",
    "analyze_exp6_AB.py",
    "analyze_exp6_sweep.py",
    "dump_seed0_full.py",
    "dump_seed2_codes.py",
    "dump_final_strategies.py",
    "dump_seed2_final_examples.py",
    "scan_high_order.py",
    "check_size.py",
    "make_zip.py",
    "print_v3_trajectories.py",
    "print_summary.py",
    "compute_v3_summary.py",
]


def should_exclude(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    for part in rel.parts:
        if part in EXCLUDE_DIRS:
            return True
    name = p.name
    # Check filename against any-level excludes (handles wildcards)
    for pat in EXCLUDE_FILES_ANYWHERE:
        if "*" in pat:
            from fnmatch import fnmatch
            if fnmatch(name, pat):
                return True
        else:
            if name == pat:
                return True
    if any(name.startswith(pat) for pat in EXCLUDE_PATTERNS):
        return True
    if p.parent == ROOT and name in EXCLUDE_ROOT_FILES:
        return True
    return False


count = 0
total_size = 0
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not should_exclude(p):
            arcname = p.relative_to(ROOT.parent)  # include llm-reputation/ as root
            zf.write(p, arcname)
            count += 1
            total_size += p.stat().st_size

print(f"Zipped {count} files, raw size {total_size/1e6:.1f} MB")
print(f"Output: {OUT}")
print(f"Output size: {OUT.stat().st_size/1e6:.1f} MB")
