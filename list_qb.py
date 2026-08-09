import os
out = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline"
if os.path.exists(out):
    items = os.listdir(out)
    print(f"Items in {out}: {len(items)}")
    for item in sorted(items):
        full = os.path.join(out, item)
        if os.path.isdir(full):
            sub = os.listdir(full)
            print(f"  [DIR]  {item}/  ({len(sub)} files: {sub[:3]})")
        else:
            print(f"  [FILE] {item}  ({os.path.getsize(full)} bytes)")
else:
    print(f"NOT EXIST: {out}")
