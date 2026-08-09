"""Move stale build PDF to OS trash (recoverable)."""
import send2trash, os
p = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\paper_zh\paper_zh.stale_20260724.pdf.bak"
if os.path.exists(p):
    send2trash.send2trash(p)
    print(f"Moved to trash: {p}")
else:
    print(f"Not found: {p}")
