"""Convert results/figures/*.png to *.pdf for LaTeX \includegraphics.

Uses Pillow (PIL) to embed PNG into a single-page PDF.
"""
from pathlib import Path
from PIL import Image

ROOT = Path('C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/figures')

pngs = sorted(ROOT.glob('*.png'))
print(f'Found {len(pngs)} PNGs to convert:')
for p in pngs:
    out = p.with_suffix('.pdf')
    img = Image.open(p).convert('RGB')
    img.save(out, 'PDF', resolution=150)
    print(f'  {p.name} -> {out.name} ({p.stat().st_size//1024}KB PNG -> {out.stat().st_size//1024}KB PDF)')

print('Done.')
