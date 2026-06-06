"""Convert PNG figures to PDF for LaTeX (pdflatex cannot include PNG directly)."""
import os
import sys
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

SRC = Path('results/figures')
for png in sorted(SRC.glob('*.png')):
    pdf = png.with_suffix('.pdf')
    img = Image.open(png).convert('RGB')
    img.save(pdf, 'PDF', resolution=150)
    print(f'  {png.name} -> {pdf.name}')
print('Done.')
