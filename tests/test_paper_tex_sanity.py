"""Sanity checks on the generated Chinese paper (paper_zh/paper.tex).

Converted from the root-level ``paper_zh/sanity.py`` script to pytest.
"""
import re
from pathlib import Path

import pytest

PAPER_TEX = Path(__file__).resolve().parents[1] / "paper_zh" / "paper.tex"

# (test-id, required substring) pairs checked against paper.tex.
REQUIRED_MARKERS = [
    ("begin document", "\\begin{document}"),
    ("end document", "\\end{document}"),
    ("ctex package", "\\usepackage{ctex}"),
    ("begin lstlisting", "\\begin{lstlisting}"),
    ("begin abstract", "\\begin{abstract}"),
    ("title in Chinese", "LLM 驱动"),
    ("hybrid kept", "Hybrid"),
    ("leading eight kept", "leading eight"),
    ("image scoring", "Image Scoring"),
    ("author anonymous", "匿名作者"),
    ("abstract translated", "捐赠博弈"),
    ("willis cited", "Willis"),
    ("discussion section", "\\section{讨论}"),
    ("conclusion section", "\\section{结论}"),
    ("data availability", "数据和代码可用性"),
    ("competing interests", "竞争利益"),
]


@pytest.fixture(scope="module")
def paper_content() -> str:
    return PAPER_TEX.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name,marker", REQUIRED_MARKERS, ids=[name for name, _ in REQUIRED_MARKERS]
)
def test_paper_contains_required_marker(paper_content, name, marker):
    assert marker in paper_content, f"missing: {name}"


def test_paper_contains_chinese_characters(paper_content):
    assert re.search(r"[一-鿿]", paper_content)


def test_paper_has_chinese_sections(paper_content):
    sections = re.findall(r"\\section\{([^}]+)\}", paper_content)
    assert len(sections) >= 3
    assert any(re.search(r"[一-鿿]", s) for s in sections)
