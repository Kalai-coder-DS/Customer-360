"""Phase 9: documentation is complete, links work, and results are generated from data."""

import re
import sys
from pathlib import Path

import pytest

from tests.conftest import build_full_pipeline

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "docs/project_guide.md", "docs/interview_guide.md", "docs/demo_script.md",
        "docs/RESULTS.md", "docs/dataset.md", "docs/architecture.md", "docs/testing.md",
        "docs/deployment.md", "docs/images/README.md"]


@pytest.mark.parametrize("path", DOCS)
def test_document_exists_and_is_not_empty(path):
    file = ROOT / path
    assert file.exists(), path
    assert len(file.read_text(encoding="utf-8")) > 200, path


def _links(markdown: str) -> list[str]:
    visible = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)  # ignore commented-out parts
    return re.findall(r"\]\(([^)#\s]+)\)", visible)


@pytest.mark.parametrize("doc", ["README.md", "docs/project_guide.md", "docs/interview_guide.md",
                                 "docs/demo_script.md"])
def test_relative_links_point_to_real_files(doc):
    path = ROOT / doc
    for link in _links(path.read_text(encoding="utf-8")):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        assert (path.parent / link).resolve().exists(), f"{doc}: broken link {link}"


def test_readme_has_diagrams_badge_and_author():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count("```mermaid") == 2          # workflow + architecture
    assert "actions/workflows/tests.yml/badge.svg" in text
    assert "github.com/Kalai-coder-DS" in text


def test_build_results_uses_real_pipeline_outputs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("results")
    try:
        db = build_full_pipeline(tmp, mp)
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_results
        out = tmp / "RESULTS.md"
        assert build_results.main(["--db", str(db), "--out", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
    finally:
        mp.undo()
    for heading in ("## 1. The customer base", "## 3. Customer segments", "## 4. Churn prediction",
                    "## 5. What the model relies on", "## 6. Retention plan"):
        assert heading in text, heading
    assert "400 customers" in text   # the test pipeline's size, read from the database


def test_build_results_without_data_fails_cleanly(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_results
    assert build_results.main(["--db", str(tmp_path / "none.db"),
                               "--out", str(tmp_path / "R.md")]) == 1
