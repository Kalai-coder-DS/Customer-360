"""Render every dashboard page headlessly with Streamlit's own test runner.

These tests need Streamlit and Plotly (installed from requirements.txt). They
are skipped automatically where those packages are missing.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import build_full_pipeline  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app"
PAGES = sorted(p.name for p in (APP / "pages").glob("*.py"))
TIMEOUT = 120


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    build_full_pipeline(tmp_path_factory.mktemp("app"), mp)
    yield
    mp.undo()


def _run(path: Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    at.run()
    return at


def _errors(at: AppTest) -> list[str]:
    return [str(e.value) for e in at.exception]


def test_main_app_starts(pipeline):
    at = _run(APP / "streamlit_app.py")
    assert not at.exception, _errors(at)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(pipeline, page):
    at = _run(APP / "pages" / page)
    assert not at.exception, _errors(at)
    assert len(at.markdown) > 0  # page header and text rendered


def _text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_overview_shows_kpis_and_takeaways(pipeline):
    at = _run(APP / "pages" / "overview.py")
    text = _text(at)
    assert "Churn rate" in text and "Total customers" in text
    assert "What this page tells you" in text


def test_customer_profile_switches_customer(pipeline):
    at = _run(APP / "pages" / "customer_profile.py")
    box = at.selectbox[0]
    box.select(box.options[3]).run()
    assert not at.exception, _errors(at)
    assert box.options[3] in _text(at) and "chance of churning" in _text(at)


def test_retention_filters_and_what_if(pipeline):
    at = _run(APP / "pages" / "retention.py")
    at.slider[0].set_value(0.5).run()          # minimum churn risk filter
    assert not at.exception, _errors(at)
    at.slider[1].set_value(1.5).run()          # success-rate multiplier
    assert not at.exception, _errors(at)


def test_empty_database_shows_guidance(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "empty.db")
    at = _run(APP / "pages" / "overview.py")
    assert not at.exception, _errors(at)
    assert any("run_eda.py" in c.value for c in at.code)
