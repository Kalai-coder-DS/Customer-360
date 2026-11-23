"""Reusable interface building blocks so every page looks and reads the same.

Visual hierarchy used on every page:
1. Hero header: workflow step, title, one-sentence purpose
2. "What this page tells you": 2-3 takeaways in words, generated from the data
3. KPI cards: the few numbers that matter, each with a context note and tooltip
4. Chart cards: elevated card = title + takeaway sentence + chart/table
5. "Learn more": the six standard questions, collapsed by default
"""

from __future__ import annotations

from contextlib import contextmanager
from html import escape

import streamlit as st

from config import theme
from src.dashboard import content

TONES = {
    "neutral": theme.MAGENTA,           # general business metric
    "good": theme.STATUS["retained"],   # retained / positive
    "bad": theme.STATUS["churned"],     # churn / loss
    "warn": theme.RISK["High"],         # high risk / warning
    "amber": theme.AMBER,               # important estimate
    "muted": theme.TEXT_MUTED,
}

CSS = f"""
<style>
:root {{
  --c360-bg: {theme.BACKGROUND}; --c360-sidebar: {theme.SIDEBAR}; --c360-card: {theme.SURFACE};
  --c360-raised: {theme.CARD_RAISED}; --c360-border: {theme.BORDER};
  --c360-violet: {theme.VIOLET}; --c360-magenta: {theme.MAGENTA}; --c360-coral: {theme.CORAL};
  --c360-text: {theme.TEXT}; --c360-text2: {theme.TEXT_SECONDARY}; --c360-muted: {theme.TEXT_MUTED};
  --c360-shadow: 0 1px 2px rgba(0,0,0,0.35), 0 6px 18px rgba(0,0,0,0.22);
}}

/* Page frame (config.toml sets the same colours; these are a fallback) */
.stApp {{ background-color: var(--c360-bg); color: var(--c360-text); }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: var(--c360-text); }}
p, li {{ line-height: 1.55; }}
a {{ color: var(--c360-magenta); }}

/* Sidebar and navigation */
section[data-testid="stSidebar"] {{ background-color: var(--c360-sidebar);
  border-right: 1px solid var(--c360-border); }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; }}
[data-testid="stSidebarNav"] a:hover {{ background: {theme.VIOLET_TINT}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {theme.VIOLET_TINT_STRONG};
  box-shadow: inset 3px 0 0 var(--c360-magenta); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: var(--c360-text);
  font-weight: 650; }}

/* Hero header */
.c360-hero {{ background: var(--c360-card); border: 1px solid var(--c360-border);
  border-radius: 18px; padding: 22px 26px 18px; margin-bottom: 14px;
  box-shadow: var(--c360-shadow); }}
.c360-eyebrow {{ color: var(--c360-magenta); font-size: 0.76rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.08em; }}
.c360-title {{ font-size: 2.05rem; font-weight: 750; color: var(--c360-text); margin: 4px 0 6px;
  line-height: 1.15; }}
.c360-lead {{ color: var(--c360-text2); font-size: 1.04rem; max-width: 920px; margin: 0; }}

/* Takeaways */
.c360-takeaways {{ background: {theme.VIOLET_TINT}; border: 1px solid {theme.VIOLET_TINT_STRONG};
  border-left: 5px solid var(--c360-violet); border-radius: 14px; padding: 14px 20px 8px;
  margin: 4px 0 16px; }}
.c360-takeaways h4 {{ margin: 0 0 6px; font-size: 0.9rem; color: var(--c360-magenta);
  text-transform: uppercase; letter-spacing: 0.06em; }}
.c360-takeaways ul {{ margin: 0 0 6px 0; padding-left: 1.1rem; }}
.c360-takeaways li {{ color: var(--c360-text); font-size: 1rem; margin-bottom: 4px; }}
.c360-takeaways b {{ color: var(--c360-text); }}

/* KPI cards */
.c360-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px; margin: 2px 0 14px; }}
.c360-kpi {{ background: var(--c360-card); border: 1px solid var(--c360-border);
  border-radius: 14px; padding: 14px 16px 12px; border-top: 3px solid var(--accent);
  box-shadow: var(--c360-shadow); transition: border-color 0.15s ease; }}
.c360-kpi:hover {{ border-color: {theme.VIOLET_TINT_STRONG}; border-top-color: var(--accent); }}
.c360-kpi .k-label {{ font-size: 0.78rem; font-weight: 650; color: var(--c360-text2);
  text-transform: uppercase; letter-spacing: 0.05em; }}
.c360-kpi .k-value {{ font-size: 1.8rem; font-weight: 750; color: var(--c360-text);
  line-height: 1.15; margin-top: 6px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }}
.c360-kpi .k-note {{ font-size: 0.8rem; color: var(--c360-muted); margin-top: 4px; }}
.c360-kpi .k-info {{ color: var(--c360-muted); cursor: help; font-weight: 400; }}

/* Cards: Streamlit bordered containers become elevated plum cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--c360-card);
  border-color: var(--c360-border) !important; border-radius: 14px;
  box-shadow: var(--c360-shadow); }}
.c360-card-title {{ font-size: 1.08rem; font-weight: 700; color: var(--c360-text); margin: 0; }}
.c360-card-caption {{ font-size: 0.9rem; color: var(--c360-text2); margin: 2px 0 6px; }}
.c360-card-caption b {{ color: var(--c360-text); }}

/* Section headings outside cards */
.c360-section {{ font-size: 1.25rem; font-weight: 700; color: var(--c360-text);
  margin: 18px 0 2px; }}
.c360-section-caption {{ font-size: 0.92rem; color: var(--c360-text2); margin-bottom: 8px; }}

/* Interactive states: buttons, inputs, tabs, expanders */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px;
  border: 1px solid var(--c360-border); background: var(--c360-raised); color: var(--c360-text);
  transition: border-color 0.15s ease, background 0.15s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--c360-magenta);
  color: var(--c360-text); background: {theme.VIOLET_TINT_STRONG}; }}
.stButton > button[kind="primary"] {{ background: var(--c360-violet); border-color: var(--c360-violet);
  color: var(--c360-text); }}
.stButton > button[kind="primary"]:hover {{ background: var(--c360-magenta);
  border-color: var(--c360-magenta); color: {theme.PLUM_BG}; }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
input:focus-visible, textarea:focus-visible {{ outline: 2px solid var(--c360-magenta) !important;
  outline-offset: 2px; }}
.stButton > button:disabled, .stDownloadButton > button:disabled {{ opacity: 0.45;
  cursor: not-allowed; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ background: var(--c360-raised);
  border-color: var(--c360-border); }}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"] > div:focus-within {{
  border-color: var(--c360-magenta); }}
span[data-baseweb="tag"] {{ background: {theme.VIOLET_TINT_STRONG} !important; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: var(--c360-text); font-weight: 650; }}
div[data-baseweb="tab-highlight"] {{ background-color: var(--c360-magenta); }}
details[data-testid="stExpander"], div[data-testid="stExpander"] details {{
  background: var(--c360-card); border-color: var(--c360-border); border-radius: 12px; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--c360-card);
  border: 1px dashed var(--c360-border); }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--c360-magenta); }}
code {{ background: var(--c360-raised); color: var(--c360-text2); }}
hr {{ border-color: var(--c360-border); }}

/* Badges, fields, chips, workflow, notes */
.c360-badge {{ display: inline-block; padding: 3px 12px; border-radius: 999px;
  font-size: 0.8rem; font-weight: 700; vertical-align: middle; }}
.c360-fields {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px 16px; margin: 4px 0 6px; }}
.c360-field .f-label {{ font-size: 0.74rem; color: var(--c360-muted); text-transform: uppercase;
  letter-spacing: 0.05em; font-weight: 650; }}
.c360-field .f-value {{ font-size: 0.98rem; color: var(--c360-text); font-weight: 600; }}
.c360-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0; }}
.c360-chip {{ background: {theme.MAGENTA_TINT}; color: var(--c360-text);
  border: 1px solid rgba(217, 70, 239, 0.35); border-radius: 999px; padding: 3px 11px;
  font-size: 0.82rem; font-weight: 600; }}
.c360-chip.off {{ background: transparent; color: var(--c360-muted);
  border: 1px dashed var(--c360-border); text-decoration: line-through; }}
.c360-flow {{ display: flex; flex-wrap: nowrap; gap: 5px; align-items: stretch; margin: 2px 0 14px; }}
.c360-step {{ flex: 1 1 0; min-width: 0; background: var(--c360-card);
  border: 1px solid var(--c360-border); border-radius: 12px; padding: 9px 10px; }}
.c360-step b {{ display: block; color: var(--c360-text); font-size: 0.88rem; }}
.c360-step span {{ color: var(--c360-text2); font-size: 0.76rem; }}
.c360-step.active {{ border-color: var(--c360-violet); background: {theme.VIOLET_TINT}; }}
.c360-num {{ display: inline-block; width: 20px; height: 20px; border-radius: 50%;
  background: var(--c360-violet); color: var(--c360-text) !important; font-size: 0.72rem !important;
  font-weight: 700; text-align: center; line-height: 20px; margin-bottom: 4px; }}
.c360-arrow {{ align-self: center; color: var(--c360-muted); }}
.c360-note {{ color: var(--c360-muted); font-size: 0.8rem; }}
.c360-brand {{ font-size: 1.25rem; font-weight: 800; color: var(--c360-text); margin-bottom: 0; }}
.c360-brand span {{ color: var(--c360-magenta); }}

/* Small screens: tighter spacing, single-column KPI cards on phones */
@media (max-width: 1100px) {{
  .c360-flow {{ flex-wrap: wrap; }}
  .c360-step {{ flex: 1 1 140px; }}
  .c360-arrow {{ display: none; }}
}}
@media (max-width: 768px) {{
  .block-container {{ padding-left: 0.8rem; padding-right: 0.8rem; }}
  .c360-title {{ font-size: 1.55rem; }}
  .c360-hero {{ padding: 16px 18px; }}
  .c360-kpis {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
  .c360-arrow {{ display: none; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(step: str, title: str, lead: str) -> None:
    st.markdown(
        f'<div class="c360-hero"><div class="c360-eyebrow">{escape(step)}</div>'
        f'<div class="c360-title">{escape(title)}</div>'
        f'<p class="c360-lead">{escape(lead)}</p></div>', unsafe_allow_html=True)


def takeaways(points: list[str], title: str = "What this page tells you") -> None:
    """Highlighted box with the page's main messages in plain words.

    Points may contain **bold** (converted to <b>); everything else is escaped.
    """
    items = "".join(f"<li>{_bold(escape(p))}</li>" for p in points if p)
    st.markdown(f'<div class="c360-takeaways"><h4>{escape(title)}</h4><ul>{items}</ul></div>',
                unsafe_allow_html=True)


def _bold(text: str) -> str:
    parts = text.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def kpi_row(items: list[dict]) -> None:
    """KPI cards. Each item: label, value, optional note, help (tooltip) and tone."""
    cards = []
    for item in items:
        accent = TONES.get(item.get("tone", "neutral"), theme.PRIMARY)
        info = (f' <span class="k-info" title="{escape(item["help"])}">ⓘ</span>'
                if item.get("help") else "")
        note = f'<div class="k-note">{escape(item["note"])}</div>' if item.get("note") else ""
        cards.append(
            f'<div class="c360-kpi" style="--accent:{accent}">'
            f'<div class="k-label">{escape(item["label"])}{info}</div>'
            f'<div class="k-value" title="{escape(str(item["value"]))}">'
            f'{escape(str(item["value"]))}</div>{note}</div>')
    st.markdown(f'<div class="c360-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


@contextmanager
def card(title: str, caption: str | None = None):
    """An elevated card with a title and a takeaway caption; put a chart or table inside."""
    with st.container(border=True):
        st.markdown(f'<p class="c360-card-title">{escape(title)}</p>'
                    + (f'<p class="c360-card-caption">{_bold(escape(caption))}</p>'
                       if caption else ""), unsafe_allow_html=True)
        yield


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="c360-section">{escape(title)}</div>'
                + (f'<div class="c360-section-caption">{_bold(escape(caption))}</div>'
                   if caption else ""), unsafe_allow_html=True)


def fields(pairs: dict[str, str]) -> None:
    """Label/value grid, e.g. for a customer profile."""
    cells = "".join(f'<div class="c360-field"><div class="f-label">{escape(k)}</div>'
                    f'<div class="f-value">{escape(str(v))}</div></div>'
                    for k, v in pairs.items())
    st.markdown(f'<div class="c360-fields">{cells}</div>', unsafe_allow_html=True)


def chips(active: list[str], inactive: list[str] | None = None) -> None:
    html = "".join(f'<span class="c360-chip">{escape(c)}</span>' for c in active)
    html += "".join(f'<span class="c360-chip off">{escape(c)}</span>' for c in inactive or [])
    st.markdown(f'<div class="c360-chips">{html}</div>', unsafe_allow_html=True)


def learn_more(module: str, title: str = "Learn more: how this works, in plain words") -> None:
    """The six standard questions for a module, collapsed by default."""
    answers = content.EXPLAINERS[module]
    with st.expander(f":material/school: {title}"):
        cols = st.columns(2, gap="large")
        for i, (key, question) in enumerate(content.SIX_QUESTIONS):
            with cols[i % 2]:
                st.markdown(f"**{i + 1}. {question}**")
                st.write(answers[key])


def risk_badge(level: str) -> str:
    colour = theme.RISK.get(level, theme.TEXT_MUTED)
    text_colour = theme.ON_COLOUR.get(colour, theme.TEXT)
    return (f'<span class="c360-badge" style="background:{colour};color:{text_colour}">'
            f'{escape(level)} risk</span>')


def workflow(active: int | None = None) -> None:
    """The project workflow as connected steps; ``active`` highlights one (0-based)."""
    parts = []
    for i, (name, note) in enumerate(content.WORKFLOW):
        cls = "c360-step active" if i == active else "c360-step"
        parts.append(f'<div class="{cls}"><span class="c360-num">{i + 1}</span>'
                     f'<b>{escape(name)}</b><span>{escape(note)}</span></div>')
        if i < len(content.WORKFLOW) - 1:
            parts.append('<div class="c360-arrow">→</div>')
    st.markdown(f'<div class="c360-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def missing_data(steps: list[str]) -> None:
    """Friendly empty state telling the user exactly which commands to run."""
    st.info("Some results are not available yet. Run these commands in the project folder, "
            "then refresh this page:", icon=":material/info:")
    st.code("\n".join(steps), language="bash")


def data_notice() -> None:
    st.markdown(f'<p class="c360-note">{escape(content.DATA_NOTICE)}</p>',
                unsafe_allow_html=True)


def brand() -> None:
    st.markdown('<p class="c360-brand">Customer<span>360</span> AI</p>', unsafe_allow_html=True)
    st.caption("Customer analytics, segmentation & churn prediction")


def money(value: float) -> str:
    return f"${value:,.0f}"
