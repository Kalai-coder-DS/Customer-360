"""Design tokens: the single source of truth for every colour in Customer360 AI.

Theme: Dark Plum-Charcoal + Violet/Magenta + Warm Coral/Amber.
Used by the Streamlit dashboard (CSS + Plotly) and by the static Matplotlib
report charts, so the whole project shares one look. ``.streamlit/config.toml``
repeats the base values because Streamlit reads its theme from that file.

Accessibility notes (checked with a colour-vision validator):
* Violet (#9333EA) has only ~3:1 contrast on cards, so it is used for fills,
  never for small text; text always uses the TEXT tokens.
* Pink and magenta, and amber and orange, are close to each other, so charts
  that use them always carry labels or a legend; colour is never the only cue.
"""

# --- Brand palette (official) -------------------------------------------------
PLUM_BG = "#18111E"       # main background
PLUM_SIDEBAR = "#22162B"  # sidebar
PLUM_CARD = "#2A1C35"     # cards and containers
VIOLET = "#9333EA"        # primary interactions, active states
MAGENTA = "#D946EF"       # secondary interactions, accent text
PINK = "#EC4899"
CORAL = "#FB7185"         # churn
ORANGE = "#FB923C"        # high risk, warnings
AMBER = "#FBBF24"         # medium risk, important metrics
TEXT = "#F8FAFC"          # primary text
TEXT_SECONDARY = "#E2D9EC"

# --- Derived neutrals (shades of the plum background, no new hues) ------------
TEXT_MUTED = "#A99BB8"    # captions, axis titles
BORDER = "#3A2A47"        # card borders
GRIDLINE = "#33243F"      # chart grid
BASELINE = "#4A3858"      # axis / zero lines
CARD_RAISED = "#33223F"   # hover / elevated surfaces, table headers

# --- Roles used by the code ---------------------------------------------------
PRIMARY = VIOLET
ACCENT = MAGENTA
SURFACE = PLUM_CARD       # chart and card background
BACKGROUND = PLUM_BG
SIDEBAR = PLUM_SIDEBAR

# Churn outcome
STATUS = {
    "retained": VIOLET,
    "churned": CORAL,
}

# Risk levels (always shown with a text label)
RISK = {
    "Low": VIOLET,
    "Medium": AMBER,
    "High": ORANGE,
}

# Text colour to use on top of each fill
ON_COLOUR = {VIOLET: TEXT, MAGENTA: PLUM_BG, PINK: PLUM_BG, CORAL: PLUM_BG,
             ORANGE: PLUM_BG, AMBER: PLUM_BG}

# Customer segments: violet, pink, magenta first (best separated order), then
# lighter/darker tints of the same family for segments 4-6.
CATEGORICAL = [VIOLET, PINK, MAGENTA, "#E9D5FF", "#BE185D", "#C084FC"]

# Model / scenario identities (fixed per entity)
MODEL_COLOURS = {"baseline": TEXT_MUTED, "logistic_regression": VIOLET,
                 "random_forest": MAGENTA}
SCENARIO_COLOURS = {"Conservative": AMBER, "Base": VIOLET, "Optimistic": MAGENTA}

# Diverging scale (e.g. correlation -1 .. +1): violet <-> neutral plum <-> coral
DIVERGING = [VIOLET, BORDER, CORAL]
# Sequential scale (low -> high within one hue): card -> violet -> light violet
SEQUENTIAL = [BORDER, VIOLET, "#E9D5FF"]

# Translucent tints for highlights (same hues, low opacity)
VIOLET_TINT = "rgba(147, 51, 234, 0.16)"
VIOLET_TINT_STRONG = "rgba(147, 51, 234, 0.28)"
MAGENTA_TINT = "rgba(217, 70, 239, 0.14)"
RISK_TINT = {"Low": "rgba(147, 51, 234, 0.22)", "Medium": "rgba(251, 191, 36, 0.20)",
             "High": "rgba(251, 146, 60, 0.24)"}

FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
