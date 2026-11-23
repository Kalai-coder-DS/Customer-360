"""Exploratory Data Analysis (EDA): describe the customers and find churn patterns.

Every function takes the *cleaned* DataFrame and returns plain tables or
numbers, so the same results feed the terminal report now and the dashboard
in Phase 7.

Statistics used, in plain words:

* **Churn rate by group**: the share of customers in a group who left.
* **Chi-square test + Cramér's V** (text columns): is churn linked to this
  column, and how strongly? V runs from 0 (no link) to 1 (perfect link).
* **Mann-Whitney U test + rank-biserial effect** (number columns): do churned
  and retained customers differ on this number, and by how much? The effect
  runs from -1 to +1; positive means churned customers have higher values.

With 7,000 rows, even tiny differences give very small p-values, so the
*effect size* is what tells us whether a difference matters. All of these
are associations: they show what goes together with churn, not what causes it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.data.cleaning import CHURN_FLAG
from src.data.schema import CATEGORICAL_VALUES, TARGET_COLUMN

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is stored as 0/1 but is really a yes/no category.
CATEGORICAL_FEATURES = ["SeniorCitizen"] + [c for c in CATEGORICAL_VALUES if c != TARGET_COLUMN]

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0–12 months", "13–24 months", "25–48 months", "49+ months"]
TENURE_BAND = "tenure_band"


def overall_churn_rate(df: pd.DataFrame) -> float:
    return float(df[CHURN_FLAG].mean()) if len(df) else 0.0


def overview(df: pd.DataFrame) -> dict:
    """Headline numbers about the customer base."""
    churned = df[CHURN_FLAG] == 1
    return {
        "customers": int(len(df)),
        "churned": int(churned.sum()),
        "retained": int((~churned).sum()),
        "churn_rate": overall_churn_rate(df),
        "avg_tenure_months": float(df["tenure"].mean()),
        "avg_monthly_charges": float(df["MonthlyCharges"].mean()),
        "total_monthly_charges": float(df["MonthlyCharges"].sum()),
        "monthly_charges_of_churned": float(df.loc[churned, "MonthlyCharges"].sum()),
    }


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """How many customers stayed vs left, and the 'always guess No' baseline."""
    counts = df[TARGET_COLUMN].value_counts().reindex(["No", "Yes"], fill_value=0)
    table = pd.DataFrame({"customers": counts, "share": counts / counts.sum()})
    table.index.name = TARGET_COLUMN
    return table


def majority_baseline_accuracy(df: pd.DataFrame) -> float:
    """Accuracy of a 'model' that predicts the most common class for everyone."""
    return float(class_balance(df)["share"].max())


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count, mean, spread, quartiles, skew and missing values for number columns."""
    data = df[NUMERIC_FEATURES].astype(float)
    summary = data.describe().T
    summary["median"] = data.median()
    summary["skew"] = data.skew()
    summary["missing"] = data.isna().sum()
    return summary[["count", "missing", "mean", "median", "std", "min", "25%", "75%",
                    "max", "skew"]].round(2)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct values, most common value and its share, per text column."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        counts = df[col].value_counts(dropna=True)
        rows.append({
            "column": col,
            "distinct_values": int(counts.size),
            "most_common": str(counts.index[0]) if counts.size else None,
            "most_common_share": float(counts.iloc[0] / counts.sum()) if counts.size else None,
            "missing": int(df[col].isna().sum()),
        })
    return pd.DataFrame(rows).set_index("column")


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with tenure grouped into readable bands."""
    out = df.copy()
    out[TENURE_BAND] = pd.cut(out["tenure"].astype(float), bins=TENURE_BINS,
                              labels=TENURE_LABELS)
    return out


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate for every value of one column, highest first.

    ``lift`` compares the group with the whole customer base: 1.5 means the
    group churns 1.5 times as often as average.
    """
    data = df[[column, CHURN_FLAG]].copy()
    if isinstance(data[column].dtype, pd.CategoricalDtype):
        data[column] = data[column].astype(object)
    data[column] = data[column].where(data[column].notna(), "Missing").astype(str)
    grouped = data.groupby(column)[CHURN_FLAG].agg(customers="size", churned="sum")
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    grouped["share_of_customers"] = grouped["customers"] / grouped["customers"].sum()
    overall = overall_churn_rate(df)
    grouped["lift"] = grouped["churn_rate"] / overall if overall else np.nan
    if column == TENURE_BAND:  # keep the natural order for bands
        return grouped.reindex([b for b in TENURE_LABELS if b in grouped.index])
    return grouped.sort_values("churn_rate", ascending=False)


def _cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    """Chi-square p-value and Cramér's V for a contingency table."""
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k else 0.0
    return float(p_value), v


def _strength_v(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.2:
        return "weak"
    if v < 0.3:
        return "moderate"
    return "strong"


def categorical_association(df: pd.DataFrame) -> pd.DataFrame:
    """Rank text columns by how strongly they are associated with churn."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        table = pd.crosstab(df[col], df[TARGET_COLUMN])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_value, v = _cramers_v(table)
        rates = churn_rate_by(df, col)["churn_rate"]
        rows.append({
            "feature": col,
            "cramers_v": round(v, 3),
            "strength": _strength_v(v),
            "p_value": p_value,
            "highest_churn_group": rates.idxmax(),
            "highest_churn_rate": float(rates.max()),
            "lowest_churn_group": rates.idxmin(),
            "lowest_churn_rate": float(rates.min()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("feature").sort_values("cramers_v", ascending=False)


def _strength_r(r: float) -> str:
    r = abs(r)
    if r < 0.1:
        return "negligible"
    if r < 0.3:
        return "small"
    if r < 0.5:
        return "medium"
    return "large"


def numeric_association(df: pd.DataFrame) -> pd.DataFrame:
    """Compare number columns between churned and retained customers."""
    rows = []
    churned_mask = df[CHURN_FLAG] == 1
    for col in NUMERIC_FEATURES:
        churned = df.loc[churned_mask, col].astype(float).dropna()
        stayed = df.loc[~churned_mask, col].astype(float).dropna()
        if churned.empty or stayed.empty:
            continue
        u_stat, p_value = stats.mannwhitneyu(churned, stayed, alternative="two-sided")
        effect = 2 * u_stat / (len(churned) * len(stayed)) - 1  # rank-biserial
        rows.append({
            "feature": col,
            "median_churned": float(churned.median()),
            "median_retained": float(stayed.median()),
            "mean_churned": float(churned.mean()),
            "mean_retained": float(stayed.mean()),
            "effect_size": round(float(effect), 3),
            "strength": _strength_r(effect),
            "p_value": float(p_value),
        })
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).set_index("feature")
    return table.reindex(table["effect_size"].abs().sort_values(ascending=False).index)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between number columns and churn.

    Spearman (rank-based) is used because charges and tenure are skewed and
    their relationships are not straight lines.
    """
    cols = NUMERIC_FEATURES + [CHURN_FLAG]
    return df[cols].astype(float).corr(method="spearman").round(2)


def services_count(df: pd.DataFrame) -> pd.Series:
    """Number of services each customer subscribes to (phone, internet, add-ons)."""
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies", "MultipleLines"]
    count = (df[addons] == "Yes").sum(axis=1)
    count += (df["PhoneService"] == "Yes").astype(int)
    count += (df["InternetService"].fillna("No") != "No").astype(int)
    return count.rename("services_count")


def key_findings(df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Plain-English findings generated from the data (no hard-coded numbers)."""
    findings: list[str] = []
    rate = overall_churn_rate(df)
    findings.append(
        f"{rate:.1%} of customers churned. A model that simply predicts 'no churn' for "
        f"everyone would be {majority_baseline_accuracy(df):.1%} accurate while catching "
        "no churners, which is why accuracy alone is a misleading measure here."
    )

    cat = categorical_association(df)
    for feature, row in cat.head(top_n).iterrows():
        findings.append(
            f"{feature} has a {row['strength']} link with churn (Cramér's V "
            f"{row['cramers_v']:.2f}): '{row['highest_churn_group']}' customers churn at "
            f"{row['highest_churn_rate']:.1%}, compared with {row['lowest_churn_rate']:.1%} "
            f"for '{row['lowest_churn_group']}'."
        )

    bands = churn_rate_by(add_tenure_band(df), TENURE_BAND)
    if len(bands) >= 2:
        first, last = bands.index[0], bands.index[-1]
        first_rate, last_rate = bands.loc[first, "churn_rate"], bands.loc[last, "churn_rate"]
        trend = ("falls" if last_rate < first_rate else
                 "rises" if last_rate > first_rate else "stays level")
        findings.append(
            f"Churn {trend} as customers stay longer: {first_rate:.1%} "
            f"in the first band ({first}) versus {last_rate:.1%} "
            f"for {last}."
        )

    num = numeric_association(df)
    if "MonthlyCharges" in num.index:
        row = num.loc["MonthlyCharges"]
        direction = "higher" if row["median_churned"] > row["median_retained"] else "lower"
        findings.append(
            f"Churned customers pay {direction} monthly charges (median "
            f"${row['median_churned']:.2f} vs ${row['median_retained']:.2f}; "
            f"{row['strength']} effect)."
        )

    findings.append(
        "These are associations, not proof of cause. A characteristic that goes together "
        "with churn may describe the kind of customer who chooses it (for example, less "
        "committed customers picking shorter contracts) rather than be the reason they leave."
    )
    return findings
