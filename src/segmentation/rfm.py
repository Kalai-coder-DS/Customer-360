"""RFM-style scoring adapted to a subscription business.

Classic RFM (Recency, Frequency, Monetary) needs purchase dates and a list of
purchases. The Telco data is a single snapshot with neither, so this project
uses three honest substitutes, each scored 1 (lowest) to 5 (highest):

| Classic RFM   | Substitute used here  | Column                 | Why it fits                           |
|---------------|-----------------------|------------------------|---------------------------------------|
| Recency       | **Loyalty**           | tenure (months)        | how long the relationship has lasted  |
| Frequency     | **Engagement**        | number of services     | how much of the product they use      |
| Monetary      | **Monetary**          | MonthlyCharges         | how much revenue they bring each month|

Scores are percentile-based: a Loyalty score of 5 means the customer is in
the top 20% for tenure. Customers with equal values always get equal scores.
Churn is NOT used to build the scores; it is only used afterwards to see how
each segment behaves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.eda import services_count
from src.data.schema import ID_COLUMN

SCORE_COLUMNS = {
    "loyalty_score": "tenure",
    "engagement_score": "services_count",
    "monetary_score": "MonthlyCharges",
}

# Rules are checked top to bottom; the first match names the segment.
RFM_SEGMENT_RULES: list[tuple[str, str]] = [
    ("Champions", "Long-standing, highly engaged, high-paying customers"),
    ("Loyal core", "Long-standing customers who are not top spenders"),
    ("High-value newcomers", "Recent customers already paying a lot each month"),
    ("New and light", "Recent customers with low spend and few services"),
    ("Developing", "Customers in the middle on loyalty and value"),
]
RFM_SEGMENT_ORDER = [name for name, _ in RFM_SEGMENT_RULES]
RFM_SEGMENT_DESCRIPTIONS = dict(RFM_SEGMENT_RULES)


def percentile_score(values: pd.Series, bins: int = 5) -> pd.Series:
    """Score values 1..bins by percentile rank; ties share the same score."""
    pct = values.astype(float).rank(method="average", pct=True)
    return np.ceil(pct * bins).clip(1, bins).astype(int)


def assign_rfm_segment(loyalty: int, engagement: int, monetary: int) -> str:
    if loyalty >= 4 and engagement >= 4 and monetary >= 4:
        return "Champions"
    if loyalty >= 4:
        return "Loyal core"
    if loyalty <= 2 and monetary >= 4:
        return "High-value newcomers"
    if loyalty <= 2 and monetary <= 2:
        return "New and light"
    return "Developing"


def compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per customer with the three scores and a named segment."""
    data = pd.DataFrame({
        ID_COLUMN: df[ID_COLUMN].values,
        "tenure": df["tenure"].astype(float).values,
        "services_count": services_count(df).values,
        "MonthlyCharges": df["MonthlyCharges"].astype(float).values,
    })
    for score, source in SCORE_COLUMNS.items():
        data[score] = percentile_score(data[source])
    data["rfm_score"] = data[list(SCORE_COLUMNS)].sum(axis=1)  # 3..15
    data["rfm_segment"] = [
        assign_rfm_segment(l, e, m)
        for l, e, m in zip(data["loyalty_score"], data["engagement_score"],
                           data["monetary_score"])
    ]
    return data
