"""K-Means clustering: let the data find natural customer groups.

How it works, in plain words:
1. Each customer is described by a few numbers (tenure, monthly charges,
   number of services).
2. The numbers are put on the same scale (standardised), otherwise monthly
   charges (up to ~120) would outweigh services (0-9) just because the values
   are bigger.
3. K-Means places K "centre points" and assigns every customer to the nearest
   one, moving the centres until the groups stop changing.
4. We try several values of K and pick one using the **silhouette score**
   (how clearly separated the groups are: -1 poor, 0 overlapping, 1 very clear),
   restricted to 3-6 groups so the result is usable by a marketing team.

TotalCharges is left out on purpose: it is roughly tenure x monthly charges,
so including it would count the same information twice.
Churn is left out on purpose too: segments describe *who customers are*;
churn is only used afterwards to see how each group behaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.analysis.eda import services_count
from src.data.cleaning import CHURN_FLAG

CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "services_count"]
SILHOUETTE_SAMPLE = 3000  # sample size for speed; fixed seed keeps it reproducible


def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "tenure": df["tenure"].astype(float).values,
        "MonthlyCharges": df["MonthlyCharges"].astype(float).values,
        "services_count": services_count(df).astype(float).values,
    }, index=df.index)


def make_pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=settings.RANDOM_STATE)),
    ])


def evaluate_k(features: pd.DataFrame, k_values=settings.K_SEARCH_RANGE) -> pd.DataFrame:
    """Inertia (for the elbow chart) and silhouette score for each K."""
    rows = []
    sample = min(SILHOUETTE_SAMPLE, len(features))
    for k in k_values:
        if k >= len(features):
            break
        pipe = make_pipeline(k).fit(features)
        scaled = pipe.named_steps["scale"].transform(features)
        labels = pipe.named_steps["kmeans"].labels_
        sil = silhouette_score(scaled, labels, sample_size=sample,
                               random_state=settings.RANDOM_STATE)
        rows.append({"k": k, "inertia": float(pipe.named_steps["kmeans"].inertia_),
                     "silhouette": float(sil)})
    return pd.DataFrame(rows).set_index("k")


def choose_k(scores: pd.DataFrame, allowed=settings.K_CHOICE_RANGE) -> int:
    """Best silhouette among the business-usable cluster counts."""
    candidates = scores.loc[[k for k in allowed if k in scores.index]]
    if candidates.empty:
        candidates = scores
    return int(candidates["silhouette"].idxmax())


@dataclass
class ClusteringResult:
    k: int
    pipeline: Pipeline
    labels: pd.Series                 # cluster id per customer, 0 = shortest tenure
    k_scores: pd.DataFrame
    profiles: pd.DataFrame = field(default_factory=pd.DataFrame)


def fit_segments(df: pd.DataFrame, k: int | None = None) -> ClusteringResult:
    """Fit K-Means and return labels ordered by average tenure (stable naming)."""
    features = build_cluster_features(df)
    k_scores = evaluate_k(features)
    k = k or choose_k(k_scores)
    pipe = make_pipeline(k).fit(features)
    raw_labels = pipe.named_steps["kmeans"].labels_

    # Re-number clusters by mean tenure so cluster 0 is always the newest group.
    order = features.groupby(raw_labels)["tenure"].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    labels = pd.Series([mapping[l] for l in raw_labels], index=df.index, name="cluster")
    pipe.cluster_order_ = mapping  # stored with the model for reuse
    return ClusteringResult(k=k, pipeline=pipe, labels=labels, k_scores=k_scores)


def profile_segments(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Average characteristics and churn rate of every cluster."""
    data = df.assign(
        cluster=labels.values,
        services_count=services_count(df).values,
        month_to_month=(df["Contract"] == "Month-to-month").astype(int).values,
        fiber=(df["InternetService"] == "Fiber optic").astype(int).values,
    )
    grouped = data.groupby("cluster")
    profile = pd.DataFrame({
        "customers": grouped.size(),
        "avg_tenure": grouped["tenure"].mean(),
        "avg_monthly_charges": grouped["MonthlyCharges"].mean(),
        "avg_total_charges": grouped["TotalCharges"].mean(),
        "avg_services": grouped["services_count"].mean(),
        "share_month_to_month": grouped["month_to_month"].mean(),
        "share_fiber": grouped["fiber"].mean(),
        "churn_rate": grouped[CHURN_FLAG].mean(),
        "monthly_revenue": grouped["MonthlyCharges"].sum(),
    })
    profile["share_of_customers"] = profile["customers"] / profile["customers"].sum()
    profile["share_of_revenue"] = profile["monthly_revenue"] / profile["monthly_revenue"].sum()
    return profile.astype(float).round(4).astype({"customers": int})


def _level(value: float, low: float, high: float, words: tuple[str, str, str]) -> str:
    return words[0] if value < low else words[2] if value > high else words[1]


def _make_unique(names: list[str], profile: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """If two clusters share a name, add how many services they use; then letters."""
    s_low, s_high = services_count(df).astype(float).quantile([1 / 3, 2 / 3])
    names = list(names)
    for name in set(names):
        idx = [i for i, n in enumerate(names) if n == name]
        if len(idx) > 1:
            for i in idx:
                word = _level(profile["avg_services"].iloc[i], s_low, s_high,
                              ("few services", "some services", "many services"))
                names[i] = f"{name}, {word}"
    for name in set(names):
        idx = [i for i, n in enumerate(names) if n == name]
        if len(idx) > 1:
            for letter, i in enumerate(idx):
                names[i] = f"{name} ({chr(65 + letter)})"
    return names


def name_segments(profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Give every cluster a readable name, a description and a suggested action.

    Names come from comparing the cluster's averages with the whole customer
    base (bottom third / middle / top third), so they adapt to the data.
    Suggested actions are starting hypotheses for the business to test.
    """
    t_low, t_high = df["tenure"].astype(float).quantile([1 / 3, 2 / 3])
    m_low, m_high = df["MonthlyCharges"].astype(float).quantile([1 / 3, 2 / 3])
    overall_churn = float(df[CHURN_FLAG].mean())

    out = profile.copy()
    names, descriptions, actions = [], [], []
    for _, row in profile.iterrows():
        tenure_word = _level(row["avg_tenure"], t_low, t_high, ("New", "Established", "Long-term"))
        spend_word = _level(row["avg_monthly_charges"], m_low, m_high,
                            ("low-spend", "mid-spend", "high-spend"))
        names.append(f"{tenure_word} {spend_word}")
        risk = row["churn_rate"] / overall_churn if overall_churn else 1.0
        risk_word = ("well above" if risk >= 1.5 else "above" if risk > 1.1 else
                     "well below" if risk <= 0.5 else "below" if risk < 0.9 else "close to")
        descriptions.append(
            f"{int(row['customers']):,} customers ({row['share_of_customers']:.0%}) with "
            f"about {row['avg_tenure']:.0f} months' tenure, ${row['avg_monthly_charges']:.0f} "
            f"per month and {row['avg_services']:.1f} services. "
            f"{row['share_month_to_month']:.0%} are on month-to-month contracts. Churn rate "
            f"{row['churn_rate']:.1%}, {risk_word} the {overall_churn:.1%} average."
        )
        high_value = spend_word == "high-spend"
        if risk >= 1.5 and high_value:
            action = ("Priority retention: valuable and at risk. Proactive outreach, contract "
                      "upgrade offers, service reviews.")
        elif risk >= 1.5:
            action = ("Onboarding and early engagement: reduce early-life churn with welcome "
                      "journeys, check-ins and incentives to move off month-to-month.")
        elif risk <= 0.5 and high_value:
            action = "Protect and reward: loyalty benefits, priority support, referral programmes."
        elif risk <= 0.5:
            action = "Grow: cross-sell add-on services to deepen the relationship."
        else:
            action = "Monitor and nurture: targeted add-on offers and satisfaction checks."
        actions.append(action)

    names = _make_unique(names, profile, df)

    out.insert(0, "segment_name", names)
    out["description"] = descriptions
    out["suggested_action"] = actions
    return out
