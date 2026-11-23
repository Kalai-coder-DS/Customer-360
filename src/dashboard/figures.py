"""Interactive Plotly charts for the dashboard, styled with the shared theme.

Charts have no titles of their own: each page puts a heading and a one-line
caption above the chart, so the message is stated in words first.
Hover tooltips carry the exact numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import settings, theme

FONT = dict(family=theme.FONT_FAMILY, size=13, color=theme.TEXT_SECONDARY)


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    """Shared dark-plum chart styling: transparent background (the card shows through),
    light text, subtle grid, readable tooltips."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=20, t=20, b=10), font=FONT,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font=dict(family=theme.FONT_FAMILY, size=13, color=theme.TEXT),
                        bgcolor=theme.CARD_RAISED, bordercolor=theme.BORDER),
        showlegend=kwargs.pop("showlegend", False),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                    font=dict(color=theme.TEXT_SECONDARY)),
        **kwargs,
    )
    axis = dict(gridcolor=theme.GRIDLINE, zeroline=False, linecolor=theme.BASELINE,
                tickfont=dict(color=theme.TEXT_SECONDARY),
                title_font=dict(color=theme.TEXT_MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_traces(textfont=dict(color=theme.TEXT), selector=dict(type="bar"))
    return fig


def churn_rate_bars(table: pd.DataFrame, category: str, overall: float,
                    sort_by_rate: bool = True) -> go.Figure:
    """Horizontal bars of churn rate per group, with the average marked."""
    data = table.sort_values("churn_rate") if sort_by_rate else table.iloc[::-1]
    labels = data[category].astype(str)
    fig = go.Figure(go.Bar(
        x=data["churn_rate"], y=labels, orientation="h", marker_color=theme.STATUS["churned"],
        text=[f"{r:.1%}" for r in data["churn_rate"]], textposition="outside",
        customdata=data["customers"],
        hovertemplate="<b>%{y}</b><br>Churn rate %{x:.1%}<br>Customers %{customdata:,}"
                      "<extra></extra>",
    ))
    fig.add_vline(x=overall, line_dash="dash", line_color=theme.TEXT_SECONDARY, line_width=1,
                  annotation_text=f"average {overall:.1%}", annotation_position="top",
                  annotation_font_size=11)
    fig.update_xaxes(tickformat=".0%", range=[0, max(data["churn_rate"].max(), overall) * 1.3])
    return _layout(fig, height=max(220, 46 * len(data) + 80))


def churn_rate_columns(table: pd.DataFrame, category: str, overall: float) -> go.Figure:
    """Column chart for ordered groups (e.g. tenure bands)."""
    fig = go.Figure(go.Bar(
        x=table[category].astype(str), y=table["churn_rate"], marker_color=theme.STATUS["churned"],
        text=[f"{r:.1%}" for r in table["churn_rate"]], textposition="outside",
        customdata=table["customers"],
        hovertemplate="<b>%{x}</b><br>Churn rate %{y:.1%}<br>Customers %{customdata:,}"
                      "<extra></extra>",
    ))
    fig.add_hline(y=overall, line_dash="dash", line_color=theme.TEXT_SECONDARY, line_width=1,
                  annotation_text=f"average {overall:.1%}", annotation_font_size=11)
    fig.update_yaxes(tickformat=".0%", range=[0, table["churn_rate"].max() * 1.3])
    return _layout(fig, height=340)


def stay_vs_churn(active: int, churned: int) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[active, churned], y=["Retained", "Churned"], orientation="h",
        marker_color=[theme.STATUS["retained"], theme.STATUS["churned"]],
        text=[f"{active:,} ({active / (active + churned):.1%})",
              f"{churned:,} ({churned / (active + churned):.1%})"],
        textposition="outside", hovertemplate="<b>%{y}</b>: %{x:,} customers<extra></extra>",
    ))
    fig.update_xaxes(range=[0, max(active, churned) * 1.35])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=180)


def risk_gauge(probability: float, level: str) -> go.Figure:
    """Half-dial showing a customer's churn probability against the risk bands."""
    med, high = (settings.RISK_THRESHOLDS["Medium"] * 100, settings.RISK_THRESHOLDS["High"] * 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(probability * 100),
        number=dict(suffix="%", font=dict(size=40, color=theme.TEXT)),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%", tickvals=[0, med, high, 100],
                      tickcolor=theme.BASELINE, tickfont=dict(color=theme.TEXT_SECONDARY)),
            bar=dict(color=theme.RISK.get(level, theme.PRIMARY), thickness=0.32),
            bgcolor=theme.CARD_RAISED, borderwidth=0,
            steps=[dict(range=[0, med], color=theme.RISK_TINT["Low"]),
                   dict(range=[med, high], color=theme.RISK_TINT["Medium"]),
                   dict(range=[high, 100], color=theme.RISK_TINT["High"])],
        )))
    fig.update_layout(height=210, margin=dict(l=25, r=25, t=10, b=0), font=FONT,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def segment_size_revenue(profiles: pd.DataFrame) -> go.Figure:
    data = profiles.sort_values("share_of_revenue")
    fig = go.Figure()
    for col, name, colour in (("share_of_customers", "Share of customers", theme.VIOLET),
                              ("share_of_revenue", "Share of monthly revenue", theme.MAGENTA)):
        fig.add_bar(x=data[col], y=data["segment_name"], orientation="h", name=name,
                    marker_color=colour, text=[f"{v:.0%}" for v in data[col]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>" + name + " %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%",
                     range=[0, data[["share_of_customers", "share_of_revenue"]].max().max() * 1.3])
    return _layout(fig, height=max(260, 70 * len(data) + 90), showlegend=True)


def segment_map(clean: pd.DataFrame, segments: pd.DataFrame, sample: int = 2500) -> go.Figure:
    data = clean[["customerID", "tenure", "MonthlyCharges"]].merge(
        segments[["customerID", "cluster", "segment_name"]], on="customerID")
    if len(data) > sample:
        data = data.sample(sample, random_state=settings.RANDOM_STATE)
    fig = go.Figure()
    order = data.groupby("segment_name")["cluster"].first().sort_values()
    for name, cluster in order.items():
        part = data[data["segment_name"] == name]
        fig.add_scatter(
            x=part["tenure"], y=part["MonthlyCharges"], mode="markers", name=name,
            marker=dict(size=6, opacity=0.7,
                        color=theme.CATEGORICAL[int(cluster) % len(theme.CATEGORICAL)],
                        line=dict(width=0.5, color=theme.SURFACE)),
            customdata=part["customerID"],
            hovertemplate="<b>%{customdata}</b><br>" + name +
                          "<br>Tenure %{x} months<br>$%{y:.2f}/month<extra></extra>")
    fig.update_xaxes(title_text="Tenure (months)")
    fig.update_yaxes(title_text="Monthly charges (USD)")
    return _layout(fig, height=440, showlegend=True)


def risk_distribution(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for level in ("Low", "Medium", "High"):
        part = predictions[predictions["risk_level"] == level]
        fig.add_histogram(x=part["churn_probability"], name=f"{level} risk",
                          marker_color=theme.RISK[level], xbins=dict(start=0, end=1, size=0.025),
                          hovertemplate="Probability %{x}<br>%{y:,} customers<extra></extra>")
    for thr in settings.RISK_THRESHOLDS.values():
        fig.add_vline(x=thr, line_dash="dash", line_color=theme.TEXT_SECONDARY, line_width=1)
    fig.update_layout(barmode="stack", bargap=0.05)
    fig.update_xaxes(title_text="Predicted churn probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title_text="Customers")
    return _layout(fig, height=340, showlegend=True)


def metric_comparison(test_metrics: pd.DataFrame, labels: dict[str, str]) -> go.Figure:
    """Grouped bars: each metric for each model (tuned threshold)."""
    metrics = {"precision": "Precision", "recall": "Recall", "f1": "F1",
               "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    colours = theme.MODEL_COLOURS
    tuned = test_metrics[test_metrics["threshold_type"] == "tuned"]
    fig = go.Figure()
    for _, row in tuned.iterrows():
        values = [row[m] for m in metrics]
        fig.add_bar(x=list(metrics.values()), y=values, name=labels.get(row["model"], row["model"]),
                    marker_color=colours.get(row["model"], theme.PRIMARY),
                    text=[f"{v:.2f}" for v in values], textposition="outside",
                    hovertemplate="%{x}: %{y:.3f}<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, height=360, showlegend=True)


def confusion_heatmap(confusion: pd.DataFrame) -> go.Figure:
    values = confusion.to_numpy()
    shares = values / values.sum(axis=1, keepdims=True)
    notes = [["Correctly left alone", "False alarm"], ["Missed churner", "Correctly flagged"]]
    text = [[f"<b>{values[i][j]:,}</b><br>{notes[i][j]}<br>{shares[i][j]:.0%} of row"
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=[[1, 0.25], [0.25, 1]], x=list(confusion.columns), y=list(confusion.index),
        text=text, texttemplate="%{text}", textfont=dict(size=13, color=theme.TEXT),
        colorscale=[[0, theme.CARD_RAISED], [1, theme.VIOLET]], showscale=False,
        hoverinfo="skip", xgap=3, ygap=3))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=320)


def importance_bars(importance: pd.DataFrame, top: int = 12) -> go.Figure:
    data = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=data["importance"], y=data["label"], orientation="h", marker_color=theme.PRIMARY,
        error_x=dict(type="data", array=data["spread"], color=theme.TEXT_MUTED, thickness=1),
        text=[f"{v:.3f}" for v in data["importance"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>ROC-AUC drop %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Drop in ROC-AUC when shuffled")
    return _layout(fig, height=max(280, 32 * len(data) + 80))


def contribution_bars(contributions: pd.Series, labels: dict[str, str], top: int = 8) -> go.Figure:
    """Coral bars raise risk, violet bars lower it; labelled so colour is not the only cue."""
    chosen = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    chosen = chosen.head(top).iloc[::-1]
    names = [labels.get(f, f) for f in chosen.index]
    colours = [theme.STATUS["churned"] if v > 0 else theme.STATUS["retained"] for v in chosen]
    words = [f"raises {v:.2f}" if v > 0 else f"lowers {abs(v):.2f}" for v in chosen]
    fig = go.Figure(go.Bar(
        x=chosen.values, y=names, orientation="h", marker_color=colours, text=words,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Push on risk (log-odds) %{x:+.3f}<extra></extra>"))
    span = max(float(chosen.abs().max()), 0.1) * 1.6
    fig.add_vline(x=0, line_color=theme.TEXT_SECONDARY, line_width=1)
    fig.update_xaxes(range=[-span, span], title_text="Push on churn risk · coral raises, violet lowers")
    return _layout(fig, height=max(260, 40 * len(chosen) + 80))


def tier_bars(tiers: pd.DataFrame, colours: dict[str, str]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=tiers["customers"], y=tiers.index, orientation="h",
        marker_color=[colours.get(t, theme.PRIMARY) for t in tiers.index],
        text=[f"{c:,} · ${v:,.0f} at risk" for c, v in
              zip(tiers["customers"], tiers["expected_loss_horizon"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} customers<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, tiers["customers"].max() * 1.6])
    return _layout(fig, height=260)


def scenario_bars(scenarios: pd.DataFrame) -> go.Figure:
    colours = theme.SCENARIO_COLOURS
    fig = go.Figure()
    for scenario in settings.SCENARIO_MULTIPLIERS:
        part = scenarios[scenarios["scenario"] == scenario]
        fig.add_bar(x=part["net_value"], y=part["target"], orientation="h", name=scenario,
                    marker_color=colours.get(scenario, theme.PRIMARY),
                    text=[f"${v:,.0f}" for v in part["net_value"]], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Net value $%{x:,.0f}<extra>" + scenario +
                                  "</extra>")
    fig.add_vline(x=0, line_color=theme.TEXT_SECONDARY, line_width=1)
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    span = float(np.abs(scenarios["net_value"]).max() or 1)
    fig.update_xaxes(title_text="Estimated net value (USD)", tickprefix="$",
                     range=[min(0, scenarios["net_value"].min()) - 0.2 * span, span * 1.35])
    return _layout(fig, height=380, showlegend=True)
