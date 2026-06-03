import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import iplot


class EDA:
    def __init__(self, color, data):
        self._color = color
        self.data = data

    def _template(self, fig, title):
        fig.update_layout(
            title=title,
            title_x=0.5,
            plot_bgcolor="rgba(88, 88, 88, 1)",
            font=dict(color=self._color),
            margin=dict(l=72, r=72, t=72, b=72),
            height=720,
        )
        return fig

    def histogram_distribution_plot(self, column, title):
        fig = px.histogram(
            self.data,
            x=column,
            nbins=100,
            color_discrete_sequence=[self._color],
        )
        fig.update_layout(
            xaxis_title="Values",
            yaxis_title="Count",
            bargap=0.1,
            xaxis=dict(gridcolor="grey"),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey"),
        )
        fig.update_traces(hovertemplate="Value: %{x:.2f}<br>Count: %{y:,}")
        iplot(self._template(fig, title))

    def plot_cv(self, scores, title, metric="Stratified C-Index"):
        fold_scores = [round(score, 3) for score in scores]
        mean_score = round(np.mean(scores), 3)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(fold_scores) + 1)),
                y=fold_scores,
                mode="markers",
                name="Fold Scores",
                marker=dict(size=27, color=self._color, symbol="diamond"),
                text=[f"{score:.3f}" for score in fold_scores],
                hovertemplate="Fold %{x}: %{text}<extra></extra>",
                hoverlabel=dict(font=dict(size=18)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[1, len(fold_scores)],
                y=[mean_score, mean_score],
                mode="lines",
                name=f"Mean: {mean_score:.3f}",
                line=dict(dash="dash", color="#B22222"),
                hoverinfo="none",
            )
        )
        fig.update_layout(
            title=f"{title} | Cross-validation Mean {metric} Score: {mean_score}",
            xaxis_title="Fold",
            yaxis_title=f"{metric} Score",
            plot_bgcolor="rgba(247, 230, 202, 1)",
            paper_bgcolor="rgba(247, 230, 202, 1)",
            font=dict(color=self._color),
            xaxis=dict(
                gridcolor="grey",
                tickmode="linear",
                tick0=1,
                dtick=1,
                range=[0.5, len(fold_scores) + 0.5],
                zerolinecolor="grey",
            ),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey"),
            margin=dict(l=72, r=72, t=72, b=72),
            height=720,
        )
        iplot(fig)

