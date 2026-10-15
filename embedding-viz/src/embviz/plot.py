from __future__ import annotations

import html
import textwrap
from pathlib import Path

import numpy as np

from .core import Corpus


def _wrap(s: str, width: int = 80, max_chars: int = 400) -> str:
    s = " ".join(s.split())[:max_chars]
    return "<br>".join(textwrap.wrap(html.escape(s), width))


def corpus_map(
    corpus: Corpus,
    coords: np.ndarray,
    *,
    color_by: str | list = "source",
    queries: list[str] | None = None,
    query_coords: np.ndarray | None = None,
    title: str = "Corpus map",
):
    """Plotly figure: one point per chunk (hover shows text), optional query overlay."""
    import plotly.graph_objects as go

    if isinstance(color_by, str):
        groups = [str(m.get(color_by, "?")) for m in corpus.metadata]
    else:
        groups = [str(g) for g in color_by]
    hover = [
        f"<b>{html.escape(i)}</b> · {html.escape(str(m.get('source', '')))}"
        f"{' p.' + str(m['page']) if 'page' in m else ''}<br>{_wrap(t)}"
        for i, t, m in zip(corpus.ids, corpus.texts, corpus.metadata)
    ]
    fig = go.Figure()
    for g in sorted(set(groups), key=lambda x: (len(x), x)):
        idx = [i for i, x in enumerate(groups) if x == g]
        fig.add_trace(go.Scattergl(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=g[-40:],
            marker={"size": 6, "opacity": 0.75}, hovertext=[hover[i] for i in idx], hoverinfo="text",
        ))
    if queries is not None and query_coords is not None:
        fig.add_trace(go.Scattergl(
            x=query_coords[:, 0], y=query_coords[:, 1], mode="markers", name="queries",
            marker={"size": 9, "symbol": "x", "color": "black", "line": {"width": 1}},
            hovertext=[_wrap(q) for q in queries], hoverinfo="text",
        ))
    fig.update_layout(
        title=title, template="plotly_white", height=760,
        legend={"itemsizing": "constant"}, xaxis={"visible": False}, yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def save_html(fig, path: str | Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
