"""Plotly-Visualisierungen der UMAP-Demo: Fuzzy-Nachbarschaft einer Tour, Fuzzy-Graph, Zielraum-Kurve, Einbettungen, Optimierungsverlauf, Sweeps, Abstandstreue, Stabilität, Out-of-sample und Rechenzeit.
Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _scatter(coords, color, name="Touren", size=7, showscale=False, label="latenter Faktor 1", opacity=1.0):
    return go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name=name, hoverinfo="skip",
        marker=dict(color=color, colorscale="Viridis", size=size, showscale=showscale, opacity=opacity, line=dict(width=0.5, color="white"),
                    colorbar=dict(title=label) if showscale else None),
    )


def build_fuzzy_neighbour_view(coords, color, focus, neighbours, weights):
    """2-D-Ansicht (erste zwei Hauptkomponenten): gewählte Tour und ihre Nachbarn, Markergröße = Fuzzy-Gewicht w_j|i."""
    sizes = 8 + 30 * weights / max(weights.max(), 1e-12)
    fig = go.Figure(_scatter(coords, color, showscale=True, opacity=0.5))
    fig.add_trace(go.Scatter(x=coords[neighbours, 0], y=coords[neighbours, 1], mode="markers", name="Nachbarn (Größe = Gewicht)", hoverinfo="skip",
                             marker=dict(color=ORANGE, size=sizes, line=dict(width=1, color="#14233B"))))
    fig.add_trace(go.Scatter(x=[coords[focus, 0]], y=[coords[focus, 1]], mode="markers", name="gewählte Tour", hoverinfo="skip",
                             marker=dict(color=RED, size=14, symbol="star", line=dict(width=1, color="#14233B"))))
    fig.update_xaxes(title="PC1 (Ansicht)")
    fig.update_yaxes(title="PC2 (Ansicht)")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_membership_bars(weights, distances, rho, sigma):
    """Fuzzy-Gewichte w_j|i der Nachbarn einer Tour (nach Nähe geordnet) und ihre Abstände: bis zum Abstand ρ (nächster Nachbar) ist das Gewicht 1, danach fällt es exponentiell mit der Breite σ."""
    ranks = list(range(1, len(weights) + 1))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=ranks, y=weights, marker_color=BLUE, name="Gewicht w_j|i", hovertemplate="Nachbar %{x}: %{y:.2f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=ranks, y=distances, mode="lines+markers", line=dict(color=ORANGE, width=3), name="Abstand", hovertemplate="Abstand %{y:.2f}<extra></extra>"), secondary_y=True)
    fig.add_hline(y=rho, line_dash="dash", line_color=GRAY, secondary_y=True, annotation_text=f"ρ = {rho:.2f}", annotation_position="top left")
    fig.update_xaxes(title=f"Nachbarn dieser Tour, nach Nähe geordnet (σ = {sigma:.2f})")
    fig.update_yaxes(title_text="Gewicht", range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="Abstand (z-Einheiten)", secondary_y=True)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_graph_view(coords, color, graph, max_edges=1500):
    """Fuzzy-Graph in der 2-D-Ansicht: starke Kanten (Gewicht > 0.8) dunkel, mittlere hell, schwache (< 0.3) sehr blass. Nur die stärksten `max_edges` Kanten werden gezeichnet."""
    iu = np.triu_indices(len(graph), 1)
    w = graph[iu]
    order = np.argsort(-w)
    order = order[w[order] > 0][:max_edges]
    rows, cols, ws = iu[0][order], iu[1][order], w[order]
    fig = go.Figure()
    for lo, hi, alpha, width, name in ((0.8, 1.01, 0.55, 1.6, "stark (> 0.8)"), (0.3, 0.8, 0.28, 1.0, "mittel (0.3 – 0.8)"), (0.0, 0.3, 0.12, 0.8, "schwach (< 0.3)")):
        sel = (ws >= lo) & (ws < hi)
        xs, ys = [], []
        for i, j in zip(rows[sel], cols[sel]):
            xs += [coords[i, 0], coords[j, 0], None]
            ys += [coords[i, 1], coords[j, 1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=f"rgba(60,60,60,{alpha})", width=width), name=name, hoverinfo="skip"))
    fig.add_trace(_scatter(coords, color, showscale=True))
    fig.update_xaxes(title="PC1 (Ansicht)")
    fig.update_yaxes(title="PC2 (Ansicht)")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_curve(a, b, min_dist):
    """Die Ähnlichkeitskurve im Zielraum 1/(1 + a·d^{2b}) gegen die Zielkurve (1 bis min_dist, dann exp(−(d − min_dist)))."""
    x = np.linspace(0, 3, 300)
    target = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=target, mode="lines", name="Ziel (min_dist)", line=dict(color=GRAY, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=1.0 / (1.0 + a * x ** (2 * b)), mode="lines", name=f"Anpassung: a = {a:.2f}, b = {b:.2f}", line=dict(color=ORANGE, width=4)))
    fig.update_xaxes(title="Abstand d im Zielraum")
    fig.update_yaxes(title="Ähnlichkeit", range=[0, 1.05])
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_embedding(coords, color, title_x, title_y, label="latenter Faktor 1", height=380):
    fig = go.Figure(_scatter(coords, color, showscale=True, label=label))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_optimization(forces, snapshot_r2, snapshot_loss, marker=None):
    """Verlauf der Optimierung: mittlere Schrittlänge von Anziehung und Abstoßung je Epoche (Lernrate fällt linear), R² der wahren Faktoren und exakte Fuzzy-Kreuzentropie an den Schnappschüssen."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Schrittlänge je Epoche", "R² der wahren Faktoren", "Fuzzy-Kreuzentropie (exakt)"))
    epochs = np.arange(1, len(forces) + 1)
    fig.add_trace(go.Scatter(x=epochs, y=np.maximum(forces[:, 0], 1e-6), mode="lines", line=dict(color=BLUE, width=3), name="Anziehung"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=np.maximum(forces[:, 1], 1e-6), mode="lines", line=dict(color=RED, width=3), name="Abstoßung"), row=1, col=1)
    r2 = sorted(snapshot_r2.items())
    fig.add_trace(go.Scatter(x=[e for e, _ in r2], y=[v for _, v in r2], mode="lines+markers", line=dict(color=ORANGE, width=3), showlegend=False,
                             hovertemplate="Epoche %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    ce = sorted(snapshot_loss.items())
    fig.add_trace(go.Scatter(x=[max(e, 1) for e, _ in ce], y=[v for _, v in ce], mode="lines+markers", line=dict(color=PURPLE, width=3), showlegend=False,
                             hovertemplate="Epoche %{x}: %{y:.0f}<extra></extra>"), row=1, col=3)
    if marker is not None:
        for col in (1, 2, 3):
            fig.add_vline(x=max(marker, 1), line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(type="log", col=1)
    fig.update_yaxes(range=[0, 1.02], col=2)
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_sweep(rows, xkey, current, xtitle, components=False):
    """Sweep über n_neighbors (mit Graph-Komponenten) oder min_dist: R², Abstandstreue ferner Paare und Trustworthiness."""
    xs = [r[xkey] for r in rows]
    titles = ["R² der wahren Faktoren", "Abstandstreue (fern) und Trustworthiness"] + (["Komponenten des Graphen"] if components else [])
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=titles)
    fig.add_trace(go.Scatter(x=xs, y=[r["r2"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="R²"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["far"] for r in rows], mode="lines+markers", line=dict(color=RED, width=3), name="Abstandstreue (fern)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=[r["trust"] for r in rows], mode="lines+markers", line=dict(color=GREEN, width=3), name="Trustworthiness"), row=1, col=2)
    if components:
        fig.add_trace(go.Scatter(x=xs, y=[r["components"] for r in rows], mode="lines+markers", line=dict(color=BLUE, width=3), name="Komponenten"), row=1, col=3)
        fig.update_yaxes(type="log", col=3)
    for col in range(1, len(titles) + 1):
        if current is not None:
            fig.add_vline(x=current, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text=xtitle, tickvals=xs, ticktext=[f"{x:g}" for x in xs], type="log" if components else "linear")
    fig.update_yaxes(range=[-0.1, 1.02], col=1)
    fig.update_yaxes(range=[-0.1, 1.02], col=2)
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_distance_fidelity(latent_pairs, panels):
    """Paarabstände der 2-D-Einbettung gegen die Abstände der wahren Faktoren (je auf Mittelwert 1 normiert). `panels`: [(Titel, Abstände, Farbe)]; auf der Diagonalen ist die Einbettung abstandstreu."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, pairs, color) in enumerate(panels, start=1):
        top = float(max(latent_pairs.max(), pairs.max())) * 1.05
        fig.add_trace(go.Scatter(x=latent_pairs, y=pairs, mode="markers", marker=dict(color=color, size=5, opacity=0.35), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GRAY, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Abstand der wahren Faktoren (normiert)")
    fig.update_yaxes(title_text="Abstand in der Einbettung (normiert)", col=1)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_stability(embeddings, color, labels):
    """Sechs Läufe mit verschiedenen Initialisierungen (2 × 3), Basis (spektral) zuerst."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=labels, vertical_spacing=0.16)
    for i, emb in enumerate(embeddings):
        fig.add_trace(go.Scatter(x=emb[:, 0], y=emb[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=color, colorscale="Viridis", size=4, line=dict(width=0.3, color="white"))), row=i // 3 + 1, col=i % 3 + 1)
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=50, b=10))
    return lock_axes(fig)


def build_out_of_sample(umap_train, umap_test, tsne_train, tsne_test, train_color, test_color):
    """Zurückgehaltene Touren (Sterne) in der UMAP-Einbettung (`transform`) und in der t-SNE-Einbettung (Näherung)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("UMAP: transform", "t-SNE: Näherung (gewichteter Mittelwert)"))
    for col, (train, test) in enumerate(((umap_train, umap_test), (tsne_train, tsne_test)), start=1):
        fig.add_trace(go.Scatter(x=train[:, 0], y=train[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=train_color, colorscale="Viridis", size=5, opacity=0.5)), row=1, col=col)
        fig.add_trace(go.Scatter(x=test[:, 0], y=test[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=test_color, colorscale="Viridis", cmin=float(train_color.min()), cmax=float(train_color.max()), size=11, symbol="star",
                                             line=dict(width=1.2, color="#14233B"))), row=1, col=col)
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_timing(rows):
    ns = np.array([r["n"] for r in rows], dtype=float)
    fig = go.Figure()
    for key, label, color in (("umap", "UMAP", PURPLE), ("tsne", "t-SNE", RED), ("isomap", "Isomap", BLUE), ("lle", "LLE", ORANGE), ("pca", "PCA", GREEN)):
        fig.add_trace(go.Scatter(x=ns, y=[max(r[key], 1e-6) for r in rows], mode="lines+markers", name=label, line=dict(color=color, width=3)))
    fig.update_xaxes(title="Anzahl Touren n", type="log")
    fig.update_yaxes(title="Rechenzeit (s)", type="log")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)
