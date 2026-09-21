"""UMAP an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - UMAP - und lässt
stattdessen das Beispiel wachsen. Fünftes Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe: UMAP verbindet die Nachbarschaftsgraph-Idee von Isomap/LLE
mit der lokalen Kräfte-Optimierung von t-SNE - und soll t-SNEs Schwächen beheben. Was davon stimmt, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import umap_constants as C
from umap_evaluation import (
    Settings, analyse, convergence_rows, make_dataset, min_dist_sweep, n_neighbors_sweep, out_of_sample, stability, timing_sweep, verdict,
)
from umap_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    n_neighbors_max,
    randomize_seed,
    sync_query_params,
)
from umap_visualization import (
    build_curve,
    build_distance_fidelity,
    build_embedding,
    build_fuzzy_neighbour_view,
    build_graph_view,
    build_membership_bars,
    build_optimization,
    build_out_of_sample,
    build_stability,
    build_sweep,
    build_timing,
)

st.set_page_config(page_title="UMAP – Sebastian Hanisch", layout="wide")

STEP_LABELS = {
    1: "1 · Fuzzy-Nachbarschaft",
    2: "2 · Fuzzy-Graph & Kurve",
    3: "3 · Optimierung",
    4: "4 · Ergebnis",
}


def _md_label(value):
    return f"{value:g}"


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _sweeps(q, curvature, noise, outlier_pct):
    return n_neighbors_sweep(q, curvature, noise, outlier_pct), min_dist_sweep(q, curvature, noise, outlier_pct)


@st.cache_data(show_spinner=False)
def _stability(data_params, settings):
    return stability(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _oos(data_params, settings):
    return out_of_sample(make_dataset(*data_params), settings)


st.title("🗺️ UMAP an Lieferrouten-Kennzahlen")
st.markdown(
    """
Dieselben **12 Kennzahlen je Lieferroute** wie in der PCA-, Isomap-, LLE- und t-SNE-Demo - erzeugt aus wenigen versteckten Faktoren, aber mit **gekrümmter** Struktur, an der die PCA scheiterte.
**UMAP** (Uniform Manifold Approximation and Projection) kombiniert zwei Ideen aus den Demos davor: einen **Nachbarschaftsgraphen** (wie Isomap und LLE) - hier mit *unscharfen* Kantengewichten,
die sagen, wie sicher zwei Touren Nachbarn sind - und eine **lokale Kräfte-Optimierung** (wie t-SNE): Nachbarn ziehen sich an, zufällig gewählte Touren stoßen sich ab. Es soll t-SNEs Schwächen beheben:
**neue Touren**, **globale Struktur**, **Geschwindigkeit**, **Stabilität**. Was davon **tatsächlich** stimmt, misst die Demo direkt gegen t-SNE - und sagt es, wenn eine Erwartung nicht aufgeht.
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - fünftes Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel: UMAP ist die Fortsetzung von t-SNE und zugleich eine Konvergenz mit der Graph-Idee von Isomap/LLE; "
    "welche seiner Versprechen sich hier messen lassen, steht unten."
)

with st.expander("So funktioniert UMAP", expanded=True):
    st.markdown(
        """
UMAP (McInnes, Healy & Melville, 2018) besteht aus drei Schritten:

1. **Fuzzy-Nachbarschaftsgraph**: jede Tour bekommt ihre *n_neighbors* nächsten Nachbarn. Der **nächste** Nachbar ist immer sicher verbunden (Gewicht 1); die anderen bekommen ein Gewicht, das mit dem
   zusätzlichen Abstand exponentiell fällt - mit einer Breite σ, die pro Tour so gewählt ist, dass die Summe der Gewichte fest ist (dichte Regionen und dünne Regionen werden vergleichbar). Beide Richtungen
   werden zu einem Graphen vereinigt: ein Nachbar-Paar ist stark, wenn *eine* Seite es sicher findet.
2. **Kurve im Zielraum**: im 2-D-Bild gilt die Ähnlichkeit 1 / (1 + a·d²ᵇ) - `min_dist` legt fest, wie eng Punkte im Bild zusammenrücken dürfen (a und b werden daran angepasst).
3. **Optimierung**: Jede Kante zieht ihre beiden Touren zusammen (häufiger bei großem Gewicht); für jede Anziehung werden zufällig gewählte Touren *abgestoßen*. Die Lernrate fällt linear auf 0.
   Gestartet wird an der spektralen Einbettung des Graphen (Eigenvektoren der Laplace-Matrix) - deshalb ist UMAP deterministischer als t-SNE.

Neue Touren bekommen ihre Nachbarn unter den Trainings-Touren, starten am gewichteten Mittel und werden mit festen Trainingspunkten wenige Epochen nachoptimiert (`transform`). Was UMAP im Vergleich zu
t-SNE **wirklich** ändert, steht unten in "🆚" - gemessen, nicht behauptet.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Mit mehr Faktoren als Zielraum-Dimensionen (2) wird die Stichprobe dünner und die Einbettung schwächer.",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (dann hat UMAP keinen Vorteil vor der PCA). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider("Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05, help="Messrauschen je Kennzahl.")
    outlier_pct = st.slider(
        "Sonderfahrten (%)", *bounds("outlier_slider"), key="outlier_slider",
        help="Anteil der Touren mit extremem Zeitdruck-Faktor (zehnfach vergrößert). PCA und Isomap behalten die Größenordnung - UMAP und t-SNE stauchen die Extreme.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**UMAP**")
    nn_max = n_neighbors_max(int(n_tours))
    if st.session_state["n_neighbors_slider"] > nn_max:
        st.session_state["n_neighbors_slider"] = nn_max
    n_neighbors = st.slider(
        "n_neighbors", C.N_NEIGHBORS_MIN, nn_max, key="n_neighbors_slider",
        help="Größe der Nachbarschaft (die Tour selbst zählt mit). Sehr klein (2-3): der Graph zerfällt in Teile. Groß: lokale Details verwischen, aber im Test blieb die Einbettung stabil.",
    )
    min_dist = st.select_slider(
        "min_dist", options=C.MIN_DIST_CHOICES, key="min_dist_select", format_func=_md_label,
        help="Wie eng Punkte im Bild zusammenrücken dürfen: 0 = Klumpen, 1 = gleichmäßig verteilt. Ändert die Kurve im Zielraum (Schritt 2) - im Test kaum das R² der Faktoren.",
    )
    n_epochs = st.slider(
        "Epochen", *bounds("n_epochs_slider"), key="n_epochs_slider", step=20,
        help="Durchläufe der Optimierung. Bei wenigen Epochen rechnet die Demo einen Referenzlauf mit 500 Epochen daneben und meldet, wenn die Qualität noch fehlt.",
    )
    neg_rate = st.slider(
        "Negative-Sample-Rate", *bounds("neg_rate_slider"), key="neg_rate_slider",
        help="Wie viele zufällige Touren je Anziehungsschritt abgestoßen werden (Standard 5). Im Test änderte das R² der Faktoren zwischen 1 und 15 kaum.",
    )
    init = st.selectbox(
        "Initialisierung", C.INITS, key="init_select", format_func=lambda i: C.INIT_LABELS[i],
        help="Startlayout. Spektral ist deterministisch (Standard). Zufällige Starts zeigen, wie stark das Ergebnis vom Start abhängt (siehe Stabilität unten).",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params({
    "n_tours_slider": int(n_tours), "q_slider": int(q), "curvature_slider": curvature, "noise_slider": noise, "outlier_slider": int(outlier_pct), "n_neighbors_slider": int(n_neighbors),
    "min_dist_select": min_dist, "n_epochs_slider": int(n_epochs), "neg_rate_slider": int(neg_rate), "init_select": init, "seed_input": int(seed),
})

data_params = (int(n_tours), int(q), float(curvature), float(noise), int(outlier_pct), int(seed))
settings = Settings(n_neighbors=int(n_neighbors), min_dist=float(min_dist), n_epochs=int(n_epochs), negative_sample_rate=int(neg_rate), init=init)
with st.spinner("Baue den Fuzzy-Graphen und optimiere die Einbettung..."):
    dataset = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
model = analysis.umap
metrics = analysis.metrics
z_color = dataset.z[:, 0]
iso_idx = analysis.iso_indices
data_key = data_params + (settings,)
snap_epochs = sorted(model.snapshots)
with st.spinner("Prüfe n_neighbors und min_dist über feste Sweep-Seeds..."):
    nn_rows, md_rows = _sweeps(int(q), float(curvature), float(noise), int(outlier_pct))
level, code, vd = verdict(analysis, dataset, settings)
r2_points = convergence_rows(analysis)

if not model.connected:
    st.warning(
        f"⚠️ **Der Nachbarschaftsgraph zerfällt** (n_neighbors = {model.n_neighbors}): {model.components} Teile, {vd['isolated']} von {model.n} Touren hängen nicht am größten Teil. "
        f"Die Teile werden unverbunden nebeneinander gelegt ({model.init_used})."
    )

# --- UMAP in Aktion --------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 UMAP in Aktion")
st.caption(
    "Die 2-D-Ansicht in den Schritten 1 und 2 zeigt die Touren in den ersten beiden Hauptkomponenten (Farbe = versteckter Faktor 1) - nur als Zeichenfläche; UMAP selbst rechnet in allen 12 Dimensionen."
)
if "umap_step" not in st.session_state or st.session_state.get("umap_step_owner") != data_key:
    st.session_state["umap_step"] = 1
    st.session_state["umap_snap"] = snap_epochs[-1]
    st.session_state["umap_step_owner"] = data_key
step_col, play_col = st.columns([5, 1])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="umap_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

view = analysis.pca_2d
centre = view.mean(0)
focus = int(np.argmin(((view - centre) ** 2).sum(1)))
nbrs = model.knn_idx[focus, 1:]
w_row = model.directed[focus, nbrs]
d_row = model.knn_dist[focus, 1:]
if st.session_state.get("umap_snap") not in model.snapshots:
    st.session_state["umap_snap"] = snap_epochs[-1]
if step == 3 and not auto_play:
    snap_ep = st.select_slider("Epoche", options=snap_epochs, key="umap_snap", format_func=lambda e: f"Start ({model.init_used.split(' ')[0]})" if e == 0 else f"Epoche {e}")
else:
    snap_ep = st.session_state.get("umap_snap", snap_epochs[-1])
    if snap_ep not in model.snapshots:
        snap_ep = snap_epochs[-1]

view_slot = st.empty()


def _render(current_step, epoch=None):
    if current_step == 1:
        with view_slot.container():
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_fuzzy_neighbour_view(view, z_color, focus, nbrs, w_row), width="stretch", key="umap_view_1")
            c2.markdown("**Fuzzy-Gewichte dieser Tour**")
            c2.plotly_chart(build_membership_bars(w_row, d_row, float(model.rho[focus]), float(model.sigma[focus])), width="stretch", key="umap_membership_bars")
    elif current_step == 2:
        with view_slot.container():
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_graph_view(view, z_color, model.graph), width="stretch", key="umap_graph_view")
            c2.markdown("**Ähnlichkeitskurve im Zielraum**")
            c2.plotly_chart(build_curve(model.a, model.b, model.min_dist), width="stretch", key="umap_curve")
    elif current_step == 3:
        ep = snap_ep if epoch is None else epoch
        with view_slot.container():
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**Einbettung nach Epoche {ep}**")
            c1.plotly_chart(build_embedding(model.snapshots[ep], z_color, "Koordinate 1", "Koordinate 2"), width="stretch", key=f"umap_snapshot_{ep}")
            c2.markdown("**Optimierung**")
            c2.plotly_chart(build_optimization(model.forces, dict(r2_points), model.snapshot_loss, marker=ep), width="stretch", key=f"umap_optimization_{ep}")
    else:
        with view_slot.container():
            c1, c2 = st.columns(2)
            c1.markdown("**UMAP: Einbettung**")
            c1.plotly_chart(build_embedding(model.embedding, z_color, "UMAP-Koordinate 1", "UMAP-Koordinate 2"), width="stretch", key="umap_embed_step")
            c2.markdown("**Zum Vergleich: PCA**")
            c2.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embed_step")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for ep in snap_epochs:
                _render(3, ep)
                time.sleep(0.35)
        else:
            _render(s)
            time.sleep(1.0)
    step = 4
else:
    _render(step)

if step == 1:
    st.caption(
        f"Die gewählte Tour (Stern, nahe der Mitte): ihr nächster Nachbar (Abstand ρ = {model.rho[focus]:.2f}) hat Gewicht 1, die übrigen {len(nbrs) - 1} fallen mit Breite σ = {model.sigma[focus]:.2f} ab. "
        f"σ ist so gewählt, dass die Gewichte dieser Tour zusammen log₂({model.n_neighbors}) = {np.log2(model.n_neighbors):.2f} ergeben - für jede Tour dieselbe Summe, egal ob sie in einer dichten oder dünnen Region liegt."
    )
elif step == 2:
    st.caption(
        f"Der Graph nach der fuzzy Vereinigung beider Richtungen: {format(model.n_edges, ',').replace(',', '.')} Kanten zwischen {model.n} Touren, {model.components} Zusammenhangskomponente(n). Links die stärksten Kanten (dunkel = sicher Nachbarn). "
        f"Rechts die Kurve, die die Einbettung an `min_dist` = {model.min_dist:g} anpasst (a = {model.a:.2f}, b = {model.b:.2f})."
    )
elif step == 3:
    first, last = model.snapshot_loss[0], model.snapshot_loss[model.n_epochs]
    st.caption(
        f"Schrittlänge von Anziehung und Abstoßung fallen mit der Lernrate auf 0 - der größte Qualitätssprung passiert in den letzten Epochen. **Achtung:** die exakte Fuzzy-Kreuzentropie (rechts) sinkt hier nicht "
        f"(Start {format(first, ',.0f').replace(',', '.')} → Ende {format(last, ',.0f').replace(',', '.')}): UMAPs stichprobenbasierte Abstoßung minimiert nicht exakt diese Größe - auch die Einbettung der Referenzimplementierung liegt im Test mit 4501 über dem Start "
        f"(Standardfall, Seed 7). Literatur: Damrich & Hamprecht (2021)."
    )
else:
    st.caption("Farbe = versteckter Faktor 1. Verläuft sie in der UMAP-Einbettung glatt und ohne Überlappung, hat UMAP die Fläche entrollt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was UMAP gefunden hat - und die anderen vier Verfahren auf denselben Daten")
st.caption(
    f"Vergleichsverfahren mit ihren guten Einstellungen (aus den Demos davor): t-SNE Perplexity {C.TSNE_PERPLEXITY}, {C.TSNE_N_ITER} Iterationen; Isomap k = {C.ISOMAP_K}; LLE k = {C.LLE_K}, Regularisierung {C.LLE_REG:g}; PCA auf z-Werten."
)
if len(iso_idx) < dataset.n:
    st.warning(f"⚠️ Der Isomap-Graph ist nicht zusammenhängend: nur {len(iso_idx)} von {dataset.n} Touren sind in der Isomap-Einbettung enthalten (siehe Isomap-Demo).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² der wahren Faktoren", f"{metrics['umap']['r2']:.2f}", delta=f"{metrics['umap']['r2'] - metrics['tsne']['r2']:+.2f} ggü. t-SNE", delta_color="normal",
          help="Wie gut lassen sich die versteckten Faktoren aus den zwei Koordinaten zurückgewinnen (quadratische Regression). t-SNE mit derselben Messung im Delta.")
m2.metric("Abstandstreue ferner Paare", f"{metrics['umap']['far']:.2f}", delta=f"{metrics['umap']['far'] - metrics['tsne']['far']:+.2f} ggü. t-SNE", delta_color="normal",
          help="Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren, nur für die obere Hälfte der wahren Abstände.")
m3.metric("Trustworthiness", f"{metrics['umap']['trust']:.2f}", delta=f"{metrics['umap']['trust'] - metrics['tsne']['trust']:+.2f} ggü. t-SNE", delta_color="normal",
          help=f"Nachbarschaft erhalten: Anteil der Nachbarn in der 2-D-Einbettung, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")
m4.metric("Graph-Komponenten", f"{model.components}", help="Zusammenhangskomponenten des Fuzzy-Graphen. Mehr als 1: der Graph zerfällt, die Teile liegen in der Einbettung unverbunden nebeneinander.")

names = (("umap", "UMAP", "UMAP", model.embedding, z_color), ("tsne", "t-SNE (Vergleich)", "t-SNE", analysis.tsne.embedding, z_color),
         ("isomap", "Isomap (Vergleich)", "Isomap", analysis.iso_2d, z_color[iso_idx]), ("lle", "LLE (Vergleich)", "LLE", analysis.lle.embedding[:, :2], z_color),
         ("pca", "PCA (Vergleich)", "PC", analysis.pca_2d, z_color))
row1 = st.columns(3)
row2 = st.columns(3)
for slot, (key, title, axis, coords, color) in zip(row1 + row2[:2], names):
    with slot:
        st.markdown(f"**{title}**")
        st.plotly_chart(build_embedding(coords, color, f"{axis}-Koordinate 1" if axis != "PC" else "PC1", f"{axis}-Koordinate 2" if axis != "PC" else "PC2"), width="stretch", key=f"{key}_embedding")

order = ("umap", "tsne", "isomap", "lle", "pca")
labels = {"umap": "UMAP", "tsne": "t-SNE", "isomap": "Isomap", "lle": "LLE", "pca": "PCA"}
st.table({
    "Verfahren": [labels[k] for k in order],
    "R² der Faktoren": [f"{metrics[k]['r2']:.2f}" for k in order],
    "Abstandstreue (gesamt)": [f"{metrics[k]['fid']:.2f}" for k in order],
    "nahe Paare": [f"{metrics[k]['near']:.2f}" for k in order],
    "ferne Paare": [f"{metrics[k]['far']:.2f}" for k in order],
    "Trustworthiness": [f"{metrics[k]['trust']:.2f}" for k in order],
})

st.markdown("---")

# --- n_neighbors und min_dist -----------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängen die Ergebnisse von n_neighbors und min_dist ab?")
st.markdown(
    """
*n_neighbors* bestimmt, wie viel der Umgebung jede Tour 'sieht', *min_dist* wie eng Punkte im Bild zusammenrücken dürfen. Live für Ihr aktuelles Szenario über **feste Sweep-Seeds**
(unabhängig vom Demo-Seed) geprüft, nicht behauptet:
"""
)
if code == "disconnected":
    st.warning(
        f"⚠️ **Der Graph zerfällt**: mit n_neighbors = {vd['n_neighbors']} gibt es {vd['components']} Komponenten, {vd['isolated']} Touren hängen nicht am größten Teil - R² der Faktoren nur {vd['r2']:.2f} "
        f"(t-SNE {vd['r2_tsne']:.2f}). Die Kurve unten zeigt, ab welchem n_neighbors der Graph zusammenhängt."
    )
elif code == "not_converged":
    st.warning(
        f"⚠️ **Noch nicht konvergiert**: nach {vd['n_epochs']} Epochen liegt die Trustworthiness bei {vd['trust']:.2f}, ein Referenzlauf mit {C.CONVERGED_EPOCHS} Epochen erreicht {vd['trust_alt']:.2f} "
        f"(R² {vd['r2']:.2f} gegen {vd['r2_alt']:.2f}). Die Qualität entsteht vor allem in den letzten Epochen, wenn die Lernrate gegen 0 geht."
    )
elif code == "global_structure":
    st.warning(
        f"⚠️ **Keine bessere globale Struktur als t-SNE**: mit {vd['outlier_pct']} % Sonderfahrten liegt das R² der Faktoren bei {vd['r2']:.2f} (t-SNE {vd['r2_tsne']:.2f}, PCA {vd['r2_pca']:.2f}, Isomap {vd['r2_iso']:.2f}), "
        f"die Abstandstreue ferner Paare bei {vd['far']:.2f} (PCA {vd['far_pca']:.2f}). UMAP erhält wie t-SNE Nachbarschaften, nicht die Größe von Abständen - die Extreme werden an den Rand gestaucht. "
        "(Das R² wird hier von den Sonderfahrten dominiert - genau darum geht es.)"
    )
elif code == "no_advantage":
    st.info(f"ℹ️ **Kein Vorteil vor der PCA**: die Daten sind gerade (Krümmung 0) - R² der Faktoren {vd['r2']:.2f} (UMAP) gegen {vd['r2_pca']:.2f} (PCA); ferne Paare {vd['far']:.2f} gegen {vd['far_pca']:.2f}.")
elif code == "umap_wins":
    st.success(
        f"✅ **UMAP entrollt die Fläche**: R² der Faktoren {vd['r2']:.2f} gegen {vd['r2_pca']:.2f} bei der PCA (t-SNE {vd['r2_tsne']:.2f}, Isomap {vd['r2_iso']:.2f}, LLE {vd['r2_lle']:.2f}), Trustworthiness {vd['trust']:.2f}. "
        f"Bei fernen Paaren: Abstandstreue {vd['far']:.2f} (t-SNE {vd['far_tsne']:.2f}, Isomap {vd['far_iso']:.2f})."
    )
else:
    st.info(f"UMAP erreicht R² {vd['r2']:.2f} (PCA {vd['r2_pca']:.2f}, t-SNE {vd['r2_tsne']:.2f}) - kein klarer Gewinn und kein klarer Bruch.")

st.markdown("**n_neighbors**")
st.plotly_chart(build_sweep(nn_rows, "n_neighbors", float(n_neighbors), "n_neighbors", components=True), width="stretch", key="n_neighbors_sweep")
best_nn = max(nn_rows, key=lambda r: r["r2"])
st.caption(
    f"Gleiche Daten-Einstellungen (q = {dataset.q}, Krümmung {curvature:.2f}, Rauschen {noise:.2f}, Sonderfahrten {int(outlier_pct)} %), nur n_neighbors wächst; min_dist {C.DEFAULT_MIN_DIST:g}, {C.SWEEP_N_EPOCHS} Epochen, "
    f"spektraler Start; Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren. Bestes R²: {best_nn['r2']:.2f} bei n_neighbors = {best_nn['n_neighbors']}. "
    "Ein Fenster nach oben gibt es hier nicht - der Einbruch liegt bei sehr kleinen Werten, an denen der Graph zerfällt."
)
st.markdown("**min_dist**")
st.plotly_chart(build_sweep(md_rows, "min_dist", float(min_dist), "min_dist"), width="stretch", key="min_dist_sweep")
st.caption(
    f"Gleiche Einstellungen, nur min_dist wächst; n_neighbors = {C.DEFAULT_N_NEIGHBORS}. min_dist verändert vor allem das *Aussehen* (Klumpung), nicht die Nachbarschaften: "
    "R² und Trustworthiness sind fast flach."
)

st.markdown("**Konvergenz** (aktueller Lauf)")
st.plotly_chart(build_optimization(model.forces, dict(r2_points), model.snapshot_loss), width="stretch", key="convergence_curve")
st.caption(f"Schrittlänge je Epoche, R² der wahren Faktoren und exakte Fuzzy-Kreuzentropie an den {len(r2_points)} Schnappschüssen dieses Laufs ({model.n_epochs} Epochen).")

st.markdown("---")

# --- Was UMAP gegenüber t-SNE ändert ---------------------------------------------------------------------------------------

st.markdown("## 🆚 Was UMAP gegenüber t-SNE ändert - gemessen")
st.markdown(
    """
| Versprechen gegenüber t-SNE | Ergebnis im Test (300 Touren; Out-of-sample und Stabilität über feste Seeds, sonst Seed 7) |
|---|---|
| **Neue Touren einbetten** | ✅ `transform`: R² 0.88–0.93 gegen 0.69–0.85 bei der t-SNE-Näherung; beim Neu-Rechnen verschieben sich die Trainings-Touren nur um Procrustes 0.03–0.15 |
| **Stabilität** (Start egal) | ✅ bei q = 3: zufällige Starts weichen im Median um Procrustes 0.02–0.17 voneinander ab, bei t-SNE um 0.56–0.76 (4 Datensätze). Bei q = 2 gemischt (UMAP 0.02–0.32, t-SNE 0.24–0.41), einzelne UMAP-Starts weichen stark ab |
| **Geschwindigkeit** | ✅ bei großem n (n = 600: 1.3 s gegen 4.9 s); bei n = 100-200 ist t-SNE schneller, der Wechsel liegt zwischen 200 und 400 Touren |
| **Globale Struktur** | ❌ Abstandstreue ferner Paare 0.56 gegen 0.69 (t-SNE) - nicht besser |
| **Sonderfahrten / Extreme** | ❌ 5 %: R² 0.14 gegen 0.12 (t-SNE), PCA 0.76 - dieselbe Schwäche |

Die Experimente mit Knopf unten prüfen die drei ✅ für Ihre Einstellungen nach.
"""
)

rng = np.random.default_rng(0)
m_iso = len(iso_idx)
pa = rng.integers(0, m_iso, size=min(1500, m_iso * (m_iso - 1) // 2))
pb = rng.integers(0, m_iso, size=len(pa))
keep = pa != pb
pa, pb = pa[keep], pb[keep]
ga, gb = iso_idx[pa], iso_idx[pb]


def _norm(d):
    return d / d.mean()


latent = _norm(np.linalg.norm(dataset.z[ga] - dataset.z[gb], axis=1))
umap_d = _norm(np.linalg.norm(model.embedding[ga] - model.embedding[gb], axis=1))
tsne_d = _norm(np.linalg.norm(analysis.tsne.embedding[ga] - analysis.tsne.embedding[gb], axis=1))
iso_d = _norm(np.linalg.norm(analysis.iso_2d[pa] - analysis.iso_2d[pb], axis=1))
st.markdown("**📏 Globale Struktur: Abstände**")
st.plotly_chart(build_distance_fidelity(latent, [("UMAP", umap_d, "#8e5fbf"), ("t-SNE", tsne_d, "#d62728"), ("Isomap", iso_d, "#1f77b4")]), width="stretch", key="distance_fidelity")
st.caption(
    f"Jeder Punkt ein Tourenpaar: Abstand in der 2-D-Einbettung gegen den Abstand der wahren Faktoren; auf der gestrichelten Diagonale wäre die Einbettung abstandstreu. Korrelation für **nahe** Paare: "
    f"UMAP {metrics['umap']['near']:.2f}, t-SNE {metrics['tsne']['near']:.2f}, Isomap {metrics['isomap']['near']:.2f} - für **ferne** Paare: UMAP {metrics['umap']['far']:.2f}, t-SNE {metrics['tsne']['far']:.2f}, "
    f"Isomap {metrics['isomap']['far']:.2f}. Sonderfahrten (Regler links) machen den Unterschied zu PCA und Isomap drastisch sichtbar."
)

st.markdown("**🆕 Neue Touren einbetten (Out-of-sample)**")
st.caption(
    "UMAP hat für neue Touren `transform`: Nachbarn unter den Trainings-Touren, Start am gewichteten Mittel, wenige Epochen mit festen Trainingspunkten. t-SNE hat keine Abbildung - hier die Behelfsnäherung "
    f"(gewichteter Mittelwert der 10 nächsten Trainings-Touren). Test: die letzten {C.HOLDOUT_FRACTION * 100:.0f} % der Touren zurückhalten; dazu, wie stark sich die bereits eingebetteten Touren beim Neu-Rechnen mit allen verschieben."
)
if st.button("🆕 Neue Touren testen", key="oos_start"):
    st.session_state["oos_on"] = True
if st.session_state.get("oos_on"):
    with st.spinner("Rechne UMAP und t-SNE ohne und mit den neuen Touren..."):
        oos = _oos(data_params, settings)
    st.plotly_chart(build_out_of_sample(oos["model"].embedding, oos["y_umap"], oos["tsne_train"], oos["y_tsne"], dataset.z[oos["train"], 0], dataset.z[oos["test"], 0]), width="stretch", key="oos_plot")
    st.caption(
        f"Sterne = zurückgehaltene Touren. R² der wahren Faktoren für die neuen Touren: **UMAP {oos['r2_umap']:.2f}**, t-SNE-Näherung {oos['r2_tsne']:.2f} (Trainings-Touren: {oos['r2_train']:.2f}). "
        f"Beim Neu-Rechnen mit allen Touren verschieben sich die Trainings-Touren um einen Procrustes-Abstand von {oos['shift_umap']:.2f} (UMAP) bzw. {oos['shift_tsne']:.2f} (t-SNE); 0 = unverändert."
    )

st.markdown("**🔀 Stabilität: derselbe Datensatz, verschiedene Starts**")
st.caption(
    "Die spektrale Einbettung ist deterministisch, die Optimierung zufällig gestartet - wie stark hängt das Bild vom Start ab? Sechs Läufe (spektral, PCA, vier zufällige Starts; die Initialisierung links spielt hier keine Rolle); "
    "Abweichung per Procrustes-Abstand zum spektralen Lauf nach bester Drehung/Spiegelung (0 = gleiches Bild, 1 = unabhängig)."
)
if st.button("🔀 Sechs Starts rechnen", key="stability_start"):
    st.session_state["stability_on"] = True
if st.session_state.get("stability_on"):
    with st.spinner("Rechne sechs UMAP-Läufe..."):
        stab = _stability(data_params, settings)
    labels_s = [f"{C.INIT_LABELS[i].split(' (')[0]}: R² {r:.2f}" for i, r in zip(stab["inits"], stab["r2"])]
    st.plotly_chart(build_stability(stab["embeddings"], z_color, labels_s), width="stretch", key="stability_plot")
    st.caption(
        f"Procrustes-Abstand zum spektralen Lauf: {', '.join(f'{p:.2f}' for p in stab['procrustes'][1:])}; R² zwischen {min(stab['r2']):.2f} und {max(stab['r2']):.2f}. "
        + ("Alle Starts finden praktisch dasselbe Bild." if max(stab["procrustes"]) < 0.1 else "Einzelne Starts unterscheiden sich sichtbar - meist eine gedrehte oder umgeklappte Teilstruktur.")
    )

st.markdown("**⏱️ Rechenzeit**")
st.caption(
    "t-SNE berechnet in jeder Iteration alle n² Paar-Ähnlichkeiten, UMAP je Epoche nur die Kanten des Graphen (etwa n·n_neighbors) plus wenige Zufalls-Paare - dafür braucht UMAP zusätzlich den Graphen und die spektrale Einbettung "
    "(dichte n×n-Eigenzerlegung). Bei kleinem n gewinnt t-SNE, bei großem UMAP."
)
if "timing_rows" not in st.session_state:
    if st.button("⏱️ Rechenzeit messen (ca. 15 s)", key="timing_start", help=f"Misst UMAP ({C.TIMING_N_EPOCHS} Epochen), t-SNE ({C.TSNE_N_ITER} Iterationen), Isomap, LLE und PCA für n = {', '.join(str(n) for n in C.TIMING_NS)} auf diesem Rechner."):
        with st.spinner("Messe..."):
            st.session_state["timing_rows"] = timing_sweep()
        st.rerun()
else:
    rows = st.session_state["timing_rows"]
    st.plotly_chart(build_timing(rows), width="stretch", key="timing_chart")
    st.table({
        "Touren n": [r["n"] for r in rows],
        "UMAP": [f"{r['umap']:.2f} s" for r in rows],
        "t-SNE": [f"{r['tsne']:.2f} s" for r in rows],
        "Isomap": [f"{r['isomap']:.3f} s" for r in rows],
        "LLE": [f"{r['lle']:.3f} s" for r in rows],
        "PCA": [f"{r['pca'] * 1000:.2f} ms" for r in rows],
        "t-SNE / UMAP": [f"{r['tsne'] / max(r['umap'], 1e-9):.1f}×" for r in rows],
    })
    st.caption(f"Gemessen auf diesem Rechner (Wandzeit, {C.TIMING_N_EPOCHS} Epochen bzw. {C.TSNE_N_ITER} Iterationen, ein Lauf je n): die Faktoren hängen von Rechner und Zwischenspeichern ab.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Fuzzy-Nachbarschaft.** Für Punkte $x_i \in \mathbb{R}^{12}$ (z-Werte) mit den $k$ nächsten Nachbarn $N(i)$ (die Tour selbst zählt mit) sei $\rho_i$ der Abstand zum nächsten *anderen* Punkt. Das Gewicht der Kante von $i$ zu $j$ ist

$$
w_{j|i} = \exp\!\Big(-\frac{\max(0,\ \lVert x_i - x_j \rVert - \rho_i)}{\sigma_i}\Big),
$$

wobei $\sigma_i$ per Binärsuche so bestimmt wird, dass $\sum_{j \in N(i)\setminus\{i\}} w_{j|i} = \log_2 k$. Die beiden Richtungen werden per **fuzzy Vereinigung** (probabilistische Summe) zusammengeführt:
$w_{ij} = w_{j|i} + w_{i|j} - w_{j|i}\, w_{i|j}$.

**Zielraum.** Die Ähnlichkeit zweier Bildpunkte ist $v_{ij} = \big(1 + a\,\lVert y_i - y_j\rVert^{2b}\big)^{-1}$; $a, b$ werden per Kurvenanpassung (Levenberg-Marquardt) so bestimmt, dass $v$ eine Stufe bei
`min_dist` (dann $\exp(-(d - \text{min\_dist}))$) bestmöglich trifft.

**Zielfunktion (nominell).** Die fuzzy Kreuzentropie

$$
C = \sum_{i<j} \Big[\, w_{ij} \log\frac{w_{ij}}{v_{ij}} + (1 - w_{ij}) \log\frac{1 - w_{ij}}{1 - v_{ij}} \,\Big].
$$

**Optimierung.** Stochastischer Gradientenabstieg über die Kanten: Anziehung entlang einer Kante mit Schritt $-\dfrac{2ab\,d^{2(b-1)}}{1 + a\,d^{2b}}(y_i - y_j)$, Abstoßung von $r$ (`negative_sample_rate`) zufällig
gezogenen Punkten $k$ mit Schritt $\dfrac{2b}{(0.001 + d^2)\,(1 + a\,d^{2b})}(y_i - y_k)$, jeweils auf $[-4, 4]$ begrenzt; Lernrate linear von 1 auf 0. Starke Kanten werden häufiger gezogen (Häufigkeit $\propto w_{ij}$).
Die Demo aktualisiert je Epoche alle aktiven Kanten gleichzeitig (vektorisiert), nicht nacheinander wie die Referenzimplementierung.

**Neue Punkte.** Ein neuer Punkt bekommt Gewichte $w_j$ zu seinen $k$ nächsten Trainings-Touren (ohne Selbst-Eintrag), startet am gewichteten Mittel $\sum_j \bar w_j y_j$ und wird $\lfloor n_\text{epochs}/3 \rfloor$ Epochen mit festen Trainingspunkten optimiert.

**Grenzen.** (1) *Keine belastbare globale Struktur*: die Zielfunktion misst Nachbarschaften; ferne Regionen und Extremwerte werden gestaucht (Demo: Sonderfahrten). (2) *Die tatsächlich optimierte Größe* ist nicht exakt die Kreuzentropie
(Damrich & Hamprecht, 2021) - die exakte Kreuzentropie sinkt in der Demo nicht. (3) *Nachbarschaftsgraph*: bei zu kleinem $k$ zerfällt er (Demo: n_neighbors = 2-3). (4) *Rechenzeit*: hier exaktes kNN und dichte
spektrale Einbettung, $O(n^2)$ bzw. $O(n^3)$ - für große Datensätze gibt es Näherungen (Nearest-Neighbor-Descent, sparse Eigenlöser), die hier nicht gebaut sind. (5) Achsen und Abstände im Bild haben keine feste Bedeutung.

**Trustworthiness** (Venna & Kaski, 2001): $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$ mit $U_i$ = Nachbarn in der Einbettung, die im Originalraum keine sind, und $r(i,j)$ ihrem Originalrang.
**Abstandstreue** = Pearson-Korrelation der Paarabstände der 2-D-Einbettung mit den Paarabständen der wahren Faktoren (nah/fern: untere/obere Hälfte der wahren Abstände). **Procrustes-Abstand**: $1 - (\sum s_i)^2$ mit $s_i$ den
Singulärwerten von $A^\top B$ nach Zentrierung und Normierung beider Einbettungen (wie `scipy.spatial.procrustes`).

Implementiert in `umap_algorithm.py` (Graph, Kurve, Initialisierung, Optimierung, Transform), `umap_tsne.py` / `umap_isomap.py` / `umap_lle.py` (Vergleichsverfahren, wortgleich aus tsne-demo / isomap-demo / lle-demo),
`umap_scenario.py` (Lieferrouten-Generator, wortgleich aus pca-demo) und `umap_evaluation.py` (Kennzahlen, Sweeps, Verdict, Zeitmessung).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
