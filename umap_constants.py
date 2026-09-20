"""Defaults, Slider-Grenzen und Presets für die UMAP-Demo. Merkmale und Erzeugungs-Konstanten sind wortgleich aus pca-demo übernommen
(dieselben Lieferrouten - dieselbe gekrümmte Fläche, an der PCA scheiterte); alles Übrige ist neu."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 1.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_OUTLIER_PCT = 0
OUTLIER_PCT_MIN, OUTLIER_PCT_MAX = 0, 10
DEFAULT_N_NEIGHBORS = 15
N_NEIGHBORS_MIN, N_NEIGHBORS_MAX = 2, 100          # die obere Grenze folgt zusätzlich der Tourenzahl (n_neighbors < n)
MIN_DIST_CHOICES = (0.0, 0.1, 0.25, 0.5, 0.8, 1.0)
DEFAULT_MIN_DIST = 0.1
DEFAULT_N_EPOCHS = 500
N_EPOCHS_MIN, N_EPOCHS_MAX = 20, 500
DEFAULT_NEG_RATE = 5
NEG_RATE_MIN, NEG_RATE_MAX = 1, 15
INITS = ("spectral", "pca") + tuple(f"random:{i}" for i in range(5))
INIT_LABELS = {"spectral": "spektral (deterministisch)", "pca": "PCA", **{f"random:{i}": f"zufällig, Start {i + 1}" for i in range(5)}}
DEFAULT_INIT = "spectral"
DEFAULT_SEED = 7

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)


# --- Auswertung --------------------------------------------------------------------------------------------------------
TRUST_NEIGHBORS = 10
TSNE_PERPLEXITY, TSNE_N_ITER = 30, 500             # Vergleichsverfahren mit ihren guten Einstellungen (siehe tsne-demo / isomap-demo / lle-demo)
ISOMAP_K = 10
LLE_K, LLE_REG = 14, 1e-2
SWEEP_SEEDS = tuple(100_000 + i for i in range(3))                 # feste Sweep-Seeds, unabhängig vom Demo-Seed
SWEEP_N_NEIGHBORS = (2, 3, 5, 8, 15, 30, 60)
SWEEP_MIN_DISTS = (0.0, 0.1, 0.25, 0.5, 1.0)
SWEEP_N_TOURS = 200
SWEEP_N_EPOCHS = 200
STABILITY_INITS = ("pca", "random:0", "random:1", "random:2", "random:3")
HOLDOUT_FRACTION = 0.2
TIMING_NS = (100, 200, 400, 600)
TIMING_N_EPOCHS = 500
SNAPSHOT_COUNT = 12
CONVERGED_EPOCHS = 500                             # Referenzlauf für die Konvergenz-Prüfung bei wenigen Epochen

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": DEFAULT_Q, "curvature": DEFAULT_CURVATURE, "noise": DEFAULT_NOISE, "outlier_pct": DEFAULT_OUTLIER_PCT, "n_neighbors": DEFAULT_N_NEIGHBORS,
         "min_dist": DEFAULT_MIN_DIST, "n_epochs": DEFAULT_N_EPOCHS, "negative_sample_rate": DEFAULT_NEG_RATE, "init": DEFAULT_INIT, "seed": DEFAULT_SEED}
PRESETS = {
    "Gekrümmte Fläche: UMAP entrollt": {**_BASE},
    "Sonderfahrten: dieselbe Schwäche wie t-SNE": {**_BASE, "outlier_pct": 5},
    "n_neighbors zu klein: Graph zerfällt": {**_BASE, "n_neighbors": 3},
    "Zu wenige Epochen": {**_BASE, "n_epochs": 20},
    "Rauschen: UMAP hält": {**_BASE, "noise": 0.8},
    "Gerade Daten: kein Vorteil": {**_BASE, "curvature": 0.0},
}
PRESET_HELP = {
    "Gekrümmte Fläche: UMAP entrollt": "Dieselbe gebogene Fläche wie in den Demos davor: UMAP gewinnt die versteckten Faktoren gut zurück (R² ≈ 0.88 gegen 0.50 der PCA; t-SNE 0.93, Isomap 0.98, LLE 0.96) - bei fernen Tourenpaaren aber nur mit Abstandstreue 0.56 (t-SNE 0.69).",
    "Sonderfahrten: dieselbe Schwäche wie t-SNE": "5 % Sonderfahrten mit extremen Werten: PCA (R² ≈ 0.76) und Isomap (0.75) behalten die Größenordnung, UMAP fällt auf 0.14 - fast genau wie t-SNE (0.12). Die oft genannte 'bessere globale Struktur' zeigt sich hier nicht.",
    "n_neighbors zu klein: Graph zerfällt": "Mit nur 3 Nachbarn je Tour (die Tour selbst zählt mit) zerfällt der Nachbarschaftsgraph in 8 Teile, 73 von 300 Touren hängen nicht am Hauptteil: R² der Faktoren 0.34, die Teile liegen unverbunden nebeneinander.",
    "Zu wenige Epochen": "Nach nur 20 Epochen hat die Optimierung noch nicht konvergiert: Trustworthiness 0.94 statt 0.98, Abstandstreue ferner Paare 0.45 statt 0.56 - die Demo rechnet einen Referenzlauf mit 500 Epochen daneben. (Der Effekt ist klein und datensatzabhängig: der Trustworthiness-Abstand lag über fünf Datensätze zwischen 0.01 und 0.04.)",
    "Rauschen: UMAP hält": "Mit viel Rauschen (0.8) liegt UMAP bei R² ≈ 0.89 - vor t-SNE (0.85), Isomap (0.84), LLE (0.58) und PCA (0.45). Der Graph mittelt über viele Nachbarn.",
    "Gerade Daten: kein Vorteil": "Krümmung 0: die Kennzahlen hängen linear von den Faktoren ab, die PCA ist optimal (R² 0.98) - UMAP erreicht 0.91 und kostet ein Vielfaches an Rechenzeit.",
}
PRESET_EXPECTED_BANDS = {
    "Gekrümmte Fläche: UMAP entrollt": {"verdict": "umap_wins", "r2": (0.75, 0.97), "r2_tsne": (0.8, 0.99), "r2_iso": (0.94, 1.0), "r2_pca": (0.4, 0.6), "far": (0.3, 0.75), "far_tsne": (0.4, 0.85)},
    "Sonderfahrten: dieselbe Schwäche wie t-SNE": {"verdict": "global_structure", "r2": (-0.2, 0.5), "r2_tsne": (-0.2, 0.5), "r2_pca": (0.6, 0.9), "r2_iso": (0.6, 0.9), "far": (-0.5, 0.35), "far_pca": (0.85, 1.0)},
    "n_neighbors zu klein: Graph zerfällt": {"verdict": "disconnected", "components": (5, 12), "isolated": (40, 110), "r2": (0.15, 0.55)},
    "Zu wenige Epochen": {"verdict": "not_converged", "trust": (0.88, 0.97), "trust_alt": (0.95, 1.0), "far": (0.15, 0.7)},
    "Rauschen: UMAP hält": {"verdict": "umap_wins", "r2": (0.7, 0.97), "r2_tsne": (0.65, 0.95), "r2_iso": (0.7, 0.95), "r2_lle": (0.4, 0.7), "r2_pca": (0.3, 0.6)},
    "Gerade Daten: kein Vorteil": {"verdict": "no_advantage", "r2": (0.85, 0.97), "r2_pca": (0.94, 1.0)},
}
