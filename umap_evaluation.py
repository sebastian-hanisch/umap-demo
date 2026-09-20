"""Auswertung: findet UMAP die wahren Faktoren zurück - und behebt es wirklich, was man ihm gegenüber t-SNE nachsagt? Alle Kennzahlen werden am Datensatz gemessen; die wahren latenten Faktoren z sind
bekannt (Lieferrouten-Erzeugung, inkl. Sonderfahrten). UMAP, t-SNE, Isomap, LLE (Vergleichsverfahren mit ihren guten Einstellungen) und PCA werden mit denselben Messungen bewertet.

- **R² der wahren Faktoren**: Rekonstruktion von z aus den zwei Koordinaten per quadratischer Regression (monotone Umparametrisierungen werden nicht bestraft).
- **Abstandstreue** (gesamt / nahe Paare / ferne Paare): Pearson-Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren; "nah" = untere Hälfte der wahren
  Paarabstände, "fern" = obere Hälfte.
- **Trustworthiness** (Venna & Kaski): bleiben Nachbarn Nachbarn?
- **Fuzzy-Kreuzentropie**: die Größe, die UMAP nominell minimiert (siehe App: sie sinkt in der Praxis nicht monoton)."""

import time
from dataclasses import dataclass

import numpy as np

import umap_constants as C
from umap_algorithm import fit_umap, transform
from umap_isomap import fit_isomap, pairwise_distances, standardize
from umap_lle import fit_lle
from umap_scenario import generate_dataset
from umap_tsne import embed_new_naive, fit_tsne, procrustes_disparity


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def _quad_features(e):
    return np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])


def r2_quadratic(coords2, z):
    """R² der Rekonstruktion von z aus zwei Koordinaten (quadratische Regression, Mittel über die Faktoren, gewichtet mit ihrer Varianz)."""
    A = _quad_features(coords2)
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(1.0 - (z - A @ beta).var(0).sum() / z.var(0).sum())


def pca_project(X, n_components=2):
    Z = standardize(X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:n_components].T


def distance_fidelity(coords2, z):
    iu = np.triu_indices(len(z), 1)
    return float(np.corrcoef(pairwise_distances(coords2)[iu], pairwise_distances(z)[iu])[0, 1])


def distance_fidelity_split(coords2, z):
    """(nahe Paare, ferne Paare): Abstandstreue getrennt für die untere und die obere Hälfte der wahren Paarabstände."""
    iu = np.triu_indices(len(z), 1)
    dz = pairwise_distances(z)[iu]
    dc = pairwise_distances(coords2)[iu]
    near = dz <= np.median(dz)
    return float(np.corrcoef(dz[near], dc[near])[0, 1]), float(np.corrcoef(dz[~near], dc[~near])[0, 1])


def make_dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


def _metrics(coords, z, Z):
    near, far = distance_fidelity_split(coords, z)
    return {"r2": r2_quadratic(coords, z), "fid": distance_fidelity(coords, z), "near": near, "far": far, "trust": trustworthiness(Z, coords)}


@dataclass(frozen=True)
class Settings:
    n_neighbors: int = C.DEFAULT_N_NEIGHBORS
    min_dist: float = C.DEFAULT_MIN_DIST
    n_epochs: int = C.DEFAULT_N_EPOCHS
    negative_sample_rate: int = C.DEFAULT_NEG_RATE
    init: str = C.DEFAULT_INIT                       # "spectral", "pca" oder "random:<Start>"


def _with(settings, **changes):
    return Settings(**{**settings.__dict__, **changes})


def run_umap(X, s):
    kind, _, start = s.init.partition(":")
    return fit_umap(X, s.n_neighbors, s.min_dist, s.n_epochs, s.negative_sample_rate, kind, int(start) if start else 0)


@dataclass(frozen=True)
class Analysis:
    umap: object
    alt: object                      # bei wenigen Epochen: derselbe Lauf mit CONVERGED_EPOCHS Epochen (Konvergenz-Prüfung), sonst None
    tsne: object
    isomap: object
    lle: object
    iso_2d: np.ndarray
    pca_2d: np.ndarray
    iso_indices: np.ndarray
    metrics: dict                    # {"umap", "tsne", "isomap", "lle", "pca", "alt"?} -> {"r2","fid","near","far","trust"}
    snapshot_r2: dict                # Epoche -> R² der Einbettung zu diesem Zeitpunkt


def analyse(dataset, settings):
    Z = standardize(dataset.X)
    model = run_umap(dataset.X, settings)
    tsne = fit_tsne(dataset.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    iso = fit_isomap(dataset.X, C.ISOMAP_K, 2)
    lle = fit_lle(dataset.X, C.LLE_K, 2, C.LLE_REG)
    pca2 = pca_project(dataset.X)
    iso2 = iso.embedding[:, :2]
    metrics = {
        "umap": _metrics(model.embedding, dataset.z, Z),
        "tsne": _metrics(tsne.embedding, dataset.z, Z),
        "isomap": _metrics(iso2, dataset.z[iso.indices], Z[iso.indices]),
        "lle": _metrics(lle.embedding[:, :2], dataset.z, Z),
        "pca": _metrics(pca2, dataset.z, Z),
    }
    alt = None
    if settings.n_epochs < 200:
        alt = run_umap(dataset.X, _with(settings, n_epochs=C.CONVERGED_EPOCHS))
        metrics["alt"] = _metrics(alt.embedding, dataset.z, Z)
    snap = {e: r2_quadratic(y, dataset.z) for e, y in model.snapshots.items() if e > 0}
    return Analysis(umap=model, alt=alt, tsne=tsne, isomap=iso, lle=lle, iso_2d=iso2, pca_2d=pca2, iso_indices=iso.indices, metrics=metrics, snapshot_r2=snap)


def verdict(analysis, dataset, settings):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten)."""
    m = analysis.metrics
    model = analysis.umap
    data = {"r2": m["umap"]["r2"], "r2_tsne": m["tsne"]["r2"], "r2_iso": m["isomap"]["r2"], "r2_lle": m["lle"]["r2"], "r2_pca": m["pca"]["r2"], "far": m["umap"]["far"],
            "far_tsne": m["tsne"]["far"], "far_pca": m["pca"]["far"], "far_iso": m["isomap"]["far"], "trust": m["umap"]["trust"], "n_neighbors": settings.n_neighbors,
            "n_epochs": settings.n_epochs, "components": model.components, "outlier_pct": dataset.outlier_pct, "isolated": _isolated(model)}
    if "alt" in m:
        data.update({"trust_alt": m["alt"]["trust"], "r2_alt": m["alt"]["r2"]})
    if not model.connected:
        return "warning", "disconnected", data
    if "alt" in m and m["alt"]["trust"] - m["umap"]["trust"] >= 0.02:
        return "warning", "not_converged", data
    if dataset.outlier_pct > 0 and m["pca"]["far"] - m["umap"]["far"] >= 0.25 and m["pca"]["r2"] - m["umap"]["r2"] >= 0.15:
        return "warning", "global_structure", data
    if dataset.curvature == 0 and m["umap"]["r2"] - m["pca"]["r2"] < 0.03:
        return "info", "no_advantage", data
    if m["umap"]["r2"] - m["pca"]["r2"] >= 0.10:
        return "success", "umap_wins", data
    return "info", "neutral", data


def _isolated(model):
    """Zahl der Touren, die nicht in der größten Zusammenhangskomponente des Graphen liegen (bei zerfallenem Graphen)."""
    graph = model.graph > 0
    n = len(graph)
    label = -np.ones(n, dtype=int)
    sizes = []
    for start in range(n):
        if label[start] >= 0:
            continue
        stack, label[start], size = [start], len(sizes), 1
        while stack:
            node = stack.pop()
            for nxt in np.nonzero(graph[node] & (label < 0))[0]:
                label[nxt] = len(sizes)
                stack.append(int(nxt))
                size += 1
        sizes.append(size)
    return int(n - max(sizes))


def n_neighbors_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, n_epochs=C.SWEEP_N_EPOCHS, values=C.SWEEP_N_NEIGHBORS, seeds=C.SWEEP_SEEDS):
    """Feste Sweep-Seeds, min_dist/Init/Negative-Rate wie im Standard: je n_neighbors mittleres R², Abstandstreue (ferne Paare), Trustworthiness und mittlere Zahl der Graph-Komponenten."""
    rows = []
    for k in values:
        acc = {"r2": [], "far": [], "trust": [], "components": []}
        for seed in seeds:
            ds = make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
            model = fit_umap(ds.X, k, C.DEFAULT_MIN_DIST, n_epochs)
            m = _metrics(model.embedding, ds.z, standardize(ds.X))
            acc["r2"].append(m["r2"]), acc["far"].append(m["far"]), acc["trust"].append(m["trust"]), acc["components"].append(model.components)
        rows.append({"n_neighbors": int(k), **{key: float(np.mean(v)) for key, v in acc.items()}})
    return rows


def min_dist_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, n_epochs=C.SWEEP_N_EPOCHS, values=C.SWEEP_MIN_DISTS, seeds=C.SWEEP_SEEDS):
    """Wie `n_neighbors_sweep`, aber über min_dist (n_neighbors = Standard)."""
    rows = []
    for md in values:
        acc = {"r2": [], "far": [], "trust": []}
        for seed in seeds:
            ds = make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
            model = fit_umap(ds.X, C.DEFAULT_N_NEIGHBORS, md, n_epochs)
            m = _metrics(model.embedding, ds.z, standardize(ds.X))
            acc["r2"].append(m["r2"]), acc["far"].append(m["far"]), acc["trust"].append(m["trust"])
        rows.append({"min_dist": float(md), **{key: float(np.mean(v)) for key, v in acc.items()}})
    return rows


def convergence_rows(analysis):
    """(Epoche, R²) an den Schnappschüssen des aktuellen Laufs - ohne Extra-Rechnung."""
    return sorted(analysis.snapshot_r2.items())


def stability(dataset, settings, inits=C.STABILITY_INITS):
    """Derselbe Datensatz, dieselben Einstellungen, verschiedene Initialisierungen (Basis: spektral): -> Einbettungen, R², Procrustes-Abstände zur spektralen Basis."""
    base = run_umap(dataset.X, _with(settings, init="spectral"))
    runs = [run_umap(dataset.X, _with(settings, init=i)) for i in inits]
    return {"inits": ["spectral"] + list(inits), "embeddings": [base.embedding] + [r.embedding for r in runs],
            "r2": [r2_quadratic(base.embedding, dataset.z)] + [r2_quadratic(r.embedding, dataset.z) for r in runs],
            "procrustes": [0.0] + [procrustes_disparity(r.embedding, base.embedding) for r in runs]}


def out_of_sample(dataset, settings, fraction=C.HOLDOUT_FRACTION):
    """Die letzten `fraction` der Touren zurückhalten: UMAP-`transform` gegen die Näherung für t-SNE (`embed_new_naive`), dazu wie stark sich die Trainings-Touren verschieben, wenn man alles neu rechnet."""
    n = dataset.n
    n_test = max(10, int(round(fraction * n)))
    train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    z_test = dataset.z[test]
    umap_train = run_umap(dataset.X[train], settings)
    y_umap = transform(umap_train, dataset.X[test])
    beta, *_ = np.linalg.lstsq(_quad_features(umap_train.embedding), dataset.z[train], rcond=None)
    r2_umap = float(1 - (z_test - _quad_features(y_umap) @ beta).var(0).sum() / z_test.var(0).sum())
    umap_full = run_umap(dataset.X, settings)
    tsne_train = fit_tsne(dataset.X[train], C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    y_tsne = embed_new_naive(tsne_train, dataset.X[test], 10)
    beta_t, *_ = np.linalg.lstsq(_quad_features(tsne_train.embedding), dataset.z[train], rcond=None)
    r2_tsne = float(1 - (z_test - _quad_features(y_tsne) @ beta_t).var(0).sum() / z_test.var(0).sum())
    tsne_full = fit_tsne(dataset.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    return {"train": train, "test": test, "model": umap_train, "tsne_train": tsne_train.embedding, "y_umap": y_umap, "y_tsne": y_tsne, "r2_umap": r2_umap, "r2_tsne": r2_tsne,
            "r2_train": r2_quadratic(umap_train.embedding, dataset.z[train]),
            "shift_umap": procrustes_disparity(umap_full.embedding[train], umap_train.embedding), "shift_tsne": procrustes_disparity(tsne_full.embedding[train], tsne_train.embedding)}


def timing_sweep(ns=C.TIMING_NS, n_epochs=C.TIMING_N_EPOCHS, seed=100_000):
    """Gemessene Rechenzeit (Sekunden) von UMAP, t-SNE, Isomap, LLE und PCA für wachsende n (eigene Messung, Rechner-abhängig)."""
    rows = []
    for n in ns:
        ds = generate_dataset(n, 2, 0.0, 0.1, 0, seed)
        out = {"n": int(n)}
        for name, fn in (("umap", lambda: fit_umap(ds.X, C.DEFAULT_N_NEIGHBORS, C.DEFAULT_MIN_DIST, n_epochs)), ("tsne", lambda: fit_tsne(ds.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)),
                         ("isomap", lambda: fit_isomap(ds.X, C.ISOMAP_K, 2)), ("lle", lambda: fit_lle(ds.X, C.LLE_K, 2, C.LLE_REG)), ("pca", lambda: pca_project(ds.X))):
            t = time.perf_counter()
            fn()
            out[name] = time.perf_counter() - t
        rows.append(out)
    return rows
