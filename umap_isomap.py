"""Isomap-Kern (wortgleich aus isomap-demo kopiert) - hier nur der Vergleichspartner von LLE. Isomap von Grund auf (numpy): Nachbarschaftsgraph → kürzeste Wege (geodätische Abstände) → klassisches MDS.

1. **kNN-Graph**: jeder Punkt wird mit seinen k nächsten Nachbarn verbunden (symmetrisiert), Kantengewicht = euklidischer Abstand.
2. **Kürzeste Wege** (Floyd-Warshall, vektorisiert über den Zwischenknoten, O(n³)): die Länge des kürzesten Weges im Graphen nähert den
   geodätischen Abstand auf der Fläche an.
3. **Klassisches MDS**: B = −½ J D² J (J = Zentriermatrix), Eigenzerlegung, Koordinaten = sqrt(λ)·v.

Ist der Graph nicht zusammenhängend, gibt es zwischen den Teilen keinen endlichen Weg: `fit_isomap` bettet dann die GRÖSSTE Komponente ein und
meldet die übrigen Punkte - es repariert nichts still."""

from dataclasses import dataclass

import numpy as np


def standardize(X):
    X = np.asarray(X, dtype=float)
    scale = X.std(0, ddof=1)
    return (X - X.mean(0)) / np.where(scale > 0, scale, 1.0)


def pairwise_distances(Z):
    sq = (Z ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    D = np.sqrt(np.maximum(d2, 0.0))
    D = (D + D.T) / 2.0                     # exakt symmetrisch, Diagonale exakt 0 (die Gram-Formel hat Rundungsfehler ~1e-8)
    np.fill_diagonal(D, 0.0)
    return D


def knn_graph(dist, k):
    """Symmetrische kNN-Adjazenz: (i, j) ist Kante, wenn j unter den k nächsten Nachbarn von i liegt ODER umgekehrt.
    -> Gewichtsmatrix (inf = keine Kante, 0 auf der Diagonale) und Kantenliste [(i, j), i < j]."""
    n = len(dist)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.argsort(d, axis=1)[:, :k]
    adjacent = np.zeros((n, n), dtype=bool)
    adjacent[np.arange(n)[:, None], nearest] = True
    adjacent |= adjacent.T
    weights = np.where(adjacent, dist, np.inf)
    np.fill_diagonal(weights, 0.0)
    ii, jj = np.nonzero(np.triu(adjacent, 1))
    return weights, np.stack([ii, jj], axis=1)


def shortest_paths(weights):
    """Floyd-Warshall: kürzeste Wege zwischen allen Knotenpaaren. O(n³), je Zwischenknoten eine vektorisierte n×n-Aktualisierung."""
    D = weights.copy()
    for m in range(len(D)):
        np.minimum(D, D[:, m:m + 1] + D[m:m + 1, :], out=D)
    return D


def shortest_path_route(weights, D, source, target):
    """Knotenfolge des kürzesten Weges (für die Anzeige) - aus der fertigen Abstandsmatrix rekonstruiert."""
    route = [source]
    current = source
    for _ in range(len(D)):
        if current == target:
            break
        neighbours = np.nonzero(np.isfinite(weights[current]) & (np.arange(len(D)) != current))[0]
        best = neighbours[np.argmin(weights[current, neighbours] + D[neighbours, target])]
        route.append(int(best))
        current = int(best)
    return route


def connected_components(finite_distance):
    """Komponenten-Label je Knoten aus der (endlich/unendlich) Abstandsmatrix. Größte Komponente hat Label 0."""
    n = len(finite_distance)
    label = -np.ones(n, dtype=int)
    current = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        members = np.nonzero(finite_distance[start])[0]
        label[members] = current
        current += 1
    sizes = np.bincount(label)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[label]


def classical_mds(D, n_components):
    """Klassisches MDS auf einer Abstandsmatrix. -> (Koordinaten [n, n_components], Eigenwerte absteigend)."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = (B + B.T) / 2
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    top = np.maximum(values[:n_components], 0.0)
    return vectors[:, :n_components] * np.sqrt(top), values


@dataclass(frozen=True)
class IsomapResult:
    embedding: np.ndarray        # [m, n_components] für die eingebetteten Punkte
    indices: np.ndarray          # [m] Indizes (in X) der eingebetteten Punkte (größte Komponente)
    geodesic: np.ndarray         # [m, m] geodätische Abstände dieser Punkte
    weights: np.ndarray          # [n, n] Graphgewichte (inf = keine Kante)
    edges: np.ndarray            # [E, 2] Kanten (i < j)
    euclid: np.ndarray           # [n, n] euklidische Abstände (z-Werte)
    components: np.ndarray       # [n] Komponenten-Label (0 = größte)
    eigenvalues: np.ndarray
    k: int

    @property
    def n(self):
        return len(self.weights)

    @property
    def connected(self):
        return int(self.components.max()) == 0

    @property
    def n_dropped(self):
        return self.n - len(self.indices)


def fit_isomap(X, k, n_components=10):
    Z = standardize(X)
    dist = pairwise_distances(Z)
    weights, edges = knn_graph(dist, k)
    D = shortest_paths(weights)
    components = connected_components(np.isfinite(D))
    indices = np.nonzero(components == 0)[0]
    geodesic = D[np.ix_(indices, indices)]
    embedding, values = classical_mds(geodesic, min(max(n_components, 1), len(indices) - 1))
    return IsomapResult(embedding=embedding, indices=indices, geodesic=geodesic, weights=weights, edges=edges, euclid=dist,
                        components=components, eigenvalues=values, k=k)


def residual_variance(geodesic, embedding_full_coords, n_dims):
    """Isomap-'Scree': 1 − R² zwischen geodätischen Abständen und den Abständen der ersten d Koordinaten, für d = 1..n_dims (Tenenbaum et al. 2000)."""
    iu = np.triu_indices(len(geodesic), 1)
    g = geodesic[iu]
    out = []
    for d in range(1, n_dims + 1):
        e = pairwise_distances(embedding_full_coords[:, :d])[iu]
        out.append(1.0 - float(np.corrcoef(g, e)[0, 1]) ** 2)
    return np.array(out)
