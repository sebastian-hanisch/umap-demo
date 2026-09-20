"""LLE (Locally Linear Embedding, Roweis & Saul 2000) von Grund auf (numpy).

1. **Nachbarn**: jede Tour bekommt ihre k nächsten Nachbarn (gerichtet, nicht symmetrisiert) im Merkmalsraum (z-Werte).
2. **Rekonstruktionsgewichte**: jede Tour wird als AFFINE Kombination ihrer Nachbarn rekonstruiert, x_i ≈ Σ_j w_ij x_j mit Σ_j w_ij = 1. Mit der lokalen
   Gram-Matrix G = (x_j − x_i)(x_l − x_i)^T ist die Lösung w ∝ G^{-1} 1. Bei k > d (mehr Nachbarn als Merkmale) ist G singulär und die Lösung nicht
   eindeutig: LLE braucht dann eine Regularisierung G += reg · tr(G) · I (Konvention wie sklearn). Ohne Regularisierung meldet `fit_lle` den Fehler
   `SingularNeighbourhood`, statt still zu reparieren.
3. **Einbettung**: die Koordinaten Y minimieren Σ_i ‖y_i − Σ_j w_ij y_j‖² bei Mittelwert 0 und Einheitsvarianz - die Eigenvektoren zu den kleinsten Eigenwerten
   von M = (I − W)^T (I − W), ohne den konstanten (Eigenwert 0). Skalierung wie sklearn: Y = sqrt(n) · Eigenvektoren.

Anders als Isomap erhält LLE keine Abstände, nur die lokale Geometrie der Nachbarschaften - und es hat eine natürliche Erweiterung auf neue Punkte (`embed_new`)."""

from dataclasses import dataclass

import numpy as np

from umap_isomap import pairwise_distances, standardize


class SingularNeighbourhood(ValueError):
    """Die lokale Gram-Matrix einer Tour ist singulär (k > Dimension) und die Regularisierung ist 0."""


def nearest_neighbours(dist, k):
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def _weights_for(point, neighbours, reg):
    """Rekonstruktionsgewichte (Summe 1) einer Tour aus ihren Nachbarn; -> (w, Rekonstruktionsfehler, singulär?)"""
    N = neighbours - point                                  # [k, d]
    G = N @ N.T
    k = len(G)
    trace = float(np.trace(G))
    values = np.linalg.eigvalsh(G)
    singular = bool(values[0] <= 1e-12 * max(values[-1], 1e-300))
    if reg > 0:
        G = G + reg * (trace if trace > 0 else 1.0) * np.eye(k)
    elif singular:
        raise SingularNeighbourhood(f"lokale Gram-Matrix singulär (k = {k} > Rang {int((values > 1e-12 * max(values[-1], 1e-300)).sum())}) und Regularisierung = 0")
    w = np.linalg.solve(G, np.ones(k))
    w = w / w.sum()
    error = float(((w @ neighbours - point) ** 2).sum())
    return w, error, singular


def reconstruction_weights(Z, neighbour_index, reg):
    """Gewichtsmatrix W [n, n] (Zeile i: Gewichte der Nachbarn von i), Rekonstruktionsfehler je Tour, Anteil singulärer Nachbarschaften."""
    n, k = neighbour_index.shape
    W = np.zeros((n, n))
    errors = np.zeros(n)
    singular = 0
    for i in range(n):
        w, err, sing = _weights_for(Z[i], Z[neighbour_index[i]], reg)
        W[i, neighbour_index[i]] = w
        errors[i] = err
        singular += sing
    return W, errors, singular / n


def component_labels(neighbour_index):
    """Zusammenhangskomponenten des (symmetrisierten) Nachbarschaftsgraphen; Label 0 = größte."""
    n = len(neighbour_index)
    adjacent = np.zeros((n, n), dtype=bool)
    adjacent[np.arange(n)[:, None], neighbour_index] = True
    adjacent |= adjacent.T
    label = -np.ones(n, dtype=int)
    current = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = current
        while stack:
            node = stack.pop()
            for nxt in np.nonzero(adjacent[node] & (label < 0))[0]:
                label[nxt] = current
                stack.append(int(nxt))
        current += 1
    sizes = np.bincount(label)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[label]


@dataclass(frozen=True)
class LLEModel:
    embedding: np.ndarray          # [n, n_components]
    eigenvalues: np.ndarray        # kleinste Eigenwerte von M (aufsteigend, inkl. des konstanten), höchstens N_SPECTRUM
    W: np.ndarray                  # [n, n] Rekonstruktionsgewichte
    neighbours: np.ndarray         # [n, k]
    errors: np.ndarray             # Rekonstruktionsfehler je Tour
    singular_share: float
    components: np.ndarray         # Komponenten-Label des Nachbarschaftsgraphen (0 = größte)
    mean: np.ndarray               # Standardisierung der Trainingsdaten (für embed_new)
    scale: np.ndarray
    Z: np.ndarray                  # standardisierte Trainingsdaten
    k: int
    reg: float

    @property
    def n(self):
        return len(self.Z)

    @property
    def reconstruction_error(self):
        return float(self.errors.sum())

    @property
    def connected(self):
        return int(self.components.max()) == 0


N_SPECTRUM = 12


def fit_lle(X, k, n_components=10, reg=1e-3):
    X = np.asarray(X, dtype=float)
    mean = X.mean(0)
    scale = X.std(0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    dist = pairwise_distances(Z)
    nbrs = nearest_neighbours(dist, k)
    W, errors, singular_share = reconstruction_weights(Z, nbrs, reg)
    A = np.eye(len(Z)) - W
    M = A.T @ A
    M = (M + M.T) / 2
    values, vectors = np.linalg.eigh(M)
    n_components = min(n_components, len(Z) - 1)
    embedding = vectors[:, 1:n_components + 1] * np.sqrt(len(Z))
    return LLEModel(embedding=embedding, eigenvalues=values[:N_SPECTRUM], W=W, neighbours=nbrs, errors=errors, singular_share=singular_share,
                    components=component_labels(nbrs), mean=mean, scale=scale, Z=Z, k=k, reg=reg)


def embed_new(model, X_new):
    """Out-of-sample: neue Touren über ihre Rekonstruktionsgewichte aus den k nächsten TRAININGS-Touren einbetten, y = Σ_j w_j y_j."""
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    d = np.sqrt(np.maximum(((Zn[:, None, :] - model.Z[None, :, :]) ** 2).sum(-1), 0.0))
    nbrs = np.argsort(d, axis=1)[:, :model.k]
    out = np.zeros((len(Zn), model.embedding.shape[1]))
    for i in range(len(Zn)):
        w, _, _ = _weights_for(Zn[i], model.Z[nbrs[i]], model.reg)
        out[i] = w @ model.embedding[nbrs[i]]
    return out
