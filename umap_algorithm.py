"""UMAP (Uniform Manifold Approximation and Projection, McInnes, Healy & Melville 2018) von Grund auf (numpy, exaktes kNN, für n ≤ 600).

Folgt der Referenzimplementierung `umap-learn` (Zahlen dort nachgeschlagen, im Test gegengeprüft):

1. **Fuzzy-Nachbarschaftsgraph** (Echo von Isomap/LLE): je Tour die n_neighbors nächsten Nachbarn (die Tour selbst zählt mit, wie in umap-learn). ρ_i = Abstand zum nächsten *anderen* Punkt - der nächste Nachbar ist
   damit sicher verbunden (Gewicht 1); σ_i per Binärsuche so, dass Σ_j exp(−(d_ij − ρ_i)/σ_i) = log₂(n_neighbors). Gewichte w_j|i = exp(−max(0, d_ij − ρ_i)/σ_i); beide Richtungen werden per
   **fuzzy Vereinigung** zusammengeführt: w_ij = w_j|i + w_i|j − w_j|i · w_i|j.
2. **Kurve im Zielraum**: 1 / (1 + a·d^{2b}), a und b so angepasst, dass sie eine Stufe bei `min_dist` (dann exponentiell fallend) möglichst gut trifft.
3. **Optimierung** (Echo von t-SNE): stochastischer Gradientenabstieg über die Kanten des Graphen - **Anziehung** entlang jeder Kante (häufiger für große Gewichte), **Abstoßung** von zufällig gezogenen
   Punkten (`negative_sample_rate` je Anziehungsschritt); Lernrate linear fallend; Start: spektrale Einbettung des Graphen (oder zufällig / PCA).
4. **Neue Punkte** (`transform`): Nachbarn unter den Trainings-Touren, Start am gewichteten Mittel ihrer Koordinaten, dann wenige Epochen mit festen Trainingspunkten.

Anders als die Referenzimplementierung wird je Epoche über alle aktiven Kanten gleichzeitig (Jacobi-artig, vektorisiert) statt nacheinander aktualisiert - Ergebnisse sind gleichwertig, nicht bitgleich."""

from dataclasses import dataclass

import numpy as np

from umap_isomap import pairwise_distances, standardize

MIN_K_DIST_SCALE = 1e-3
SMOOTH_TOLERANCE = 1e-5
CLIP = 4.0


# --- Kurve a, b --------------------------------------------------------------------------------------------------------------

def fit_ab(min_dist, spread=1.0, n_iter=200):
    """Levenberg-Marquardt-Anpassung von 1/(1 + a·x^{2b}) an die Zielkurve (1 bis min_dist, dann exp(−(x − min_dist)/spread)); Start (1, 1) wie scipy.curve_fit in umap-learn."""
    x = np.linspace(0.0, spread * 3.0, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    p = np.array([1.0, 1.0])
    lam = 1e-3

    def model(p):
        return 1.0 / (1.0 + p[0] * x ** (2.0 * p[1]))

    def jac(p):
        u = x ** (2.0 * p[1])
        denom = (1.0 + p[0] * u) ** 2
        logx = np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), 0.0)
        return np.column_stack([-u / denom, -p[0] * u * 2.0 * logx / denom])

    cost = float(((model(p) - y) ** 2).sum())
    for _ in range(n_iter):
        r = y - model(p)
        J = jac(p)
        A, g = J.T @ J, J.T @ r
        candidate = p + np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)      # gedämpfter Gauß-Newton-Schritt: (JᵀJ + λ·diag) δ = Jᵀ r
        if not np.all(np.isfinite(candidate)) or candidate[0] <= 0 or candidate[1] <= 0:
            lam *= 10
            continue
        new_cost = float(((1.0 / (1.0 + candidate[0] * x ** (2.0 * candidate[1])) - y) ** 2).sum())
        if new_cost < cost:
            done = abs(cost - new_cost) < 1e-14 * max(cost, 1e-30)
            p, cost, lam = candidate, new_cost, max(lam / 10, 1e-12)
            if done:
                break
        else:
            lam *= 10
            if lam > 1e12:
                break
    return float(p[0]), float(p[1])


# --- Fuzzy-Graph --------------------------------------------------------------------------------------------------------------

def knn(dist, k):
    """k nächste Nachbarn je Zeile inklusive des Punktes selbst an erster Stelle (wie umap-learn). -> (Indizes [n,k], Abstände [n,k])."""
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    others = np.argsort(d, axis=1, kind="stable")[:, :k - 1]
    order = np.column_stack([np.arange(len(dist)), others])
    return order, np.take_along_axis(dist, order, axis=1)


def smooth_knn(distances, target, n_steps=64):
    """ρ und σ je Zeile für Abstände [n,k] (erste Spalte = Selbstabstand 0). -> (ρ [n], σ [n]); alle Zeilen gleichzeitig, wie `smooth_knn_dist` in umap-learn (local_connectivity = 1)."""
    n, k = distances.shape
    rest = distances[:, 1:]
    positive = np.where(rest > 0, rest, np.inf)
    rho = positive.min(1)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    lo, hi, mid = np.zeros(n), np.full(n, np.inf), np.ones(n)
    for _ in range(n_steps):
        d = rest - rho[:, None]
        psum = np.where(d > 0, np.exp(-d / mid[:, None]), 1.0).sum(1)
        settled = np.abs(psum - target) < SMOOTH_TOLERANCE
        too_big = psum > target
        hi = np.where(~settled & too_big, mid, hi)
        lo = np.where(~settled & ~too_big, mid, lo)
        new_mid = np.where(too_big, (lo + hi) / 2.0, np.where(np.isinf(hi), mid * 2.0, (lo + hi) / 2.0))
        mid = np.where(settled, mid, new_mid)
    mean_row = distances.mean(1)
    mean_all = float(distances.mean())
    floor = np.where(rho > 0, MIN_K_DIST_SCALE * mean_row, MIN_K_DIST_SCALE * mean_all)
    return rho, np.maximum(mid, floor)


def membership(distances, rho, sigma):
    """Gewichte w_j|i = exp(−max(0, d_ij − ρ_i)/σ_i) für Abstände [n,k] (Abstände bis ρ bekommen Gewicht 1; der Selbst-Eintrag wird von `fuzzy_graph` auf 0 gesetzt)."""
    d = distances - rho[:, None]
    w = np.where(d > 0, np.exp(-np.where(d > 0, d, 0.0) / sigma[:, None]), 1.0)
    return w


def fuzzy_graph(dist, n_neighbors):
    """-> (directed [n,n] w_j|i, symmetric [n,n] fuzzy Vereinigung, knn_idx, knn_dist, rho, sigma)."""
    n = len(dist)
    idx, kd = knn(dist, n_neighbors)
    rho, sigma = smooth_knn(kd, np.log2(n_neighbors))
    w = membership(kd, rho, sigma)
    w[:, 0] = 0.0
    directed = np.zeros((n, n))
    directed[np.arange(n)[:, None], idx] = w
    np.fill_diagonal(directed, 0.0)
    sym = directed + directed.T - directed * directed.T
    return directed, sym, idx, kd, rho, sigma


# --- Initialisierung ------------------------------------------------------------------------------------------------------------

def spectral_init(graph, seed):
    """Eigenvektoren der normalisierten Laplace-Matrix (zweit- und drittkleinster Eigenwert) des Fuzzy-Graphen, auf ±10 skaliert + Rauschen. Nicht zusammenhängend: -> None."""
    n = len(graph)
    if _components(graph) > 1:
        return None
    deg = graph.sum(1)
    inv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    L = np.eye(n) - inv[:, None] * graph * inv[None, :]
    values, vectors = np.linalg.eigh((L + L.T) / 2)
    Y = vectors[:, 1:3]
    return Y * (10.0 / np.abs(Y).max()) + np.random.default_rng(seed).normal(scale=1e-4, size=(n, 2))


def _components(graph):
    n = len(graph)
    label = -np.ones(n, dtype=int)
    count = 0
    for start in range(n):
        if label[start] >= 0:
            continue
        stack = [start]
        label[start] = count
        while stack:
            node = stack.pop()
            for nxt in np.nonzero((graph[node] > 0) & (label < 0))[0]:
                label[nxt] = count
                stack.append(int(nxt))
        count += 1
    return count


def graph_components(graph):
    return _components(graph)


def initial_embedding(Z, graph, init, seed):
    """-> (Einbettung [n,2], tatsächlich verwendete Initialisierung). `init`: spectral | random | pca; nicht zusammenhängende Graphen fallen bei spectral auf random zurück (wie umap-learn ohne Mehrkomponenten-Layout)."""
    n = len(Z)
    rng = np.random.default_rng(seed)
    if init == "spectral":
        Y = spectral_init(graph, seed)
        if Y is not None:
            return Y, "spectral"
        return rng.uniform(-10, 10, size=(n, 2)), "random (Graph nicht zusammenhängend)"
    if init == "random":
        return rng.uniform(-10, 10, size=(n, 2)), "random"
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    Y = Zc @ vt[:2].T
    return Y * (10.0 / np.abs(Y).max()) + rng.normal(scale=1e-4, size=(n, 2)), "pca"


# --- Optimierung ----------------------------------------------------------------------------------------------------------------

def _clip(g):
    return np.clip(g, -CLIP, CLIP)


def attraction_coefficient(d2, a, b):
    """Faktor vor (y_i − y_j) beim Anziehungsschritt einer Kante: −2ab·d^{2(b−1)} / (1 + a·d^{2b}) (Abstand d, d2 = d²); Gradient von −log v(d) mit Vorzeichen des Abstiegs."""
    out = np.zeros_like(d2)
    pos = d2 > 0
    out[pos] = -2.0 * a * b * d2[pos] ** (b - 1.0) / (a * d2[pos] ** b + 1.0)
    return out


def repulsion_coefficient(d2, a, b, gamma=1.0):
    """Faktor vor (y_i − y_k) beim Abstoßungsschritt: 2γb / ((0.001 + d²)(1 + a·d^{2b})) (die 0.001 verhindert eine Division durch 0); Abstieg auf −log(1 − v(d))."""
    out = np.zeros_like(d2)
    pos = d2 > 0
    out[pos] = 2.0 * gamma * b / ((0.001 + d2[pos]) * (a * d2[pos] ** b + 1.0))
    return out


def optimize_layout(head, tail, edges, weights, n_epochs, a, b, alpha0, negative_sample_rate, rng, move_other, snapshot_epochs=(), gamma=1.0):
    """Stochastischer Gradientenabstieg über die Kanten (h, t, w). Verändert `head` (und `tail`, wenn move_other) in-place.
    Je Epoche sind die Kanten aktiv, deren nächster Stichprobenzeitpunkt erreicht ist (Häufigkeit ∝ Gewicht); je aktiver Kante gibt es negative_sample_rate Abstoßungs-Stichproben.
    -> (Schnappschüsse {Epoche: Kopie von head}, Statistik je Epoche [attraktiv, abstoßend] mittlere Schrittlänge)."""
    h_idx, t_idx = edges
    n_head, n_tail = len(head), len(tail)
    eps = weights.max() / weights                                                 # Epochen je Stichprobe: starke Kanten (Gewicht = Maximum) jede Epoche
    eps_neg = eps / negative_sample_rate if negative_sample_rate > 0 else np.full_like(eps, np.inf)
    next_pos = eps.copy()
    next_neg = eps_neg.copy()
    snapshots = {0: head.copy()} if 0 in snapshot_epochs else {}
    stats = np.zeros((n_epochs, 2))
    for epoch in range(1, n_epochs + 1):
        alpha = alpha0 * (1.0 - (epoch - 1) / n_epochs)
        active = next_pos <= epoch
        if active.any():
            hi, ti = h_idx[active], t_idx[active]
            delta = head[hi] - tail[ti]
            d2 = (delta ** 2).sum(1)
            coeff = attraction_coefficient(d2, a, b)
            grad = _clip(coeff[:, None] * delta) * alpha
            head += np.stack([np.bincount(hi, weights=grad[:, c], minlength=n_head) for c in range(2)], axis=1)
            if move_other:
                tail -= np.stack([np.bincount(ti, weights=grad[:, c], minlength=n_tail) for c in range(2)], axis=1)
            stats[epoch - 1, 0] = float(np.linalg.norm(grad, axis=1).mean())
            next_pos[active] += eps[active]
            n_neg = np.zeros(int(active.sum()), dtype=int)
            if negative_sample_rate > 0:
                n_neg = np.maximum(np.floor((epoch - next_neg[active]) / eps_neg[active]).astype(int), 0)
            rep = 0.0
            count = 0
            for j in range(int(n_neg.max()) if n_neg.size else 0):
                sel = n_neg > j
                hs = hi[sel]
                ks = rng.integers(0, n_tail, size=int(sel.sum()))
                delta = head[hs] - tail[ks]
                d2 = (delta ** 2).sum(1)
                coeff = repulsion_coefficient(d2, a, b, gamma)
                grad = np.where(coeff[:, None] > 0, _clip(coeff[:, None] * delta), CLIP) * alpha
                if head is tail:
                    grad[hs == ks] = 0.0                                           # ein Punkt stößt sich nicht von sich selbst ab
                head += np.stack([np.bincount(hs, weights=grad[:, c], minlength=n_head) for c in range(2)], axis=1)
                rep += float(np.linalg.norm(grad, axis=1).sum())
                count += len(hs)
            stats[epoch - 1, 1] = rep / max(count, 1)
            if negative_sample_rate > 0:
                next_neg[active] += n_neg * eps_neg[active]
        if epoch in snapshot_epochs:
            snapshots[epoch] = head.copy()
    return snapshots, stats


def snapshot_epochs(n_epochs, count=12):
    grid = np.unique(np.round(np.geomspace(1, max(n_epochs, 2), count)).astype(int))
    return sorted(set(int(e) for e in grid) | {0, int(n_epochs)})


def cross_entropy(P, Y, a, b):
    """Fuzzy-Kreuzentropie zwischen Graph P und der Kurve im Zielraum über alle Paare (i < j): das Optimierungsziel von UMAP."""
    d2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    Q = 1.0 / (1.0 + a * d2 ** b)
    iu = np.triu_indices(len(Y), 1)
    p, q = P[iu], np.clip(Q[iu], 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0) / q), 0.0)
        term2 = np.where(p < 1, (1 - p) * np.log((1 - np.where(p < 1, p, 0.0)) / (1 - q)), 0.0)
    return float((term1 + term2).sum())


@dataclass(frozen=True)
class UMAPModel:
    embedding: np.ndarray            # [n, 2]
    snapshots: dict                  # Epoche -> Einbettung (Kopie), inkl. 0 und n_epochs
    snapshot_loss: dict              # Epoche -> Fuzzy-Kreuzentropie
    forces: np.ndarray               # [n_epochs, 2]: mittlere Schrittlänge Anziehung / Abstoßung je Epoche
    directed: np.ndarray             # w_j|i
    graph: np.ndarray                # fuzzy Vereinigung
    knn_idx: np.ndarray
    knn_dist: np.ndarray
    rho: np.ndarray
    sigma: np.ndarray
    a: float
    b: float
    n_neighbors: int
    min_dist: float
    n_epochs: int
    negative_sample_rate: int
    init: str
    init_used: str
    seed: int
    components: int                  # Zusammenhangskomponenten des Graphen
    mean: np.ndarray
    scale: np.ndarray
    Z: np.ndarray

    @property
    def n(self):
        return len(self.Z)

    @property
    def n_edges(self):
        return int((np.triu(self.graph, 1) > 0).sum())

    @property
    def connected(self):
        return self.components == 1


def fit_umap(X, n_neighbors=15, min_dist=0.1, n_epochs=500, negative_sample_rate=5, init="spectral", seed=0):
    X = np.asarray(X, dtype=float)
    if init not in ("spectral", "random", "pca"):
        raise ValueError("init in {spectral, random, pca}")
    mean = X.mean(0)
    scale = X.std(0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    n = len(Z)
    if not 2 <= n_neighbors <= n - 1:
        raise ValueError(f"n_neighbors {n_neighbors} muss zwischen 2 und n − 1 = {n - 1} liegen")
    dist = pairwise_distances(Z)
    directed, graph, idx, kd, rho, sigma = fuzzy_graph(dist, n_neighbors)
    a, b = fit_ab(min_dist)
    Y, used = initial_embedding(Z, graph, init, seed)
    rows, cols = np.nonzero(graph > 0)
    w = graph[rows, cols]
    keep = w >= w.max() / n_epochs                                                    # sehr schwache Kanten würden nie stichprobiert
    rows, cols, w = rows[keep], cols[keep], w[keep]
    wanted = snapshot_epochs(n_epochs)
    rng = np.random.default_rng(seed + 1_000_003)
    snaps, stats = optimize_layout(Y, Y, (rows, cols), w, n_epochs, a, b, 1.0, negative_sample_rate, rng, True, wanted)
    loss = {e: cross_entropy(graph, y, a, b) for e, y in snaps.items()}
    return UMAPModel(embedding=Y, snapshots=snaps, snapshot_loss=loss, forces=stats, directed=directed, graph=graph, knn_idx=idx, knn_dist=kd, rho=rho, sigma=sigma, a=a, b=b,
                     n_neighbors=int(n_neighbors), min_dist=float(min_dist), n_epochs=int(n_epochs), negative_sample_rate=int(negative_sample_rate), init=init, init_used=used,
                     seed=int(seed), components=_components(graph), mean=mean, scale=scale, Z=Z)


def transform(model, X_new, seed=0, n_epochs=None):
    """Neue Touren einbetten (wie `UMAP.transform`): Nachbarn unter den Trainings-Touren -> Fuzzy-Gewichte -> Start am gewichteten Mittel -> n_epochs/3 Epochen mit festen Trainingspunkten."""
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    d = np.sqrt(np.maximum(((Zn[:, None, :] - model.Z[None, :, :]) ** 2).sum(-1), 0.0))
    k = model.n_neighbors
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    kd = np.take_along_axis(d, idx, axis=1)
    # neue Punkte sind nicht in den Trainingsdaten: alle k Nachbarn zählen (kein Selbst-Eintrag), Ziel wie beim Training log2(k)
    padded = np.column_stack([np.zeros(len(Zn)), kd])
    rho, sigma = smooth_knn(padded, np.log2(k))
    w = membership(kd, rho, sigma)
    norm = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    head = np.einsum("ik,ikc->ic", norm, model.embedding[idx])
    epochs = int(model.n_epochs // 3) if n_epochs is None else int(n_epochs)
    keep = w >= w.max() / max(epochs, 1)
    rows = np.repeat(np.arange(len(Zn)), k).reshape(len(Zn), k)[keep]
    cols = idx[keep]
    optimize_layout(head, model.embedding.copy(), (rows, cols), w[keep], epochs, model.a, model.b, 0.25, model.negative_sample_rate, np.random.default_rng(seed + 7), False)
    return head
