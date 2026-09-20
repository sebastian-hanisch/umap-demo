"""t-SNE (t-distributed Stochastic Neighbor Embedding, van der Maaten & Hinton 2008) von Grund auf (numpy, exakte O(n²)-Variante).

1. **Affinitäten im Hochdimensionalen**: je Tour i eine Gauß-Verteilung über alle anderen Touren, p_j|i ∝ exp(−‖x_i − x_j‖² · β_i). Die Breite β_i = 1/(2σ_i²) wird per Binärsuche so
   gewählt, dass die **Perplexität** 2^H(P_i) einen Zielwert trifft - die "effektive Zahl der Nachbarn". Symmetrisiert: p_ij = (p_j|i + p_i|j) / (2n).
2. **Zielraum**: q_ij ∝ (1 + ‖y_i − y_j‖²)⁻¹ (Student-t mit einem Freiheitsgrad, schwere Ränder). Optional `kernel="gauss"`: q_ij ∝ exp(−‖y_i − y_j‖²) - das ursprüngliche symmetrische SNE,
   das das **Crowding-Problem** zeigt (in wenigen Dimensionen ist zu wenig Platz für die vielen mittleren Abstände, alles wird zusammengedrückt).
3. **Optimierung**: Gradientenabstieg auf KL(P ‖ Q) mit Momentum, adaptiven Gains (delta-bar-delta) und **Early Exaggeration** (P × 12 in der ersten Phase). Wie in scikit-learn:
   Momentum 0.5 -> 0.8, Lernrate `auto` = max(n / Exaggeration / 4, 50), PCA-Initialisierung auf Std 1e-4.

t-SNE hat **keine Abbildung für neue Punkte** - die einzige billige Näherung (`embed_new_naive`) ist hier ein Zusatz, kein Teil des Verfahrens."""

from dataclasses import dataclass

import numpy as np

EXAGGERATION_ITERS = 250
MOMENTUM_EARLY, MOMENTUM_LATE = 0.5, 0.8
MIN_GAIN = 0.01
INIT_STD = 1e-4
DIVERGENCE_LIMIT = 1e8


def squared_distances(Z):
    sq = (Z ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    np.fill_diagonal(d2, 0.0)
    return np.maximum((d2 + d2.T) / 2, 0.0)


def conditional_affinities(d2, perplexity, n_steps=100):
    """Zeilenweise Gauß-Affinitäten p_j|i mit Perplexität `perplexity` (Binärsuche auf log β_i, alle Zeilen gleichzeitig). -> (P_cond [n,n], β [n])."""
    n = len(d2)
    if not 1.0 <= perplexity < n - 1:
        raise ValueError(f"Perplexität {perplexity} muss zwischen 1 und n − 1 = {n - 1} liegen")
    target = np.log(perplexity)
    off = ~np.eye(n, dtype=bool)
    shifted = np.where(off, d2, np.inf)
    shifted = shifted - shifted.min(1, keepdims=True)                     # numerisch stabil: nächster Nachbar bekommt exp(0)
    shifted = np.where(off, shifted, 0.0)                                 # Diagonale endlich halten (0 · inf = nan); sie wird über `off` ausgeblendet
    lo, hi = np.full(n, -30.0), np.full(n, 30.0)                          # Suche über log β
    for _ in range(n_steps):
        log_beta = (lo + hi) / 2
        beta = np.exp(log_beta)[:, None]
        w = np.where(off, np.exp(-shifted * beta), 0.0)
        total = w.sum(1, keepdims=True)
        entropy = np.log(total[:, 0]) + (beta[:, 0] * (shifted * w).sum(1) / total[:, 0])
        too_wide = entropy > target                                        # zu hohe Entropie -> Verteilung zu breit -> β erhöhen
        lo = np.where(too_wide, log_beta, lo)
        hi = np.where(too_wide, hi, log_beta)
    beta = np.exp((lo + hi) / 2)[:, None]
    w = np.where(off, np.exp(-shifted * beta), 0.0)
    return w / w.sum(1, keepdims=True), beta[:, 0]


def joint_affinities(d2, perplexity):
    """Symmetrisierte gemeinsame Affinitäten P (Summe 1, Diagonale 0) und die Gauß-Breiten σ_i."""
    cond, beta = conditional_affinities(d2, perplexity)
    n = len(d2)
    return (cond + cond.T) / (2.0 * n), cond, np.sqrt(1.0 / (2.0 * beta))


def row_perplexity(cond):
    """Perplexität 2^H je Zeile der bedingten Affinitäten (zur Kontrolle der Binärsuche; entspricht exp(H) mit natürlichem Logarithmus)."""
    p = np.where(cond > 0, cond, 1.0)
    return np.exp(-(cond * np.log(p)).sum(1))


def low_dim_affinities(Y, kernel="student"):
    """Q (Summe 1, Diagonale 0) und die Kernel-Matrix W, aus der es entsteht."""
    d2 = squared_distances(Y)
    if kernel == "student":
        W = 1.0 / (1.0 + d2)
    else:                                                                # exp(−d²), um den kleinsten Abstand verschoben: kürzt sich in Q heraus, verhindert aber Unterlauf bei weit gestreuten Punkten
        off = ~np.eye(len(Y), dtype=bool)
        W = np.exp(-(np.where(off, d2, d2[off].min()) - d2[off].min()))
    np.fill_diagonal(W, 0.0)
    return np.maximum(W / W.sum(), 1e-12), W


def gradient(P, Y, kernel="student"):
    """∂KL(P‖Q)/∂y_i = 4 Σ_j (p_ij − q_ij) · (y_i − y_j) · [(1 + d_ij²)⁻¹ beim Student-t-Kern, 1 beim Gauß-Kern]."""
    Q, W = low_dim_affinities(Y, kernel)
    A = (P - Q) * (W if kernel == "student" else 1.0)
    np.fill_diagonal(A, 0.0)
    return 4.0 * (A.sum(1)[:, None] * Y - A @ Y), Q


def kl_divergence(P, Q):
    mask = P > 0
    return float((P[mask] * np.log(P[mask] / Q[mask])).sum())


@dataclass(frozen=True)
class TSNEModel:
    embedding: np.ndarray            # [n, 2]
    kl_history: np.ndarray           # KL(P‖Q) mit dem echten (nicht übertriebenen) P je Iteration
    snapshots: dict                  # Iteration -> Einbettung (Kopie), inkl. 0, Ende der Early Exaggeration und n_iter
    P: np.ndarray                    # gemeinsame Affinitäten
    cond: np.ndarray                 # bedingte Affinitäten p_j|i (Zeilen)
    sigma: np.ndarray                # Gauß-Breite je Tour (in z-Einheiten)
    mean: np.ndarray
    scale: np.ndarray
    Z: np.ndarray
    perplexity: float
    n_iter: int
    learning_rate: float
    exaggeration: float
    exaggeration_iters: int
    kernel: str
    init: str
    seed: int
    diverged_at: int = 0             # 0 = nicht divergiert, sonst die Iteration, in der die Punkte ins Unendliche flogen (danach eingefroren)

    @property
    def n(self):
        return len(self.Z)

    @property
    def kl(self):
        return float(self.kl_history[-1])


def snapshot_iterations(n_iter, exaggeration_iters, count=12):
    grid = np.unique(np.round(np.geomspace(1, max(n_iter, 2), count)).astype(int))
    return sorted(set(int(i) for i in grid) | {0, int(exaggeration_iters), int(n_iter)})


def pca_init(Z, n_components=2):
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    Y = Zc @ vt[:n_components].T
    return Y / np.std(Y[:, 0]) * INIT_STD


def fit_tsne(X, perplexity=30.0, n_iter=750, learning_rate=None, exaggeration=12.0, kernel="student", init="pca", seed=0):
    """`learning_rate=None` = auto (max(n / exaggeration / 4, 50)). Die Early-Exaggeration-Phase dauert min(250, n_iter // 4) Iterationen."""
    if kernel not in ("student", "gauss") or init not in ("pca", "random"):
        raise ValueError("kernel in {student, gauss}, init in {pca, random}")
    X = np.asarray(X, dtype=float)
    mean = X.mean(0)
    scale = X.std(0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    n = len(Z)
    P, cond, sigma = joint_affinities(squared_distances(Z), perplexity)
    lr = float(learning_rate) if learning_rate else max(n / exaggeration / 4.0, 50.0)
    exag_iters = int(min(EXAGGERATION_ITERS, n_iter // 4))
    Y = pca_init(Z) if init == "pca" else np.random.default_rng(seed).standard_normal((n, 2)) * INIT_STD
    update, gains = np.zeros_like(Y), np.ones_like(Y)
    kl_history = np.zeros(n_iter)
    diverged_at = 0
    wanted = set(snapshot_iterations(n_iter, exag_iters))
    snapshots = {0: Y.copy()} if 0 in wanted else {}
    for it in range(1, n_iter + 1):
        exaggerating = it <= exag_iters
        grad, Q = gradient(P * exaggeration if exaggerating else P, Y, kernel)
        kl_history[it - 1] = kl_divergence(P, Q)
        inc = update * grad < 0.0
        gains = np.where(inc, gains + 0.2, gains * 0.8)
        np.maximum(gains, MIN_GAIN, out=gains)
        update = (MOMENTUM_EARLY if exaggerating else MOMENTUM_LATE) * update - lr * gains * grad
        candidate = Y + update
        if not np.isfinite(candidate).all() or np.abs(candidate).max() > DIVERGENCE_LIMIT:      # z. B. Gauß-Kern mit riesiger Lernrate: die Punkte fliegen auseinander
            diverged_at = it
            kl_history[it - 1:] = kl_history[it - 2] if it > 1 else kl_history[it - 1]
            for frozen in sorted(w for w in wanted if w >= it):
                snapshots[frozen] = Y.copy()
            break
        Y = candidate - candidate.mean(0)
        if it in wanted:
            snapshots[it] = Y.copy()
    return TSNEModel(embedding=Y, kl_history=kl_history, snapshots=snapshots, P=P, cond=cond, sigma=sigma, mean=mean, scale=scale, Z=Z, perplexity=float(perplexity),
                     n_iter=int(n_iter), learning_rate=lr, exaggeration=float(exaggeration), exaggeration_iters=exag_iters, kernel=kernel, init=init, seed=int(seed), diverged_at=diverged_at)


def embed_new_naive(model, X_new, k=10):
    """NÄHERUNG, kein Teil von t-SNE: neue Tour = mit 1/Abstand gewichteter Mittelwert der Koordinaten ihrer k nächsten Trainings-Touren."""
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    d = np.sqrt(np.maximum(((Zn[:, None, :] - model.Z[None, :, :]) ** 2).sum(-1), 0.0))
    nbrs = np.argsort(d, axis=1)[:, :k]
    out = np.zeros((len(Zn), 2))
    for i in range(len(Zn)):
        w = 1.0 / np.maximum(d[i, nbrs[i]], 1e-9)
        out[i] = (w / w.sum()) @ model.embedding[nbrs[i]]
    return out


def procrustes_disparity(A, B):
    """Wie weit unterscheiden sich zwei Einbettungen derselben Touren nach bester Drehung/Spiegelung/Skalierung? 0 = gleich, 1 = unabhängig (wie scipy.spatial.procrustes)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(max(0.0, 1.0 - s.sum() ** 2))
