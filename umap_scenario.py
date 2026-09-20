"""Erzeugt Lieferrouten-Kennzahlen aus wenigen versteckten Faktoren (die wahre Dimension q ist bekannt und damit prüfbar).

Modell: latente Faktoren z ~ N(0, I_q); jedes der 12 Merkmale lädt vor allem auf den Faktor seiner Gruppe (bei q < 4 teilen sich mehrere
Gruppen einen Faktor: Gruppe g lädt auf Faktor g mod q), dazu kleine Querladungen. Signal in z-Einheiten:

    s = z @ L.T  +  kappa * h(z) @ B.T          (L: 12 x q, B: 12 x m, h: nichtlineare Terme)

Die Krümmung kappa biegt die q-dimensionale Fläche in höhere Dimensionen: h(z) enthält sin/cos, Quadrate und Produkte der latenten
Koordinaten. kappa = 0 ist exakt linear (PCA ist dann perfekt), kappa > 0 ist die Linearitätsannahme, an der PCA scheitert.
Rauschen: s + noise * eps. Rohdaten = Mittelwert + typische Streuung * s (Einheiten wie gemessen). Sonderfahrten (Ausreißer): wenige
Touren mit stark vergrößertem Faktor. Die Matrizen L und B sind fest (LAYOUT_SEED), nur Touren, Rauschen und Ausreißer hängen vom Seed ab."""

from dataclasses import dataclass

import numpy as np

import umap_constants as C


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray            # [n, 12] Rohdaten in Einheiten
    z: np.ndarray            # [n, q] latente Faktoren (Wahrheit)
    outlier: np.ndarray      # [n] bool: Sonderfahrt
    q: int
    curvature: float
    noise: float
    outlier_pct: int

    @property
    def n(self):
        return len(self.X)


def _nonlinear_terms(z):
    """h(z): sin/cos jeder latenten Koordinate, Quadrate (zentriert) und Produkte - [n, m]."""
    n, q = z.shape
    terms = []
    for f in range(q):
        terms.append(np.sin(C.CURVATURE_FREQUENCY * z[:, f]))
        terms.append(np.cos(C.CURVATURE_FREQUENCY * z[:, f]) - np.exp(-0.5 * C.CURVATURE_FREQUENCY ** 2))   # E[cos(w z)] abziehen
        terms.append(z[:, f] ** 2 - 1.0)
    for a in range(q):
        for b in range(a + 1, q):
            terms.append(z[:, a] * z[:, b])
    return np.stack(terms, axis=1)


def n_nonlinear_terms(q):
    return 3 * q + q * (q - 1) // 2


def loading_matrix(q):
    """Feste 12 x q-Ladungsmatrix: Merkmal j lädt auf den Faktor seiner Gruppe (Gruppe g -> Faktor g mod q), dazu kleine Querladungen."""
    rng = np.random.default_rng(C.LAYOUT_SEED)
    cross = rng.uniform(-1.0, 1.0, size=(C.N_FEATURES, 4))
    L = np.zeros((C.N_FEATURES, q))
    for j in range(C.N_FEATURES):
        group = C.GROUP_OF_FEATURE[j]
        L[j, group % q] += C.WITHIN_LOADINGS[j % 3]
        for g in range(4):
            if g != group:
                L[j, g % q] += C.CROSS_LOADING * cross[j, g]
    return L


def curvature_matrix(q):
    """Feste 12 x m-Matrix, die die nichtlinearen Terme auf die Merkmale verteilt (jeder Term wirkt auf alle Merkmale)."""
    rng = np.random.default_rng(C.LAYOUT_SEED + 1)
    B = rng.standard_normal((C.N_FEATURES, n_nonlinear_terms(q)))
    return B / np.linalg.norm(B, axis=0, keepdims=True) * C.CURVATURE_AMPLITUDE


def generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_tours, q))
    eps = rng.standard_normal((n_tours, C.N_FEATURES))
    outlier_draw = rng.random(n_tours)                                 # zuletzt gezogen: gleiche Touren mit/ohne Ausreißer
    outlier_sign = rng.choice([-1.0, 1.0], size=n_tours)
    outlier = outlier_draw < outlier_pct / 100.0
    z_used = z.copy()
    if outlier.any():
        factor = 1 if q >= 2 else 0                                    # Sonderfahrten strecken den Faktor "Zeitdruck" (bei q = 1 den einzigen)
        z_used[outlier, factor] = z_used[outlier, factor] * C.OUTLIER_SCALE * outlier_sign[outlier]
    signal = z_used @ loading_matrix(q).T
    if curvature > 0:
        signal = signal + curvature * _nonlinear_terms(z_used) @ curvature_matrix(q).T
    signal = signal + noise * eps
    means = np.array([f[2] for f in C.FEATURES])
    scales = np.array([f[3] for f in C.FEATURES])
    X = means + signal * scales
    return Dataset(X=X, z=z_used, outlier=outlier, q=q, curvature=float(curvature), noise=float(noise), outlier_pct=int(outlier_pct))
