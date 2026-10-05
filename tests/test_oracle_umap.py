"""Orakel-Tests (unabhängiger Rechenweg, klein und schnell, ohne umap-learn): σ per scipy-Nullstellensuche, Fuzzy-Vereinigung und Kreuzentropie als
Paarschleifen, die vektorisierte Optimierung gegen eine Schleifen-Neuimplementierung (gleicher Zufallsstrom, wenige Epochen - danach verstärkt die
Singularität d^(2(b-1)) bei b < 1 Rundungsunterschiede chaotisch), spektraler Start gegen die Laplace-Matrix aus scipy, R² und Abstandstreue gegen sklearn/scipy."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from umap_algorithm import cross_entropy, fit_ab, fuzzy_graph, graph_components, optimize_layout, spectral_init
from umap_evaluation import distance_fidelity, distance_fidelity_split, r2_quadratic
from umap_isomap import pairwise_distances


def test_sigma_rho_weights_and_fuzzy_union_match_loops_and_root_finding():
    rng = np.random.default_rng(1)
    for _ in range(12):
        n, k = int(rng.integers(10, 35)), int(rng.integers(3, 10))
        dist = pairwise_distances(rng.standard_normal((n, 3)))
        directed, sym, idx, kd, rho, sigma = fuzzy_graph(dist, k)
        for i in range(n):
            rest = kd[i, 1:]
            assert abs(rho[i] - rest.min()) < 1e-12
            root = brentq(lambda s: np.exp(-np.maximum(rest - rho[i], 0) / s).sum() - np.log2(k), 1e-9, 1e6, xtol=1e-13)
            assert abs(root - sigma[i]) < 1e-3 * root
            for jj in range(1, k):
                assert abs(directed[i, idx[i, jj]] - np.exp(-max(0.0, dist[i, idx[i, jj]] - rho[i]) / sigma[i])) < 1e-12
        union = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    union[i, j] = directed[i, j] + directed[j, i] - directed[i, j] * directed[j, i]
        assert np.allclose(union, sym, atol=1e-12)
        assert graph_components(sym) == connected_components(sym > 0, directed=False)[0]


def test_cross_entropy_matches_a_pair_loop():
    rng = np.random.default_rng(2)
    for scale in (0.1, 1.0, 5.0):
        n = 14
        P = np.triu(rng.random((n, n)) * (rng.random((n, n)) < 0.4), 1)
        P = P + P.T
        P[0, 1] = P[1, 0] = 1.0
        Y = rng.standard_normal((n, 2)) * scale
        a, b = fit_ab(0.3)
        ref = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(((Y[i] - Y[j]) ** 2).sum())
                q = min(max(1 / (1 + a * d ** (2 * b)), 1e-12), 1 - 1e-12)
                p = P[i, j]
                ref += p * np.log(p / q) if p > 0 else 0.0
                ref += (1 - p) * np.log((1 - p) / (1 - q)) if p < 1 else 0.0
        assert abs(cross_entropy(P, Y, a, b) - ref) < 1e-8 * max(1.0, abs(ref))


def _reference_optimize(Y0, rows, cols, w, n_epochs, a, b, alpha0, rate, rng):
    head = Y0.copy()
    n = len(head)
    eps = w.max() / w
    eps_neg = eps / rate
    next_pos, next_neg = eps.copy(), eps_neg.copy()
    for epoch in range(1, n_epochs + 1):
        alpha = alpha0 * (1 - (epoch - 1) / n_epochs)
        active = [e for e in range(len(w)) if next_pos[e] <= epoch]
        if not active:
            continue
        update = np.zeros_like(head)
        for e in active:
            delta = head[rows[e]] - head[cols[e]]
            d2 = (delta ** 2).sum()
            coeff = 0.0 if d2 <= 0 else -2 * a * b * d2 ** (b - 1) / (a * d2 ** b + 1)
            g = np.clip(coeff * delta, -4, 4) * alpha
            update[rows[e]] += g
            update[cols[e]] -= g
        head = head + update
        counts = [max(int(np.floor((epoch - next_neg[e]) / eps_neg[e])), 0) for e in active]
        for e in active:
            next_pos[e] += eps[e]
        for j in range(max(counts)):
            chosen = [active[m] for m in range(len(active)) if counts[m] > j]
            targets = rng.integers(0, n, size=len(chosen))
            update = np.zeros_like(head)
            for e, t in zip(chosen, targets):
                if rows[e] == t:
                    continue
                delta = head[rows[e]] - head[t]
                d2 = (delta ** 2).sum()
                coeff = 0.0 if d2 <= 0 else 2 * b / ((0.001 + d2) * (a * d2 ** b + 1))
                update[rows[e]] += (np.clip(coeff * delta, -4, 4) if coeff > 0 else np.full(2, 4.0)) * alpha
            head = head + update
        for m, e in enumerate(active):
            next_neg[e] += counts[m] * eps_neg[e]
    return head


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_optimisation_matches_a_loop_reimplementation(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(8, 20)), int(rng.integers(3, 8))
    _, sym, *_ = fuzzy_graph(pairwise_distances(rng.standard_normal((n, 3))), k)
    n_epochs = int(rng.integers(1, 5))
    rows, cols = np.nonzero(sym > 0)
    w = sym[rows, cols]
    keep = w >= w.max() / n_epochs
    rows, cols, w = rows[keep], cols[keep], w[keep]
    a, b = fit_ab(float(rng.uniform(0, 1)))
    Y0 = rng.uniform(-10, 10, (n, 2))
    rate = int(rng.integers(1, 6))
    Y = Y0.copy()
    optimize_layout(Y, Y, (rows, cols), w, n_epochs, a, b, 1.0, rate, np.random.default_rng(seed + 100), True)
    ref = _reference_optimize(Y0, rows, cols, w, n_epochs, a, b, 1.0, rate, np.random.default_rng(seed + 100))
    assert np.abs(Y - ref).max() < 1e-8


def test_spectral_start_spans_the_laplacian_eigenvectors():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(30):
        n, k = int(rng.integers(15, 40)), int(rng.integers(5, 12))
        _, sym, *_ = fuzzy_graph(pairwise_distances(rng.standard_normal((n, 3))), k)
        Y = spectral_init(sym, 0)
        if connected_components(sym > 0, directed=False)[0] > 1:
            assert Y is None
            continue
        values, vectors = np.linalg.eigh(laplacian(sym, normed=True))
        if min(values[1] - values[0], values[2] - values[1], values[3] - values[2]) < 1e-6:
            continue
        for c in range(2):
            assert abs(np.corrcoef(vectors[:, c + 1], Y[:, c])[0, 1]) > 1 - 1e-6
        assert abs(np.abs(Y).max() - 10.0) < 1e-2
        checked += 1
    assert checked >= 10


def test_r2_and_distance_fidelity_match_sklearn_and_scipy():
    rng = np.random.default_rng(5)
    for it in range(20):
        n = int(rng.integers(15, 50))
        z = rng.standard_normal((n, int(rng.integers(1, 4))))
        c = np.column_stack([z[:, 0], z[:, 0] ** 2 + 0.1 * rng.standard_normal(n)]) if it % 2 else rng.standard_normal((n, 2))
        A = PolynomialFeatures(2).fit_transform(c)
        residual = z - LinearRegression(fit_intercept=False).fit(A, z).predict(A)
        assert abs(r2_quadratic(c, z) - (1 - residual.var(0).sum() / z.var(0).sum())) < 1e-8
        dz, dc = pdist(z), pdist(c)
        assert abs(distance_fidelity(c, z) - pearsonr(dc, dz)[0]) < 1e-10
        near = dz <= np.median(dz)
        got_near, got_far = distance_fidelity_split(c, z)
        assert abs(got_near - pearsonr(dc[near], dz[near])[0]) < 1e-10 and abs(got_far - pearsonr(dc[~near], dz[~near])[0]) < 1e-10
