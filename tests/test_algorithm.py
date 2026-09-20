import numpy as np
import pytest
import umap as umap_learn
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

from umap_algorithm import (
    attraction_coefficient, cross_entropy, fit_ab, fit_umap, fuzzy_graph, graph_components, knn, membership, optimize_layout, repulsion_coefficient, smooth_knn, snapshot_epochs,
    spectral_init, transform,
)
from umap_isomap import fit_isomap, pairwise_distances, standardize
from umap_lle import SingularNeighbourhood, fit_lle
from umap_scenario import generate_dataset
from umap_tsne import fit_tsne, procrustes_disparity


def _data(n=300, q=2, curvature=1.0, noise=0.25, seed=7):
    return generate_dataset(n, q, curvature, noise, 0, seed)


@pytest.mark.parametrize("min_dist", [0.0, 0.1, 0.5, 1.0])
def test_curve_parameters_match_umap_learn_and_scipy(min_dist):
    a, b = fit_ab(min_dist)
    ref_a, ref_b = find_ab_params(1.0, min_dist)
    assert abs(a - ref_a) < 1e-3 and abs(b - ref_b) < 1e-3
    x = np.linspace(0, 3, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist)))
    (sa, sb), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), x, y)
    assert abs(a - sa) < 1e-3 and abs(b - sb) < 1e-3


def test_knn_puts_the_point_itself_first_and_matches_sklearn():
    Z = standardize(_data(120).X)
    dist = pairwise_distances(Z)
    idx, kd = knn(dist, 8)
    assert (idx[:, 0] == np.arange(120)).all() and (kd[:, 0] == 0).all() and (np.diff(kd[:, 1:], axis=1) >= -1e-12).all()
    sk = NearestNeighbors(n_neighbors=8).fit(Z).kneighbors(Z, return_distance=False)
    assert np.mean([set(a[1:]) == set(b[1:]) for a, b in zip(idx, sk)]) > 0.99


def test_smooth_knn_hits_the_target_sum_and_rho_is_the_nearest_other_distance():
    Z = standardize(_data(150).X)
    idx, kd = knn(pairwise_distances(Z), 12)
    rho, sigma = smooth_knn(kd, np.log2(12))
    w = membership(kd, rho, sigma)[:, 1:]
    assert np.allclose(rho, kd[:, 1]) and np.allclose(w.sum(1), np.log2(12), atol=1e-4)
    assert np.allclose(w[:, 0], 1.0) and (w >= 0).all() and (w <= 1).all() and (sigma > 0).all()


def test_fuzzy_graph_is_symmetric_bounded_and_matches_umap_learn():
    Z = standardize(_data(200).X)
    directed, graph, idx, kd, rho, sigma = fuzzy_graph(pairwise_distances(Z), 15)
    assert np.allclose(graph, graph.T) and graph.min() >= 0 and graph.max() <= 1 and np.allclose(np.diag(graph), 0)
    assert np.allclose(graph, directed + directed.T - directed * directed.T)
    nn = NearestNeighbors(n_neighbors=15).fit(Z)
    kdist, kidx = nn.kneighbors(Z)
    ref, ref_sigma, ref_rho = fuzzy_simplicial_set(Z, 15, np.random.RandomState(0), "euclidean", knn_indices=kidx, knn_dists=kdist)
    assert np.abs(ref.toarray() - graph).max() < 1e-4 and np.abs(ref_rho - rho).max() < 1e-5 and np.abs(ref_sigma - sigma).max() < 1e-4


def test_a_small_neighbourhood_disconnects_the_graph_and_a_large_one_connects_it():
    Z = standardize(_data().X)
    dist = pairwise_distances(Z)
    assert graph_components(fuzzy_graph(dist, 3)[1]) == 8
    assert graph_components(fuzzy_graph(dist, 15)[1]) == 1


def test_spectral_init_is_scaled_deterministic_and_none_for_a_disconnected_graph():
    Z = standardize(_data().X)
    dist = pairwise_distances(Z)
    graph = fuzzy_graph(dist, 15)[1]
    a, b = spectral_init(graph, 0), spectral_init(graph, 0)
    assert np.array_equal(a, b) and abs(np.abs(a).max() - 10.0) < 0.01 and a.shape == (300, 2)
    assert spectral_init(fuzzy_graph(dist, 3)[1], 0) is None


def test_force_coefficients_are_the_gradients_of_the_cross_entropy_terms():
    """Anziehung = Abstieg auf −log v(d) = log(1 + a·d^{2b}); Abstoßung = Abstieg auf −log(1 − v(d)) (ohne die 0.001-Stabilisierung, deshalb bei großem d verglichen)."""
    a, b = 1.58, 0.9
    y_i, y_j = np.array([0.4, -0.3]), np.array([1.5, 0.9])

    def attract_loss(y):
        d2 = float(((y - y_j) ** 2).sum())
        return np.log(1.0 + a * d2 ** b)

    def repel_loss(y):
        d2 = float(((y - y_j) ** 2).sum())
        return -np.log(a * d2 ** b / (1.0 + a * d2 ** b))

    d2 = np.array([float(((y_i - y_j) ** 2).sum())])
    step_a = attraction_coefficient(d2, a, b)[0] * (y_i - y_j)
    step_r = repulsion_coefficient(d2, a, b)[0] * (y_i - y_j)
    for c in range(2):
        e = np.zeros(2)
        e[c] = 1e-6
        assert abs(-(attract_loss(y_i + e) - attract_loss(y_i - e)) / 2e-6 - step_a[c]) < 1e-6
        assert abs(-(repel_loss(y_i + e) - repel_loss(y_i - e)) / 2e-6 - step_r[c]) < 2e-3      # 0.001-Stabilisierung im Nenner
    assert attraction_coefficient(np.array([0.0]), a, b)[0] == 0 and repulsion_coefficient(np.array([0.0]), a, b)[0] == 0


def test_optimisation_pulls_neighbours_together_and_pushes_random_points_apart():
    rng = np.random.default_rng(0)
    head = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 10.0]])
    edges = (np.array([0, 1]), np.array([1, 0]))
    before = np.linalg.norm(head[0] - head[1])
    optimize_layout(head, head, edges, np.array([1.0, 1.0]), 30, 1.58, 0.9, 1.0, 0, rng, True)          # ohne negative Stichproben: nur Anziehung
    assert np.linalg.norm(head[0] - head[1]) < before


def test_fit_is_deterministic_records_snapshots_and_forces():
    ds = _data(150)
    m1, m2 = fit_umap(ds.X, 10, 0.1, 100), fit_umap(ds.X, 10, 0.1, 100)
    assert np.array_equal(m1.embedding, m2.embedding) and not np.array_equal(m1.embedding, fit_umap(ds.X, 10, 0.1, 100, seed=3).embedding)
    assert set(m1.snapshots) == set(snapshot_epochs(100)) and np.array_equal(m1.snapshots[100], m1.embedding) and m1.forces.shape == (100, 2)
    assert m1.snapshot_loss.keys() == m1.snapshots.keys() and m1.init_used == "spectral" and m1.connected


def test_disconnected_graph_falls_back_to_a_random_start_and_says_so():
    m = fit_umap(_data().X, 3, 0.1, 50)
    assert not m.connected and m.components == 8 and m.init_used.startswith("random")


def test_invalid_options_are_rejected():
    X = _data(60).X
    with pytest.raises(ValueError):
        fit_umap(X, 10, 0.1, 20, init="tsne")
    with pytest.raises(ValueError):
        fit_umap(X, 60, 0.1, 20)
    with pytest.raises(ValueError):
        fit_umap(X, 1, 0.1, 20)


def test_quality_is_comparable_to_umap_learn_on_the_default_surface():
    from umap_evaluation import r2_quadratic, trustworthiness
    ds = _data()
    Z = standardize(ds.X)
    ours = fit_umap(ds.X, 15, 0.1, 500)
    ref = umap_learn.UMAP(n_neighbors=15, min_dist=0.1, n_epochs=500, random_state=0, init="spectral").fit(Z)
    assert r2_quadratic(ours.embedding, ds.z) > r2_quadratic(ref.embedding_, ds.z) - 0.1
    assert trustworthiness(Z, ours.embedding) > trustworthiness(Z, ref.embedding_) - 0.03


def test_exact_cross_entropy_does_not_fall_during_optimisation_also_for_umap_learn():
    """Belegt die Aussage in der App: die exakte Fuzzy-Kreuzentropie liegt am Ende über dem spektralen Start - auch bei der Referenzimplementierung."""
    ds = _data()
    ours = fit_umap(ds.X, 15, 0.1, 500)
    assert ours.snapshot_loss[500] > 1.5 * ours.snapshot_loss[0]
    ref = umap_learn.UMAP(n_neighbors=15, min_dist=0.1, n_epochs=500, random_state=0, init="spectral").fit(standardize(ds.X))
    assert cross_entropy(ours.graph, ref.embedding_, ours.a, ours.b) > ours.snapshot_loss[0]
    assert abs(ours.snapshot_loss[500] - cross_entropy(ours.graph, ours.embedding, ours.a, ours.b)) < 1e-6


def test_transform_of_training_points_lands_near_their_own_coordinates():
    ds = _data()
    m = fit_umap(ds.X, 15, 0.1, 500)
    y = transform(m, ds.X[:30])
    assert np.abs(y - m.embedding[:30]).max() < 0.15 * np.abs(m.embedding).max()
    assert y.shape == (30, 2) and (np.abs(y) <= np.abs(m.embedding).max() * 1.2).all()


def test_transform_is_deterministic():
    ds = _data(200)
    m = fit_umap(ds.X[:160], 12, 0.1, 200)
    assert np.array_equal(transform(m, ds.X[160:]), transform(m, ds.X[160:]))


def test_copied_isomap_lle_tsne_and_scenario_match_their_reference_values():
    ds = _data()
    r = fit_isomap(ds.X, 8, 2)
    assert abs(float(np.abs(r.embedding).sum()) - 1786.6726227517283) < 1e-6 and abs(float(r.geodesic.sum()) - 621595.593373937) < 1e-3
    lle = fit_lle(ds.X, 14, 2, 1e-2)
    assert np.allclose(lle.embedding.mean(0), 0, atol=1e-9) and abs(lle.eigenvalues[0]) < 1e-10
    with pytest.raises(SingularNeighbourhood):
        fit_lle(ds.X, 20, 2, 0.0)
    assert abs(fit_tsne(ds.X, 30, 750).kl - 0.3098433168465976) < 2e-3            # t-SNE ist chaotisch: auf CI (andere BLAS) 0.30990 statt 0.30984
    assert procrustes_disparity(np.eye(2), np.eye(2) * 3) < 1e-12
    assert abs(float(ds.X.sum()) - 14553337.310875032) < 1e-6 and abs(float(generate_dataset(200, 3, 0.4, 0.3, 5, 42).X.sum()) - 10763498.969287368) < 1e-6
