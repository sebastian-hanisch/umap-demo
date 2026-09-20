import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

import umap_constants as C
from umap_evaluation import (
    Settings, analyse, convergence_rows, distance_fidelity, distance_fidelity_split, make_dataset, min_dist_sweep, n_neighbors_sweep, out_of_sample, pca_project, r2_quadratic, stability,
    timing_sweep, trustworthiness, verdict,
)


def test_trustworthiness_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for embedding in (X[:, :2], rng.standard_normal((80, 2)), pca_project(X)):
        assert abs(trustworthiness(X, embedding, 8) - sk_trustworthiness(X, embedding, n_neighbors=8)) < 1e-9


def test_r2_quadratic_recovers_monotone_reparametrisations_and_rejects_noise():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200, 2))
    coords = np.column_stack([z[:, 0] + 0.3 * z[:, 0] ** 2, z[:, 1] + 0.2 * z[:, 1] ** 2])
    assert r2_quadratic(coords, z) > 0.9
    assert r2_quadratic(rng.standard_normal((200, 2)), z) < 0.1


def test_distance_fidelity_is_one_for_a_scaled_copy_and_split_reports_near_and_far():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((100, 2))
    assert distance_fidelity(3.0 * z, z) > 0.999999
    near, far = distance_fidelity_split(3.0 * z, z)
    assert near > 0.999999 and far > 0.999999
    squashed = z * (1.0 / (1.0 + np.linalg.norm(z, axis=1, keepdims=True) ** 2))
    near2, far2 = distance_fidelity_split(squashed, z)
    assert near2 > far2


def test_analysis_on_the_default_surface_reproduces_the_measured_headline():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    a = analyse(ds, Settings())
    m = a.metrics
    assert 0.82 < m["umap"]["r2"] < 0.95 and m["tsne"]["r2"] > 0.88 and m["isomap"]["r2"] > 0.94 and m["pca"]["r2"] < 0.6
    assert m["umap"]["far"] < m["tsne"]["far"]                                  # "bessere globale Struktur als t-SNE": nicht bestätigt
    assert a.alt is None and "alt" not in m and a.umap.connected
    assert len(a.iso_indices) == 300 and sorted(a.snapshot_r2) == [e for e in sorted(a.umap.snapshots) if e > 0]
    assert convergence_rows(a)[-1][0] == 500 and abs(convergence_rows(a)[-1][1] - m["umap"]["r2"]) < 1e-12


def test_umap_has_the_same_sonderfahrten_weakness_as_tsne():
    ds = make_dataset(300, 2, 1.0, 0.25, 5, 7)
    m = analyse(ds, Settings()).metrics
    assert m["umap"]["r2"] < 0.3 and m["tsne"]["r2"] < 0.3 < 0.6 < m["pca"]["r2"] and m["isomap"]["r2"] > 0.6
    assert m["umap"]["far"] < 0.3 and m["pca"]["far"] > 0.85


def test_too_few_epochs_are_compared_against_a_converged_reference_run():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    a = analyse(ds, Settings(n_epochs=20))
    assert a.alt is not None and a.alt.n_epochs == C.CONVERGED_EPOCHS and a.metrics["alt"]["trust"] - a.metrics["umap"]["trust"] >= 0.02
    assert analyse(ds, Settings(n_epochs=200)).alt is None


@pytest.mark.parametrize("q,curv,noise,out,settings,code", [
    (2, 1.0, 0.25, 0, Settings(), "umap_wins"),
    (2, 1.0, 0.25, 5, Settings(), "global_structure"),
    (2, 1.0, 0.25, 0, Settings(n_neighbors=3), "disconnected"),
    (2, 1.0, 0.25, 0, Settings(n_epochs=20), "not_converged"),
    (2, 1.0, 0.8, 0, Settings(), "umap_wins"),
    (2, 0.0, 0.25, 0, Settings(), "no_advantage"),
])
def test_verdict_codes(q, curv, noise, out, settings, code):
    ds = make_dataset(300, q, curv, noise, out, 7)
    assert verdict(analyse(ds, settings), ds, settings)[1] == code


def test_disconnected_verdict_reports_components_and_isolated_tours():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    s = Settings(n_neighbors=3)
    data = verdict(analyse(ds, s), ds, s)[2]
    assert data["components"] == 8 and data["isolated"] == 73


def test_sweeps_are_deterministic_and_show_the_disconnected_end():
    kw = dict(n_tours=120, n_epochs=60, seeds=(100_000, 100_001))
    a = n_neighbors_sweep(2, 1.0, 0.25, 0, values=(2, 15), **kw)
    assert a == n_neighbors_sweep(2, 1.0, 0.25, 0, values=(2, 15), **kw)
    assert a[0]["components"] > 1 and a[1]["components"] == 1 and a[0]["r2"] < a[1]["r2"]
    md = min_dist_sweep(2, 1.0, 0.25, 0, values=(0.0, 1.0), **kw)
    assert [r["min_dist"] for r in md] == [0.0, 1.0] and all(np.isfinite(r["r2"]) for r in md)


def test_sweep_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED


def test_stability_covers_the_start_variants_and_measures_procrustes_to_the_spectral_run():
    ds = make_dataset(150, 2, 1.0, 0.25, 0, 7)
    s = Settings(n_neighbors=10, n_epochs=100)
    a, b = stability(ds, s), stability(ds, s)
    assert a["inits"] == ["spectral"] + list(C.STABILITY_INITS) and a["procrustes"][0] == 0.0 and a["procrustes"] == b["procrustes"]
    assert len(a["embeddings"]) == 6 and all(0.0 <= v <= 1.0 for v in a["procrustes"])


def test_stability_claim_of_the_app_holds_on_the_default_surface():
    """Belegt die Tabelle in der App: zufällige Starts weichen im Standardfall höchstens um Procrustes 0.3 vom spektralen Lauf ab (bei kleinem n / wenigen Epochen kann es mehr sein)."""
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    assert max(stability(ds, Settings())["procrustes"]) < 0.3


def test_out_of_sample_transform_beats_the_tsne_approximation_and_moves_less():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 100_000)
    oos = out_of_sample(ds, Settings())
    assert len(oos["test"]) == 60 and oos["test"][0] == 240 and oos["y_umap"].shape == (60, 2) and oos["y_tsne"].shape == (60, 2)
    assert oos["r2_umap"] > oos["r2_tsne"] and oos["r2_umap"] > 0.8 and oos["shift_umap"] < 0.3


def test_timing_sweep_has_the_expected_shape_and_the_crossover_holds():
    rows = timing_sweep(ns=(100, 600), n_epochs=200)
    assert [r["n"] for r in rows] == [100, 600] and all(r[key] > 0 for r in rows for key in ("umap", "tsne", "isomap", "lle", "pca"))
    assert rows[0]["tsne"] < rows[0]["umap"] and rows[1]["umap"] < rows[1]["tsne"]              # bei kleinem n gewinnt t-SNE, bei großem UMAP
