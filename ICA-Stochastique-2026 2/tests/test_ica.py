import numpy as np
import pytest

from ica_stoch import SGDICA, amari_index, make_mixing, make_sources, run_benchmark, whiten
from ica_stoch.data import excess_kurtosis
from ica_stoch.fmri import pica, select_q_bic


# ---------------------------------------------------------------- données
@pytest.mark.parametrize("scenario", [1, 2, 3])
def test_sources_reproductibles_et_normalisees(scenario):
    S1, t = make_sources(scenario)
    S2, _ = make_sources(scenario)
    np.testing.assert_array_equal(S1, S2)
    assert S1.shape == (3, t.size)
    np.testing.assert_allclose(S1.std(axis=1), 1.0)


def test_scenario_3_entierement_sub_gaussien():
    S, _ = make_sources(3)
    assert np.all(excess_kurtosis(S) < 0)


def test_scenario_inconnu():
    with pytest.raises(ValueError):
        make_sources(4)


# ---------------------------------------------------------------- indice d'Amari
def test_amari_nul_pour_une_permutation_dilatation():
    A = make_mixing(3, seed=0)
    P = np.array([[0, 2.0, 0], [0, 0, -0.5], [3.0, 0, 0]])
    W = P @ np.linalg.inv(A)
    assert amari_index(A, W) == pytest.approx(0.0, abs=1e-12)


def test_amari_positif_et_borne():
    rng = np.random.default_rng(1)
    value = amari_index(rng.standard_normal((4, 4)), rng.standard_normal((4, 4)))
    assert 0 < value <= 1


# ---------------------------------------------------------------- SGD-ICA
def test_blanchiment():
    S, _ = make_sources(1)
    Z, V, _ = whiten(make_mixing(3, 0) @ S)
    np.testing.assert_allclose(np.cov(Z), np.eye(3), atol=1e-8)


@pytest.mark.parametrize("natural", [True, False])
@pytest.mark.parametrize("scenario", [1, 2, 3])
def test_sgd_extended_separe_tous_les_scenarios(scenario, natural):
    S, _ = make_sources(scenario)
    A = make_mixing(3, seed=0)
    ica = SGDICA(nonlinearity="extended", natural_gradient=natural, n_epochs=30, random_state=0).fit(A @ S)
    assert amari_index(A, ica.components_) < 0.06


def test_sgd_signes_estimes():
    S, _ = make_sources(1)  # sinus et carré sub-gaussiens, Laplace super-gaussienne
    ica = SGDICA(n_epochs=30, random_state=0).fit(make_mixing(3, 0) @ S)
    assert sorted(ica.signs_) == [-1, -1, 1]


def test_sgd_hypothese_fausse_echoue():
    """Avec une hypothèse super-gaussienne, des sources sub-gaussiennes ne sont pas séparées."""
    S, _ = make_sources(3)
    A = make_mixing(3, seed=0)
    ica = SGDICA(nonlinearity="super", n_epochs=30, random_state=0).fit(A @ S)
    assert amari_index(A, ica.components_) > 0.3


def test_transform_coherent_avec_fit():
    S, _ = make_sources(2)
    X = make_mixing(3, 1) @ S
    ica = SGDICA(n_epochs=10, random_state=0)
    np.testing.assert_allclose(ica.fit_transform(X), ica.transform(X))


def test_benchmark_fastica():
    res = run_benchmark("fastica", 1, n_runs=3)
    assert res.scores.shape == (3,)
    assert res.mean < 0.05


# ---------------------------------------------------------------- IRMf (données synthétiques)
def _fake_fmri(T=120, p=3000, k=5, seed=0):
    """k cartes spatiales parcimonieuses (super-gaussiennes) × séries temporelles + bruit."""
    rng = np.random.default_rng(seed)
    maps = rng.laplace(size=(k, p)) * (rng.random((k, p)) < 0.1)
    ts = rng.standard_normal((T, k))
    return ts @ maps * 3 + rng.standard_normal((T, p)), k


def test_select_q_bic_retrouve_le_rang():
    X, k = _fake_fmri()
    q, scores = select_q_bic(X, q_max=30)
    assert q == k
    assert scores.shape == (30,)


@pytest.mark.parametrize("method", ["fastica", "sgd"])
def test_pica_formes_et_cartes(method):
    X, k = _fake_fmri()
    res = pica(X, q=k, method=method, **({"n_epochs": 20} if method == "sgd" else {}))
    assert res.maps.shape == (k, X.shape[1])
    assert res.sources.shape == (X.shape[0], k)
    assert res.map_kurtosis.shape == (k,)
    assert res.map_kurtosis.mean() > 1  # les cartes retrouvées sont nettement non gaussiennes
