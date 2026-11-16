"""Protocole expérimental commun à tous les algorithmes.

Pour chaque scénario, les sources sont générées une fois ; à chaque tirage on tire une
nouvelle matrice de mélange ``A``, on ajuste l'algorithme et on calcule l'indice d'Amari.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from .data import make_mixing, make_sources
from .metrics import amari_index
from .sgd_ica import SGDICA

METHODS = ("infomax", "fastica", "sgd")


def fit_unmixing(method: str, X: np.ndarray, seed: int, **params) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ajuste ``method`` sur ``X`` (n_sources, n_samples).

    Renvoie ``(W, S_est, info)`` où ``W`` est la matrice de séparation dans l'espace d'origine.
    """
    n_sources = X.shape[0]

    if method == "infomax":
        import mne  # import local : dépendance lourde, utile pour cette méthode seulement
        from mne.preprocessing import ICA

        info = mne.create_info([f"CH{i + 1}" for i in range(n_sources)], sfreq=250.0, ch_types="eeg")
        raw = mne.io.RawArray(X, info, verbose=False)
        ica = ICA(
            n_components=n_sources,
            method="infomax",
            fit_params=dict(extended=params.get("extended", False)),
            random_state=seed,
            verbose=False,
        )
        ica.fit(raw, verbose=False)
        W = ica.unmixing_matrix_ @ ica.pca_components_
        return W, ica.get_sources(raw).get_data(), {}

    if method == "fastica":
        ica = FastICA(
            n_components=n_sources,
            algorithm="parallel",
            fun=params.get("fun", "logcosh"),
            max_iter=params.get("max_iter", 2000),
            tol=params.get("tol", 1e-6),
            whiten="unit-variance",
            random_state=seed,
        )
        S_est = ica.fit_transform(X.T).T
        return ica.components_, S_est, {"n_iter": ica.n_iter_}

    if method == "sgd":
        ica = SGDICA(random_state=seed, **params)
        S_est = ica.fit_transform(X)
        return ica.components_, S_est, {"objective": ica.objective_, "n_epochs": ica.n_epochs_}

    raise ValueError(f"méthode inconnue {method!r}, attendu l'une de {METHODS}")


@dataclass
class BenchmarkResult:
    method: str
    scenario: int
    params: dict
    scores: np.ndarray
    runs: list = field(repr=False)
    label: str = ""

    @property
    def mean(self) -> float:
        return float(self.scores.mean())

    @property
    def std(self) -> float:
        return float(self.scores.std())

    @property
    def best_run(self) -> dict:
        return self.runs[int(np.argmin(self.scores))]

    def summary(self) -> str:
        return f"{self.label or self.method:28s} | jeu {self.scenario} | Amari = {self.mean:.4f} ± {self.std:.4f}"


def run_benchmark(
    method: str, scenario: int, n_runs: int = 20, source_seed: int = 42, label: str = "", **params
) -> BenchmarkResult:
    """Évalue ``method`` sur ``n_runs`` matrices de mélange différentes."""
    S, _ = make_sources(scenario, seed=source_seed)
    scores, runs = [], []
    for run in range(n_runs):
        A = make_mixing(S.shape[0], seed=run)
        X = A @ S
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, S_est, info = fit_unmixing(method, X, seed=run, **params)
        score = amari_index(A, W)
        scores.append(score)
        runs.append({"run": run, "A": A, "X": X, "S_est": S_est, "amari": score, **info})
    return BenchmarkResult(method, scenario, params, np.array(scores), runs, label or method)


def grid_search_sgd(
    scenario: int, learning_rates, n_runs: int = 5, **params
) -> pd.DataFrame:
    """Indice d'Amari moyen de SGD-ICA pour chaque pas initial. Trié du meilleur au moins bon."""
    rows = []
    for lr in learning_rates:
        try:
            res = run_benchmark("sgd", scenario, n_runs=n_runs, learning_rate=lr, **params)
            rows.append({"learning_rate": lr, "amari_mean": res.mean, "amari_std": res.std})
        except FloatingPointError:  # pas trop grand : l'algorithme diverge
            rows.append({"learning_rate": lr, "amari_mean": np.nan, "amari_std": np.nan})
    return pd.DataFrame(rows).sort_values("amari_mean").reset_index(drop=True)
