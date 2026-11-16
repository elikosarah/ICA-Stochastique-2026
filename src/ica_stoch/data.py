"""Génération des sources simulées et des mélanges.

Toutes les méthodes sont évaluées sur **exactement les mêmes sources** : elles sont
générées une seule fois par scénario, à partir d'un générateur aléatoire local
(``numpy.random.default_rng``) et jamais à partir de l'état global de NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SAMPLES = 2000
DURATION = 8.0


@dataclass(frozen=True)
class Scenario:
    key: int
    name: str
    description: str


SCENARIOS = {
    1: Scenario(1, "Jeu 1 — mixte", "sinusoïde, signal carré, Laplace (sub- et super-gaussiennes)"),
    2: Scenario(2, "Jeu 2 — super-gaussien", "sinusoïde, exponentielle, Laplace"),
    3: Scenario(3, "Jeu 3 — sub-gaussien", "sinusoïde, signal carré, uniforme"),
}


def make_sources(
    scenario: int, n_samples: int = N_SAMPLES, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie ``(S, time)`` : 3 sources indépendantes de variance 1, de forme (3, n_samples)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario doit être dans {sorted(SCENARIOS)}, reçu {scenario!r}")

    rng = np.random.default_rng(seed)
    time = np.linspace(0, DURATION, n_samples)
    sinus = np.sin(2 * np.pi * 3 * time)  # sub-gaussienne (kurtosis ≈ -1,5)
    square = np.sign(np.sin(3 * np.pi * time))  # sub-gaussienne (kurtosis = -2)

    if scenario == 1:
        S = np.vstack([sinus, square, rng.laplace(0, 1, n_samples)])
    elif scenario == 2:
        S = np.vstack([sinus, rng.exponential(1, n_samples), rng.laplace(0, 1, n_samples)])
    else:
        S = np.vstack([sinus, square, rng.uniform(-np.sqrt(3), np.sqrt(3), n_samples)])

    S = S - S.mean(axis=1, keepdims=True)
    S /= S.std(axis=1, keepdims=True)
    return S, time


def make_mixing(n_sources: int, seed: int) -> np.ndarray:
    """Matrice de mélange gaussienne, une par tirage (``seed`` = numéro du tirage)."""
    return np.random.default_rng(seed).standard_normal((n_sources, n_sources))


def excess_kurtosis(S: np.ndarray) -> np.ndarray:
    """Kurtosis en excès de chaque ligne (0 pour une gaussienne)."""
    Z = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    return np.mean(Z**4, axis=1) - 3.0
