"""Figures réutilisées dans les notebooks et le script de benchmark."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROW_STYLE = [("Source originale", "#2a6f97"), ("Mélange observé", "#8d99ae"), ("Source estimée", "#d1495b")]


def plot_separation(time, S, X, S_est, title: str = ""):
    """Grille 3 × D : sources, mélanges, sources estimées."""
    D = S.shape[0]
    fig, axes = plt.subplots(3, D, figsize=(4.5 * D, 6.5), sharex=True)
    for row, (data, (label, color)) in enumerate(zip((S, X, S_est), ROW_STYLE)):
        for i in range(D):
            ax = axes[row, i]
            ax.plot(time, data[i], color=color, lw=0.7)
            ax.set_title(f"{label} {i + 1}", fontsize=9)
            ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Temps (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_amari_comparison(results, title: str = "Indice d'Amari sur 20 tirages (plus bas = mieux)"):
    """Boîtes à moustaches par méthode, un panneau par scénario. ``results`` : liste de BenchmarkResult."""
    scenarios = sorted({r.scenario for r in results})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.5 * len(scenarios), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenarios):
        res = [r for r in results if r.scenario == sc]
        labels = [r.label or r.method for r in res]
        ax.boxplot([r.scores for r in res], tick_labels=labels, widths=0.6)
        for i, r in enumerate(res, start=1):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(r.scores))
            ax.scatter(np.full(len(r.scores), i) + jitter, r.scores, s=10, alpha=0.6, color="#2a6f97", zorder=3)
        ax.set_title(f"Jeu {sc}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Indice d'Amari")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_grid_search(df: pd.DataFrame, title: str = ""):
    d = df.sort_values("learning_rate")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(d["learning_rate"], d["amari_mean"], yerr=d["amari_std"], marker="o", capsize=3, color="#2a6f97")
    ax.set_xscale("log")
    ax.set_xlabel("Pas initial η₀")
    ax.set_ylabel("Indice d'Amari moyen")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_convergence(objective, title: str = "Convergence de SGD-ICA"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(objective) + 1), objective, color="#2a6f97")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Log-vraisemblance moyenne")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
