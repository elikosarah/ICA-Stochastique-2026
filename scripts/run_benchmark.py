"""Reproduit le tableau de résultats et les figures des simulations.

Usage : python scripts/run_benchmark.py [--runs 20]
Sorties : results/amari_summary.csv, results/sgd_grid_search.csv, results/figures/*.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

from ica_stoch import grid_search_sgd, make_sources, run_benchmark  # noqa: E402
from ica_stoch.plots import plot_amari_comparison, plot_grid_search, plot_separation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = OUT / "figures"

LEARNING_RATES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def configurations(lr: float):
    sgd = dict(learning_rate=lr, n_epochs=50)
    return [
        ("Infomax (MNE, tanh)", "infomax", {}),
        ("FastICA (log cosh)", "fastica", {}),
        ("SGD, hyp. super-gauss.", "sgd", dict(nonlinearity="super", natural_gradient=False, **sgd)),
        ("SGD, hyp. sub-gauss.", "sgd", dict(nonlinearity="sub", natural_gradient=False, **sgd)),
        ("SGD étendu", "sgd", dict(nonlinearity="extended", natural_gradient=False, **sgd)),
        ("SGD étendu, grad. naturel", "sgd", dict(nonlinearity="extended", natural_gradient=True, **sgd)),
    ]


def main(n_runs: int) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Choix du pas initial pour SGD-ICA étendu (gradient naturel), 5 tirages par valeur
    grids = []
    for sc in (1, 2, 3):
        g = grid_search_sgd(sc, LEARNING_RATES, n_runs=5, nonlinearity="extended", n_epochs=50)
        g["scenario"] = sc
        grids.append(g)
        plot_grid_search(g, f"Recherche du pas — jeu {sc}").savefig(FIG / f"grid_search_jeu{sc}.png", dpi=150)
    grid = pd.concat(grids, ignore_index=True)
    grid.to_csv(OUT / "sgd_grid_search.csv", index=False)
    lr = float(grid.groupby("learning_rate")["amari_mean"].mean().idxmin())
    print(f"Pas initial retenu (meilleure moyenne sur les 3 jeux) : {lr}")

    # 2. Benchmark : mêmes sources et mêmes matrices de mélange pour toutes les méthodes
    results, rows = [], []
    for sc in (1, 2, 3):
        for label, method, params in configurations(lr):
            res = run_benchmark(method, sc, n_runs=n_runs, label=label, **params)
            print(res.summary())
            results.append(res)
            rows.append({"methode": label, "jeu": sc, "amari_moyen": res.mean, "amari_ecart_type": res.std})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "amari_summary.csv", index=False)
    table = summary.pivot(index="methode", columns="jeu", values="amari_moyen")
    table = table.loc[[c[0] for c in configurations(lr)]]
    print("\n", table.round(4).to_string())

    plot_amari_comparison(results).savefig(FIG / "amari_comparaison.png", dpi=150)

    # 3. Exemple de séparation : meilleur tirage de SGD étendu (gradient naturel), jeu 1
    S, t = make_sources(1)
    best = next(r for r in results if r.scenario == 1 and r.label == "SGD étendu, grad. naturel").best_run
    plot_separation(t, S, best["X"], best["S_est"], f"SGD-ICA étendu — jeu 1 (Amari = {best['amari']:.4f})").savefig(
        FIG / "separation_jeu1.png", dpi=150
    )
    print(f"\nTerminé en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    main(parser.parse_args().runs)
