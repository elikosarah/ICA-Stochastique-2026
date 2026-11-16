"""Indice d'Amari : qualité de séparation d'un algorithme d'ICA."""

from __future__ import annotations

import numpy as np


def amari_index(mixing: np.ndarray, unmixing: np.ndarray) -> float:
    r"""Indice d'Amari entre la vraie matrice de mélange ``A`` et la matrice de séparation ``W``.

    On forme :math:`C = W A`. Si la séparation est parfaite, :math:`C` est une matrice de
    permutation à un facteur d'échelle près et l'indice vaut 0. Il vaut au plus 1.

    .. math::
        r(C) = \frac{1}{2D(D-1)}\Big[\sum_i\Big(\sum_j \frac{|c_{ij}|}{\max_k |c_{ik}|}-1\Big)
               + \sum_j\Big(\sum_i \frac{|c_{ij}|}{\max_k |c_{kj}|}-1\Big)\Big]
    """
    C = np.abs(np.asarray(unmixing) @ np.asarray(mixing))
    D = C.shape[0]
    if C.shape != (D, D):
        raise ValueError(f"W @ A doit être carrée, forme obtenue {C.shape}")
    rows = (C / C.max(axis=1, keepdims=True)).sum(axis=1) - 1
    cols = (C / C.max(axis=0, keepdims=True)).sum(axis=0) - 1
    return float((rows.sum() + cols.sum()) / (2 * D * (D - 1)))
