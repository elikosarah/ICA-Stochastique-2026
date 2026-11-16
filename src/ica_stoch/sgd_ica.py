r"""ICA par montée de gradient stochastique (implémentation maison).

Modèle : :math:`x = A s`. On blanchit les données (:math:`z = V x`, :math:`\mathrm{Cov}(z) = I`),
puis on cherche une matrice :math:`W` qui maximise la log-vraisemblance

.. math:: \ell(W) = \log|\det W| + \mathbb{E}\big[\textstyle\sum_i \log p_i(w_i^\top z)\big].

Densités des sources (familles de l'Infomax étendu, Lee et al. 1999), avec
:math:`k_i = +1` pour une source super-gaussienne et :math:`k_i = -1` pour une sub-gaussienne :

.. math:: \log p_i(y) = -k_i \log\cosh(y) - y^2/2 + \text{cste},
          \qquad \varphi_i(y) = \frac{d}{dy}\log p_i(y) = -k_i \tanh(y) - y.

Mises à jour, pour un exemple :math:`z_t` et :math:`y_t = W z_t` :

* gradient ordinaire : :math:`W \leftarrow W + \eta_t\,(W^{-\top} + \varphi(y_t) z_t^\top)`
* gradient naturel (Amari, 1998) : :math:`W \leftarrow W + \eta_t\,(I + \varphi(y_t) y_t^\top)\,W`,
  qui évite d'inverser :math:`W` à chaque pas. Comme les données sont blanchies, la solution est
  une rotation : on utilise la version restreinte aux matrices orthogonales (gradient relatif
  antisymétrique, cf. Cardoso & Laheld, 1996), :math:`W \leftarrow W + \eta_t\,(G - G^\top) W`
  avec :math:`G = \varphi(y_t) y_t^\top`, appliquée par une transformation de Cayley
  :math:`W \leftarrow (I - \tfrac{\eta_t}{2}K)^{-1}(I + \tfrac{\eta_t}{2}K)\,W`, :math:`K = G - G^\top`,
  qui garde :math:`W` exactement orthogonale : la méthode ne diverge pas, même avec un grand pas.

Le pas suit un schéma de Robbins-Monro, :math:`\eta_e = \eta_0/(1+e)` où :math:`e` est
l'**époque** : ainsi le pas ne s'effondre pas dès la première passe sur les données.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Nonlinearity = Literal["super", "sub", "extended"]
Decay = Literal["inv", "sqrt", "constant"]


def whiten(X: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre et blanchit ``X`` (forme (D, n)). Renvoie ``(Z, V, mean)`` avec ``Z = V (X - mean)``."""
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    eigval, eigvec = np.linalg.eigh(np.atleast_2d(np.cov(Xc)))
    eigval = np.maximum(eigval, eps)
    V = (eigvec / np.sqrt(eigval)).T
    return V @ Xc, V, mean


def _sym_orthogonalize(W: np.ndarray) -> np.ndarray:
    """Projection sur les matrices orthogonales : W ← (W Wᵀ)^{-1/2} W."""
    eigval, eigvec = np.linalg.eigh(W @ W.T)
    return eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T @ W


def _source_signs(Y: np.ndarray) -> np.ndarray:
    """Critère de l'Infomax étendu : +1 si la source estimée est super-gaussienne, -1 sinon."""
    crit = np.mean(1 / np.cosh(Y) ** 2, axis=1) * np.mean(Y**2, axis=1) - np.mean(np.tanh(Y) * Y, axis=1)
    signs = np.sign(crit)
    signs[signs == 0] = 1.0
    return signs


class SGDICA:
    """ICA par gradient stochastique, API proche de scikit-learn.

    Parameters
    ----------
    nonlinearity : {"super", "sub", "extended"}
        Hypothèse sur les sources. ``"extended"`` choisit le signe de chaque composante
        à chaque époque (Infomax étendu) et fonctionne sur des sources mixtes.
    natural_gradient : bool
        Utilise le gradient naturel au lieu du gradient ordinaire.
    learning_rate : float
        Pas initial :math:`\\eta_0`.
    decay : {"inv", "sqrt", "constant"}
        Décroissance du pas par époque : :math:`\\eta_0/(1+e)`, :math:`\\eta_0/\\sqrt{1+e}` ou constant.
    n_epochs : int
        Nombre maximal de passes sur les données.
    ortho_every : int
        Reprojection sur les matrices orthogonales tous les ``ortho_every`` exemples (et à chaque
        fin d'époque). Stabilise le gradient naturel quand le pas est grand.
    tol : float
        Arrêt quand la variation de l'objectif entre deux époques passe sous ``tol``.
    random_state : int | None
        Graine (initialisation et ordre de parcours).

    Attributes
    ----------
    components_ : ndarray (D, D)
        Matrice de séparation dans l'espace d'origine (``S_est = components_ @ (X - mean_)``).
    rotation_ : ndarray (D, D)
        Matrice :math:`W` apprise dans l'espace blanchi.
    whitening_ : ndarray (D, D)
        Matrice de blanchiment :math:`V`.
    signs_ : ndarray (D,)
        Type estimé de chaque source (+1 super-gaussienne, -1 sub-gaussienne).
    objective_ : list[float]
        Log-vraisemblance moyenne (à une constante près) à la fin de chaque époque.
    n_epochs_ : int
        Nombre d'époques effectuées.
    """

    def __init__(
        self,
        nonlinearity: Nonlinearity = "extended",
        natural_gradient: bool = True,
        learning_rate: float = 0.05,
        decay: Decay = "inv",
        n_epochs: int = 50,
        tol: float = 1e-6,
        ortho_every: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if nonlinearity not in ("super", "sub", "extended"):
            raise ValueError("nonlinearity doit valoir 'super', 'sub' ou 'extended'")
        if decay not in ("inv", "sqrt", "constant"):
            raise ValueError("decay doit valoir 'inv', 'sqrt' ou 'constant'")
        self.nonlinearity = nonlinearity
        self.natural_gradient = natural_gradient
        self.learning_rate = learning_rate
        self.decay = decay
        self.n_epochs = n_epochs
        self.tol = tol
        self.ortho_every = ortho_every
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ utilitaires
    def _step_size(self, epoch: int) -> float:
        if self.decay == "inv":
            return self.learning_rate / (1 + epoch)
        if self.decay == "sqrt":
            return self.learning_rate / np.sqrt(1 + epoch)
        return self.learning_rate

    def _signs(self, Y: np.ndarray) -> np.ndarray:
        if self.nonlinearity == "super":
            return np.ones(Y.shape[0])
        if self.nonlinearity == "sub":
            return -np.ones(Y.shape[0])
        return _source_signs(Y)

    def _project(self, W: np.ndarray, epoch: int) -> np.ndarray:
        if not np.all(np.isfinite(W)):
            raise FloatingPointError(
                f"SGD-ICA a divergé à l'époque {epoch} (pas initial {self.learning_rate}) : réduire learning_rate"
            )
        return _sym_orthogonalize(W)

    @staticmethod
    def _objective(W: np.ndarray, Z: np.ndarray, signs: np.ndarray) -> float:
        Y = W @ Z
        log_p = -signs[:, None] * np.log(np.cosh(Y)) - 0.5 * Y**2
        return float(np.log(abs(np.linalg.det(W))) + log_p.sum(axis=0).mean())

    # ------------------------------------------------------------------ API
    def fit(self, X: np.ndarray) -> "SGDICA":
        """Apprend la séparation sur ``X`` de forme (n_sources, n_samples)."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        Z, self.whitening_, self.mean_ = whiten(X)
        D, n = Z.shape
        identity = np.eye(D)

        W = _sym_orthogonalize(rng.standard_normal((D, D)))
        self.objective_ = []

        for epoch in range(self.n_epochs):
            signs = self._signs(W @ Z)
            eta = self._step_size(epoch)

            for step, i in enumerate(rng.permutation(n), start=1):
                z = Z[:, i : i + 1]
                y = W @ z
                phi = -signs[:, None] * np.tanh(y) - y
                if self.natural_gradient:
                    # Gradient naturel restreint aux rotations : partie antisymétrique K de φ(y) yᵀ
                    # (le terme I - y yᵀ, d'espérance nulle sur données blanchies, disparaît).
                    # Pas de Cayley : W reste exactement orthogonale, quel que soit le pas.
                    G = phi @ y.T
                    K = 0.5 * eta * (G - G.T)
                    W = np.linalg.solve(identity - K, (identity + K) @ W)
                else:
                    W = W + eta * (np.linalg.inv(W).T + phi @ z.T)
                if step % self.ortho_every == 0:
                    W = self._project(W, epoch)

            # Les données étant blanchies, la solution est une rotation : on reprojette.
            W = self._project(W, epoch)
            self.objective_.append(self._objective(W, Z, signs))

            if self.verbose and epoch % 10 == 0:
                print(f"époque {epoch:4d} | objectif = {self.objective_[-1]:.6f}")
            if epoch > 0 and abs(self.objective_[-1] - self.objective_[-2]) < self.tol:
                break

        self.rotation_ = W
        self.signs_ = self._signs(W @ Z)
        self.components_ = W @ self.whitening_
        self.n_epochs_ = epoch + 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Sources estimées pour ``X`` (n_sources, n_samples)."""
        return self.components_ @ (np.asarray(X, dtype=float) - self.mean_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
