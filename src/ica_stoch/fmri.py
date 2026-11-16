r"""ICA probabiliste (PICA) sur données d'IRM fonctionnelle.

Pipeline (Beckmann & Smith, 2004, version simplifiée) :

1. PCA préliminaire : on passe de ``p`` voxels à ``T - 1`` dimensions (``T`` = nombre de volumes) ;
2. PPCA : on garde ``q`` composantes et on estime la variance du bruit :math:`\sigma^2` ;
3. blanchiment corrigé du bruit : chaque composante est divisée par :math:`\sqrt{\lambda_j - \sigma^2}` ;
4. ICA (FastICA ou SGD-ICA) dans cet espace réduit ;
5. retour dans l'espace des voxels pour obtenir une carte cérébrale par composante.

Le choix de ``q`` est fait par :func:`select_q_bic`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA

from .sgd_ica import SGDICA


def load_adhd_subject(subject: int = 0, smoothing_fwhm: float = 6.0, data_dir: str | None = None):
    """Télécharge un sujet ADHD-200 (via nilearn) et renvoie ``(X, masker)``, ``X`` de forme (T, p)."""
    from nilearn import datasets
    from nilearn.maskers import NiftiMasker

    dataset = datasets.fetch_adhd(n_subjects=subject + 1, data_dir=data_dir)
    masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm, standardize="zscore_sample", memory="nilearn_cache", verbose=0)
    X = masker.fit_transform(dataset.func[subject])
    return X, masker


def pca_eigenvalues(X: np.ndarray) -> np.ndarray:
    """Valeurs propres non nulles de la covariance empirique de ``X`` (T, p), par SVD (sans matrice p × p)."""
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s**2 / X.shape[0]


def select_q_bic(X: np.ndarray, q_max: int = 50, noise: str = "ml") -> tuple[int, np.ndarray]:
    r"""Choisit le nombre de composantes ``q`` d'une PPCA par critère BIC (Minka, 2000, §4).

    Log-vraisemblance de la PPCA au maximum, pour ``T`` observations en dimension ``p`` :

    .. math:: \ell(q) = -\frac{T}{2}\Big[\sum_{j\le q}\log\lambda_j + (p-q)\log\hat\sigma^2 + p\Big],
              \qquad \mathrm{BIC}(q) = \ell(q) - \frac{1}{2}\Big(pq - \frac{q(q-1)}{2}\Big)\log T.

    Parameters
    ----------
    noise : {"ml", "legacy"}
        ``"ml"`` : estimateur du maximum de vraisemblance
        :math:`\hat\sigma^2 = \sum_{j>q}\lambda_j / (p-q)`, où la somme porte sur **toutes** les
        directions restantes (y compris les :math:`p - T` valeurs propres nulles quand :math:`p > T`).
        ``"legacy"`` : moyenne des seules valeurs propres non nulles restantes (version du rapport
        d'avril 2026), qui surestime :math:`\sigma^2` quand :math:`p \gg T`.

    Returns
    -------
    q_star, scores : int, ndarray de longueur ``q_max`` (``scores[q-1]`` = BIC(q))
    """
    T, p = X.shape
    lam = pca_eigenvalues(X)
    q_max = min(q_max, len(lam) - 1)
    total = lam.sum()
    scores = np.empty(q_max)
    for q in range(1, q_max + 1):
        if noise == "ml":
            sigma2 = (total - lam[:q].sum()) / (p - q)
        elif noise == "legacy":
            sigma2 = lam[q:].mean()
        else:
            raise ValueError("noise doit valoir 'ml' ou 'legacy'")
        sigma2 = max(sigma2, 1e-12)
        loglik = -0.5 * T * (np.log(lam[:q]).sum() + (p - q) * np.log(sigma2) + p)
        n_params = p * q - q * (q - 1) / 2
        scores[q - 1] = loglik - 0.5 * n_params * np.log(T)
    return int(np.argmax(scores) + 1), scores


@dataclass
class PICAResult:
    q: int
    sigma2: float
    sources: np.ndarray  # (T, q) séries temporelles
    maps: np.ndarray  # (q, p) cartes spatiales
    map_kurtosis: np.ndarray  # (q,) kurtosis en excès de chaque carte
    eigenvalues: np.ndarray  # spectre de la PPCA

    def ranking(self) -> np.ndarray:
        """Indices des composantes, de la plus non gaussienne à la moins non gaussienne."""
        return np.argsort(-self.map_kurtosis)


def pica(X: np.ndarray, q: int | str = "mle", method: str = "fastica", seed: int = 42, **ica_params) -> PICAResult:
    """PICA sur ``X`` (T, p). ``q`` : entier, ou ``"mle"`` pour le choix automatique de scikit-learn."""
    T = X.shape[0]
    pca_pre = PCA(n_components=T - 1, svd_solver="full")
    X_pre = pca_pre.fit_transform(X)

    ppca = PCA(n_components=q, svd_solver="full").fit(X_pre)
    sigma2 = float(ppca.noise_variance_)
    scale = 1.0 / np.sqrt(np.maximum(ppca.explained_variance_ - sigma2, 1e-10))
    X_white = ((X_pre - ppca.mean_) @ ppca.components_.T) * scale  # (T, q)
    n_comp = X_white.shape[1]

    if method == "fastica":
        ica = FastICA(fun="logcosh", whiten=False, max_iter=1000, random_state=seed, **ica_params)
        sources = ica.fit_transform(X_white)
        unmixing = ica.components_
    elif method == "sgd":
        ica = SGDICA(random_state=seed, **ica_params).fit(X_white.T)
        sources = ica.transform(X_white.T).T
        unmixing = ica.components_  # inclut le blanchiment interne de SGDICA
    else:
        raise ValueError("method doit valoir 'fastica' ou 'sgd'")

    W_ppca = ppca.components_ * scale[:, None]  # (q, T-1)
    maps = unmixing @ W_ppca @ pca_pre.components_  # (q, p)
    return PICAResult(
        q=n_comp,
        sigma2=sigma2,
        sources=sources,
        maps=maps,
        map_kurtosis=kurtosis(maps, axis=1),
        eigenvalues=ppca.explained_variance_,
    )
