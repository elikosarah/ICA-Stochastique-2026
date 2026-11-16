# ICA et optimisation stochastique — comparaison d'algorithmes et application à l'IRM fonctionnelle

Projet de Master 2 Data Science (Université Paris-Saclay, cours de Réduction de dimension, avril 2026).

**En bref :** j'ai implémenté et comparé plusieurs algorithmes d'Analyse en Composantes Indépendantes (ICA) sur des signaux simulés, en mesurant la qualité de séparation par l'indice d'Amari sur 20 tirages. J'ai ensuite appliqué une ICA probabiliste (PICA) à des données réelles d'IRM fonctionnelle (cohorte ADHD-200), avec un critère bayésien de Minka pour choisir le nombre de composantes.

📄 [Rapport complet (PDF, 16 pages)](Rapport_ICA_Sarah_Eliko.pdf) · 💻 [Notebook](ICA_Sarah_Eliko-code.ipynb)

---

## Question étudiée

Quand on observe un mélange linéaire de signaux `x = A s`, peut-on retrouver les sources `s` sans connaître `A` ? L'ICA y répond en cherchant des composantes aussi **non gaussiennes** que possible. Le projet regarde :

1. comment les algorithmes classiques (Infomax, FastICA) et une version **stochastique** (SGD-ICA, pas de Robbins-Monro) se comportent selon la nature des sources ;
2. si ces méthodes permettent d'extraire des composantes interprétables dans un signal d'IRM fonctionnelle réel.

## Méthode

**Données simulées** : 3 sources, 2 000 points, 3 scénarios, et une nouvelle matrice de mélange aléatoire à chaque tirage (20 tirages).

| Jeu | Sources | Type |
|---|---|---|
| 1 | sinusoïde, signal carré, Laplace | mixte (sub- et super-gaussiennes) |
| 2 | sinusoïde, exponentielle, Laplace | majoritairement super-gaussiennes |
| 3 | sinusoïde, signal carré, uniforme | sub-gaussiennes |

**Métrique** : indice d'Amari entre la vraie matrice de mélange et la matrice de séparation estimée (0 = séparation parfaite).

**Algorithmes**
- Infomax batch (Bell & Sejnowski, 1995), via MNE, avec `g = tanh`
- FastICA (Hyvärinen & Oja, 1997), via scikit-learn, avec `G = log cosh`
- **SGD-ICA, implémentée à la main** : blanchiment, montée de gradient stochastique sur la log-vraisemblance, pas décroissant, recherche du taux d'apprentissage par grille

**Données réelles** : un sujet de la cohorte ADHD-200 (176 volumes × 69 681 voxels), chargé avec `nilearn`, lissé et standardisé. On réduit d'abord la dimension (PPCA), on choisit `q*`, puis on applique l'ICA (FastICA ou SGD-ICA). Les composantes sont classées par kurtosis.

## Résultats

### Indice d'Amari moyen sur 20 tirages (plus bas = mieux)

| Algorithme | Jeu 1 (mixte) | Jeu 2 (super-gaussien) | Jeu 3 (sub-gaussien) |
|---|---|---|---|
| Infomax batch (`tanh`) | 0,383 ± 0,057 | 0,141 ± 0,025 | 0,692 ± 0,018 |
| FastICA (`log cosh`) | **0,007** | **0,023** | **0,008** |
| SGD-ICA (implémentation maison) | 0,274 ± 0,160 | 0,399 ± 0,127 | 0,037 ± 0,007 |

**À retenir**
- **FastICA** sépare presque parfaitement les sources dans les trois scénarios.
- **Infomax avec `g = tanh` fixe** suppose des sources super-gaussiennes. Il marche correctement sur le jeu 2, mais échoue sur des sources sub-gaussiennes (jeu 3). Le choix de la non-linéarité doit donc correspondre au type de source.
- **SGD-ICA** est sensible au taux d'apprentissage, ce qui explique sa variance plus élevée. Bien réglée, elle donne de très bons résultats sur le jeu 3.

### IRM fonctionnelle (ADHD-200)

- L'estimation automatique de la dimension par la PPCA renvoie `q* = 173` pour seulement 176 volumes. C'est une sur-estimation, et les composantes extraites restent peu non gaussiennes (kurtosis moyen 0,54).
- Avec le **critère de Minka**, on obtient `q* = 6` : les composantes sont nettement plus non gaussiennes (kurtosis moyen 1,95), donc mieux séparées.
- **Conclusion** : sur des données d'IRM, le choix de la dimension compte plus que le choix de l'algorithme d'ICA.

## Reproduire

```bash
git clone https://github.com/elikosarah/ICA-Stochastique-2026.git
cd ICA-Stochastique-2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ICA_Sarah_Eliko-code.ipynb
```

Les données ADHD-200 sont téléchargées automatiquement par `nilearn` (environ 100 Mo pour un sujet). Les graines aléatoires sont fixées.

## Structure

```
├── ICA_Sarah_Eliko-code.ipynb   # code et résultats (simulations + IRMf)
├── Rapport_ICA_Sarah_Eliko.pdf  # rapport : théorie, expériences, discussion
├── requirements.txt
└── README.md
```

## Limites et pistes

- La version **gradient naturel** (Amari, 1998) et une version **Infomax étendue**, qui adapte `g` au type de source, sont les suites naturelles pour corriger l'échec d'Infomax sur des sources sub-gaussiennes.
- L'application à l'IRM fonctionnelle porte sur un seul sujet. Une analyse de groupe (ICA de groupe) serait l'étape suivante.

## Stack

Python · NumPy · scikit-learn · MNE · nilearn · SciPy · Matplotlib

