# ICA et Optimisation Stochastique : Analyse Comparative et PICA

Ce dépôt contient le projet final de Master 2 Data Science (Université Paris-Saclay) pour le cours de Réduction de Dimension. L'objectif est d'étudier l'Analyse en Composantes Indépendantes (ICA) sous l'angle de l'optimisation stochastique et de l'appliquer à l'imagerie médicale (fMRI).

## 📝 Présentation du projet
Le projet se divise en deux parties majeures :
1. **Analyse comparative** : Étude de 4 variantes d'ICA (Infomax Batch, FastICA, SGD-ICA et Gradient Naturel) sur des jeux de données synthétiques avec mesure de performance via l'**Indice d'Amari**.
2. **Contribution Créative (PICA)** : Implémentation de la *Probabilistic ICA* (mélange de PPCA et ICA) appliquée au nettoyage de signaux IRM fonctionnelle (Sujet de la cohorte **ADHD-200**).

## 🚀 Installation et Reproduction

### 1. Prérequis
Il est recommandé d'utiliser un environnement virtuel. Les dépendances incluent `scikit-learn` pour les baselines et `nilearn` pour la manipulation des données neuroimagerie.

```bash
pip install -r requirements.txt
