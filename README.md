# BDS_Fault_Detection — Détection de défauts de roulements (banc CWRU)

Projet de détection et classification des défauts de roulements à partir de signaux
vibratoires issus du banc d'essai **CWRU** (Case Western Reserve University) :
détection sain/défectueux, localisation du défaut (IR = bague interne, B = billes,
OR = bague externe), sévérité (diamètre 7/14/21 mils), sur deux capteurs :
**DE** (Drive End) et **FE** (Fan End), échantillonnés à 12 kHz.

## Contenu du dépôt

| Dossier / fichier | Contenu |
|---|---|
| `Sources/` | Modules du projet : `data.py` (chargement CWRU), `features.py` (fenêtrage, tables, splits), `indicateurs.py` (indicateurs temporels/spectraux), `signaux.py` (filtrage, enveloppe), `models.py` (modèles ML + Conv1D-AE), `config.py`, `path.py` |
| `Sources/Fitted_models/` | Modèles de deep learning sauvegardés (`DAE_fitted.keras`, ...) |
| `Codes/` | Notebooks d'analyse et de modélisation (`01_analyse_exploratoire.qmd`, `03_indicateurs_par_fenetre.ipynb`, `07_cnn1d.ipynb`, `08_reference_rms.ipynb`, `09_ae_transfert_DE_FE.ipynb`, `Jeu_Apprentissage_Blaise.qmd`, ...) |
| `Codes/Robin/` | Travaux de l'équipe : `0.DataImport`, `1.AnalyseExploratoire`, `2.ML`, `3.DL/AE_1D` (auto-encodeurs AE/DAE, transfert DE↔FE), `_Rapports` |
| `code.py` | **Script autonome Random Forest** (voir ci-dessous) |
| `data/` | Données (cache CWRU dans `data/cwru_cache/`, gitigné) et tables d'indicateurs (`data/features/`, gitigné) |
| `Docs/`, `images/` | Rapports et figures |

## Prérequis

```bash
pip install -r requirements.txt        # numpy, scipy, pandas, matplotlib, scikit-learn, ...
python -m pip install tensorflow keras joblib   # pour les notebooks de deep learning
```

## Démarrage rapide — Random Forest (`code.py`)

`code.py` (à la racine ou dans `Codes/`) est un script autonome, inspiré des notebooks
`Codes/Robin/2.ML`. Trois fonctionnalités :

1. `preparer_donnees()` — construit la table d'indicateurs par fenêtre (0,2 s) et
   découpe **train = capteur DE / test = capteur FE** (standardisation fit sur le train
   seul, équilibrage des classes sur le train uniquement) ;
2. `entrainer_rf()` / `optimiser_rf()` — Random Forest par défaut, ou optimisé par
   `RandomizedSearchCV` (grille élargie, CV groupée par fichier, pas de fuite) ;
3. `predire(rf, X)` — prédiction des classes (`predict_proba` pour les probabilités).

```bash
python code.py        # premier run : télécharge CWRU + calcule les indicateurs (quelques minutes)
```

Sauvegarde du modèle entraîné (contient les hyperparamètres optimisés) :

```python
import joblib
joblib.dump(rf, "data/modele_RF.joblib")     # puis : rf = joblib.load("data/modele_RF.joblib")
```

⚠️ Recharger avec la même version de scikit-learn, et prédire avec les mêmes
features dans le même ordre (`FEATURES` en tête de `code.py`).

## Deux approches dans le projet

1. **ML sur indicateurs** (`code.py`, `Codes/Jeu_Apprentissage_Blaise.qmd`,
   `Codes/Robin/2.ML/`) : Random Forest retenu pour la soutenance, réglé par
   validation croisée groupée (leave-one-load-out), vote majoritaire par fichier.
2. **Deep learning sur signaux bruts fenêtrés** (`Codes/Robin/3.DL/AE_1D/`,
   `Codes/09_ae_transfert_DE_FE.ipynb`) : auto-encodeur convolutionnel Conv1D-AE / DAE
   entraîné sur les seules fenêtres saines ; score d'anomalie = MSE de reconstruction ;
   seuil = p99 des scores sains d'entraînement ; normalisation par fenêtre (z-score
   sans état). Voir aussi les protocoles de transfert DE↔FE.

## Bonnes pratiques retenues (anti-fuite)

- Split train/test **par fichier** (groupes) : les fenêtres se recouvrant à 50 %,
  un même fichier des deux côtés = fuite.
- Standardisation fit sur le **train uniquement**.
- Normalisation par fenêtre (z-score sans état) pour l'AE : identique à l'entraînement
  et à l'inférence, aucune statistique du test utilisée.
- Le seuil (p99 des scores sains d'entraînement) ne se transpose pas tel quel d'un
  capteur à l'autre : voir les protocoles de transfert DE↔FE.

## Données

La base CWRU est téléchargée automatiquement au premier appel de `Sources.data.charger()`
(dans `data/cwru_cache/`). Les tables d'indicateurs sont générées dans `data/features/`
(par ex. via `Codes/01_analyse_exploratoire.qmd` ou `features.sauvegarder_table`).