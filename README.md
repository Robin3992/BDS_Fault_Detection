# BDS_Fault_Detection — Détection de défauts de roulements (banc CWRU)

Projet de détection et classification des défauts de roulements à partir de signaux
vibratoires issus du banc d'essai **CWRU** (Case Western Reserve University) :
détection sain/défectueux, type de défaut (IR = bague interne, B = billes,
OR = bague externe), sévérité (diamètre 7/14/21 mils), sur deux capteurs :
**DE** (Drive End) et **FE** (Fan End), échantillonnés à 12 kHz.

## Contenu du dépôt

| Dossier / fichier | Contenu |
|---|---|
| `RandomForest.py` | **Script final** — pipeline Random Forest complet : préparation → optimisation → prédiction → évaluation |
| `Sources/` | Modules du projet : `data.py` (catalogue + chargement CWRU), `features.py` (fenêtrage, tables d'indicateurs, splits, équilibrage), `indicateurs.py` (indicateurs temporels/spectraux + listes de colonnes), `signaux.py` (filtrage passe-bande, enveloppe, spectre d'enveloppe), `models.py` (métriques, matrice de confusion, Conv1D-AE, classifieurs ML), `config.py`, `path.py` |
| `Sources/Fitted_models/` | Modèles sauvegardés : auto-encodeurs DL (`DAE_fitted.keras`, ...) et `modele_RF.joblib` (Random Forest entraîné avec ses hyperparamètres optimisés) |
| `Codes/` | Notebooks et rapports Quarto d'analyse et de modélisation (`01_analyse_exploratoire.qmd`, `07_cnn1d.ipynb`, `09_ae_transfert_DE_FE.ipynb`, `Jeu_Apprentissage_Blaise.qmd`, ...) |
| `data/` | Données (gitignoré) : cache CWRU dans `data/cwru_cache/`, tables d'indicateurs dans `data/features/` |
| `Docs/`, `images/` | Documents du projet et figures |
| `requirements.txt` | Dépendances (versions épinglées, inclut scikit-learn, tensorflow, keras, joblib...) |

## Prérequis

```bash
pip install -r requirements.txt
```

## Démarrage rapide — Random Forest (`RandomForest.py`)

`RandomForest.py` (à la racine du dépôt) est le script final, autonome et commenté.
Il fonctionne aussi bien lancé depuis la racine que depuis `Codes/` (la racine du dépôt
est retrouvée automatiquement).

```bash
python RandomForest.py
```

Le premier run télécharge les données CWRU dans `data/cwru_cache/` puis calcule les
indicateurs par fenêtre (quelques minutes).

### Pipeline (4 étapes dans `__main__`)

1. **`preparer_donnees()`** — préparation des données :
   - `data.selectionner(...)` : plan de données CWRU — sources `12kDE` / `12kFE` / `Normal`,
     défauts IR/B/OR, diamètres 7/14/21, position OR@6, charges 0–3 HP, avec
     `canaux={"12kDE": ("DE",), "12kFE": ("FE",), "Normal": ("DE", "FE")}` ;
   - `features.construire_table_SpecMourad(selection, data.charger)` : une ligne par
     (fichier, fenêtre de 0,2 s = 2400 points à 12 kHz, canal) + indicateurs
     (4 temporels + 17 spectraux + 5 « Mourad ») ;
   - `features.preparer_modeling_DE_FE(..., canal_train="DE", canal_test="FE", target="defaut")` :
     **split transverse** train = capteur DE / test = capteur FE, `StandardScaler` fit sur le
     **train seul** (aucune fuite) ;
   - `features.equilibrer(..., type="train")` : équilibrage des classes **uniquement sur le train**
     (le test reste complet).
   - Renvoie `X_train, X_test, y_train, y_test, g_train` (fichiers du train, pour la CV groupée).
   - Features du modèle : `FEATURES = indicateurs.COLONNES_TEMPORELS + indicateurs.COLONNES_Mourad`
     (9 colonnes), cible `CIBLE = "defaut"` (4 classes : Sain / IR / B / OR).
2. **`optimiser_rf(X_train, y_train, groupes=g_train)`** — optimisation des hyperparamètres par
   `RandomizedSearchCV` (grille `GRILLE_RF` : 1080 combinaisons, `n_iter=30` par défaut,
   `scoring="f1_macro"`, `refit=True`) avec **CV groupée par fichier** (`StratifiedGroupKFold`) :
   les fenêtres d'un même fichier ne sont jamais des deux côtés d'un pli → pas de fuite.
   Renvoie le meilleur modèle, réentraîné sur tout le train.
3. **Sauvegarde** : `joblib.dump(rf, RACINE / "Sources" / "Fitted_models" / "modele_RF.joblib")`
   — le `.joblib` contient le modèle entraîné **avec les hyperparamètres optimisés**
   (vérifiable après chargement avec `rf.get_params()`) ; rechargement :
   `rf = joblib.load(RACINE / "Sources" / "Fitted_models" / "modele_RF.joblib")`.
4. **`predire(rf, X_test)`** puis **`evaluer(rf, X_test, y_test)`** — prédictions sur le test
   (capteur FE, jamais vu à l'entraînement), puis matrice de confusion + métriques
   (`models.evaluer_modele` : accuracy, balanced accuracy, F1-macro, rapport par classe)
   + les deux taux du projet : **FPR** (fausses alertes) et **FNR** (défauts manqués).

### Utilisation étape par étape

```python
from RandomForest import preparer_donnees, optimiser_rf, predire, evaluer

X_train, X_test, y_train, y_test, g_train = preparer_donnees()
rf = optimiser_rf(X_train, y_train, groupes=g_train)   # ou RandomForestClassifier(...) directement
y_pred = predire(rf, X_test)                           # rf.predict_proba(X_test) pour les probabilités
evaluer(rf, X_test, y_test)                            # matrice de confusion + FPR / FNR
```

⚠️ Recharger un `.joblib` avec la **même version de scikit-learn**, et prédire avec les
**mêmes features dans le même ordre** (`FEATURES`, défini en tête de `RandomForest.py`).

## Deux approches dans le projet

1. **ML sur indicateurs** (`RandomForest.py`, `Codes/Jeu_Apprentissage_Blaise.qmd`,
   `Codes/Robin/2.ML/`) : Random Forest retenu pour la soutenance — features calculées par
   fenêtre, validation croisée groupée par fichier (leave-one-load-out, protocoles P2/P3/P4).
2. **Deep learning sur signaux bruts fenêtrés** (`Codes/Robin/3.DL/AE_1D/`,
   `Codes/09_ae_transfert_DE_FE.ipynb`, `Codes/10_dae_transfert_DE_FE.ipynb`) :
   auto-encodeur convolutionnel Conv1D-AE / DAE entraîné sur les seules fenêtres saines ;
   score d'anomalie = MSE de reconstruction ; seuil = p99 des scores sains d'entraînement ;
   normalisation **par fenêtre** (`features.standardiser_par_fenetre` : z-score sans état,
   équivalent à `StandardScaler().fit_transform(X.T).T`, utilisable telle quelle à
   l'inférence sur un nouveau fichier). Voir les protocoles de transfert DE↔FE.

## Bonnes pratiques retenues (anti-fuite)

- Split train/test **par fichier** (`preparer_modeling`, groupes = fichier) ou **par canal** (transverse DE→FE, `preparer_modeling_DE_FE`) : aucune ligne (fichier, canal) des deux côtés, et dans le split transverse **aucun canal en commun** entre train et test (les fichiers `Normal_*` contribuent aux deux côtés, mais via des canaux différents : DE au train, FE au test).
- Standardisation (`StandardScaler`) fit sur le **train uniquement** ; pour l'AE, z-score
  **par fenêtre** sans état → aucune statistique du test utilisée, rien à sauvegarder.
- CV de réglage des hyperparamètres **groupée par fichier** (`StratifiedGroupKFold` avec `g_train`).
- Équilibrage des classes uniquement sur le train ; le test reste complet (évaluation réaliste).
- Le seuil p99 des scores sains d'entraînement ne se transpose pas tel quel d'un capteur à
  l'autre : voir les protocoles de transfert DE↔FE (FPR asymétrique observé : train DE → test FE
  plus pénalisant que l'inverse).

## Données

La base CWRU est téléchargée automatiquement au premier appel de `Sources.data.charger()`
(dans `data/cwru_cache/`, gitignore). Les tables d'indicateurs sont générées dans
`data/features/` (gitignore), par ex. via `features.sauvegarder_table(table)` ou
`Codes/01_analyse_exploratoire.qmd`.