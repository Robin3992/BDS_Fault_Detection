# -*- coding: utf-8 -*-
"""
code.py — Random Forest (détection de défauts de roulements, banc CWRU)
=======================================================================

Fonctionnalités :
    1. preparer_donnees()  : construit la table d'indicateurs et la découpe
                             train = capteur DE / test = capteur FE
    2. entrainer_rf()      : entraîne le Random Forest (hyperparamètres par défaut)
       optimiser_rf()      : optimise les hyperparamètres (RandomizedSearchCV, CV groupée par fichier)
    3. predire()           : prédit les classes d'un jeu de fenêtres

Exécution :  python code.py
(Le premier run télécharge les données CWRU dans data/cwru_cache/.)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
import joblib

# --- Racine du dépôt (pour importer Sources/) ---
def trouver_racine(dossier=Path(__file__).resolve()):
    for parent in (dossier, *dossier.parents):
        if "BDS_Fault_Detection" in parent.name:
            return parent
    raise FileNotFoundError("Dossier 'BDS_Fault_Detection' introuvable.")

RACINE = trouver_racine()
sys.path.insert(0, str(RACINE))

from Sources import data, features, indicateurs, models

# Features utilisées : 4 temporels + spectres (amplitudes raies caractéristiques : fondamentale + 2 harmoniques)
FEATURES = indicateurs.COLONNES_TEMPORELS + indicateurs.COLONNES_Mourad
CIBLE    = "defaut"     # "defaut" = 4 classes (Sain/IR/B/OR) ; "classe" = 10 classes


# =====================================================================
# 1. PRÉPARATION DES DONNÉES
# =====================================================================
def preparer_donnees():
    """Construit la table des indicateurs (fenêtres de 0,2 s) puis la découpe :
    train = capteur DE, test = capteur FE (split transverse, standardisé sur le train).
    Renvoie X_train, X_test, y_train, y_test, g_train (fichiers du train, pour la CV groupée).
    """
    selection = data.selectionner(
        sources=("12kDE", "12kFE", "Normal"),
        defauts=("IR", "B", "OR"),
        diametres=(7, 14, 21),
        positions_or=("6",),
        charges=(0, 1, 2, 3),
        canaux={"12kDE": ("DE",), "12kFE": ("FE",), "Normal": ("DE", "FE")},
    )

    # Chargement des données + ajout des indicateurs par fenêtre   
    table = features.construire_table_SpecMourad(selection, data.charger)   
    print(f"Table construite : {len(table)} fenêtres, {table['fichier'].nunique()} fichiers")

    # Split transverse DE -> FE + standardisation (fit sur le train uniquement)
    X_train, X_test, y_train, y_test, g_train, _ = features.preparer_modeling_DE_FE(
        table, FEATURES, canal_train="DE", canal_test="FE", target=CIBLE)

    # Équilibrage des classes UNIQUEMENT sur le train (le test reste complet)
    X_train, y_train, X_test, y_test = features.equilibrer(
        X_train, y_train, X_test, y_test, type="train")
    g_train = g_train.loc[X_train.index]     # groupes (fichiers) alignés sur le train équilibré

    print(f"Train : {len(X_train)} fenêtres | Test : {len(X_test)} fenêtres")
    return X_train, X_test, y_train, y_test, g_train


# =====================================================================
# 2. ENTRAÎNEMENT DU RANDOM FOREST
# =====================================================================

# Grille élargie pour l'optimisation (1080 combinaisons possibles)
GRILLE_RF = {
    "n_estimators":      [200, 400, 800],
    "max_depth":         [None, 8, 16, 32],
    "min_samples_leaf":  [1, 2, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2", None],
    "class_weight":      [None, "balanced"],
}


def optimiser_rf(X_train, y_train, groupes=None, grille=None, cv=3, n_iter=30, scoring="f1_macro"):
    """Optimise les hyperparamètres du Random Forest par RandomizedSearchCV.

    - groupes : noms de fichiers du train -> CV groupée par fichier (pas de fuite entre plis)
    - n_iter  : nombre de combinaisons testées (30 par défaut)
    - refit=True : le meilleur modèle est réentraîné sur TOUT le train.
    Renvoie le meilleur modèle entraîné.
    """
    if grille is None:
        grille = GRILLE_RF
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    if groupes is not None:
        cv_split = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        groupes = None

    recherche = RandomizedSearchCV(rf, grille, n_iter=n_iter, scoring=scoring,
                                   cv=cv_split, n_jobs=-1, refit=True,
                                   random_state=42, verbose=1)
    recherche.fit(X_train, y_train, groups=groupes)

    print(f"\n{n_iter} combinaisons testées sur {cv} plis ({scoring})")
    print("Meilleurs hyperparamètres :", recherche.best_params_)
    print(f"Meilleur score ({scoring}) en validation croisée : {recherche.best_score_:.3f}")
    return recherche.best_estimator_


# =====================================================================
# 3. PRÉDICTION
# =====================================================================
def predire(rf, X):
    """Prédit la classe de chaque fenêtre de X. Renvoie un array de classes.
    (rf.predict_proba(X) donne les probabilités par classe si besoin.)"""
    return rf.predict(X)


# =====================================================================
# 4. ÉVALUATION (matrice de confusion + métriques) — via Sources.models
# =====================================================================
def evaluer(rf, X_test, y_test, titre="Random Forest (train DE -> test FE)"):
    """Prédit X_test, puis affiche métriques + matrice de confusion.

    Utilise Sources.models.evaluer_modele : accuracy, balanced accuracy, F1-macro,
    rapport de classification par classe et matrice de confusion.
    Ajoute les deux taux du projet : fausses alertes (FPR) et défauts manqués (FNR).
    Renvoie la matrice de confusion.
    """
    y_pred = predire(rf, X_test)
    cm = models.evaluer_modele(y_test, y_pred, classes=list(rf.classes_), titre=titre)

    # Taux spécifiques au projet (classe "Sain" = référence)
    y_true = np.asarray(y_test); y_p = np.asarray(y_pred)
    sains = y_true == "Sain"
    defauts = ~sains
    fpr = (y_p[sains] != "Sain").mean() if sains.any() else float("nan")
    fnr = (y_p[defauts] == "Sain").mean() if defauts.any() else float("nan")
    print(f"Fausses alertes (FPR)   : {fpr:.2%}")
    print(f"Défauts manqués (FNR)   : {fnr:.2%}")
    return cm



# =====================================================================
# Exemple d'utilisation
# =====================================================================
if __name__ == "__main__":
    # 1. préparation (train = DE, test = FE)
    X_train, X_test, y_train, y_test, g_train = preparer_donnees()

    # 2. Entrainement avec optimisation des hyperparamètres via RandomizedSearchCV
    rf = optimiser_rf(X_train, y_train, groupes=g_train)

    # Sauvegarde optionnelle du modèle :
    joblib.dump(rf, RACINE / "Sources" / "Fitted_models" / "modele_RF.joblib")

    # Chargement du modèle optimisé et entrainé
    # rf = joblib.load(RACINE / "Sources" / "Fitted_models" / "modele_RF.joblib")

    # 3. prédiction sur le test (capteur FE, jamais vu à l'entraînement)
    y_pred = predire(rf, X_test)
    print("\nPrédictions sur le test (FE) :", pd.Series(y_pred).value_counts().to_dict())

    evaluer(rf, X_test, y_test)    # matrice de confusion + métriques