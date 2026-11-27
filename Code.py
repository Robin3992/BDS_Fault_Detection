"""
code.py — Random Forest (détection de défauts de roulements, banc CWRU)
=======================================================================
Version simplifiée, fidèle aux notebooks de Codes/Robin/2.ML (1.ML_1.ipynb).

3 fonctionnalités :
    1. preparer_donnees()  : construit la table d'indicateurs et la découpe
                             train = capteur DE / test = capteur FE
    2. entrainer_rf()      : entraîne le Random Forest sur le train
    3. predire()           : prédit les classes d'un jeu de fenêtres

Exécution :  python code.py
(Le premier run télécharge les données CWRU dans data/cwru_cache/.)
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# --- Racine du dépôt (pour importer Sources/) ---
def trouver_racine(dossier=Path(__file__).resolve()):
    for parent in (dossier, *dossier.parents):
        if "BDS_Fault_Detection" in parent.name:
            return parent
    raise FileNotFoundError("Dossier 'BDS_Fault_Detection' introuvable.")

RACINE = trouver_racine()
sys.path.insert(0, str(RACINE))

from Sources import data, features, indicateurs

# Features utilisées (comme dans le notebook 1.ML_1) : 4 temporels + spectres (amp_raies)
FEATURES = indicateurs.COLONNES_TEMPORELS + indicateurs.COLONNES_SPECTRAUX
CIBLE    = "defaut"     # "defaut" = 4 classes (Sain/IR/B/OR) ; "classe" = 10 classes


# =====================================================================
# 1. PRÉPARATION DES DONNÉES
# =====================================================================
def preparer_donnees():
    """Construit la table des indicateurs (fenêtres de 0,2 s) puis la découpe :
    train = capteur DE, test = capteur FE (split transverse, standardisé sur le train).
    Renvoie X_train, X_test, y_train, y_test.
    """
    selection = data.selectionner(
        sources=("12kDE", "12kFE", "Normal"),
        defauts=("IR", "B", "OR"),
        diametres=(7, 14, 21),
        positions_or=("6",),
        charges=(0, 1, 2, 3),
        canaux={"12kDE": ("DE",), "12kFE": ("FE",), "Normal": ("DE", "FE")},
    )

    table = features.construire_table(selection, data.charger)   # indicateurs par fenêtre
    print(f"Table construite : {len(table)} fenêtres, {table['fichier'].nunique()} fichiers")

    # Split transverse DE -> FE + standardisation (fit sur le train uniquement)
    X_train, X_test, y_train, y_test, _, _ = features.preparer_modeling_DE_FE(
        table, FEATURES, canal_train="DE", canal_test="FE", target=CIBLE)

    # Équilibrage des classes UNIQUEMENT sur le train (le test reste complet)
    X_train, y_train, X_test, y_test = features.equilibrer(
        X_train, y_train, X_test, y_test, type="train")

    print(f"Train : {len(X_train)} fenêtres | Test : {len(X_test)} fenêtres")
    return X_train, X_test, y_train, y_test


# =====================================================================
# 2. ENTRAÎNEMENT DU RANDOM FOREST
# =====================================================================
def entrainer_rf(X_train, y_train):
    """Entraîne le Random Forest (200 arbres, classes équilibrées) et renvoie le modèle."""
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Random Forest entraîné sur {len(X_train)} fenêtres "
          f"({len(rf.classes_)} classes : {list(rf.classes_)})")
    return rf


# =====================================================================
# 3. PRÉDICTION
# =====================================================================
def predire(rf, X):
    """Prédit la classe de chaque fenêtre de X. Renvoie un array de classes.
    (rf.predict_proba(X) donne les probabilités par classe si besoin.)"""
    return rf.predict(X)


# =====================================================================
# Exemple d'utilisation
# =====================================================================
if __name__ == "__main__":
    # 1. préparation
    X_train, X_test, y_train, y_test = preparer_donnees()

    # 2. entraînement
    rf = entrainer_rf(X_train, y_train)

    # 3. prédiction sur le test (capteur FE, jamais vu à l'entraînement)
    y_pred = predire(rf, X_test)
    print("\nPrédictions sur le test (FE) :", pd.Series(y_pred).value_counts().to_dict())

    # Sauvegarde optionnelle du modèle :
    # import joblib
    # joblib.dump(rf, RACINE / "data" / "modele_RF.joblib")
    # Pour recharger : rf = joblib.load(RACINE / "data" / "modele_RF.joblib")