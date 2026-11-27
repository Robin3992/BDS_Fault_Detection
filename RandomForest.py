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

# ---------------------------------------------------------------------
# Imports standards
# ---------------------------------------------------------------------
import sys                      # pour modifier sys.path (accès au dossier Sources/)
from pathlib import Path        # chemins de fichiers indépendants de l'OS

# Imports data / ML
import pandas as pd             # tables de données (X, y)
import numpy as np               # calculs numériques (FPR/FNR)
from sklearn.ensemble import RandomForestClassifier                 # le modèle retenu
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold  # optimisation + CV
import joblib                   # sauvegarde / chargement du modèle (.joblib)

# --- Racine du dépôt (pour importer Sources/) ---
# On remonte l'arborescence depuis l'emplacement de CE fichier jusqu'au dossier
# contenant "BDS_Fault_Detection" : comme ça, le script marche qu'il soit
# placé à la racine du dépôt ou dans Codes/.
def trouver_racine(dossier=Path(__file__).resolve()):
    for parent in (dossier, *dossier.parents):
        if "BDS_Fault_Detection" in parent.name:
            return parent
    raise FileNotFoundError("Dossier 'BDS_Fault_Detection' introuvable.")

RACINE = trouver_racine()
sys.path.insert(0, str(RACINE))   # rend le dossier racine importable (Sources/...)

# Modules du projet (dossier Sources/ à la racine du dépôt)
from Sources import data, features, indicateurs, models

# Features utilisées : 4 temporels + spectres (amplitudes des raies caractéristiques)
# -> ce sont les colonnes de la table d'indicateurs fournies au modèle.
#    Ne pas modifier sans réentraîner le modèle (le .joblib est lié à cet ordre de features).
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
    # Sélection du plan de données CWRU : défauts IR/B/OR, diamètres 7/14/21,
    # position OR@6, charges 0-3 HP, sur les deux capteurs DE et FE.
    selection = data.selectionner(
        sources=("12kDE", "12kFE", "Normal"),
        defauts=("IR", "B", "OR"),
        diametres=(7, 14, 21),
        positions_or=("6",),
        charges=(0, 1, 2, 3),
        canaux={"12kDE": ("DE",), "12kFE": ("FE",), "Normal": ("DE", "FE")},
    )

    # Chargement des données + ajout des indicateurs par fenêtre (RMS_bande et Env_BPFI/BPFO/BSF/FTF)
    table = features.construire_table_SpecMourad(selection, data.charger)
    print(f"Table construite : {len(table)} fenêtres, {table['fichier'].nunique()} fichiers")

    # Split transverse DE -> FE + standardisation (fit sur le train uniquement)
    # -> le test (FE) est un capteur jamais vu à l'entraînement ; la standardisation
    #    n'utilise que les statistiques du train : pas de fuite.
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
# Chaque clé = un hyperparamètre du RandomForestClassifier, chaque valeur = les essais.
GRILLE_RF = {
    "n_estimators":      [200, 400, 800],     # nombre d'arbres de la forêt
    "max_depth":         [None, 8, 16, 32],    # profondeur max (None = illimitée)
    "min_samples_leaf":  [1, 2, 5, 10, 20],    # taille minimale d'une feuille
    "min_samples_split": [2, 5, 10],           # échantillons min pour couper un nœud
    "max_features":      ["sqrt", "log2", None],  # features testées à chaque split
    "class_weight":      [None, "balanced"],   # pondération des classes (déséquilibre Sain vs défauts)
}


def optimiser_rf(X_train, y_train, groupes=None, grille=None, cv=3, n_iter=30, scoring="f1_macro"):
    """Optimise les hyperparamètres du Random Forest par RandomizedSearchCV.

    - groupes : noms de fichiers du train -> CV groupée par fichier (pas de fuite entre plis)
    - n_iter  : nombre de combinaisons testées (30 par défaut)
    - refit=True : le meilleur modèle est réentraîné sur TOUT le train.
    Renvoie le meilleur modèle entraîné.
    """
    if grille is None:
        grille = GRILLE_RF                    # grille élargie ci-dessus si aucune fournie
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)   # modèle de base (sera cloné par la recherche)

    # CV groupée par fichier si on a les groupes -> les fenêtres d'un même fichier
    # ne sont jamais des deux côtés d'un pli (pas de fuite dans le réglage).
    if groupes is not None:
        cv_split = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        groupes = None                        # GridSearchCV sans groupes

    # Recherche aléatoire : n_iter combinaisons tirées dans la grille, évaluées en CV,
    # puis refit du meilleur modèle sur tout le train.
    recherche = RandomizedSearchCV(rf, grille, n_iter=n_iter, scoring=scoring,
                                   cv=cv_split, n_jobs=-1, refit=True,
                                   random_state=42, verbose=1)
    recherche.fit(X_train, y_train, groups=groupes)

    print(f"\n{n_iter} combinaisons testées sur {cv} plis ({scoring})")
    print("Meilleurs hyperparamètres :", recherche.best_params_)
    print(f"Meilleur score ({scoring}) en validation croisée : {recherche.best_score_:.3f}")
    return recherche.best_estimator_          # modèle entraîné avec les meilleurs hyperparamètres


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
    # evaluer_modele affiche accuracy / balanced accuracy / F1-macro / classification_report
    # et trace la matrice de confusion ; on lui passe les classes du modèle pour l'ordre.
    cm = models.evaluer_modele(y_test, y_pred, classes=list(rf.classes_), titre=titre)

    # Taux spécifiques au projet (classe "Sain" = référence)
    y_true = np.asarray(y_test); y_p = np.asarray(y_pred)
    sains = y_true == "Sain"        # masque des vraies fenêtres saines
    defauts = ~sains                # masque des vrais défauts
    # FPR = part des sains prédits défectueux (fausses alertes)
    fpr = (y_p[sains] != "Sain").mean() if sains.any() else float("nan")
    # FNR = part des défauts prédits sains (défauts manqués)
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
    #    (CV groupée par fichier grâce à g_train : pas de fuite entre plis)
    rf = optimiser_rf(X_train, y_train, groupes=g_train)

    # Sauvegarde optionnelle du modèle :
    # -> le .joblib contient le modèle entraîné AVEC les hyperparamètres optimisés
    #    (vérifiable après chargement avec rf.get_params())
    joblib.dump(rf, RACINE / "Sources" / "Fitted_models" / "modele_RF.joblib")

    # Chargement du modèle optimisé et entrainé
    # rf = joblib.load(RACINE / "Sources" / "Fitted_models" / "modele_RF.joblib")

    # 3. prédiction sur le test (capteur FE, jamais vu à l'entraînement)
    y_pred = predire(rf, X_test)
    print("\nPrédictions sur le test (FE) :", pd.Series(y_pred).value_counts().to_dict())

    evaluer(rf, X_test, y_test)    # matrice de confusion + métriques