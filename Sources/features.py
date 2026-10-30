"""Construction de la table des indicateurs et préparation de la modélisation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from .config import DUREE_FENETRE, FEATURES
from .indicateurs import indicateurs_spectraux, indicateurs_temporels


def fenetrer(x, fs, duree=DUREE_FENETRE):
    """Découpe un signal en fenêtres sans recouvrement.
    Renvoie une liste de fenêtres (tableaux numpy)."""
    taille = int(duree * fs)
    return [x[i * taille:(i + 1) * taille] for i in range(len(x) // taille)]


def construire_table(selection, charger, canaux=("DE",)):
    """Construit la table des indicateurs : une ligne par (fichier, fenêtre, canal).

    canaux : ("DE",) par défaut ; ("DE", "FE") pour l'analyse transverse.
    La colonne `canal` identifie le capteur, et `roulement` est déduit du canal
    pour les fréquences caractéristiques (DE→6205, FE→6203).
    """
    lignes = []
    for nom, info in selection.iterrows():
        s = charger(nom)
        fs, rpm = s["fs"], s["rpm"]
        for canal in canaux:
            roulement = canal if canal in ("DE", "FE") else "DE"
            x = s[canal]
            for i, w in enumerate(fenetrer(x, fs)):
                res = {"fichier": nom, "fenetre": i, "rpm": rpm, "canal": canal,
                       **info.to_dict()}
                res.update(indicateurs_temporels(w))
                res.update(indicateurs_spectraux(w, fs, rpm, roulement=roulement))
                lignes.append(res)
    table = pd.DataFrame(lignes)
    table["classe"] = np.where(table["defaut"] == "Sain", "Sain",
                               table["defaut"] + table["diametre_mils"].astype(str).str.zfill(3))
    return table


def sauvegarder_table(table, nom_fichier="table_indicateurs.csv"):
    """Écrit la table dans data/features/ (dossier créé si besoin)."""
    FEATURES.mkdir(parents=True, exist_ok=True)
    chemin = FEATURES / nom_fichier
    table.to_csv(chemin, index=False)
    return chemin


def preparer_modeling(table, features, test_size=0.25, seed=42):
    """Sépare les données par FICHIER (jamais par fenêtre : > 90 % de la
    variabilité est inter-fichiers) et standardise (moyenne/écart-type
    calculés sur le train uniquement, équivalent manuel de
    GroupShuffleSplit + StandardScaler, sans dépendance sklearn).

    Renvoie (X_train, X_test, y_train, y_test, g_train, g_test).
    """
    X, y, groupes = table[features], table["classe"], table["fichier"]
    fichiers = np.sort(groupes.unique())
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_size * len(fichiers))))
    fichiers_test = rng.choice(fichiers, size=n_test, replace=False)
    est_test = groupes.isin(fichiers_test)
    moy = X[~est_test].mean()
    std = X[~est_test].std().replace(0, 1)      # évite la division par zéro
    return ((X[~est_test] - moy) / std, (X[est_test] - moy) / std,
            y[~est_test], y[est_test],
            groupes[~est_test], groupes[est_test])




def preparer_modeling_DE_FE(table, features, test_size=0.25, seed=42):
    X, y, groupes = table[features], table["classe"], table["fichier"]

    n_splits = round(1 / test_size)          # test_size=0.25 -> 4 splits
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    i_train, i_test = next(cv.split(X, y, groupes))
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X.iloc[i_train]),
                           columns=features, index=X.index[i_train])
    X_test = pd.DataFrame(scaler.transform(X.iloc[i_test]),
                          columns=features, index=X.index[i_test])
    return (X_train, X_test,
        y.iloc[i_train], y.iloc[i_test],
        groupes.iloc[i_train], groupes.iloc[i_test])








def preparer_modeling_strat(table, features, test_size=0.25, seed=42):

    """Sépare les données par FICHIER, de façon STRATIFIÉE : chaque classe
    est représentée dans le test (évite qu'une classe minoritaire soit
    absente du test par hasard). Standardise sur le train uniquement.
 
     Renvoie (X_train, X_test, y_train, y_test, g_train, g_test).
     """
    X, y, groupes = table[features], table["classe"], table["fichier"]

    n_splits = round(1 / test_size)          # test_size=0.25 -> 4 splits
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=seed)
    i_train, i_test = next(cv.split(X, y, groupes))
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X.iloc[i_train]),
                           columns=features, index=X.index[i_train])
    X_test = pd.DataFrame(scaler.transform(X.iloc[i_test]),
                          columns=features, index=X.index[i_test])
    return (X_train, X_test,
        y.iloc[i_train], y.iloc[i_test],
        groupes.iloc[i_train], groupes.iloc[i_test])