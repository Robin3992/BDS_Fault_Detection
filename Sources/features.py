"""Construction de la table des indicateurs et préparation de la modélisation."""
import numpy as np
import pandas as pd

from .config import DUREE_FENETRE, FEATURES
from .indicateurs import indicateurs_spectraux, indicateurs_temporels


def fenetrer(x, fs, duree=DUREE_FENETRE):
    """Découpe un signal en fenêtres sans recouvrement.
    Renvoie une liste de fenêtres (tableaux numpy)."""
    taille = int(duree * fs)
    return [x[i * taille:(i + 1) * taille] for i in range(len(x) // taille)]


def construire_table(selection, charger, canal="DE"):
    """Construit la table des indicateurs : une ligne par fenêtre de 0,2 s.

    selection : DataFrame du catalogue (ex. data.selectionner()) indexé par nom.
    charger   : fonction data.charger (injectée pour éviter le couplage).

    Colonnes : identification (fichier, fenetre, rpm), conditions d'essai
    (recopiées du catalogue), étiquette `classe`, 4 indicateurs temporels
    et 17 indicateurs spectraux.
    """
    lignes = []
    for nom, info in selection.iterrows():
        s = charger(nom)
        x, fs, rpm = s[canal], s["fs"], s["rpm"]
        for i, w in enumerate(fenetrer(x, fs)):
            res = {"fichier": nom, "fenetre": i, "rpm": rpm, **info.to_dict()}
            res.update(indicateurs_temporels(w))
            res.update(indicateurs_spectraux(w, fs, rpm))
            lignes.append(res)
    table = pd.DataFrame(lignes)
    table["classe"] = np.where(table["defaut"] == "Sain", "Sain",
                               table["defaut"]
                               + table["diametre_mils"].astype(str).str.zfill(3))
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