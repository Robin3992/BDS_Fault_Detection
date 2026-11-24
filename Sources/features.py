"""
Fonctions relatives à la création de la table de données
Utilise notamment les fonctions selectionner() et charger() pour l'import de données
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


from .config import DUREE_FENETRE, FEATURES
from .indicateurs import indicateurs_spectraux, indicateurs_temporels, indicateurs_spectraux_Mourad
from .signaux import creer_filtre, M_up_enveloppe_bande


# def fenetrer(x, fs, duree=DUREE_FENETRE):
#     """Découpe un signal en fenêtres sans recouvrement.
#     Renvoie une liste de fenêtres (tableaux numpy)."""
#     taille = int(duree * fs)
#     return [x[i * taille:(i + 1) * taille] for i in range(len(x) // taille)]


def fenetrer(x, fs, duree=DUREE_FENETRE, recouvrement=0.0):
    """
    Découpe un signal en fenêtres avec x secondes (duree), avec un recouvrement paramétrable : sans recouvrement par défaut.
    """
    taille = round(duree * fs)
    pas = int(taille * (1 - recouvrement))
    return [x[i:i + taille] for i in range(0, len(x) - taille + 1, pas)]


def construire_table(selection, charger, canaux=None):
    """Construit la table des indicateurs : une ligne par (fichier, fenêtre, canal).

    canaux : None = lu depuis la colonne "canaux" de `selection`, LIGNE PAR LIGNE
             (remplie par data.selectionner2, éventuellement via un dict
             {source: canaux}) ; sinon tuple/str de canaux identiques pour tous.
    """
    lignes = []
    sos_cache = {}
    for nom, info in selection.iterrows():
        liste = (info["canaux"] if "canaux" in selection.columns else None) if canaux is None else canaux
        if liste is None:
            liste = ("DE",)
        if isinstance(liste, str):
            liste = (liste,)
        s = charger(nom, canaux=liste)          # ← ne charge que les canaux utiles
        fs, rpm = s["fs"], s["rpm"]
        if fs not in sos_cache:
            sos_cache[fs] = creer_filtre(fs)
        sos = sos_cache[fs]
        for canal in liste:                     # ← canaux de CE fichier
            roulement = "FE" if "FE" in nom else "DE"   # selon le FICHIER, pas le canal
            x = s[canal]
            for i, w in enumerate(fenetrer(x, fs)):
                res = {"fichier": nom, "fenetre": i, "rpm": rpm, "canal": canal,
                       **{k: v for k, v in info.to_dict().items() if k != "canaux"}}
                res.update(indicateurs_temporels(w))
                res.update(indicateurs_spectraux(w, fs, rpm, roulement=roulement, sos=sos))
                lignes.append(res)
    table = pd.DataFrame(lignes)
    table["classe"] = np.where(table["defaut"] == "Sain", "Sain",
                               table["defaut"] + table["diametre_mils"].astype(str).str.zfill(3))
    return table



def construire_table_SpecMourad(selection, charger, canaux=None):
    """Construit la table des indicateurs : une ligne par (fichier, fenêtre, canal).

    canaux : None = lu depuis la colonne "canaux" de `selection`, LIGNE PAR LIGNE
             (remplie par data.selectionner2, éventuellement via un dict
             {source: canaux}) ; sinon tuple/str de canaux identiques pour tous.
    """
    lignes = []
    sos_cache = {}
    for nom, info in selection.iterrows():
        liste = (info["canaux"] if "canaux" in selection.columns else None) if canaux is None else canaux
        if liste is None:
            liste = ("DE",)
        if isinstance(liste, str):
            liste = (liste,)
        s = charger(nom, canaux=liste)          # ← ne charge que les canaux utiles
        fs, rpm = s["fs"], s["rpm"]
        if fs not in sos_cache:
            sos_cache[fs] = creer_filtre(fs)
        sos = sos_cache[fs]
        for canal in liste:                     # ← canaux de CE fichier
            roulement = "FE" if "FE" in nom else "DE"   # selon le FICHIER, pas le canal
            x = s[canal]
            x_filt, env = M_up_enveloppe_bande(x, fs, sos=sos)   # une fois par fichier/canal
            for i, w in enumerate(fenetrer(x, fs)):
                res = {"fichier": nom, "fenetre": i, "rpm": rpm, "canal": canal,
                       **{k: v for k, v in info.to_dict().items() if k != "canaux"}}
                res.update(indicateurs_temporels(w))
                res.update(indicateurs_spectraux(w, fs, rpm, roulement=roulement, sos=sos))
                sl = slice(i * len(w), (i + 1) * len(w))
                res.update(indicateurs_spectraux_Mourad(x_filt[sl], env[sl], fs, rpm,
                                                         roulement=roulement))
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


def preparer_modeling(table, features, test_size=0.25, seed=42, strat=True, target="classe"):
    """Sépare par FICHIER et standardise sur le train uniquement.

    strat=True  (défaut) : StratifiedGroupKFold — chaque classe garantie dans le test. Recommandé.
    strat=False          : GroupShuffleSplit — split aléatoire simple par fichier.
    """
    X, y, groupes = table[features], table[target], table["fichier"]

    if strat:
        n_splits = round(1 / test_size)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    i_train, i_test = next(cv.split(X, y, groupes))

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X.iloc[i_train]),
                           columns=features, index=X.index[i_train])
    X_test  = pd.DataFrame(scaler.transform(X.iloc[i_test]),
                           columns=features, index=X.index[i_test])
    return (X_train, X_test,
            y.iloc[i_train], y.iloc[i_test],
            groupes.iloc[i_train], groupes.iloc[i_test])


def preparer_modeling_DE_FE(table, features, canal_train="DE", canal_test="FE", target="classe"):
    """Split TRANSVERSE par canal : 
    -> train sur `canal_train` (DE)
    -> test sur `canal_test` (FE)

    target : colonne utilisée comme étiquette de classe
             ("classe" = 10 classes IR007/..., "defaut" = 4 classes Sain/IR/B/OR)
    ..."""
    assert "canal" in table.columns, "La table doit contenir une colonne 'canal'"
    assert target in table.columns, f"Colonne cible {target!r} absente de la table"
    est_train = (table["canal"] == canal_train) & (table["source"] != f"12k{canal_test}")
    est_test  = (table["canal"] == canal_test)  & (table["source"] != f"12k{canal_train}")
    assert est_train.any() and est_test.any(), f"Canaux {canal_train!r}/{canal_test!r} absents de la table"

    X_tr_raw = table.loc[est_train, features]
    X_te_raw = table.loc[est_test,  features]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_tr_raw),
                           columns=features, index=X_tr_raw.index)
    X_test  = pd.DataFrame(scaler.transform(X_te_raw),
                           columns=features, index=X_te_raw.index)

    return (X_train, X_test,
            table.loc[est_train, target], table.loc[est_test, target],
            table.loc[est_train, "fichier"], table.loc[est_test, "fichier"])



def equilibrer(X_train, y_train, X_test, y_test, type="train", n=None, seed=42):
    """
    Équilibre les classes par sous-échantillonnage (tirage aléatoire reproductible).

    type='min'       : train et test équilibrés chacun sur leur propre classe minoritaire
    type='train'     : seul le train est équilibré (sur sa classe minoritaire, ou n) — test intact
    type='train2test': comme 'train', et le test est plafonné au même n
    n                : plafond de fenêtres par classe (None = min de la classe concernée)
    """
    def sous_echantillonner(X, y, n):
        idx = []
        for c in y.unique():
            idx += y[y == c].sample(n=min(n, (y == c).sum()),
                                     random_state=seed).index.tolist()
        return X.loc[idx], y.loc[idx]
    
    if type == "nope":
        return X_train, y_train, X_test, y_test
    if type == "min":
        n_tr = n or y_train.value_counts().min()
        n_te = n or y_test.value_counts().min()
        return (*sous_echantillonner(X_train, y_train, n_tr),
                *sous_echantillonner(X_test, y_test, n_te))
    if type == "train":
        n = n or y_train.value_counts().min()
        return (*sous_echantillonner(X_train, y_train, n), X_test, y_test)
    if type == "train2test":
        n = n or y_train.value_counts().min()
        return (*sous_echantillonner(X_train, y_train, n),
                *sous_echantillonner(X_test, y_test, n))
    raise ValueError(f"type inconnu : {type!r} (attendu : 'min', 'train', 'train2test')")