"""
=== Chargement des données CWRU ===
https://engineering.case.edu/bearingdatacenter/download-data-file

Catalogue de données extrait du site CWRU (161 fichiers, hardcoded)
Fonctions permettant de charger les données souhaitées :
- selectionner()    : permet de sélectionner plusieurs fichiers du catalogue par critères
- charger()         : permet de charger les données sélectionnées
"""

import urllib.request

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import CACHE, URL_CWRU, RPM_NOMINAL

# ------------------------------------------------------------------
# Numéros de fichiers du site, page par page
#     clé = (défaut, diamètre en millièmes de pouce)
#     valeur = numéros pour les charges 0, 1, 2, 3 HP (None = absent)
# ------------------------------------------------------------------
NORMAL = [97, 98, 99, 100]

PAGES = {
    "12kDE": (12000, {
        ("IR", 7): [105, 106, 107, 108],      ("B", 7): [118, 119, 120, 121],
        ("OR@6", 7): [130, 131, 132, 133],    ("OR@3", 7): [144, 145, 146, 147],
        ("OR@12", 7): [156, 158, 159, 160],
        ("IR", 14): [169, 170, 171, 172],     ("B", 14): [185, 186, 187, 188],
        ("OR@6", 14): [197, 198, 199, 200],
        ("IR", 21): [209, 210, 211, 212],     ("B", 21): [222, 223, 224, 225],
        ("OR@6", 21): [234, 235, 236, 237],   ("OR@3", 21): [246, 247, 248, 249],
        ("OR@12", 21): [258, 259, 260, 261],
        ("IR", 28): [3001, 3002, 3003, 3004], ("B", 28): [3005, 3006, 3007, 3008],
    }),
    "48kDE": (48000, {
        ("IR", 7): [109, 110, 111, 112],      ("B", 7): [122, 123, 124, 125],
        ("OR@6", 7): [135, 136, 137, 138],    ("OR@3", 7): [148, 149, 150, 151],
        ("OR@12", 7): [161, 162, 163, 164],
        ("IR", 14): [174, 175, 176, 177],     ("B", 14): [189, 190, 191, 192],
        ("OR@6", 14): [201, 202, 203, 204],
        ("IR", 21): [213, 214, 215, 217],     ("B", 21): [226, 227, 228, 229],
        ("OR@6", 21): [238, 239, 240, 241],   ("OR@3", 21): [250, 251, 252, 253],
        ("OR@12", 21): [262, 263, 264, 265],
    }),
    "12kFE": (12000, {
        ("IR", 7): [278, 279, 280, 281],      ("B", 7): [282, 283, 284, 285],
        ("OR@6", 7): [294, 295, 296, 297],    ("OR@3", 7): [298, 299, 300, 301],
        ("OR@12", 7): [302, 305, 306, 307],
        ("IR", 14): [274, 275, 276, 277],     ("B", 14): [286, 287, 288, 289],
        ("OR@6", 14): [313, None, None, None], ("OR@3", 14): [310, 309, 311, 312],
        ("IR", 21): [270, 271, 272, 273],     ("B", 21): [290, 291, 292, 293],
        ("OR@6", 21): [315, None, None, None], ("OR@3", 21): [None, 316, 317, 318],
    }),
}


def construire_catalogue():
    """Construit le catalogue : une ligne par fichier, avec un nom lisible.
    ex. 'Normal_L0', '12kDE_IR007_L0', '12kDE_OR021at3_L2', '48kDE_B014_L1'
    """
    lignes = []
    for charge, num in enumerate(NORMAL):
        lignes.append({"nom": f"Normal_L{charge}", "numero": num, "source": "Normal",
                       "fs": 12000, "defaut": "Sain", "position_OR": None,
                       "diametre_mils": 0, "charge_hp": charge})
    for source, (fs, table) in PAGES.items():
        for (defaut, diametre), numeros in table.items():
            localisation, _, position = defaut.partition("@")
            for charge, num in enumerate(numeros):
                if num is None:
                    continue
                nom = (f"{source}_{localisation}{diametre:03d}"
                       + (f"at{position}" if position else "")
                       + f"_L{charge}")
                lignes.append({"nom": nom, "numero": num, "source": source, "fs": fs,
                               "defaut": localisation, "position_OR": position or None,
                               "diametre_mils": diametre, "charge_hp": charge})
    catalogue = pd.DataFrame(lignes).set_index("nom")
    catalogue["rpm_nominal"] = catalogue["charge_hp"].map(RPM_NOMINAL)
    return catalogue


catalogue = construire_catalogue()

def charger(nom, canaux=("DE", "FE", "BA")):
    """Renvoie {canal: signal, 'fs', 'rpm'} pour un fichier du catalogue
    (ex. '12kDE_IR007_L0'). Télécharge le fichier la 1re fois, puis le
    relit en local (cache).

    canaux : canal(x) à extraire du .mat — seuls ceux-là sont gardés
            en mémoire (ex. charger(nom, canaux=("DE",)) pour le DE seul).
    """
    # Infos du fichier dans le catalogue (numéro, fs, charge, diamètre...)
    info = catalogue.loc[nom]
    numero = int(info["numero"])
    # Chemin du .mat en cache local (data/cwru_cache/)
    chemin = CACHE / f"{nom}__{numero}.mat"     # ex. 12kDE_IR007_L0__105.mat

    # 1er appel : on télécharge le .mat depuis le site CWRU ; les appels suivants relisent le cache
    if not chemin.exists():                     # 1re fois : téléchargement
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL_CWRU + f"{numero}.mat") as reponse:
            chemin.write_bytes(reponse.read())
    # Contenu du .mat (variables X<numéro>_DE_time, X<numéro>_FE_time, X<numéro>_RPM...)
    mat = loadmat(chemin)

    # Retrouve la clé .mat se terminant par ce suffixe (ex. "_DE_time"),
    # en privilégiant la variable du bon numéro de fichier (X105_...)
    def choisir(suffixe):                       # privilégie la variable du bon numéro
        cles = [k for k in mat if k.endswith(suffixe)]
        bonne = [k for k in cles if k.startswith(f"X{numero:03d}")]
        return (bonne or cles or [None])[0]

    # Accepte aussi un canal unique sous forme de str, ex. charger(nom, "DE")
    if isinstance(canaux, str):
        canaux = (canaux,)
    # Dictionnaire résultat : "fs" + un signal par canal demandé + "rpm"
    signaux = {"fs": int(info["fs"])}
    # Boucle sur les SEULS canaux demandés (et non les 3 systématiquement)
    for canal in canaux:                        # ← seulement les canaux demandés
        cle = choisir(f"_{canal}_time")
        # Signal 1D du canal (None si le canal est absent du fichier .mat)
        signaux[canal] = None if cle is None else np.ravel(mat[cle])
    # Régime mesuré dans le fichier (ex. X105_RPM)
    cle_rpm = choisir("RPM")
    # Si pas de RPM dans le .mat : fallback sur le régime nominal du catalogue
    signaux["rpm"] = (float(np.squeeze(mat[cle_rpm])) if cle_rpm
                      else float(info["rpm_nominal"]))
    # Renvoie ex. {"DE": array(...), "FE": array(...), "fs": 12000, "rpm": 1797.0}
    return signaux


def _en_liste(valeur):
    """Normalise un critère de sélection en liste (None -> pas de filtre)."""
    if valeur is None:
        return None
    if isinstance(valeur, (list, tuple, set)):
        return list(valeur)
    return [valeur]


def selectionner(sources=("12kDE", "Normal"), defauts=("IR", "B", "OR"),
                 diametres=(7, 14, 21), positions_or=("6",),
                 charges=(0, 1, 2, 3), canaux=("DE",)):
    """Sélection de fichiers du catalogue, critère par critère.
    (docstring inchangée — voir ton fichier)
    """
    # Chaque critère accepte une valeur unique OU un tuple -> normalisé en liste
    # (None = filtre désactivé : toutes les valeurs sont conservées)
    sources = _en_liste(sources)
    defauts = _en_liste(defauts)
    diametres = _en_liste(diametres)
    positions_or = _en_liste(positions_or)
    charges = _en_liste(charges)

    # "Normal"/"Sain" dans sources -> source "Normal" du catalogue
    sources_norm = ["Normal" if str(src).lower() in ("normal", "sain") else src
                    for src in sources]
    avec_normal = "Normal" in sources_norm

    est_cible = catalogue["source"].isin(sources_norm)
    est_sain = catalogue["source"] == "Normal"
    # Garde les sources demandées + les fichiers "Normal" (sains) si "Normal"/"Sain" est demandé
    masque = est_cible | (est_sain & avec_normal)

    # Filtre sur le type de défaut (IR/B/OR) ; les fichiers sains passent toujours
    if defauts is not None:
        masque = masque & (catalogue["defaut"].isin(defauts) | est_sain)
    # Filtre sur le diamètre du défaut en millièmes de pouce (7/14/21) ; sains -> gardés via est_sain
    if diametres is not None:
        masque = masque & (catalogue["diametre_mils"].isin(diametres) | est_sain)
    # Filtre sur la position de l'OR ("3"/"6"/"12") ; sains/IR/B (position NaN) -> gardés
    if positions_or is not None:
        masque = masque & ((catalogue["position_OR"].isin(positions_or)
                            | catalogue["position_OR"].isna()) | est_sain)
    # Filtre sur la charge moteur en HP (0, 1, 2, 3)
    if charges is not None:
        masque = masque & catalogue["charge_hp"].isin(charges)

    # Catalogue filtré selon tous les critères ci-dessus
    selection = catalogue[masque].copy()

    def _valide(c):
        """Normalise un canal / tuple de canaux en tuple validé."""
        c = tuple(_en_liste(c))
        # Vérifie que chaque canal est bien DE / FE / BA
        for x in c:
            assert x in ("DE", "FE", "BA"), f"canal inconnu : {x!r} (attendu DE/FE/BA)"
        return c

    # Cas utilisé par RandomForest.py : canaux différents selon la source,
    # ex. {"12kDE": ("DE",), "12kFE": ("FE",), "Normal": ("DE", "FE")}
    if isinstance(canaux, dict):
        # canaux différents selon la source ; "*" = valeur par défaut
        # Valeur par défaut pour les sources absentes du dict ("*" ou ("DE",))
        defaut = _valide(canaux.get("*", ("DE",)))
        # Colonne "canaux" remplie LIGNE PAR LIGNE : le tuple de canaux de CE fichier
        selection["canaux"] = [_valide(canaux.get(src, defaut))
                               for src in selection["source"]]
    else:
        # Sinon : le même tuple de canaux est appliqué à toutes les lignes
        c = _valide(canaux)                     # tuple/str identique pour tous
        selection["canaux"] = [c] * len(selection)
    # Renvoie le DataFrame filtré + colonne "canaux" -> passé à features.construire_table_SpecMourad
    return selection