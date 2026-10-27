"""Localisation de la racine du dépôt, quel que soit le dossier de travail."""
from pathlib import Path


def trouver_racine(repo="BDS_Fault_Detection", depart=None):
    """Remonte l'arborescence depuis `depart` jusqu'au dossier du dépôt
    et renvoie son chemin absolu.

    depart : dossier ou fichier de départ (défaut : dossier courant).
             Script :   trouver_racine(depart=Path(__file__).parent)
             Notebook/Quarto : trouver_racine() suffit (cwd = dossier du fichier).
    """
    if depart is not None:
        dossier = Path(depart)
    else:
        # Par défaut : le dossier contenant le package Sources/ (défini dans
        # Sources/config.py), et non le dossier courant, ainsi le package
        # fonctionne quel que soit l'endroit d'où il est importé.
        dossier = Path(__file__).resolve().parent.parent
    dossier = dossier.resolve()
    if dossier.is_file():
        dossier = dossier.parent
    for parent in (dossier, *dossier.parents):
        if repo in parent.name:
            return parent
    raise FileNotFoundError(f"Racine '{repo}' introuvable depuis {dossier}")