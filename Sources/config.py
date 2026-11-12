"""Constantes globales du projet (une seule source de vérité)."""
from .path import trouver_racine

RACINE = trouver_racine()

# Chemins de données
CACHE = RACINE / "data" / "cwru_cache"
FEATURES = RACINE / "data" / "features"

# Acquisition CWRU
URL_CWRU = "https://engineering.case.edu/sites/default/files/"
RPM_NOMINAL = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# Prétraitement
DUREE_FENETRE = 0.2            # s, durée d'une fenêtre d'indicateurs
BANDE_RESONANCE = (1000, 4000)  # Hz, bande de résonance des chocs
N_FFT = 2 ** 14                # zero-padding des spectres
HARMONIQUES = (1, 2, 3)        # harmoniques relevées par raie caractéristique
DEMI_BANDE = 5.0               # Hz, = résolution FFT d'une fenêtre de 0,2 s

# Coefficients × fréquence de rotation (source CWRU)
COEFF_FREQ = {
    "DE": {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.39828},
    "FE": {"BPFI": 4.9469, "BPFO": 3.0530, "BSF": 3.9874, "FTF": 0.3817},
}

# Style des graphiques du rapport
STYLES_RAIES = {"BPFI": ("purple", "-"), "BPFO": ("blue", "--"),
                "BSF": ("orange", "-."), "FTF": ("green", ":")}
STYLES_CONDITION = {"Sain": "#149C07", "Défaut": "#C82909"}
STYLES_CANAL = {"DE": "-", "FE": "--"}