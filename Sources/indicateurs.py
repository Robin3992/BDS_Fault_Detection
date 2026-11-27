"""
fonctions de calcul d'indicateurs (features) sur une fenêtre de signal.

Deux familles :
  - indicateurs_temporels : RMS, Pic, facteur de crête, kurtosis
  - indicateurs_spectraux : amplitudes et ratios des raies caractéristiques
    (BPFI, BPFO, BSF, FTF) + fréquence du pic dominant, issus du spectre
    d'enveloppe (passe-bande + Hilbert + FFT).

Les indicateurs spectraux ont 2 versions relativement proche, car elles ont été mises en places 
par 2 membres du projets, puis comparées ensuite pour tester leurs avantages et limites.
"""
import numpy as np
from scipy.stats import kurtosis

from .config import DEMI_BANDE, HARMONIQUES, COEFF_FREQ
from .signaux import amplitude_raie, freqs_caracteristiques, spectre_enveloppe

RAIES = ("BPFI", "BPFO", "BSF", "FTF")


def indicateurs_temporels(x):
    """4 indicateurs temporels classiques de la surveillance vibratoire."""
    rms = np.sqrt(np.mean(x ** 2))
    pic = np.max(np.abs(x))
    # Remplacement de la fonciton Kurtosis de scipy.stats par la version manuelle pour améliorer la vitesse de calcul
    z = (x - x.mean()) / x.std()

    return {"RMS": rms, "Pic": pic, "Crete": pic / rms,
            # "Kurtosis": kurtosis(x, fisher=False)} remplacement Kurtosis de scipy.stats par la version manuelle
            "Kurtosis": float(np.mean(z**4))}

def indicateurs_spectraux(x, fs, rpm, roulement="DE",
                          harmoniques=HARMONIQUES, demi_bande=DEMI_BANDE, sos=None):
    """17 indicateurs spectraux extraits du spectre d'enveloppe d'une fenêtre.

    Pour chaque raie F ∈ {BPFI, BPFO, BSF, FTF} :
      - amp_F_k : amplitude max du spectre d'enveloppe à k×F (± demi_bande Hz)
      - ratio_F : amp_F_1 / somme des amp_*_1 (part d'énergie de la raie,
                  indépendante de l'amplitude absolue du signal)
    + f_pic : fréquence du pic dominant du spectre d'enveloppe sous 1 kHz.
    """

    
    freqs, amp = spectre_enveloppe(x, fs, sos=sos)

    # Récupération des fréquences caractéristiques associées au roulement souhaité
    fc = freqs_caracteristiques(roulement, rpm)
    res = {}
    for nom, f in fc.items():
        for k in harmoniques:
            res[f"amp_{nom}_{k}"] = amplitude_raie(freqs, amp, k * f, demi_bande)
    total = sum(res[f"amp_{F}_1"] for F in RAIES)
    for F in RAIES:
        res[f"ratio_{F}"] = res[f"amp_{F}_1"] / total if total > 0 else 0.0
    # /!\ f_pic : indique une fréquence brute en Hz, il n'est donc pas transposable sur différent roulement.
    dans_bande = freqs <= 1000
    res["f_pic"] = float(freqs[dans_bande][np.argmax(amp[dans_bande])])
    return res

# 4 features temporelles (RMS, Pic, facteur de crête, Kurtosis)
# -> 1ère partie de FEATURES dans RandomForest.py
COLONNES_TEMPORELS = ["RMS", "Pic", "Crete", "Kurtosis"]
COLONNES_SPECTRAUX = ([f"amp_{F}_{k}" for F in RAIES for k in HARMONIQUES]
                      + [f"ratio_{F}" for F in RAIES] 
                      + ["f_pic"]
                    )




# Code Mourad
COEF_DE = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135}
def M_indicateurs_spectraux(x_filt, env, fs, rpm):
    """Indicateurs d'une fenêtre : énergie de la bande et rapports signal/bruit
    du spectre d'enveloppe aux fréquences BPFI, BPFO et BSF."""
    
    n = len(env)
    amp = np.abs(np.fft.rfft(env - env.mean())) * 2 / n
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fond = np.median(amp[(freqs >= 10) & (freqs <= 1000)])   # niveau de bruit
    f_rot = rpm / 60
    res = {"RMS_bande": np.sqrt(np.mean(x_filt ** 2))}
    for nom, coef in COEF_DE.items():
        total = 0.0
        for k in (1, 2, 3):                                   # 3 premières harmoniques
            f = k * coef * f_rot
            tol = max(0.02 * f, 1.5 * freqs[1])               # tolérance de ±2 %
            total += amp[(freqs >= f - tol) & (freqs <= f + tol)].max()
        res[f"Env_{nom}"] = total / fond
    return res

# Code Mourad adapté à l'environement du main
def indicateurs_spectraux_Mourad(x_filt, env, fs, rpm, roulement="DE",
                              harmoniques=HARMONIQUES):
    """Indicateurs "rapport signal/bruit" d'une fenêtre, sur le signal filtré
    et l'enveloppe calculés sur le FICHIER ENTIER (voir signaux.enveloppe_bande,
    bande = BANDE_RESONANCE par défaut).

    Transformations identiques à la méthode du rapport : spectre d'enveloppe
    sans fenêtrage ni zero-padding ; pour chaque raie F de COEFF_FREQ[roulement]
    (DE ou FE : BPFI, BPFO, BSF, FTF), Env_F = (somme des amplitudes max aux
    harmoniques k×F, tolérance ±max(2 % de f, 1,5×résolution FFT)) / fond,
    avec fond = médiane du spectre entre 10 et 1000 Hz.
    + RMS_bande : RMS du signal filtré dans la bande de résonance.
    """
    bande_fond = (10, 1000)
    n = len(env)
    amp = np.abs(np.fft.rfft(env - env.mean())) * 2 / n
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fond = np.median(amp[(freqs >= bande_fond[0]) & (freqs <= bande_fond[1])])
    f_rot = rpm / 60
    res = {"RMS_bande": np.sqrt(np.mean(x_filt ** 2))}
    for nom, coef in COEFF_FREQ[roulement].items():
        total = 0.0
        for k in harmoniques:
            f = k * coef * f_rot
            tol = max(0.02 * f, 1.5 * freqs[1])
            total += amp[(freqs >= f - tol) & (freqs <= f + tol)].max()
        res[f"Env_{nom}"] = total / fond
    return res

# 5 features "Mourad" : RMS_bande + Env_BPFI/BPFO/BSF/FTF
# -> 2ème partie de FEATURES ; produites uniquement par features.construire_table_SpecMourad
COLONNES_Mourad = ["RMS_bande"] + [f"Env_{F}" for F in COEFF_FREQ["DE"]]
# -> ['RMS_bande', 'Env_BPFI', 'Env_BPFO', 'Env_BSF', 'Env_FTF']


COLONNES_INDICATEURS = COLONNES_TEMPORELS + COLONNES_SPECTRAUX + COLONNES_Mourad