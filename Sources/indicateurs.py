"""Extraction d'indicateurs (features) sur une fenêtre de signal.

Trois familles :
  - indicateurs_temporels : RMS, Pic, facteur de crête, kurtosis
  - indicateurs_spectraux : amplitudes et ratios des raies caractéristiques
    (BPFI, BPFO, BSF, FTF) + fréquence du pic dominant, issus du spectre
    d'enveloppe (passe-bande + Hilbert + FFT).
  - indicateurs_spectraux_snr : RMS de la bande et rapports signal/bruit
    Env_BPFI / Env_BPFO / Env_BSF du spectre d'enveloppe (méthode 2-5 kHz,
    niveau de fond = médiane du spectre entre 10 et 1000 Hz).
"""
import numpy as np
from scipy.stats import kurtosis

from .config import BANDE_FOND, COEFF_FREQ, DEMI_BANDE, HARMONIQUES, RAIES_SNR
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
    fc = freqs_caracteristiques(roulement, rpm)
    res = {}
    for nom, f in fc.items():
        for k in harmoniques:
            res[f"amp_{nom}_{k}"] = amplitude_raie(freqs, amp, k * f, demi_bande)
    total = sum(res[f"amp_{F}_1"] for F in RAIES)
    for F in RAIES:
        res[f"ratio_{F}"] = res[f"amp_{F}_1"] / total if total > 0 else 0.0
    dans_bande = freqs <= 1000
    res["f_pic"] = float(freqs[dans_bande][np.argmax(amp[dans_bande])])
    return res


def indicateurs_spectraux_snr(x_filt, env, fs, rpm, roulement="DE",
                              harmoniques=HARMONIQUES, bande_fond=BANDE_FOND):
    """Indicateurs "rapport signal/bruit" d'une fenêtre, sur le signal filtré
    et l'enveloppe calculés sur le FICHIER ENTIER (voir signaux.enveloppe_bande).

    Spectre d'enveloppe sans fenêtrage ni zero-padding ; pour chaque raie
    F ∈ {BPFI, BPFO, BSF}, Env_F = (somme des amplitudes max aux harmoniques
    k×F, k = 1..3, tolérance ±max(2 % de f, 1,5×résolution)) / fond, avec
    fond = médiane du spectre entre 10 et 1000 Hz. + RMS_bande : RMS du signal
    filtré dans la bande de résonance.
    """
    n = len(env)
    amp = np.abs(np.fft.rfft(env - env.mean())) * 2 / n
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fond = np.median(amp[(freqs >= bande_fond[0]) & (freqs <= bande_fond[1])])
    f_rot = rpm / 60
    res = {"RMS_bande": np.sqrt(np.mean(x_filt ** 2))}
    for nom, coef in COEFF_FREQ[roulement].items():
        if nom not in RAIES_SNR:
            continue
        total = 0.0
        for k in harmoniques:
            f = k * coef * f_rot
            tol = max(0.02 * f, 1.5 * freqs[1])
            total += amp[(freqs >= f - tol) & (freqs <= f + tol)].max()
        res[f"Env_{nom}"] = total / fond
    return res


COLONNES_TEMPORELS = ["RMS", "Pic", "Crete", "Kurtosis"]
COLONNES_SNR = ["RMS_bande"] + [f"Env_{F}" for F in RAIES_SNR]
COLONNES_SPECTRAUX = ([f"amp_{F}_{k}" for F in RAIES for k in HARMONIQUES]
                      + [f"ratio_{F}" for F in RAIES] 
                      # + ["f_pic"] # Suppression de f_pic car il indique des fréquences brutes, et n'est donc pas transposable sur différents roulement
                    )
COLONNES_INDICATEURS = COLONNES_TEMPORELS + COLONNES_SPECTRAUX