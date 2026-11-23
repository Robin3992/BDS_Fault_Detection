"""
Différentes fonctions relatives au traitement du signal vibratoire : filtrage, enveloppe, spectres.

"""

import numpy as np
import scipy.signal as sci
from scipy.signal import hilbert

from .config import BANDE_RESONANCE, COEFF_FREQ, N_FFT


def freqs_caracteristiques(roulement, rpm):
    """Fréquences caractéristiques (Hz) du roulement 'DE' ou 'FE',
    proportionnelles à la fréquence de rotation."""
    f_rot = rpm / 60
    return {nom: coef * f_rot for nom, coef in COEFF_FREQ[roulement].items()}


def spectre_fft(x, fs, n_fft=None):
    """Spectre d'amplitude unilatéral du signal (centré, fenêtre de Hann).
    Renvoie (fréquences, amplitudes)."""
    n = len(x)
    n_fft = n_fft or n
    fenetre = np.hanning(n)
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    amp = np.abs(np.fft.rfft((x - np.mean(x)) * fenetre, n=n_fft)) / np.sum(fenetre) * 2
    return freqs, amp

def creer_filtre(fs, bande=BANDE_RESONANCE, ordre=4):
    """Crée le filtre passe-bande (SOS) UNE FOIS pour toutes les fenêtres."""
    return sci.butter(ordre, [bande[0]/(fs/2), bande[1]/(fs/2)],
                      btype="bandpass", output="sos")

def filtrer_passe_bande(x, fs, bande=BANDE_RESONANCE, ordre=4, sos=None):
    if sos is None:
        sos = creer_filtre(fs, bande, ordre)
    return sci.sosfiltfilt(sos, x)


def extraire_enveloppe(x):
    """Enveloppe par transformée de Hilbert, centrée (retire le pic à 0 Hz avec - np.mean(env))."""
    env = np.abs(hilbert(x))
    return env - np.mean(env)

# 
def spectre_enveloppe(x, fs, bande=BANDE_RESONANCE, n_fft=N_FFT, sos=None):
    """Chaîne complète d'analyse d'enveloppe : passe-bande + Hilbert + FFT.
    Renvoie (fréquences, amplitudes) du spectre de l'enveloppe."""
    env = extraire_enveloppe(filtrer_passe_bande(x, fs, bande, sos=sos))
    fenetre = np.hanning(len(env))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    amp = np.abs(np.fft.rfft(env * fenetre, n=n_fft)) / np.sum(fenetre) * 2
    return freqs, amp


def amplitude_raie(freqs, amp, f0, demi_bande):
    """Amplitude maximale du spectre dans [f0 - demi_bande, f0 + demi_bande]."""
    masque = (freqs >= f0 - demi_bande) & (freqs <= f0 + demi_bande)
    return float(amp[masque].max()) if masque.any() else 0.0





# Code Mourad
def M_enveloppe(x, fs, bande=BANDE_RESONANCE):
    """Filtre passe-bande, puis enveloppe (module du signal analytique)."""
    m_sos = sci.butter(4, bande, btype="bandpass", fs=fs, output="sos")
    x_filt = sci.sosfiltfilt(m_sos, x)
    env = np.abs(hilbert(x_filt))
    return x_filt, env
    
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
def M_up_enveloppe_bande(x, fs, bande=BANDE_RESONANCE, ordre=4, sos=None):
    """Filtre passe-bande puis enveloppe (module du signal analytique) appliqués
    au signal ENTIER, une fois par fichier : renvoie (x_filtre, enveloppe),
    à découper ensuite fenêtre par fenêtre.
    bande par défaut = BANDE_RESONANCE (même bande que spectre_enveloppe, SOS
    partageable via le cache de creer_filtre).
    """
    if sos is None:
        sos = creer_filtre(fs, bande, ordre)
    x_filt = sci.sosfiltfilt(sos, x)
    env = np.abs(hilbert(x_filt))
    return x_filt, env