"""Traitement du signal vibratoire : filtrage, enveloppe, spectres.

Fonctions pures : entrée = array numpy, sortie = array ou (freqs, amp).
Aucune entrée/sortie fichier, aucune variable globale modifiée.
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


def filtrer_passe_bande(x, fs, bande=BANDE_RESONANCE, ordre=4):
    """Filtre Butterworth passe-bande, filtrage zero-phase (sosfiltfilt) :
    les chocs ne sont pas décalés dans le temps."""
    sos = sci.butter(ordre, [bande[0] / (fs / 2), bande[1] / (fs / 2)],
                     btype="bandpass", output="sos")
    return sci.sosfiltfilt(sos, x)


def extraire_enveloppe(x):
    """Enveloppe par transformée de Hilbert, centrée (retire le pic à 0 Hz)."""
    env = np.abs(hilbert(x))
    return env - np.mean(env)


def spectre_enveloppe(x, fs, bande=BANDE_RESONANCE, n_fft=N_FFT):
    """Chaîne complète d'analyse d'enveloppe : passe-bande + Hilbert + FFT.
    Renvoie (fréquences, amplitudes) du spectre de l'enveloppe."""
    env = extraire_enveloppe(filtrer_passe_bande(x, fs, bande))
    fenetre = np.hanning(len(env))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    amp = np.abs(np.fft.rfft(env * fenetre, n=n_fft)) / np.sum(fenetre) * 2
    return freqs, amp


def amplitude_raie(freqs, amp, f0, demi_bande):
    """Amplitude maximale du spectre dans [f0 - demi_bande, f0 + demi_bande]."""
    masque = (freqs >= f0 - demi_bande) & (freqs <= f0 + demi_bande)
    return float(amp[masque].max()) if masque.any() else 0.0