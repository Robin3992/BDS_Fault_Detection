
#| tbl-cap: "Nombre de fichiers du catalogue CWRU par source et type de défaut."
#| label: tbl-catalogue
#Bloc 1 : remplace les deux anciens blocs de chargement
#Ce que fait chaque partie
#1.1 recopie les tableaux des quatre pages du site. Les numéros ont été vérifiés aujourd’hui sur les pages elles-mêmes, et None correspond aux astérisques (données non disponibles).
#1.2 transforme ces tableaux en une table, le catalogue, avec une ligne par fichier et un nom qui résume tout. Par exemple, 12kDE_OR021at3_L2 se lit : 12 kHz, défaut côté entraînement, bague extérieure, 0,021", position 3 h, charge 2 HP.
#1.3 définit charger(nom). Elle télécharge le fichier la première fois, puis le relit en local. Elle choisit la bonne variable quand un fichier en contient deux, et prend la vitesse nominale si la vitesse mesurée manque. Ce sont les deux défauts du code précédent.

#Résultat attendu : 161 fichiers au catalogue, puis un tableau de synthèse :
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sci
from scipy.io import loadmat
from scipy.signal import hilbert
from scipy.stats import kurtosis
from scipy.fftpack import fft, fftfreq

# ------------------------------------------------------------------
# 1.1 Les numéros de fichiers du site, page par page
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

RPM_NOMINAL = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# ------------------------------------------------------------------
# 1.2 Le catalogue : une ligne par fichier, avec un nom lisible
#     ex. 'Normal_L0', '12kDE_IR007_L0', '12kDE_OR021at3_L2', '48kDE_B014_L1'
# ------------------------------------------------------------------
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
            nom = f"{source}_{localisation}{diametre:03d}" + (f"at{position}" if position else "") + f"_L{charge}"
            lignes.append({"nom": nom, "numero": num, "source": source, "fs": fs,
                           "defaut": localisation, "position_OR": position or None,
                           "diametre_mils": diametre, "charge_hp": charge})

catalogue = pd.DataFrame(lignes).set_index("nom")
catalogue["rpm_nominal"] = catalogue["charge_hp"].map(RPM_NOMINAL)

# ------------------------------------------------------------------
# 1.3 Lecture d'un fichier depuis le site (copie locale sous un nom lisible)
# ------------------------------------------------------------------
URL_CWRU = "https://engineering.case.edu/sites/default/files/"
CACHE = Path("../data/cwru_cache")

def charger(nom):
    """Renvoie {'DE', 'FE', 'BA', 'fs', 'rpm'} pour un fichier du catalogue
    (ex. '12kDE_IR007_L0')."""
    info = catalogue.loc[nom]
    numero = int(info["numero"])
    chemin = CACHE / f"{nom}__{numero}.mat"            # ex. 12kDE_IR007_L0__105.mat

    if not chemin.exists():                            # 1re fois : téléchargement depuis le site
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL_CWRU + f"{numero}.mat") as reponse:
            chemin.write_bytes(reponse.read())
    mat = loadmat(chemin)

    def choisir(suffixe):                              # privilégie la variable du bon numéro
        cles = [k for k in mat if k.endswith(suffixe)]
        bonne = [k for k in cles if k.startswith(f"X{numero:03d}")]
        return (bonne or cles or [None])[0]

    signaux = {"fs": int(info["fs"])}
    for canal in ["DE", "FE", "BA"]:
        cle = choisir(f"_{canal}_time")
        signaux[canal] = None if cle is None else np.ravel(mat[cle])
    cle_rpm = choisir("RPM")
    signaux["rpm"] = float(np.squeeze(mat[cle_rpm])) if cle_rpm else float(info["rpm_nominal"])
    return signaux

from IPython.display import Markdown

tab = catalogue.groupby(["source", "defaut"]).size().unstack(fill_value=0)
tab = tab[["Sain", "IR", "B", "OR"]].reindex(["Normal", "12kDE", "48kDE", "12kFE"])
tab = tab.reset_index().rename(columns={"source": "Source", "IR": "Bague int. (IR)",
                                        "B": "Bille (B)", "OR": "Bague ext. (OR)"})
Markdown(tab.to_markdown(index=False))

####################################################################################################################################################



#| include: false
#| label: tbl-controle
#| tbl-cap: "Contrôle de la lecture sur deux fichiers de référence (capteur DE, 12 kHz)."
#| tbl-colwidths: [30, 17, 15, 13, 25]
# Bloc 2 : contrôle de la lecture sur deux fichiers
lignes = []
for nom in ["Normal_L0", "12kDE_IR007_L0"]:
    s = charger(nom)
    canaux = [c for c in ["DE", "FE", "BA"] if s[c] is not None]
    lignes.append({"Fichier": nom,
                   "Points": f"{len(s['DE']):,}".replace(",", " "),
                   "Durée (s)": f"{len(s['DE']) / s['fs']:.1f}".replace(".", ","),
                   "tr/min": round(s["rpm"]),
                   "Canaux": ", ".join(canaux)})

Markdown(pd.DataFrame(lignes).to_markdown(
    index=False, colalign=("left", "right", "right", "right", "left")))


####################################################################################################################################################
#| label: selection
#| code-summary: "Sélection des 40 fichiers"
#| tbl-cap: "Nombre de fichiers sélectionnés par localisation et diamètre du défaut."
est_sain   = catalogue["source"] == "Normal"
est_12kDE  = catalogue["source"] == "12kDE"
bon_defaut = catalogue["defaut"].isin(["IR", "B"]) | (catalogue["position_OR"] == "6")
bon_diam   = catalogue["diametre_mils"].isin([7, 14, 21])

selection = catalogue[est_sain | (est_12kDE & bon_defaut & bon_diam)]
selection.groupby(["defaut", "diametre_mils"]).size().unstack(fill_value=0)






#| label: table-indicateurs
#| code-summary: "Calcul des indicateurs par fenêtre"
#| tbl-cap: "Nombre de fenêtres de 0.2s après découpage du signal"
def indicateurs(x):
    rms = np.sqrt(np.mean(x**2))
    pic = np.max(np.abs(x))
    return {"RMS": rms, "Pic": pic, "Crete": pic / rms,
            "Kurtosis": kurtosis(x, fisher=False)}

DUREE_FENETRE = 0.2      # secondes

lignes = []
for nom, info in selection.iterrows():
    s = charger(nom)                                   # téléchargement la 1re fois
    x, fs = s["DE"], s["fs"]
    taille = int(DUREE_FENETRE * fs)
    for i in range(len(x) // taille):
        res = {"fichier": nom, "fenetre": i, "rpm": s["rpm"], **info.to_dict()}
        res.update(indicateurs(x[i*taille:(i+1)*taille]))
        lignes.append(res)

table = pd.DataFrame(lignes)
table["classe"] = np.where(table["defaut"] == "Sain", "Sain",
                           table["defaut"] + table["diametre_mils"].astype(str).str.zfill(3))

Path("../data/features").mkdir(parents=True, exist_ok=True)
table.to_csv("../data/features/table_indicateurs.csv", index=False)

#print(f"{len(table)} fenêtres issues de {table['fichier'].nunique()} fichiers")
table.groupby(["classe", "charge_hp"]).size().unstack()





#| label: tbl-extrait
#| tbl-cap: "Extrait de la table des indicateurs (trois premières fenêtres de deux fichiers)."
extrait = (table[table["fichier"].isin(["Normal_L0", "12kDE_IR007_L0"])]
           .groupby("fichier").head(3)
           [["fichier", "fenetre", "charge_hp", "classe",
             "RMS", "Pic", "Crete", "Kurtosis"]]
           .round(3))
Markdown(extrait.to_markdown(index=False))




#| label: parametres-explo
#| include: false
INDICATEURS = ["RMS", "Pic", "Crete", "Kurtosis"]
ORDRE = ["Sain", "IR007", "IR014", "IR021", "B007", "B014", "B021",
         "OR007", "OR014", "OR021"]




#| label: tbl-descriptives
#| tbl-cap: "Moyenne et écart-type des indicateurs par classe (toutes charges confondues)."
table.groupby("classe")[INDICATEURS].agg(["mean", "std"]).round(3).reindex(ORDRE)





#| label: fig-boxplots
#| fig-cap: "Distribution des indicateurs par classe (fenêtres de 0,2 s, toutes charges)."
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, ind in zip(axes.ravel(), INDICATEURS):
    donnees = [table.loc[table["classe"] == c, ind] for c in ORDRE]
    ax.boxplot(donnees, tick_labels=ORDRE, showfliers=False)
    ax.set_title(ind)
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()




#| label: fig-charge
#| fig-cap: "Médiane du RMS et du kurtosis selon la charge, pour chaque classe."
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, ind in zip(axes, ["RMS", "Kurtosis"]):
    med = table.groupby(["charge_hp", "classe"])[ind].median().unstack()[ORDRE]
    med.plot(ax=ax, marker="o", legend=False)
    ax.set_title(f"{ind} médian selon la charge")
    ax.set_xlabel("Charge (HP)")
    ax.set_xticks(range(4))
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()







#| label: fig-nuage
#| fig-cap: "RMS et kurtosis de chaque fenêtre, par classe."
fig, ax = plt.subplots(figsize=(10, 7))
for c in ORDRE:
    g = table[table["classe"] == c]
    ax.scatter(g["RMS"], g["Kurtosis"], s=8, alpha=0.5, label=c)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("RMS (échelle log)")
ax.set_ylabel("Kurtosis (échelle log)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
plt.tight_layout()
plt.show()





#| label: tbl-variance
#| tbl-cap: 'Part de la variabilité expliquée par les différences entre fichiers ($\eta^2$).'
var = []
for ind in INDICATEURS:
    x = table[ind]
    moyenne_fichier = table.groupby("fichier")[ind].transform("mean")
    sc_totale = ((x - x.mean()) ** 2).sum()
    sc_intra  = ((x - moyenne_fichier) ** 2).sum()
    var.append({"Indicateur": ind,
                "Part entre fichiers (%)": round(100 * (1 - sc_intra / sc_totale), 1)})
pd.DataFrame(var)


'''

#| label: fig-spectre
#| fig-cap: "Décomposition d'un signal synthétique : composantes temporelles (gauche) et spectrales (droite)."
# Définitions fonctions
def spectre_fft(x, fs):
    """Spectre d'amplitude : renvoie (fréquences, amplitudes)."""
    n = len(x)
    freqs = fftfreq(n, d=1 / fs)[: n // 2]
    amp = np.abs(fft(x - np.mean(x)))[: n // 2] * 2 / n # ajout de "- np.mean(x)" pour supprimer la composante continue du signal
    return freqs, amp

fs = 12_000
t = np.arange(0, 1, 1 / fs)
x30hz = 1.0 * np.sin(2 * np.pi * 30 * t)      # "rotation" à 30 Hz
x162hz = 0.5 * np.sin(2 * np.pi * 162 * t)    # "défaut" à 162 Hz
xbruit = 0.3 * np.random.randn(len(t))        # bruit 
xtot = x30hz + x162hz + xbruit

# --- Calcul des spectres (une seule fois par composante) ---
f_30hz, a30 = spectre_fft(x30hz, fs)
f_162hz, a162 = spectre_fft(x162hz, fs)
f_bruit, a_bruit = spectre_fft(xbruit, fs)
f_tot, a_tot = spectre_fft(xtot, fs)

fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col")

# Colonne 0 : signaux temporels — Colonne 1 : spectres
signaux = [
    (x30hz,  f_30hz,  a30,     "30 Hz (rotation)"),
    (x162hz, f_162hz, a162,    "162 Hz (défaut)"),
    (xbruit, f_bruit, a_bruit, "Bruit"),
    (xtot,   f_tot,   a_tot,   "Signal complet"),
]
n = int(0.1 * fs)   # 0,1 seconde

for ligne, (x, freqs, amp, titre) in enumerate(signaux):
    # --- Colonne 0 : temporel ---
    axes[ligne, 0].plot(t[:n], x[:n], color="steelblue", lw=0.8)
    axes[ligne, 0].set_title(f"{titre} — signal temporel", loc="left")
    axes[ligne, 0].set_ylabel("Amplitude")
    axes[ligne, 0].grid(alpha=0.3)

    # --- Colonne 1 : spectre, zoomé sur 0–300 Hz ---
    axes[ligne, 1].plot(freqs, amp, color="crimson", lw=0.8)
    axes[ligne, 1].set_xlim(0, 300)
    axes[ligne, 1].set_title(f"{titre} — spectre", loc="left")
    axes[ligne, 1].grid(alpha=0.3)

axes[-1, 0].set_xlabel("Temps (s)")
axes[-1, 1].set_xlabel("Fréquence (Hz)")
plt.tight_layout(); plt.show()




#| fig-cap: "Le repliement (aliasing) à 12 kHz."

# Cas du projet : fs = 12 kHz -> Nyquist = 6 kHz
FS_DEMO = 12_000
F_VRAIE = 11_600                 # résonance réelle, au-dessus de Nyquist
F_ALIAS = FS_DEMO - F_VRAIE     # 400 Hz : là où elle se replie

# Temporel : 10 ms (indices entiers, pas d'erreur flottante)
n_ech = np.arange(int(0.01 * FS_DEMO))
points = np.sin(2 * np.pi * F_VRAIE * n_ech / FS_DEMO)

# FFT : la même sinusoïde observée 2 s (raie fine)
n_fft = np.arange(int(2.0 * FS_DEMO))
f_fft, a_fft = spectre_fft(np.sin(2 * np.pi * F_VRAIE * n_fft / FS_DEMO), FS_DEMO)

t_cont = np.arange(0, 0.01, 1 / 1_000_000)    # courbe continue (dense)

fig, axes = plt.subplots(4, 1, figsize=(12, 12.5))

# 1. Le signal réel (continu) avec les points échantillonnés transparents
axes[0].plot(t_cont * 1000, np.sin(2 * np.pi * F_VRAIE * t_cont),
             color="crimson", lw=0.8, label="Signal réel (continu)")
axes[0].plot(n_ech / FS_DEMO * 1000, points, "o", ms=4, color="black",
             alpha=0.35, label="Points échantillonnés (fs = 12 kHz)")
axes[0].set_title(f"1. Le signal réel : une résonance à {F_VRAIE/1000:.1f} kHz", loc="left")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlabel("Temps (ms)")
axes[0].legend(fontsize=9, loc="upper right")

# 2. Ce qu'on mesure : les points échantillonnés, seuls
axes[1].plot(n_ech / FS_DEMO * 1000, points, "o", ms=4, color="crimson")
axes[1].set_title("2. Ce qu'on mesure : fs = 12 kHz "
                  "(Nyquist = 6 kHz < 11,6 kHz : trop lent !)", loc="left")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Temps (ms)")

# 3. Ce que croit l'analyste : l'alias. Attention : une fréquence entre
#    fs/2 et fs se replie AVEC INVERSION DE PHASE -> on trace -sin.
#    (Vérifié : écart points/courbe ~1e-13)
axes[2].plot(n_ech / FS_DEMO * 1000, points, "o", ms=5, color="crimson",
             label="Points mesurés")
axes[2].plot(t_cont * 1000, -np.sin(2 * np.pi * F_ALIAS * t_cont),
             color="steelblue", lw=2,
             label=f"Sinusoïde perçue : {F_ALIAS:.0f} Hz (repliée, phase inversée)")
axes[2].set_title(f"3. Ce que croit l'analyste : une sinusoïde de {F_ALIAS:.0f} Hz "
                  "(repliement : fs − f = 12 000 − 11 600)", loc="left")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Temps (ms)")
axes[2].legend(fontsize=9, loc="upper right")

# 4. La même illusion dans le spectre : la fausse raie, dans la bande observable
ax = axes[3]
ax.plot(f_fft / 1000, a_fft, lw=1.2, color="steelblue", label="FFT du signal mesuré")
ax.axvspan(0, FS_DEMO / 2000, color="lightgreen", alpha=0.25,
           label="Bande observable (0 – Nyquist = 6 kHz)")
ax.axvline(F_ALIAS / 1000, color="steelblue", ls="--", lw=1.8,
           label=f"Raie PERÇUE : {F_ALIAS/1000:.1f} kHz (fausse)")
ax.axvline(F_VRAIE / 1000, color="crimson", ls="--", lw=1.8,
           label=f"Vraie raie : {F_VRAIE/1000:.1f} kHz (hors bande)")
ax.annotate("le repliement déplace\nla vraie raie ICI",
            xy=(F_ALIAS / 1000, a_fft.max() * 0.75),
            xytext=(2.5, a_fft.max() * 0.9), fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))
ax.annotate("vraie fréquence : invisible\n(hors de la bande observable)",
            xy=(F_VRAIE / 1000, a_fft.max() * 0.5),
            xytext=(6.5, a_fft.max() * 0.75), fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round", fc="mistyrose", ec="crimson"))
ax.set_xlim(0, 12)
ax.set_title("4. FFT vue par l'analyste : une raie à 0,4 kHz — "
             "la vraie (11,6 kHz) est hors bande", loc="left")
ax.set_xlabel("Fréquence (kHz)")
ax.set_ylabel("Amplitude")
ax.legend(fontsize=8, loc="upper right")

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()





#| fig-cap: "La résolution spectrale 1/T"

FS_DEMO = 12_000
F1, F2 = 30, 33          # deux raies séparées de seulement 3 Hz

t_court = np.arange(0, 0.1, 1 / FS_DEMO)    # observation de 0,1 s
t_long  = np.arange(0, 1.0, 1 / FS_DEMO)    # observation de 1 s
x_court = np.sin(2 * np.pi * F1 * t_court) + 0.8 * np.sin(2 * np.pi * F2 * t_court)
x_long  = np.sin(2 * np.pi * F1 * t_long)  + 0.8 * np.sin(2 * np.pi * F2 * t_long)

f_c, a_c = spectre_fft(x_court, FS_DEMO)
f_l, a_l = spectre_fft(x_long, FS_DEMO)

def tracer_resolution(ax, amp, resolution):
    """Flèche de largeur EXACTEMENT égale à la résolution (1/T Hz),
    calée sur la raie 30 Hz (de F1 à F1 + résolution), texte centré."""
    y = amp.max() * 0.55
    ax.annotate("", xy=(F1 + resolution, y), xytext=(F1, y),
                arrowprops=dict(arrowstyle="<->", color="darkgreen",
                                lw=2.5, mutation_scale=14))
    ax.text(F1 + resolution / 2, y * 1.08,
            f"résolution = {resolution:.0f} Hz",
            ha="center", va="bottom", fontsize=11, color="darkgreen",
            fontweight="bold")

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# 1. Le signal temporel : les deux sinusoïdes, indiscernables à l'œil
axes[0].plot(t_court * 1000, x_court, lw=0.8, color="crimson")
axes[0].set_title("1. Le signal : deux sinusoïdes (30 + 33 Hz) "
                  "— indiscernables dans le temporel", loc="left")
axes[0].set_xlabel("Temps (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(alpha=0.3)

# 2. Observation courte (T = 0,1 s) : résolution 10 Hz -> raies confondues
axes[1].plot(f_c, a_c, lw=1.2, color="crimson", label="Spectre observé sur 0,1 s")
axes[1].axvline(F1, color="purple", ls="--", lw=1, label="vraie raie 30 Hz")
axes[1].axvline(F2, color="blue",   ls="--", lw=1, label="vraie raie 33 Hz")
axes[1].set_xlim(18, 46)               # marge gauche : pic visible en entier
axes[1].set_ylim(0, a_c.max() * 1.2)
axes[1].set_xticks(np.arange(18, 47, 2))
tracer_resolution(axes[1], a_c, resolution=10)   # flèche : 30 -> 40 Hz
axes[1].set_title("2. Observé 0,1 s : résolution 1/T = 10 Hz "
                  "→ UNE seule raie : les deux se confondent", loc="left")
axes[1].set_xlabel("Fréquence (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].legend(fontsize=9, loc="upper right")
axes[1].grid(alpha=0.3)

# 3. Observation longue (T = 1 s) : résolution 1 Hz -> raies séparées
axes[2].plot(f_l, a_l, lw=1.2, color="crimson", label="Spectre observé sur 1 s")
axes[2].axvline(F1, color="purple", ls="--", lw=1, label="vraie raie 30 Hz")
axes[2].axvline(F2, color="blue",   ls="--", lw=1, label="vraie raie 33 Hz")
axes[2].set_xlim(18, 46)               # même fenêtre que le panneau 2
axes[2].set_ylim(0, a_l.max() * 1.2)
axes[2].set_xticks(np.arange(18, 47, 2))
tracer_resolution(axes[2], a_l, resolution=1)    # flèche : 30 -> 31 Hz
axes[2].set_title("3. Observé 1 s : résolution 1/T = 1 Hz "
                  "→ DEUX raies distinctes", loc="left")
axes[2].set_xlabel("Fréquence (Hz)")
axes[2].set_ylabel("Amplitude")
axes[2].legend(fontsize=9, loc="upper right")
axes[2].grid(alpha=0.3)

plt.tight_layout(); plt.show()







# --- Fichiers étudiés : baseline saine et défaut bague intérieure, 0 HP ---
NOM_SAIN   = "Normal_L0"
NOM_DEFAUT = "12kDE_IR007_L0"

sain   = charger(NOM_SAIN)
defaut = charger(NOM_DEFAUT)

STYLES = {
    "condition": {"Sain": "#149C07", "Défaut": "#C82909"},
    "canal":     {"DE": "-",        "FE": "--"},
}

STYLES_RAIES = {  # fréquences caractéristiques
    "BPFI": ("purple", "-"),
    "BPFO": ("blue",   "--"),
    "BSF":  ("orange", "-."),
    "FTF":  ("green",  ":"),
}




#|  tbl-cap: "Résumé des signaux, durées calculées avec le fs annoncé par le catalogue."

SIGNAUX = {  # nom: (signaux, canal, condition)
    "Sain — moteur (DE)":        (sain,   "DE", "Sain"),
    "Sain — ventilateur (FE)":   (sain,   "FE", "Sain"),
    "Défaut — moteur (DE)":      (defaut, "DE", "Défaut"),
    "Défaut — ventilateur (FE)": (defaut, "FE", "Défaut"),
}

resume = pd.DataFrame([
    {"Condition": nom,
     "Nombre de points": len(s[canal]),
     "fs (Hz)": s["fs"],
     #"Durée (s)": round(len(s[canal]) / s["fs"], 2),
     #"NaN": int(np.sum(~np.isfinite(s[canal])))}
     "NaN": int(np.sum(pd.isna(s[canal])))}
    for nom, (s, canal, _) in SIGNAUX.items()
])
resume




#|  tbl-cap: "Contrôle du fs sur base de fréquence de rotation moteur."
def spectre_fft(x, fs):
    """Spectre d'amplitude : renvoie (fréquences, amplitudes)."""
    n = len(x)
    freqs = fftfreq(n, d=1 / fs)[: n // 2]
    # amp = np.abs(fft(x))[: n // 2] * 2 / n
    # ajout de "- np.mean(x)" pour supprimer la composante continue du signal
    amp = np.abs(fft(x - np.mean(x)))[: n // 2] * 2 / n
    return freqs, amp

x = sain["DE"]  # signal du capteur DE (Drive End)
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
for ax, fs in zip(axes, [48000, 24000, 12000]):
    f, a = spectre_fft(x, fs)
    ax.plot(f, a)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.015)
    ax.axvline(1797/60, color="red", ls="--", label="rotation moteur attendue ≈ 30 Hz")
    ax.set_title(f"Signal sain, en supposant fs = {fs} Hz")
    ax.legend()
axes[1].set_xlabel("Fréquence (Hz)")
plt.tight_layout(); plt.show()




COEFF_FREQ = {  # coefficients × fréquence de rotation, source CWRU
    "DE": {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.39828},
    "FE": {"BPFI": 4.9469, "BPFO": 3.0530, "BSF": 3.9874, "FTF": 0.3817},
}
RPM = defaut["rpm"] # vitesse du fichier si présente, sinon nominale du catalogue

def freqs_caracteristiques(roulement, rpm=RPM):
    """Fréquences caractéristiques (Hz) du roulement 'DE' ou 'FE'."""
    f_rot = rpm / 60
    return {nom: coef * f_rot for nom, coef in COEFF_FREQ[roulement].items()}

pd.DataFrame({
    "vitesse de rotation": [f"{RPM:.0f} tr/min"] * len(COEFF_FREQ["DE"]),
    "coefficient": pd.Series(COEFF_FREQ["DE"]),
    "f. caractéristique (DE)": pd.Series(freqs_caracteristiques("DE")).round(2).astype(str) + " Hz",
}).round(4)







#| fig-cap: "0,1 seconde de signal ; vert = sain, rouge = défaut ; trait plein = Drive End, tirets = Fan End."

def tracer_temps(x, fs, ax, condition, canal, titre):
    n = int(0.1 * fs)
    ax.plot(np.arange(n) / fs, x[:n],
            color=STYLES["condition"][condition],
            ls=STYLES["canal"][canal], lw=0.7)
    ax.set_title(titre, loc="left")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, (nom, (s, canal, cond)) in zip(axes.flat, SIGNAUX.items()):
    tracer_temps(s[canal], s["fs"], ax, cond, canal, nom)
plt.tight_layout(); plt.show()





#| fig-cap: "Zoom sur 30 ms du signal défectueux ; les repères sont espacés de la période BPFI théorique."

BPFI_DE = freqs_caracteristiques("DE")["BPFI"]
fs = defaut["fs"]

x = defaut["DE"]
n = int(0.03 * fs)
t = np.arange(n) / fs
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t, x[:n], color="red", ls="-", lw=0.8, label="Signal")
ax.vlines(np.arange(0, t[-1], 1 / BPFI_DE), *ax.get_ylim(),
          colors="crimson", ls=":", lw=1,
          label=f"Période BPFI ≈ {BPFI_DE:.0f} Hz")
ax.set_title("Défaut bague intérieure — repérage des chocs", loc="left")
ax.set_xlabel("Temps (s)"); ax.set_ylabel("Accélération (g)")
ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()





#| fig-cap: "Spectres d'amplitude FFT : le défaut ajoute de l'énergie haute fréquence."

def tracer_fft(ax, x, fs, condition, canal, titre,
               roulement=None, x_max=None):
    freqs, amp = spectre_fft(x, fs)
    ax.plot(freqs, amp, color=STYLES["condition"][condition],
            ls=STYLES["canal"][canal], lw=0.5)
    ax.set_title(titre, loc="left")
    ax.set_xlim(0, x_max if x_max else fs / 2)
    ax.set_xlabel("Fréquence (Hz)")
    ax.grid(alpha=0.3)
    if roulement:  # raies caractéristiques + harmoniques 2x, 3x
        for nom, val in freqs_caracteristiques(roulement).items():
            couleur, ls = STYLES_RAIES[nom]
            ax.axvline(val, color=couleur, ls=ls, lw=1.8, alpha=0.4,
                       label=f"{nom} = {val:.1f} Hz")
            for k in range(2, 4):
                if k * val < (x_max or fs / 2):
                    ax.axvline(k * val, color=couleur, ls=ls,
                               lw=0.8, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
for colonne, canal in enumerate(("DE", "FE")):
    for ligne, condition in enumerate(("Sain", "Défaut")):
        nom = [n for n in SIGNAUX
               if n.endswith(f"({canal})") and n.startswith(condition)][0]
        s, can, cond = SIGNAUX[nom][0], canal, condition
        tracer_fft(axes[ligne, colonne], s[canal], s["fs"], cond, can, nom)
        if colonne == 0:
            axes[ligne, colonne].set_ylabel("Amplitude")
plt.tight_layout(); plt.show()




#| fig-cap: "FFT basse fréquence du signal défectueux avec raies caractéristiques (fondamentales et harmoniques)."

fs = defaut["fs"]

# Fenêtre d'affichage : jusqu'à 3,5 × la raie BPFI la plus haute
X_MAX = max(freqs_caracteristiques(r)["BPFI"] for r in ("DE", "FE")) * 3.5

# Chaque spectre n'est calculé qu'une seule fois
SPECTRES_DEFAUT = {canal: spectre_fft(defaut[canal], fs)
                   for canal in ("DE", "FE")}
Y_MAX = max(amp[(freqs >= 0) & (freqs <= X_MAX)].max()
            for freqs, amp in SPECTRES_DEFAUT.values())

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
for ax, (canal, (freqs, amp)) in zip(axes, SPECTRES_DEFAUT.items()):
    tracer_fft(ax, defaut[canal], fs, "Défaut", canal, f"Défaut — {canal}",
               roulement=canal, x_max=X_MAX)
    ax.set_ylim(0, Y_MAX * 1.05)
    ax.set_ylabel("Amplitude")
plt.tight_layout(); plt.show()





#| fig-cap: "Les étapes de l'analyse d'enveloppe sur le signal défectueux (côté moteur)."

BANDE_RESONANCE = (1000, 4000)  # bande de résonance isolée (Hz)
N_FFT = 2 ** 14

def filtrer_passe_bande(x, fs, bande=BANDE_RESONANCE):
    sos = sci.butter(4, [bande[0] / (fs / 2), bande[1] / (fs / 2)],
                     btype="bandpass", output="sos")
    return sci.sosfiltfilt(sos, x)     # zero-phase : chocs non décalés

def extraire_enveloppe(x):
    env = np.abs(hilbert(x))
    return env - np.mean(env)          # retire le pic à 0 Hz

def spectre_enveloppe(x, fs, bande=BANDE_RESONANCE, n_fft=N_FFT):
    """Chaîne complète : passe-bande + Hilbert + FFT."""
    env = extraire_enveloppe(filtrer_passe_bande(x, fs, bande))
    fenetre = np.hanning(len(env))
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
    amp = np.abs(np.fft.rfft(env * fenetre, n=n_fft)) / np.sum(fenetre) * 2
    return freqs, amp

fs = defaut["fs"]
x = defaut["DE"]
x_filtre = filtrer_passe_bande(x, fs)
enveloppe = extraire_enveloppe(x_filtre)
t = np.arange(len(x)) / fs

fig, axes = plt.subplots(4, 1, figsize=(12, 11))
vues = [
    (f"1. Signal brut — chocs noyés dans les résonances (Défaut — DE)", x, "red", "-"),
    (f"2. Après filtre passe-bande {BANDE_RESONANCE[0]}–{BANDE_RESONANCE[1]} Hz",
     x_filtre, "tab:blue", "-"),
    ("3. Enveloppe (Hilbert) — le rythme des chocs apparaît",
     enveloppe, "tab:green", "-"),
]
for ax, (titre, sig, couleur, ls) in zip(axes, vues):
    n = int(0.05 * fs)
    ax.plot(t[:n], sig[:n], color=couleur, ls=ls, lw=0.8)
    ax.set_title(titre, loc="left")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

freqs_env, amp_env = spectre_enveloppe(x, fs)
ax = axes[3]
ax.plot(freqs_env, amp_env, color="tab:red", lw=0.8)
ax.set_title("4. FFT de l'enveloppe — raies caractéristiques nettes", loc="left")
ax.set_ylabel("Amplitude d'enveloppe")
ax.set_xlabel("Fréquence (Hz)")
ax.set_xlim(0, X_MAX)
for nom, val in freqs_caracteristiques("DE").items():
    if nom != "BPFI":
        continue
    couleur, ls = STYLES_RAIES[nom]
    ax.axvline(val, color=couleur, ls=ls, lw=1.5, alpha=0.5,
               label=f"{nom} = {val:.1f} Hz")

    for k in range(2, 4):              # harmoniques 2x et 3x
        if k * val < X_MAX:
            ax.axvline(k * val, color=couleur, ls=ls, lw=0.8, alpha=0.2)


ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()






#| fig-cap: "Spectres d'enveloppe avec raies caractéristiques, pour les canaux moteur et ventilateur."

fs = defaut["fs"]

# Chaque spectre d'enveloppe n'est calculé qu'une seule fois
ENVELOPPES_DEFAUT = {canal: spectre_enveloppe(defaut[canal], fs)
                     for canal in ("DE", "FE")}
Y_MAX = max(amp[(freqs >= 0) & (freqs <= X_MAX)].max()
            for freqs, amp in ENVELOPPES_DEFAUT.values())

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharey=True)
for ax, (canal, (freqs_env, amp_env)) in zip(axes, ENVELOPPES_DEFAUT.items()):
    ax.plot(freqs_env, amp_env, color="red",
            ls=STYLES["canal"][canal], lw=0.5)
    ax.set_title(f"Défaut — {canal}  (filtre "
                 f"{BANDE_RESONANCE[0]}–{BANDE_RESONANCE[1]} Hz)", loc="left")
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX * 1.05)
    for nom, val in freqs_caracteristiques(canal).items():
        couleur, ls = STYLES_RAIES[nom]
        ax.axvline(val, color=couleur, ls=ls, lw=1.8, alpha=0.5,
                   label=f"{nom} = {val:.1f} Hz")
        for k in range(2, 4):
            if k * val < X_MAX:
                ax.axvline(k * val, color=couleur, ls=ls, lw=0.8, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_ylabel("Amplitude d'enveloppe")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Fréquence (Hz)")
plt.tight_layout(); plt.show()

'''