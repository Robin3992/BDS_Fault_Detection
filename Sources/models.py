"""Modèles de classification et évaluation.

Contient :
  - evaluer_modele : métriques (accuracy, balanced accuracy, F1-macro),
    rapport par classe et matrice de confusion
  - entrainer_comparer : entraîne plusieurs modèles et renvoie un tableau comparatif
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, classification_report, confusion_matrix)


def evaluer_modele(y_test, y_pred, classes=None, titre="", report= True, dispMatrice=True):
    """Affiche accuracy, balanced accuracy, F1-macro, le rapport par classe
    et trace la matrice de confusion. Renvoie la matrice cm."""
    if classes is None:
        classes = sorted(set(y_test) | set(y_pred))
    print(f"=== {titre} ===")
    print(f"Accuracy          : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Balanced accuracy : {balanced_accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-macro          : {f1_score(y_test, y_pred, average='macro'):.3f}")

    if report:
        print("\nRapport par classe :")
        print(classification_report(y_test, y_pred, labels=classes, zero_division=0))
        
    cm = None
    if dispMatrice:
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.set_title(f"Matrice de confusion — {titre}", loc="left")
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cm[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if v > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.show()

    return cm


def entrainer_comparer(modeles, X_train, y_train, X_test, y_test,
                       classes=None, tracer=True, report=True, dispMatrice=True):
    """Entraîne chaque modèle de `modeles` (dict nom -> instance sklearn),
    évalue sur le test et renvoie un DataFrame comparatif des métriques.

    Exemple :
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        modeles = {
            "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        }
        tableau = entrainer_comparer(modeles, X_train, y_train, X_test, y_test)
    """
    if classes is None:
        classes = sorted(set(y_train) | set(y_test))
    lignes = []
    for nom, clf in modeles.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        lignes.append({
            "modèle": nom,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "F1-macro": f1_score(y_test, y_pred, average="macro"),
        })
        if tracer:
            evaluer_modele(y_test, y_pred, classes=classes, titre=nom, report=report, dispMatrice=dispMatrice)
    return pd.DataFrame(lignes).round(3)