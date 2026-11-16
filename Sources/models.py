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



# 3. Convolution 1 dimension : auto-encodeur
import tensorflow as tf
import keras
from keras import layers

def conv1d_ae(taille):
    Input = keras.Input(shape=(taille, 1))
    # Encodeur : 2 poolings
    x = layers.Conv1D(32, 7, activation="relu", padding="same")(Input) 
    x = layers.MaxPooling1D(2, padding="same")(x)                       
    x = layers.Conv1D(16, 5, activation="relu", padding="same")(x)      
    x = layers.MaxPooling1D(2, padding="same")(x)                       
    latent = layers.Conv1D(8, 3, activation="relu", padding="same")(x)
    # Decodeur : miroir exact
    x = layers.Conv1D(16, 5, activation="relu", padding="same")(latent)  
    x = layers.UpSampling1D(2)(x)                                         
    x = layers.Conv1D(32, 7, activation="relu", padding="same")(x)        
    x = layers.UpSampling1D(2)(x)                                       
    sortie = layers.Conv1D(1, 7, padding="same")(x)           
    modele = keras.Model(Input, sortie)
    modele.compile(optimizer="adam", loss="mse")
    return modele












## FONCTION FNR (False Negative Rate) POUR LA CLASSE DEFAUT


import numpy as np
import pandas as pd

## FONCTION FNR (False Negative Rate) POUR LA CLASSE DEFAUT

classe_saine = "Sain"  # à remplacer par le nom exact dans la colonne defaut

def F_FNR(y_true, y_pred, classe_saine):
    """Calcule le FNR : part des vrais défauts prédits comme sains."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred n'ont pas le même nombre de lignes.")

    if classe_saine not in y_true:
        raise ValueError(
            f"Classe saine {classe_saine!r} introuvable. Classes présentes : {list(np.unique(y_true))}"
        )

    defauts_reels = y_true != classe_saine
    nb_defauts = defauts_reels.sum()

    if nb_defauts == 0:
        raise ValueError("y_true ne contient aucun défaut réel.")

    nb_manques = (y_pred[defauts_reels] == classe_saine).sum()
    fnr = nb_manques / nb_defauts

    detail = pd.DataFrame({
        "réels": pd.Series(y_true[defauts_reels]).value_counts(),
        "prédits sain": pd.Series(y_true[defauts_reels & (y_pred == classe_saine)]).value_counts(),
    }).fillna(0).astype(int)

    print(detail)
    print(f"\nFNR = {nb_manques} / {nb_defauts} = {fnr:.3f}")

    return fnr







import numpy as np
from sklearn.metrics import fbeta_score, make_scorer

CLASSE_SAINE = "Sain"  # à remplacer par le nom exact dans la colonne defaut


def F_TAUX_DEFAUTS_MANQUES(y_true, y_pred, classe_saine=CLASSE_SAINE):
    """Part des vrais défauts prédits comme sains."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    defauts_reels = y_true != classe_saine
    return (y_pred[defauts_reels] == classe_saine).mean()


def F_TAUX_FAUSSES_ALERTES(y_true, y_pred, classe_saine=CLASSE_SAINE):
    """Part des vrais éléments sains prédits comme défectueux."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sains_reels = y_true == classe_saine
    return (y_pred[sains_reels] != classe_saine).mean()


def F_F2_DETECTION(y_true, y_pred, classe_saine=CLASSE_SAINE):
    """Score F2 sur la détection : défectueux ou sain."""
    return fbeta_score(
        np.asarray(y_true) != classe_saine,
        np.asarray(y_pred) != classe_saine,
        beta=2,
    )


score_f2_detection = make_scorer(F_F2_DETECTION)











## FONCTION RANDOM FOREST 



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import matplotlib.pyplot as plt

def F_RANDOM_FOREST(X_train, X_test, y_train, y_test, groupes_train, n_plis=3, graine=42):
    """Optimise une Random Forest et l'évalue sur X_test, y_test."""
    grille = {
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", 2, 3],
        "n_estimators": [100, 300, 500],
    }

    recherche = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=graine),
        param_grid=grille,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine),
        n_jobs=-1,
    )
    recherche.fit(X_train, y_train, groups=groupes_train)

    meilleur_modele = recherche.best_estimator_
    y_pred = meilleur_modele.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average="macro")

    print("Meilleurs paramètres :", recherche.best_params_)
    print(f"F1 macro en validation : {recherche.best_score_:.3f}")
    print(f"F1 macro sur le test : {f1_test:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    
    print("Taux de défauts manqués :", round(F_TAUX_DEFAUTS_MANQUES(y_test, y_pred), 3))
    print("Taux de fausses alertes :", round(F_TAUX_FAUSSES_ALERTES(y_test, y_pred), 3))
    print("F2 détection :", round(F_F2_DETECTION(y_test, y_pred), 3))
    
    print("FNR (False Negative Rate) :")
    fnr_rf = F_FNR(y_test, y_pred, classe_saine="Sain")
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.title("Matrice de confusion : Random Forest")
    plt.show()

    return meilleur_modele, recherche.best_params_, f1_test




## FONCTION LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def F_LOGISTIC_REGRESSION(X_train, X_test, y_train, y_test, groupes_train, n_plis=3, graine=42):
    """Optimise une régression logistique et l'évalue sur X_test, y_test."""
    grille = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=graine),
    )

    recherche = GridSearchCV(
        pipeline,
        param_grid=grille,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine),
        n_jobs=-1,
    )
    recherche.fit(X_train, y_train, groups=groupes_train)

    meilleur_modele = recherche.best_estimator_
    y_pred = meilleur_modele.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average="macro")

    print("Meilleurs paramètres :", recherche.best_params_)
    print(f"F1 macro en validation : {recherche.best_score_:.3f}")
    print(f"F1 macro sur le test : {f1_test:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Taux de défauts manqués :", round(F_TAUX_DEFAUTS_MANQUES(y_test, y_pred), 3))
    print("Taux de fausses alertes :", round(F_TAUX_FAUSSES_ALERTES(y_test, y_pred), 3))
    print("F2 détection :", round(F_F2_DETECTION(y_test, y_pred), 3))
    
    print("FNR (False Negative Rate) :")
    fnr_rf = F_FNR(y_test, y_pred, classe_saine="Sain")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.title("Matrice de confusion : Logistic Regression")
    plt.show()

    return meilleur_modele, recherche.best_params_, f1_test



## FONCTION SVM Noyau RBF

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def F_SVM_noyau_RBF(X_train, X_test, y_train, y_test, groupes_train, n_plis=3, graine=42):
    """Optimise un SVM à noyau RBF et l'évalue sur X_test, y_test."""
    grille = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.1, 1],
    }

    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=graine),
    )

    recherche = GridSearchCV(
        pipeline,
        param_grid=grille,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine),
        n_jobs=-1,
    )
    recherche.fit(X_train, y_train, groups=groupes_train)

    meilleur_modele = recherche.best_estimator_
    y_pred = meilleur_modele.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average="macro")

    print("Meilleurs paramètres :", recherche.best_params_)
    print(f"F1 macro en validation : {recherche.best_score_:.3f}")
    print(f"F1 macro sur le test : {f1_test:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Taux de défauts manqués :", round(F_TAUX_DEFAUTS_MANQUES(y_test, y_pred), 3))
    print("Taux de fausses alertes :", round(F_TAUX_FAUSSES_ALERTES(y_test, y_pred), 3))
    print("F2 détection :", round(F_F2_DETECTION(y_test, y_pred), 3))
    print("FNR (False Negative Rate) :")
    fnr_rf = F_FNR(y_test, y_pred, classe_saine="Sain")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.title("Matrice de confusion : SVM Noyau RBF")
    plt.show()

    return meilleur_modele, recherche.best_params_, f1_test





## FONCTION GRADIENT_BOOSTING _HistGradientBoostingClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
import matplotlib.pyplot as plt


def F_GRADIENT_BOOSTING(X_train, X_test, y_train, y_test, groupes_train, n_plis=3, graine=42):
    """Optimise un gradient boosting et l'évalue sur X_test, y_test."""
    grille = {
        "learning_rate": [0.03, 0.1, 0.3],
        "max_iter": [100, 300],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [10, 20, 50],
    }

    recherche = GridSearchCV(
        HistGradientBoostingClassifier(
            class_weight="balanced", early_stopping=False, random_state=graine
        ),
        param_grid=grille,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine),
        n_jobs=-1,
    )
    recherche.fit(X_train, y_train, groups=groupes_train)

    meilleur_modele = recherche.best_estimator_
    y_pred = meilleur_modele.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average="macro")

    print("Meilleurs paramètres :", recherche.best_params_)
    print(f"F1 macro en validation : {recherche.best_score_:.3f}")
    print(f"F1 macro sur le test : {f1_test:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Taux de défauts manqués :", round(F_TAUX_DEFAUTS_MANQUES(y_test, y_pred), 3))
    print("Taux de fausses alertes :", round(F_TAUX_FAUSSES_ALERTES(y_test, y_pred), 3))
    print("F2 détection :", round(F_F2_DETECTION(y_test, y_pred), 3))
    
    print("FNR (False Negative Rate) :")
    fnr_rf = F_FNR(y_test, y_pred, classe_saine="Sain")
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.title("Matrice de confusion : Gradient Boosting")
    plt.show()

    return meilleur_modele, recherche.best_params_, f1_test




## FONCTION K-NN  (K-Nearest Neighbors )

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def F_K_NN(X_train, X_test, y_train, y_test, groupes_train, n_plis=3, graine=42):
    """Optimise un k-NN et l'évalue sur X_test, y_test."""
    grille = {"kneighborsclassifier__n_neighbors": [3, 5, 11, 21]}

    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())

    recherche = GridSearchCV(
        pipeline,
        param_grid=grille,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=n_plis, shuffle=True, random_state=graine),
        n_jobs=-1,
    )
    recherche.fit(X_train, y_train, groups=groupes_train)

    meilleur_modele = recherche.best_estimator_
    y_pred = meilleur_modele.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average="macro")

    print("Meilleurs paramètres :", recherche.best_params_)
    print(f"F1 macro en validation : {recherche.best_score_:.3f}")
    print(f"F1 macro sur le test : {f1_test:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Taux de défauts manqués :", round(F_TAUX_DEFAUTS_MANQUES(y_test, y_pred), 3))
    print("Taux de fausses alertes :", round(F_TAUX_FAUSSES_ALERTES(y_test, y_pred), 3))
    print("F2 détection :", round(F_F2_DETECTION(y_test, y_pred), 3))
    
    print("FNR (False Negative Rate) :")
    fnr_knn = F_FNR(y_test, y_pred, classe_saine="Sain")
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.title("Matrice de confusion : K-NN_Nearest Neighbors")
    plt.show()

    return meilleur_modele, recherche.best_params_, f1_test







# " APPEL DES FONCTIONS: 
                    


#  appeler fonction TRAIN TEST SPLIT

#X_train, X_test, y_train, y_test, groupes_train = F_TRAIN_TEST_SPLIT(table_indicateurs_classe)

#X_train.head()


#  appeler fonction RANDOM FOREST

#modele_rf, params_rf, f1_rf = F_RANDOM_FOREST(X_train, X_test, y_train, y_test, groupes_train)


#  appeler fonction LOGISTIC REGRESSION

# modele_lr, params_lr, f1_lr = F_LOGISTIC_REGRESSION(X_train, X_test, y_train, y_test, groupes_train)



#  appeler fonction SVM Noyau RBF

#modele_svm, params_svm, f1_svm = F_SVM_noyau_RBF(X_train, X_test, y_train, y_test, groupes_train)


#  appeler fonction GRADIENT_BOOSTING

# modele_gb, params_gb, f1_gb = F_GRADIENT_BOOSTING(X_train, X_test, y_train, y_test, groupes_train)


#  appeler fonction K-NN ()

#modele_knn, params_knn, f1_knn = F_K_NN(X_train, X_test, y_train, y_test, groupes_train)



