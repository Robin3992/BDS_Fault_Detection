# Proposition : rapports Quarto et règles de travail sur le dépôt BDS_Fault_Detection

*Document à discuter ensemble : commentez directement dans la pull request.*
   >>> Lors de la merge pull request ?? Pas encore clair pour moi.
   
## 1. Pourquoi cette proposition

Nous commençons à produire plusieurs analyses sur les données CWRU (indicateurs temporels, FFT, analyse d'enveloppe…). Pour bien travailler à plusieurs, il nous faut :

- des **rapports reproductibles**, que chacun peut régénérer à l'identique ;
- des **relectures faciles** dans GitHub, en voyant ligne par ligne ce qui a changé ;
- **peu de conflits** quand deux personnes modifient le dépôt ;
- des **règles communes** de nommage et de rangement.

Je propose d'utiliser **Quarto** pour les analyses partagées.Un fichier `.qmd` contient le texte, le code et les options de mise en forme. Il produit un rapport HTML, PDF ou Word, avec ou sans le code visible.

## 2. Essai en 15 minutes

Un rapport d'exemple est disponible : `Codes/rapport_IR.qmd` (comparaison sain / défaut de bague intérieure).

1. **Installer Quarto** depuis quarto.org, puis l'extension **Quarto** dans VS Code. Redémarrer VS Code.
2. **Vérifier l'installation** dans le terminal :
   ```
   quarto check
   ```
   Si Jupyter n'est pas trouvé, lancer `pip install jupyter`.
3. **Récupérer la branche d'essai** : dans VS Code, faire **Pull**, puis cliquer sur le nom de la branche en bas à gauche et choisir `origin/rapport-quarto`.
4. **Ouvrir** `Codes/rapport_IR.qmd`, puis cliquer sur **Preview** (`Ctrl+Maj+K`). Le rapport s'affiche à côté et se met à jour à chaque enregistrement.
5. **Tester** : modifier une phrase, passer `echo: false` à `true`, enregistrer et observer.
6. *(Facultatif)* Pour obtenir un PDF :
   ```
   quarto install tinytex
   quarto render Codes/rapport_IR.qmd --to pdf
   ```
      >>> Done

## 3. Règles de travail proposées

1. **Branches et pull requests.** Personne ne pousse directement sur `main`. Chaque travail se fait sur une branche, puis passe par une pull request relue par un collègue avant fusion.
   >>> Ok, mais je ne maitrise pas encore bien git.
2. **Notebooks pour explorer, Quarto pour partager.** Chacun peut explorer librement dans des notebooks. Une analyse destinée à l'équipe est rédigée en `.qmd`.
   >>> OK
3. **On versionne les sources, pas les rendus.** Les fichiers HTML produits par Quarto et les dossiers techniques sont ignorés par Git. Un PDF final peut être déposé volontairement dans `Docs/` à une étape clé.
   >>> Par source, tu entends les analyses en '.qmd' et rendus les '.pdf' généré ? >>> Oui.
4. **Un environnement commun.** Un fichier `requirements.txt` à la racine liste les bibliothèques nécessaires. Chacun l'installe avec `pip install -r requirements.txt`.
   >>> Cela implique une synchro de nos environnements.
5. **Nommage des fichiers.** Préfixe numérique, puis sujet, sans espaces ni accents : `05_robustesse_charges.qmd`. De préférence, un fichier par analyse et par auteur, pour éviter de modifier le même fichier à deux.
   >>> Ok, mais ne doit-on pas ajouter les initiales au nommage, puisque c'est par auteur ?
6. **Reproductibilité avant chaque commit.** Pour un notebook : *Restart* puis *Run All*. Pour un `.qmd` : un rendu complet sans erreur.
   >>> OK
7. **Pas de réglages personnels dans le dépôt.** Le dossier `.vscode/` est déjà ignoré, car il contient des chemins propres à chaque poste.
   >>> OK


## 4. Points à décider ensemble

- [ ] **Les données dans le dépôt.** Aujourd'hui, les fichiers `.mat` sont versionnés. Avec toutes les charges et toutes les tailles de défaut, cela représentera plus d'une centaine de Mo. Les garde-t-on dans le dépôt, ou ajoute-t-on un script qui les télécharge depuis le site CWRU ?
   >>> Impossible des les mettres dans git, trop lourd et pas pertinent.
   >>> J'ai un script pour télécharger les fichiers en local, pour avoir un accès facile et rapide au fichiers
   >>> Je pense que nous avons besoin d'un code pour récupérer ces dont en direct depuis CWRU, mais la vitesse de transfert du site est faible et instable.
- [ ] **L'emplacement des rapports.** Dans `Codes/`, ou dans un dossier dédié `Rapports/` ?
   >>> Rapports/
- [ ] **Les noms de dossiers de données.** Harmoniser `normal` et `12k-DE`, par exemple tout en minuscules : `normal`, `12k_de`.
   >>> Arborescence déjà proposé via slack avec les fichiers associés. À décider ensemble.
- [ ] **La relecture.** Qui relit et fusionne les pull requests ? Un relecteur fixe, ou n'importe quel membre de l'équipe ?
   >>> Je suppose qu'un relecteur fixe aurait un meilleur suivi, ça serait donc plus safe pour le projet. À voir ensemble.


## 5. Prochaine étape
Merci d'essayer la partie 2, puis de laisser vos remarques en commentaire de la pull request (accord, désaccord ou modification sur chaque point). Une fois les règles validées, je mettrai à jour ce document et le `README`.
