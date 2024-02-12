# Dossier d'étude du Projet Plotambora 

### Introduction 

Plotambora est un projet de prédiction du nombre de décès lors des catastrophes naturelles futures. À ces fins, nous utilisons des techniques de traitement des données, de visualisation et de modélisation prédictive.

### Etapes : 

- Importer et nettoyer les données des catastrophes passées.
- Étudier les données afin d'en comprendre les tendances et de décider des paramètres importants.
- Entrainer différents modèles de prédictions 
- Évaluer les résultats et décider de leur pertinence pour choisir le meilleur. 

### Implémentation 

Nous avons choisi d'implémenter de différentes façons nos prévisions, l'une complétant l'autre : 
- La première est une classification afin de déterminer si des morts ont été enregistrés.
- La seconde est une régression qui déterminera le nombre de morts enregistrés.

## 1 - Les données 

Afin d'effectuer ces prévisions nous avons sélectionner certaines données plus pertinentes que les autres, pour ce faire nous avons utiliser des histogrammes et graphiques et avons pu déterminer lesquels étaient pertinents : 
- Les colonnes de Localisation
- Les colonnes de Date 
- Les colonnes d'identification des catastrophes 
- Les colonnes de bilans humains 

#### # Gestion de la colonne 'Total Deaths' 

'Total Deaths' étant la colonne à prévoir, la gestion de celle-ci a un impact majeur sur la pertinence de notre modèle d'aide à la décision. Nous avons effectué les modifications suivantes pour une plus grande précision.
    
- **Remplir les valeurs manquantes.** Pour ce faire nous avions plusieurs possibilités, nous avons testé de remplir par la moyenne des valeurs déjà présentes, par la moyenne des valeurs groupés par différentes colonnes, pour finir par choisir la remplissage par zéros : Aucune ligne ne contenait de zéros, les valeurs non-remplis correspondent donc à 0.

-  **Exclure les valeurs extrèmes.** Certaines catastrophe ont eu de terribles conséquences et ont donc un rôle négatif sur la précision du modèle. Nous avons dû choisir d'exclure le quantile à 0.8 pour avoir le meilleur rapport entre exclure un minimum de lignes et avoir une meilleure précision.

La colonne est alors prête à entraîner le jeu de données.


#### # Gestion de la colonne 'Duration' 

'Duration' était une colonne manquante du DataSet, mais pouvant montrer son importance, car elle a pu augmenter la précision des modèles. Pour la créer nous avons effectuer les actions suivantes : 

- **Création des colonnes Start Date et End Date** à partir de Year,Start Month,Start Day,End Year,End Month, et End Day.

- **Remplissage des valeurs manquantes** par des valeurs par défaut qui définiront une duration égale à 1. 1 car les dates manquantes représentent généralement une catastrophe qui n'a duré qu'un instant. 

- **Calcul de Duration à partir de Start Date et End Date** pour obtenir une durée en jours.


## Gestion de la colonne 'Lethality'

Afin d'effectuer la classification pour déterminer si des morts ont été enregistrées, il nous a fallu ajouter cette colonne représentant cette donnée.

- **Une valeur booléenne déterminée** à partir de Total Deaths : si > 0 => True else => False
- **Convertir cette valeur booléenne en Int** pour le modèle d'apprentissage.

## Gestion des autres colonnes

Les autres colonnes représentant des valeurs en chaîne de caractères ont été mappées avec des valeurs en entier afin que le modèle les comprenne.
Enfin, les valeurs inutiles ont été rejetées du DataSet.

## 2 - Classification

La première valeur à déterminer est celle qui désigne s'il y a eu des victimes dans les catastrophes. Dans le cas contraire, l'appel au second modèle sera donc évité.
Pour ce faire, nous avons choisi en premier 2 modèles d'apprentissage :
- **SVC (Support Vector Classification)** : cherche à trouver un hyperplan qui sépare de manière optimale les différentes classes en maximisant la marge entre les points les plus proches de chaque classe, appelés vecteurs de support.
- **Régression Logistique** : modélise la probabilité qu'un exemple appartienne à une classe donnée en utilisant une fonction logistique pour estimer les coefficients de chaque variable indépendante.

Afin de déterminer le modèle à utiliser, nous effectuons les actions suivantes :
- Paramétrage des modèles
- Entraînement des modèles
- Prédiction des valeurs
- Affichage de la précision

Exemple :

    ----------- Classification de la Léthalité -----------
    SVC Accuracy Score : 0.6298701298701299
    Logistic Regression Accuracy Score : 0.6298701298701299

On peut voir que le modèle n'est pas très précis au niveau de la classification.  
Nous obtenons alors que le modèle SVC est légèrement plus précis que la régression Logistique.

### Le modèle SVC sera alors choisi pour la Classification.

## 3 - Régression 

Après avoir déterminé la présence de victimes dans les catastrophes, nous pouvons passer à en estimer le nombre par la régression.

Pour ce faire, nous avons choisi en premier 3 modèles d'apprentissage :
- **Régression Linéaire** : Cherche à établir une relation linéaire entre une variable dépendante et plusieurs variables indépendantes en ajustant un modèle qui minimise les erreurs de prédiction.
- **Lasso** : Ajoute une pénalité L1 à la fonction de coût de la régression linéaire, ce qui favorise la réduction du nombre de coefficients non nuls dans le modèle, ce qui peut conduire à la sélection de variables.
- **ElasticNet** : Combine à la fois les pénalités L1 et L2 dans la fonction de coût de la régression linéaire, ce qui permet de bénéficier des avantages de la sélection de caractéristiques du Lasso tout en atténuant certains de ses inconvénients, comme la tendance à sélectionner un seul prédicteur lorsque plusieurs prédicteurs sont hautement corrélés.

Encore une fois, afin de déterminer le meilleur modèle, nous effectuons les actions suivantes: 
- Paramétrage des modèles 
- Entraînement des modèles 
- Prédiction des valeurs 
- Affichage de la précisions 

Exemple :


    ----------- Régression des totaux de morts -----------
    -- Linear Regression : 
    Mean Squared Error - Linear Regression: 164.3541836165224
    R2 Score - Linear Regression: 0.1332531970469013
    -- Lasso : 
    Mean Squared Error - Lasso: 167.19754925547417
    R2 Score - Lasso: 0.1182582755732966
    -- ElasticNet : 
    Mean Squared Error - ElasticNet: 166.75633415225224
    R2 Score - ElasticNet: 0.12058509057561206

La régression a aussi des valeurs qui sont peu précises malgré les actions sur les données.
On peut voir, par un MSE plus petit et un R2 plus proche de 0, que la régression linéaire est la plus efficace.

### Le modèle Régression Linéaire sera alors choisi pour la Régression.

---

### Conclusion 

Nous pouvons voir que malgré de nombreuses actions afin d'améliorer la précision des modèles, celle-ci reste basse. Cela s'explique facilement par le fait qu'il est difficile de prévoir ce type de données très aléatoires. 

Cependant, la précision pourrait être encore améliorée en effectuant les actions suivantes : 

- Trouver les paramètres optimales pour chaque modèles grâce à la fonction GridSearchCV qui vient tester les différents paramètres avec de nombreuses valuers.
- Trouver des modèles plus efficaces comme : 
    - Les arbres de décisions 
    - Les forêts aléatoires 
    - Régression PLS
    - Régressio Ridge
    - etc ...

--- 

#### Auteurs :

- Corentin RICHARD : corentin.richard@etu.uca.fr
- Dorian HODIN : dorian.hodin@etu.uca.fr

<div align="center">
<a href = "https://codefirst.iut.uca.fr/git/corentin.richard">
<img src="https://codefirst.iut.uca.fr/git/avatars/4372364870f18ab9104f13222fa84d2e?size=870" width="50" >
</a>
<a href = "https://codefirst.iut.uca.fr/git/dorian.hodin">
<img src="https://codefirst.iut.uca.fr/git/avatars/d6f97dbdf66352b0b66685e144aa1ee5?size=870" width="50" >
</a>
</div>
