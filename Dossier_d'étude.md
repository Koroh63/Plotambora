# Dossier d'étude du Projet Plotambora 

## Introduction 

Plotambora est un projet de prédiction du nombre de décès lors des catastrophes naturelles futures. A ces fins nous utilisons techniques de traitement des données, de visualisation et de modélisation prédictive.

## Etapes : 

- Importer et Nettoyer les données des catastrophes passées.
- Etudier les données afin d'en comprendre les tendances et décider des paramètres importants.
- Entrainer différents modèles de prédictions 
- Evaluer les résultats et décider de leur pertinence pour choisir le meilleur. 

## Implémentation 

Nous avons choisi d'implémenter de différentes façon nos prévisions, l'une complétant l'autre : 
- La première est une classification afin déterminer si des morts ont été enregistrées 
- La seconde est une régression qui déterminera le nombre de morts enregistrés 

### 1 - Les données 

Afin d'effectuer ces prévisions nous avons sélectionner certaines données plus pertinentes que les autres, pour ce faire nous avons utiliser des historigrammes et graphiques et avons pu déterminer lesquels étaient pertinents : 
- Les colonnes de Localisation
- Les colonnes de Date 
- Les colonnes d'identification des catastrophes 
- Les colonnes de bilans humains 

#### Gestion de la colonne 'Total Deaths' 

Etant la colonne à prévoir, la gestion de celle-ci a un impact majeur sur la pertinence de notre modèle d'aide à la décision