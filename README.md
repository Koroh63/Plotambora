<div align = center>

<img src="https://cdn.discordapp.com/attachments/1150019887473901569/1201891884935680090/istockphoto-1333043586-612x6121.jpg?ex=65cb780b&is=65b9030b&hm=0d9fa0fd17d7217d976a10fed5cbcd757e27154256ebbe464c9d738cdc1a0f17&" width="1080" height="">

# **PlotAmbora** 
### L'outils de prédiction des catastrophes naturelles
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
</div>


# Présentation

Ce projet a pour but de determiner le nombre de morts/bléssés/personnes touchées des catastrophes naturelles en fonction des différentes informations que nous fournit notre Dataset.
Nous utiliserons Python et Panda pour traiter ce dataset et pour réaliser de l'apprentissage continu.

# Démarrage 🚀

### Prérequis 
- Avoir Python >= 3.11
- Avoir les librairies : 
  - sklearn 
  - pandas
  - matplotlib
  - numpy 

### Lancement 

Afin de lancer le projet il faut se situer à la source du projet et lancer la commande : 

    python3 src/main.py

# Dossier d'étude 📄

Nous avons effectué un dossier d'études afin d'expliquer notre raisonnement au cours de ce projet et les conclusions que l'ont peu apporter.  
Celui-ci est disponible ici : <a href="./Dossier_d&apos;étude.md" target="_blank">Dossier d'étude</a>

<a href="https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis?resource=download" target="_blank">Lien du Dataset ici</a>

# Développeurs 🧑‍💻

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

---
# Colonnes utilisées 📝 : 

* ### Pour ce projet nous allons utiliser les colonnes suivantes de notre dataset : 
  * **Colonne *Year* :** Cette colonne répertorie l'année où la catastrophe est arrivée
  * **Colonne *Disaster Subgroup* :** Cette colonne permet de classer les catastrophes naturelles en fonction de leur groupe (par exemple Geophysical, Meteorogical, etc...)
  * **Colonne *Disaster Type* :** Cette colonne classe les catastrophes naturelles (Tremblement de terre, Tsunami, etc...)
  * **Colonne *Disaster Subtype* :** Cette colonne précise encore plus le type de catastrophe (Chute de cendre, Cyclone Tropical, etc...)
  * **Colonne *Event Name* :** Cette colonne indique le nom de l'événement s'il a été nommé
  * **Colonne *ISO* :** Cette colonne donne le code ISO du pays où la catastrophe s'est produite (IND,CHN,etc...)
  * **Colonne *Region* :** Cette colonne donne la région ou la catastrophe naturelle s'est déroulée
  * **Colonne *Continent* :** Cette colonne indique le continent sur lequel la catastrophe a eu lieu
  * **Colonne *Origin* :** Cette colonne donne la raison de cette catastrophe s'il y en a une (exemple : un séisme qui déclenche un tsunami)
  * **Colonne *Latitude* :** Cette colonne donne la latitude précise de l'apparition de la catastrophe
  * **Colonne *Longitude* :** Cette colonne donne la longitude précise de l'apparition de la catastrophe
  * **Colonne *Local Time* :** Cette colonne donne l'heure locale du déclenchement de la catastrophe
  * **Colonne *Start Month* :**  Cette colonne donne le mois de début de la catastrophe si elle a duré sur une période
  * **Colonne *Start Day* :** Cette colonne donne le jour de début de la catastrophe
  * **Colonne *End Year* :** Cette colonne indique l'année de fin de la  catastrophe
  * **Colonne *End Month* :** Cette colonne indique le mois de fin de la catastrophe
  * **Colonne *End Day* :** Cette colonne indique le jour de la fin de catastrophe
  * **Colonne *Total Deaths* :** Cette colonne donne le nombre total de morts dues à la catastrophe naturelle
  * **Colonne *No Injured* :** Cette colonne précise le nombre de personnes blessées durant la catastrophe
  * **Colonne *No Affected* :** Cette colonne donne le nombre total de personnes affectées par cette catastrophe
  * **Colonne *No Homeless* :** Cette colonne indique le nombre total de personnes qui ont perdu leur logement à cause de cette catastrophe
  * **Colonne *Total Affected* :** Pour finir, cette colonne donne le nombre total de personnes affectées par la catastrophe