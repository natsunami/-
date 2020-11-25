---
layout: post
title: Parallel computing with Dask
subtitle: An easy way to go into Big Data
cover-img: /assets/img/dask/dask_head.jpeg
thumbnail-img: /assets/img/dask/dask-logo-2.png
share-img:
tags: [Dask,Big Data,Cloud, Cluster, Parallel,Coiled]
---

# Introduction #

Considéré comme l'un des grands défis informatiques de notre décennie, l'émergence du Big Data est souvent estimée comme l'une des innovations technologiques les plus significatives. 
Chaque jour, tout secteurs confondus, une quantité astronomique de donnée est crée et jusqu'à récemment, les ressources étaient insuffisantes pour traiter de tels volumes de données. Ainsi, le Big Data répond à la question de "comment traitrer des données massives ?"

Bien que nous ayons aujourd'hui à notre disposition des machines beaucoup plus puissantes en terme de CPU, RAM et GPU, les ressources requises pour traiter 

Comment exploiter un jeu de donnée nécessitant des ressources supérieures à celle d'une seule et unique machine ? En effet, imaginons que le jeu de donnée ait une taille supérieure à la RAM de l'ordinateur, ce dernier ne pourra pas etre stocké dans la mémoire vive et donc inutilisable. De meme,  réaliser une computation

=> Le parallèle computing 

Qu'est ce que le parallel computing 

Comme son nom l'indique, le parallele computing consiste à travailler de facon parallèle, en opposition au ttravail sériel. Tr

L'idée est relativement simple, répartir le travail entre plusieurs machines rendra la tache moins ardue. Ainsi, plusieurs machines vont se coordonner et se diviser la tâche pour la realiser.

[photo graph paralelel computing]

- workers
- client
- scheduler

Ainsi, afin de travailler dans un environnement Big Data, des technologies ont été élaborée et certaines sont aujourd'hui très connue du public. On citera le plus célèbre, [Hadoop](https://fr.wikipedia.org/wiki/Hadoop) et son  modèle de programmation [MapReduce](https://fr.wikipedia.org/wiki/MapReduce), et bien entendu [Spark](https://fr.wikipedia.org/wiki/Apache_Spark). Initié en 2012, soit 6 ans après Hadoop, Spark (Apache Spark) tend aujourd'hui à etre de plus en plus utilisé que Hadoop dans la mesure où la où Hadoop lit et écrit les fichiers en HDFS (Hadoop Distributed File System)pour traiter la donnée, Spark est plus rapide en utilisant la mémoire vive (RAM) via son concept de RDD ( Resilient Distributed Dataset). Par ailleurs, Spark possède une riche librairie pour réaliser du machine learning (SparkML) de facon distribuée. Si vous désirez connaitre les principales caractéristiques/ différences entre Hadoop et Spark, je vous invite à consulter ce lien: [Difference Between Hadoop and Spark](https://www.geeksforgeeks.org/difference-between-hadoop-and-spark/). 

Maintenant que nous avons en tete certaines notions/concepts propre au Big Data (Qu'est-ce que le Big Data, ses enjeux, comment cela fonctionne et les technologies associés), nous allons enfin pouvoir traiter de la techno au coeur de cet article, [Dask](https://dask.org).


#Présentation de Dask

## Qu'est-ce que c'est ? ##

Si vous deviez lire une chose dans cette article lisez le paragraphe suivant. Elaboré par Matthew Rocklin ([2015](https://conference.scipy.org/proceedings/scipy2015/pdfs/matthew_rocklin.pdf), Dask est une librairie écrite en python qui, comme Hadoop et Apache Spark, permet de traiter des données massives en exploitant le parallel computing. A ce stade,nous serions tenté de se demander quel est l'interet de Dask sachant qu'il existe deja des frameworks open source reconnues, validés et hautement utilisés. La réponse est relativement simple, Dask exploite le potentiel de librairies bien connues dans le milieu de la data science tels que Numpy, Pandas, Scikit-Learn. En reposant sur ce riche ecosysteme Dask permet de réaliser du traitement de données distribuée en utilisant des librairies largement connues, avec aucune, voir peu, de modifications à réaliser. Par ailleurs, Dask bénéficie en plus du soutien des communautés de cet écosystème ce qui permet d'enrichir, développer la librairie.

Après cette rapide présentation, nous allons nous interesser plus en détail aux spécificités de dask et pourquoi l'utiliser pour vos projets Big Data.

## Pourquoi Dask ? ##

Si l'on regarde la 2020 Developer Survey de Stack Overflow, on peut constater que Python est l'un des languages de programmation les plus utilisé dans le monde chez les developpeurs (44.1 %) juste derriere JavaScript (Web), HTML/CSS (Web) et SQL (base de données), ce qui en fait le language dominant en matière de programmation générale et à destination de la data.



[voir graph most used programming language]

Il est évident que la simplicité de sa syntaxe et le développement de nombreuses librairies (Numpy, Pandas, Matplotlib, Scikit-learn, Tensorflow/ Keras) en parallèle de l'engouement massif pour la data/machine learning/ A.I,  ont contribué à sa popularité. Si l'on travaille avec un volume de donnée "convenable", utiliser ces librairies ne pose aucun problème. Mais vient tot ou tard le moment où l'on cherche à travailler dans un environnement Big Data, la, ca ne fonctionne plus. La raison est que ces librairies n'ont pas été crée initialement pour etre scalable (i.e. Appliquer à une large quantité de données). La solution serait donc d'utiliser les frameworks (e.g Spark) pour réaliser du calcul distribué, mais cela implique au préalable de connaitre son fonctionnement, l'API et il se peut qu'il faille réecrire le code dans un autre language. Par exemple, si l'on veut tirer pleine performances de Spark il faudeait que le code soit écrit en Scala et non pas en python dans la mesure où Scala est 10x plus rapide que python pour le traitement et l'analyse de données. Ce processus pouvant etre fastidieux et frustrant, il serait préférable de travailler avec les librairies scalées propres à python. Et c'est exactement ce que nous permet Dask.

Dask va donc scaler entre autre numpy, pandas, sckikit-learn et cloner leur API afin de fournir un environment familier et réduire au maximum la réecriture du code ( Notes: Il est important de noter que meme si Dask copie les API des librairies data les plus connues de python, il n'implémente pas encore aujourd'hui tous leurs contenus, ce qui nécessite parfois de faire davantage "from scratch").

Enfin, Dask propose 2 grandes facons de réaliser du calcul distribué. La première étant bien entendu de scaler le travail entre plusieurs machines, mais Dask permet également de distribuer les computations à l'echelle d'une seule machine. En effet, avec Dask il est tout à fait possible de scaler les taches entre les coeurs du processeur. Il est important de comprendre qu' un cluster de machines n'est pas forcément la meilleur décision pour travailler dans un envirnement Big Data. d'une part, les ordinateurs utilisant Dask et ayant des composants relativement récents ( CPU multi-coeurs dernière génération, GPU, RAM allant de 16 à 64gb, SSD) permettent de travailler avec des jeux de données de plus de 100gb sans grande difficulté. D'autre part, travailler en local évite bon nombre de contraintes telles que le fait d'etre limité par la bande passante ( dans un cluster les données doivent circuler dans le réseau), mais aussi le fait de devoir gérer des images docker plutot que de travailler avec un software environment local par exemple.


## Comment Dask fonctionne ? ##

Après avoir présenté Dask je pense que vous comprenez désormais son interet. Au sein de cette partie nous allons voir plus en détails le fonctionnement interne de Dask.

Tout d'abord,il faut savoir de quoi est constitué un réseau distribué Dask. En effet, ce dernier repose sur 3 concepts fondamentaux:

- Le scheduler: 

Comme son nom l'indique, le rôle du scheduler est de planifier les taches de facon distribué. Ce dernier assimile les taches à effectuer sous la forme de graph (Task graph) crée par Dask au préalable, et va  demander ensuite aux workers de realiser ces taches.

[scheduler.png]

- Le client:

Le client est tout simplement ce qui va nous permettre de nous connecter au cluster dask. Après avoir créer le cluster, on initialise le client en lui passant l'adresse du scheduler. Le client s'enregistre en tant que scheduler par défaut, et permet d'exécuter toutes les collections de dask (dask.array, dask.bag, dask.dataframe et dask.delayed).

- Les Workers:

Si l'on décide d'utiliser dask sur une seule machine, les workers sont les coeurs du processeur tandis que dans un cluster ce sont les les différentes machines. Les workers recoivent les informations du scheduler et exécutent les tâches. Ils rapportent au scheduler lorsqu'ils ont terminé, tout en conservant les résultats stockés dans les workers où il ont été calculé.

[photo dask distributed graph]

- Dask Distributed Local:

```py
from dask.distributed import Client
client = Client(cluster)
print('Dashboard:', client.dashboard_link)
```

- Dask Distributed cluster: 

-Graph worker/ ( client/ workers/scheduler)
-creat a cluster distribué (local and 
-ex avec dashboard
-Chunks/partitions

-array
-dataframe
-machine learning


## Difference avec Spark ?
Spark: Spark is a newer project, initially developed in 2012, at the AMPLab at UC Berkeley. It’s a top-level Apache project focused on processing data in parallel across a cluster, but the biggest difference is that it works in memor

Whereas Hadoop reads and writes files to HDFS, Spark processes data in RAM using a concept known as an RDD, Resilient Distributed Dataset

## Dask en exemples ##

- Creer un cluster local
- Creer un cluster sur le cloud avec coiled (montrer le dashboard)
- try dask (dask dataframe, processing, dask ml(xgboost)

