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

Ainsi, afin de travailler dans un environnement Big Data, des technologies ont été élaborée et certaines sont aujourd'hui très connue du public. On citera le plus célèbre, [Hadoop](https://fr.wikipedia.org/wiki/Hadoop) et son  modèle de programmation [MapReduce](https://fr.wikipedia.org/wiki/MapReduce), et bien entendu [Spark](https://fr.wikipedia.org/wiki/Apache_Spark). Initié en 2012, soit 6 ans après Hadoop, Spark (Apache Spark) tend aujourd'hui à etre plus utilisé que Hadoop dans la mesure où la où Hadoop lit et écrit les fichiers en HDFS (Hadoop Distributed File System)pour traiter la donnée, Spark est plus rapide en utilisant la mémoire vive (RAM) via son concept de RDD ( Resilient Distributed Dataset). Par ailleurs, Spark possède une riche librairie pour réaliser du machine


#Présentation de Dask

## Comment Dask fonctionne ? ##

- familiar API
-Scales up to cluster / down to machine 
-Graph worker/


## Difference avec Spark ?
Spark: Spark is a newer project, initially developed in 2012, at the AMPLab at UC Berkeley. It’s a top-level Apache project focused on processing data in parallel across a cluster, but the biggest difference is that it works in memor

Whereas Hadoop reads and writes files to HDFS, Spark processes data in RAM using a concept known as an RDD, Resilient Distributed Dataset
