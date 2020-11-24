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

Si vous chezchez à aller à l'essentiel, alors retenez le paragraphe suivant. Elaboré par Matthew Rocklin ([2015](https://conference.scipy.org/proceedings/scipy2015/pdfs/matthew_rocklin.pdf), Dask est une librairie écrite en python qui, comme Hadoop et Apache Spark, permet de traiter des données massives en exploitant le parallel computing. A ce stade,nous serions tenté de se demander quel est l'interet de Dask sachant qu'il existe deja des frameworks open source reconnues, validés et hautement utilisés. La réponse est relativement simple, Dask exploite le potentiel de librairies bien connues dans le milieu de la data science tels que Numpy, Pandas, Scikit-Learn. En reposant sur ce riche ecosysteme Dask permet de réaliser du traitement de données distribuée en utilisant des librairies largement connues, avec aucune, voir peu, de modifications à réaliser. Par ailleurs, Dask bénéficie en plus du soutien des communautés de cet écosystème ce qui permet d'enrichir, développer la librairie.



## Comment Dask fonctionne ? ##

- familiar API
-Scales up to cluster / down to machine 
-Graph worker/


## Difference avec Spark ?
Spark: Spark is a newer project, initially developed in 2012, at the AMPLab at UC Berkeley. It’s a top-level Apache project focused on processing data in parallel across a cluster, but the biggest difference is that it works in memor

Whereas Hadoop reads and writes files to HDFS, Spark processes data in RAM using a concept known as an RDD, Resilient Distributed Dataset

## Dask en exemples ##

- Creer un cluster local
- Creer un cluster sur le cloud avec coiled (montrer le dashboard)
- try dask (dask dataframe, processing, dask ml(xgboost)

