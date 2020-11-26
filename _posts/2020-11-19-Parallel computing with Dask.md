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

Selon l'édition 2019 du Statista Digital Economy Compass, 33 zettaoctets (Zo) de données, soit 2.64e+14 gigabits(Gb), auraient été créées dans le monde l'année dernière. Si vous avez du mal à vous représenter un tel volume, c'est tout à fait normal. Peut-être que l'infographie suivante pourrait vous éclairer.

![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/data_volume_2018.jpg)


_Fig 1. Volume de données stocké au cours de l'année 2018_
_(Source: Statista Digital Economy Compass 2019)_


Le big data répond donc à la question suivante: **Comment traiter des données massives ?**

Bien que nous ayons aujourd'hui à notre disposition des machines dont la puissance en terme de CPU, RAM et GPU ne cesse de croître, la quantité de données évolue plus rapidement que les capacités des ordinateurs à traiter ces données. C'est pourquoi de nouvelles solutions ont dû émerger afin de répondre à cette problématique. La réponse fût le _"parallel computing"_. En effet, si une machine s'avère insuffisante pour faire face à ces données massives, alors plusieurs y parviendront. 

Comme son nom l'indique, le "parallel computing" consiste à tout simplement à répartir le travail entre plusieurs machines et l'éxecuter simultanément. La logique est évidente, plutôt que de faire peser toute la charge sur un élément, la répartir entre plusieurs éléments va rendre son éxecution moins "fatiguante" et plus rapide.

De nouvelles technologies ont ainsi vu le jour afin de travailler dans un environnement Big Data. Aujourd'hui, certaines sont très connues du public. On citera notamment le précurseur,[Hadoop](https://fr.wikipedia.org/wiki/Hadoop) et son  modèle de programmation [MapReduce](https://fr.wikipedia.org/wiki/MapReduce), et bien entendu,[Spark](https://fr.wikipedia.org/wiki/Apache_Spark). Initié en 2010, soit 4 ans après Hadoop, Spark (Apache Spark) tend aujourd'hui à être le framework leader pour le traitement de données massives. Brièvement, cela s'explique dans la mesure où la où Hadoop lit et écrit les fichiers en HDFS (Hadoop Distributed File System) pour traiter la donnée, Spark est plus rapide en utilisant la mémoire vive (RAM) via son concept de RDD ( Resilient Distributed Dataset). Par ailleurs, Spark possède une riche librairie de machine learning (SparkML). Si vous désirez connaitre les principales caractéristiques/différences entre Hadoop et Spark, je vous invite à consulter ce lien: [Difference Between Hadoop and Spark](https://www.geeksforgeeks.org/difference-between-hadoop-and-spark/). 

Cette petite introduction sur le Big Data étant faite, nous allons pouvoir plonger au coeur du sujet de cet article, le parallel computing avec [Dask](https://dask.org).


# Présentation de Dask #

## Qu'est-ce que c'est ? ##

Si vous deviez retenir une chose sur le sujet de cet article, cela serait sans doute le paragraphe suivant:

Elaboré par Matthew Rocklin ([2015](https://conference.scipy.org/proceedings/scipy2015/pdfs/matthew_rocklin.pdf), Dask est une librairie écrite en Python qui, comme Hadoop et Apache Spark, permet de traiter des données massives en exploitant le parallel computing. À ce stade, vous seriez tenté de vous demander quel serait l'intêret d'utiliser Dask sachant qu'il existe deja des frameworks open source reconnues, validés par une large communauté et utilisés dans le monde. La réponse est relativement simple, Dask exploite le potentiel de librairies bien connues dans le milieu de la data tels que Numpy, Pandas, Scikit-Learn. En s'appuyant sur ce riche écosystème Dask permet de réaliser du traitement de données distribué en copiant des librairies connues, avec aucune, voir peu de modifications de code à réaliser. Qui plus est, Dask bénéficie du soutien des communautés de cet écosystème, ce qui contribue fortement à son développement.

## Pourquoi Dask ? ##

Si l'on regarde la [2020 Developer Survey](https://insights.stackoverflow.com/survey/2020) de Stack Overflow, on peut constater que Python est l'un des langages de programmation les plus utilisés dans le monde chez les developpeurs (44.1 %) juste derriere JavaScript (Web), HTML/CSS (Web) et SQL (base de données). De ce fait, Python est le langage dominant en matière de programmation générale et à destination de la data.



![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/most_used_languages.png)

_Fig 2. Langages de programmation les plus utilisés dans le monde en 2020_
_(Source: Stack Overflow 2020 Developer Survey)_


Il est évident que la simplicité de sa syntaxe et le développement de nombreuses librairies (Numpy, Pandas, Matplotlib, Scikit-learn, Tensorflow/ Keras) en parallèle de l'engouement massif pour la data/machine learning/ A.I,  ont contribué à sa popularité. Si l'on travaille avec un volume de donnée "convenable", utiliser ces librairies ne pose aucun problème. Mais vient tot ou tard le moment où l'on cherche à travailler dans un environnement Big Data, la, ca ne fonctionne plus. La raison est que ces librairies n'ont pas été crée initialement pour etre scalable (i.e. Appliquer à une large quantité de données). La solution serait donc d'utiliser les frameworks (e.g Spark) pour réaliser du calcul distribué, mais cela implique au préalable de connaitre son fonctionnement, l'API et il se peut qu'il faille réecrire le code dans un autre language. Par exemple, si l'on veut tirer pleine performances de Spark il faudeait que le code soit écrit en Scala et non pas en python dans la mesure où Scala est 10x plus rapide que python pour le traitement et l'analyse de données. Ce processus pouvant etre fastidieux et frustrant, il serait préférable de travailler avec les librairies scalées propres à python. Et c'est exactement ce que nous permet Dask.

Dask va donc scaler entre autre numpy, pandas, sckikit-learn et cloner leur API afin de fournir un environment familier et réduire au maximum la réecriture du code ( Notes: Il est important de noter que meme si Dask copie les API des librairies data les plus connues de python, il n'implémente pas encore aujourd'hui tous leurs contenus, ce qui nécessite parfois de faire davantage "from scratch").

Enfin, Dask propose 2 grandes facons de réaliser du calcul distribué. La première étant bien entendu de scaler le travail entre plusieurs machines, mais Dask permet également de distribuer les computations à l'echelle d'une seule machine. En effet, avec Dask il est tout à fait possible de scaler les taches entre les coeurs du processeur. Il est important de comprendre qu' un cluster de machines n'est pas forcément la meilleur décision pour travailler dans un envirnement Big Data. d'une part, les ordinateurs utilisant Dask et ayant des composants relativement récents ( CPU multi-coeurs dernière génération, GPU, RAM allant de 16 à 64gb, SSD) permettent de travailler avec des jeux de données de plus de 100gb sans grande difficulté. D'autre part, travailler en local évite bon nombre de contraintes telles que le fait d'etre limité par la bande passante ( dans un cluster les données doivent circuler dans le réseau), mais aussi le fait de devoir gérer des images docker plutot que de travailler avec un software environment local par exemple.


## Comment Dask fonctionne ? ##

Après avoir présenté Dask je pense que vous comprenez désormais son interet. Au sein de cette partie nous allons voir plus en détails le fonctionnement interne de Dask.

### Concepts ###
Tout d'abord,il faut savoir de quoi est constitué un réseau distribué Dask. En effet, ce dernier repose sur 3 concepts fondamentaux:

- Le scheduler: 

Comme son nom l'indique, le rôle du scheduler est de planifier les taches de facon distribué. Ce dernier assimile les tâches à effectuer sous la forme de graph (Task graph) crée par Dask au préalable, et va  demander ensuite aux workers de realiser ces taches.


![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/scheduler.png)


- Le client:

Le client est tout simplement ce qui va nous permettre de nous connecter au cluster dask. Après avoir créer le cluster, on initialise le client en lui passant l'adresse du scheduler. Le client s'enregistre en tant que scheduler par défaut, et permet d'exécuter toutes les collections de dask (dask.array, dask.bag, dask.dataframe et dask.delayed).

- Les Workers:

Si l'on décide d'utiliser dask sur une seule machine, les workers sont les coeurs du processeur tandis que dans un cluster ce sont les les différentes machines. Les workers recoivent les informations du scheduler et exécutent les tâches. Ils rapportent au scheduler lorsqu'ils ont terminé, tout en conservant les résultats stockés dans les workers où il ont été calculé.


![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/parallel_computing_graph.png)


### Créer un un réseau distribué ###
Maintenant que nous sommes familier avec les concepts de base, nous allons voir comment implémenter un réseau distribué avec Dask. Comme nous l'avons énoncé précédemment, nous avons la possibilité de créer un réséau distribué en local ( une seule machine) et en cluster (plusieurs machines). Nous  allons voir brièvement les deux cas de figure:

- Dask Distributed Local:

```py
from dask.distributed import Client

client = Client()
# or
client = Client(processes=False)

print('Dashboard:', client.dashboard_link)
```

```
Client

    Scheduler: tcp://127.0.0.1:39663
    Dashboard: http://127.0.0.1:8787/status

	
Cluster

    Workers: 4
    Cores: 8
    Memory: 16.51 GB
```

Ici, nous venons tout simplement d'initialiser le client sans paramètres pour lui indiquer que l'on veut se connecter à un réseau distribué local. Dask renvoit par ailleurs l'adresse du scheduler ainsi que les caractéristiques du cluster (celles de ma machine)(Notes: Il est tout à fait possible de faire varier le nombre de workers).


- Dask Distributed cluster: 

```py
# We have to creat a software environment that's gonna be run as a docker image on all workers in the cloud
# Workers and Client NEED to have same packages versions

import coiled

coiled.create_software_environment(
    name="ml-env",
    conda="environment.yml",
)

coiled.create_cluster_configuration(
    name="ml-env",
    software="ml-env",
    # there are other inputs here you can also adjust
)
```

```py
#Using our previous software environment on a docker image, we creat a cluster with 10 workers and initialize the client

cluster = coiled.Cluster(n_workers=10,
                         software="new_ml_env")

from dask.distributed import Client

client = Client(cluster)
print('Dashboard:', client.dashboard_link)
```

```
Client

    Scheduler: tls://ec2-18-218-215-162.us-east-2.compute.amazonaws.com:8786
    Dashboard: http://ec2-18-218-215-162.us-east-2.compute.amazonaws.com:8787

	
Cluster

    Workers: 10
    Cores: 40
    Memory: 171.80 GB
```

Le code présenté ci-dessus permet de créer un cluster dans le cloud (via AWS). Pour etre plus précis nous avons utilisé [Coiled Cloud](https://docs.coiled.io/user_guide/getting_started.html) qui permet de scaler très simplement dans le cloud. 
Dans un premier temps il est nécessaire de creer une image docker qui contient l'ensemble des packages nécessaires qui sera ensuite runner sur chaque workers ( Chaque workers, ainsi que le client doivent posséder les memes dépendances sous risque de poser des problèmes par la suite). Une fois le software environment crée, on peut créer le cluster et initialiser le client avec. Comme indiqué précedemment, Dask renvoit l'adresse du scheduler ainsi que les caractéristiques du cluster.  

Vous l'aurez sous doute remarqué, Dask renvoit également l'adresse de ce qu'il appelle le dashboard. Le dashboard est un outil très utile puisqu'il permet notamment de comprendre et suivre comment Dask processe les tâches à réaliser, comment elles sont réparties, les capacités utilisées de chaque workers, et permet également d'avoir accès aux logs.


![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/dask_dashboard.png)



Pour en savoir plus sur le dashboard, je vous invite à consulter la vidéo ci-dessous:


[![](http://img.youtube.com/vi/nTMGbkS761Q/0.jpg)](http://www.youtube.com/watch?v=nTMGbkS761Q "Dask Dashboard")


Dans la section suivante nous allons nous interesser aux API de Dask, avec un focus sur Dask.arrays et Dask.dataframes.

### Dask API ###

Dans cette section  nous allons nous interesser aux différentes API de Dask. 

Dask est composé de:

- Arrays (S'appuye sur NumPy)
- DataFrames (Repose sur Pandas)
- Bag (Suit map/filter/groupby/reduce)
- Dask-ML (Suit entre autre Scikit-Learn)
- Delayed (Couvre de manière générale Python)
- Futures follows concurrent.futures from the standard library for real-time computation.

Nous allons voir brièvement comment utiliser Dask.arrays, Dask.dataframes et pour finir Dask.ml, avec quelques exemples de code puisque ce sont les 3 API que nous sommes le plus susceptibles d'utiliser dans un contexte d'analyse de données.

#### Dask arrays ####

Les Dask arrays coordonnent de nombreux Numpy arrays, disposés en "chunks" (morceaux) à l'intérieur d'une grille. L'API de Dask prend en charge un large partie de l'API Numpy.

![](https://raw.githubusercontent.com/natsunami/website/9a0d72b882e33755b4b1de778588e746b7c8da3b/assets/img/dask/dask-array-1.svg)

```py
import dask.array as da

x = da.random.randint(1, 100, size=(20000, 20000),   # 400 million element array 
                              chunks=(1000, 1000))   # Cut into 1000x1000 sized chunks
y = x.mean()
```

```py
x
``` 
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/dask_array.png)

```py
y.compute()
```
```
49.99919512
```
Si vous etes familier avec Numpy vous aurez constaté que la seule différence réside uniquement dans le fait d'importer dask.array (da) plutot que Numpy (np). Si l'on cherche à afficher l'objet x on peut voir cependant qu'il n'apparait pas comme si l'on utilisait Numpy mais au lieu de ca, affiche les propriétés de l'objet. L'explication est la suivante, Dask ne réalise pas la computation tant qu'elle n'est pas explicitement demandé via la méthode .compute(). En effet, Dask est par défault "Lazy" (Notes: Utiliser la methode .compute() va stocker la computation en mémoire. La méthode doit donc etre appelée uniquement si la computation à suffisemment de place sous peine de voir l'erreur *KilledWorker*).


#### Dask dataframes #####

Les Dask dataframes permettent de coordonner plusieurs dataframe Pandas partitionnés selon l'index. De meme que l'API Dask.arrays, l'API Dask.dataframes possèdent une grande partie des fonctionnalités de l'API Pandas.

![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/dask_dataframe.jpg)

```py
import dask.dataframe as dd

df = dask.datasets.timeseries()
df
```
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/dask_lazy_dataframe.png)

```py
df.head()
```
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/dask/dask_dataframe_head.png)

```py
x_mean_by_name = df.groupby('name')['x'].mean()
x_mean_by_name
```

```
Dask Series Structure:
npartitions=1
    float64
        ...
Name: x, dtype: float64
Dask Name: truediv, 101 tasks
```

```py
x_mean_by_name.compute()
```

```
name
Alice      -0.000354
Bob        -0.001484
Charlie    -0.001285
Dan         0.003877
Edith      -0.000339
Frank       0.000321
George     -0.002556
Hannah     -0.001059
Ingrid     -0.000804
Jerry      -0.002137
Kevin       0.001354
Laura       0.000851
Michael    -0.000793
Norbert     0.001598
Oliver     -0.002574
Patricia   -0.001985
Quinn      -0.000443
Ray        -0.000632
Sarah      -0.002952
Tim         0.000621
Ursula      0.003599
Victor      0.004504
Wendy      -0.001640
Xavier      0.001189
Yvonne     -0.001253
Zelda      -0.002810
Name: x, dtype: float64
```

Pas de grand changement par rapport à Dask.arrays. Si l'on cherche à afficher un dataframe, Dask le renvoit sous format "lazy", donc sans les valeurs. On peut cependant afficher un partie du dataframe avec .head() ou encore .tail() et si l'on désir réaliser une computation, on utilise comme vu précédemment la méthode .compute() (Note:  Connaitre le nombre de partitions est important afin de realiser certaines opérations (e.g concat()) dans la mesure où les dataframes doivent etre partitionnés de la meme manière).

#### Dask ML ####

Comme son nom l'indique, tout comme Spark ML, Dask ML permet de réaliser du machine learning distribué. L'API repose entre autre sur celle de scikit-learn et d'autres, tel que XGBoost. De manière générale, l'API peut etre utilisée pour le preprocessing, réaliser de la cross validation, faire des hyperparameters search, créer des pipelines et autres.

Dask ML peut etre utilisé pour pallier à 2 types de contraintes. La première étant liée à la mémoire (Memory-Bound), et la deuxième étant computationelle (CPU-Bound).

![](https://ml.dask.org/_images/dimensions_of_scale.svg)

##### Memory-Bound #####

Ce problème se pose lorsque la taille du jeu de données est supérieur à la RAM. Dans ce contexte, utiliser Numpy ou Pandas ne fonctionnerait pas et il serait donc impossible de réaliser du machine learning en utilisant scikit-learn par exemple. Prenons un exemple concret pour comprendre le problème. Le dataset [Microsoft Malware Prediction](https://www.kaggle.com/c/microsoft-malware-prediction), challenge Kaggle d'il y a 2 ans, comporte approximativement 10M de lignes et 80 colonnes. Il est tout à fait possible de lire le dataset avec pandas, bien que cela soit beaucoup plus challenging qu'avec Dask, mais le problème étant que l'on risque de faire face à des difficultés plus importantes au moment du preprocessing. Par exemple, utiliser un One-Hot Encoder sur les variables catégorielles ferait exploser la taille du jeu de données. L'extrait de code ci-dessous permet de préprocesser le jeu de données sans grande difficultés gràce à dask-ml.

```py
import dask.array as da
import dask.dataframe as dd

from sklearn.pipeline import Pipeline

from dask_ml.compose import ColumnTransformer
from dask_ml.preprocessing import StandardScaler, OneHotEncoder
from dask_ml.impute import SimpleImputer

binary_processor = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent'))])

numerical_processor = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

categorical_processor = Pipeline(steps=[('encoder',OneHotEncoder())])

#category_processor = Pipeline(steps=[('encoder',TargetEncoder(cols=category_columns))])

preprocessor = ColumnTransformer(transformers=
                                [('bin',binary_processor,binary_columns),
                                ('num',numerical_processor,numerical_columns),
                                ('cat',categorical_processor,category_columns)])
				

X_train_processed = preprocessor.fit_transform(X_train)
```
```
CPU times: user 8.09 s, sys: 101 ms, total: 8.19 s
Wall time: 3min 10s
```

##### CPU-Bound #####

Ce cas de figure apparait lorsque le modèle ML crée est si large/ complexe que cela à un impact sur le flux de travail (e.g.une tâche d'apprentissage, recherche d'hyperparamètre beaucoup trop longue). Dans ce contexte, paralléliser la tâche à effectuer en utilisant les estimateurs de dask-ml  permet de faire face à ces difficultés. L'extrait ci-dessous montre comment une hyperparameters search via Dask-ML et son estimateur HuperbandSearchCV.

```py
import numpy as np

from dask_ml.model_selection import HyperbandSearchCV
from dask_ml.datasets import make_classification
from sklearn.linear_model import SGDClassifier

X, y = make_classification(chunks=20)

est = SGDClassifier(tol=1e-3)

param_dist = {'alpha': np.logspace(-4, 0, num=1000),

              'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],

              'average': [True, False]}

search = HyperbandSearchCV(est, param_dist)

search.fit(X, y, classes=np.unique(y))

search.best_params_
```
```
{'loss': 'log', 'average': False, 'alpha': 0.0080502}
```


## Différences avec Spark  ##

Parut en 2010, grâce à son efficacité et son intégration dans un écosystème riche ( Apache projects), Spark est aujourd'hui incontournable pour travailler dans un environnement Big Data. Développé plus tardivement, Dask est un projet plus léger que Spark. Grâce à son intégration dans l'écosystème Python (Numpy, Pandas, Scikit-Learn, ect...), Dask permet de réaliser des opérations plus complexes que Spark. Comparer Spark et Dask afin de déterminer lequel est meilleur n'aurait pas vraiment de sens puisque leur utilisation dépend avant tout de ce que l'on cherche à réaliser. Par ailleurs, Dask et Spark peuvent très bien être utilisés ensemble sur un même cluster. Nous allons toutefois dresser quelques différences notables entre les deux technologies.


- Langage:  

   - Comme nous l'avons mentionné plus haut, Spark est codé en Scala mais supporte également Python et R (Scala reste le langage offrant la meilleure synergie avec Spark). 
   - Quant à Dask, il est codé en Python et ne supporte uniquement que ce langage. 
 
- Ecosystème: 

   - Spark fait partie de l'écosystème Apache et est de ce fait bien integré à de nombreux autres projets (e.g. Hadoop/Hive, HBase, Cassandra, etc...)
   - Dask est integré dans l'ecosystème Python et a été crée pour scaler les librairies Data les plus connues (e.g. Numpy, Pandas, Scikit-Learn)

- Age:

   - Développé en 2010, Spark est aujourd'hui dominant dans le milieu Big Data.
   - Plus jeune ( crée en 2014), Spark est en constante amélioration afin de rendre l'expérience la plus similaire aux librairies  Data Python.

- Champ d'application:

   - Spark se concentre davantage sur l'aspect business intelligence, analytics, en permettant de faire de la requête de données via SQL et du machine learning "léger".
   - Dask est plus adapté pour des applications scientifiques ou business nécéssitant du machine learning plus poussé.

-Design: 
 
   - Le modèle interne de Spark est de niveau supérieur, offrant de bonnes optimisations de haut niveau sur des calculs appliqués uniformément, mais manquant de flexibilité pour des algorithmes plus complexes ou des systèmes ad hoc. Il s'agit fondamentalement d'une extension du paradigme Map-Shuffle-Reduce.
   - Le modèle interne de Dask est de niveau inférieur et manque donc d'optimisations de haut niveau, mais il est capable d'implémenter des algorithmes plus sophistiqués et de construire des systèmes sur mesure plus complexes. Il est fondamentalement basé sur une planification générique des tâches.

- Scaling: 
  
   - Spark comme Dask permettent de créer des clusters composés jusqu'à mille machines.

Si vous hésitez sur la technologie à choisir entre Spark et Dask, ceci devrait vous aider:

- Les raisons pour lesquelles vous devriez choisir Spark:

   - Vous preferez utiliser Scala ou SQL.
   - Vous disposez principalement d'une infrastructure JVM et de systèmes hérités.
   - Vous voulez une solution de confiance, largement utilisée dans le milieu de l'entreprise.
   - Vous faites de la business analytics ou vos applications nécéssitent du Machine learning peu complexe.
   - Vous avez besoin d'une solution tout-en-un (Ecosystème Apache).

- Les raisons pour lesquelles vous devriez choisir Dask:

   - Vous preferrez utiliser Python et êtes deja familier avec ses librairies.
   - Vous avez des projets plus complexes que ce que Spark permet de résoudre.
   - Vous souhaitez une transition plus légère d'une utilisation locale à une utlisation en cluster.
   - Vous voulez interagir avec d'autres technologies et installer plusieurs paquets ne vous dérange pas.
 

## Conclusion ##

Cet article touche malheureusement à sa fin. Dask est une librairie très riche à parcourir, qui aux premiers abord peu sembler très facile d'accès mais qui nécessite beaucoup de temps à maitriser. L'objectif de cet article était avant tout de vous introduire à Dask, de comprendre à quoi cela pouvait etre utile, comment cette librairie fonctionne en vous présentant les concepts fondamentaux et les API les plus importantes pour traiter la data, et nous avons vu à la fin ses principales différences avec Spark. Je suis convaincu qu'avec le temps Dask sera ammené à se développer davantage mais aussi à être de plus en plus utilisé dans le monde de l'entreprise tant sa prise en main est simpliste ( si on est déja familier avec Python) car cela permettrait de rendre le Big Data plus accessible.




