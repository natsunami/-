---
layout: post
title: French to SQL query API using transformers | Fast-Api | Docker 
subtitle: Fatigué d'écrire des requêtes SQL ? Flemme d'apprendre le SQL ? Cet article peut vous interesser...
cover-img: /assets/img/background_api_sql_query.png
thumbnail-img: /assets/img/french_sql_api_logo.png
share-img: /assets/img/
tags: [SQL, Transformers, Hugging face, query, fast-api, docker]

---
## Introduction ##

Il y a quatre ans, un papier scientifique entraîna une petite révolution dans le milieu de l'IA. Considéré comme une véritable avancée, le papier [Attention is all you need (2017)](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) de Vaswani, et al., développa le concept d'**attention**. Entrainant dans son sillage une amélioration des performances des réseaux de neurones utilisés pour le NLP (Naturel Language Processing) (les RNN), naquirent à la suite les Transformers. Non, je ne parle pas des auto-bots et decepticons se livrant à une lutte sans merci, mais bel et bien de neural networks reposant sur le concept d'attention, et étant particulièrement efficaces pour le NLP. Petite anecdote, si vous avez remarquez une amélioration dans la traduction générée par Google trad au cours de cette même période, vous savez désormais que les transformers sont à l'œuvre !

Depuis cette fameuse publication, les avancées sur les transformers n'ont cessées de s'enrichir, contribuant ainsi à l'apparition d'un nombre incroyable de modèles impliqués dans la résolution de tâches NLP aussi diverses que variées. On donnera comme exemple la classification de phrases, l'analyse de sentiments, la traduction mais aussi la génération de texte (Essayez [AI Dungeon](https://play.aidungeon.io/main/landing) pour voir la puissance des modèles GPT-2 & 3 et surtout quelques heures de fun) et autres. Grâce à [Hugging Face] 🤗, société française 🇫🇷 fondé en 2016 par *Clément Delangue* et *Julien Chaumond*, il est désormais d’accéder à la pleine puissance des transformers, et ceci en toute simplicité. En effet, avec la librairie [transformers](https://huggingface.co/transformers/), de nombreux modèles pré-entrainés sont mis à disposition. 

## Contexte ##

En scrollant de nombreuses minutes sur LinkedIn, je me suis un jour retrouvé devant un post de Hugging Face qui faisait état des capacités du modèle GPT-Neo, cousin open-source de GPT-3. Ce post m'intrigua puisqu'il mentionna la possibilité de traduire du texte en requête SQL. En tant que Data Analyst, le SQL tient une place fondamentale dans mon métier, je dirai même qu'il est partout. Ce post tomba à pic car je m’étais déjà demandé s’il existait un système/appli permettant de réaliser des requêtes « textuelles » (ex. : "Quels sont les 10 magasins ayant le chiffre d'affaire le plus élevé dans l'ordre décroissant ?"). Même si le SQL est un langage extrêmement répandu et largement utilisé pour accéder aux données,on peut imaginer qu'une application permettant de convertir du texte en requête SQL pourrait trouver son utilité chez les personnes n'ayant pas/peu de notions en SQL.

Vous l’auriez compris en lisant le titre de cet article, ce post m’a donné une idée. Bien que L'API pour traduire de l’anglais en SQL existe déjà (Testez la [ici](https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL) !), cette dernière n’existe pas pour le français 🇫🇷 ! Cocorico, nous allons créer une API toute simple qui prendra en input du texte français pour la convertir en requête SQL, et of course, nous la déplorerons. Trêve de bavardage, il est temps de passer à l'action 💪.

## La recette de cuisine pour l'API ##

Pour débuter, vous aurez besoin :
1. 3 œufs
2. 100 g de fa....Ah désolé, c'est la recette du gâteau au chocolat marmiton ça...

Non, plus sérieusement, voici ce dont nous allons vraiment avoir besoin :

### Transformers 🤗 ### 

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1200%2F1*Bp8K-_PJrG2NQxLlzk7hlw.png&f=1&nofb=1)

Comme mentionné au dessus, nous allons utiliser la librairie **Transformers** de Hugging Face. La librairie est indispensable puisqu'elle contient les modèles pré-entrainés que nous allons utiliser pour la traduction du français en SQL. Pour l'installation, si vous avez déjà Tensorflow 2.0 et/ou PyTorch, vous pouvez directement l'installer avec pip (Pour plus de précisions, la doc d'installation est consultable [ici](https://huggingface.co/transformers/installation.html):
```console
pip install transformers
```
Pour vérifier que l'installation s'est bien passé, vous pouvez lancer la commande suivante dans votre terminal bash:
```console
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```
Vous devriez voir apparaître ceci :
```console
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```
### FastApi ⚡️ ###

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1023%2F1*du7p50wS_fIsaC_lR18qsg.png&f=1&nofb=1)

[FastApi](https://fastapi.tiangolo.com/) est un framework web qui, comme son nom l'indique, va nous permettre de créer rapidement des API ultra-performante. En trois mots, FastApi c’est : **Rapide**, **simple** et **robuste**. La rapidité de FastAPI est possible grâce à Pydantic, Starlette et Uvicorn. Pydantic est utilisé pour la validation des données et Starlette pour l'outillage, ce qui le rend extrêmement rapide par rapport à Flask et qui lui confère des performances comparables à celles des API Web à haut débit en Node ou Go. Starlette est un framework/toolkit ASGI léger, idéal pour créer des services asynchrones à haute performance. Uvicorn est un serveur ASGI rapide comme l'éclair, construit sur uvloop et httptools. 

Pour installer FastApi, rien de plus simple :
```console
pip install fastapi uvicorn[standard]
```

### Docker  🐋  ###

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.1min30.com%2Fwp-content%2Fuploads%2F2018%2F04%2FLogo-Docker.jpg&f=1&nofb=1)

Je pense que je n'ai pas vraiment besoin de présenter Docker tant il est populaire. Pour rappel, Docker va nous permettre de « containeriser »  notre code ainsi que ses dépendances (ex . : Transformers, FastApi) afin notre API puisse être exécutée sur n'importe quel serveur. 

Si vous n’avez pas Docker installé, je vous renvoie à la doc, très bien écrite : [Get Docker](https://docs.docker.com/get-docker/) 
 
## Création de l'API ##

Maintenant que nous avons passé en revu tout ce dont nous avions besoin pour créer notre API et in fine, la déployer, nous allons pouvoir rentrer dans le vif du sujet ! Vous allez voir qu'en quelques lignes de code la magie va opérer. 🧙🏻‍♀️  


### Création du script ###

Dans cette partie nous allons montrer comment créer l'API étape par étape. 

1. Importer les librairies

Pour commencer, il faut Importer les librairies dont nous avons besoin. Dans l'ordre, on importe tout d'abord **FastApi** et **uvicorn**, puis les auto classes **AutoModelWithLMHead** et **AutoTokenizer** de la librairie transformers que nous utiliserons juste après, et que j'expliquerai plus en détails. La librairie logging n'est pas essentielle ici, elle permet juste d'émettre des messages en cas d’anomalies.
```py
# Import packages:

from fastapi import FastAPI
import uvicorn
from transformers import AutoModelWithLMHead, AutoTokenizer
#import logging
```
2. Créer l'instance FastApi

Une fois les librairies importées, on instancie notre application. Pour cela, rien de plus simple :
```py
# Instanciate the app:

app = FastAPI(title='French to SQL translation API 🤗', description='API for SQL query translation from french text using Hugging Face transformers')

#my_logger = logging.getLogger()
#my_logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, filename='logs.log')
```
FastApi est une classe python qui contient l'ensemble des fonctionnalités pour l'API. De plus, par convention on appellera la variable 'app' mais vous pouvez l'appeler comme bon vous semble. Pour finir, il est possible de renseigner dans les paramètres le titre ainsi qu'une brève description de l'application, ce qui permettra aux utilisateurs de savoir à quoi sert l'API.

3. Télécharger et instancier les modèles transformers 

Cette étape est très importante dans le script puisque c'est ici que nous allons télécharger les modèles pré-entrainés et les tokenizers associés et les instancier. Pour cela, on utilise les deux classes précédemment importées :

- **AutoModelWithLMHead**: Classe de modèle qui sera instanciée lorsque l'on utilisera la méthode de classe AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path). Il existe une très grande variété de modèles qui peuvent être téléchargés, chacun ayant des spécificités et permettant de résoudre des problématiques différentes. La liste des modèles est consultable [ici](https://huggingface.co/transformers/v3.0.2/model_summary.html).

- **AutoTokenizer**: Classe de tokenizer qui sera instanciée lorsque l'on utilisera la méthode de classe AutoTokenizer.from_pretrained(pretrained_model_name_or_path). Le tokenizer utilisé est celui mentionné par l'utilisateur dans la méthode from_pretrained (ex. : distillert, roberta, t5,...) Petit rappel concernant la tokenization, l'objectif du tokenizer est tout simplement de prétraiter le texte. Il va diviser le texte en mots (ou parties de mots, symboles de ponctuation, etc.) que l'on appelle **tokens**. A retenir, lors de l'utilisation d'un modèle il faut s'assurer que le tokenizer instancié correspond au tokenizer ayant été utilisé pour entrainer le modèle.

```py
#Download and instanciate vocabulary and Fr-to-En model :
model_trad_fr_to_eng = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
tokenizer_translation = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

#Download and instanciate vocabulary and En-to-SQL model :
tokenizer_sql = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model_sql = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
```

Le code ci-dessus nous permet donc de télécharger et instancier dans un premier temps le modèle [opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en). Ce modèle va tout simplement nous permettre de convertir du francais en anglais. Puis on télécharge et instancie le modèle [t5-base-finetuned-wikiSQL](https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL qui va convertir le texte anglais en SQL.

4.Définir le *path operation decorator*:

Cette étape va faire en sorte que l'API puisse réaliser certaines actions (dans notre cas, traduire le texte en SQL) en communiquant avec elle via des méthodes de requêtes HTTP. Pour parvenir à notre fin, on va renseigner à FastApi le type de méthode (ex. : POST, GET, DELETE, PUT,) et le chemin (L’endroit/endpoint où se fait la requête). Voici un exemple d'opération que l'on pourrait réaliser:

```py
@app.get('/')
def get_root():

    return {'Message': 'Welcome to the french SQL query translator !'}
```

Le ``` @app.get('/')``` dit à FastApi que la fonction juste en dessous est chargée de traiter les requêtes qui vont vers le chemin ```/``` et qui utilisent la méthode GET, ce qui renverra le dict ```{'Message': 'Welcome to the french SQL query translator !'}```

Maintenant que nous comprenons mieux comment réaliser des opérations, nous allons pouvoir écrire la fonction qui va traduire le texte français en SQL:

```py
@app.get('/get_query/{query}', tags=['query'])
async def text_to_sql_query(query:str):
    
    '''This function allows to convert french text to a SQL query using opus-mt-fr-en transformer for text translation and a t5 base finedtuned on wikiSQL for english to SQL traduction'''
    
    # Encoding: Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.
    # Decoding: Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.
   
    inputs_trad = tokenizer_translation.encode(query, return_tensors="pt")
    outputs_trad = model_trad_fr_to_eng.generate(inputs_trad, max_length=600, num_beams=4, early_stopping=True)
    
    text_trad = tokenizer_translation.decode(outputs_trad[0]).replace('<pad>','')

    text_to_convert_query = "translate English to SQL: %s </s>" % text_trad
    features = tokenizer_sql([text_to_convert_query], return_tensors='pt')

    output_sql = model_sql.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'])

    sql_query = tokenizer_sql.decode(output_sql[0]).replace('<pad>','').replace('</s>','')

    return { 'SQL QUERY' : sql_query} 
```

Ce que l'on fait ici est relativement simple à comprendre. Concernant la fonction en elle-même, cette dernière prend en argument le texte français, que j'ai désigné ici par ```query```.Au passage vous pouvez noter que dans le path de la méthode get on retrouve notre ```query``` entre ```{}``` pour indiquer à l'utilisateur qu'il a directement la possibilité de rentrer directement sa requête version texte français dans l'url.

Comme nous l'avons dit précédemment, on cherche à convertir le texte français en anglais puis de l'anglais vers le SQL. La procédure à réaliser est la même dans les deux cas :
- Encodage : On encode le texte avec la méthode ```encode```. La ```string``` est convertit en dictionnaire de la forme suivante au sein duquel la liste d’Integer représente les index des tokens:
```py
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
- Génération : Génère les index des tokens de la séquence que l'on cherche à obtenir à partir des index obtenus par le Tokenizer et l'utilisation du modèle pré-entrainé.
- Décodage : Décode les index des tokens en une nouvelle string.

Pour conclure cette partie j'insiste sur le fait que la fonction doit renvoyer un ```dict```.

5. Runner l'API

On arrive enfin à l'étape finale qui va tout simplement consister à faire tourner en local notre API sur un serveur Uvicorn. Pour cela, il existe deux possibilités.
Vous pouvez rajouter dans le script ces 2 lignes de code :
```py
# Running the API on our local network:

if __name__ == '__main__':    
    uvicorn.run(app, host='127.0.0.1', port=8000)
```
Ou bien, vous pouvez directement lancer le serveur dans le terminal :
```console
uvicorn french_text_to_sql_query:app --reload
```
Au passage, voici le script au complet :

```py
# Import packages 
import uvicorn
import logging
from fastapi import FastAPI
from transformers import AutoModelWithLMHead, AutoTokenizer

app = FastAPI(title='French to SQL translation API 🤗', description='API for SQL query translation from french text using Hugging Face transformers')

my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='logs.log')

model_trad_fr_to_eng = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

tokenizer_translation = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

tokenizer_sql = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

model_sql = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

@app.get('/')
def get_root():

    return {'Message': 'Welcome to the french SQL query translator !'}

@app.get('/get_query/{query}', tags=['query'])
async def text_to_sql_query(query:str):
    
    '''This function allows to convert french text to a SQL query using opus-mt-fr-en transformer for text translation and a t5 base finedtuned on wikiSQL for english to SQL traduction'''
   

    inputs_trad = tokenizer_translation.encode(query, return_tensors="pt")
    outputs_trad = model_trad_fr_to_eng.generate(inputs_trad, max_length=600, num_beams=4, early_stopping=True)

    text_trad = tokenizer_translation.decode(outputs_trad[0]).replace('<pad>','')

    text_to_convert_query = "translate English to SQL: %s </s>" % text_trad
    features = tokenizer_sql([text_to_convert_query], return_tensors='pt')

    output_sql = model_sql.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'])

    sql_query = tokenizer_sql.decode(output_sql[0]).replace('<pad>','').replace('</s>','')

    return { 'SQL QUERY' : sql_query} 

if __name__ == '__main__':    
    uvicorn.run(app, host='127.0.0.1', port=8000)
```
Une fois tout ceci réalisé, notre script est terminé et nous n'avons plus qu'à le faire tourner. Si tout fonctionne correctement, vous devriez voir apparaitre ceci dans le terminal :
```console
INFO:     Started server process [31847]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:33262 - "GET / HTTP/1.1" 200 OK
```
**Attention**: L'exécution peut prendre un peu de temps selon votre connexion internet car il faut dans un premier temps télécharger les modèles.

Il est désormais temps de se rendre sur notre API et de la tester ! Pour cela nous allons nous rendre au dashboard FastApi qui va nous permettre d'envoyer des requêtes à l’API : [http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

![](https://user-images.githubusercontent.com/52154100/125428785-3cd376b1-667b-49ad-a92b-e4a53f97365b.png)

6. Test de l'API
7. 
Le moment de vérité est arrivé ! Pour vérifier que l'API fonctionne correctement nous allons exécuter une requête relativement simple du type ```Sélectionner les magasins ayant un chiffre d'affaire supérieur à 100000``` ce qui se traduirait en SQL par la query ```SELECT shops FROM table WHERE turnover > 100 000```. Tout ce que nous faisons reste  expérimental donc ne soyez pas surpris si la query renvoyée n'est pas exactement ce dont vous vous attendiez. Mais tout cela reste quand même assez prometteur pour la suite et amusant (enfin c'est mon avis).

![](https://user-images.githubusercontent.com/52154100/125434336-c806462e-b4bc-4b04-a433-41588483c5e2.png)

Comme vous le voyez, j'ai directement exécuté la requête dans le dashboard de FastApi (mais vous pouvez très bien l'exécuter dans le terminal) qui nous renverrai ceci :
```console
{
  "SQL QUERY": " SELECT Stores FROM table WHERE Turnover > 100000"
}
```
Le résultat n’est pas trop mal non ? 🥳
Après, comme je vous l'ai dit, ne vous attendez pas à ce que l'API puisse résoudre des query complexes. Dans l'idée, si l'on voulait obtenir de meilleures performances il faudrait non pas passer par la traduction français-anglais mais directement entrainés un modèle sur des phrases écrites en français.

## Tout le monde dans le conteneur 🐳 ## 

L'API étant fonctionnelle, nous allons faire en sorte de la rendre utilisable par tous. Afin de rendre cela possible, nous allons devoir utiliser Docker. Plus précisément il va falloir convertir notre application en image Docker. Pour cela, nous allons créer un Dockerfile qui prendra la forme suivante :
```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 

WORKDIR /app/

COPY . /app/

RUN pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
&& pip install transformers[sentencepiece] 

CMD ["python","french_text_to_sql_query.py"]
```
Pour faire simple, le Dockerfile va donner plusieurs instructions à Docker. Docker va créer un fichier /app dans l'image base ```uvicorn-gunicorn-fastapi:python3.7``` (Notes: Cette image permet d'utiliser FastApi dans un conteneur Docker) puis va copier dans ce fichier tout ce qui se trouve dans le répertoire actif, et installer les différents packages nécessaires. Enfin, on demande à Docker de lancer notre script python sur le port 8000, et tout ça dans un conteneur Docker. 

Désormais, nous n’avons plus qu'à lancer deux commandes dans notre terminal: 
```console
docker build -t french_sql_query
docker run french_sql_query
```
Comme tout à l'heure, on se rends à l'adresse [127.0.0.1:8000/docs#/](127.0.0.1:8000/docs#/) pour voir si tout fonctionne correctement. Si vous voyez le dashboard FastApi, c'est que nous avons réussi à déployer notre API dans un conteneur Docker 👌🏼.

## We made it ##

Yeaaaaah ! Nous avons déployé une API qui permet de traduire du française en une requête SQL ! Si ce n’est pas cool ça !? Merci d'avoir suivi cet article/tutoriel jusqu'au bout. Si besoin, vous pouvez directement télécharger l'image Docker du projet sur [mon repo Dockerhub](https://hub.docker.com/repository/docker/natsunami/content) 👈🏽

On se retrouve bientôt pour un nouvel article ! 🖖

