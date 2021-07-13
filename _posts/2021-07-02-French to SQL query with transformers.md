---
layout: post
title: French to SQL query API using transformers | Fast-Api | Docker 
subtitle: Tired to write down SQL queries ? Lazy to learn SQL ? This article might be for you...
cover-img: /assets/img/
thumbnail-img: /assets/img/article french sql query logo.png
share-img: /assets/img/
tags: [SQL, Transformers, Hugging face, query, fast-api, docker]

---
## Introduction ##

Il y a quatre ans, un papier scientifique entraina une profonde révolution dans le milieu de l'intelligence artificielle et plus spécifiquement dans le deep learning. En effet, considéré comme une véritable avancée, le papier [Attention is all you need (2017)](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) de Vaswani, et al., développa le concept d'*attention* permettant une amélioration significative des performances des neural networks utilisés jusqu'à lors pour des tâches de traitement naturel du language (NLP = Naturel Language Processing). Ainsi, nacquirent les Transformers. Non, je ne parle pas des auto-bots et decepticons se livrant une lutte sans merci et ayant la faculté de se transformer en voitures, mais bel et bien de neural networks reposant sur le concept d'attention, et très puissant en  NLP. Petite anecdote, si vous avez remarquez une amélioration dans la traduction générée par Google trad au cours de cette période,maintenant vous savez que se sont les transformers à l'oeuvre !

Depuis la publication, les découvertes et avancées dans l'étude des transformers se sont considérablement enrichient, contribuant ainsi au développement d'un nombre incroyable de modèles utilisés pour la résolution de tâches de NLP telles que la classification de phrases, l'analyse de sentiments, la traduction mais aussi la génération de texte (Essayez [AI Dungeon](https://play.aidungeon.io/main/landing) pour voir la puissance des modèles GPT-2 & 3 et surtout quelques heures de fun) et autres. Grâce à [Hugging Face] 🤗, société francaise 🇫🇷 fondé en 2016 par Clément Delangue et Julien Chaumond, il est désormais possible d'implémenter le State-of-the-Art du NLP en toute simplicité. En effet,avec la création de la librairie [transformers](https://huggingface.co/transformers/), de nombreux modèles pré-entrainés sur des tâches spécifiques,avec possibilité de les fine-tuner, peuvent être utilisés sans etre un crack sur le fonctionnement de ce type de neural network.

Un jour, après avoir scrollé de nombreuses minutes sur Linkedin, je me retrouva devant un post de Hugging Face qui faisait état des capacités du modèle GPT-Neo, cousin open-source de GPT-3. Ce post m'intrigua fortement, d'autant plus qu'il était mention de la possibilité de traduire du texte en requetes SQL. En tant que Data Analyst, le language SQL tient une place fondamentale dans mon métier. Le SQL étant partout, il m'est déja arrivé d'imaginer un monde où il serait possible de requeter ses données directement (e.g: "Quels sont les 10 magasins ayant le chiffre d'affaire le plus elevé dans l'ordre décroissant ?") sans passer par le language SQL. Et bien je crois qu'aujourd'hui ce monde est à porté de main et que de plus en plus de solution permettant de requeter des données de facon textuelles ( = no SQL) seront amenées à etre developpées. En effet, même si le SQL est un language extremement répandu, utilisé non pas seulement par des individus dans la data, il existe bien des personnes n'ayant pas de notions en SQL ( ou qui haissent le language type code) qui pourraient être amenées à réaliser des requetes. Ddans ce contexte, 

Bon, trêve de bavardage, il est temps de passer à l'action et je crois que vous voyez ou je veux en venir...L'API pour traduire du texte en anglais en SQL existe déja (Testez la [ici](https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL) !), l'idée n'est pas de la recréer, cela n'aurait aucun interet. Mais...en bon francais que je suis, je me suis dit: "Hey ! Mais ca serait vraiment cool de pouvoir directement ecrire en francais 🇫🇷!". Cocorico, j'ai décidé de créer une API toute simple qui prend en input du texte francais pour la convertir en requete SQL et de la déployer. Voyons tout de suite comment nous allons nous y prendre !

## La recette de cuisine pour l'API ##

Pour débuter, vous aurez besoin :
1. 3 oeufs
2. 100 g de fa....Ah désolé, c'est la recette du gateau au chocolat marmiton ca...

Non, plus sérieusement, voici ce dont nous allons vraiment avoir besoin:

### Transformers ### 

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1200%2F1*Bp8K-_PJrG2NQxLlzk7hlw.png&f=1&nofb=1)

Comme mentionné auparavant, nous allons utilisé la librairie **Transformers** de Hugging Face. La librairie est indispensable puisqu'elle contient les modèles pré-entrainés que nous allons utiliser pour la traduction du francais en SQL.. Pour l'installation, si vous avez deja Tensorflow 2.0 et/ou PyTorch, vous pouvez directement l'installer avec pip (Pour plus de précisions, la doc d'installation est consultable [ici](https://huggingface.co/transformers/installation.html):
```console
pip install transformers
```
Pour vérifier que l'installation s'est bien passé, vous pouvez runner la commande suivante dans votre terminal bash:
```console
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```
Vous devriez voir apparaître ceci:
```console
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### FastApi ###

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1023%2F1*du7p50wS_fIsaC_lR18qsg.png&f=1&nofb=1)

[FastApi](https://fastapi.tiangolo.com/) est un framework web qui, comme son nom l'indique, va nous permettre de créer rapidement des API ultra-performante. En trois mots, FastApi c'est: Rapide, simple et robuste. La rapidité de FastAPI est possible grâce à Pydantic, Starlette et Uvicorn. Pydantic est utilisé pour la validation des données et Starlette pour l'outillage, ce qui le rend extrêmement rapide par rapport à Flask et lui confère des performances comparables à celles des API Web à haut débit en Node ou Go. Il s'agit d'un cadre innovant construit sur Starlette et Uvicorn. Starlette est un framework/toolkit ASGI léger, idéal pour créer des services asynchrones à haute performance. Uvicorn est un serveur ASGI rapide comme l'éclair, construit sur uvloop et httptools. 

Pour installer FastApi, rien de plus simple:
```console
pip install fastapi uvicorn[standard]
```

### Docker ###

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.1min30.com%2Fwp-content%2Fuploads%2F2018%2F04%2FLogo-Docker.jpg&f=1&nofb=1)

Bon, je pense que je n'ai pas besoin de présenter Docker en long et en large. Pour faire court, Docker va nous permettre de  "containeriser"  notre code ainsi que ses dépendances (e.g. Transformers, FastApi) afin notre API puisse etre executé sur n'importe quel serveur. Pour cela, nous allons créer un DOCKERFILE et builder l'image DOCKER.

Si Docker n'est pas déja installé, je vous renvoie à la doc, qui est très bien ecrite : [Get Docker](https://docs.docker.com/get-docker/) 
 
## Création de l'API ##

Maintenant que nous avons passé en revu tout ce dont nous avions besoin pour créer notre API et in fine, la déployer, il est temps de passer à l'action! Vous allez voir qu'en quelques lignes de code la magie va opérer. 🧙🏻‍♀️  

### Création du script ###

Dans cette partie nous allons montrer comment créer l'API étape par étape. 

1. Importer les librairies

importer dans notre petit script python les librairies dont nous avons besoin pour notre API. Dans l'ordre, on import tout d'abord FastApi et uvicorn, puis les auto classes  **AutoModelWithLMHead** et **AutoTokenizer** de la librairie transformers que nous utiliserons juste après et que j'expliquerai plus en détails. La librairie logging n'est pas essentielle ici, elle permet juste d'émettre des messages suites à des évenèments et ainsi résoudre des anomalies.
```py
# Import packages:

from fastapi import FastAPI
import uvicorn
from transformers import AutoModelWithLMHead, AutoTokenizer
#import logging
```
2. Créer l'instance FastApi

Une fois les librairies importées, la première chose à faire est d'instancier notre application. Pour cela, rien de plus simple :
```py
# Instanciate the app:

app = FastAPI(title='French to SQL translation API 🤗', description='API for SQL query translation from french text using Hugging Face transformers')

#my_logger = logging.getLogger()
#my_logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, filename='logs.log')
```
FastApi est une classe python qui contient l'ensemble des fonctionnalités pour l'API. De plus, par convention on appellera la variable 'app' mais vous pouvez l'appeler comme bon vous semble. Pour finir, il est possible de renseigner dans les paramètres le titre ainsi qu'une brève description de l'application, ce qui permettra aux utilisateurs de savoir à quoi sert l'API.

3. Instanciate transformers models

Cette étape est la plus importante dans le script puisque c'est ici que nous allons telecharger les modèles pré-entrainés et les tokenizers associés et les instancier. Pour cela, on utilise les deux classes importées précédemment:

- AutoModelWithLMHead: Classe de modèle qui sera instanciée lorsque l'on utilisera la méthode de classe AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path). Il existe une très grande variété de modèles qui peuvent etre télécharger, chacun ayant des spécificités et permettant de résourdre des problématiques différentes. La liste des modèles est consultable [ici](https://huggingface.co/transformers/v3.0.2/model_summary.html) !

- AutoTokenizer: Classe de tokenizer qui sera instanciée lorsque l'on utilisera la méthode de classe AutoTokenizer.from_pretrained(pretrained_model_name_or_path). Le tokenizer utilisé est celui mentionné par l'utilisateur dans la méthode from_pretrained ( e.g. distillert, roberta, t5,...) Petit rappel conernant la tokenization,l'objectif du tokenizer est tout simplement de prétraiter le texte. Il va diviser le texte en mots (ou parties de mots, symboles de ponctuation, etc.) que l'on appelle **tokens**. Lors de l'utilisation d'un modèle, il faut s'assurer que le tokenizer instancié correspond au tokenizer ayant été utilisé pour entrainer le modèle.
```py
#Download and instanciate vocabulary and Fr-to-En model :
model_trad_fr_to_eng = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
tokenizer_translation = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

#Download and instanciate vocabulary and En-to-SQL model :
tokenizer_sql = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model_sql = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
```
Le code ci-dessus nous permet donc de telecharger et instancier dans un premier temps le modèle [opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en). Ce modèle va tout simplement nous permettre de convertir du francais en anglais. Puis on télécharge et instancie le modèle [t5-base-finetuned-wikiSQL](https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL qui va convertire le texte anglais en SQL.

4.Définir le *path operation decorator*:

Cette étape va faire en sorte que l'API puisse réaliser certaines actions ( dans notre cas, traduire le texte en SQL) en communiquant avec elle via des méthodes de requetes HTTP. Pour parvenir à notre fin, on va renseigner à FastApi le type de méthode (e.g. POST, GET, DELETE, PUT,...) et le chemin ( L'endroit/endpoint où se fait la requete). Voici un exemple d'operation que l'on pourait réaliser:
```py
@app.get('/')
def get_root():

    return {'Message': 'Welcome to the french SQL query translator !'}
```
Le ``` @app.get('/')``` dit à FastApi que la fonction juste en dessous est chargée de traiter les requetes qui vont vers le chemin ```/``` et qui utilisent la méthode GET, ce qui renverra le dict ```{'Message': 'Welcome to the french SQL query translator !'}```

Maintenant que nous comprenons mieux comment réaliser des opérations, nous allons pouvoir écrire la fonction qui va traduire le texte en francais en SQL: 
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
Ce que l'on fait ici est relativement simple à comprendre. Concernant la fonction en elle-meme, cette dernière prend en argument le texte francais, que j'ai désigné ici par ```query```.Au passage vous pouvez noter que dans le path de la méthode get on retrouve notre ```query``` entre ```{}``` pour indiquer à l'utilisateur qu'il a directement la possibilité de rentrer directement sa requête version texte francais dans l'url.

Comme nous l'avons dit précédemment, on cherche à convertir le texte francais en anglais puis de l'anglais vers le SQL. La procédure à réaliser est la même dans les deux cas:
- Encodage: On encode le texte avec la methode ```encode```. La ```string``` est convertit en dictionnaire de la forme suivante au sein duquel la list d'integer représente les index des tokens:
```py
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
- Generation: Génère les index des tokens pour la séquence que l'on cherche à obtenir à partir des index obtenus par le Tokenizer et l'utilisation du modèle pré-entrainé.
- Décodage: Décode les index des tokens en une nouvelle string.

Pour conclure cette partie j'insiste sur le fait que la fonction doit renvoyer un ```dict```.

5. Runner l'API

On arrive  enfin à l'étape finale qui va tout simplement consister à faire tourner en local notre API sur un serveur Uvicorn. Pour cela, il existe deux possibilités.
Vous pouvez rajouter dans le script ces 2 lignes de code:
```py
# Running the API on our local network:

if __name__ == '__main__':    
    uvicorn.run(app, host='127.0.0.1', port=8000)
```
Ou bien, vous pouvez directement lancer le serveur dans le terminal:
```console
uvicorn french_text_to_sql_query:app --reload
```
Au passage,voici le script au complet:
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
Une fois tout ceci réalisé, notre script est terminé et nous n'avons plus qu'à le faire tourner. Si tout fonctionne correctement, vous devriez voir apparaitre ceci dans votre terminal :
```console
INFO:     Started server process [31847]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:33262 - "GET / HTTP/1.1" 200 OK
```
**Attention**: L'execution peut prendre un peu de temps selon votre connexion internet car il faut dans un premier temps télécharger les modèles.

Il est désormais temps de se rendre sur notre API et de la tester ! Pour cela nous allons nous rendre au dashboard FastApi qui va nous permettre d'envoyer des requêtes à l'API: [http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

![](https://user-images.githubusercontent.com/52154100/125428785-3cd376b1-667b-49ad-a92b-e4a53f97365b.png)

6. Test de l'API

Le moment de verité est enfin arrivé ! Pour vérifier que l'API fonctionne nous allons essayer avec une requete relativement simple du type ```selectionner les magasins ayant un chiffre d'affaire supérieur à 100000``` ce qui se traduirait en SQL par la query ```SELECT shops FROM table WHERE turnover > 100 000```. En effet, tout cela reste relativement expérimental donc ne soyez pas surpris si la query renvoyée n'est pas exactement se dont vous vous attendiez. Mais tout cela reste quand meme assez prometteur pour la suite et amusant(enfin c'est mon avis).

![](https://user-images.githubusercontent.com/52154100/125432056-2a2ef1cd-2e27-4200-b17f-eb928c313555.png)

Comme vous le voyez, j'ai directement executé la requête dans le dashboard de FastApi (mais vous pouvez très bien l'exécuter dans le terminal) qui nous renvoie ceci:
```console
{
  "SQL QUERY": " SELECT Stores FROM table WHERE Turnover > 100000"
}
```


