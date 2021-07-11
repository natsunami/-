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

Bon, trêve de bavardage, il est temps de passer à l'action et je crois que vous voyez ou je veux en venir...L'API pour traduire du texte en SQL existe déja

```py
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
