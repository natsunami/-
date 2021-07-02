---
layout: post
title: French to SQL query API using transformers | Fast-Api | Docker | Heroku
subtitle: Tired to write down SQL queries ? Lazy to learn SQL ? This article might be for you...
cover-img: /assets/img/
thumbnail-img: /assets/img/article french sql query logo.png
share-img: /assets/img/
tags: [SQL, Transformers, Hugging face, query, fast-api, docker]

---

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
