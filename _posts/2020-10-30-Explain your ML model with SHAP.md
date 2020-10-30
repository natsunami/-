---
layout: post
title: Explain your ML model with SHAP
subtitle: Predict is good, explain is better
cover-img: /assets/img/shap_background.jpg
thumbnail-img: /assets/img/shap_logo_white.png
share-img:
tags: [SHAP, Explainability]
---

Avant de rentrer dans le vif du sujet, il me semble fondamental de s'interesser à la notion d'interpretabilité/explicabilité ( _Ces deux termes seront considérés comme sunonymes dans cet article_) et ce qu'elle signifie lorsque l'on tend à l'appliquer au machine learning (*ML*). 

Miller (2017) définit l'interprétabilité comme _la faculté grâce à laquelle un être humain peut comprendre la cause d'une décision_. Dès lors, si nous cherchons à appliquer cette définition au ML, l'interpretabilité d'un modèle consiste au niveau de compréhension qu'un individu peut avoir dans sa prédiction. Ainsi,  explicable  aboutitmeilleur  compréhension des données


si le modèle n'est pas explicable alors il est considéré comme étant une boite-noire (black-box). Il parait evident de penser qu'à choisir entre un modele peu interpretable et un autre parfaitement compréhensible, il paraiy intuitif de choisir le modèle à au niveau d'explicabilité pour la simple est bonne raison que nous pouvons tirer parti de cette compréhension et obtenir des informations à haute valeur ajoutée. Pour étayer mes propos je vais utiliser l'exemple suivant: _Imaginons que l'on veuille prédire le prix d'un appartement parisien ? Quelles sont les features/caractéristiques permettant d'expliquer le prix prédit, et à quels sont leur importances dans la prédiction ? Par exemple, nous pourrions apprendre que l'arrondissement dans lequel se situe l'appartement est un critère determinant dans la prédiction du modèle (e.g: Un appartement dans le 16e sera prédit plus cher qu'un appartement dans le 20e par exemple). De meme, nous pourrions constater que la présence d'un balcon ou non influe également sur la prédiction. 

Je vous accorde que ces examples peuvent sembler relativement intuitif mais cela ne veut pas dire pour autant que ces décisions ne doivent pas etre expliquées. Reprenons l'exemple du secteur immobilier. Imaginons une estimation de bien en s'appuyant sur la décision d'un modèle de ML. Il faut etre capable de justifier aux propriétaires les raisons de cette estimation. 

Cela nous ammène donc a considerer l'aspect légal.

    Aspect légal : L’article 22 du RGPD prévoit des règles pour éviter que l’homme ne subisse des décisions émanant uniquement de machines. Les modèles sans explication risquent d’entraîner une sanction qui peut s’élever à 20 000 000 d’euros ou, dans le cas d’une entreprise, à 4% du chiffre d’affaires mondial total de l’exercice précédent (le montant le plus élevé étant retenu),
    Validation du modèle : Le modèle a une bonne précision, mais nous cherchons à connaître les variables influentes afin de vérifier la cohérence avec la connaissance métier du domaine. D’autre part, pour certaines applications, nous devons également contrôler le risque du modèle, ce qui nécessite une compréhension approfondie de celui-ci,
   
