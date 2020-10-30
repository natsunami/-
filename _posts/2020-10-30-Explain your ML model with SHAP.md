---
layout: post
title: Explain your ML model with SHAP
subtitle: Predict is good, explain is better
cover-img: /assets/img/shap_background.jpg
thumbnail-img: /assets/img/shap_logo_white.png
share-img:
tags: [SHAP, Explainability]
---

Avant de rentrer dans le vif du sujet, il me semble fondamental de s'interesser à la notion d'interpretabilité/explicabilité ( _Ces deux termes seront considérés comme synonymes dans cet article_) et ce qu'elle signifie lorsque l'on tend à l'appliquer au machine learning (*ML*). 

Miller (2017) définit l'interprétabilité comme _la faculté grâce à laquelle un être humain peut comprendre la cause d'une décision_. Dès lors, si nous cherchons à appliquer cette définition au ML, l'interpretabilité d'un modèle consiste au niveau de compréhension qu'un individu peut avoir dans sa prédiction. Ainsi,  explicable  aboutitmeilleur  compréhension des données


si le modèle n'est pas explicable alors il est considéré comme étant une boite-noire (black-box). Il parait evident de penser qu'à choisir entre un modele peu interpretable et un autre parfaitement compréhensible, il parait intuitif de choisir le modèle à au niveau d'explicabilité pour la simple est bonne raison que nous pouvons tirer parti de cette compréhension et obtenir des informations à haute valeur ajoutée. Pour étayer mes propos je vais utiliser l'exemple suivant: _Imaginons que l'on veuille prédire le prix d'un appartement parisien ? Quelles sont les features/caractéristiques permettant d'expliquer le prix prédit, et à quels sont leur importances dans la prédiction ? Par exemple, nous pourrions apprendre que l'arrondissement dans lequel se situe l'appartement est un critère determinant dans la prédiction du modèle (e.g: Un appartement dans le 16e sera prédit plus cher qu'un appartement dans le 20e par exemple). De meme, nous pourrions constater que la présence d'un balcon ou non influe également sur la prédiction. 

Je vous accorde que ces examples peuvent sembler relativement intuitif mais cela ne veut pas dire pour autant que ces décisions prisent par le modèle ne doivent pas etre expliquées. Reprenons l'exemple du secteur immobilier. Imaginons un conseiller faisant une estimation de bien en s'appuyant sur la décision d'un modèle de ML. Et bien ce dernier doit  etre capable de justifier aux propriétaires les raisons de cette estimation sans quoi cette dernière pourrait s'averer caduque. 

L'exemple que nous venons juste de developper qui nous ammène  a considerer l'aspect légal. En effet, l’[article 22](https://www.cnil.fr/fr/reglement-europeen-protection-donnees/chapitre3#Article22) du RGPD prévoit des règles pour éviter que l’homme ne subisse des décisions émanant uniquement de machines:
>La personne concernée a le droit de ne pas faire l'objet d'une décision fondée exclusivement sur un traitement automatisé, y compris le profilage, produisant des effets juridiques la concernant ou l'affectant de manière significative de façon similaire.
Dès lors, Les modèles sans explication risquent d’entraîner une sanction qui peut s’élever à 20 000 000 d’euros ou, dans le cas d’une entreprise, à 4% du chiffre d’affaires mondial total de l’exercice précédent (le montant le plus élevé étant retenu).

Je pense que vous comprenez désormais l'importance de pouvoir expliquer un modèle de ML et les enjeux associés. Cependant, à ce stade nous ne savons toujours pas comment interepreter notre modèle...Pas de panique , c'est ce que nous allons voir juste à présent grâce à la librairie [SHAP](https://shap.readthedocs.io/en/latest/index.html) (SHapley Additive exPlanations)! 


Développé par [Lundberg and Lee (2016)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf), SHAP est une librairie permettant d'expliquer chaque prédiction d'un modèle de ML. Pour cela, SHAP s'appuie sur la theorie des jeux en utilisant le concept de [valeur de Shapley](https://fr.wikipedia.org/wiki/Valeur_de_Shapley.




    
