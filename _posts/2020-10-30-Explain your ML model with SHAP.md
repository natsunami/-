---
layout: post
title: Explain your ML model with SHAP
subtitle: Making predictions is fun, explaining them is even better
cover-img: /assets/img/shap_background.jpg
thumbnail-img: /assets/img/shap_logo_white.png
share-img:
tags: [SHAP, Explainability]
---

## Introduction ##

Avant de rentrer dans le vif du sujet, il me semble fondamental de s'interesser à la notion d'interpretabilité/explicabilité ( _Ces deux termes seront considérés comme synonymes dans cet article_) et ce qu'elle signifie lorsque l'on tend à l'appliquer au machine learning (*ML*). 

Miller (2017) définit l'interprétabilité comme _la faculté grâce à laquelle un être humain peut comprendre la cause d'une décision_. Dès lors, si nous cherchons à appliquer cette définition au ML, l'interpretabilité d'un modèle consiste au niveau de compréhension qu'un individu peut avoir dans sa prédiction. Ainsi,  explicable  aboutitmeilleur  compréhension des données

## Expliquer, à quelle finalité ? ##

si le modèle n'est pas explicable alors il est considéré comme étant une boite-noire (black-box). Il parait evident de penser qu'à choisir entre un modele peu interpretable et un autre parfaitement compréhensible, il parait intuitif de choisir le modèle à au niveau d'explicabilité pour la simple est bonne raison que nous pouvons tirer parti de cette compréhension et obtenir des informations à haute valeur ajoutée. Pour étayer mes propos je vais utiliser l'exemple suivant: _Imaginons que l'on veuille prédire le prix d'un appartement parisien ? Quelles sont les features/caractéristiques permettant d'expliquer le prix prédit, et à quels sont leur importances dans la prédiction ? Par exemple, nous pourrions apprendre que l'arrondissement dans lequel se situe l'appartement est un critère determinant dans la prédiction du modèle (e.g: Un appartement dans le 16e sera prédit plus cher qu'un appartement dans le 20e par exemple). De meme, nous pourrions constater que la présence d'un balcon ou non influe également sur la prédiction. 

Je vous accorde que ces examples peuvent sembler relativement intuitif mais cela ne veut pas dire pour autant que ces décisions prisent par le modèle ne doivent pas etre expliquées. Reprenons l'exemple du secteur immobilier. Imaginons un conseiller faisant une estimation de bien en s'appuyant sur la décision d'un modèle de ML. Et bien ce dernier doit  etre capable de justifier aux propriétaires les raisons de cette estimation sans quoi cette dernière pourrait s'averer caduque. 

L'exemple que nous venons juste de developper qui nous ammène  a considerer l'aspect légal. En effet, l’[article 22](https://www.cnil.fr/fr/reglement-europeen-protection-donnees/chapitre3#Article22) du RGPD prévoit des règles pour éviter que l’homme ne subisse des décisions émanant uniquement de machines:
>La personne concernée a le droit de ne pas faire l'objet d'une décision fondée exclusivement sur un traitement automatisé, y compris le profilage, produisant des effets juridiques la concernant ou l'affectant de manière significative de façon similaire.
Dès lors, Les modèles sans explication risquent d’entraîner une sanction qui peut s’élever à 20 000 000 d’euros ou, dans le cas d’une entreprise, à 4% du chiffre d’affaires mondial total de l’exercice précédent (le montant le plus élevé étant retenu).

Je pense que vous comprenez désormais l'importance de pouvoir expliquer un modèle de ML et les enjeux associés. Cependant, à ce stade nous ne savons toujours pas comment interepreter notre modèle...Pas de panique , c'est ce que nous allons voir juste à présent grâce à la librairie [SHAP](https://shap.readthedocs.io/en/latest/index.html)! 

## SHAP (SHapley Additive exPlanations) ##

### Qu'est ce que c'est ? ###

Développé par [Lundberg and Lee (2016)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf), SHAP est une librairie permettant d'expliquer chaque prédiction d'un modèle de ML. Pour cela, SHAP s'appuie sur la theorie des jeux en utilisant le concept de [valeur de Shapley](https://fr.wikipedia.org/wiki/Valeur_de_Shapley).

L'idée est la suivante: Pour chaque feature de chaque exemple du jeu de données va etre calculé les valeurs de Shapley correspondantes:

![](https://raw.githubusercontent.com/natsunami/website/3adf860daf5e4ccba3983e8f131bcf9a78c53bf1/assets/img/shap_value_formula.svg)

En calculant ces valeurs nous allons pouvoir expliquer la prédiction dans la mesure où si l'on somme toutes les valeurs de shapley pour un exemple donné à la prédiction moyenne (de tout les examples), nous obtenons bien la sorti du modèle pour cet exemple:

![](https://raw.githubusercontent.com/natsunami/website/b4b8d28c5e11b6286e65cf91cdd69abd020ef2af/assets/img/shap_value_additivity_1.svg)

Ce qu'il faut retenir et comprendre c'est que les valeurs de shapley vont permettre de determiner l'effet de chaque variable dans la prédiction. Plus la valeur de shapley est elevée ( en valeur absolue), plus l'effet de la variable est important.

![](https://github.com/natsunami/website/blob/master/assets/img/shap_value_additivity2.png)

Si ce que je viens de dire n'est pas clair, alors peut etre qu'avec cet exemple ca le sera ! Reprenons notre exemple  un de prédiction immobilière. Imaginez un appartement dont la valeur est prédite à 530 000 €. L'appartement à une **superficie** de 75m*2, possède un **balcon** et est situé dans le 16e **arrondissement**. Par ailleurs, il a été calculé que le prix moyen d'un logement est de 500 000€. Notre appartement est donc 30 000€ plus cher que le prix moyen prédit et l'objectif est d'expliquer cette différence. Et bien il est tout a fait probable que la superficie contribue à hauteur de 15 000€ , la présence d'un balcon de 5 000€ et l'arrondissement à 10 000€. Ces valeurs sont les valeurs de shapley.(_Note: Dans le cadre d'une classification les valeurs de shapley augmentent/diminuent la probabilité moyenne prédite_).

### Comment calculer une valeur de shapley ? ###

S'il faudrait retenir une chose, ca serait la suivante: La veleur de shapley est la contribution marginale moyenne d'un feature dans toutes les coalitions possibles.

Pour comprendre la définition citée précédemment nous allons chercher à évaluer la contribution de l'arrondissement lorsqu'elle est ajoutée à la coalition 'superficie - balcon'. Dans l'exemple précedent nous avions prédit le prix d'un appartement en considérement sa superficie, la présence d'un balcon ou non et son arrondissement (16eme). Nous avons prédit un prix de 530 000€. Si on enlève **arrondissement** de la coalition en remplacant la valeur '16eme' par une valeur aléatoire de ce meme feature (par exemple 13eme), nous prédirons un prix de 490 000€. L'arrondissement du 16eme contribue donc à hauteur de 40 000€ (530 000€ - 490 000€).
Ainsi nous venons donc de calculer la contribution d'une valeur d'un feature dans **une seule** coalition. Maintenant l'opération doit etre répétée pour toutes les combinaisons de coalitions possibles afin de dterminer la contribution marginale moyenne (_Note: Le temps pour calculer les valeurs de shapley augmente de facon exponentielle en fonction du nombre de features).

En conclusion, on calcule pour chaque coalition le prix de l'appartement avec et sans la valeur '16eme' du feature **arrondissement** pour determiner la moyenne des differences (contribution marginale moyenne). Ce processus


