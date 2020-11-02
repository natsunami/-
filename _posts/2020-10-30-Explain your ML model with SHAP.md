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

Développé par [Lundberg and Lee (2016)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf), SHAP est une librairie permettant d'expliquer chaque prédiction d'un modèle de ML. Pour cela, SHAP s'appuie sur la theorie des jeux en utilisant le concept de [valeur de Shapley](https://fr.wikipedia.org/wiki/Valeur_de_Shapley)

L'idée est la suivante: Pour chaque feature de chaque exemple du dataset vont être calculé les valeurs de Shapley \varphi_i:

![](https://raw.githubusercontent.com/natsunami/website/3adf860daf5e4ccba3983e8f131bcf9a78c53bf1/assets/img/shap_value_formula.svg)

_Avec M, le nombre de variables, S est un sous-ensemble de variables, x est le vecteur des valeurs des features de l'example à expliquer. f(x) est la prédiction utilisant les valeurs des features dans l'ensemble S qui sont marginalisées par rapport aux features qui ne sont pas inclus dans l'ensemble S._

Chacune des prédictions pour chaque example peut s'écrire comme la somme des valeurs de shapley ajoutée à la prédiction moyenne notée  \varphi_0 (valeur de base):

![](https://raw.githubusercontent.com/natsunami/website/b4b8d28c5e11b6286e65cf91cdd69abd020ef2af/assets/img/shap_value_additivity_1.svg)

Avec, _y_pred_ la valeur prédite du modèle pour cette exemple, \varphi_0 la valeur de base du model, z'\in \{0,1\}^M quand la variable est observée z'_i=1 ou inconnue z'_i=0.

Au final, Ce qui est important de retenir c'est que les valeurs de shapley représentent l'effet de chaque variable dans la prédiction (voir Fig.1). Plus la valeur de shapley est elevée (en valeur absolue), plus elle est importante dans la prédiction.

![](https://github.com/natsunami/website/blob/master/assets/img/shap_value_additivity2.png)

Figure 1: Additivité des valeurs de shapley (La somme des valeurs de shapley ajoutée à la valeur de base est égale à la prédiction)

Pour finir cette partie, quoi de mieux qu'un exemple pour illustrer mes propos! Pour cela, reprenons l'exemple de la prédiction immobilière. Imaginez un appartement dont la valeur est prédite à 530 000 €. L'appartement à une **superficie** de 75m*2, possède un **balcon** et est situé dans le 16e **arrondissement**. Par ailleurs, il a été calculé que le prix moyen d'un logement est de 500 000€. Notre appartement est donc 30 000€ plus cher que le prix moyen prédit et l'objectif est d'expliquer cette différence. Et bien il est tout a fait probable que la superficie contribue à hauteur de 15 000€ , la présence d'un balcon de 5 000€ et l'arrondissement à 10 000€. Ces valeurs sont les valeurs de shapley.(_Note: Dans le cadre d'une classification les valeurs de shapley augmentent/diminuent la probabilité moyenne prédite_).

### Comment calculer une valeur de shapley ? ###

S'il faudrait retenir une chose, ca serait la suivante: La veleur de shapley est la contribution marginale moyenne d'un feature dans toutes les coalitions possibles.

Pour comprendre la définition citée précédemment nous allons chercher à évaluer la contribution de l'arrondissement lorsqu'elle est ajoutée à la coalition 'superficie - balcon'. Dans l'exemple précedent nous avions prédit le prix d'un appartement en considérement sa superficie, la présence d'un balcon ou non et son arrondissement (16eme). Nous avons prédit un prix de 530 000€. Si on enlève **arrondissement** de la coalition en remplacant la valeur '16eme' par une valeur aléatoire de ce meme feature (par exemple 13eme), nous prédirons un prix de 490 000€. L'arrondissement du 16eme contribue donc à hauteur de 40 000€ (530 000€ - 490 000€).
Ainsi nous venons donc de calculer la contribution d'une valeur d'un feature dans **une seule** coalition. Maintenant l'opération doit etre répétée pour toutes les combinaisons de coalitions possibles afin de dterminer la contribution marginale moyenne (_Note: Le temps pour calculer les valeurs de shapley augmente de facon exponentielle en fonction du nombre de features).

En conclusion, on calcule pour chaque coalition le prix de l'appartement avec et sans la valeur '16eme' du feature **arrondissement** pour determiner la moyenne des differences (contribution marginale moyenne). Ce processus

## SHAP en exemple ##

Maintenant que nous sommes familier avec SHAP et les valeurs de shapley, nous allons pouvoir etudier un cas concret d'explicabilité de modèle. L'exemple que nous allons prendre s'appuie sur le dataset [Health Insurance Cross Sell Prediction](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).

### Descriptif ###

Travaillant pour le compte d'une entreprise d'assurance, notre objectif est de déterminer si les clients seraient potentiellement intéréssés à la souscription d'une assurance auto. Pour mener à bien cette tâche nous allons construire un modèle classifiant si oui ou non le client serait intéréssé. A notre disposition nous avons des informations sur les clients (_Ex: genre, age, sexe, région_), l'etat de leur vehicule (_Ex:age du véhicule,présence de dommages_) et leur contrat d'assurance (_Ex:somme versée par le client, moyen de contact du client_). Bien entendu, nous expliquerons les prédictions du modèle (_Ex: Pourquoi ce client a été classifié comme susceptible de souscrire à l'assurance auto ?_).

Dans cet article nous allons directement traiter de l'explicabilité avec SHAP. Si cela vous interesse, je vous invite à consulter le notebook complet [ici](https://natsunami.github.io/website/Portfolio/Insurance-cost-sell-prediction/insurance_cross_sell_prediction.html) avec les plots interactifs.

### SHAP feature importance ###

Quand on parle de feature importance, on peut penser à plusieurs choses:
- Aux poids (_weights_) dans le cadre d'une régression linéaire. En effet, les variables ayant un poids elevé sont plus importantes dans le modèle (_Notes: Uniquement si les variables sont à la même échelle_).
- Le gain d'information (_information gain_) pour les modèles arborescents (les features qui réduisent davantage l'impurity sont plus importants).

SHAP peut être utilisé pour obtenir l'importance des features sur la base des valeurs de Shapley. Plus les features ont une moyenne de |valeurs de Shapley| (_en valeur absolue_) élevée, plus elles contribuent aux prédictions.

![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/shap_plot/shap_feature_importance.png)

En s'interessant de plus près au feature importance plot, nous comprenons que les caractéristiques **Previously_insured**, **Vehicle_damage** et **Policy_sales_channel** sont les trois variables qui contribuent le plus aux prédictions.

(_Notes: L'utilisation de l'importance de la caractéristique SHAP dans un cas de haute dimensionnalité pourrait être une bonne idée pour réduire la dimensionnalité en supprimant les caractéristiques ayant une faible  moyenne des |valeurs de Shapley|_).

### SHAP summary plot ###

Le SHAP summary plot fournit des informations sur l'importance des features et leurs effets.

Chaque point du graphique est une valeur de Shapley pour chaque feature de chaque observation. La position est définie par la valeur de Shapley sur l'axe des x et les caractéristiques sur l'axe des y. Sur la droite, la barre de couleur représente la valeur du feature, de faible (_Bleu_) à élevée (_Rouge_).

![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/shap_plot/shap_summary_plot.png)

Ici, une faible valeur de **Previously_insured** (_0 : non assuré_) signifie une valeur SHAP négative qui diminue la probabilité d'être intéressé par une assurance automobile (_Rappel: la prédiction est représentée comme la somme des valeurs SHAP_). Au contraire, une valeur élevée de **Previously_insured** (_1 : assuré_) signifie une valeur SHAP positive qui augmente la probabilité d'être intéressé par une assurance automobile.

Néanmoins, nous pouvons voir que certains effets de caractéristiques sont difficiles à interpréter sur ce graphique, car nous avons effectué une mise à l'échelle sur certaines variables données. Pour une meilleure compréhension, il pourrait être utile d'explorer les effets au niveau individuel.

### SHAP individual observation ###

Le force plot fournit par SHAP permet de comprendre les effets de chaque feature pour chaque observation. Nous allons voir directement comment cela se présente en étudiant le cas de deux clients.

#### Client 1 ####
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/shap_plot/client1.png)
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/shap_plot/client1_shap_force_plot.png)

Pour ce client, nous constatons que le modèle prédit une probabilité de 0,61 qu'il soit interéssé par l'assurance et, par conséquent, qu'il soit classifié comme tel. En examinant le force plot, nous constatons que **Age**, **Policy_Sales_Channel**, **Vehicule_Damage** et **Previously_Insured** sont les principales variables qui augmentent la probabilité.

Comment cela peut-il être interprété ?
Le fait que ce client soit âgé de 21 ans, qu'il ait déjà été assuré, que son véhicule n'ait pas été endommagé et qu'il ait été contacté par la chaîne en utilisant le code 152, augmente la probabilité d'être intéressé par une assurance automobile.

(_Notes:Les valeurs continues sur le diagramme de force sont mises à l'échelle_)

#### Client 2 ####
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/shap_plot/client2.png)
![](https://raw.githubusercontent.com/natsunami/website/master/assets/img/shap_plot/client2_shap_force_plot.png)

