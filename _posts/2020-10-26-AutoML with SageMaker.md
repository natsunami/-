---
layout: post
title: AutoML with SageMaker
subtitle: 
cover-img: /assets/img/network.jpeg
thumbnail-img: /assets/img/sagemaker_pic.png
share-img: /assets/img/network.jpeg
tags: [aws, sagemaker, automl]
---

Comme premier post je voulais partager avec vous une découverte que j’ai faite en travaillant sur l'écosystème d'AWS (Amazon Web Services): [AWS SageMaker Autopilot](https://aws.amazon.com/fr/sagemaker/autopilot/).

Pour ceux qui ne connaissent pas Autopilot, il s’agit d’une solution permettant de réaliser du machine learning de façon automatisé, dit **AutoML**. Jusqu’à très récemment la génération de modèle automatisé était un processus offrant peu de visibilité sur leur construction (**Black-box**), ce qui m’a fortement intrigué sur le fonctionnement d’Autopilot. Ainsi, j’ai décidé de tester Autopilot sur un cas concret de prédiction de prix et de le comparer à ce que j’aurai réalisé (data preparation, choix du modèle, hypertuning). Mon dévolu s’est porté sur le dataset **Airbnb Paris** qui recense les airbnb de Paris, l’objectif étant de prédire le prix.

Vous pourrez trouver à l’adresse suivante mon notebook avec l’ensemble des informations détaillant le processus de la création du modèle ainsi que le lien permettant de télécharger le dataset: [Airbnb Price Prediction](https://natsunami.github.io/website/Portfolio/Airbnb/Airbnb-paris-Price-Prediction.html).

Autopilot de son coté suit une pipeline consistant à analyser les données, faire du feature engineering puis du model tuning. Ce qui est plutôt impressionnant c’est qu’au cours de ce processus Autopilot va générer 2 notebooks tout en détaillant le type de preprocessing réalisé ( Imputer, scaler, encoder, etc.), les différents modèles proposés ( ex : linear, booster,etc..) et les meilleurs hyperparamètres pour chaque modèle ( le processus a duré environ 1 heure sur ce dataset ):

1. [Exploration des données](https://natsunami.github.io/website/Portfolio/Airbnb/sagemaker_autopilot_data_exploration_notebook.html)

2. [Modèles candidats](https://natsunami.github.io/website/Portfolio/Airbnb/sagemaker_autopilot_candidate_notebook.html)

Au final de mon coté je suis parvenu à une MSE de 0.144 sans hypertuning et Autopilot à une MSE de 0.131 avec hypertuning ( voir les hyperparamètres dans mon notebook) sur un validation set. (Note: Appliquer les hyperparametres d'Autopilot sur mon modèle a permis de diminuer ma MSE à 0.134).

Dès lors, les résultats obtenus par Autopilot poussent à la réflexion. En effet, nous pouvons supposer que les solutions de types AutoML vont de plus en plus être accessibles aux personnes n'ayant pas forcément de background en Machine Learning, et qui plus est, fournissent de très bons résultats pour ce type de problématiques. Ainsi, la question se pose de comment évoluera le travail du machine learning engineer au cours des années à venir ? Superviser la création automatique des modèles ? Se concentrer sur des problématiques spécifiques la où l'AutoML ne fonctionne pas aujourd'hui (ex: deep learning) ?

Qu'en pensez vous ?

J'espère que vous aviez trouvé ce post intéressant. Dorénavant je posterai davantage de contenu. Par ailleurs, le prochain post s'intéressera à l'explicabilité de modèle appliqué au secteur de l'assurance en utilisant SHAP !
