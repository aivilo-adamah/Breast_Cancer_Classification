# Project Reference / Référence du projet

🇬🇧 [English](#english) · 🇫🇷 [Français](#français)

---

<a id="english"></a>
## English

This file summarizes the main models used in the project and explains how threshold selection was done during evaluation.

### Models Used

#### 1. Baseline CNN

Run name: `baseline_cnn_seed42`

The original custom CNN built from scratch for `50x50` IDC patches. It uses:

- 3 convolutional blocks
- batch normalization
- ReLU activations
- max pooling
- a small fully connected classifier
- dropout before the final output layer

Concretely, the network is:

1. `ConvBlock(3 -> 32)`
2. `ConvBlock(32 -> 64)`
3. `ConvBlock(64 -> 128)`
4. flatten
5. `Linear(128 * 6 * 6 -> 256)`
6. `ReLU`
7. `Dropout`
8. `Linear(256 -> 1)`

Training setup:

- patient-level `train / val / test` split
- class-weighted `BCEWithLogitsLoss`
- basic geometric augmentation: horizontal flip, vertical flip, 90-degree rotations
- pixel scaling to `[0, 1]`

#### 2. Improved Baseline CNN

Run name: `baseline_cnn_improved_seed42`

Same custom CNN architecture as the baseline, with an improved preprocessing and training setup:

- train-split mean/std normalization
- light color jitter during training (brightness, contrast, color intensity)
- `ReduceLROnPlateau` learning-rate scheduler
- slightly adjusted early stopping configuration

What did not change: patient-level split, CNN architecture, class-weighted loss, evaluation protocol.

This model is the current selected from-scratch model.

Repeated runs: `baseline_cnn_improved_seed42`, `baseline_cnn_improved_seed123`, `baseline_cnn_improved_seed2024`.

Why multiple seeds were used: the patient split was kept fixed, only the training randomness changed - this checked whether the improved baseline was stable across runs.

Stability summary across the 3 improved-baseline runs:

- selected thresholds ranged from `0.70` to `0.80`
- fixed-threshold (`0.50`) test `F1`: `0.8030 ± 0.0025`
- test `AP`: `0.8993 ± 0.0126`
- test `ROC-AUC`: `0.9465 ± 0.0053`
- tuned-threshold test `F1`: `0.8215 ± 0.0104`

Interpretation: the improved baseline performed consistently across the 3 seeds; the variation is small enough to support the claim that the model is reasonably stable. Among the three runs, `baseline_cnn_improved_seed123` was the strongest single run.

#### 3. Pretrained ResNet18 Comparison

Run name: `resnet18_transfer_seed42`

Setup:

- pretrained `ResNet18` backbone from `torchvision`
- ImageNet classification head replaced with a binary classifier head
- fine-tuned on the IDC dataset
- resized input patches from `50x50` to `128x128`
- ImageNet normalization

Classifier head: `Dropout` => `Linear(in_features -> 1)`.

Important note: this was direct fine-tuning from the start - the full model was trainable, the backbone was not frozen first.

### What The Model Actually Learns

During training, the model does not learn a hard rule like "predict 1 if score ≥ 0.50, else 0". Instead, it learns to produce a logit for each patch. After applying a sigmoid, that logit becomes a probability-like score between `0` and `1`.

Example: patch A => `0.91`, patch B => `0.63`, patch C => `0.18`.

The threshold is only used **after** the model has already produced these scores.

### How The Best Threshold Was Chosen

The threshold was selected in the evaluation stage, not in the training stage.

Workflow:

1. Train the model once and save the best checkpoint.
2. Run that checkpoint on the validation split.
3. Collect one predicted probability for every validation patch.
4. Sweep several thresholds from `0.10` to `0.90`.
5. Compute metrics at each threshold.
6. Pick the threshold that gives the best validation `F1`.

The selected threshold is the best threshold among the tested values, for `F1`, on the validation set only. The test set is used only after the threshold has already been selected.

### Why Validation Was Used Instead Of Test

Threshold tuning is a model-selection decision. If the threshold were chosen directly on the test set, the test results would no longer be a clean final evaluation.

- validation: choose the operating point
- test: report final performance

### What Threshold Changes

At threshold `0.50`, the model is more aggressive about predicting IDC: higher recall, lower precision, more false positives.

At a higher threshold such as `0.70` or `0.75`, the model becomes more conservative: higher precision, fewer false positives, lower recall.

So the "best" threshold depends on the goal:

- lower threshold if catching as many IDC patches as possible is the priority
- higher threshold if a more balanced `F1` score or fewer false alarms is the priority

### Important Nuance

The selected threshold is the best *tested* threshold, not necessarily the exact mathematical optimum. A finer sweep could find a slightly better nearby value. For this project, the current sweep is sufficient for a proper evaluation.

---

<a id="français"></a>
## Français

Ce document résume les principaux modèles utilisés dans le projet et explique comment le seuil de décision a été choisi lors de l'évaluation.

### Modèles utilisés

#### 1. CNN de base (Baseline)

Nom du run : `baseline_cnn_seed42`

Le CNN custom original, entraîné from scratch sur des patches `50x50`. Il utilise :

- 3 blocs convolutionnels
- batch normalization
- activations ReLU
- max pooling
- un petit classifieur entièrement connecté
- du dropout avant la couche de sortie finale

Concrètement, le réseau est :

1. `ConvBlock(3 -> 32)`
2. `ConvBlock(32 -> 64)`
3. `ConvBlock(64 -> 128)`
4. flatten
5. `Linear(128 * 6 * 6 -> 256)`
6. `ReLU`
7. `Dropout`
8. `Linear(256 -> 1)`

Configuration d'entraînement :

- split `train / val / test` au niveau patient
- `BCEWithLogitsLoss` pondérée par classe
- augmentation géométrique simple : flip horizontal, flip vertical, rotations de 90°
- mise à l'échelle des pixels dans `[0, 1]`

#### 2. CNN de base amélioré

Nom du run : `baseline_cnn_improved_seed42`

Même architecture CNN que la baseline, avec un prétraitement et un entraînement améliorés :

- normalisation moyenne/écart-type calculée sur le split d'entraînement
- léger color jitter pendant l'entraînement (luminosité, contraste, intensité de couleur)
- scheduler de taux d'apprentissage `ReduceLROnPlateau`
- configuration d'early stopping légèrement ajustée

Ce qui n'a pas changé : le split par patient, l'architecture CNN, la loss pondérée par classe, le protocole d'évaluation.

C'est le modèle from-scratch actuellement retenu.

Runs répétés : `baseline_cnn_improved_seed42`, `baseline_cnn_improved_seed123`, `baseline_cnn_improved_seed2024`.

Pourquoi plusieurs seeds : le split patient est resté fixe, seul l'aléa d'entraînement a changé - cela permet de vérifier la stabilité du modèle amélioré d'un run à l'autre.

Résumé de stabilité sur les 3 runs :

- seuils sélectionnés entre `0.70` et `0.80`
- `F1` test à seuil fixe (`0.50`) : `0.8030 ± 0.0025`
- `AP` test : `0.8993 ± 0.0126`
- `ROC-AUC` test : `0.9465 ± 0.0053`
- `F1` test à seuil ajusté : `0.8215 ± 0.0104`

Interprétation : le modèle amélioré est stable sur les 3 seeds ; la variation est suffisamment faible pour affirmer une stabilité raisonnable. Parmi les trois runs, `baseline_cnn_improved_seed123` est le plus performant.

#### 3. ResNet18 pré-entraîné (comparaison)

Nom du run : `resnet18_transfer_seed42`

Configuration :

- backbone `ResNet18` pré-entraîné (`torchvision`)
- tête de classification ImageNet remplacée par une tête binaire
- fine-tuné sur le dataset IDC
- patches redimensionnés de `50x50` à `128x128`
- normalisation ImageNet

Tête de classification : `Dropout` => `Linear(in_features -> 1)`.

Remarque importante : fine-tuning direct dès le début - le modèle entier était entraînable, le backbone n'a pas été gelé au préalable.

### Ce que le modèle apprend réellement

Pendant l'entraînement, le modèle n'apprend pas une règle dure du type « prédire 1 si le score ≥ 0.50, sinon 0 ». Il apprend à produire un logit pour chaque patch. Après application d'une sigmoïde, ce logit devient un score proche d'une probabilité, entre `0` et `1`.

Exemple : patch A => `0.91`, patch B => `0.63`, patch C => `0.18`.

Le seuil n'est utilisé qu'**après** que le modèle a déjà produit ces scores.

### Comment le meilleur seuil a été choisi

Le seuil a été sélectionné lors de l'étape d'évaluation, pas pendant l'entraînement.

Déroulé :

1. Entraîner le modèle une fois et sauvegarder le meilleur checkpoint.
2. Exécuter ce checkpoint sur le split de validation.
3. Collecter une probabilité prédite pour chaque patch de validation.
4. Balayer plusieurs seuils de `0.10` à `0.90`.
5. Calculer les métriques à chaque seuil.
6. Choisir le seuil donnant le meilleur `F1` en validation.

Le seuil retenu est le meilleur parmi les valeurs testées, pour le `F1`, uniquement sur le jeu de validation. Le jeu de test n'est utilisé qu'après cette sélection.

### Pourquoi la validation plutôt que le test

Le réglage du seuil est une décision de sélection de modèle. Si le seuil était choisi directement sur le jeu de test, les résultats de test ne constitueraient plus une évaluation finale propre.

- validation : choix du point de fonctionnement
- test : mesure de la performance finale

### Ce que change le seuil

À un seuil de `0.50`, le modèle est plus agressif pour prédire l'IDC : rappel plus élevé, précision plus faible, plus de faux positifs.

À un seuil plus élevé, comme `0.70` ou `0.75`, le modèle devient plus prudent : précision plus élevée, moins de faux positifs, rappel plus faible.

Le « meilleur » seuil dépend donc de l'objectif :

- seuil plus bas si la priorité est de détecter un maximum de patches IDC
- seuil plus élevé si la priorité est un `F1` plus équilibré ou moins de fausses alertes

### Nuance importante

Le seuil sélectionné est le meilleur *parmi ceux testés*, pas nécessairement l'optimum mathématique exact. Un balayage plus fin pourrait trouver une valeur voisine légèrement meilleure. Pour ce projet, le balayage actuel est suffisant pour une évaluation correcte.
