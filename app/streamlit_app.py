"""Streamlit demo for the IDC breast-cancer histopathology classifier.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import emoji
import pandas as pd
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from idc_pipeline.inference import LoadedModel, load_model_for_inference, predict_patch  # noqa: E402

DEMO_MODEL_DIR = PROJECT_ROOT / "models" / "demo"
CHECKPOINT_PATH = DEMO_MODEL_DIR / "best_model.pt"
METRICS_PATH = DEMO_MODEL_DIR / "metrics.json"
EVALUATION_DIR = DEMO_MODEL_DIR / "evaluation"
SAMPLE_GALLERY_DIR = DEMO_MODEL_DIR / "sample_patches"
TSNE_GIF_PATH = DEMO_MODEL_DIR / "tsne" / "tsne_animation.gif"

st.set_page_config(
    page_title="IDC Breast Cancer Classifier",
    page_icon=emoji.emojize(":reminder_ribbon:"),
    layout="wide",
)


@st.cache_resource
def get_loaded_model() -> LoadedModel:
    return load_model_for_inference(CHECKPOINT_PATH)


@st.cache_data
def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


LANGUAGES = {
    "English": {
        "title": ":reminder_ribbon: Breast Cancer Histopathology Classifier (IDC)",
        "subtitle": (
            "Upload a 50x50 histopathology patch and get an instant IDC "
            "(Invasive Ductal Carcinoma) prediction."
        ),
        "tab_predict": "Predict",
        "tab_model": "Model",
        "tab_context": "Context & Glossary",
        "upload_label": "Upload one or more histopathology patches (PNG/JPG)",
        "threshold_label": "Decision threshold",
        "threshold_help": (
            "Probability above which a patch is classified as IDC-present. "
            "Lower = more sensitive (catches more IDC, more false alarms). "
            "Higher = more specific (fewer false alarms, may miss some IDC)."
        ),
        "result_header": "Result",
        "probability_label": "IDC probability",
        "prediction_label": "Prediction",
        "confidence_label": "Confidence",
        "class_present": "IDC present (malignant)",
        "class_absent": "IDC absent (benign)",
        "no_upload": "Upload an image to get a prediction.",
        "metrics_intro": "Metrics computed on the held-out validation/test split (patient-level, never seen during training).",
        "metric_accuracy": "Accuracy",
        "metric_precision": "Precision",
        "metric_recall": "Recall",
        "metric_f1": "F1 score",
        "metric_roc_auc": "ROC-AUC",
        "confusion_matrix": "Confusion matrix",
        "cm_actual": "Actual",
        "cm_predicted": "Predicted",
        "curves_header": "ROC & Precision-Recall curves",
        "metrics_missing": "No metrics.json found for the demo model yet.",
        "roc_curve_caption": "ROC curve (x=FPR, y=TPR)",
        "pr_curve_caption": "Precision-Recall curve",
        "about_header": "About this model",
        "about_body": (
            "This demo uses a compact CNN trained on a patient-level subsample "
            "of the IDC histopathology dataset (kept small so it trains quickly "
            "on CPU). See `explanation.md` in the repository for the full "
            "methodology, including how the decision threshold is selected and "
            "why patient-level splitting matters."
        ),
        "language_label": "Language",
    },
    "Français": {
        "title": ":reminder_ribbon: Classifieur de cancer du sein par histopathologie (IDC)",
        "subtitle": (
            "Uploadez un patch histopathologique de 50x50 et obtenez une "
            "prédiction instantanée d'IDC (Carcinome Canalaire Invasif)."
        ),
        "tab_predict": "Prédiction",
        "tab_model": "Modèle",
        "tab_context": "Contexte & Glossaire",
        "upload_label": "Uploadez un ou plusieurs patches histopathologiques (PNG/JPG)",
        "threshold_label": "Seuil de décision",
        "threshold_help": (
            "Probabilité au-delà de laquelle un patch est classé IDC-présent. "
            "Plus bas = plus sensible (détecte plus d'IDC, plus de fausses alertes). "
            "Plus haut = plus spécifique (moins de fausses alertes, peut manquer des IDC)."
        ),
        "result_header": "Résultat",
        "probability_label": "Probabilité d'IDC",
        "prediction_label": "Prédiction",
        "confidence_label": "Confiance",
        "class_present": "IDC présent (malin)",
        "class_absent": "IDC absent (bénin)",
        "no_upload": "Uploadez une image pour obtenir une prédiction.",
        "metrics_intro": "Métriques calculées sur le split de validation/test (au niveau patient, jamais vu pendant l'entraînement).",
        "metric_accuracy": "Exactitude",
        "metric_precision": "Précision",
        "metric_recall": "Rappel",
        "metric_f1": "Score F1",
        "metric_roc_auc": "ROC-AUC",
        "confusion_matrix": "Matrice de confusion",
        "cm_actual": "Réel",
        "cm_predicted": "Prédit",
        "curves_header": "Courbes ROC & Precision-Recall",
        "metrics_missing": "Aucun fichier metrics.json trouvé pour le modèle de démo pour l'instant.",
        "roc_curve_caption": "Courbe ROC (x=FPR, y=TPR)",
        "pr_curve_caption": "Courbe Precision-Recall",
        "about_header": "À propos de ce modèle",
        "about_body": (
            "Cette démo utilise un CNN compact entraîné sur un sous-échantillon "
            "du dataset IDC respectant le split par patient (volontairement "
            "réduit pour s'entraîner rapidement sur CPU). Voir `explanation.md` "
            "dans le dépôt pour la méthodologie complète, notamment le choix du "
            "seuil de décision et l'importance du split par patient."
        ),
        "language_label": "Langue",
    },
}


CONTEXT_MARKDOWN = {
    "English": r"""
### :stethoscope: Project context

**IDC** (Invasive Ductal Carcinoma) is the most common type of breast cancer. To diagnose it,
pathologists examine **histopathology slides** - thin tissue sections stained and viewed under a
microscope. A whole slide is far too large to feed to a model directly, so it is cut into small
**patches** (here, `50x50` pixel RGB tiles). The model's job is to look at one patch and decide
whether it shows IDC tissue or not.

- **Benign** (`class 0`, "IDC absent"): normal / non-cancerous tissue in that patch
- **Malignant** (`class 1`, "IDC present"): tissue showing invasive ductal carcinoma in that patch

This is a **patch-level** label, not a patient-level diagnosis - one patient's slide typically
contains both benign and malignant patches, and a full clinical diagnosis would aggregate
predictions across many patches, not rely on a single one.

### :open_book: Key vocabulary

| Term | Meaning |
|---|---|
| **Patch** | A small fixed-size crop (`50x50px`) of a whole-slide histopathology image |
| **Benign / IDC absent** | No invasive carcinoma detected in this patch |
| **Malignant / IDC present** | Invasive ductal carcinoma detected in this patch |
| **Patient-level split** | Train/val/test are split by *patient*, not by patch, so no patch from the same patient leaks across splits |
| **Logit / probability** | The raw model output (logit) is passed through a sigmoid to get a probability in `[0, 1]` |
| **Threshold** | The probability cutoff above which a patch is classified as malignant (default `0.50`, tunable) |

### :hammer_and_wrench: What preprocessing / "feature engineering" was actually done

This project uses raw pixels with a CNN, not hand-crafted features - but several preprocessing
choices matter a lot for a medical imaging task like this one:

- **Data cleaning**: every patch is validated (correct filename pattern, correct patient ID,
  correct label, readable PNG, exact `50x50` size) before it enters the pipeline - invalid samples
  are dropped and logged.
- **Patient-level splitting**: because patches from the same patient are visually correlated
  (same staining, same tissue), splitting at the *patch* level would leak information from train
  into test and make the evaluation over-optimistic. Splitting at the *patient* level avoids this.
- **Pixel scaling**: raw pixel values (`0–255`) are rescaled to `[0, 1]`.
- **Normalization**: mean/std computed **on the training split only** (never on val/test, to avoid
  leakage), then applied to all splits.
- **Light data augmentation**: horizontal/vertical flips, 90° rotations, and mild color jitter
  (brightness/contrast/color) - kept deliberately light because the patches are small and
  aggressive transforms can destroy the local texture cues that distinguish tissue types.

### :balance_scale: Class imbalance

In the full dataset, roughly **72% of patches are benign and 28% are malignant** - a real-world
imbalance, since most tissue on a slide is normal. Training a model naively on this would bias it
toward always predicting "benign" (it would still score ~72% accuracy while catching almost no
cancer). To fix this, the loss function (`BCEWithLogitsLoss`) applies a **positive class weight**
equal to `negative_count / positive_count`, so misclassifying a malignant patch is penalized more
heavily than misclassifying a benign one during training.

### :bar_chart: Why these metrics, and why not just "accuracy"

For an imbalanced medical detection problem, **accuracy is misleading** - a model that always
predicts "benign" can still look accurate. Instead this project tracks:

- **Precision** = of all patches predicted malignant, how many actually are? (cost of being wrong: false alarms)
- **Recall (sensitivity)** = of all truly malignant patches, how many did the model catch? (cost of being wrong: missed cancer)
- **F1 score** = harmonic mean of precision and recall - a single number balancing both
- **ROC-AUC** = threshold-independent ranking quality (can the model separate the two classes at all?)
- **Average Precision (AP)** = threshold-independent, and more informative than ROC-AUC on
  imbalanced data because it focuses on the positive (minority) class

The **decision threshold** is tuned by sweeping values from `0.10` to `0.90` on the **validation**
set (never the test set) and picking the one that maximizes F1 - see `explanation.md` for the full
protocol.

### :input_numbers: The confusion matrix, explained

| | Predicted: Benign | Predicted: Malignant |
|---|---|---|
| **Actual: Benign** | :check_mark_button: True Negative (TN) | :cross_mark: False Positive (FP) |
| **Actual: Malignant** | :cross_mark: False Negative (FN) | :check_mark_button: True Positive (TP) |

- **TP (True Positive)**: correctly flagged malignant tissue
- **TN (True Negative)**: correctly cleared benign tissue
- **FP (False Positive)**: benign tissue wrongly flagged as malignant - a **false alarm**
- **FN (False Negative)**: malignant tissue wrongly cleared as benign - a **missed cancer**

### :bullseye: What actually matters in a medical context

In cancer screening, **not all errors are equal**:

- A **False Negative (FN)** means real cancer tissue is missed - the patient doesn't get flagged
  for follow-up. This is the **most dangerous type of error** in a screening context: a delayed
  diagnosis can cost lives.
- A **False Positive (FP)** means healthy tissue is flagged as suspicious - this typically leads
  to unnecessary follow-up (a second look, a biopsy), which causes patient anxiety and extra cost,
  but is not directly life-threatening.

Because of this asymmetry, **recall (sensitivity)** is usually prioritized over precision in a
screening tool: it is generally considered better to over-flag (more FPs, caught by a human
reviewer downstream) than to under-flag (more FNs, cancer missed entirely). This is exactly why
this project reports the **precision/recall trade-off explicitly** rather than optimizing accuracy
alone, and why the app lets you **lower the decision threshold** to trade some precision for
higher recall when that is the priority.

> This tool is a **patch-level screening aid demo**, not a diagnostic device. Any real clinical
> workflow would combine model output with pathologist review, patient history, and additional
> tests - never a single patch-level probability in isolation.
""",
    "Français": r"""
### :stethoscope: Contexte du projet

L'**IDC** (Carcinome Canalaire Invasif) est le type de cancer du sein le plus fréquent. Pour le
diagnostiquer, les pathologistes examinent des **lames d'histopathologie** - de fines coupes de
tissu colorées et observées au microscope. Une lame entière est bien trop grande pour être donnée
directement à un modèle, elle est donc découpée en petits **patches** (ici, des tuiles RGB de
`50x50` pixels). Le rôle du modèle est d'examiner un patch et de décider s'il montre du tissu IDC
ou non.

- **Bénin** (`classe 0`, « IDC absent ») : tissu normal / non cancéreux sur ce patch
- **Malin** (`classe 1`, « IDC présent ») : tissu montrant un carcinome canalaire invasif sur ce patch

C'est un label **au niveau du patch**, pas un diagnostic au niveau du patient - la lame d'un même
patient contient en général à la fois des patches bénins et malins, et un vrai diagnostic clinique
agrégerait les prédictions sur de nombreux patches, sans jamais se fier à un seul.

### :open_book: Vocabulaire clé

| Terme | Signification |
|---|---|
| **Patch** | Une petite portion de taille fixe (`50x50px`) découpée dans une lame d'histopathologie |
| **Bénin / IDC absent** | Aucun carcinome invasif détecté sur ce patch |
| **Malin / IDC présent** | Carcinome canalaire invasif détecté sur ce patch |
| **Split par patient** | Train/val/test sont séparés par *patient*, pas par patch, pour qu'aucun patch d'un même patient ne se retrouve dans plusieurs splits |
| **Logit / probabilité** | La sortie brute du modèle (logit) passe par une sigmoïde pour donner une probabilité dans `[0, 1]` |
| **Seuil** | La probabilité au-delà de laquelle un patch est classé malin (par défaut `0.50`, ajustable) |

### :hammer_and_wrench: Le prétraitement / "feature engineering" réellement effectué

Ce projet utilise les pixels bruts avec un CNN, pas de features fabriquées à la main - mais
plusieurs choix de prétraitement comptent beaucoup pour une tâche d'imagerie médicale comme celle-ci :

- **Nettoyage des données** : chaque patch est validé (nom de fichier conforme, ID patient
  correct, label correct, PNG lisible, taille exacte `50x50`) avant d'entrer dans le pipeline - les
  échantillons invalides sont supprimés et journalisés.
- **Split par patient** : les patches d'un même patient étant visuellement corrélés (même
  coloration, même tissu), un split au niveau *patch* laisserait fuir de l'information de
  l'entraînement vers le test et rendrait l'évaluation trop optimiste. Le split au niveau *patient*
  évite ce problème.
- **Mise à l'échelle des pixels** : les valeurs brutes (`0–255`) sont ramenées dans `[0, 1]`.
- **Normalisation** : moyenne/écart-type calculés **uniquement sur le split d'entraînement**
  (jamais sur val/test, pour éviter toute fuite), puis appliqués à tous les splits.
- **Légère augmentation de données** : flips horizontal/vertical, rotations de 90°, et color
  jitter modéré (luminosité/contraste/couleur) - volontairement léger car les patches sont petits
  et des transformations trop agressives détruiraient les indices de texture locale qui
  distinguent les types de tissu.

### :balance_scale: Déséquilibre des classes

Dans le dataset complet, environ **72% des patches sont bénins et 28% sont malins** - un
déséquilibre réaliste, la majorité du tissu d'une lame étant normal. Entraîner un modèle
naïvement sur ces données le biaiserait vers prédire toujours « bénin » (il obtiendrait quand même
~72% d'accuracy tout en ratant presque tous les cancers). Pour corriger cela, la fonction de perte
(`BCEWithLogitsLoss`) applique un **poids de classe positive** égal à
`nombre_négatifs / nombre_positifs`, pour pénaliser davantage une erreur sur un patch malin qu'une
erreur sur un patch bénin pendant l'entraînement.

### :bar_chart: Pourquoi ces métriques, et pas juste "l'accuracy"

Pour un problème de détection médicale déséquilibré, **l'accuracy est trompeuse** - un modèle qui
prédit toujours « bénin » peut sembler précis. Ce projet suit plutôt :

- **Précision** = parmi tous les patches prédits malins, combien le sont réellement ? (coût d'erreur : fausses alertes)
- **Rappel (sensibilité)** = parmi tous les patches réellement malins, combien le modèle en a détecté ? (coût d'erreur : cancer manqué)
- **Score F1** = moyenne harmonique de la précision et du rappel - un seul chiffre équilibrant les deux
- **ROC-AUC** = qualité de classement indépendante du seuil (le modèle sépare-t-il bien les deux classes ?)
- **Average Precision (AP)** = indépendante du seuil, et plus informative que ROC-AUC sur des
  données déséquilibrées car elle se concentre sur la classe positive (minoritaire)

Le **seuil de décision** est ajusté en balayant des valeurs de `0.10` à `0.90` sur le jeu de
**validation** (jamais le jeu de test) et en choisissant celui qui maximise le F1 - voir
`explanation.md` pour le protocole complet.

### :input_numbers: La matrice de confusion, expliquée

| | Prédit : Bénin | Prédit : Malin |
|---|---|---|
| **Réel : Bénin** | :check_mark_button: Vrai Négatif (TN) | :cross_mark: Faux Positif (FP) |
| **Réel : Malin** | :cross_mark: Faux Négatif (FN) | :check_mark_button: Vrai Positif (TP) |

- **TP (Vrai Positif)** : tissu malin correctement détecté
- **TN (Vrai Négatif)** : tissu bénin correctement écarté
- **FP (Faux Positif)** : tissu bénin détecté à tort comme malin - une **fausse alerte**
- **FN (Faux Négatif)** : tissu malin détecté à tort comme bénin - un **cancer manqué**

### :bullseye: Ce qui compte vraiment dans un contexte médical

En dépistage du cancer, **toutes les erreurs ne se valent pas** :

- Un **Faux Négatif (FN)** signifie qu'un vrai tissu cancéreux est manqué - le patient n'est pas
  orienté vers un suivi. C'est le **type d'erreur le plus dangereux** dans un contexte de dépistage :
  un diagnostic retardé peut coûter des vies.
- Un **Faux Positif (FP)** signifie qu'un tissu sain est signalé comme suspect - cela entraîne
  généralement un suivi supplémentaire inutile (second avis, biopsie), source d'anxiété et de coût
  pour le patient, mais sans mise en danger directe.

À cause de cette asymétrie, le **rappel (sensibilité)** est généralement priorisé par rapport à la
précision dans un outil de dépistage : il est en général préférable de sur-détecter (plus de FP,
rattrapés ensuite par un relecteur humain) que de sous-détecter (plus de FN, cancer totalement
manqué). C'est exactement pourquoi ce projet expose explicitement le **compromis
précision/rappel** plutôt que d'optimiser uniquement l'accuracy, et pourquoi l'app te permet de
**baisser le seuil de décision** pour échanger un peu de précision contre plus de rappel quand
c'est la priorité.

> Cet outil est une **démo d'aide au dépistage au niveau du patch**, pas un dispositif de
> diagnostic. Tout vrai workflow clinique combinerait la sortie du modèle avec la relecture d'un
> pathologiste, l'historique du patient et des examens complémentaires - jamais une seule
> probabilité de patch isolée.
""",
}


def _emojize_strings(value: object) -> object:
    """Recursively turn ``:shortcode:`` markers into real emoji characters.

    Keeps emoji out of the source as literal unicode glyphs (which are easy to
    mis-render or accidentally corrupt in an editor) while still being able to
    use them in the UI text.
    """
    if isinstance(value, str):
        return emoji.emojize(value, language="alias")
    if isinstance(value, dict):
        return {key: _emojize_strings(item) for key, item in value.items()}
    return value


LANGUAGES = _emojize_strings(LANGUAGES)
CONTEXT_MARKDOWN = _emojize_strings(CONTEXT_MARKDOWN)


def render_probability_gauge(probability: float, threshold: float) -> None:
    st.progress(probability)
    st.caption(f"{probability:.1%}  (threshold: {threshold:.2f})")


def render_showcase(text: dict) -> None:
    """Visual banner shown at the top of the Context tab, before any text."""
    gallery_files = sorted(SAMPLE_GALLERY_DIR.glob("*.png")) if SAMPLE_GALLERY_DIR.exists() else []
    if not TSNE_GIF_PATH.exists() and not gallery_files:
        return

    if TSNE_GIF_PATH.exists():
        gif_columns = st.columns([1, 5, 1])
        gif_columns[1].image(str(TSNE_GIF_PATH), width="stretch")

    malignant_files = [f for f in gallery_files if f.name.startswith("class1")]
    benign_files = [f for f in gallery_files if f.name.startswith("class0")]

    def render_row(files: list[Path], label: str) -> None:
        if not files:
            return
        st.caption(f"**{label}**")
        row_columns = st.columns(len(files))
        for column, file_path in zip(row_columns, files):
            column.image(str(file_path), width="stretch")

    render_row(malignant_files, text["class_present"])
    render_row(benign_files, text["class_absent"])


def main() -> None:
    with st.sidebar:
        language = st.selectbox(
            emoji.emojize(":globe_with_meridians: Language / Langue", language="alias"),
            list(LANGUAGES.keys()),
        )
    text = LANGUAGES[language]

    st.title(text["title"])
    st.write(text["subtitle"])

    if not CHECKPOINT_PATH.exists():
        st.error(
            f"No checkpoint found at `{CHECKPOINT_PATH}`. "
            "Run `python scripts/train_demo_model.py` first, or see the README."
        )
        return

    loaded_model = get_loaded_model()
    metrics = get_metrics()

    tab_context, tab_model, tab_predict = st.tabs(
        [text["tab_context"], text["tab_model"], text["tab_predict"]]
    )

    with tab_predict:
        with st.sidebar:
            threshold = st.slider(
                text["threshold_label"],
                min_value=0.05,
                max_value=0.95,
                value=float(loaded_model.default_threshold),
                step=0.05,
                help=text["threshold_help"],
            )

        uploaded_files = st.file_uploader(
            text["upload_label"],
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.write(text["no_upload"])
        else:
            columns = st.columns(min(len(uploaded_files), 3))
            for index, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                prediction = predict_patch(image, loaded_model, threshold=threshold)

                with columns[index % len(columns)]:
                    st.image(image, caption=uploaded_file.name, width=180)
                    label_text = (
                        text["class_present"]
                        if prediction.predicted_label == 1
                        else text["class_absent"]
                    )
                    if prediction.predicted_label == 1:
                        st.error(f"**{label_text}**")
                    else:
                        st.success(f"**{label_text}**")

                    render_probability_gauge(prediction.probability, threshold)
                    st.caption(f"{text['confidence_label']}: {prediction.confidence:.1%}")

    with tab_model:
        st.subheader(text["tab_model"])
        st.caption(text["metrics_intro"])

        test_metrics = metrics.get("test_metrics", {})
        if test_metrics:
            metric_columns = st.columns(5)
            metric_columns[0].metric(text["metric_accuracy"], f"{test_metrics.get('accuracy', 0):.3f}")
            metric_columns[1].metric(text["metric_precision"], f"{test_metrics.get('precision', 0):.3f}")
            metric_columns[2].metric(text["metric_recall"], f"{test_metrics.get('recall', 0):.3f}")
            metric_columns[3].metric(text["metric_f1"], f"{test_metrics.get('f1', 0):.3f}")
            roc_auc = test_metrics.get("roc_auc")
            metric_columns[4].metric(text["metric_roc_auc"], f"{roc_auc:.3f}" if roc_auc else "n/a")

            st.markdown(f"**{text['confusion_matrix']}**")
            tp, tn = test_metrics.get("tp", 0), test_metrics.get("tn", 0)
            fp, fn = test_metrics.get("fp", 0), test_metrics.get("fn", 0)
            st.table(
                {
                    f"{text['cm_predicted']}: {text['class_absent']}": [tn, fn],
                    f"{text['cm_predicted']}: {text['class_present']}": [fp, tp],
                }
            )
            st.caption(
                f"Rows = {text['cm_actual']} "
                f"({text['class_absent']}, {text['class_present']})"
            )
        else:
            st.warning(text["metrics_missing"])

        roc_curve_path = EVALUATION_DIR / "test_roc_curve.csv"
        pr_curve_path = EVALUATION_DIR / "test_precision_recall_curve.csv"
        if roc_curve_path.exists() or pr_curve_path.exists():
            st.markdown(f"**{text['curves_header']}**")
            curve_columns = st.columns(2)
            if roc_curve_path.exists():
                roc_df = pd.read_csv(roc_curve_path)
                curve_columns[0].line_chart(roc_df.set_index("fpr")["tpr"])
                curve_columns[0].caption(text["roc_curve_caption"])
            if pr_curve_path.exists():
                pr_df = pd.read_csv(pr_curve_path)
                curve_columns[1].line_chart(pr_df.set_index("recall")["precision"])
                curve_columns[1].caption(text["pr_curve_caption"])

        st.divider()
        st.subheader(text["about_header"])
        st.write(text["about_body"])
        st.markdown(
            f"""
- **Architecture**: `{loaded_model.architecture}`
- **Default threshold**: `{loaded_model.default_threshold:.2f}`
- **Best validation epoch**: `{loaded_model.checkpoint_metadata.get('best_epoch')}`
- **Selection metric**: `{loaded_model.checkpoint_metadata.get('best_metric_name')}` = `{loaded_model.checkpoint_metadata.get('best_metric_value')}`
- **Positive class weight (train)**: `{loaded_model.checkpoint_metadata.get('pos_weight')}`
"""
        )

    with tab_context:
        render_showcase(text)
        st.divider()
        st.markdown(CONTEXT_MARKDOWN[language])


if __name__ == "__main__":
    main()
