# Notebooks

🇬🇧 These notebooks were run on **Google Colab** against the full ~555k-patch dataset (mounted from Google Drive). They are kept as-is for reproducibility of the full-scale results reported in [`../explanation.md`](../explanation.md). For a fast, local, CPU-friendly reproduction, see [`../scripts/`](../scripts/) and the Streamlit demo instead.

| Notebook | Purpose |
|---|---|
| `01_manifest_and_patient_split.ipynb` | Build the cleaned manifest and the patient-level train/val/test split |
| `02_data_visualization.ipynb` | Sanity-check class balance, patient distribution, sample patches |
| `03_training_baseline.ipynb` | Train the first custom CNN baseline from scratch |
| `04_baseline_improvement.ipynb` | Same CNN, improved preprocessing/training (normalization, color jitter, LR scheduler) |
| `05_pretrained_resnet18.ipynb` | Train a pretrained ResNet18 comparison model |
| `06_evaluation.ipynb` | Reload checkpoints, sweep thresholds, compare all models |

🇫🇷 Ces notebooks ont été exécutés sur **Google Colab** sur le dataset complet (~555k patches, monté depuis Google Drive). Ils sont conservés tels quels pour la reproductibilité des résultats complets décrits dans [`../explanation.md`](../explanation.md). Pour une reproduction rapide et locale (CPU), voir plutôt [`../scripts/`](../scripts/) et la démo Streamlit.

| Notebook | Rôle |
|---|---|
| `01_manifest_and_patient_split.ipynb` | Construire le manifest nettoyé et le split train/val/test par patient |
| `02_data_visualization.ipynb` | Vérifier l'équilibre des classes, la distribution par patient, des exemples de patches |
| `03_training_baseline.ipynb` | Entraîner le premier CNN de base from scratch |
| `04_baseline_improvement.ipynb` | Même CNN, prétraitement/entraînement amélioré (normalisation, color jitter, scheduler) |
| `05_pretrained_resnet18.ipynb` | Entraîner un ResNet18 pré-entraîné à titre de comparaison |
| `06_evaluation.ipynb` | Recharger les checkpoints, balayer les seuils, comparer les modèles |
