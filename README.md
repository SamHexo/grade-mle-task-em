# create-submission-em

Agent Emily qui run automatiquement des scripts d'entraînement ML, grade leurs soumissions, et remonte les scores à Emily à chaque step.

---

## Concept

Chaque step = un fichier Python dans le dossier `codes/`. L'agent les run dans l'ordre alphabétique, passe les chemins de données en arguments, récupère le fichier de soumission produit, le grade, et remonte le score MAE à Emily.

```
workspace/
├── codes/          ← tes scripts ML (un par step)
│   ├── model_a.py
│   ├── model_b.py
│   └── ...
├── train.csv       ← dataset d'entraînement
├── submissions_<experiment_id>/   ← créé automatiquement
│   ├── submission_model_a.csv
│   └── submission_model_b.csv
└── grid_<experiment_id>/          ← créé automatiquement
    ├── metric_model_a.json
    └── metric_model_b.json

<competition_id>/   ← dossier de compétition (ex: ventilator-pressure-prediction/)
├── test.csv                ← test public (sans labels)
├── private_test.csv        ← test privé (avec labels, pour grader)
├── sample_submission.csv   ← format attendu
└── grade.py                ← calcul du score (MAE)
```

---

## Ce que chaque script doit faire

Chaque fichier Python dans `codes/` doit :

1. Accepter ces arguments obligatoires :
   ```
   --train-dataset-path      chemin vers train.csv
   --test-dataset-path       chemin vers le test public de la compétition
   --output-submission-path  chemin où écrire la soumission (passé par l'agent)
   ```

2. Accepter `--epochs` (ou tout autre arg listé dans `additional_args`)

3. Écrire un fichier CSV à `--output-submission-path` avec exactement les colonnes de `sample_submission.csv`

```python
# Exemple minimal
import argparse, pandas as pd
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--train-dataset-path", required=True, type=Path)
p.add_argument("--test-dataset-path",  required=True, type=Path)
p.add_argument("--output-submission-path", required=True, type=Path)
p.add_argument("--epochs", type=int, default=5)
args = p.parse_args()

# ... entraîner le modèle ...

submission = pd.DataFrame({"id": test["id"], "pressure": predictions})
args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(args.output_submission_path, index=False)
```

Voir `workspace/codes/example_train.py` (sklearn) et `workspace/codes/example_train_torch.py` (PyTorch) comme exemples complets.

---

## Configuration (`test_config.yaml`)

```yaml
EXPERIMENT_ID: "local-test-001"
PROJECT_ID: "local-project"
TASK_DESCRIPTION: "..."
MAX_ITERATIONS: 10       # max scripts à runner (s'arrête si moins de scripts disponibles)

AGENT_CONFIG:
  competition_id: "ventilator-pressure-prediction"   # nom du dossier compétition
  code_folder_path: "workspace/codes"                # relatif à la racine du projet
  train_dataset_path: "workspace/train.csv"          # relatif à la racine du projet
  additional_args:
    - "--epochs"
    - "5"
```

### `MAX_ITERATIONS` et nombre de scripts

`effective_steps = min(MAX_ITERATIONS, nombre de scripts dans codes/)`

- Plus de scripts que de steps → seuls les N premiers (ordre alphabétique) sont runné
- Plus de steps que de scripts → l'agent s'arrête proprement quand tous les scripts ont tourné

### Chemins relatifs

Tous les chemins dans `AGENT_CONFIG` sont relatifs à la **racine du projet** :
- En local : le dossier contenant `custom_agent.py`
- Sur Emily : `/` (donc `workspace/codes` → `/workspace/codes`)

---

## Préparer les données de compétition

Les fichiers `test.csv`, `private_test.csv` et `sample_submission.csv` doivent être dans le dossier de la compétition. Pour les générer depuis le CSV Kaggle brut :

```bash
cd grade-mle-task-agent-em
python ventilator-pressure-prediction/prepare_data.py \
    --raw-train-csv /chemin/vers/kaggle/train.csv
```

Des fichiers synthétiques (20 breaths × 80 steps) sont déjà inclus pour les tests locaux.

---

## Lancer en local

```bash
cd grade-mle-task-agent-em

# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Copier et éditer la config
cp test_config.yaml.example test_config.yaml

# 3. Lancer
python test_agent_locally.py
```

Les webhooks sont sauvegardés dans `webhooks/` pour inspection.

---

## Outputs produits à chaque step

### `workspace/submissions_<experiment_id>/submission_<script_stem>.csv`
Le fichier de soumission produit par le script.

### `workspace/grid_<experiment_id>/metric_<script_stem>.json`
Les métriques du step :
```json
{
  "python_file": "/path/to/codes/model_a.py",
  "submission_file": "/path/to/submissions_.../submission_model_a.csv",
  "score": 0.284,
  "format_valid": true,
  "grade_message": "MAE = 0.284102",
  "execution_time_seconds": 12.4,
  "error": null
}
```

`score` est `null` si le script a crashé ou si le format de soumission est invalide.

---

## Archives ZIP/TAR dans `codes/`

L'agent extrait automatiquement les archives `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.gz` trouvées dans `codes/` avant de lancer les scripts. Tous les `.py` sont remontés à la racine de `codes/`.

> **Note macOS** : Finder ajoute des fichiers `._*` dans les zips (métadonnées AppleDouble). L'agent les détecte et les supprime automatiquement après extraction — ton vrai script `foo.py` est toujours préservé.

---

## Ajouter une compétition

1. Créer un dossier `<competition_id>/` à la racine
2. Y mettre :
   - `test.csv` — test public (colonnes features, sans labels)
   - `private_test.csv` — test privé (avec labels, pour grader)
   - `sample_submission.csv` — format attendu (ex: `id,pressure`)
   - `grade.py` — fonction `grade(submission: DataFrame, answers: DataFrame) -> float`
3. Mettre `competition_id: "<competition_id>"` dans `AGENT_CONFIG`
