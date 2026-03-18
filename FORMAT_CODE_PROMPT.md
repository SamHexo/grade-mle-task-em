# FORMAT_CODE_PROMPT

Instructions for an AI agent to reformat a Python ML script so it can be invoked correctly from the command line — **without changing the business logic**.

---

## Objective

The script must be invocable as follows :

```bash
python script.py \
  --train-dataset-path /path/to/train.csv \
  --test-dataset-path  /path/to/test.csv \
  --output-submission-path /path/to/submission.csv \
  [--epochs 10] \
  [other optional args...]
```

---

## Instructions for the agent

You will reformat the provided Python code. **You do not change the logic** (no change to algorithm, feature engineering, default hyperparameters, or model structure). Only the CLI wiring and path handling change.

### Mandatory rules

1. **Add `argparse`** at the top of the file (if missing) with the 3 required arguments :
   ```python
   import argparse
   from pathlib import Path

   parser = argparse.ArgumentParser()
   parser.add_argument("--train-dataset-path", required=True, type=Path)
   parser.add_argument("--test-dataset-path",  required=True, type=Path)
   parser.add_argument("--output-submission-path", required=True, type=Path)
   ```

2. **Keep all existing hyperparameters** as optional arguments with their current default values. Example :
   ```python
   parser.add_argument("--epochs",     type=int,   default=<current_value>)
   parser.add_argument("--lr",         type=float, default=<current_value>)
   parser.add_argument("--batch-size", type=int,   default=<current_value>)
   ```

3. **Replace hardcoded paths** with the `args.*` variables :
   - Any read of the training dataset → `args.train_dataset_path`
   - Any read of the test dataset → `args.test_dataset_path`
   - Any write of the submission → `args.output_submission_path`

4. **Create the parent directory** before writing the submission :
   ```python
   args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
   submission.to_csv(args.output_submission_path, index=False)
   ```

5. **Wrap in `if __name__ == "__main__":`** if not already done.

---

### Special case: code with train + predict (no separate test)

If the original code uses an internal `train_test_split` to evaluate then predict on that same split, **you must adapt** as follows — this is the only allowed logic change, because the real test is provided separately :

**Before (pattern to replace) :**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
model.fit(X_train, y_train)
preds = model.predict(X_test)
# submission write on internal X_test
```

**After :**
```python
# Train on ALL of train
model.fit(X, y)

# Load real test and predict
test_df = pd.read_csv(args.test_dataset_path)
# ... same preprocessing as train ...
preds = model.predict(X_test_real)
```

> Keep all preprocessing (encoding, scaling, feature engineering) — apply it to the real test the same way as to the train.

> If the split was only used to compute a local validation score, remove it (grading is done externally). If the split was used to select a model or for early stopping, keep that logic but train the final model on the full set.

---

### Submission format

The CSV written to `--output-submission-path` must have **exactly** the columns of the competition’s `sample_submission.csv`. If the original code does not match that format, align column names without changing the values.

---

### What you must NOT do

- Change the algorithm or model used
- Modify default hyperparameters
- Add feature engineering not present in the original code
- Refactor or reorganize the code beyond what is strictly necessary
- Add unnecessary imports
- Change preprocessing logic

---

## Full example of expected output

```python
import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--train-dataset-path",      required=True, type=Path)
parser.add_argument("--test-dataset-path",        required=True, type=Path)
parser.add_argument("--output-submission-path",   required=True, type=Path)
parser.add_argument("--n-estimators", type=int,   default=100)
parser.add_argument("--max-depth",    type=int,   default=None)
args = parser.parse_args()

if __name__ == "__main__":
    train = pd.read_csv(args.train_dataset_path)
    test  = pd.read_csv(args.test_dataset_path)

    # ... feature engineering identical to the original ...
    X      = train.drop(columns=["target"])
    y      = train["target"]
    X_test = test.drop(columns=["id"])

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    model.fit(X, y)
    preds = model.predict(X_test)

    submission = pd.DataFrame({"id": test["id"], "target": preds})
    args.output_submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_submission_path, index=False)
```
