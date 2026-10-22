# CIFAR-100 image classification

A restored graduate-school notebook by **Ronak Mistry**. This project explores a compact
convolutional neural network for 100 natural-image classes.

## Method

Six convolution layers in three blocks (32, 64, 128 filters), batch normalization, L2 regularization, pooling and dropout. Training-only augmentation preserves the original rotations, shifts, and horizontal/vertical flips.

Normalize uint8 pixels once to [0, 1]. Keep 10% of the supplied training partition for
validation using a fixed seeded permutation. Train with sparse categorical cross entropy;
use the provided test partition only after fitting. Sparse labels replace one-hot targets
without changing the objective. No model tuning uses test results.
## Setup and verification

Use Python **3.12** in an isolated environment. CPU execution is the default.
Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python scripts/execute_notebooks.py
```

On Windows, activate with `.venv\Scripts\activate`. The notebook runner starts a fresh
kernel for each supported root notebook and writes executed copies and HTML to
`artifacts/notebooks/`. Open the HTML locally to read the rendered walkthrough.
Root notebooks default to generated fixtures and perform no downloads.
Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt`
records the complete tested environment. The lock targets CPython 3.12 on macOS
Apple Silicon and Linux; optional platform packages use environment markers.
CI installs the lock, runs fixture tests, executes notebooks, and uploads the rendered results.

## Evidence and preservation

**No real-data reproduction has been performed in this restoration.** Fixture metrics
validate code paths only. They do not establish model quality or recover historical results.
See [verification](docs/verification.md) for checks and limitations, and
[inventory](docs/inventory.json) for the starting commit, original files, and missing resources.
The original notebook sources and saved outputs are preserved in [historical/](historical/).
They use obsolete APIs and are not supported execution entry points.

Generated data, features, models, reports, and notebook outputs stay under ignored
`artifacts/` or `data/` directories. Review dataset permissions before adding any data to Git.

## Quick run

```bash
python classifier.py smoke --output artifacts/smoke
```

Generated random pixels and labels exercise the **full original CNN architecture** for one
epoch, validation, held-out fixture evaluation, error plots, and model save/load equality.
The fixture has 18 training, 2 validation, and 8 test examples. Its accuracy is meaningless.

## Supply real data later

Provide `data/cifar100.npz` with `x_train`, `y_train`, `x_test`, `y_test` arrays:

- Raw uint8 images shaped `(N, 32, 32, 3)`.
- Integer labels 0–99, one per image; `(N,)` and `(N, 1)` labels are supported.
- Preserve the official training/test split when preparing the file from the original dataset.

A cached MNIST NPZ already follows this schema. For CIFAR-100, convert a locally held
original dataset to these arrays; automated acquisition is deliberately absent.

```bash
python classifier.py train --data data/cifar100.npz --epochs 25 --batch-size 64 --output artifacts/cifar100
```

Outputs include `model.keras`, `results.json`, `split_indices.npz`, and `errors.png`.
The report records the seed, configuration, library versions, dataset SHA-256, split sizes,
training/validation history, final test loss/accuracy, and confusion matrix. Error examples
use integer class IDs. Load models with `tensorflow.keras.models.load_model` and apply the
same normalization before inference.

## Results and limitations

Historical saved outputs are unverified; they remain in `historical/`. The restored run
has not reproduced real-data accuracy. Changed dependencies and a separate validation
partition mean a future rerun may differ from the coursework output.

The old notebook used test data as training-time validation. The restored pipeline removes
that exposure. Vertical flips are retained to preserve the original approach, but may create
unrealistic examples for some classes; document any later augmentation changes as a separate
experiment. Augmentation is disabled automatically for validation, test, and inference.

## References and attribution

Original author attribution is preserved in the historical source and Git history.
The existing MIT license is retained.
References: [Keras](https://keras.io/), [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
