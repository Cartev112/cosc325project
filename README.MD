# Cuisine Prediction from Recipe Ingredients

## Project overview

This COSC 325 Intro to Machine Learning semester project will predict a recipe's cuisine from its ingredient list. The project will compare accessible classification models, investigate which ingredients distinguish cuisines, and analyze where predictions become difficult.

The project uses the pre-approved [Kaggle Recipe Ingredients Dataset](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset), associated with the [What's Cooking? competition](https://www.kaggle.com/c/whats-cooking/data). The original labeled data contains approximately 39,774 recipes across 20 cuisines. Each record contains a recipe ID, a cuisine label, and a variable-length list of ingredient strings. These counts and fields will be verified when the data is downloaded.

The central question is: **How accurately can classical machine-learning models identify cuisine from ingredients alone, and how does performance vary across cuisines?** A focused secondary experiment will test whether a change to features or class weighting improves the selected baseline.

Project setup and initial data preparation are implemented. Model training remains the next stage.

## Current status and quick start

The setup/data-preparation responsibility covers the environment, download, audit, cleanup, and fixed splits. The implementation uses two small scripts with plain functions and concise comments. No model has been trained or scored yet.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root:

```powershell
uv sync --locked
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run pytest -q
```

Python 3.12 is selected in `.python-version`; `uv.lock` records the exact dependency versions. The environment lives in `.venv`. NumPy is installed through scikit-learn; pandas is not required for this initial workflow.

The downloader gets `train.json` directly from the approved Kaggle dataset and leaves an existing download unchanged. If Kaggle requires login, download the ZIP from the [dataset page](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset), extract the labeled `train.json`, and place it at `data/raw/train.json` before running preparation. The source listing currently identifies its license as unknown, so downloaded recipe files stay outside Git.

### Preparation rules and measured results

The initial run validated all 39,774 labeled records across 20 cuisines. Validation stops on malformed records or repeated IDs. Ingredients are lowercased, whitespace is collapsed, and repeated ingredients within a recipe are removed. Whole phrases are preserved: `olive oil` remains one ingredient. Synonyms such as `garlic` and `garlic cloves` remain distinct.

Duplicate detection compares sorted sets of normalized ingredients, independent of the original ingredient order. Same-label groups retain the lowest-ID recipe; all members of conflicting-label groups are excluded. This removed 516 redundant same-label records and 30 records from 14 conflicting-label groups, leaving 39,228 unique recipes. Raw data is preserved, and every excluded ID and reason is recorded locally. This defines a cleaner benchmark and means results should not be directly compared with papers using the full original dataset.

| Partition | Recipes | Approximate share |
| --- | ---: | ---: |
| Training | 27,459 | 70% |
| Validation | 5,884 | 15% |
| Test | 5,885 | 15% |

The split uses seed `325`, stratification by cuisine, and a consistent ID ordering. All 20 cuisines occur in each partition. Recipe IDs and normalized ingredient-list signatures are disjoint across partitions. Exact duplicates are addressed; near-duplicate recipes may still remain.

The preparation script saves:

- `data/prepared/train.json`, `validation.json`, and `test.json`: normalized records with the same `id`, `cuisine`, and `ingredients` fields as the source.
- `data/prepared/split_ids.json`: the fixed partition assignments.
- `data/prepared/excluded.json`: excluded IDs and reasons.
- `data/prepared/manifest.json`: source SHA-256, seed, preparation rules, and audit counts.
- [results/data_audit.json](results/data_audit.json): a compact shareable audit, including per-partition class counts.
- [results/training_summary.png](results/training_summary.png): training cuisine frequencies and ingredient-list lengths.

Training data contains 6,051 unique normalized ingredients, with a median of 10 ingredients per recipe. Detailed ingredient summaries and plots use training data only. Validation and test records are reserved for the agreed evaluation stages.

Preparation refuses to overwrite a nonempty output directory. To investigate an alternative preparation without changing the shared split, use separate output and report directories:

```powershell
uv run python scripts/prepare_data.py --output data/alternative --reports data/alternative-reports
```

The tests cover malformed data, phrase preservation, duplicate/conflict handling, deterministic splits, partition separation, and overwrite protection. The initial run passed all 15 tests.

### Handoff to modeling

Use the locked environment and the saved splits. Run preparation once per fresh checkout; keep seed `325` and the source file unchanged to reproduce the assignments. Load a partition with standard Python JSON tools:

```python
import json
from pathlib import Path

train = json.loads(Path("data/prepared/train.json").read_text(encoding="utf-8"))
ingredients = [record["ingredients"] for record in train]
labels = [record["cuisine"] for record in train]
```

Fit ingredient vocabulary and TF-IDF weights on training data only, preserving whole ingredient strings as features. Use validation data for model comparison and tuning. Keep the test partition for the frozen final evaluation. The next deliverable is the dummy and first learned baselines.

## Scope and evaluation

- **Input:** a list of ingredient strings; the recipe ID is used only for tracking records.
- **Target:** one cuisine label per recipe.
- **Primary selection metric:** macro-F1, which gives each cuisine equal weight.
- **Additional metrics:** accuracy, per-cuisine precision and recall, and a confusion matrix.
- **Tools:** Python, pandas, NumPy, scikit-learn, and Matplotlib or Seaborn. Notebooks will support exploration, while scripts will support reproducible training and evaluation.
- **Success criteria:** outperform the majority-class baseline on accuracy and macro-F1, produce a fair comparison of learned models, and explain strengths and weaknesses with concrete examples. A failed improvement experiment is still informative if its outcome is documented.

Ingredient lists do not provide cooking instructions, ingredient quantities, or preparation order. The project will describe these as limits of the available data. Cuisine labels also simplify overlapping culinary traditions, and class imbalance can make overall accuracy misleading.

## Development milestones

The stages below follow dependency order. Dates can be added around course deadlines; each stage should finish with a reviewable output before the next begins.

### 1. Confirm the task and set up the project

Review the assignment requirements, agree on metrics and project scope, and assign responsibilities for data preparation, modeling, and analysis. Set up a shared Python environment and record dependencies. Read the dataset documentation and the related work linked below.

**Checkpoint:** the problem definition and evaluation criteria are established, and the development environment works.

### 2. Download and audit the data

Validate record structure, missing or empty ingredients, class counts, ingredient-list lengths, and vocabulary size. Look for duplicate recipes using an order-independent ingredient-list signature and inspect duplicates with conflicting cuisine labels. Document a consistent policy for invalid records and conflicting labels. Keep identical ingredient lists within one split, or remove redundant copies before splitting.

Create a small exploratory analysis with cuisine-frequency plots and examples of distinctive or commonly shared ingredients. Use training data for subsequent feature-driven exploration once the split is established. Keep downloaded data local and document its source and usage terms.

**Checkpoint:** a reproducible loading and audit script, an EDA notebook, and a short account of data quality and class imbalance.

### 3. Establish the evaluation protocol

Create a fixed, approximately 70% training / 15% validation / 15% test split, stratified by cuisine and respecting duplicate groups. Save the recipe IDs assigned to each partition and the random seed. All models must use the same partitions. Use a held-out portion of the labeled data as the project test set; the competition's unlabeled test file cannot provide local evaluation metrics.

Fit vocabulary, TF-IDF weights, feature selection, and other learned preprocessing on training data only. Use a [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html) to keep feature fitting and classifier training together. The validation set supports model selection; the test set remains untouched until the final configuration is fixed.

**Checkpoint:** saved split assignments and a documented protocol that prevents data leakage.

### 4. Build an end-to-end baseline pipeline

Start with a dummy classifier that always predicts the most common training cuisine. Then build a sparse ingredient representation and train Multinomial Naive Bayes and logistic regression. Preserve complete ingredient phrases such as `olive oil` as individual features rather than accidentally splitting every ingredient into separate words. Begin with lowercase and whitespace normalization, retaining meaningful ingredient distinctions.

Use binary ingredient-presence features as the initial representation. Compare these with TF-IDF weights over the same ingredient vocabulary once the basic pipeline works. Save validation metrics and enough configuration information to reproduce each run.

**Checkpoint:** raw JSON can be loaded, transformed, used for training, and evaluated through a repeatable workflow; the first baseline results are recorded.

### 5. Compare models and tune a limited set of parameters

Compare the following models using the shared evaluation protocol:

| Model | Purpose |
| --- | --- |
| Majority-class dummy | Establish the minimum reference performance |
| Multinomial Naive Bayes | Provide a fast learned baseline for sparse nonnegative features |
| Logistic regression | Provide a strong, interpretable linear baseline |
| Linear SVM | Test an alternative linear classifier suited to sparse features |

Keep model and feature comparisons controlled: compare classifiers with the same representation, and compare representations with the same classifier. Tune a small predefined set of smoothing or regularization values on validation data. Select the leading configuration by macro-F1, considering accuracy, runtime, and simplicity when scores are close.

**Checkpoint:** a comparison table with validation accuracy, macro-F1, training time, and the selected configuration.

### 6. Analyze errors and run one focused improvement experiment

Inspect validation errors, commonly confused cuisine pairs, minority-cuisine recall, and influential ingredients from the linear models. Use these observations to choose one bounded experiment, such as class weighting, careful ingredient-name normalization, or selected ingredient-pair features. If using pairs, treat them as unordered ingredient combinations rather than adjacent text words.

Write down the hypothesis and comparison before running the experiment. Retain the modification only if it improves validation macro-F1 without an unacceptable tradeoff in accuracy or complexity. Record negative results as well as positive ones, and stop after this planned experiment unless the scope is explicitly revised.

**Checkpoint:** representative prediction examples, an error-analysis summary, and a documented decision about the proposed improvement.

### 7. Freeze the configuration and perform final evaluation

Lock preprocessing, features, classifier, and hyperparameters. Refit the selected pipeline on the combined training and validation data, then evaluate it once on the untouched test partition. Evaluate the predetermined baseline configurations on that same partition for the final comparison. Save test metrics, predictions, and a confusion matrix. Do not use test results to restart tuning.

**Checkpoint:** final model and baseline results, a saved inference pipeline, and conclusions supported by held-out evidence.

### 8. Prepare the report, presentation, and reproducible handoff

Explain the research question, dataset, related work, methods, evaluation design, results, and limitations. Include the model-comparison table, confusion matrix, and a few illustrative predictions. Discuss whether errors reflect shared ingredients, ambiguous labels, sparse recipes, or class imbalance without claiming causes that were not tested.

Document installation, data placement, training, evaluation, and prediction commands. Reproduce the main result from a clean environment. A small command-line ingredient-input demonstration is optional if time remains after the core analysis is complete.

**Checkpoint:** the final report and presentation are ready, and the documented workflow reproduces the main result.

## Repository organization

```text
scripts/       Implemented download and preparation scripts
tests/         Data-preparation tests
docs/          Initial implementation plan
data/          Local raw and prepared data (ignored)
results/       Shareable audit and training plots
pyproject.toml Dependency declarations and test configuration
uv.lock        Exact dependency versions
README.MD      Project status, commands, and development plan
```

Raw/prepared recipe data, the environment, caches, and model artifacts are ignored by Git. Source code, dependency information, and compact results can be versioned. Add modeling scripts or notebooks as their implementation begins; a separate EDA notebook is still optional follow-up work.

## Related work and implementation references

- **Li, B., & Wang, M. (2015). _Cuisine Classification from Ingredients_. Stanford University CS229 project report.** [Full report](https://cs229.stanford.edu/proj2015/313_report.pdf). A closely matched starting point: it compares classical classifiers on the same Yummly dataset and reports 78.4% accuracy for logistic regression. It is a course project report rather than a peer-reviewed publication.
- **Kazama, M., Sugimoto, M., Hosokawa, C., Matsushima, K., Varshney, L. R., & Ishikawa, Y. (2018). A Neural Network System for Transformation of Regional Cuisine Style. _Frontiers in ICT, 5_, Article 14.** [Article](https://doi.org/10.3389/fict.2018.00014). Uses binary ingredient vectors and a neural network on the same 20-cuisine dataset, reporting approximately 79% test accuracy. It provides inspiration for ingredient representation and analysis of overlapping cuisine styles.
- **Sharma, T., Upadhyay, U., & Bagler, G. (2020). _Classification of Cuisines from Sequentially Structured Recipes_.** [Preprint](https://arxiv.org/abs/2004.14165). Uses a different dataset with cooking processes, utensils, and sequence information; it helps explain the limits of ingredient-only classification.
- **Scikit-learn documentation:** [feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html), [pipelines](https://scikit-learn.org/stable/modules/compose.html), and [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).

Published accuracy values provide context, not directly comparable benchmarks: splits, preprocessing, and evaluation procedures may differ from this project's protocol.
