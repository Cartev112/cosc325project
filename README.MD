# Cuisine Prediction from Recipe Ingredients

## Project overview

This COSC 325 Intro to Machine Learning semester project will predict a recipe's cuisine from its ingredient list. The team will compare accessible classification models, investigate which ingredients distinguish cuisines, and analyze where predictions become difficult.

The project uses the pre-approved [Kaggle Recipe Ingredients Dataset](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset), associated with the [What's Cooking? competition](https://www.kaggle.com/c/whats-cooking/data). The original labeled data contains approximately 39,774 recipes across 20 cuisines. Each record contains a recipe ID, a cuisine label, and a variable-length list of ingredient strings. These counts and fields will be verified when the data is downloaded.

The central question is: **How accurately can classical machine-learning models identify cuisine from ingredients alone, and how does performance vary across cuisines?** A focused secondary experiment will test whether a change to features or class weighting improves the selected baseline.

This README describes planned work; model results will be added as implementation progresses.

## Scope and evaluation

- **Input:** a list of ingredient strings; the recipe ID is used only for tracking records.
- **Target:** one cuisine label per recipe.
- **Primary selection metric:** macro-F1, which gives each cuisine equal weight.
- **Additional metrics:** accuracy, per-cuisine precision and recall, and a confusion matrix.
- **Tools:** Python, pandas, NumPy, scikit-learn, and Matplotlib or Seaborn. Notebooks will support exploration, while scripts will support reproducible training and evaluation.
- **Success criteria:** outperform the majority-class baseline on accuracy and macro-F1, produce a fair comparison of learned models, and explain strengths and weaknesses with concrete examples. A failed improvement experiment is still informative if its outcome is documented.

Ingredient lists do not provide cooking instructions, ingredient quantities, or preparation order. The project will describe these as limits of the available data. Cuisine labels also simplify overlapping culinary traditions, and class imbalance can make overall accuracy misleading.

## Development milestones

The stages below follow dependency order. Dates and team assignments can be added around course deadlines; each stage should finish with a reviewable output before the next begins.

### 1. Confirm the task and set up the project

Review the assignment requirements, agree on metrics and project scope, and assign responsibilities for data preparation, modeling, and analysis. Set up a shared Python environment and record dependencies. Read the dataset documentation and the related work linked below.

**Checkpoint:** the team has a shared problem definition, agreed evaluation criteria, and a working development environment.

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

Write down the hypothesis and comparison before running the experiment. Retain the modification only if it improves validation macro-F1 without an unacceptable tradeoff in accuracy or complexity. Record negative results as well as positive ones, and stop after this planned experiment unless the team explicitly revises the scope.

**Checkpoint:** representative prediction examples, an error-analysis summary, and a documented decision about the proposed improvement.

### 7. Freeze the configuration and perform final evaluation

Lock preprocessing, features, classifier, and hyperparameters. Refit the selected pipeline on the combined training and validation data, then evaluate it once on the untouched test partition. Evaluate the predetermined baseline configurations on that same partition for the final comparison. Save test metrics, predictions, and a confusion matrix. Do not use test results to restart tuning.

**Checkpoint:** final model and baseline results, a saved inference pipeline, and conclusions supported by held-out evidence.

### 8. Prepare the report, presentation, and reproducible handoff

Explain the research question, dataset, related work, methods, evaluation design, results, and limitations. Include the model-comparison table, confusion matrix, and a few illustrative predictions. Discuss whether errors reflect shared ingredients, ambiguous labels, sparse recipes, or class imbalance without claiming causes that were not tested.

Document installation, data placement, training, evaluation, and prediction commands. Have another team member reproduce the main result from a clean environment. A small command-line ingredient-input demonstration is optional if time remains after the core analysis is complete.

**Checkpoint:** the final report and presentation are ready, and the documented workflow reproduces the main result.

## Planned repository organization

```text
data/          Local downloaded data and saved split assignments
notebooks/     Exploratory analysis and result interpretation
src/           Loading, preprocessing, training, evaluation, and prediction
results/       Metrics, plots, and experiment summaries
models/        Saved fitted pipelines
README.MD      Project overview, plan, and eventual run instructions
```

Add ignore rules for raw data, environments, and large generated artifacts before creating them. Keep source code, dependency information, experiment settings, and compact result summaries in version control. This organization is proposed; implementation files have not yet been created.

## Related work and implementation references

- **Li, B., & Wang, M. (2015). _Cuisine Classification from Ingredients_. Stanford University CS229 project report.** [Full report](https://cs229.stanford.edu/proj2015/313_report.pdf). A closely matched starting point: it compares classical classifiers on the same Yummly dataset and reports 78.4% accuracy for logistic regression. It is a course project report rather than a peer-reviewed publication.
- **Kazama, M., Sugimoto, M., Hosokawa, C., Matsushima, K., Varshney, L. R., & Ishikawa, Y. (2018). A Neural Network System for Transformation of Regional Cuisine Style. _Frontiers in ICT, 5_, Article 14.** [Article](https://doi.org/10.3389/fict.2018.00014). Uses binary ingredient vectors and a neural network on the same 20-cuisine dataset, reporting approximately 79% test accuracy. It provides inspiration for ingredient representation and analysis of overlapping cuisine styles.
- **Sharma, T., Upadhyay, U., & Bagler, G. (2020). _Classification of Cuisines from Sequentially Structured Recipes_.** [Preprint](https://arxiv.org/abs/2004.14165). Uses a different dataset with cooking processes, utensils, and sequence information; it helps explain the limits of ingredient-only classification.
- **Scikit-learn documentation:** [feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html), [pipelines](https://scikit-learn.org/stable/modules/compose.html), and [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).

Published accuracy values provide context, not directly comparable benchmarks: splits, preprocessing, and evaluation procedures may differ from this project's protocol.
