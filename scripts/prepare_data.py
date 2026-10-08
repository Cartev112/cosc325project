"""Validate, deduplicate, split, and summarize the labeled recipe data."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from sklearn.model_selection import train_test_split

SEED = 325
SOURCE = "https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset"


def normalize(text):
    return " ".join(text.lower().split())


def validate(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON list of labeled recipes.")
    cleaned, ids = [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index}: expected an object.")
        recipe_id = record.get("id")
        if type(recipe_id) is not int or recipe_id in ids:
            raise ValueError(f"Record {index}: missing, invalid, or repeated ID.")
        cuisine = record.get("cuisine")
        if not isinstance(cuisine, str) or not cuisine.strip():
            raise ValueError(f"Recipe {recipe_id}: missing cuisine label.")
        ingredients = record.get("ingredients")
        if not isinstance(ingredients, list) or not ingredients:
            raise ValueError(f"Recipe {recipe_id}: empty or invalid ingredient list.")
        if any(not isinstance(item, str) or not item.strip() for item in ingredients):
            raise ValueError(f"Recipe {recipe_id}: invalid ingredient string.")
        ids.add(recipe_id)
        cleaned.append({"id": recipe_id, "cuisine": normalize(cuisine),
                        "ingredients": sorted({normalize(item) for item in ingredients})})
    return sorted(cleaned, key=lambda record: record["id"])


def deduplicate(records):
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record["ingredients"])].append(record)
    retained, excluded = [], []
    duplicate_groups = conflicting_groups = 0
    for group in groups.values():
        group = sorted(group, key=lambda record: record["id"])
        if len(group) > 1:
            duplicate_groups += 1
        if len({record["cuisine"] for record in group}) > 1:
            # Do not choose a cuisine arbitrarily when identical inputs disagree.
            conflicting_groups += 1
            removed, reason = group, "conflicting_labels"
        else:
            # Keep the lowest ID so the choice is stable across input order.
            retained.append(group[0])
            removed, reason = group[1:], "duplicate_same_label"
        excluded.extend({"id": record["id"], "cuisine": record["cuisine"], "reason": reason}
                        for record in removed)
    return sorted(retained, key=lambda record: record["id"]), excluded, {
        "duplicate_groups": duplicate_groups, "conflicting_label_groups": conflicting_groups,
        "excluded_by_reason": dict(sorted(Counter(item["reason"] for item in excluded).items())),
    }


def split_records(records, seed=SEED):
    train, remaining = train_test_split(
        records, test_size=0.30, random_state=seed,
        stratify=[record["cuisine"] for record in records])
    validation, test = train_test_split(
        remaining, test_size=0.50, random_state=seed,
        stratify=[record["cuisine"] for record in remaining])
    splits = {"train": train, "validation": validation, "test": test}
    labels = {record["cuisine"] for record in records}
    seen_ids, seen_signatures = set(), set()
    for name, partition in splits.items():
        if {record["cuisine"] for record in partition} != labels:
            raise ValueError(f"Not every cuisine appears in {name}; more examples are required.")
        ids = {record["id"] for record in partition}
        signatures = {tuple(record["ingredients"]) for record in partition}
        if seen_ids & ids or seen_signatures & signatures:
            raise ValueError("Recipe overlap between partitions.")
        seen_ids.update(ids)
        seen_signatures.update(signatures)
        splits[name] = sorted(partition, key=lambda record: record["id"])
    return splits


def class_counts(records):
    return dict(sorted(Counter(record["cuisine"] for record in records).items()))


def training_summary(records):
    lengths = [len(record["ingredients"]) for record in records]
    ingredients = Counter(item for record in records for item in record["ingredients"])
    return {"unique_ingredients": len(ingredients),
            "ingredients_per_recipe": {"min": min(lengths), "median": statistics.median(lengths),
                                       "mean": statistics.mean(lengths), "max": max(lengths)},
            "most_common_ingredients": ingredients.most_common(15)}


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plot_training(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = Counter(record["cuisine"] for record in records)
    names = sorted(counts, key=counts.get)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(names, [counts[name] for name in names])
    axes[0].set(title="Training recipes by cuisine", xlabel="Recipe count")
    axes[1].hist([len(record["ingredients"]) for record in records], bins=30)
    axes[1].set(title="Training ingredient-list lengths", xlabel="Unique ingredients", ylabel="Recipes")
    fig.tight_layout()
    fig.savefig(output / "training_summary.png", dpi=150)
    plt.close(fig)


def prepare(source, output, report_dir, seed=SEED):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains files. Use a new output directory.")
    content = source.read_bytes()
    records = validate(json.loads(content))
    retained, excluded, duplicate_audit = deduplicate(records)
    splits = split_records(retained, seed)
    report = {"source_url": SOURCE, "source_sha256": hashlib.sha256(content).hexdigest(),
              "normalization": "lowercase, collapse whitespace, sorted unique whole ingredients",
              "duplicate_policy": "keep lowest ID for same-label groups; exclude conflicting-label groups",
              "split_method": "stratified 70/30, then stratified 50/50 of the remainder",
              "seed": seed, "raw_records": len(records), "retained_records": len(retained),
              "raw_class_counts": class_counts(records), **duplicate_audit,
              "splits": {name: {"records": len(partition), "class_counts": class_counts(partition)}
                         for name, partition in splits.items()},
              "training_summary": training_summary(splits["train"])}
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in splits.items():
        save_json(output / f"{name}.json", partition)
    save_json(output / "split_ids.json", {name: [record["id"] for record in partition]
                                         for name, partition in splits.items()})
    save_json(output / "excluded.json", excluded)
    save_json(output / "manifest.json", report)
    save_json(report_dir / "data_audit.json", report)
    plot_training(splits["train"], report_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/train.json"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--reports", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output, args.reports, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(f"Retained {report['retained_records']:,} / {report['raw_records']:,} recipes")
    print("Split sizes:", {name: info["records"] for name, info in report["splits"].items()})
    print(f"Audit and plots: {args.reports}; prepared data: {args.output}")


if __name__ == "__main__":
    main()
