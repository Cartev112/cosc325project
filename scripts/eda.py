"""Create two ingredient heatmaps using only the prepared training partition."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


def ingredient_profiles(records):
    cuisines = sorted({record["cuisine"] for record in records})
    vocabulary = sorted({item for record in records for item in record["ingredients"]})
    rows = {name: i for i, name in enumerate(cuisines)}
    columns = {name: i for i, name in enumerate(vocabulary)}
    counts = np.zeros((len(cuisines), len(vocabulary)))
    sizes = Counter(record["cuisine"] for record in records)
    for record in records:
        for item in set(record["ingredients"]):
            counts[rows[record["cuisine"]], columns[item]] += 1
    return cuisines, vocabulary, counts, np.array([sizes[name] for name in cuisines])


def distinctive_ingredients(counts, sizes):
    # Half-count smoothing keeps ratios finite when an ingredient is absent elsewhere.
    inside = (counts + 0.5) / (sizes[:, None] + 1)
    outside = (counts.sum(axis=0) - counts + 0.5) / (sizes.sum() - sizes[:, None] + 1)
    log_ratio = np.log2(inside / outside)
    selections = []
    for row, size in enumerate(sizes):
        eligible = np.flatnonzero(counts[row] >= max(10, np.ceil(0.02 * size)))
        ranked = eligible[np.argsort(-log_ratio[row, eligible], kind="stable")]
        selections.append([int(col) for col in ranked[:2] if log_ratio[row, col] > 0])
    return log_ratio, selections


def cuisine_similarity(counts, sizes):
    prevalence = counts / sizes[:, None]
    # Recipe-level IDF reduces the influence of ubiquitous ingredients such as salt.
    idf = np.log((sizes.sum() + 1) / (counts.sum(axis=0) + 1)) + 1
    similarity = cosine_similarity(prevalence * idf)
    distance = np.clip(1 - similarity, 0, 2)
    np.fill_diagonal(distance, 0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average",
                                optimal_ordering=True))
    return similarity, order


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/eda"))
    args = parser.parse_args()
    source = Path("data/prepared/train.json")
    content = source.read_bytes()
    records = json.loads(content)
    cuisines, vocabulary, counts, sizes = ingredient_profiles(records)
    log_ratio, selections = distinctive_ingredients(counts, sizes)
    similarity, order = cuisine_similarity(counts, sizes)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = [name.replace("_", " ").title().replace(" Us", " US") for name in cuisines]

    # Keep each ingredient once, in the order of the cuisine that selected it.
    selected = list(dict.fromkeys(col for row in selections for col in row))
    if not selected:
        raise ValueError("No ingredients meet the minimum support threshold.")
    fig, ax = plt.subplots(figsize=(17, 9))
    limit = 6
    plot = ax.imshow(log_ratio[:, selected], aspect="auto", cmap="RdBu_r",
                     norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(len(selected)), [vocabulary[col] for col in selected],
                  rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(len(cuisines)), labels)
    fig.colorbar(plot, ax=ax, shrink=0.75, extend="both",
                 label="Log₂ prevalence ratio: cuisine vs. remaining recipes\n+1 = twice as prevalent; −1 = half as prevalent (smoothed)")
    fig.tight_layout()
    save_figure(fig, args.output, "distinctive_ingredients")

    fig, ax = plt.subplots(figsize=(12, 10))
    plot = ax.imshow(similarity[np.ix_(order, order)], cmap="YlGnBu", vmin=0, vmax=1)
    ordered_labels = [labels[i] for i in order]
    ax.set_xticks(range(len(order)), ordered_labels, rotation=55, ha="right")
    ax.set_yticks(range(len(order)), ordered_labels)
    fig.colorbar(plot, ax=ax, shrink=0.75, label="Cosine similarity (1 = identical profile)")
    fig.tight_layout()
    save_figure(fig, args.output, "cuisine_similarity")

    pairs = sorted([{"cuisines": [cuisines[i], cuisines[j]], "similarity": float(similarity[i, j])}
                    for i in range(len(cuisines)) for j in range(i)],
                   key=lambda pair: pair["similarity"], reverse=True)
    summary = {"source": source.as_posix(), "source_sha256": hashlib.sha256(content).hexdigest(),
               "recipes": len(records), "cuisines": cuisines, "vocabulary_size": len(vocabulary),
               "distinctive_ingredients": {
                   name: [{"ingredient": vocabulary[col], "recipes": int(counts[row, col]),
                           "prevalence": float(counts[row, col] / sizes[row]),
                           "log2_ratio": float(log_ratio[row, col])} for col in selections[row]]
                   for row, name in enumerate(cuisines)},
               "most_similar_pairs": pairs[:10],
               "similarity_matrix": similarity.tolist(),
               "cluster_order": [cuisines[i] for i in order]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved two heatmaps (PNG + SVG) and summary.json to {args.output}")
    print("Most similar pairs:", pairs[:3])


if __name__ == "__main__":
    main()
