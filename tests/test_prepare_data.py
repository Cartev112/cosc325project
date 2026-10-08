import json

import pytest

from scripts.prepare_data import deduplicate, prepare, split_records, validate


def recipe(recipe_id, cuisine="italian", ingredients=None):
    return {"id": recipe_id, "cuisine": cuisine,
            "ingredients": ingredients if ingredients is not None else [f"ingredient {recipe_id}"]}


def test_normalization_preserves_phrases_and_collapses_repeats():
    records = validate([recipe(1, ingredients=["  Olive   Oil ", "olive oil", "Sea Salt"])])
    assert records[0]["ingredients"] == ["olive oil", "sea salt"]


@pytest.mark.parametrize("records", [[], {}, [None], [recipe(True)],
    [recipe(1), recipe(1)], [recipe(1, cuisine="")],
    [recipe(1, ingredients=[])], [recipe(1, ingredients=[""])],
    [recipe(1, ingredients=[None])], [{"id": 1, "ingredients": ["salt"]}]])
def test_invalid_data_fails(records):
    with pytest.raises(ValueError):
        validate(records)


def test_duplicate_and_conflict_policy():
    records = validate([recipe(2, ingredients=["salt", "rice"]),
                        recipe(1, ingredients=["rice", "salt"]),
                        recipe(3, "thai", ["oil"]), recipe(4, "italian", ["oil"])])
    retained, excluded, audit = deduplicate(records)
    assert [record["id"] for record in retained] == [1]
    assert {item["id"]: item["reason"] for item in excluded} == {
        2: "duplicate_same_label", 3: "conflicting_labels", 4: "conflicting_labels"}
    assert audit["conflicting_label_groups"] == 1


def sample_records():
    return validate([recipe(i, "italian" if i < 100 else "thai") for i in range(200)])


def test_split_is_reproducible_complete_and_disjoint():
    records = sample_records()
    splits = split_records(records)
    assert splits == split_records(records)
    assert {name: len(partition) for name, partition in splits.items()} == {
        "train": 140, "validation": 30, "test": 30}
    all_ids = [record["id"] for partition in splits.values() for record in partition]
    signatures = [tuple(record["ingredients"]) for partition in splits.values() for record in partition]
    assert len(set(all_ids)) == len(set(signatures)) == len(records)
    for partition in splits.values():
        assert {record["cuisine"] for record in partition} == {"italian", "thai"}


def test_insufficient_class_examples_fail():
    with pytest.raises(ValueError):
        split_records(validate([recipe(1, "italian"), recipe(2, "thai")]))


def test_preparation_writes_outputs_and_protects_existing_split(tmp_path):
    source, output, reports = tmp_path / "train.json", tmp_path / "prepared", tmp_path / "reports"
    source.write_text(json.dumps(sample_records()), encoding="utf-8")
    audit = prepare(source, output, reports)
    assert audit["raw_records"] == 200
    assert audit["training_summary"]["unique_ingredients"] == 140
    assert (reports / "training_summary.png").is_file()
    assert json.loads((output / "split_ids.json").read_text())["train"] == [
        record["id"] for record in json.loads((output / "train.json").read_text())]
    with pytest.raises(FileExistsError):
        prepare(source, output, reports)
