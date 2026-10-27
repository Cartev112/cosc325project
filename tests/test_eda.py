import numpy as np

from scripts.eda import cuisine_similarity, distinctive_ingredients, ingredient_profiles


def test_profiles_count_recipe_presence_and_preserve_phrases():
    records = [{"cuisine": "a", "ingredients": ["olive oil", "olive oil", "salt"]},
               {"cuisine": "b", "ingredients": ["salt"]}]
    cuisines, vocabulary, counts, sizes = ingredient_profiles(records)
    assert cuisines == ["a", "b"]
    assert vocabulary == ["olive oil", "salt"]
    np.testing.assert_array_equal(counts, [[1, 1], [0, 1]])
    np.testing.assert_array_equal(sizes, [1, 1])


def test_distinctiveness_uses_prevalence_and_support():
    counts = np.array([[40, 20, 1], [20, 40, 0]])
    ratios, selected = distinctive_ingredients(counts, np.array([100, 100]))
    assert selected == [[0], [1]]
    assert ratios[0, 0] > 0 and ratios[1, 0] < 0
    assert np.isfinite(ratios).all()


def test_similarity_is_independent_of_cuisine_sample_size():
    counts = np.array([[40, 20, 0], [20, 10, 0], [0, 0, 30]])
    similarity, order = cuisine_similarity(counts, np.array([100, 50, 50]))
    np.testing.assert_allclose(similarity, similarity.T)
    np.testing.assert_allclose(np.diag(similarity), 1)
    assert np.isclose(similarity[0, 1], 1)
    assert similarity[0, 2] == 0
    assert sorted(order) == [0, 1, 2]
