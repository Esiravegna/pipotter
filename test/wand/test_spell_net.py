import numpy as np
import pytest

from wand.spell_net2.model import SpellNet


@pytest.fixture(scope="module")
def random_image():
    return np.random.rand(224, 224, 3)


def test_create_model():
    m = SpellNet()
    assert m


def test_a_random_prediction(random_image):
    prediction = SpellNet().classify(np.array(random_image))
    assert len(prediction) == 6  # Six classes so far
    assert "lumos" in prediction.keys()
