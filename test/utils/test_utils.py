import numpy as np
import pytest
from random import randint
from src.core.config import settings
from src.core.utils import pad_to_square


@pytest.fixture(scope="module")
def random_image():
    w = randint(20, 500)
    h = randint(20, 500)
    return np.random.rand(w, h, 3)


def test_pad_to_square(random_image):
    new_im = pad_to_square(random_image)
    assert new_im.shape == (
        settings["PIPOTTER_SIDE_SPELL_NET"],
        settings["PIPOTTER_SIDE_SPELL_NET"],
        3,
    )
    assert isinstance(new_im, np.ndarray)
