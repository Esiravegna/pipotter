import pytest

from core.error import SFXError
from sfx.audio.effect import AudioEffect

class MockPlayer(object):
    """
    Just a mock for the player
    """

    def play(self, filename):
        pass


class MockPlayerError(object):
    """
    Just a mock for the player when raises an error
    """

    def play(self, filename):
        raise SFXError()


@pytest.fixture(scope="module")
def a_player():
    return MockPlayer()


@pytest.fixture(scope="module")
def a_broken_player():
    return MockPlayerError()


@pytest.fixture(scope="module")
def a_valid_file():
    return pytest.__file__


@pytest.fixture(scope="module")
def not_a_valid_file():
    return "/this/will/never/work"


def test_happy_path(a_player, a_valid_file):
    fx = AudioEffect(a_player, a_valid_file)
    fx.run()


def test_invalid_file(a_player, not_a_valid_file):
    with pytest.raises(SFXError):
        fx = AudioEffect(a_player, not_a_valid_file)


def test_faulty_palyer_response(a_broken_player, a_valid_file):
    fx = AudioEffect(a_broken_player, a_valid_file)
    with pytest.raises(SFXError):
        fx.run()
