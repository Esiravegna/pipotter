import pytest

from core.error import SFXError
from sfx.lights.effect import LightEffect


class MockController(object):
    """
    Just a mock for the controller with the basic commands
    """

    def cancel(self, key):
        pass

    def repeat(self, seq, rep):
        pass


class MockLightBulb(object):
    """
    Just a mock for the light object with some commands
    """

    def brightness(self, payload, group):
        pass

    def warmness(self, payload, group):
        pass

    def fade_up(self, group):
        pass


@pytest.fixture(scope="module")
def a_controller():
    return MockController()


@pytest.fixture(scope="module")
def valid_json():
    return '[{"group": "1","command": "brightness","payload": "50"},{"group": "1","command": "warmness","payload": ' \
           '"50"},{"group": "1","command": "fade_up"}]'


@pytest.fixture(scope="module")
def broken_json():
    return '[{}{}"group": "1","command": "brightness","payload": "50"},{"group": "1","command": "warmness","payload": ' \
           '"50"},{"group": "1","command": "fade_up"}]'


@pytest.fixture(scope="module")
def invalid_command_json():
    return '[{"group": "1","command": "brightness","payload": "50"},{"group": "1","command": "warmness","payload": ' \
           '"50"},{"group": "1","command": "this_will_not_work","payload": ""}]'


@pytest.fixture(scope="module")
def a_bulb():
    return MockLightBulb()


def test_happy_path(a_controller, a_bulb, valid_json):
    fx = LightEffect(valid_json, a_controller, a_bulb, time_to_sleep=0)
    fx.run()


def test_invalid_json_will_work_but_log_a_error(a_controller, a_bulb, invalid_command_json):
    fx = LightEffect( invalid_command_json, a_controller, a_bulb, time_to_sleep=0)
    fx.run()


def test_a_broken_json_will_fail(a_controller, a_bulb, broken_json):
    with pytest.raises(SFXError):
        fx = LightEffect(broken_json,  a_controller, a_bulb, time_to_sleep=0)

