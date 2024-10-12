import pytest
import json

from core.error import SFXError
from sfx.factory import EffectFactory
from sfx.effect import Effect


class AnEffect(Effect):
    """
    Just an Effect-like class for testing
    """

    def __init__(self, *args, **kwargs):
        self.initialized_value = args

    def run(self, *args, **kwar):
        pass


@pytest.fixture(scope="module")
def effect_list():
    return {"AnEffect": AnEffect}


@pytest.fixture(scope="module")
def create_valid_config():
    config = {
        "alohomora": [
            {"AnEffect": "this_is_a_value"},
            {"AnEffect": ["element1", "element2"]},
        ],
        "incendio": [
            {"AnEffect": "this_must_be_the_first_value"},
            {"AnEffect": ["element1", "element2", "element3"]},
            {"AnEffect": ["element1"]},
            {"AnEffect": ["element1", "element2", "element3"]},
            {"AnEffect": "this_must_be_the_last_value"},
        ],
    }
    return json.dumps(config)


@pytest.fixture(scope="module")
def create_broken_config():
    config = {"this_will_not_work": []}
    return json.dumps(config)


def test_the_happy_path(effect_list, create_valid_config, tmpdir):
    p = tmpdir.mkdir("pipotter_test").join("valid_config.json")
    p.write(create_valid_config)
    f = EffectFactory(config_file=str(p), effects_list=effect_list)
    # can we run the effects?
    for a_key in json.loads(create_valid_config):
        f.run(a_key)

    # do we have all our spells?
    assert "alohomora" in list(f.spells.keys())
    assert "incendio" in list(f.spells.keys())
    # do we have all our effects?
    assert len(f.spells["incendio"]) == 5
    # are all the queued effects in the right order?
    assert (
        f.spells["incendio"].queue[0].initialized_value[0]
        == "this_must_be_the_first_value"
    )
    assert (
        f.spells["incendio"].queue[4].initialized_value[0]
        == "this_must_be_the_last_value"
    )


def test_invalid_filename(effect_list):
    with pytest.raises(SFXError):
        f = EffectFactory(config_file="/this/will/not/work", effects_list=effect_list)


def test_broken_json(effect_list, create_broken_config, tmpdir):
    p = tmpdir.mkdir("pipotter_test").join("valid_config.json")
    p.write(create_valid_config)
    with pytest.raises(SFXError):
        f = EffectFactory(config_file=str(p), effects_list=effect_list)
