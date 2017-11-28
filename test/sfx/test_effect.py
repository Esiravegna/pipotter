import pytest
from sfx.effect import Effect
from sfx.effect_container import EffectContainer


@pytest.fixture(scope="module")
def this_is_an_effect():
    return Effect()

@pytest.fixture(scope="module")
def this_is_a_working_effect():
    class AWorkignEffect(Effect):
        def run(self, *args, **kwargs):
            return True
    return AWorkignEffect()


def test_create_container_ok(this_is_an_effect):
    container = EffectContainer()
    an_effect = this_is_an_effect
    container.append(an_effect)
    assert len(container) == 1


def test_create_container_broken_element():
    container = EffectContainer()
    an_effect = "This is a Flowerpot"
    with pytest.raises(ValueError):
        container.append(an_effect)


def test_running_things(this_is_a_working_effect):
    container = EffectContainer()
    an_effect = this_is_a_working_effect
    container.append(an_effect)
    assert container.run()
