import pytest

from wand.spellscontainer import SpellsContainer


@pytest.fixture(scope='module')
def list_of_lines():
    return ([
                [619, 224, 761, 567],
                [573, 7, 583, 621],
                [136, 87, 169, 211],
                [162, 607, 783, 729]],

            [[131, 290, 765, 328],
             [59, 51, 196, 687],
             [80, 97, 703, 313],
             [12, 47, 52, 259],
             [32, 32, 428, 54]],

            [[85, 3, 646, 236],
             [89, 67, 462, 138],
             [28, 66, 385, 439]])


def test_happy_path(list_of_lines):
    spells = SpellsContainer()
    for i, item in enumerate(list_of_lines):
        for a_dot in item:
            spells[i] = a_dot
    assert spells.get_box() == (2, 22, 775, 697)


def test_exceptions_validations():
    spells = SpellsContainer()
    with pytest.raises(ValueError):
        spells.get_box()
    with pytest.raises(ValueError):
        spells[0] = 0


def test_inverted_path_edge_case_single_line():
    spells = SpellsContainer()

    lines = [[
        [459, 228, 463, 223]
    ]]

    for i, item in enumerate(lines):
        for a_dot in item:
            spells[i] = a_dot
    assert spells.get_box() == (449, 213, 473, 238)


def test_all_the_same_sigils():
    same_length_sigils = [
        [[352, 202, 358, 211]],
        [[118, 316, 118, 316]],
        [[514, 274, 514, 274]],
    ]
    spells = SpellsContainer()
    for i, item in enumerate(same_length_sigils):
        for a_dot in item:
            spells[i] = a_dot
    assert spells.get_box() == (342, 192, 368, 221)
