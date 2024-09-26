import pytest
from unittest.mock import patch

from wand.spellscontainer import SpellsContainer, SpellSigil


@pytest.fixture(scope="module")
def list_of_lines():
    return (
        [
            [619, 224, 761, 567],
            [573, 7, 583, 621],
            [136, 87, 169, 211],
            [162, 607, 783, 729],
        ],
        [
            [131, 290, 765, 328],
            [59, 51, 196, 687],
            [80, 97, 703, 313],
            [12, 47, 52, 259],
            [32, 32, 428, 54],
        ],
        [[85, 3, 646, 236], [89, 67, 462, 138], [28, 66, 385, 439]],
    )


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

    lines = [[[459, 228, 463, 223]]]

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


# Test cases for the SpellSigil class
class TestSpellSigil:

    def test_add_line(self):
        sigil = SpellSigil()
        assert len(sigil) == 0

        sigil.add(10, 20, 30, 40)
        assert len(sigil) == 1
        assert sigil.history == [[10, 20, 30, 40]]

    def test_no_duplicate_lines(self):
        sigil = SpellSigil()
        sigil.add(10, 20, 30, 40)
        sigil.add(10, 20, 30, 40)  # Adding the same line should not increase history
        assert len(sigil) == 1

    def test_negative_coordinates_handling(self):
        sigil = SpellSigil()
        sigil.add(-10, -20, 30, 40)
        assert sigil.history == [
            [0, 0, 30, 40]
        ]  # Negative values should be clipped to 0

    def test_get_box_empty_history(self):
        sigil = SpellSigil()
        with pytest.raises(
            ValueError,
            match="The bounding boxes cannot be extracted from an empty history",
        ):
            sigil.get_box()

    def test_get_box(self):
        sigil = SpellSigil(pixels_margin=5)
        sigil.add(10, 20, 30, 40)
        sigil.add(15, 25, 35, 45)
        assert sigil.get_box() == (5, 15, 40, 50)  # Bounding box with margin


# Test cases for the SpellsContainer class
class TestSpellsContainer:

    @pytest.fixture
    def container(self):
        return SpellsContainer(expiration_time=5)

    def test_add_sigil(self, container):
        container["spell1"] = (10, 20, 30, 40)
        assert len(container) == 1

    def test_expiration_staleness(self, container):
        with patch("time.time", return_value=1000):
            container["spell1"] = (10, 20, 30, 40)
        with patch("time.time", return_value=1006):  # 6 seconds later
            assert container.is_stale()

    def test_not_stale(self, container):
        with patch("time.time", return_value=1000):
            container["spell1"] = (10, 20, 30, 40)
        with patch("time.time", return_value=1004):  # 4 seconds later
            assert not container.is_stale()

    def test_auto_clear_stale_container(self, container):
        with patch("time.time", return_value=1000):
            container["spell1"] = (10, 20, 30, 40)
        with patch("time.time", return_value=1006):  # 6 seconds later
            assert container.auto_clear()
            assert len(container) == 0

    def test_auto_clear_active_container(self, container):
        with patch("time.time", return_value=1000):
            container["spell1"] = (10, 20, 30, 40)
        with patch("time.time", return_value=1004):  # 4 seconds later
            assert not container.auto_clear()
            assert len(container) == 1

    def test_reset_container(self, container):
        container["spell1"] = (10, 20, 30, 40)
        container.reset()
        assert len(container) == 0
        assert container.last_update_time is None

    def test_get_box_empty_container(self, container):
        with pytest.raises(
            ValueError, match="Cannot proceed with an empty sigils history"
        ):
            container.get_box()

    def test_get_box_non_empty_container(self, container):
        container["spell1"] = (10, 20, 30, 40)
        assert container.get_box() == (0, 10, 40, 50)
