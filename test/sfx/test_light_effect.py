import pytest
from unittest.mock import patch, MagicMock
import json
from src.sfx.lights.effect import (
    LightEffect,
    SFXError,
)  # Assuming your code is in light_effect.py


@pytest.fixture
def mock_neopixel():
    with patch("neopixel.NeoPixel") as mock_strip:
        yield mock_strip


@pytest.fixture
def mock_board():
    with patch("board.D18", 18):  # Simulate board.D18 as integer pin 18
        yield


@pytest.fixture
def mock_pixel_order():
    with patch("neopixel.GRB", "GRB"):  # Simulate neopixel.GRB as string 'GRB'
        yield


def test_valid_json_config(mock_neopixel, mock_board, mock_pixel_order):
    """
    Test that the NeoPixel strip is initialized correctly from valid JSON configuration.
    """
    json_config = """
    {
        "gpio_pin": "D18",
        "num_leds": 16,
        "brightness": 0.5,
        "pixel_order": "GRB",
        "commands": [
            {"command": "rollcall_cycle", "payload": {"wait": 0.1}}
        ]
    }
    """
    effect = LightEffect(json_config)

    # Ensure the NeoPixel object was created with the correct parameters
    mock_neopixel.assert_called_once_with(
        18, 16, brightness=0.5, auto_write=False, pixel_order="GRB"
    )

    # Run the effect (will use the mock)
    effect.run()
    assert (
        mock_neopixel.return_value.show.call_count > 0
    ), "Expected show() to be called"


def test_missing_required_params(mock_neopixel, mock_board, mock_pixel_order):
    """
    Test missing required configuration parameters raises SFXError.
    """
    json_config = """
    {
        "brightness": 0.5,
        "pixel_order": "GRB",
        "commands": []
    }
    """  # Missing 'gpio_pin' and 'num_leds'

    with pytest.raises(SFXError, match="Missing required configuration parameter"):
        LightEffect(json_config)


def test_invalid_json_format(mock_neopixel, mock_board, mock_pixel_order):
    """
    Test that invalid JSON format raises SFXError.
    """
    invalid_json = """{ "gpio_pin": "D18", "num_leds": 16, """  # Incomplete JSON

    with pytest.raises(SFXError, match="Invalid JSON configuration"):
        LightEffect(invalid_json)


def test_invalid_gpio_or_pixel_order(mock_neopixel, mock_board, mock_pixel_order):
    """
    Test that invalid GPIO pin or pixel order raises SFXError.
    """
    json_config = """
    {
        "gpio_pin": "INVALID_PIN",
        "num_leds": 16,
        "brightness": 0.5,
        "pixel_order": "INVALID_ORDER",
        "commands": []
    }
    """
    with pytest.raises(SFXError, match="Invalid GPIO pin or pixel order"):
        LightEffect(json_config)
