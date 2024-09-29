import pytest
from unittest.mock import patch, MagicMock
from sfx.led.effect import LEDControl, SFXError


# Fixture to mock GPIO operations
@pytest.fixture
def mock_gpio():
    """Mock RPi.GPIO methods."""
    with patch("RPi.GPIO.setmode") as mock_setmode, patch(
        "RPi.GPIO.setup"
    ) as mock_setup, patch("RPi.GPIO.output") as mock_output, patch(
        "RPi.GPIO.cleanup"
    ) as mock_cleanup:

        yield {
            "setmode": mock_setmode,
            "setup": mock_setup,
            "output": mock_output,
            "cleanup": mock_cleanup,
        }


def test_ledcontrol_init_with_valid_json(mock_gpio):
    """
    Test LEDControl initialization with valid JSON configuration.
    """
    json_config = """
    {
        "led_pins": {
            "LED1": 17,
            "LED2": 27
        },
        "commands": [
            {"command": "turn_on", "payload": {"led": "LED1"}},
            {"command": "turn_off", "payload": {"led": "LED2"}}
        ]
    }
    """

    # Create LEDControl object
    effect = LEDControl(json_config)

    # Ensure GPIO.setmode is called with GPIO.BCM (value of GPIO.BCM is 11)
    mock_gpio["setmode"].assert_called_once_with(11)

    # Ensure GPIO.setup is called twice for both LEDs
    assert mock_gpio["setup"].call_count == 2

    # Ensure GPIO.output is called to initialize the LEDs to LOW
    mock_gpio["output"].assert_any_call(17, 0)
    mock_gpio["output"].assert_any_call(27, 0)


def test_ledcontrol_run_commands(mock_gpio):
    """
    Test LEDControl run method with multiple commands.
    """
    json_config = """
    {
        "led_pins": {
            "LED1": 17,
            "LED2": 27
        },
        "commands": [
            {"command": "turn_on", "payload": {"led": "LED1"}},
            {"command": "turn_off", "payload": {"led": "LED2"}}
        ]
    }
    """

    effect = LEDControl(json_config)

    # Run the effect (execute commands)
    effect.run()

    # Ensure the correct GPIO output commands were made
    mock_gpio["output"].assert_any_call(17, 1)  # Turn on LED1
    mock_gpio["output"].assert_any_call(27, 0)  # Turn off LED2


def test_ledcontrol_cleanup(mock_gpio):
    """
    Test LEDControl cleanup command.
    """
    json_config = """
    {
        "led_pins": {
            "LED1": 17,
            "LED2": 27
        },
        "commands": [
            {"command": "cleanup"}
        ]
    }
    """

    effect = LEDControl(json_config)

    # Run the effect (execute commands)
    effect.run()

    # Ensure the GPIO cleanup method is called
    mock_gpio["cleanup"].assert_called_once()


def test_ledcontrol_invalid_json(mock_gpio):
    """
    Test LEDControl initialization with invalid JSON format.
    """
    invalid_json_config = """
    {
        "led_pins": {
            "LED1": 17,
            "LED2": 27
        },
        "commands": [
            {"command": "turn_on", "payload": {"led": "LED1"}}
        ]
        # Missing closing bracket
    """

    with pytest.raises(SFXError, match="Invalid JSON configuration"):
        LEDControl(invalid_json_config)


def test_ledcontrol_missing_led_in_commands(mock_gpio):
    """
    Test LEDControl command with missing LED in configuration.
    """
    json_config = """
    {
        "led_pins": {
            "LED1": 17
        },
        "commands": [
            {"command": "turn_on", "payload": {"led": "LED2"}}
        ]
    }
    """

    effect = LEDControl(json_config)

    # Run the effect (execute commands)
    effect.run()

    # Ensure GPIO.output is called to initialize LED1 to LOW during setup
    mock_gpio["output"].assert_any_call(17, 0)

    # Ensure no GPIO.output calls were made for the missing LED 'LED2'
    # We expect that no other output calls besides initialization were made
    assert mock_gpio["output"].call_count == 1


def test_ledcontrol_enter_exit(mock_gpio):
    """
    Test LEDControl usage in a context manager.
    """
    json_config = """
    {
        "led_pins": {
            "LED1": 17,
            "LED2": 27
        },
        "commands": [
            {"command": "turn_on", "payload": {"led": "LED1"}}
        ]
    }
    """

    with LEDControl(json_config) as effect:
        effect.run()

    # Ensure the cleanup method is called upon exiting the context manager
    mock_gpio["cleanup"].assert_called_once()
