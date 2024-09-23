from sfx.servomotor.effect import ServoMotor
import pytest
from unittest.mock import patch, MagicMock


# Fixture to selectively mock only methods, not constants
@pytest.fixture
def mock_gpio():
    """Mock RPi.GPIO methods, including setmode, setup, and PWM, but not constants like GPIO.OUT."""
    with patch("RPi.GPIO.setmode") as mock_setmode, patch(
        "RPi.GPIO.setup"
    ) as mock_setup, patch("RPi.GPIO.PWM") as mock_pwm, patch(
        "RPi.GPIO.cleanup"
    ) as mock_cleanup:
        mock_pwm_instance = MagicMock()  # Create a mock instance for PWM
        mock_pwm.return_value = mock_pwm_instance  # PWM() returns the mock instance

        yield {
            "setmode": mock_setmode,
            "setup": mock_setup,
            "pwm": mock_pwm_instance,
            "cleanup": mock_cleanup,  # Include cleanup in the mock
        }


def test_servomotor_init_with_valid_json(mock_gpio):
    """
    Test ServoMotor initialization with valid JSON.
    """
    json_config = """
    {
        "gpio_pin": 17,
        "frequency": 50,
        "commands": [
            {"command": "set_angle", "payload": {"angle": 90}},
            {"command": "stop"}
        ]
    }
    """

    # Create ServoMotor object
    effect = ServoMotor(json_config)

    # Ensure GPIO.setmode is called with GPIO.BCM (value of GPIO.BCM is 11)
    mock_gpio["setmode"].assert_called_once_with(11)

    # Ensure GPIO.setup is called for the correct pin with GPIO.OUT (real value of GPIO.OUT is 1)
    mock_gpio["setup"].assert_called_once_with(17, 0)

    # Ensure PWM is started correctly for the servo
    assert mock_gpio["pwm"].start.call_count == 1  # PWM is started once

    # Ensure cleanup is called after stop
    effect.stop()
    mock_gpio["pwm"].stop.assert_called_once()
    mock_gpio["cleanup"].assert_called_once_with(
        17
    )  # GPIO.cleanup should be called after stop


def test_servomotor_run_commands(mock_gpio):
    """
    Test ServoMotor run method with multiple commands.
    """
    json_config = """
    {
        "gpio_pin": 17,
        "frequency": 50,
        "commands": [
            {"command": "set_angle", "payload": {"angle": 90}},
            {"command": "set_angle", "payload": {"angle": 45}},
            {"command": "stop"}
        ]
    }
    """

    effect = ServoMotor(json_config)

    # Run the effect (execute commands)
    effect.run()

    # Ensure PWM duty cycle is calculated for angles 90 and 45
    duty_cycle_90 = (0.05 * 50) + (90 / 18.0)
    duty_cycle_45 = (0.05 * 50) + (45 / 18.0)

    mock_gpio["pwm"].ChangeDutyCycle.assert_any_call(duty_cycle_90)
    mock_gpio["pwm"].ChangeDutyCycle.assert_any_call(duty_cycle_45)

    # Ensure PWM stop was called
    effect.pwm.stop.assert_called_once()

    # Ensure cleanup is called after stopping
    mock_gpio["cleanup"].assert_called_once_with(17)
