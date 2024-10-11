import pytest
from unittest.mock import patch, MagicMock
from sfx.servomotor.effect import ServoMotor


# Fixture to selectively mock only methods, not constants
@pytest.fixture
def mock_gpio():
    """
    Mock RPi.GPIO methods, including setmode, setup, and PWM, but not constants like GPIO.OUT.
    """
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
            "cleanup": mock_cleanup,
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
            {"command": "set_angle", "payload": {"angle": 90}}
        ]
    }
    """

    # Create ServoMotor object
    effect = ServoMotor(json_config)

    # Ensure GPIO setup hasn't happened during initialization
    mock_gpio["setmode"].assert_not_called()
    mock_gpio["setup"].assert_not_called()
    mock_gpio["pwm"].start.assert_not_called()

    # Run the effect (this should trigger GPIO setup)
    effect.run()

    # Now ensure GPIO.setmode is called with GPIO.BCM (value of GPIO.BCM is 11)
    mock_gpio["setmode"].assert_called_once_with(11)

    # Ensure GPIO.setup is called for the specified GPIO pin (17)
    mock_gpio["setup"].assert_called_once_with(17, mock_gpio["setup"].call_args[0][1])

    # Ensure PWM is initialized and started with 0 duty cycle
    mock_gpio["pwm"].start.assert_called_once_with(0)


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
            {"command": "set_angle", "payload": {"angle": 45}}
        ]
    }
    """

    effect = ServoMotor(json_config)

    # Run the effect (execute commands)
    effect.run()

    # Ensure PWM duty cycle is calculated for angles 90 and 45
    duty_cycle_90 = float(90) / 18.0 + 2.5
    duty_cycle_45 = float(45) / 18.0 + 2.5

    mock_gpio["pwm"].ChangeDutyCycle.assert_any_call(duty_cycle_90)
    mock_gpio["pwm"].ChangeDutyCycle.assert_any_call(duty_cycle_45)

    # Ensure PWM stop was called after execution
    mock_gpio["pwm"].stop.assert_called_once()

    # Ensure cleanup is called after stopping
    mock_gpio["cleanup"].assert_called_once_with(17)


def test_servo_motor_delayed_initialization(mock_gpio):
    """
    Test that ServoMotor delays GPIO and PWM initialization until the effect is run.
    """
    json_config = """
    {
        "gpio_pin": 18,
        "frequency": 50,
        "commands": [
            {"command": "set_angle", "payload": {"angle": 90}}
        ]
    }
    """

    # Create a ServoMotor object (PWM should not be initialized yet)
    effect = ServoMotor(json_config)

    # Ensure that GPIO setup and PWM start are NOT called during object creation
    mock_gpio["setmode"].assert_not_called()
    mock_gpio["setup"].assert_not_called()
    mock_gpio["pwm"].start.assert_not_called()

    # Run the effect (this should initialize GPIO and PWM)
    effect.run()

    # Now ensure that GPIO setup and PWM start are called
    mock_gpio["setmode"].assert_called_once_with(11)
    mock_gpio["setup"].assert_called_once_with(18, mock_gpio["setup"].call_args[0][1])
    mock_gpio["pwm"].start.assert_called_once_with(0)

    # Ensure the PWM and GPIO are cleaned up after the stop command
    mock_gpio["pwm"].stop.assert_called_once()
    mock_gpio["cleanup"].assert_called_once_with(18)


def test_servo_motor_multiple_angles(mock_gpio):
    """
    Test that ServoMotor sets multiple angles and cleans up correctly.
    """
    json_config = """
    {
        "gpio_pin": 18,
        "frequency": 50,
        "commands": [
            {"command": "set_angle", "payload": {"angle": 0}},
            {"command": "set_angle", "payload": {"angle": 180}}
        ]
    }
    """

    effect = ServoMotor(json_config)
    effect.run()

    # Check duty cycles for 0 and 180 degrees
    duty_cycle_0 = float(0) / 18.0 + 2.5
    duty_cycle_180 = float(180) / 18.0 + 2.5

    mock_gpio["pwm"].ChangeDutyCycle.assert_any_call(duty_cycle_0)
    mock_gpio["pwm"].ChangeDutyCycle.assert_any_call(duty_cycle_180)

    # Ensure PWM stop was called once at the end
    mock_gpio["pwm"].stop.assert_called_once()

    # Ensure cleanup is called after stopping
    mock_gpio["cleanup"].assert_called_once_with(18)


def test_servo_motor_cleanup_on_delete(mock_gpio):
    """
    Test that ServoMotor cleans up resources when deleted.
    """
    json_config = """
    {
        "gpio_pin": 18,
        "frequency": 50,
        "commands": [
            {"command": "set_angle", "payload": {"angle": 90}}
        ]
    }
    """

    # Create ServoMotor object
    effect = ServoMotor(json_config)

    # Run the effect to initialize and set the angle
    effect.run()

    # Delete the object to trigger cleanup
    del effect

    # Ensure PWM stop and GPIO cleanup were called
    mock_gpio["pwm"].stop.assert_called_once()
    mock_gpio["cleanup"].assert_called_once_with(18)
