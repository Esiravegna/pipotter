import pytest
from unittest.mock import patch, MagicMock
from sfx.led.effect import LEDControl, SFXError 

# Fixture to selectively mock only methods, not constants
@pytest.fixture
def mock_gpio():
    """Mock RPi.GPIO methods, including setmode, setup, and PWM, but not constants like GPIO.OUT."""
    with patch('RPi.GPIO.setmode') as mock_setmode, \
         patch('RPi.GPIO.setup') as mock_setup, \
         patch('RPi.GPIO.PWM') as mock_pwm:

        mock_pwm_instance = MagicMock()  # Create a mock instance for PWM
        mock_pwm.return_value = mock_pwm_instance  # PWM() returns the mock instance

        yield {
            'setmode': mock_setmode,
            'setup': mock_setup,
            'pwm': mock_pwm_instance  # Pass the PWM mock instance
        }

def test_ledcontrol_init_with_valid_json(mock_gpio):
    """
    Test LEDControl initialization with valid JSON and PWM.
    """
    json_config = '''
    {
        "led_pins": {
            "led1": 18,
            "led2": 23
        },
        "use_pwm": true,
        "frequency": 100,
        "commands": [
            {"command": "set_brightness", "payload": {"led": "led1", "brightness": 50}},
            {"command": "turn_off", "payload": {"led": "led2"}}
        ]
    }
    '''
    
    # Create LEDControl object
    effect = LEDControl(json_config)

    # Ensure GPIO.setmode is called with GPIO.BCM (value of GPIO.BCM is 11)
    mock_gpio['setmode'].assert_called_once_with(11)


    # Ensure PWM is started correctly for both LEDs
    assert mock_gpio['pwm'].start.call_count == 2  # Two LEDs, both using PWM

def test_ledcontrol_run_commands(mock_gpio):
    """
    Test LEDControl run method with multiple commands.
    """
    json_config = '''
    {
        "led_pins": {
            "led1": 18
        },
        "use_pwm": true,
        "commands": [
            {"command": "set_brightness", "payload": {"led": "led1", "brightness": 50}},
            {"command": "turn_off", "payload": {"led": "led1"}}
        ]
    }
    '''
    
    effect = LEDControl(json_config)
    
    # Run the effect (execute commands)
    effect.run()

    # Ensure brightness was set to 50% for led1
    mock_gpio['pwm'].ChangeDutyCycle.assert_any_call(50)

    # Ensure led1 was turned off (brightness set to 0)
    mock_gpio['pwm'].ChangeDutyCycle.assert_any_call(0)
