import json
import logging
import RPi.GPIO as GPIO
from core.error import SFXError
from sfx.effect import Effect

logger = logging.getLogger(__name__)


class LEDControl(Effect):
    """
    A class to control individual LEDs connected to GPIO pins via a JSON-based configuration.
    """

    def __init__(self, jsonable_string):
        """
        Initialize the LEDControl class from a JSON string.

        :param jsonable_string: A JSON string with the LED configuration and commands.
        """
        super().__init__()
        self.led_pins = {}
        self.commands = []
        self.use_pwm = False
        self.name = "LEDControlEffect"

        logger.info("Initializing LEDControl")
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string):
        """
        Parse the JSON configuration and initialize the LED control.
        """
        logger.debug("Parsing LEDControl JSON configuration")
        try:
            config = json.loads(jsonable_string)
        except ValueError as e:
            logger.error(f"Invalid JSON configuration: {e}")
            raise SFXError(f"Invalid JSON configuration: {e}")

        try:
            led_pins = config["led_pins"]
            self.use_pwm = config.get("use_pwm", False)
            frequency = config.get("frequency", 100)

            logger.info("Setting up GPIO and LED pins")
            GPIO.setmode(GPIO.BCM)
            for led, pin in led_pins.items():
                GPIO.setup(pin, GPIO.OUT)
                if self.use_pwm:
                    pwm = GPIO.PWM(pin, frequency)
                    pwm.start(0)
                    self.led_pins[led] = pwm
                    logger.info(f"LED {led} (Pin {pin}) initialized with PWM")
                else:
                    self.led_pins[led] = GPIO.LOW
                    logger.info(f"LED {led} (Pin {pin}) initialized without PWM")

            # Parse commands
            logger.info("Parsing LEDControl commands")
            for command in config.get("commands", []):
                cmd_type = command.get("command")
                payload = command.get("payload", {})

                if cmd_type == "turn_on":
                    led = payload.get("led")
                    self.commands.append(("turn_on", led))
                    logger.debug(f"Command added: Turn on LED {led}")

                elif cmd_type == "turn_off":
                    led = payload.get("led")
                    self.commands.append(("turn_off", led))
                    logger.debug(f"Command added: Turn off LED {led}")

                elif cmd_type == "set_brightness":
                    led = payload.get("led")
                    brightness = payload.get("brightness", 100)
                    self.commands.append(("set_brightness", led, brightness))
                    logger.debug(
                        f"Command added: Set brightness of LED {led} to {brightness}%"
                    )

                else:
                    logger.warning(f"Unknown command: {cmd_type}")

        except KeyError as e:
            logger.error(f"Missing required configuration parameter: {e}")
            raise SFXError(f"Missing required configuration parameter: {e}")

    def turn_on(self, led):
        """Turn on the LED."""
        logger.info(f"Turning on LED {led}")
        if self.use_pwm:
            self.led_pins[led].ChangeDutyCycle(100)  # 100% brightness
        else:
            GPIO.output(self.led_pins[led], GPIO.HIGH)

    def turn_off(self, led):
        """Turn off the LED."""
        logger.info(f"Turning off LED {led}")
        if self.use_pwm:
            self.led_pins[led].ChangeDutyCycle(0)  # 0% brightness
        else:
            GPIO.output(self.led_pins[led], GPIO.LOW)

    def set_brightness(self, led, brightness):
        """Set the brightness of the LED using PWM."""
        logger.info(f"Setting brightness of LED {led} to {brightness}%")
        if not self.use_pwm:
            logger.error("Attempted to set brightness without PWM enabled")
            raise SFXError("PWM is not enabled for this instance.")
        self.led_pins[led].ChangeDutyCycle(brightness)

    def run(self):
        """Run the commands for controlling LEDs."""
        logger.info("Running LEDControl commands")
        for command in self.commands:
            action = command[0]

            if action == "turn_on":
                led = command[1]
                self.turn_on(led)

            elif action == "turn_off":
                led = command[1]
                self.turn_off(led)

            elif action == "set_brightness":
                led, brightness = command[1], command[2]
                self.set_brightness(led, brightness)

    def cleanup(self):
        """Cleanup the GPIO pins."""
        logger.info("Cleaning up GPIO for LEDControl")
        for led in self.led_pins.values():
            if self.use_pwm:
                led.stop()
        GPIO.cleanup()
