import json
import logging
import RPi.GPIO as GPIO
from core.error import SFXError
from sfx.effect import Effect

logger = logging.getLogger(__name__)


class LEDControl(Effect):
    """
    A class to control individual LEDs connected to GPIO pins via a JSON-based configuration.

    Example:

    {
        "led_pins": {
            "LED1": 17,
            "LED2": 27
        },
        "commands": [
            {"command": "turn_on", "payload": {"led": "LED1"}},
            {"command": "turn_off", "payload": {"led": "LED2"}},
            {"command": "cleanup"}
        ]
    }
    """

    def __init__(self, jsonable_string: str):
        """
        Initialize the LEDControl class from a JSON string.

        :param jsonable_string: A JSON string with the LED configuration and commands.
        """
        super().__init__()
        self.led_pins = {}
        self.commands = []
        self.name = "LEDControlEffect"

        logger.info("Initializing LEDControl")
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string: str):
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
            logger.info("Setting up GPIO and LED pins")
            GPIO.setmode(GPIO.BCM)

            # Setup each pin as output and initialize to LOW
            for led, pin in led_pins.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)  # Initialize LED to OFF
                self.led_pins[led] = pin
                logger.info(f"LED {led} (Pin {pin}) initialized and set to LOW")

            # Parse commands
            logger.info("Parsing LEDControl commands")
            for command in config.get("commands", []):
                cmd_type = command.get("command")
                payload = command.get("payload", {})

                if cmd_type == "turn_on":
                    led = payload.get("led")
                    if led not in self.led_pins:
                        logger.error(f"LED {led} not found in configuration.")
                        continue
                    self.commands.append(("turn_on", led))
                    logger.debug(f"Command added: Turn on LED {led}")

                elif cmd_type == "turn_off":
                    led = payload.get("led")
                    if led not in self.led_pins:
                        logger.error(f"LED {led} not found in configuration.")
                        continue
                    self.commands.append(("turn_off", led))
                    logger.debug(f"Command added: Turn off LED {led}")

                elif cmd_type == "cleanup":
                    self.commands.append(("cleanup",))
                    logger.debug("Command added: Cleanup GPIO")

                else:
                    logger.warning(f"Unknown command: {cmd_type}")

        except KeyError as e:
            logger.error(f"Missing required configuration parameter: {e}")
            raise SFXError(f"Missing required configuration parameter: {e}")

    def turn_on(self, led: str):
        """Turn on the LED."""
        if led in self.led_pins:
            pin = self.led_pins[led]
            logger.info(f"Turning on LED {led} (Pin {pin})")
            GPIO.output(pin, GPIO.HIGH)
        else:
            logger.error(f"LED {led} not found in configuration")

    def turn_off(self, led: str):
        """Turn off the LED."""
        if led in self.led_pins:
            pin = self.led_pins[led]
            logger.info(f"Turning off LED {led} (Pin {pin})")
            GPIO.output(pin, GPIO.LOW)
        else:
            logger.error(f"LED {led} not found in configuration")

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

            elif action == "cleanup":
                self.cleanup()

    def cleanup(self):
        """Cleanup the GPIO pins."""
        logger.info("Cleaning up GPIO for LEDControl")
        GPIO.cleanup()

    def __enter__(self):
        """Enter method for use in a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method for use in a context manager, ensuring cleanup is called."""
        self.cleanup()
