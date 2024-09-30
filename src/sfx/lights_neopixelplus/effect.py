import json
import logging
import time
import neopixel_plus as npp
from sfx.effect import Effect

logger = logging.getLogger(__name__)


class LightEffectsNPP(Effect):
    """
    A class to control NeoPixel LEDs using NeoPixelPlus for advanced effects, based on a JSON configuration.
        {
            "gpio_pin": 10,
            "num_leds": 16,
            "commands": [
                {
                "command": "color",
                "payload": {
                    "customization_json": {
                    "rgb_color": [255, 0, 0]
                    }
                }
                },
                {
                "command": "beats",
                "payload": {
                    "customization_json": {
                    "loop_limit": 10,
                    "rgb_colors": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    "duration_ms": 200,
                    "pause_ms": 300,
                    "brightness": 0.8
                    }
                }
                },
                {
                "command": "moving_dot",
                "payload": {
                    "customization_json": {
                    "loop_limit": 5,
                    "rgb_colors": [[255, 255, 0], [0, 255, 255]],
                    "duration_ms": 100,
                    "pause_a_ms": 100,
                    "pause_b_ms": 200,
                    "brightness": 0.9
                    }
                }
                },
                {
                "command": "light_up",
                "payload": {
                    "customization_json": {
                    "rgb_colors": [[0, 255, 0], [255, 0, 255]],
                    "sections": [0, 1],
                    "duration_ms": 500,
                    "pause_ms": 200,
                    "loop_limit": 2
                    }
                }
                },
                {
                "command": "transition",
                "payload": {
                    "customization_json": {
                    "rgb_colors": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    "duration_ms": 1000,
                    "pause_ms": 500,
                    "brightness": 1.0,
                    "loop_limit": 3
                    }
                }
                },
                {
                "command": "on",
                "payload": {
                    "pixel_number": 5
                }
                },
                {
                "command": "wait",
                "payload": {
                    "wait": 2
                }
                },
                {
                "command": "off",
                "payload": {
                    "pixel_number": 5
                }
                },
                {
                "command": "wait",
                "payload": {
                    "wait": 1
                }
                },
                {
                "command": "off",
                "payload": {}
                }
            ]
        }

    """

    def __init__(self, jsonable_string: str):
        """
        Initialize the NeoPixel strip and execute the effects.

        :param jsonable_string: A JSON string with NeoPixel configurations and commands.
        """
        super().__init__()
        self.strip = None
        self.commands = []
        self.name = "LightEffectsNPP"

        # Parse the configuration from the provided JSON string
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string: str):
        """
        Read and parse the JSON configuration for the NeoPixel strip and commands.
        """
        logger.debug("Reading JSON config file")
        try:
            config = json.loads(jsonable_string)
        except (AttributeError, ValueError) as e:
            logger.error(f"Invalid JSON configuration: {e}")
            raise ValueError(f"Invalid JSON configuration: {e}")

        # Extract and initialize NeoPixelPlus strip configurations
        try:
            gpio_pin = config.get("gpio_pin", "10")
            num_leds = config.get("num_leds", 16)

            # Use NeoPixelPlus for advanced effects with 'adafruit' target for Raspberry Pi
            self.strip = npp.NeoPixel(gpio_pin, num_leds, target="adafruit")
            self.commands = config.get("commands", [])

        except KeyError as e:
            logger.error(f"Missing configuration parameter: {e}")
            raise ValueError(f"Missing configuration parameter: {e}")

    def run(self):
        """
        Execute the commands defined in the JSON configuration.
        """
        logger.info("Running NeoPixel commands")
        for command in self.commands:
            action = command.get("command")
            payload = command.get("payload", {})

            try:
                if action == "beats":
                    self.strip.beats(
                        customization_json=payload.get(
                            "customization_json", {"loop_limit": 10}
                        )
                    )
                elif action == "moving_dot":
                    self.strip.moving_dot(
                        customization_json=payload.get(
                            "customization_json", {"loop_limit": 10}
                        )
                    )
                elif action == "light_up":
                    self.strip.light_up(
                        customization_json=payload.get(
                            "customization_json", {"loop_limit": 1}
                        )
                    )
                elif action == "transition":
                    self.strip.transition(
                        customization_json=payload.get(
                            "customization_json", {"loop_limit": 10}
                        )
                    )
                elif action == "color":
                    self.strip.beats(
                        customization_json=payload.get(
                            "customization_json", {"rgb_color": [100, 200, 200]}
                        )
                    )
                elif action == "off":
                    num_pix = payload.get("pixel_number", False)
                    if num_pix:
                        self.strip.off(num_pix)
                        self.srrip.write()
                    else:
                        self.strip.off()
                elif action == "on":
                    num_pix = payload.get("pixel_number", False)
                    if num_pix:
                        self.strip.on(num_pix)
                        self.srrip.write()
                    else:
                        self.strip.on()
                elif action == "wait":
                    wait_time = payload.get("wait", 1)
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Unknown command: {action}")

            except Exception as e:
                logger.error(f"Error executing command {action}: {e}")
                raise ValueError(f"Error executing command {action}: {e}")

    def cleanup(self):
        """Turn off all LEDs and clean up resources."""
        logger.info("Cleaning up and turning off all LEDs")
        self.strip.off()
        self.strip.write()
