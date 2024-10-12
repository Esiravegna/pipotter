import json
import logging
import random
import time
import board
import neopixel
import numpy as np
from core.error import SFXError
from sfx.effect import Effect

logger = logging.getLogger(__name__)


class LightEffect(Effect):
    """
    A NeoPixel-based effect for controlling addressable LEDs (like WS2812).
    This class creates the NeoPixel object from a JSON configuration and supports multiple effects.

    Example:

    {
            "gpio_pin": "D18",
            "num_leds": 16,
            "brightness": 0.5,
            "pixel_order": "GRB",
            "commands": [
                {
                "command": "rollcall_cycle",
                "payload": {
                    "wait": 0.1,
                    "colors": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
                }
                },
                {
                "command": "wait",
                "payload": {
                    "wait": 2
                }
                },
                {
                "command": "fire_effect",
                "payload": {
                    "num_random": 5,
                    "fire_color": "#FF4500",
                    "fade_by": -5,
                    "delay": 0.1
                }
                },
                {
                "command": "wait",
                "payload": {
                    "wait": 3
                }
                },
                {
                "command": "fade_up",
                "payload": {
                    "brightness": 100
                }
                },
                {
                "command": "wait",
                "payload": {
                    "wait": 1
                }
                },
                {
                "command": "fade_down",
                "payload": {
                    "brightness": 10
                }
                },
                {
                "command": "color",
                "payload": {
                    "led": 0,
                    "color": "#FFFFFF"
                }
                },
                {
                "command": "color",
                "payload": {
                    "led": 1,
                    "color": "#FF00FF"
                }
                },
                {
                "command": "color",
                "payload": {
                    "led": 2,
                    "color": "#00FF00"
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
                "payload": {
                    "led": 0
                }
                },
                {
                "command": "off",
                "payload": {
                    "led": 1
                }
                },
                {
                "command": "off",
                "payload": {
                    "led": 2
                }
                },
                {
                "command": "brightness",
                "payload": {
                    "brightness": 50
                }
                }
            ]
    }

    """

    def __init__(self, jsonable_string: str):
        """
        Initialize the NeoPixel strip and commands from a JSON string.

        :param jsonable_string: A JSON string with configuration for the NeoPixel object and commands.
        """
        super().__init__()
        self.strip = None
        self.commands = []  # Store parsed commands
        self.default_gokai_colours = [
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 0),
            (255, 105, 180),
            (192, 192, 192),
        ]
        self.name = "LightEffect"

        logger.info("Initializing LightEffect")
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string: str):
        """
        Read the JSON config and translate it into NeoPixel configurations and commands.
        """
        logger.debug("Reading JSON config file")
        try:
            config = json.loads(jsonable_string)
        except (AttributeError, ValueError) as e:
            logger.error(f"Invalid JSON configuration: {e}")
            raise SFXError(f"Invalid JSON configuration: {e}")

        # Parse NeoPixel configuration
        try:
            gpio_pin = getattr(board, config.get("gpio_pin", "D18"))
            num_leds = config["num_leds"]
            brightness = config.get("brightness", 1.0)
            pixel_order_str = config.get("pixel_order", "GRB")
            pixel_order = getattr(neopixel, pixel_order_str, neopixel.GRB)
        except KeyError as e:
            logger.error(f"Missing required configuration parameter: {e}")
            raise SFXError(f"Missing required configuration parameter: {e}")
        except AttributeError as e:
            logger.error(f"Invalid GPIO pin or pixel order: {e}")
            raise SFXError(f"Invalid GPIO pin or pixel order: {e}")

        # Initialize NeoPixel strip
        self.strip = neopixel.NeoPixel(
            gpio_pin,
            num_leds,
            brightness=brightness,
            auto_write=False,
            pixel_order=pixel_order,
        )

        # Parse commands
        try:
            commands = config.get("commands", [])
            for command in commands:
                cmd_type = command.get("command")
                payload = command.get("payload", {})

                if cmd_type == "rollcall_cycle":
                    wait = payload.get("wait", 0.2)
                    colors = payload.get("colors", self.default_gokai_colours)
                    colors_rgb = [
                        self._hex_to_rgb(c) if isinstance(c, str) else c for c in colors
                    ]
                    self.commands.append(("rollcall_cycle", wait, colors_rgb))

                elif cmd_type == "fire_effect":
                    num_random = payload.get("num_random", 3)
                    fire_color = payload.get("fire_color", "#FF4500")
                    fade_by = payload.get("fade_by", -3)
                    delay = payload.get("delay", 0.05)
                    fire_color = self._hex_to_rgb(fire_color)
                    self.commands.append(
                        ("fire_effect", num_random, fire_color, fade_by, delay)
                    )

                elif cmd_type == "fade_up":
                    brightness = payload.get("brightness", 100)
                    self.commands.append(("fade_up", brightness))

                elif cmd_type == "fade_down":
                    brightness = payload.get("brightness", 0)
                    self.commands.append(("fade_down", brightness))

                elif cmd_type == "color":
                    led_index = payload.get("led", 0)
                    color = payload.get("color", "#FFFFFF")
                    color_rgb = self._hex_to_rgb(color)
                    self.commands.append(("color", led_index, color_rgb))

                elif cmd_type == "brightness":
                    brightness = payload.get("brightness", 100)
                    self.commands.append(("brightness", brightness))

                elif cmd_type == "off":
                    led_index = payload.get("led", 0)
                    self.commands.append(("off", led_index))

                elif cmd_type == "wait":
                    wait_time = payload.get("wait", 0)
                    self.commands.append(("wait", wait_time))

                else:
                    logger.warning(f"Unknown command: {cmd_type}")
        except Exception as e:
            logger.error(f"Error reading commands: {e}")
            raise SFXError(f"Error reading commands: {e}")

        logger.debug("JSON processed successfully")

    def _fade(self, color1: tuple, color2: tuple, percent: float) -> tuple:
        """
        Transition between two colors based on the percentage.
        :param color1: The starting RGB color tuple (e.g., (255, 0, 0)).
        :param color2: The target RGB color tuple (e.g., (0, 255, 0)).
        :param percent: The percentage (0.0 to 1.0) indicating how much to transition.
        :return: An RGB tuple representing the color at the given percentage between color1 and color2.
        """
        # Convert the colors to numpy arrays for easy calculation
        color1 = np.array(color1)
        color2 = np.array(color2)

        # Interpolate between the two colors based on the percentage
        blended_color = color1 + (color2 - color1) * percent

        # Ensure the resulting color values are valid RGB integers (0-255)
        return tuple(int(channel) for channel in blended_color)

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert a hex color (string) to an RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def _rollcall_cycle(self, wait: float, colors: list):
        """Cycle through a set of colors with smooth transitions."""
        for j in range(len(colors)):
            for i in range(10):
                colour1 = colors[j]
                colour2 = colors[0] if j == len(colors) - 1 else colors[j + 1]
                percent = i * 0.1
                transition_color = self._fade(colour1, colour2, percent)
                self.strip.fill(transition_color)
                self.strip.show()
                time.sleep(wait)

    def _fire_effect(
        self, num_random: int, fire_color: tuple, fade_by: int, delay: float
    ):
        """
        Run the fire effect by lighting random LEDs and fading all LEDs for a duration calculated
        based on the number of LEDs and delay.
        """
        num_leds = len(self.strip)  # Total number of LEDs
        duration = (num_leds * delay) * 1.5
        start_time = time.time()
        while (
            time.time() - start_time < duration
        ):  # Run the effect for the calculated duration
            self._light_random_leds(num_random, fire_color)
            self._fade_leds(fade_by)
            self.strip.show()
            time.sleep(delay)

    def _light_random_leds(self, num_random: int, color: tuple):
        """Light up random LEDs with a specified color."""
        for _ in range(num_random):
            random_index = random.randint(0, len(self.strip) - 1)
            self.strip[random_index] = color

    def _fade_leds(self, fade_by: int):
        """Fade down all LEDs by a certain value."""
        for i in range(len(self.strip)):
            self.strip[i] = tuple(
                min(max(channel + fade_by, 0), 255) for channel in self.strip[i]
            )

    def _fade_brightness(
        self, start_brightness: float, end_brightness: float, duration: float
    ):
        """Smoothly fade the brightness from start_brightness to end_brightness over the specified duration."""
        steps = 50  # Number of steps for smooth fading
        step_duration = duration / steps  # Time per step
        brightness_delta = (
            end_brightness - start_brightness
        ) / steps  # Change in brightness per step

        for step in range(steps + 1):
            # Calculate new brightness for this step
            new_brightness = start_brightness + (brightness_delta * step)
            self.strip.brightness = max(
                0, min(1, new_brightness)
            )  # Clamp between 0 and 1
            self.strip.show()
            time.sleep(step_duration)

    def run(self):
        """Run the commands for controlling the NeoPixel strip."""
        logger.info("Running the commands")
        try:
            for command in self.commands:
                action = command[0]

                if action == "rollcall_cycle":
                    wait, colors = command[1], command[2]
                    self._rollcall_cycle(wait, colors)

                elif action == "fire_effect":
                    num_random, fire_color, fade_by, delay = (
                        command[1],
                        command[2],
                        command[3],
                        command[4],
                    )
                    self._fire_effect(num_random, fire_color, fade_by, delay)

                elif action == "fade_up":
                    brightness = command[1] / 100.0
                    self.strip.brightness = brightness
                    self.strip.show()

                elif action == "fade_down":
                    brightness = command[1] / 100.0
                    self.strip.brightness = brightness
                    self.strip.show()

                elif action == "color":
                    led_index, color_rgb = command[1], command[2]
                    self.strip[led_index] = color_rgb
                    self.strip.show()

                elif action == "brightness":
                    brightness = command[1] / 100.0
                    self.strip.brightness = brightness
                    self.strip.show()

                elif action == "off":
                    led_index = command[1]
                    self.strip[led_index] = (0, 0, 0)
                    self.strip.show()

                elif action == "wait":
                    wait_time = command[1]
                    time.sleep(wait_time)

            logger.info("Commands ran successfully")
        except Exception as e:
            logger.error(f"Unable to run the sequence due to {e}")
            raise SFXError(f"Unable to run the sequence due to {e}")

    def cleanup(self):
        """Turn off all LEDs and clean up resources."""
        logger.info("Cleaning up and turning off all LEDs")
        self.strip.fill((0, 0, 0))
        self.strip.show()

    def __enter__(self):
        """Enter method for context management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method for context management, ensuring cleanup is called."""
        self.cleanup()
