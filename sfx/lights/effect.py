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
    
    Now creates the NeoPixel object from a JSON configuration and supports multiple effects.
    
    Usage:
    ```
    json_config = '{"gpio_pin": "D18", "num_leds": 16, "brightness": 0.4, "pixel_order": "GRB", "commands": [{"command": "rollcall_cycle", "payload": {"wait": 0.1}}]}'
    an_effect = LightEffect(json_config)
    an_effect.run()
    ```
    """

    def __init__(self, jsonable_string):
        """
        The constructor, which initializes the NeoPixel strip from a JSON string.
        
        :param jsonable_string: A JSON string with configuration for the NeoPixel object and commands.
        """
        self.strip = None
        self.commands = []  # This will store the parsed commands
        self.default_gokai_colours = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 105, 180), (192, 192, 192)]
        self.name = "LightEffect"

        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string):
        """
        Reads the JSON config string and translates it into NeoPixel configurations and commands.
        """
        logger.debug("Reading JSON config file")
        try:
            config = json.loads(jsonable_string)
        except (AttributeError, ValueError) as e:
            raise SFXError(f"Invalid JSON configuration: {e}")
        
        # Parse NeoPixel configuration
        try:
            gpio_pin = getattr(board, config.get("gpio_pin", "D18"))
            num_leds = config["num_leds"]
            brightness = config.get("brightness", 1.0)
            pixel_order_str = config.get("pixel_order", "GRB")
            pixel_order = getattr(neopixel, pixel_order_str, neopixel.GRB)
        except KeyError as e:
            raise SFXError(f"Missing required configuration parameter: {e}")
        except AttributeError as e:
            raise SFXError(f"Invalid GPIO pin or pixel order: {e}")
        except Exception as e:
            raise SFXError(f"Error reading configuration: {e}")
        
        # Initialize NeoPixel strip
        self.strip = neopixel.NeoPixel(gpio_pin, num_leds, brightness=brightness, auto_write=False, pixel_order=pixel_order)

        # Parse commands
        try:
            commands = config.get("commands", [])
            for command in commands:
                cmd_type = command.get("command")
                payload = command.get("payload", {})

                if cmd_type == "rollcall_cycle":
                    wait = payload.get("wait", 0.2)
                    colors = payload.get("colors", self.default_gokai_colours)
                    colors_rgb = [self._hex_to_rgb(c) if isinstance(c, str) else c for c in colors]
                    self.commands.append(('rollcall_cycle', wait, colors_rgb))

                elif cmd_type == "fire_effect":
                    num_random = payload.get("num_random", 3)
                    fire_color = payload.get("fire_color", "#FF4500")
                    fade_by = payload.get("fade_by", -3)
                    delay = payload.get("delay", 0.05)
                    fire_color = self._hex_to_rgb(fire_color)
                    self.commands.append(('fire_effect', num_random, fire_color, fade_by, delay))

                elif cmd_type == "fade_up":
                    brightness = payload.get("brightness", 100)
                    self.commands.append(('fade_up', brightness))

                elif cmd_type == "fade_down":
                    brightness = payload.get("brightness", 0)
                    self.commands.append(('fade_down', brightness))

                elif cmd_type == "color":
                    led_index = payload.get("led", 0)
                    color = payload.get("color", "#FFFFFF")
                    color_rgb = self._hex_to_rgb(color)
                    self.commands.append(('color', led_index, color_rgb))

                elif cmd_type == "brightness":
                    brightness = payload.get("brightness", 100)
                    self.commands.append(('brightness', brightness))

                elif cmd_type == "off":
                    led_index = payload.get("led", 0)
                    
                elif cmd_type == "wait":
                    wait_time = payload.get("wait", 0)
                    self.commands.append(('wait', wait_time))
                else:
                    logger.warning(f"Unknown command: {cmd_type}")
        except Exception as e:
            raise SFXError(f"Error reading commands: {e}")
        
        logger.debug("JSON processed successfully")

    def _hex_to_rgb(self, hex_color):
        """
        Convert a hex color (string) to an RGB tuple.
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _fade(self, colour1, colour2, percent):
        """
        Transition between two colors based on the percentage.
        """
        colour1 = np.array(colour1)
        colour2 = np.array(colour2)
        vector = colour2 - colour1
        newcolour = (int((colour1 + vector * percent)[0]),
                     int((colour1 + vector * percent)[1]),
                     int((colour1 + vector * percent)[2]))
        return newcolour

    def _rollcall_cycle(self, wait, colors):
        """
        Cycle through a set of colors with smooth transitions.
        """
        for j in range(len(colors)):
            for i in range(10):
                colour1 = colors[j]
                # Cycle back to the first color after the last one
                colour2 = colors[0] if j == len(colors) - 1 else colors[j + 1]
                percent = i * 0.1  # 0.1*100 so 10% increments between colors
                transition_color = self._fade(colour1, colour2, percent)
                self.strip.fill(transition_color)
                self.strip.show()
                time.sleep(wait)

    def _fire_effect(self, num_random, fire_color, fade_by, delay):
        """Run the fire effect by lighting random LEDs and fading all LEDs."""
        while True:
            # Light random LEDs with the fire color
            self._light_random_leds(num_random, fire_color)

            # Measure start time
            start_time = time.monotonic()

            # Fade all LEDs
            self._fade_leds(fade_by)

            # Measure elapsed time
            elapsed_time = time.monotonic() - start_time
            print(f"Update took {int(elapsed_time * 1000)} ms")

            # Update the strip
            self.strip.show()

            # Small delay before next frame
            time.sleep(delay)

    def _light_random_leds(self, num_random, color):
        """Light up random LEDs with a specified color."""
        for _ in range(num_random):
            random_index = random.randint(0, len(self.strip) - 1)
            self.strip[random_index] = color

    def _fade_leds(self, fade_by):
        """Fade down all LEDs by a certain value (fade_by)."""
        for i in range(len(self.strip)):
            self.strip[i] = tuple(min(max(channel + fade_by, 0), 255) for channel in self.strip[i])

    def run(self):
        """
        Runs the effects on the NeoPixel strip.
        """
        logger.info("Running the commands")
        try:
            for command in self.commands:
                action = command[0]

                if action == 'rollcall_cycle':
                    wait, colors = command[1], command[2]
                    self._rollcall_cycle(wait, colors)

                elif action == 'fire_effect':
                    num_random, fire_color, fade_by, delay = command[1], command[2], command[3], command[4]
                    self._fire_effect(num_random, fire_color, fade_by, delay)

                elif action == 'fade_up':
                    brightness = command[1] / 100.0
                    self.strip.brightness = brightness
                    self.strip.show()
                    
                elif action == 'fade_down':
                    brightness = command[1] / 100.0
                    self.strip.brightness = brightness
                    self.strip.show()
                    
                elif action == 'color':
                    led_index, color_rgb = command[1], command[2]
                    self.strip[led_index] = color_rgb
                    self.strip.show()

                elif action == 'brightness':
                    brightness = command[1] / 100.0
                    self.strip.brightness = brightness
                    self.strip.show()

                elif action == 'off':
                    led_index = command[1]
                    self.strip[led_index] = (0, 0, 0)
                    self.strip.show()

                time.sleep(1)  # Simulate time between commands
                
            logger.info("Commands ran successfully")
        except Exception as e:
            raise SFXError(f"Unable to run the sequence due to {e}")
