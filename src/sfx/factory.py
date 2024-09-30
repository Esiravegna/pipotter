import json
import logging

from core.error import SFXError

# Should you add new effects, import them below and add to the available effects dict
from sfx.audio.effect import AudioEffect
from sfx.effect_container import EffectContainer
from sfx.led.effect import LEDControl
from sfx.lights.effect import LightEffect
from sfx.lights_neopixelplus.effect import LightEffectsNPP
from sfx.servomotor.effect import ServoMotor

logger = logging.getLogger(__name__)


class EffectFactory(object):
    """
    Given a configuration file containing names and effects, this class creates objects that can generate all the specified effects.
    This object is tightly coupled to the sfx module: to add new effects, update the EFFECTS_LIST and the methods of this class.

    Usage:
        ```python
        from sfx.factory import EffectFactory
        effects = EffectFactory('/path/to/the/spells.json')
        effects.run('alohomora')
        ```

    Example JSON config:
        ```json
        {
            "alohomora": [
                {"AudioEffect": "file/to/play"},
                {"LightEffect": {
                    "gpio_pin": "D18",
                    "num_leds": 16,
                    "brightness": 0.5,
                    "pixel_order": "GRB",
                    "commands": [
                        {"command": "rollcall_cycle", "payload": {"wait": 0.1}},
                        {"command": "fade_up", "payload": {"brightness": 50}},
                        {"command": "color", "payload": {"led": 1, "color": "#FF0000"}}
                    ]
                }},
                {"LEDControl": {
                    "led_pins": {
                        "led1": 18,
                        "led2": 23
                    },
                    "use_pwm": true,
                    "commands": [
                        {"command": "set_brightness", "payload": {"led": "led1", "brightness": 50}},
                        {"command": "turn_off", "payload": {"led": "led2"}}
                    ]
                }},
                {"ServoMotor": {
                    "gpio_pin": 17,
                    "commands": [
                        {"command": "set_angle", "payload": {"angle": 90}},
                        {"command": "set_angle", "payload": {"angle": 45}},
                        {"command": "stop"}
                    ]
                }}
            ]
        }
        ```

    Please note that the order of effects is not mandatory, but it is suggested that audio effects run first as they may go in the background.
    """

    def __init__(self, config_file, effects_list=None):
        """
        The constructor accepts two parameters:
        :param config_file: string, path to the JSON config file
        :param effects_list: a dict containing the effect names and their corresponding class.
            Defaults to including AudioEffect, LightEffect, LEDControl, and ServoMotor.
        """
        logger.info("Creating a group of effects")

        if effects_list is None:
            # Default effects list includes AudioEffect, LightEffect, LEDControl, and ServoMotor
            self.effects_list = {
                "AudioEffect": AudioEffect,
                "LightEffect": LightEffect,
                "LightEffectNPP": LightEffectsNPP,
                "LEDControl": LEDControl,
                "ServoMotor": ServoMotor,
            }
        else:
            self.effects_list = effects_list

        try:
            with open(config_file, "r") as fp:
                str_json = fp.read()
                config = json.loads(str_json)
        except FileNotFoundError as e:
            raise SFXError(f"Unable to find {config_file}: {e}")
        except json.decoder.JSONDecodeError as e:
            raise SFXError(f"{config_file} does not contain a valid JSON file: {e}")

        self.spells = {}

        try:
            for spell_name, spell_value in config.items():
                logger.debug(f"Creating the config for {spell_name}")
                self.spells[spell_name] = EffectContainer()
                for an_effect in spell_value:
                    for effect_name, value in an_effect.items():
                        if effect_name not in self.effects_list:
                            logger.error(f"{effect_name} not a valid effect name")
                            continue
                        self.spells[spell_name].append(
                            self._create_effect(effect_name, value)
                        )
                        logger.debug(f"{effect_name} added to {spell_name}")
        except (TypeError, AttributeError, IndexError) as e:
            raise SFXError(f"Cannot parse config file due to {e}")

        logger.info(f"Configuration created with {len(self.spells)} spells")

    def __getitem__(self, spellname):
        return self.spells[spellname]

    def __iter__(self):
        return iter(self.spells)

    def _create_effect(self, effect, effect_value):
        """
        Given the valid effect name and its value, create a corresponding effect.
        :param effect: a valid effect name (from the EFFECTS_LIST keys)
        :param effect_value: the payload to initialize the effect
        """
        if type(effect_value) is not str:
            effect_value = json.dumps(effect_value)
        return self.effects_list[effect](effect_value)

    def run(self, spell):
        """
        Run the effects in the EffectContainer for the given spell.
        :param spell: str, the name of the spell
        """
        logger.info(f"Attempting to run {spell}")
        if spell not in self.spells:
            logger.error(f"{spell} not found in spells list, aborting")
            return
        self.spells[spell].run()
