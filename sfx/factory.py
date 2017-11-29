import json
import logging

from sfx.effect_container import EffectContainer

from core.error import SFXError

# Should you to add new effects, import them below and add to the available effects dict
from sfx.audio.effect import AudioEffect
from sfx.lights.effect import LightEffect

logger = logging.getLogger(__name__)

class EffectFactory(object):
    """
    Given a configuration file containing names and effects, will create an object that can create all the specified effects.
    This object is tightly coupled to the sfx module: Should you want to add any new effect, please take a look at the EFFECTS_LIST 
    and the methods of this class.
    
    Usage:
        ```
        from sfx.factory import EffectFactory
        effects = EffectFactory('/path/to/the/spells.json')
        effects['alohomora'].run()
        ```
    
    The JSON file would look something like:
        ```
        [
            "spell_name_1": [
                {"AudioEffect": "file/to/play"},
                {"LightEffect": [
                    {'group':'1', 'command': 'fade_up'},
                     ...
                    {'group':'1', 'command': 'brighten', 'payload': '50'},
                ]}, 
            ],
            ...
            "spell_name_N": [
                {"AudioEffect": "file/to/play"},
                {"LightEffect": [
                    {'group':'1', 'command': 'fade_up'},
                     ...
                    {'group':'1', 'command': 'brighten', 'payload': '50'},
                ]}, 
            ]
        
        ]
        ```
    Please note that the order is not mandatory, however it is suggested that the audio effects runs first as they may go in the background.
    """

    # the available effects
    EFFECTS_LIST = {
        'AudioEffect': AudioEffect,
        'LightEffect': LightEffect
    }

    def _create_effect(self, effect, effect_value):
        """
        Given the VALID_EFFECT_NAME constant and the effect variable, creates a valid effect
        :param effect: a valid effect name as per the EFFECT_LIST keys
        :param effect_value: the payload
        """
        return self.EFFECTS_LIST[effect]()

    def __init__(self, config_file):
        logger.info("Creating a group of effects")
        try:
            with open(config_file, 'rb') as fp:
                config = json.loads(fp.read())
        except FileNotFoundError as e:
            raise SFXError("Unable to find {} : {}".format(config_file, e))
        except json.decoder.JSONDecodeError as e:
            raise SFXError("{} does not contains a valid json file : {}".format(config_file, e))
        self.spells = {}
        # Let's read our config file
        try:
            for a_spell in config:
                # Let's get the Name
                spell_name = a_spell.keys()[0]
                # Let's create a container with that name
                self.spells[spell_name] = EffectContainer()
                for an_effect in a_spell[spell_name]:
                    if an_effect not in self.EFFECTS_LIST:
                        logger.error("{} not a valid effect name".format(an_effect))
                        continue





        except (TypeError, AttributeError, IndexError) as e:
            raise SFXError("Cannot parse config file due to {}".format(e))



