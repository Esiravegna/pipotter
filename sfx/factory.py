import json
import logging

from core.error import SFXError
# Should you to add new effects, import them below and add to the available effects dict
from sfx.audio.effect import AudioEffect
from sfx.effect_container import EffectContainer
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
        effects.run('alohomora']
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

    def __init__(self, config_file, effects_list={'AudioEffect': AudioEffect, 'LightEffect': LightEffect}):
        """
        The constructor. It accept two parameters:
        :param config_file: string, path where to find the json config file
        :param effects_list: a list containing the effect name and the module to use. Look at the default for examples. 
            Remeber that the values of that dictionary MUST be callable and Effect-like.
        """
        logger.info("Creating a group of effects")
        self.effects_list = effects_list
        try:
            with open(config_file, 'r') as fp:
                str_json = fp.read()
                config = json.loads(str_json)
        except FileNotFoundError as e:
            raise SFXError("Unable to find {} : {}".format(config_file, e))
        except json.decoder.JSONDecodeError as e:
            raise SFXError("{} does not contains a valid json file : {}".format(config_file, e))
        self.spells = {}
        # Let's read our config file
        try:
            for a_spell_name, a_spell_value in config.items():
                # Let's get the Name
                # Let's create a container with that name
                logger.debug("Creating the config for {}".format(a_spell_name))
                self.spells[a_spell_name] = EffectContainer()
                # For each effect in the list of spells:
                for an_effect in a_spell_value:
                    # The effect is a dictionary of 'EffectName':'payload', so:
                    for effect_name, value in an_effect.items():
                        # Is the effect name a valid one?
                        if effect_name not in self.effects_list.keys():
                            logger.error("{} not a valid effect name".format(effect_name))
                            continue
                        # if it is, let's add to the container
                        self.spells[a_spell_name].append(self._create_effect(effect_name, value))
                        logger.debug("{} added to {}".format(effect_name, a_spell_name))
        except (TypeError, AttributeError, IndexError) as e:
            raise SFXError("Cannot parse config file due to {}".format(e))
        logger.info("Configuration created with {} spells on it".format(len(self.spells)))

    def __getitem__(self, spellname):
        return self.spells[spellname]

    def _create_effect(self, effect, effect_value):
        """
        Given the VALID_EFFECT_NAME constant and the effect variable, creates a valid effect
        :param effect: a valid effect name as per the EFFECT_LIST keys
        :param effect_value: the payload
        """
        # As per the effect defilition, effect_value should be a JSONAble string, so:
        if type(effect_value) is not str:
            effect_value = json.dumps(effect_value)
        return self.effects_list[effect](effect_value)

    def run(self, spell):
        """
        Actually runs an EffectContainer for the given spell as key
        :param spell: str, the spell name, or key
        """
        logger.info("Attempting to run it for {}".format(spell))
        if spell not in self.spells.keys():
            logger.error("{} not found in spells list, aborting".format(spell))
        self.spells[spell].run()
