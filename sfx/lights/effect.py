import json
import logging
import milight
from time import sleep

from core.config import settings
from core.error import SFXError
from sfx.effect_container import Effect

logger = logging.getLogger(__name__)


class LightEffect(Effect):
    """
    A MiLight bassed effect using
    https://github.com/McSwindler/python-milight
    
    Usage:
    ```
    an_effect = LightEffect("[{'group':'1', 'command': 'fade_up', 'payload': '50'}]",a_controller, a_light)
    ```
    
    So, this command will create an effect from a controller, a light and a json string. In this case, the json string contains the fade_up command, 
    will run it on the light group 1 and the value for the aforementioned command will be 50.
    Please do mind that a single LightEffect object can pile up commands and there's no need to pile up multiple LightEffects commands, yet possible.
    Important: if you use the color command, as it requires a specific helper to translate the value (either in hsl, rgb or hex), only hex is supported.
    """

    def __init__(self,
                 jsonable_string,
                 milight_controller=milight.MiLight({'host': settings['PIPOTTER_MILIGHT_SERVER'], 'port': settings['PIPOTTER_MILIGHT_PORT']}, wait_duration=0.025),
                 milight_light=milight.LightBulb(['rgbw']), #TODO Evaluate if makes sense to restrict this to rgbw as we'd need color for the spells
                 time_to_sleep=1):
        """
        The constructor
        :param jsonable_string: a string that can be turn into a json element. Must be in the form: 
        [
            {"group":"1", "command":"command", "payload": "value"},
            ...,
            {"group":"1", "command":"command", "payload": "value"}
        ]
        :param milight_controller: a MiLight controller object
        :param milight_light: a Milight LigthBulb object
        :param time_to_sleep: int, seconds to wait before the 
        """
        super().__init__()
        logger.info("Creating LightEffect object")
        self.name = "LightEffect"
        self.controller = milight_controller
        self.light = milight_light
        self.commands = [] # This will be overwritten by a milight.Command object later on
        self.time_to_sleep = time_to_sleep
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string):
        """
        Reads the config json file out from jsonable_string and creates a Milight sequence to later run on
        :param jsonable_string: a string that can be turn into a json element. Must be in the form: 
        [
            {"group":"1", "command":"command", "payload": "value"},
            ...,
            {"group":"1", "command":"command", "payload": "value"}
        ]
        """
        logger.debug("Reading JSON config file")
        try:
            sequence = json.loads(jsonable_string)
        except (AttributeError, ValueError) as e:
            raise SFXError("Unable to parse {} due to {}".format(jsonable_string, e))
        logger.debug("Sequence created, adding elements")
        for a_command in sequence:
            try:
                # Essentially: for each str command attribute, get the command from the light element
                # and piles it up into the list using the payload and group element
                the_command = getattr(self.light, a_command['command'])
                value = a_command.get('payload', None)
                group = a_command.get('group', None)
                if value:
                    # If the command got a parameter...
                    if a_command['command'] == 'color':
                        #  The color processing is different for all the rest of them as
                        # it uses a 0.255 value calculated from  a helper
                        # For simplicity sake, let's use only hex values.
                        value = milight.color_from_hex(value)
                    the_command_result = the_command(value, group) if group else the_command(value) # the wait command does not take a group parameter
                elif not group:
                    # This is a generic wait command, so:
                    the_command_result = the_command(a_command.get('value', 1))
                else:
                    # Nope, just the bulb group as a parameter
                    the_command_result = the_command(group)
                if not self.commands:
                    self.commands = the_command_result
                else:
                    self.commands += the_command_result
            except (ValueError, AttributeError) as e:
                logger.error("Unable to decode {} due to {}".format(a_command, e))
        logger.debug("JSON processed sucessfully")

    def run(self):
        """
        Runs the effects previously piled up
        """
        logger.info("Running the commands")
        try:
            logger.info("Running the commands")
            key = self.controller.repeat(self.commands, reps=0)
            sleep(self.time_to_sleep)
            self.controller.cancel(key)
            logger.info("commands ran")
        except (ValueError, AttributeError) as e:
            raise SFXError("Unable to run the sequence due to".format(e))
