import logging

logger = logging.getLogger(__name__)
from os.path import exists
from core.error import SFXError

from sfx.effect_container import Effect

logger = logging.getLogger(__name__)


class AudioEffect(Effect):
    """
    Runs an audio effect from a given file using the mplayer command.
    IT DOES REQUIRE TO HAVE THE MPLAYER INSTALLED
    Usage:
    ```
    an_effect = AudioEffect(A_mediaPlayer, a_variable_containing_the_path_to_run)
    an_effect.run()
    ```
    :param mediaplayer a valid sfx.audio.player.MediaPlayer instance
    :param the_filename: a valid existing filename
    """

    def __init__(self, mediaplayer, the_filename):
        self.player = mediaplayer
        self.filename = the_filename
        super().__init__()
        logger.info("Creating AudioEffect object")
        self.name = "AudioEffect"
        if not exists(the_filename):
            raise SFXError("Unable to reproduce {}".format(the_filename))

    def run(self):
        """
        Just runs the player
        """
        logging.debug("Attemting to reproduce sound effect")
        try:
            self.player.play(self.filename)
        except SFXError as e:
            raise SFXError("Unable to play due to {}".format(e))
