import logging
from os.path import exists
from errno import ENOENT
import subprocess
from core.error import SFXError
from core.config import settings

log = logging.getLogger(__name__)


class Mediaplayer(object):
    """
    Just a media player using mplayer
    """

    player = None

    def play(self, file_to_play):
        """
        Given a file to play, calls a mplayer instance
        :param file_to_play: <str> path to play a file
        """
        log.debug("About to play {}".format(file_to_play))
        if not exists(file_to_play):
            raise SFXError("Unable to reproduce {}".format(file_to_play))
        try:
            log.info("calling mplayer")
            self.player = subprocess.Popen(
                ["mplayer", file_to_play] + settings["PIPOTTER_EXTRA_AUDIO_COMMANDS"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            if e.errno == ENOENT:
                raise SFXError("Check that mplayer is installed: {}".format(e))
            else:
                raise SFXError("Error when calling mplayer: {}".format(e))

    def stop(self):
        """
        Stops the player shoud it be active
        """
        log.info("Stopping mplayer")
        self.player.stdin.write("q")
